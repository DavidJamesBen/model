"""Independently fitted DDE forecasting baseline for synthetic robustness data.

Only the first 100 observed bins enter least squares. The next 20 bins choose
among three independently initialized fits. Truth parameters and states are
read for recovery diagnostics only after fitting and start selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from revision_dde import LOWER, UPPER, NAMES
from robustness_synthetic import OUT, solve_integrated, HISTORY

SEEDS = (11, 22, 33)
FIT_MESH = 2000
EVAL_MESH = 4000
MAX_NFEV = 400
HORIZONS = (20, 40, 80)
LO, HI = np.log(LOWER), np.log(UPPER)


def metrics(y, prediction, train_y):
    scale = np.maximum(np.sqrt(np.mean(train_y ** 2, axis=0)), 1.0)
    delta = prediction - y
    nrmse_channels = np.sqrt(np.mean(delta ** 2, axis=0)) / scale
    nmae_channels = np.mean(np.abs(delta), axis=0) / scale
    return {"nrmse": float(np.mean(nrmse_channels)),
        "nmae": float(np.mean(nmae_channels)),
        "rumor_nrmse": float(nrmse_channels[0]),
        "debunk_nrmse": float(nrmse_channels[1])}


def freeze_protocol():
    path = OUT / "dde_baseline_protocol.json"
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {"version": 1, "method": "RK4 method of steps + bounded log-parameter least squares",
        "unknown_parameters": NAMES, "lower": LOWER.tolist(), "upper": UPPER.tolist(),
        "fixed_history": HISTORY.tolist(), "fixed_media_decay": 1.0,
        "count_scale": 10000.0, "fit_mesh": FIT_MESH, "evaluation_mesh": EVAL_MESH,
        "refinement_mesh": 8000, "initialization_seeds": SEEDS,
        "initialization": "Independent PCG64 log-uniform draw in 5%-95% of each bounded log range; no true parameters or neural estimates",
        "training_bins": [0, 100], "validation_bins": [100, 120], "test_bins": [120, 200],
        "test_horizons": HORIZONS, "max_nfev_per_start": MAX_NFEV,
        "least_squares": {"method": "trf", "ftol": 1e-9, "xtol": 1e-9,
                          "gtol": 1e-8, "diff_step": 1e-4, "jac": "2-point"},
        "loss": "Training two-channel mean squared error divided by training RMS (floored at 1)",
        "selection": "Minimum validation macro NRMSE among all three fitted starts",
        "record_failures": "Retain finite budget-limited results with convergence status; never restart according to test error",
        "budgets": "400 residual evaluations reported by scipy per start, plus numerical Jacobian calls counted separately. Not equivalent to Adam updates.",
        "source_sha256": source_hash,
    }
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
            raise RuntimeError("Frozen DDE baseline protocol does not match current settings/source")
    else:
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def fit_start(y, edges, seed):
    train_y = y[:100]
    scales = np.maximum(np.sqrt(np.mean(train_y ** 2, axis=0)), 1.0)
    rng = np.random.Generator(np.random.PCG64(seed))
    start = LO + rng.uniform(.05, .95, 8) * (HI - LO)
    calls = 0
    def fun(z):
        nonlocal calls
        calls += 1
        _, predicted = solve_integrated(np.exp(z), edges[:101], FIT_MESH)
        return ((predicted - train_y) / scales).ravel()
    beginning = time.perf_counter()
    result = least_squares(fun, start, bounds=(LO, HI), method="trf",
        diff_step=1e-4, ftol=1e-9, xtol=1e-9, gtol=1e-8, max_nfev=MAX_NFEV)
    runtime = time.perf_counter() - beginning
    theta = np.exp(result.x)
    states, predicted = solve_integrated(theta, edges, EVAL_MESH)
    states_refined, prediction_refined = solve_integrated(theta, edges, 8000)
    validation = metrics(y[100:120], predicted[100:120], train_y)
    record = {"seed": seed, "parameters": dict(zip(NAMES, theta.tolist())),
        "initial_parameters": dict(zip(NAMES, np.exp(start).tolist())),
        "runtime_seconds": runtime, "success": bool(result.success),
        "status": int(result.status), "message": result.message,
        "nfev": int(result.nfev), "njev": int(result.njev or 0), "actual_residual_calls": calls,
        "optimality": float(result.optimality), "training_rss": float(2 * result.cost),
        "training_nrmse": metrics(train_y, predicted[:100], train_y)["nrmse"],
        "validation_nrmse": validation["nrmse"],
        "relative_observation_mesh_refinement": float(np.max(abs(predicted - prediction_refined) / np.maximum(abs(prediction_refined), 1e-12))),
        "maximum_state_mesh_refinement": float(np.max(abs(states - states_refined))),
        "minimum_predicted_state": float(states.min()),
        "mass_error": float(np.max(abs(states[:, :7].sum(1) - 1.0)))}
    if record["relative_observation_mesh_refinement"] > 1e-5 or states.min() < -1e-9:
        raise AssertionError("Fitted DDE numerical verification failed")
    return record, predicted, states


def run_dataset(path):
    destination = OUT / "dde_baseline" / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    data = np.load(path)
    # Ground-truth parameters and states are not used here.
    y = data["y"]
    edges = data["bin_edges"]
    fitted = []
    for seed in SEEDS:
        record_path = destination / f"seed{seed}.json"
        prediction_path = destination / f"seed{seed}.npz"
        if record_path.exists() and prediction_path.exists():
            record = json.loads(record_path.read_text())
            predictions = np.load(prediction_path)
            predicted, states = predictions["predicted"], predictions["states"]
        else:
            record, predicted, states = fit_start(y, edges, seed)
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            np.savez_compressed(prediction_path, predicted=predicted, states=states,
                times=edges[1:], seed=seed)
        fitted.append((record, predicted, states))
        print("DDE", path.stem, seed, f"validation={record['validation_nrmse']:.6g}",
              f"success={record['success']}", f"nfev={record['nfev']}", flush=True)
    selected_seed = min(fitted, key=lambda item: item[0]["validation_nrmse"])[0]["seed"]
    # Selection is complete before any test/recovery metric is calculated.
    true_theta, true_states = data["theta"], data["states"]
    partition, scenario = str(data["partition"]), str(data["scenario"])
    dataset_rows = []
    for record, predicted, states in fitted:
        theta = np.array([record["parameters"][name] for name in NAMES])
        for horizon in HORIZONS:
            stop = 120 + horizon
            row = {"dataset": path.stem, "partition": partition, "scenario": scenario,
                "generator_seed": int(data["generator_seed"]), "model": "DDE-LS",
                "seed": record["seed"], "horizon": horizon,
                "selected": record["seed"] == selected_seed,
                "runtime_seconds": record["runtime_seconds"], "success": record["success"],
                "nfev": record["nfev"], "actual_residual_calls": record["actual_residual_calls"],
                "validation_nrmse": record["validation_nrmse"]}
            row.update(metrics(y[120:stop], predicted[120:stop], y[:100]))
            cumulative = metrics(np.cumsum(y, axis=0)[120:stop],
                np.cumsum(predicted, axis=0)[120:stop], np.cumsum(y[:100], axis=0))
            row["cumulative_nrmse"] = cumulative["nrmse"]
            state_scale = np.maximum(np.sqrt(np.mean(true_states[1:101] ** 2, axis=0)), 1e-8)
            state_error = np.sqrt(np.mean((states[121:stop + 1] - true_states[121:stop + 1]) ** 2, axis=0)) / state_scale
            row["state_nrmse"] = float(np.mean(state_error))
            row["latent_E_H_U_M1_M2_nrmse"] = float(np.mean(state_error[[2, 3, 6, 7, 8]]))
            row["parameter_log_rmse"] = float(np.sqrt(np.mean(np.log(theta / true_theta) ** 2)))
            row["parameter_mean_relative_error"] = float(np.mean(abs(theta / true_theta - 1)))
            for k, name in enumerate(NAMES):
                row[f"relative_error_{name}"] = float(abs(theta[k] / true_theta[k] - 1))
            dataset_rows.append(row)
    pd.DataFrame(dataset_rows).to_csv(destination / "metrics.csv", index=False)
    (destination / "selection.json").write_text(json.dumps({"selected_seed": selected_seed,
        "criterion": "Validation-only macro NRMSE", "training_bins": 100,
        "validation_bins": 20, "horizons": HORIZONS}, indent=2), encoding="utf-8")
    return dataset_rows


def summarize():
    paths = sorted((OUT / "dde_baseline").glob("*/metrics.csv"))
    if not paths:
        return
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame.to_csv(OUT / "dde_metrics.csv", index=False)
    quantities = ["nrmse", "nmae", "cumulative_nrmse", "runtime_seconds",
        "state_nrmse", "latent_E_H_U_M1_M2_nrmse", "parameter_log_rmse",
        "parameter_mean_relative_error"]
    per_seed = frame.groupby(["partition", "scenario", "horizon", "seed"])[quantities].mean().reset_index()
    per_seed.to_csv(OUT / "dde_seed_aggregates.csv", index=False)
    summary = per_seed.groupby(["partition", "scenario", "horizon"])[quantities].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(OUT / "dde_all_start_summary.csv", index=False)
    selected = frame[frame.selected]
    selected.groupby(["partition", "scenario", "horizon"])[quantities].mean().reset_index().to_csv(
        OUT / "dde_validation_selected_summary.csv", index=False)
    print(summary[["nrmse_mean", "nrmse_std"]].to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("development", "test", "all"), default="all")
    args = parser.parse_args()
    freeze_protocol()
    # Exclude one-time JIT compilation from individual fitting runtimes.
    solve_integrated(np.sqrt(LOWER * UPPER), mesh=FIT_MESH)
    paths = sorted((OUT / "datasets").glob("*.npz"))
    for path in paths:
        if args.partition != "all" and not path.stem.startswith(args.partition + "_"):
            continue
        run_dataset(path)
        summarize()


if __name__ == "__main__":
    main()
