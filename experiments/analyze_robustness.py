"""Independent, balanced audits and paired inference for the follow-up experiments.

This module deliberately does not import a training module. Saved predictions,
validation scores and available-training denominators determine every result.
Positive paired differences mean lower error for TransPINN. Bootstrap intervals
describe comparison uncertainty, not prediction-band coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
import tempfile

import numpy as np
import pandas as pd


CHANNELS = ("rumor", "debunk")
MODELS = ("TransPINN", "Matched-PINN", "NoAttn-PINN", "Data-Transformer", "Data-MLP")
PRIMARY_BASELINES = ("Data-Transformer", "Data-MLP")
METRICS = ("nrmse", "nmae", "cumulative_nrmse", "cumulative_nmae")
PARAMETER_NAMES = ("beta_1", "beta_2", "tau_normalized", "q_R", "q_D", "chi", "reporting_R", "reporting_D")


@dataclass
class AuditedRun:
    event: str
    fold: int
    model: str
    seed: int
    validation_score: float
    runtime_seconds: float
    denominator: np.ndarray
    predictions: pd.DataFrame
    metadata: dict
    metrics: list[dict]

    @property
    def key(self) -> tuple[str, int, str, int]:
        return self.event, self.fold, self.model, self.seed


def long_path(path: Path) -> Path:
    """Use Windows extended-length paths locally without changing file names."""
    if os.name != "nt":
        return Path(path)
    absolute = os.path.abspath(os.fspath(path))
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def sha256(path: Path) -> str:
    return hashlib.sha256(long_path(path).read_bytes()).hexdigest()


def normalized_metrics(actual: np.ndarray, predicted: np.ndarray,
                       denominator: np.ndarray) -> dict[str, np.ndarray]:
    error = (predicted - actual) / denominator
    return {"nrmse": np.sqrt(np.mean(error * error, axis=0)),
            "nmae": np.mean(np.abs(error), axis=0)}


def audit_run(payload: dict, prediction_path: Path, *,
              expected_epochs: int | None = None,
              expected_parameter_count: int | None = 18619,
              horizon: int | None = None) -> AuditedRun:
    """Recompute scores, including selection, rather than trust training summaries.

    The training code must save ``metric_denominator`` (two channels), or the
    exact boolean ``train_observed`` mask in its prediction CSV. Complete-data
    runs may omit the mask only when the record explicitly states retention=1.
    """
    prediction_path = long_path(prediction_path)
    run = dict(payload["run"])
    frame = pd.read_csv(prediction_path)
    required = {"split", *[f"{kind}_{c}" for kind in ("actual", "predicted") for c in CHANNELS]}
    if not required <= set(frame.columns):
        raise ValueError(f"Prediction columns missing in {prediction_path}")
    split = frame["split"].astype(str).to_numpy()
    train, validation, test = (split == label for label in ("train", "validation", "test"))
    counts = [int(mask.sum()) for mask in (train, validation, test)]
    if min(counts) < 1 or split.tolist() != (["train"] * counts[0]
            + ["validation"] * counts[1] + ["test"] * counts[2]):
        raise ValueError(f"Invalid chronological split in {prediction_path}")
    actual = frame[[f"actual_{c}" for c in CHANNELS]].to_numpy(float)
    predicted = frame[[f"predicted_{c}" for c in CHANNELS]].to_numpy(float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError(f"Nonfinite prediction in {prediction_path}")
    if (actual < 0).any() or (predicted < -1e-10).any():
        raise ValueError(f"Negative activity in {prediction_path}")
    recorded_d = payload.get("metric_denominator", run.get("metric_denominator"))
    if "train_observed" in frame:
        observed_raw = frame["train_observed"]
        if observed_raw.dtype == object:
            observed = observed_raw.astype(str).str.lower().map({"true": True, "false": False,
                                                                 "1": True, "0": False})
            if observed.isna().any():
                raise ValueError("Invalid training observation mask")
            observed = observed.to_numpy(bool)
        else:
            observed = observed_raw.to_numpy(bool)
        if np.any(observed & ~train) or not observed.any():
            raise ValueError("Observation mask must contain only available training bins")
        denominator = np.maximum(np.sqrt(np.mean(actual[observed] ** 2, axis=0)), 1.)
        if recorded_d is not None:
            np.testing.assert_allclose(denominator, recorded_d, rtol=3e-6, atol=1e-9)
    elif recorded_d is not None:
        denominator = np.asarray(recorded_d, dtype=float)
    elif float(payload.get("retention", run.get("retention", -1))) == 1.:
        denominator = np.maximum(np.sqrt(np.mean(actual[train] ** 2, axis=0)), 1.)
    else:
        raise ValueError("Sparse-run audit requires metric_denominator or train_observed")
    if denominator.shape != (2,) or not np.isfinite(denominator).all() or np.any(denominator < 1):
        raise ValueError("Invalid normalization denominator")
    if payload.get("mask_seed", 0) and payload.get("noise_seed", 0) and "train_observed" in frame:
        # Reconstruct the prescribed sampling/noise mechanism from the saved
        # independent seeds. A missing observation never becomes a zero label.
        count_train = counts[0]
        order = np.random.default_rng(int(payload["mask_seed"])).permutation(count_train)
        expected_mask = np.zeros(len(frame), dtype=bool)
        expected_mask[order[:max(1, int(np.ceil(count_train * float(payload["retention"]))))]] = True
        np.testing.assert_array_equal(observed, expected_mask)
        normal = np.random.default_rng(int(payload["noise_seed"])).standard_normal((count_train, 2))
        expected_labels = np.maximum(0., actual[:count_train] + float(payload["noise"]) * denominator * normal)
        saved_labels = frame[[f"training_label_{c}" for c in CHANNELS]].to_numpy(float)
        np.testing.assert_allclose(saved_labels[observed], expected_labels[observed[:count_train]],
                                   rtol=3e-7, atol=1e-9)
        if np.isfinite(saved_labels[~observed]).any():
            raise ValueError("Unavailable labels must be excluded")
        estimated_total = np.maximum(actual[observed].sum(0) * count_train / observed.sum(), 1.)
        np.testing.assert_allclose(estimated_total, payload["scales"], rtol=3e-7, atol=1e-9)
    val_score = float(normalized_metrics(actual[validation], predicted[validation], denominator)["nrmse"].mean())
    np.testing.assert_allclose(val_score, run["validation_score"], rtol=5e-5, atol=1e-8,
                               err_msg=f"Validation selection mismatch: {prediction_path}")
    if expected_epochs is not None and int(run["epochs_completed"]) != expected_epochs:
        raise ValueError(f"Unexpected update count in {prediction_path}")
    if expected_parameter_count is not None and int(run["parameter_count"]) != expected_parameter_count:
        raise ValueError(f"Unexpected trainable count in {prediction_path}")
    if payload.get("gradient_connected") is False:
        raise ValueError(f"Inactive trainable parameters in {prediction_path}")
    if not np.isfinite(run["runtime_seconds"]) or float(run["runtime_seconds"]) <= 0:
        raise ValueError(f"Invalid runtime in {prediction_path}")
    trace = payload.get("trace", payload.get("history"))
    trace_path = prediction_path.with_suffix(".trace.csv")
    if trace_path.exists():
        trace = pd.read_csv(trace_path).to_dict("records")
        epochs = [int(item["epoch"]) for item in trace]
        completed = int(run["epochs_completed"])
        if epochs != list(range(10, completed + 1, 10)):
            raise ValueError(f"Incomplete validation/loss trace in {trace_path}")
        row = next((item for item in trace if int(item["epoch"]) == int(run["best_epoch"])), None)
        if row is None:
            raise ValueError("Saved best epoch is absent from validation trace")
        np.testing.assert_allclose(val_score, row["validation_nrmse"], rtol=5e-5, atol=1e-8)
    if isinstance(trace, list) and trace and isinstance(trace[0], dict):
        score_key = next((key for key in ("validation_score", "validation_nrmse", "val_nrmse")
                          if all(key in item for item in trace)), None)
        if score_key:
            # The chosen checkpoint must minimize validation error, allowing only
            # floating-point save/reload error. The test score is never consulted.
            best_trace_score = min(float(item[score_key]) for item in trace)
            np.testing.assert_allclose(val_score, best_trace_score, rtol=5e-5, atol=1e-8)
    used_test = np.flatnonzero(test)
    if horizon is not None:
        if len(used_test) < horizon:
            raise ValueError("Requested horizon exceeds saved test interval")
        used_test = used_test[:horizon]
    values = normalized_metrics(actual[used_test], predicted[used_test], denominator)
    if horizon is None:
        for metric in ("nrmse", "nmae"):
            if f"test_{metric}" in run:
                np.testing.assert_allclose(values[metric].mean(), run[f"test_{metric}"], rtol=2e-7, atol=1e-12)
    # Auxiliary cumulative errors remain normalized by their own training RMS.
    actual_cumulative, predicted_cumulative = np.cumsum(actual, axis=0), np.cumsum(predicted, axis=0)
    cumulative_denominator = np.maximum(np.sqrt(np.mean(actual_cumulative[train] ** 2, axis=0)), 1.)
    cumulative = normalized_metrics(actual_cumulative[used_test], predicted_cumulative[used_test], cumulative_denominator)
    key = {k: run[k] for k in ("event", "fold", "model", "seed")}
    rows = [{**key, "channel": channel, **{k: float(v[i]) for k, v in values.items()},
             **{f"cumulative_{k}": float(v[i]) for k, v in cumulative.items()}}
            for i, channel in enumerate(CHANNELS)]
    if horizon is not None:
        # Truncate only after cumulative evaluation, preserving forecast origin.
        frame = frame.iloc[:int(used_test[-1]) + 1].copy()
    return AuditedRun(str(run["event"]), int(run["fold"]), str(run["model"]), int(run["seed"]),
                      val_score, float(run["runtime_seconds"]), denominator, frame,
                      {**payload, "prediction_sha256": sha256(prediction_path)}, rows)


def check_complete_grid(runs: Sequence[AuditedRun], models: Sequence[str], seeds: Sequence[int],
                        folds_by_event: Mapping[str, Sequence[int]]) -> dict:
    expected = {(e, int(f), m, int(s)) for e, folds in folds_by_event.items()
                for f in folds for m in models for s in seeds}
    keys = [r.key for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate run records")
    actual = set(keys)
    if expected != actual:
        raise ValueError(f"Incomplete experiment: missing={sorted(expected-actual)[:8]}, "
                         f"unexpected={sorted(actual-expected)[:8]}")
    groups: dict[tuple, list[AuditedRun]] = {}
    for r in runs:
        groups.setdefault((r.event, r.fold, r.seed), []).append(r)
    for key, paired in groups.items():
        first = paired[0]
        for other in paired[1:]:
            np.testing.assert_array_equal(first.predictions["split"], other.predictions["split"])
            for channel in CHANNELS:
                np.testing.assert_allclose(first.predictions[f"actual_{channel}"],
                                           other.predictions[f"actual_{channel}"], rtol=2e-15, atol=1e-12)
            np.testing.assert_allclose(first.denominator, other.denominator, rtol=1e-12)
            for col in ("train_observed", "training_label_rumor", "training_label_debunk"):
                if col in first.predictions or col in other.predictions:
                    if col == "train_observed":
                        np.testing.assert_array_equal(first.predictions[col], other.predictions[col])
                    else:
                        np.testing.assert_allclose(first.predictions[col], other.predictions[col],
                                                   rtol=2e-15, atol=1e-12, equal_nan=True)
            for field in ("mask_seed", "noise_seed", "mask_sha256", "noise_sha256", "input_sha256",
                          "budget", "retention", "noise"):
                if field == "budget" and "DDE-LS" in (first.model, other.model):
                    # Function evaluations of a numerical inverse solver are not
                    # Adam update units. Its cost is reported separately.
                    continue
                a = first.metadata.get(field, first.metadata.get("run", {}).get(field))
                b = other.metadata.get(field, other.metadata.get("run", {}).get(field))
                if a != b:
                    raise ValueError(f"Unpaired {field} for {key}")
    return {"complete": True, "fit_count": len(runs), "expected_fits": len(expected),
            "models": list(models), "seeds": list(seeds), "events": list(folds_by_event),
            "paired_observations_and_denominators": True,
            "validation_score_recomputations": len(runs)}


def balanced_summaries(runs: Sequence[AuditedRun]) -> dict[str, pd.DataFrame]:
    metrics = pd.DataFrame([row for r in runs for row in r.metrics])
    manifest = pd.DataFrame([{k: getattr(r, k) for k in ("event", "fold", "model", "seed",
                "validation_score", "runtime_seconds")} for r in runs])
    # Each reduction explicitly retains equal weighting at the next level.
    fold_seed = metrics.groupby(["event", "fold", "model", "seed"], as_index=False)[list(METRICS)].mean()
    event_seed = fold_seed.groupby(["event", "model", "seed"], as_index=False)[list(METRICS)].mean()
    macro_seed = event_seed.groupby(["model", "seed"], as_index=False)[list(METRICS)].mean()
    summary = macro_seed.groupby("model")[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    selected = manifest.sort_values(["validation_score", "seed"], kind="stable").drop_duplicates(["event", "fold", "model"])
    selected_rows = metrics.merge(selected[["event", "fold", "model", "seed"]], on=["event", "fold", "model", "seed"])
    selected_fold = selected_rows.groupby(["event", "fold", "model"], as_index=False)[list(METRICS)].mean()
    selected_event = selected_fold.groupby(["event", "model"], as_index=False)[list(METRICS)].mean()
    selected_macro = selected_event.groupby("model")[list(METRICS)].mean().add_prefix("selected_")
    resources = manifest.groupby("model")["runtime_seconds"].agg(["mean", "std", "size"])
    resources.columns = ["runtime_seconds_mean", "runtime_seconds_std", "fit_count"]
    summary = summary.join(selected_macro).join(resources)
    return {"metrics": metrics, "run_manifest": manifest, "fold_seed_metrics": fold_seed,
            "event_seed_metrics": event_seed, "macro_seed_metrics": macro_seed,
            "selected_initializations": selected, "selected_event_metrics": selected_event,
            "summary": summary.reset_index()}


def prepare_squared_errors(runs: Sequence[AuditedRun], models: Sequence[str], seeds: Sequence[int],
                           folds_by_event: Mapping[str, Sequence[int]], selected: bool = False
                           ) -> dict[tuple[str, int], np.ndarray]:
    """Return arrays with axes initialization, method, channel, test time."""
    by_key = {r.key: r for r in runs}
    result = {}
    for event, folds in folds_by_event.items():
        for fold in folds:
            block = []
            for seed in ([None] if selected else seeds):
                methods = []
                for model in models:
                    if selected:
                        r = min((by_key[(event, int(fold), model, int(s))] for s in seeds),
                                key=lambda x: (x.validation_score, x.seed))
                    else:
                        r = by_key[(event, int(fold), model, int(seed))]
                    test = r.predictions["split"].eq("test")
                    actual = r.predictions.loc[test, [f"actual_{c}" for c in CHANNELS]].to_numpy(float)
                    predicted = r.predictions.loc[test, [f"predicted_{c}" for c in CHANNELS]].to_numpy(float)
                    methods.append((((predicted - actual) / r.denominator) ** 2).T)
                block.append(methods)
            result[(event, int(fold))] = np.asarray(block)
    return result


def hierarchical_bootstrap(squared_errors: Mapping[tuple[str, int], np.ndarray],
                           folds_by_event: Mapping[str, Sequence[int]], *,
                           replicates: int = 50000, random_seed: int = 20260919,
                           batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized paired event/fold/seed/circular-time-block resampling.

    A single draw resamples all methods together. Time blocks and channel pairs
    are shared across compared methods. Repeated event, fold or seed selections
    receive independent time-block draws, matching the hierarchical estimator.
    """
    if replicates < 1 or batch_size < 1:
        raise ValueError("Positive replicate and batch counts required")
    events = tuple(folds_by_event)
    first = next(iter(squared_errors.values()))
    n_seeds, n_models = first.shape[:2]
    point = np.mean([np.mean([np.sqrt(squared_errors[(e, int(f))].mean(-1)).mean((0, 2))
                              for f in folds_by_event[e]], axis=0) for e in events], axis=0)
    rng = np.random.default_rng(random_seed)
    draws = np.empty((replicates, n_models))
    models_ix = np.arange(n_models)[None, None, :, None, None]
    channels_ix = np.arange(2)[None, None, None, :, None]
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        aggregate = np.zeros((count, n_models))
        sampled_events = rng.integers(0, len(events), size=(count, len(events)))
        for event_ix, event in enumerate(events):
            rep_ix, _ = np.nonzero(sampled_events == event_ix)
            if not len(rep_ix):
                continue
            folds = tuple(folds_by_event[event])
            sampled_folds = rng.integers(0, len(folds), size=(len(rep_ix), len(folds)))
            for fold_ix, fold in enumerate(folds):
                positions, _ = np.nonzero(sampled_folds == fold_ix)
                if not len(positions):
                    continue
                error = squared_errors[(event, int(fold))]
                if error.shape[:3] != (n_seeds, n_models, 2):
                    raise ValueError("Unbalanced seed/model/channel grid")
                n_time = error.shape[-1]
                n_draws = len(positions)
                seed_draws = rng.integers(0, n_seeds, size=(n_draws, n_seeds))
                block_len = int(math.ceil(math.sqrt(n_time)))
                block_count = int(math.ceil(n_time / block_len))
                block_starts = rng.integers(0, n_time, size=(n_draws, n_seeds, block_count))
                time_draws = ((block_starts[..., None] + np.arange(block_len)) % n_time)
                time_draws = time_draws.reshape(n_draws, n_seeds, -1)[..., :n_time]
                sampled_error = error[seed_draws[..., None, None, None], models_ix,
                                      channels_ix, time_draws[:, :, None, None, :]]
                value = np.sqrt(sampled_error.mean(-1)).mean((1, 3))
                np.add.at(aggregate, rep_ix[positions], value / (len(events) * len(folds)))
        draws[start:start + count] = aggregate
    return point, draws


def paired_intervals(runs: Sequence[AuditedRun], models: Sequence[str], seeds: Sequence[int],
                     folds_by_event: Mapping[str, Sequence[int]], *,
                     replicates: int = 50000, random_seed: int = 20260919,
                     multiplicity: int = 36) -> pd.DataFrame:
    reference = list(models).index("TransPINN")
    rows = []
    for selected in (False, True):
        errors = prepare_squared_errors(runs, models, seeds, folds_by_event, selected)
        point, draws = hierarchical_bootstrap(errors, folds_by_event, replicates=replicates,
                                              random_seed=random_seed + 7919 * int(selected))
        for i, model in enumerate(models):
            if i == reference:
                continue
            differences = draws[:, i] - draws[:, reference]
            primary = model in PRIMARY_BASELINES
            tail = .05 / (2 * multiplicity) if primary else .025
            low, high = np.quantile(differences, [tail, 1 - tail])
            nominal_low, nominal_high = np.quantile(differences, [.025, .975])
            rows.append({"evaluation": "validation_selected" if selected else "all_seeds",
                "baseline": model, "baseline_macro_nrmse": float(point[i]),
                "transpinn_macro_nrmse": float(point[reference]),
                "baseline_minus_transpinn_nrmse": float(point[i] - point[reference]),
                "relative_reduction_percent": float(100 * (point[i] - point[reference]) / max(point[i], 1e-12)),
                "ci95_low": float(nominal_low), "ci95_high": float(nominal_high),
                "simultaneous_low": float(low) if primary else None,
                "simultaneous_high": float(high) if primary else None,
                "primary_family": primary, "family_size": multiplicity if primary else 0,
                "familywise_confidence": .95 if primary else None,
                "transpinn_advantage_after_correction": bool(primary and low > 0),
                "bootstrap_probability_baseline_worse": float(np.mean(differences > 0)),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": random_seed + 7919 * int(selected),
                "expected_tail_draws": replicates * tail,
                "selection_inference": "conditional on saved validation selection" if selected else "paired initialization resampling",
                "time_block_rule": "circular blocks of ceil(sqrt(n_test))"})
    return pd.DataFrame(rows)


def analyze_group(runs: Sequence[AuditedRun], output_dir: Path, *, models: Sequence[str] = MODELS,
                  seeds: Sequence[int] = (11, 22, 33),
                  folds_by_event: Mapping[str, Sequence[int]], replicates: int = 50000,
                  random_seed: int = 20260919, multiplicity: int = 36) -> dict[str, pd.DataFrame]:
    output_dir = long_path(output_dir)
    audit = check_complete_grid(runs, models, seeds, folds_by_event)
    tables = balanced_summaries(runs)
    tables["paired_intervals"] = paired_intervals(runs, models, seeds, folds_by_event,
        replicates=replicates, random_seed=random_seed, multiplicity=multiplicity)
    # Verify the point estimates in inference and summaries agree exactly.
    summary = tables["summary"].set_index("model")
    for row in tables["paired_intervals"].itertuples():
        metric = "selected_nrmse" if row.evaluation == "validation_selected" else "nrmse_mean"
        np.testing.assert_allclose(row.baseline_macro_nrmse, summary.loc[row.baseline, metric], atol=1e-12)
        np.testing.assert_allclose(row.transpinn_macro_nrmse, summary.loc["TransPINN", metric], atol=1e-12)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    audit.update({"bootstrap_replicates": replicates, "primary_family_size": multiplicity,
        "weighting": "equal channels, then folds, then events; seed SD uses ddof=1",
        "normalization": "saved available-training RMS, lower bounded by 1",
        "selection": "validation macro incremental NRMSE only; lowest seed breaks ties",
        "runtime_kinds": sorted({str(r.metadata.get("run", {}).get("runtime_kind", "unspecified")) for r in runs}),
        "bootstrap_limitations": f"{len(folds_by_event)} outer trajectory/event clusters. Percentile bootstrap is an approximation; intervals are not prediction coverage.",
        "source_sha256": sha256(Path(__file__))})
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return tables


def empirical_groups(root: Path, scope: str = "all", condition: str | None = None) -> list[dict]:
    """Prespecified logical grids; shared clean fits are not duplicated on disk."""
    root = long_path(root)
    events = ("harvey", "csdc", "ottawa", "sydney")
    specifications = []
    if scope in ("all", "stage1"):
        specifications += [(f"stage1_b{budget}", budget, 100, 0, (11, 22, 33), 8)
                           for budget in (260, 1040)]
    if scope in ("all", "stage2"):
        specifications += [(f"stage2_r{retention}_n{noise}", 1040, retention, noise,
                            tuple(range(11, 111, 11)), 36)
                           for retention in (100, 50, 25) for noise in (0, 5, 10)]
    groups = []
    for name, budget, retention, noise, seeds, multiplicity in specifications:
        if condition and f"r{retention}_n{noise}" != condition:
            continue
        paths = [root / "runs" / f"{event}_fold{fold}_b{budget}_r{retention}_n{noise}_{model}_seed{seed}.json"
                 for event in events for fold in (1, 2, 3) for model in MODELS for seed in seeds]
        groups.append({"name": name, "paths": paths, "seeds": seeds, "budget": budget,
                       "retention": retention / 100, "noise": noise / 100,
                       "multiplicity": multiplicity, "folds_by_event": {e: (1, 2, 3) for e in events}})
    return groups


def empirical_coverage(groups: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for group in groups:
        paths = group["paths"]
        complete = sum(p.exists() and p.with_suffix(".csv").exists() and p.with_suffix(".pt").exists()
                       for p in paths)
        rows.append({"group": group["name"], "expected_fits": len(paths), "complete_artifact_triples": complete,
                     "missing_artifact_triples": len(paths) - complete,
                     "complete": complete == len(paths)})
    return pd.DataFrame(rows)


def audit_nested_empirical_inputs(root: Path) -> dict:
    """Audit complete noise/sparsity inputs using one paired representative model."""
    root = long_path(root)
    checked = 0
    for event in ("harvey", "csdc", "ottawa", "sydney"):
        for fold in (1, 2, 3):
            for seed in range(11, 111, 11):
                masks = {}
                for retention in (100, 50, 25):
                    baseline_mask, baseline_actual, baseline_d = None, None, None
                    for noise in (0, 5, 10):
                        stem = root / "runs" / f"{event}_fold{fold}_b1040_r{retention}_n{noise}_TransPINN_seed{seed}"
                        payload = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
                        frame = pd.read_csv(stem.with_suffix(".csv"))
                        mask = frame["train_observed"].to_numpy(bool)
                        actual = frame[[f"actual_{c}" for c in CHANNELS]].to_numpy(float)
                        denominator = np.asarray(payload["metric_denominator"], dtype=float)
                        if baseline_mask is None:
                            baseline_mask, baseline_actual, baseline_d = mask, actual, denominator
                        else:
                            np.testing.assert_array_equal(mask, baseline_mask)
                            np.testing.assert_allclose(actual, baseline_actual, rtol=0, atol=0)
                            np.testing.assert_allclose(denominator, baseline_d, rtol=0, atol=0)
                        labels = frame[[f"training_label_{c}" for c in CHANNELS]].to_numpy(float)
                        if np.any(np.isfinite(labels[~mask])):
                            raise ValueError("Unavailable training values must be excluded, not encoded as zero labels")
                        if noise == 0:
                            np.testing.assert_allclose(labels[mask], actual[mask], rtol=3e-6, atol=1e-10)
                        if retention == 100:
                            np.testing.assert_array_equal(mask, frame["split"].eq("train").to_numpy())
                        checked += 1
                    masks[retention] = baseline_mask
                if np.any(masks[25] & ~masks[50]) or np.any(masks[50] & ~masks[100]):
                    raise ValueError("Sparse observation masks are not nested")
    return {"representative_inputs_checked": checked, "noise_levels_share_masks": True,
            "retention_masks_nested": True, "unobserved_labels_excluded": True,
            "clean_observed_labels_match_source": True}


def analyze_empirical(root: Path, output: Path, *, scope: str = "all", condition: str | None = None,
                      replicates: int = 50000, coverage_only: bool = False) -> None:
    root, output = long_path(root), long_path(output)
    groups = empirical_groups(root, scope, condition)
    if not groups:
        raise ValueError("No prespecified groups match request")
    coverage = empirical_coverage(groups)
    print(coverage.to_string(index=False), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output / "coverage.csv", index=False)
    if coverage_only:
        return
    if not coverage["complete"].all():
        raise ValueError("Requested experiment grids are incomplete; use --coverage-only for progress")
    if scope in ("all", "stage2") and condition is None:
        cross_condition = audit_nested_empirical_inputs(root)
        (output / "cross_condition_input_audit.json").write_text(
            json.dumps(cross_condition, indent=2), encoding="utf-8")
    summaries, intervals = [], []
    for group_number, group in enumerate(groups):
        records = []
        for path in group["paths"]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in ("budget", "retention", "noise"):
                np.testing.assert_allclose(payload[key], group[key], rtol=0, atol=1e-12)
            record = audit_run(payload, path.with_suffix(".csv"), expected_epochs=group["budget"])
            record.metadata["record_sha256"] = sha256(path)
            records.append(record)
        tables = analyze_group(records, output / group["name"], seeds=group["seeds"],
            folds_by_event=group["folds_by_event"], replicates=replicates,
            # Group-specific bootstrap randomness is set from the fixed group name,
            # independent of which other groups are requested on this invocation.
            random_seed=20260919 + int(hashlib.sha256(group["name"].encode()).hexdigest()[:6], 16),
            multiplicity=group["multiplicity"])
        for name, collection in (("summary", summaries), ("paired_intervals", intervals)):
            table = tables[name].copy()
            table.insert(0, "group", group["name"])
            collection.append(table)
        print(f"Audited {group['name']}: {len(records)} fits", flush=True)
        print(tables["summary"].to_string(index=False), flush=True)
    pd.concat(summaries, ignore_index=True).to_csv(output / f"{scope}_summary.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(output / f"{scope}_paired_intervals.csv", index=False)


def combine_empirical_reports(root: Path, output: Path) -> None:
    """Combine complete per-group CSV artifacts without rerunning any inference."""
    root, output = long_path(root), long_path(output)
    groups = empirical_groups(root)
    combined = {"summary": [], "paired_intervals": []}
    audits = {}
    for group in groups:
        directory = output / group["name"]
        audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
        if not audit["complete"] or audit["fit_count"] != len(group["paths"]):
            raise ValueError(f"Incomplete audited group: {group['name']}")
        audits[group["name"]] = audit
        for name in combined:
            table = pd.read_csv(directory / f"{name}.csv")
            table.insert(0, "group", group["name"])
            combined[name].append(table)
    for name, tables in combined.items():
        pd.concat(tables, ignore_index=True).to_csv(output / f"all_{name}.csv", index=False)
    report = {"unique_empirical_fits": 5580, "stage1_logical_fits": 360,
        "stage2_logical_fits": 5400, "shared_clean_1040_fits": 180,
        "rebootstrap_performed": False, "group_audits": audits}
    (output / "combined_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Combined 11 completed empirical groups; 5580 unique fits; no repeated bootstrap.")


def synthetic_run_path(root: Path, dataset: str, model: str, seed: int) -> Path:
    return long_path(root) / "synthetic_neural" / "runs" / f"{dataset}_fold1_b1040_r100_n0_{model}_seed{seed}.json"


def load_dde_run(root: Path, dataset: str, seed: int, horizon: int) -> AuditedRun:
    root = long_path(root)
    data_path = root / "synthetic" / "datasets" / f"{dataset}.npz"
    source = np.load(data_path, allow_pickle=False)
    destination = root / "synthetic" / "dde_baseline" / dataset
    record_path = destination / f"seed{seed}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    predictions_path = destination / f"seed{seed}.npz"
    saved = np.load(predictions_path, allow_pickle=False)
    actual, predicted = source["y"], saved["predicted"]
    if actual.shape != (200, 2) or predicted.shape != actual.shape:
        raise ValueError("Unexpected DDE observation dimensions")
    if not np.isfinite(predicted).all() or np.any(predicted < 0):
        raise ValueError("Invalid DDE forecasts")
    denominator = np.maximum(np.sqrt(np.mean(actual[:100] ** 2, axis=0)), 1.)
    validation = float(normalized_metrics(actual[100:120], predicted[100:120], denominator)["nrmse"].mean())
    np.testing.assert_allclose(validation, record["validation_nrmse"], rtol=2e-7, atol=1e-12)
    stop = 120 + horizon
    increment = normalized_metrics(actual[120:stop], predicted[120:stop], denominator)
    actual_cumulative, predicted_cumulative = actual.cumsum(0), predicted.cumsum(0)
    cumulative_denominator = np.maximum(np.sqrt(np.mean(actual_cumulative[:100] ** 2, axis=0)), 1.)
    cumulative = normalized_metrics(actual_cumulative[120:stop], predicted_cumulative[120:stop], cumulative_denominator)
    frame = pd.DataFrame({"split": ["train"] * 100 + ["validation"] * 20 + ["test"] * horizon,
        "actual_rumor": actual[:stop, 0], "actual_debunk": actual[:stop, 1],
        "predicted_rumor": predicted[:stop, 0], "predicted_debunk": predicted[:stop, 1],
        "train_observed": np.arange(stop) < 100,
        "training_label_rumor": np.where(np.arange(stop) < 100, actual[:stop, 0], np.nan),
        "training_label_debunk": np.where(np.arange(stop) < 100, actual[:stop, 1], np.nan)})
    rows = [{"event": dataset, "fold": 1, "model": "DDE-LS", "seed": seed, "channel": c,
             **{k: float(v[j]) for k, v in increment.items()},
             **{f"cumulative_{k}": float(v[j]) for k, v in cumulative.items()}}
            for j, c in enumerate(CHANNELS)]
    metadata = {"run": {"runtime_kind": "serial_cpu_numerical_inverse", "parameter_count": 8},
        "budget": 400, "retention": 1., "noise": 0., "mask_seed": 0, "noise_seed": 0,
        "input_sha256": sha256(data_path), "record_sha256": sha256(record_path),
        "prediction_sha256": sha256(predictions_path), "dde_record": record}
    return AuditedRun(dataset, 1, "DDE-LS", seed, validation, float(record["runtime_seconds"]),
                      denominator, frame, metadata, rows)


def synthetic_recovery(root: Path, dataset: str, model: str, seed: int, horizon: int) -> dict:
    """Recovery metrics use truth only after forecast/validation selection exists."""
    root = long_path(root)
    data = np.load(root / "synthetic" / "datasets" / f"{dataset}.npz", allow_pickle=False)
    truth, true_states = data["theta"], data["states"]
    row = {"event": dataset, "fold": 1, "model": model, "seed": seed, "horizon": horizon,
           "generator_seed": int(data["generator_seed"]), "partition": str(data["partition"]),
           "scenario": str(data["scenario"]), "has_latent_state_or_parameter_estimates": not model.startswith("Data-")}
    if model.startswith("Data-"):
        return row
    if model == "DDE-LS":
        stem = root / "synthetic" / "dde_baseline" / dataset / f"seed{seed}"
        record = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        theta = np.array([record["parameters"][name] for name in PARAMETER_NAMES])
        states = np.load(stem.with_suffix(".npz"), allow_pickle=False)["states"]
        row.update({key: record[key] for key in ("success", "status", "nfev", "actual_residual_calls")})
    else:
        path = synthetic_run_path(root, dataset, model, seed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        theta = np.asarray(payload["run"]["parameters"], dtype=float).copy()
        factor = np.asarray(payload["synthetic"]["gain_raw_to_normalized"], dtype=float)
        expected_factor = 10000 * float(np.diff(data["bin_edges"])[0]) / np.maximum(data["y"][:100].sum(0), 1.)
        np.testing.assert_allclose(factor, expected_factor, rtol=2e-7)
        theta[6:8] /= factor
        np.testing.assert_allclose(theta, payload["parameters_raw"], rtol=3e-6)
        np.testing.assert_allclose(payload["synthetic"]["theta"], truth, rtol=0, atol=0)
        latent = np.load(path.with_suffix(".latent.npz"), allow_pickle=False)
        np.testing.assert_allclose(latent["actual"], true_states, rtol=0, atol=0)
        states = latent["predicted"]
        relative_l2 = float(np.sqrt(np.mean(((theta - truth) / truth) ** 2)))
        np.testing.assert_allclose(relative_l2, payload["parameter_relative_l2"], rtol=3e-6)
    if states.shape != true_states.shape or theta.shape != (8,) or np.any(theta <= 0):
        raise ValueError("Invalid synthetic parameter/state predictions")
    if not np.isfinite(states).all() or np.any(states < -1e-8):
        raise ValueError("Invalid synthetic hidden states")
    stop = 120 + horizon
    scale = np.maximum(np.sqrt(np.mean(true_states[1:101] ** 2, axis=0)), 1e-8)
    state_error = np.sqrt(np.mean((states[121:stop + 1] - true_states[121:stop + 1]) ** 2, axis=0)) / scale
    row.update({"state_nrmse": float(state_error.mean()),
        "latent_E_H_U_M1_M2_nrmse": float(state_error[[2, 3, 6, 7, 8]].mean()),
        "parameter_log_rmse": float(np.sqrt(np.mean(np.log(theta / truth) ** 2))),
        "parameter_mean_relative_error": float(np.mean(abs(theta / truth - 1))),
        "parameter_relative_l2": float(np.sqrt(np.mean((theta / truth - 1) ** 2)))})
    row.update({f"relative_error_{name}": float(abs(theta[j] / truth[j] - 1)) for j, name in enumerate(PARAMETER_NAMES)})
    if model != "DDE-LS":
        for key in ("state_nrmse", "latent_E_H_U_M1_M2_nrmse"):
            reported = payload.get("horizons", {}).get(str(horizon), {}).get(key)
            if reported is not None:
                np.testing.assert_allclose(row[key], reported, rtol=3e-6, atol=1e-10)
        for key in ("parameter_log_rmse", "parameter_mean_relative_error"):
            if key in payload:
                np.testing.assert_allclose(row[key], payload[key], rtol=3e-6, atol=1e-10)
    return row


def synthetic_coverage(root: Path, partition: str = "all") -> pd.DataFrame:
    root = long_path(root)
    manifest = json.loads((root / "synthetic" / "dataset_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        if partition != "all" and item["partition"] != partition:
            continue
        dataset = item["dataset"]
        for model in (*MODELS, "DDE-LS"):
            complete = 0
            for seed in (11, 22, 33):
                if model == "DDE-LS":
                    stem = root / "synthetic" / "dde_baseline" / dataset / f"seed{seed}"
                    available = all(stem.with_suffix(ext).exists() for ext in (".json", ".npz"))
                else:
                    path = synthetic_run_path(root, dataset, model, seed)
                    extensions = [".json", ".csv", ".pt", ".trace.csv"]
                    if not model.startswith("Data-"):
                        extensions.append(".latent.npz")
                    available = all(path.with_suffix(ext).exists() for ext in extensions)
                complete += int(available)
            rows.append({"dataset": dataset, "partition": item["partition"], "scenario": item["scenario"],
                         "model": model, "expected_fits": 3, "complete_fits": complete, "complete": complete == 3})
    return pd.DataFrame(rows)


def analyze_synthetic(root: Path, output: Path, *, partition: str = "all", replicates: int = 50000,
                      coverage_only: bool = False) -> None:
    root, output = long_path(root), long_path(output)
    coverage = synthetic_coverage(root, partition)
    print(coverage.groupby(["partition", "scenario", "model"])[["expected_fits", "complete_fits"]].sum().to_string(), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output / "coverage.csv", index=False)
    if coverage_only:
        return
    if not coverage["complete"].all():
        raise ValueError("Requested synthetic grids are incomplete")
    manifest = json.loads((root / "synthetic" / "dataset_manifest.json").read_text(encoding="utf-8"))
    models, seeds = (*MODELS, "DDE-LS"), (11, 22, 33)
    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in ("summary", "paired_intervals", "recovery_metrics", "recovery_summary")}
    for subset in ("development", "test"):
        if partition not in ("all", subset):
            continue
        for scenario in ("consistent", "decay_mismatch"):
            items = [item for item in manifest if item["partition"] == subset and item["scenario"] == scenario]
            expected_trajectories = 3 if subset == "development" else 10
            if len(items) != expected_trajectories:
                raise ValueError("Synthetic generation grid is incomplete")
            events = [item["dataset"] for item in items]
            for horizon in (20, 40, 80):
                name = f"{subset}_{scenario}_h{horizon}"
                records, recovery_rows = [], []
                for item in items:
                    dataset = item["dataset"]
                    data_path = root / "synthetic" / "datasets" / f"{dataset}.npz"
                    if sha256(data_path) != item["sha256"]:
                        raise ValueError("Frozen synthetic data hash changed")
                    source = np.load(data_path, allow_pickle=False)
                    for model in models:
                        for seed in seeds:
                            if model == "DDE-LS":
                                record = load_dde_run(root, dataset, seed, horizon)
                            else:
                                path = synthetic_run_path(root, dataset, model, seed)
                                payload = json.loads(path.read_text(encoding="utf-8"))
                                if payload["input_sha256"] != item["sha256"]:
                                    raise ValueError("Neural fit does not match frozen synthetic data")
                                record = audit_run(payload, path.with_suffix(".csv"), expected_epochs=1040, horizon=horizon)
                                np.testing.assert_allclose(record.predictions[[f"actual_{c}" for c in CHANNELS]],
                                                           source["y"][:120 + horizon], rtol=3e-15, atol=1e-12)
                                computed_nrmse = float(np.mean([r["nrmse"] for r in record.metrics]))
                                np.testing.assert_allclose(computed_nrmse, payload["horizons"][str(horizon)]["nrmse"], rtol=2e-7, atol=1e-12)
                            records.append(record)
                            recovery_rows.append(synthetic_recovery(root, dataset, model, seed, horizon))
                tables = analyze_group(records, output / name, models=models, seeds=seeds,
                    folds_by_event={event: (1,) for event in events}, replicates=replicates,
                    random_seed=20260919 + int(hashlib.sha256(name.encode()).hexdigest()[:6], 16),
                    multiplicity=24)
                tables["paired_intervals"]["held_out_synthetic_partition"] = subset == "test"
                if subset == "development":
                    # Development trajectories are diagnostics. They never join
                    # the held-out test family or enlarge its effective sample.
                    tables["paired_intervals"]["primary_family"] = False
                    tables["paired_intervals"]["family_size"] = 0
                    tables["paired_intervals"]["transpinn_advantage_after_correction"] = False
                    for column in ("simultaneous_low", "simultaneous_high", "familywise_confidence"):
                        tables["paired_intervals"][column] = np.nan
                tables["paired_intervals"].to_csv(output / name / "paired_intervals.csv", index=False)
                recovery = pd.DataFrame(recovery_rows)
                choices = tables["selected_initializations"][["event", "fold", "model", "seed"]].assign(selected=True)
                recovery = recovery.merge(choices, on=["event", "fold", "model", "seed"], how="left")
                recovery["selected"] = recovery["selected"].eq(True)
                quantities = [key for key in recovery if key.startswith("relative_error_")] + [
                    "state_nrmse", "latent_E_H_U_M1_M2_nrmse", "parameter_log_rmse",
                    "parameter_mean_relative_error", "parameter_relative_l2"]
                per_seed = recovery.groupby(["model", "seed"])[quantities].mean()
                recovery_summary = per_seed.groupby("model")[quantities].agg(["mean", "std"])
                recovery_summary.columns = [f"{key}_{stat}" for key, stat in recovery_summary.columns]
                selected_recovery = recovery[recovery["selected"]].groupby("model")[quantities].mean().add_prefix("selected_")
                tables["recovery_summary"] = recovery_summary.join(selected_recovery).reset_index()
                tables["recovery_metrics"] = recovery
                for table_name in ("recovery_metrics", "recovery_summary"):
                    tables[table_name].to_csv(output / name / f"{table_name}.csv", index=False)
                for table_name, aggregate in collected.items():
                    table = tables[table_name].copy()
                    for key, value in reversed((("partition", subset), ("scenario", scenario), ("horizon", horizon))):
                        if key not in table:
                            table.insert(0, key, value)
                    aggregate.append(table)
                print(f"Audited {name}: {len(records)} fits including independently fitted DDE", flush=True)
                print(tables["summary"][["model", "nrmse_mean", "nrmse_std", "selected_nrmse"]].to_string(index=False), flush=True)
    for name, tables in collected.items():
        pd.concat(tables, ignore_index=True).to_csv(output / f"{partition}_{name}.csv", index=False)


def self_test() -> None:
    """Small, deterministic checks of pairing, nesting and weighting identities."""
    folds = {"a": (1, 2), "b": (1,)}
    constants = np.array([1., 4., 9., 16., 25.])
    errors = {(e, f): np.broadcast_to(constants[None, :, None, None], (3, 5, 2, 11)).copy()
              for e, fs in folds.items() for f in fs}
    point, draws = hierarchical_bootstrap(errors, folds, replicates=251, batch_size=32)
    np.testing.assert_allclose(point, np.sqrt(constants), rtol=0, atol=1e-14)
    np.testing.assert_allclose(draws, np.broadcast_to(point, draws.shape), rtol=0, atol=1e-14)
    # Equal methods remain identical in every draw despite nonconstant errors.
    random = np.random.default_rng(77)
    paired = {(e, f): np.repeat(random.uniform(.1, 2., (3, 1, 2, 11)), 5, axis=1)
              for e, fs in folds.items() for f in fs}
    _, paired_draws = hierarchical_bootstrap(paired, folds, replicates=251, batch_size=32)
    np.testing.assert_allclose(paired_draws, np.repeat(paired_draws[:, :1], 5, axis=1), rtol=0, atol=0)
    # Unequal numbers of folds must not change the equal-event point weights.
    unequal = {("a", 1): np.ones((1, 5, 2, 4)), ("a", 2): np.ones((1, 5, 2, 4)),
               ("b", 1): np.full((1, 5, 2, 4), 9.)}
    point, _ = hierarchical_bootstrap(unequal, folds, replicates=100, batch_size=13)
    np.testing.assert_allclose(point, 2., rtol=0, atol=1e-14)
    groups = empirical_groups(Path("unused"))
    assert sum(len(g["paths"]) for g in groups if g["name"].startswith("stage1")) == 360
    assert sum(len(g["paths"]) for g in groups if g["name"].startswith("stage2")) == 5400
    assert len({p for g in groups for p in g["paths"]}) == 5580
    # Give seed11 better validation but much worse test error. Selection must
    # choose seed11, proving that test performance cannot drive this decision.
    records = []
    for event, event_folds in folds.items():
        for fold in event_folds:
            for model in MODELS:
                for seed in (11, 22):
                    value = 2. if seed == 11 else .1
                    frame = pd.DataFrame({"split": ["train"] * 4 + ["validation"] * 2 + ["test"] * 2,
                        "actual_rumor": np.ones(8), "actual_debunk": np.ones(8),
                        "predicted_rumor": np.full(8, 1 + value), "predicted_debunk": np.full(8, 1 + value)})
                    rows = [{"event": event, "fold": fold, "model": model, "seed": seed,
                             "channel": c, **{metric: value for metric in METRICS}} for c in CHANNELS]
                    records.append(AuditedRun(event, fold, model, seed, seed / 100, 1., np.ones(2), frame, {}, rows))
    check_complete_grid(records, MODELS, (11, 22), folds)
    tables = balanced_summaries(records)
    assert tables["selected_initializations"]["seed"].eq(11).all()
    np.testing.assert_allclose(tables["summary"]["selected_nrmse"], 2.)
    np.testing.assert_allclose(tables["summary"]["nrmse_mean"], 1.05)
    try:
        check_complete_grid(records[:-1], MODELS, (11, 22), folds)
    except ValueError as exc:
        assert "Incomplete" in str(exc)
    else:
        raise AssertionError("Missing run was silently accepted")
    # Exercise the artifact audit independently of any real training files.
    with tempfile.TemporaryDirectory(prefix="transpinn-analysis-fixture-") as temporary:
        path = Path(temporary) / "fixture.csv"
        frame = pd.DataFrame({"split": ["train"] * 4 + ["validation"] * 2 + ["test"] * 2,
            "actual_rumor": np.ones(8), "actual_debunk": np.ones(8),
            "predicted_rumor": np.full(8, 1.1), "predicted_debunk": np.full(8, 1.1),
            "train_observed": np.arange(8) < 4})
        frame.to_csv(path, index=False)
        trace = pd.DataFrame({"epoch": np.arange(10, 261, 10), "validation_nrmse": .1})
        trace.to_csv(path.with_suffix(".trace.csv"), index=False)
        payload = {"run": {"event": "fixture", "fold": 1, "model": "TransPINN", "seed": 11,
            "validation_score": .1, "runtime_seconds": 1., "best_epoch": 10,
            "epochs_completed": 260, "parameter_count": 18619}, "metric_denominator": [1., 1.]}
        audited = audit_run(payload, path, expected_epochs=260)
        np.testing.assert_allclose([row["nrmse"] for row in audited.metrics], .1)
        trace.loc[4, "validation_nrmse"] = .05
        trace.to_csv(path.with_suffix(".trace.csv"), index=False)
        try:
            audit_run(payload, path, expected_epochs=260)
        except AssertionError:
            pass
        else:
            raise AssertionError("A nonminimal validation checkpoint was accepted")
    print("Self-test passed: exact pairing, equal-event weights, validation-only selection, "
          "checkpoint trace audit, missing-run rejection, and 5580 unique prescribed fits.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", type=Path,
        help="JSON with runs [{json,csv}], folds_by_event, models, seeds, optional epochs/multiplicity")
    mode.add_argument("--empirical-root", type=Path)
    mode.add_argument("--synthetic-root", type=Path, help="Common robustness root containing synthetic/ and synthetic_neural/")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=50000)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--scope", choices=("all", "stage1", "stage2"), default="all")
    parser.add_argument("--condition", help="Optional fixed cell, e.g. r50_n5; correction still covers all9 cells")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--combine-existing", action="store_true", help="Combine existing complete empirical reports without rerunning bootstrap")
    parser.add_argument("--partition", choices=("all", "development", "test"), default="all")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.empirical_root:
        if args.combine_existing:
            combine_empirical_reports(args.empirical_root, args.output or args.empirical_root / "analysis")
            return
        analyze_empirical(args.empirical_root, args.output or args.empirical_root / "analysis",
            scope=args.scope, condition=args.condition, replicates=args.bootstrap_reps,
            coverage_only=args.coverage_only)
        return
    if args.synthetic_root:
        analyze_synthetic(args.synthetic_root, args.output or args.synthetic_root / "synthetic" / "comparison",
            partition=args.partition, replicates=args.bootstrap_reps, coverage_only=args.coverage_only)
        return
    if args.output is None:
        parser.error("--output required with --index")
    index = json.loads(long_path(args.index).read_text(encoding="utf-8"))
    records = []
    for item in index["runs"]:
        metadata_path = long_path(Path(item["json"]))
        prediction_path = long_path(Path(item.get("csv", str(metadata_path.with_suffix(".csv")))))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        records.append(audit_run(payload, prediction_path, expected_epochs=index.get("epochs"),
            expected_parameter_count=index.get("parameter_count", 18619), horizon=args.horizon))
    tables = analyze_group(records, args.output, models=index.get("models", MODELS),
        seeds=index["seeds"], folds_by_event=index["folds_by_event"], replicates=args.bootstrap_reps,
        random_seed=index.get("bootstrap_seed", 20260919), multiplicity=index.get("multiplicity", 36))
    print(tables["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
