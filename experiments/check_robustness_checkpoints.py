"""Bounded CPU reload audit of saved independent-batch checkpoints.

Checks first/last completed real-data fits for each architecture and two existing
synthetic smoke fits. It performs no optimization and never uses the GPU.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import robustness_models as models
import run_multievent_baselines as base
import run_robustness as runner


ROOT = runner.ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def completed_paths(folder, name):
    return sorted(path for path in folder.glob(f"*_{name}_seed*.json")
                  if path.with_suffix(".pt").exists() and path.with_suffix(".csv").exists())


def empirical_case(payload):
    record = payload["run"]
    fold = next(item for item in base.load_event_folds(record["event"]) if item[0] == record["fold"])
    case = runner.empirical_case(record["event"], fold[0], fold[1], fold[2], record["seed"],
        int(payload["budget"]), round(100 * payload["retention"]), round(100 * payload["noise"]))
    assert case.input_sha256 == payload["input_sha256"]
    return case


def audit_checkpoint(path, case, smoke=False):
    payload = read(path)
    record = payload["run"]
    model = models.make_model(record["model"], case.reporting_bounds)
    checkpoint = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    expected_bounds = np.array([[0.25, 12], [0.25, 12]]) if case.reporting_bounds is None else case.reporting_bounds
    np.testing.assert_allclose(model.reporting_bounds.numpy(), expected_bounds, rtol=1e-7)
    times = torch.as_tensor(case.times, dtype=torch.float32)
    quadrature = torch.as_tensor(case.quadrature, dtype=torch.float32)
    with torch.no_grad():
        state, theta = model(times)
        normalized = runner.observations(state[None], theta[None], model.physical,
                                         case.split[-1], quadrature)[0].numpy()
    predictions = normalized * case.scales
    saved_frame = pd.read_csv(path.with_suffix(".csv"))
    saved_predictions = saved_frame[["predicted_rumor", "predicted_debunk"]].to_numpy(float)
    delta = abs(predictions - saved_predictions)
    normalized_delta = delta / case.denominator
    maximum = float(normalized_delta.max())
    assert maximum <= 3e-5, (str(path), maximum)
    np.testing.assert_allclose(case.denominator, payload["metric_denominator"], rtol=1e-12)
    np.testing.assert_allclose(case.scales, payload["scales"], rtol=1e-12)
    result = {
        "record": str(path.relative_to(ROOT)), "model": record["model"],
        "event": record["event"], "seed": record["seed"],
        "budget": record["epochs_completed"], "smoke_run": smoke,
        "max_absolute_prediction_difference": float(delta.max()),
        "max_prediction_difference_over_training_RMS": maximum,
        "max_relative_bounds_difference": float(np.max(abs(model.reporting_bounds.numpy() / expected_bounds - 1))),
        "checkpoint_sha256": sha256(path.with_suffix(".pt")),
        "prediction_sha256": sha256(path.with_suffix(".csv")),
        "record_sha256": sha256(path),
    }
    if model.physical:
        np.testing.assert_allclose(theta.numpy(), record["parameters"], rtol=3e-6, atol=1e-7)
        result["saved_physical_coefficients_reproduced"] = True
    if smoke:
        factor = np.asarray(payload["synthetic"]["gain_raw_to_normalized"])
        raw_theta = theta.numpy().astype(float)
        raw_theta[6:8] /= factor
        np.testing.assert_allclose(raw_theta, payload["parameters_raw"], rtol=3e-6)
        # Deliberately construct different bounds. The checkpoint's registered
        # bounds must replace them, proving they survive serialization.
        wrong_initial_bounds = [[0.02, 0.7], [0.04, 0.8]]
        reloaded = models.make_model(record["model"], wrong_initial_bounds)
        reloaded.load_state_dict(checkpoint, strict=True)
        reloaded.eval()
        with torch.no_grad():
            restored_state, restored_theta = reloaded(times)
        torch.testing.assert_close(restored_state, state, rtol=0, atol=0)
        torch.testing.assert_close(restored_theta, theta, rtol=0, atol=0)
        result.update({"reporting_bounds": expected_bounds.tolist(),
            "raw_reporting_coefficients": raw_theta[6:8].tolist(),
            "normalized_reporting_coefficients": theta[6:8].tolist(),
            "raw_to_normalized_factors": factor.tolist(),
            "checkpoint_bounds_override_constructor_bounds": True,
            "raw_coefficient_round_trip_verified": True})
    return result


def main():
    torch.set_num_threads(2)
    models.configure_attention()
    selected = []
    empirical_folder = ROOT / "output/robustness/empirical/runs"
    available = {}
    for name in models.MODEL_NAMES:
        paths = completed_paths(empirical_folder, name)
        assert len(paths) >= 2, (name, len(paths))
        available[name] = len(paths)
        selected.extend((path, empirical_case(read(path)), False) for path in (paths[0], paths[-1]))
    smoke_folder = ROOT / "tmp/robustness_cuda_smoke/synthetic_neural/runs"
    for name in ("TransPINN", "Matched-PINN"):
        paths = completed_paths(smoke_folder, name)
        assert paths
        path = paths[0]
        payload = read(path)
        source = ROOT / payload["synthetic"]["dataset"]
        case = runner.synthetic_case(source, payload["run"]["seed"])
        selected.append((path, case, True))
    records = []
    for path, case, smoke in selected:
        record = audit_checkpoint(path, case, smoke)
        records.append(record)
        print(record["record"], f"normalized difference={record['max_prediction_difference_over_training_RMS']:.3g}", flush=True)
    sources = [Path(__file__), Path(runner.__file__), Path(models.__file__), Path(base.__file__),
               Path(models.fair.__file__)]
    report = {
        "passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "device": "cpu", "threads": 2,
        "selection": "First and last completed empirical JSON in lexical order per model, snapshotted before checking",
        "available_empirical_counts_at_selection": available,
        "empirical_checkpoints_checked": 10, "synthetic_smoke_checkpoints_checked": 2,
        "maximum_allowed_RMS_normalized_difference": 3e-5,
        "maximum_RMS_normalized_difference": max(item["max_prediction_difference_over_training_RMS"] for item in records),
        "synthetic_scope": "Both existing smoke checkpoints use the same development dataset, with distinct channel bounds. Different inter-case bounds were checked separately by the serial/batched parity tests.",
        "source_sha256": {path.name: sha256(path) for path in sources},
        "records": records,
    }
    target = ROOT / "output/benchmark_design/checkpoint_reload_audit.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: 12 CPU reloads, maximum normalized difference {report['maximum_RMS_normalized_difference']:.3g}", flush=True)


if __name__ == "__main__":
    main()
