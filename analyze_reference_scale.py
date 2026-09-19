"""Secondary, post-hoc normalization diagnostic for the complete robustness grid.

The fixed reference is max(RMS(original complete training bins), 1), separately
per event/fold/channel. It is used only to score existing predictions. Original
training inputs, checkpoints, validation scores, selected complete replicates,
and all primary analyses remain unchanged. This diagnoses normalization effects;
it cannot isolate initialization selection from mask/noise realization selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_robustness import CHANNELS, MODELS, long_path


def sha256(path: Path) -> str:
    return hashlib.sha256(long_path(path).read_bytes()).hexdigest()


def analyze(root: Path, destination: Path | None = None) -> pd.DataFrame:
    root = long_path(root)
    destination = long_path(destination or root / "secondary" / "reference_rms")
    source = root / "empirical"
    primary = source / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    protocol = {
        "analysis_role": "Secondary post-hoc scoring-scale diagnostic, authorized after stage1 results",
        "reference_denominator": "max(RMS(all original unperturbed training bins),1), per event/fold/channel",
        "metric": "Incremental test NRMSE only",
        "training_information": "Full original training labels enter diagnostic scoring only; no training or checkpoint changes",
        "selection": "Reuse original validation-selected complete replicate for every event/fold/model",
        "aggregation": "Equal channels, then folds, then events; sample SD across ten paired replicate seeds",
        "selected_SD": "Not defined across seeds after selecting one replicate; left missing",
        "coverage": "All9 retention/noise conditions, all5 models, both all-seed and original-selected views",
        "inference": "Point estimates and sample SD only; no new intervals, p-values, or significance claims",
        "limitation": "Shared scoring scales do not remove selection of different mask/noise realizations",
        "source_sha256": sha256(Path(__file__)),
    }
    (destination / "diagnostic_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    all_rows, all_selected, summaries, differences, provenance = [], [], [], [], []
    reference_cache = {}
    for retention in (100, 50, 25):
        for noise in (0, 5, 10):
            condition = f"r{retention}_n{noise}"
            directory = primary / f"stage2_{condition}"
            audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
            if audit.get("complete") is not True or audit.get("fit_count") != 600:
                raise ValueError(f"Primary condition not completely audited: {condition}")
            selected_path = directory / "selected_initializations.csv"
            summary_path = directory / "summary.csv"
            choices = pd.read_csv(selected_path)
            primary_summary = pd.read_csv(summary_path).set_index("model")
            recorded = pd.read_csv(directory / "metrics.csv").set_index(["event", "fold", "model", "seed", "channel"])
            expected_selected = {(event, fold, model) for event in ("harvey", "csdc", "ottawa", "sydney")
                                 for fold in (1, 2, 3) for model in MODELS}
            actual_selected = set(choices[["event", "fold", "model"]].itertuples(index=False, name=None))
            if len(choices) != 60 or actual_selected != expected_selected:
                raise ValueError(f"Incomplete original validation selections: {condition}")
            condition_rows = []
            for event in ("harvey", "csdc", "ottawa", "sydney"):
                for fold in (1, 2, 3):
                    for model in MODELS:
                        for seed in range(11, 111, 11):
                            stem = source / "runs" / f"{event}_fold{fold}_b1040_{condition}_{model}_seed{seed}"
                            payload = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
                            frame = pd.read_csv(stem.with_suffix(".csv"))
                            train, test = frame["split"].eq("train"), frame["split"].eq("test")
                            actual = frame[[f"actual_{c}" for c in CHANNELS]].to_numpy(float)
                            predicted = frame[[f"predicted_{c}" for c in CHANNELS]].to_numpy(float)
                            reference = np.maximum(np.sqrt(np.mean(actual[train] ** 2, axis=0)), 1.)
                            key = (event, fold)
                            if key not in reference_cache:
                                reference_cache[key] = reference.copy()
                            np.testing.assert_allclose(reference, reference_cache[key], rtol=1e-12, atol=1e-12)
                            original = np.asarray(payload["metric_denominator"], dtype=float)
                            rmse = np.sqrt(np.mean((predicted[test] - actual[test]) ** 2, axis=0))
                            for k, channel in enumerate(CHANNELS):
                                primary_nrmse, reference_nrmse = float(rmse[k] / original[k]), float(rmse[k] / reference[k])
                                np.testing.assert_allclose(primary_nrmse, recorded.loc[(event, fold, model, seed, channel), "nrmse"],
                                                           rtol=2e-7, atol=1e-12)
                                condition_rows.append({"condition": condition, "retention": retention, "noise": noise,
                                    "event": event, "fold": fold, "model": model, "seed": seed, "channel": channel,
                                    "primary_denominator": float(original[k]), "reference_denominator": float(reference[k]),
                                    "primary_to_reference_denominator": float(original[k] / reference[k]),
                                    "primary_nrmse": primary_nrmse, "reference_nrmse": reference_nrmse})
            rows = pd.DataFrame(condition_rows)
            columns = ["primary_nrmse", "reference_nrmse"]
            folds = rows.groupby(["event", "fold", "model", "seed"], as_index=False)[columns].mean()
            events = folds.groupby(["event", "model", "seed"], as_index=False)[columns].mean()
            macro_seed = events.groupby(["model", "seed"], as_index=False)[columns].mean()
            aggregate = macro_seed.groupby("model")[columns].agg(["mean", "std"])
            aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
            chosen = rows.merge(choices[["event", "fold", "model", "seed", "validation_score"]],
                                on=["event", "fold", "model", "seed"], validate="many_to_one")
            selected_fold = chosen.groupby(["event", "fold", "model"], as_index=False)[columns].mean()
            selected_event = selected_fold.groupby(["event", "model"], as_index=False)[columns].mean()
            selected_macro = selected_event.groupby("model")[columns].mean()
            for model in MODELS:
                np.testing.assert_allclose(aggregate.loc[model, "primary_nrmse_mean"], primary_summary.loc[model, "nrmse_mean"], rtol=2e-7)
                np.testing.assert_allclose(aggregate.loc[model, "primary_nrmse_std"], primary_summary.loc[model, "nrmse_std"], rtol=2e-7, atol=1e-12)
                np.testing.assert_allclose(selected_macro.loc[model, "primary_nrmse"], primary_summary.loc[model, "selected_nrmse"], rtol=2e-7)
                for evaluation in ("all_seeds", "validation_selected"):
                    row = {"condition": condition, "retention": retention, "noise": noise,
                           "model": model, "evaluation": evaluation, "fit_count": 120,
                           "selection_candidates_per_event_fold": 10}
                    for metric in columns:
                        row[f"{metric}_mean"] = float(aggregate.loc[model, f"{metric}_mean"] if evaluation == "all_seeds" else selected_macro.loc[model, metric])
                        row[f"{metric}_std"] = float(aggregate.loc[model, f"{metric}_std"]) if evaluation == "all_seeds" else np.nan
                    row["reference_minus_primary_nrmse"] = row["reference_nrmse_mean"] - row["primary_nrmse_mean"]
                    summaries.append(row)
                if retention == 100:
                    # Complete observations have the same primary/reference scale,
                    # even if their training labels receive synthetic noise.
                    np.testing.assert_allclose(aggregate.loc[model, "primary_nrmse_mean"], aggregate.loc[model, "reference_nrmse_mean"], rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(selected_macro.loc[model, "primary_nrmse"], selected_macro.loc[model, "reference_nrmse"], rtol=1e-12, atol=1e-12)
            all_rows.append(rows)
            all_selected.append(chosen)
            provenance.append({"condition": condition, "primary_selection_sha256": sha256(selected_path),
                               "primary_summary_sha256": sha256(summary_path), "audited_fits": 600})
            print(f"Reference-scale diagnostic complete: {condition}, 600 existing fits; original selections retained.", flush=True)
    summary = pd.DataFrame(summaries)
    for (condition, evaluation), block in summary.groupby(["condition", "evaluation"], sort=False):
        block = block.set_index("model")
        for model in MODELS:
            if model == "TransPINN":
                continue
            row = {"condition": condition, "evaluation": evaluation, "baseline": model}
            for scale in ("primary", "reference"):
                column = f"{scale}_nrmse_mean"
                row[f"{scale}_baseline_minus_transpinn_nrmse"] = float(block.loc[model, column] - block.loc["TransPINN", column])
            differences.append(row)
    pd.concat(all_rows, ignore_index=True).to_csv(destination / "per_run_channel_metrics.csv", index=False)
    pd.concat(all_selected, ignore_index=True).to_csv(destination / "original_selected_replicates.csv", index=False)
    summary.to_csv(destination / "summary.csv", index=False)
    pd.DataFrame(differences).to_csv(destination / "paired_point_differences.csv", index=False)
    (destination / "audit.json").write_text(json.dumps({
        "complete": True, "fits_scored": 5400, "per_run_channel_records": 10800,
        "conditions": 9, "models": list(MODELS), "summary_rows": 90,
        "original_primary_values_reproduced": True, "original_selections_unchanged": True,
        "new_training": False, "new_significance_claims": False, "primary_provenance": provenance}, indent=2), encoding="utf-8")
    documentation = (
        "# Secondary reference-scale diagnostic\n\n"
        "This post-hoc diagnostic scores every completed robustness prediction using one common denominator per event, fold and channel: the larger of the complete original training RMS and 1. The complete original training labels are used for diagnostic scoring only. No model is trained again. No checkpoint, original validation score, or selected replicate is changed.\n\n"
        "The complete nine-cell grid includes all five models. summary.csv reports original-scale and common-scale NRMSE. All-seed means use equal channel, fold and event weights. Their standard deviations are sample SD across ten paired initialization and observation-perturbation replicates. Selected values retain the original validation-selected complete replicate. They have no across-seed SD after selection.\n\n"
        "This removes differing scoring denominators from the diagnostic comparison. It does not turn selection across different masks/noise realizations into initialization-only selection. Complete-observation conditions must have identical original and reference scores. No new intervals, p-values or significance claims are introduced, and the frozen primary analyses remain authoritative.\n\n"
        "Files: per_run_channel_metrics.csv contains both denominators and scores. original_selected_replicates.csv identifies the unchanged selections. paired_point_differences.csv uses baseline minus TransPINN, so positive values favor TransPINN. audit.json verifies complete coverage and records hashes of the primary selections and summaries.\n"
    )
    (destination / "README.md").write_text(documentation, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/robustness"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output)
