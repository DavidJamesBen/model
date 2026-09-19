"""Audit and analyse completed rolling-origin multi-event experiments.

This module never fits a model. It reads ``baseline_runs.csv`` and only the
fold-qualified prediction files produced by ``run_multievent_baselines.py``
(for example, ``harvey_fold2_transpinn_seed11.csv``). Legacy fixed-split files
without ``fold<N>`` in their names are never discovered or loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keep the publication figures consistent with the manuscript and with the
# robustness figures.  The explicit TrueType settings ensure that labels and
# symbols remain selectable in the exported PDF.
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
PREDICTIONS = ROOT / "output" / "predictions"
PROCESSED = ROOT / "data" / "processed"
RUNS_PATH = TABLES / "baseline_runs.csv"
RUN_MANIFEST_PATH = TABLES / "baseline_run_manifest.json"
CHANNELS = ("rumor", "debunk")
PREFERRED_EVENTS = ("harvey", "csdc", "ottawa", "sydney")
PREFERRED_MODELS = ("PINN", "LSTM-PINN", "TimePINN", "NoAttn-PINN", "TransPINN")
FORECAST_MODELS = ("LSTM-PINN", "TimePINN", "NoAttn-PINN", "TransPINN")
EVENT_LABELS = {
    "harvey": "Hurricane Harvey",
    "csdc": "CSDC-Rumor",
    "ottawa": "Ottawa shooting",
    "sydney": "Sydney cafe siege",
}
COLORS = {
    "PINN": "#777777",
    "LSTM-PINN": "#2f6da8",
    "TimePINN": "#2f8f55",
    "NoAttn-PINN": "#8b5fbf",
    "TransPINN": "#cf3c42",
}
BOOTSTRAP_SEED = 20260822


@dataclass(frozen=True)
class ExperimentIndex:
    events: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    folds_by_event: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class TestSeries:
    actual: np.ndarray
    predicted: np.ndarray
    denominator: float


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def _ordered(values: Iterable[str], preferred: tuple[str, ...]) -> tuple[str, ...]:
    unique = set(values)
    return tuple([v for v in preferred if v in unique] + sorted(unique - set(preferred)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(model: str) -> str:
    return model.lower().replace("-", "_")


def prediction_path(event: str, fold: int, model: str, seed: int) -> Path:
    """Resolve one rolling-origin prediction without ever consulting old files."""
    path = PREDICTIONS / f"{event}_fold{fold}_{slug(model)}_seed{seed}.csv"
    if f"_fold{fold}_" not in path.name:
        raise AssertionError(f"Prediction path is not fold-qualified: {path}")
    return path


def load_inputs() -> tuple[pd.DataFrame, dict[str, object], ExperimentIndex]:
    if not RUNS_PATH.exists() or not RUN_MANIFEST_PATH.exists():
        raise FileNotFoundError("Rolling-origin run table or manifest is missing")
    runs = pd.read_csv(RUNS_PATH)
    manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    required = {
        "event", "fold", "model", "seed", "channel", "train_bins",
        "validation_bins", "test_bins", "parameter_count", "runtime_seconds",
        "physics_residual_rms", "best_epoch", "nrmse", "nmae", "wape",
        "mae",
    }
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"baseline_runs.csv is missing columns: {sorted(missing)}")
    if runs.empty:
        raise ValueError("baseline_runs.csv is empty")
    for column in ("fold", "seed", "train_bins", "validation_bins", "test_bins"):
        runs[column] = pd.to_numeric(runs[column], errors="raise").astype(int)
    numeric_required = sorted(required - {"event", "model", "channel"})
    if runs[numeric_required].isna().any().any():
        raise ValueError("Run table contains missing required numeric values")
    unexpected_channels = set(runs["channel"]) - set(CHANNELS)
    if unexpected_channels:
        raise ValueError(f"Unexpected channels: {sorted(unexpected_channels)}")

    events = _ordered(runs["event"].astype(str), PREFERRED_EVENTS)
    models = _ordered(runs["model"].astype(str), PREFERRED_MODELS)
    seeds = tuple(sorted(int(value) for value in runs["seed"].unique()))
    folds_by_event = {
        event: tuple(sorted(int(v) for v in runs.loc[runs.event.eq(event), "fold"].unique()))
        for event in events
    }
    index = ExperimentIndex(events, models, seeds, folds_by_event)
    validate_complete_grid(runs, manifest, index)
    return runs, manifest, index


def validate_complete_grid(
    runs: pd.DataFrame, manifest: dict[str, object], index: ExperimentIndex
) -> None:
    keys = ["event", "fold", "model", "seed", "channel"]
    duplicates = runs.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate run/channel rows: "
            f"{runs.loc[duplicates, keys].to_dict('records')[:8]}"
        )
    actual = set(map(tuple, runs[keys].itertuples(index=False, name=None)))
    expected = {
        (event, fold, model, seed, channel)
        for event in index.events
        for fold in index.folds_by_event[event]
        for model in index.models
        for seed in index.seeds
        for channel in CHANNELS
    }
    if actual != expected:
        raise ValueError(
            "Incomplete experiment grid; "
            f"missing={sorted(expected - actual)[:12]}, extra={sorted(actual - expected)[:12]}"
        )
    manifest_seeds = tuple(sorted(int(value) for value in manifest.get("seeds", [])))
    if manifest_seeds != index.seeds:
        raise ValueError(f"Manifest seeds {manifest_seeds} != run seeds {index.seeds}")
    manifest_keys = {
        (str(item["event"]), int(item["fold"]), str(item["model"]), int(item["seed"]))
        for item in manifest.get("runs", [])
    }
    if manifest_keys != {key[:4] for key in expected}:
        raise ValueError("Run manifest does not match the completed experiment grid")


def _read_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing rolling-origin prediction: {path}")
    required = {
        "timestamp", "split", "actual_rumor", "actual_debunk",
        "predicted_rumor", "predicted_debunk",
    }
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    numeric = sorted(required - {"timestamp", "split"})
    if frame.empty or frame[numeric].isna().any().any():
        raise ValueError(f"{path.name} is empty or contains missing values")
    return frame


def validate_predictions_and_metrics(
    runs: pd.DataFrame, index: ExperimentIndex
) -> tuple[pd.DataFrame, dict[tuple[str, int, str, int], pd.DataFrame]]:
    """Audit temporal splits and independently recompute normalized test errors."""
    audited = runs.copy()
    for column in (
        "train_activity_rms", "validation_activity_total", "test_activity_total"
    ):
        audited[column] = np.nan
    cache: dict[tuple[str, int, str, int], pd.DataFrame] = {}
    references: dict[tuple[str, int], pd.DataFrame] = {}

    for event in index.events:
        for fold in index.folds_by_event[event]:
            for model in index.models:
                for seed in index.seeds:
                    key = (event, fold, model, seed)
                    path = prediction_path(*key)
                    frame = _read_prediction(path)
                    cache[key] = frame
                    split = frame["split"].astype(str)
                    if not split.isin(("train", "validation", "test")).all():
                        raise ValueError(f"Unexpected split label in {path.name}")
                    split_code = split.map({"train": 0, "validation": 1, "test": 2}).to_numpy()
                    if np.any(np.diff(split_code) < 0):
                        raise ValueError(f"Non-contiguous temporal split in {path.name}")
                    counts = split.value_counts().to_dict()
                    if any(int(counts.get(part, 0)) <= 0 for part in ("train", "validation", "test")):
                        raise ValueError(f"Empty split in {path.name}: {counts}")

                    unit_rows = audited[
                        audited.event.eq(event) & audited.fold.eq(fold)
                        & audited.model.eq(model) & audited.seed.eq(seed)
                    ]
                    for field, part in (
                        ("train_bins", "train"), ("validation_bins", "validation"),
                        ("test_bins", "test"),
                    ):
                        reported = set(int(v) for v in unit_rows[field])
                        if reported != {int(counts[part])}:
                            raise ValueError(f"{path.name}: {field} {reported} != {counts[part]}")

                    reference_columns = ["timestamp", "split", "actual_rumor", "actual_debunk"]
                    event_fold = (event, fold)
                    if event_fold not in references:
                        references[event_fold] = frame[reference_columns].copy()
                    else:
                        reference = references[event_fold]
                        if len(reference) != len(frame):
                            raise ValueError(f"Prediction lengths disagree for {event}, fold {fold}")
                        if not reference["timestamp"].equals(frame["timestamp"]):
                            raise ValueError(f"Prediction timestamps disagree for {event}, fold {fold}")
                        if not reference["split"].astype(str).equals(split):
                            raise ValueError(f"Prediction splits disagree for {event}, fold {fold}")
                        for channel in CHANNELS:
                            if not np.array_equal(
                                reference[f"actual_{channel}"].to_numpy(float),
                                frame[f"actual_{channel}"].to_numpy(float),
                            ):
                                raise ValueError(
                                    f"Actual {channel} values disagree for {event}, fold {fold}"
                                )

                    for channel in CHANNELS:
                        actual = frame[f"actual_{channel}"].to_numpy(float)
                        predicted = frame[f"predicted_{channel}"].to_numpy(float)
                        if np.any(~np.isfinite(actual)) or np.any(~np.isfinite(predicted)):
                            raise ValueError(f"Non-finite {channel} values in {path.name}")
                        if np.any(actual < 0.0):
                            raise ValueError(f"Negative observed activity in {path.name}")
                        validation_total = float(actual[split_code == 1].sum())
                        test_total = float(actual[split_code == 2].sum())
                        if validation_total <= 0.0 or test_total <= 0.0:
                            raise ValueError(
                                f"Degenerate held-out {channel} activity in {event}, fold {fold}: "
                                f"validation={validation_total}, test={test_total}"
                            )
                        train_actual = actual[split_code == 0]
                        test_actual = actual[split_code == 2]
                        test_predicted = predicted[split_code == 2]
                        train_rms = float(np.sqrt(np.mean(train_actual ** 2)))
                        denominator = max(train_rms, 1.0)
                        nrmse = float(np.sqrt(np.mean((test_predicted - test_actual) ** 2))) / denominator
                        nmae = float(np.mean(np.abs(test_predicted - test_actual))) / denominator
                        selector = (
                            audited.event.eq(event) & audited.fold.eq(fold)
                            & audited.model.eq(model) & audited.seed.eq(seed)
                            & audited.channel.eq(channel)
                        )
                        reported = float(audited.loc[selector, "nrmse"].iloc[0])
                        if not np.isclose(reported, nrmse, rtol=2e-7, atol=1e-10):
                            raise ValueError(
                                f"NRMSE mismatch in {path.name}/{channel}: "
                                f"run={reported:.12g}, recomputed={nrmse:.12g}"
                            )
                        audited.loc[selector, "nrmse"] = nrmse
                        audited.loc[selector, "nmae"] = nmae
                        audited.loc[selector, "train_activity_rms"] = train_rms
                        audited.loc[selector, "validation_activity_total"] = validation_total
                        audited.loc[selector, "test_activity_total"] = test_total

    checked = audited[["nrmse", "nmae", "validation_activity_total", "test_activity_total"]]
    if not np.isfinite(checked.to_numpy(float)).all():
        raise ValueError("Audited metrics contain non-finite values")
    if (audited[["validation_activity_total", "test_activity_total"]] <= 0).any().any():
        raise ValueError("At least one validation/test event-channel is degenerate")
    return audited, cache


def summaries(
    runs: pd.DataFrame, index: ExperimentIndex
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ["nrmse", "nmae", "wape", "mae"]

    # Equal folds before computing across-seed variability.
    event_channel_seed = runs.groupby(
        ["event", "channel", "model", "seed"], as_index=False
    )[metrics].mean()
    event_channel = event_channel_seed.groupby(
        ["event", "channel", "model"]
    )[metrics].agg(["mean", "std"]).reset_index()
    event_channel.columns = [
        "event", "channel", "model",
        *[f"{metric}_{stat}" for metric in metrics for stat in ("mean", "std")],
    ]
    event_channel["seed_count"] = len(index.seeds)

    # Equal channels within fold, then equal folds within event.
    event_seed = (
        runs.groupby(["event", "fold", "model", "seed"], as_index=False)[metrics].mean()
        .groupby(["event", "model", "seed"], as_index=False)[metrics].mean()
    )
    event_summary = event_seed.groupby(["event", "model"])[metrics].agg(
        ["mean", "std"]
    ).reset_index()
    event_summary.columns = [
        "event", "model",
        *[f"event_{metric}_{stat}" for metric in metrics for stat in ("mean", "std")],
    ]
    event_summary["seed_count"] = len(index.seeds)

    # Equal events per seed; reported SD is sample SD across seeds (pandas ddof=1).
    macro_seed = event_seed.groupby(["model", "seed"], as_index=False)[metrics].mean()
    macro = macro_seed.groupby("model")[metrics].agg(["mean", "std"]).reset_index()
    macro.columns = [
        "model", *[f"macro_{metric}_{stat}" for metric in metrics for stat in ("mean", "std")]
    ]
    macro["seed_count"] = len(index.seeds)
    macro = macro.sort_values("macro_nrmse_mean", kind="stable").reset_index(drop=True)

    # Runtime and residual are fit-level quantities, so remove duplicate channels.
    fit_keys = ["event", "fold", "model", "seed"]
    fits = runs[fit_keys + [
        "parameter_count", "runtime_seconds", "physics_residual_rms", "best_epoch"
    ]].drop_duplicates(fit_keys)
    for model, group in fits.groupby("model"):
        if group["parameter_count"].nunique() != 1:
            raise ValueError(f"Parameter count varies across {model} fits")
    resources = fits.groupby("model", as_index=False).agg(
        parameter_count=("parameter_count", "first"),
        fit_count=("runtime_seconds", "size"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
        runtime_seconds_std=("runtime_seconds", "std"),
        physics_residual_rms_mean=("physics_residual_rms", "mean"),
        physics_residual_rms_std=("physics_residual_rms", "std"),
        best_epoch_mean=("best_epoch", "mean"),
        best_epoch_std=("best_epoch", "std"),
    )
    return event_channel, event_summary, event_seed, macro_seed, macro, resources


def validation_selected_seeds(
    run_manifest: dict[str, object], index: ExperimentIndex
) -> tuple[dict[tuple[str, int, str], int], pd.DataFrame]:
    """Select one prespecified initialization using validation loss only.

    The selection is made independently for every event, rolling-origin fold,
    and model.  Test predictions and test errors are not consulted.  Applying
    the identical rule to every architecture permits an operational comparison
    of the estimators that would actually be deployed after validation.
    """
    candidates = pd.DataFrame(run_manifest.get("runs", []))
    required = {"event", "fold", "model", "seed", "validation_score"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Run manifest lacks validation-selection fields: {sorted(missing)}")
    candidates = candidates[list(required)].copy()
    candidates["fold"] = pd.to_numeric(candidates["fold"], errors="raise").astype(int)
    candidates["seed"] = pd.to_numeric(candidates["seed"], errors="raise").astype(int)
    candidates["validation_score"] = pd.to_numeric(
        candidates["validation_score"], errors="raise"
    )
    if not np.isfinite(candidates["validation_score"].to_numpy(float)).all():
        raise ValueError("Non-finite validation score in run manifest")
    candidates = candidates.sort_values(
        ["event", "fold", "model", "validation_score", "seed"], kind="stable"
    )
    selected = candidates.drop_duplicates(["event", "fold", "model"], keep="first")
    expected = {
        (event, fold, model)
        for event in index.events
        for fold in index.folds_by_event[event]
        for model in index.models
    }
    actual = set(
        map(tuple, selected[["event", "fold", "model"]].itertuples(index=False, name=None))
    )
    if actual != expected:
        raise ValueError("Validation-selected estimators do not cover the complete grid")
    selected = selected.rename(columns={"seed": "selected_seed"}).reset_index(drop=True)
    mapping = {
        (str(row.event), int(row.fold), str(row.model)): int(row.selected_seed)
        for row in selected.itertuples(index=False)
    }
    return mapping, selected


def validation_selected_summaries(
    runs: pd.DataFrame,
    selected_seeds: dict[tuple[str, int, str], int],
    index: ExperimentIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize held-out errors after validation-only initialization choice."""
    pieces: list[pd.DataFrame] = []
    for (event, fold, model), seed in selected_seeds.items():
        piece = runs[
            runs.event.eq(event) & runs.fold.eq(fold)
            & runs.model.eq(model) & runs.seed.eq(seed)
        ].copy()
        if len(piece) != len(CHANNELS):
            raise ValueError(f"Selected run is incomplete: {(event, fold, model, seed)}")
        pieces.append(piece)
    selected_runs = pd.concat(pieces, ignore_index=True)
    metrics = ["nrmse", "nmae", "wape", "mae"]
    event_channel = selected_runs.groupby(
        ["event", "channel", "model"], as_index=False
    )[metrics].mean()
    fold = selected_runs.groupby(
        ["event", "fold", "model"], as_index=False
    )[metrics].mean()
    event = fold.groupby(["event", "model"])[metrics].agg(["mean", "std"]).reset_index()
    event.columns = [
        "event", "model",
        *[f"event_{metric}_{stat}" for metric in metrics for stat in ("mean", "std")],
    ]
    event["fold_count"] = fold["fold"].nunique()
    macro = event.groupby("model", as_index=False).agg(
        macro_nrmse=("event_nrmse_mean", "mean"),
        macro_nmae=("event_nmae_mean", "mean"),
        macro_wape=("event_wape_mean", "mean"),
        macro_mae=("event_mae_mean", "mean"),
        event_nrmse_sd=("event_nrmse_mean", "std"),
    ).sort_values("macro_nrmse", kind="stable")
    macro["selection_candidates_per_fold"] = len(index.seeds)
    return selected_runs, event_channel, event, macro


def _series_nrmse(actual: np.ndarray, predicted: np.ndarray, denominator: float) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2))) / denominator


def _hierarchical_model_macro(
    model: str,
    series: dict[tuple[str, int, str, int, str], TestSeries],
    index: ExperimentIndex,
) -> float:
    event_values: list[float] = []
    for event in index.events:
        fold_values: list[float] = []
        for fold in index.folds_by_event[event]:
            seed_values: list[float] = []
            for seed in index.seeds:
                channel_values: list[float] = []
                for channel in CHANNELS:
                    item = series[(event, fold, model, seed, channel)]
                    channel_values.append(
                        _series_nrmse(item.actual, item.predicted, item.denominator)
                    )
                seed_values.append(float(np.mean(channel_values)))
            fold_values.append(float(np.mean(seed_values)))
        event_values.append(float(np.mean(fold_values)))
    return float(np.mean(event_values))


def _prepare_test_series(
    cache: dict[tuple[str, int, str, int], pd.DataFrame], index: ExperimentIndex
) -> dict[tuple[str, int, str, int, str], TestSeries]:
    """Materialize NumPy test arrays once so 5,000 bootstrap draws stay fast."""
    result: dict[tuple[str, int, str, int, str], TestSeries] = {}
    for event in index.events:
        for fold in index.folds_by_event[event]:
            for model in index.models:
                for seed in index.seeds:
                    frame = cache[(event, fold, model, seed)]
                    train = frame["split"].eq("train").to_numpy()
                    test = frame["split"].eq("test").to_numpy()
                    for channel in CHANNELS:
                        actual_full = frame[f"actual_{channel}"].to_numpy(float)
                        predicted_full = frame[f"predicted_{channel}"].to_numpy(float)
                        denominator = max(
                            float(np.sqrt(np.mean(actual_full[train] ** 2))), 1.0
                        )
                        result[(event, fold, model, seed, channel)] = TestSeries(
                            actual=actual_full[test],
                            predicted=predicted_full[test],
                            denominator=denominator,
                        )
    return result


def _circular_block_indices(length: int, rng: np.random.Generator) -> np.ndarray:
    block_length = max(1, int(math.ceil(math.sqrt(length))))
    block_count = int(math.ceil(length / block_length))
    starts = rng.integers(0, length, size=block_count)
    return np.concatenate(
        [(start + np.arange(block_length)) % length for start in starts]
    )[:length]


def paired_hierarchical_block_bootstrap(
    cache: dict[tuple[str, int, str, int], pd.DataFrame],
    index: ExperimentIndex,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    """Paired event/fold/seed bootstrap with blocks over held-out time.

    Events form the outer, equally weighted cluster. Within each sampled event,
    folds and seeds are resampled. The same circular-block indices are used for
    both compared models and channels, retaining all relevant pairing.
    """
    if replicates < 1:
        raise ValueError("--bootstrap-reps must be positive")
    if "TransPINN" not in index.models:
        raise ValueError("TransPINN is required for paired comparisons")
    series = _prepare_test_series(cache, index)
    trans_macro = _hierarchical_model_macro("TransPINN", series, index)
    baselines = [model for model in index.models if model != "TransPINN"]
    rows: list[dict[str, object]] = []

    for comparison_index, baseline in enumerate(baselines):
        baseline_macro = _hierarchical_model_macro(baseline, series, index)
        point_difference = baseline_macro - trans_macro
        event_channel_differences: list[float] = []
        event_differences: list[float] = []
        for event in index.events:
            channel_differences: list[float] = []
            for channel in CHANNELS:
                units: list[float] = []
                for fold in index.folds_by_event[event]:
                    for seed in index.seeds:
                        base = series[(event, fold, baseline, seed, channel)]
                        trans = series[(event, fold, "TransPINN", seed, channel)]
                        base_error = _series_nrmse(
                            base.actual, base.predicted, base.denominator
                        )
                        trans_error = _series_nrmse(
                            trans.actual, trans.predicted, trans.denominator
                        )
                        units.append(base_error - trans_error)
                channel_differences.append(float(np.mean(units)))
            event_channel_differences.extend(channel_differences)
            event_differences.append(float(np.mean(channel_differences)))

        rng = np.random.default_rng(random_seed + 1009 * comparison_index)
        draws = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            sampled_events = rng.choice(index.events, size=len(index.events), replace=True)
            sampled_event_values: list[float] = []
            for sampled_event in sampled_events:
                event = str(sampled_event)
                folds = index.folds_by_event[event]
                sampled_folds = rng.choice(folds, size=len(folds), replace=True)
                sampled_fold_values: list[float] = []
                for sampled_fold in sampled_folds:
                    fold = int(sampled_fold)
                    sampled_seeds = rng.choice(index.seeds, size=len(index.seeds), replace=True)
                    sampled_seed_values: list[float] = []
                    for sampled_seed in sampled_seeds:
                        seed = int(sampled_seed)
                        first = series[(event, fold, baseline, seed, CHANNELS[0])]
                        selected = _circular_block_indices(len(first.actual), rng)
                        channel_values: list[float] = []
                        for channel in CHANNELS:
                            base = series[(event, fold, baseline, seed, channel)]
                            trans = series[(event, fold, "TransPINN", seed, channel)]
                            if len(base.actual) != len(first.actual):
                                raise ValueError("Channels have different test-window lengths")
                            base_error = _series_nrmse(
                                base.actual[selected], base.predicted[selected], base.denominator,
                            )
                            trans_error = _series_nrmse(
                                trans.actual[selected], trans.predicted[selected], trans.denominator,
                            )
                            channel_values.append(base_error - trans_error)
                        sampled_seed_values.append(float(np.mean(channel_values)))
                    sampled_fold_values.append(float(np.mean(sampled_seed_values)))
                sampled_event_values.append(float(np.mean(sampled_fold_values)))
            draws[replicate] = float(np.mean(sampled_event_values))

        atomic = np.asarray(event_channel_differences)
        event_array = np.asarray(event_differences)
        rows.append({
            "comparison": f"TransPINN vs {baseline}",
            "baseline": baseline,
            "baseline_macro_nrmse": baseline_macro,
            "transpinn_macro_nrmse": trans_macro,
            "baseline_minus_transpinn_nrmse": point_difference,
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "bootstrap_probability_baseline_worse": float(np.mean(draws > 0.0)),
            "relative_reduction_percent": 100.0 * point_difference / max(baseline_macro, 1e-12),
            "event_channel_wins": int(np.sum(atomic > 0.0)),
            "event_channel_ties": int(np.sum(np.isclose(atomic, 0.0))),
            "event_channel_losses": int(np.sum(atomic < 0.0)),
            "event_wins": int(np.sum(event_array > 0.0)),
            "event_ties": int(np.sum(np.isclose(event_array, 0.0))),
            "event_losses": int(np.sum(event_array < 0.0)),
            "events": len(index.events),
            "seeds": len(index.seeds),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": random_seed + 1009 * comparison_index,
            "block_length_rule": "ceil(sqrt(n_test)); circular blocks",
        })
    return pd.DataFrame(rows)


def paired_validation_selected_bootstrap(
    cache: dict[tuple[str, int, str, int], pd.DataFrame],
    selected_seeds: dict[tuple[str, int, str], int],
    index: ExperimentIndex,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    """Paired block bootstrap for validation-selected deployed estimators."""
    if replicates < 1:
        raise ValueError("--bootstrap-reps must be positive")

    series: dict[tuple[str, int, str, str], TestSeries] = {}
    for event in index.events:
        for fold in index.folds_by_event[event]:
            for model in index.models:
                seed = selected_seeds[(event, fold, model)]
                frame = cache[(event, fold, model, seed)]
                train = frame["split"].eq("train").to_numpy()
                test = frame["split"].eq("test").to_numpy()
                for channel in CHANNELS:
                    actual_full = frame[f"actual_{channel}"].to_numpy(float)
                    predicted_full = frame[f"predicted_{channel}"].to_numpy(float)
                    denominator = max(
                        float(np.sqrt(np.mean(actual_full[train] ** 2))), 1.0
                    )
                    series[(event, fold, model, channel)] = TestSeries(
                        actual_full[test], predicted_full[test], denominator
                    )

    def selected_item(event: str, fold: int, model: str, channel: str) -> TestSeries:
        return series[(event, fold, model, channel)]

    def macro(model: str) -> float:
        events: list[float] = []
        for event in index.events:
            folds: list[float] = []
            for fold in index.folds_by_event[event]:
                channel_values: list[float] = []
                for channel in CHANNELS:
                    item = selected_item(event, fold, model, channel)
                    channel_values.append(
                        _series_nrmse(item.actual, item.predicted, item.denominator)
                    )
                folds.append(float(np.mean(channel_values)))
            events.append(float(np.mean(folds)))
        return float(np.mean(events))

    trans_macro = macro("TransPINN")
    rows: list[dict[str, object]] = []
    for comparison_index, baseline in enumerate(
        model for model in index.models if model != "TransPINN"
    ):
        baseline_macro = macro(baseline)
        point_difference = baseline_macro - trans_macro
        event_channel_differences: list[float] = []
        event_differences: list[float] = []
        for event in index.events:
            channel_differences: list[float] = []
            for channel in CHANNELS:
                fold_differences: list[float] = []
                for fold in index.folds_by_event[event]:
                    base = selected_item(event, fold, baseline, channel)
                    trans = selected_item(event, fold, "TransPINN", channel)
                    fold_differences.append(
                        _series_nrmse(base.actual, base.predicted, base.denominator)
                        - _series_nrmse(trans.actual, trans.predicted, trans.denominator)
                    )
                channel_differences.append(float(np.mean(fold_differences)))
            event_channel_differences.extend(channel_differences)
            event_differences.append(float(np.mean(channel_differences)))

        rng = np.random.default_rng(random_seed + 7919 + 1009 * comparison_index)
        draws = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            sampled_events = rng.choice(index.events, size=len(index.events), replace=True)
            sampled_event_values: list[float] = []
            for sampled_event in sampled_events:
                event = str(sampled_event)
                folds = index.folds_by_event[event]
                sampled_folds = rng.choice(folds, size=len(folds), replace=True)
                sampled_fold_values: list[float] = []
                for sampled_fold in sampled_folds:
                    fold = int(sampled_fold)
                    first = selected_item(event, fold, baseline, CHANNELS[0])
                    selected = _circular_block_indices(len(first.actual), rng)
                    channel_values: list[float] = []
                    for channel in CHANNELS:
                        base = selected_item(event, fold, baseline, channel)
                        trans = selected_item(event, fold, "TransPINN", channel)
                        channel_values.append(
                            _series_nrmse(
                                base.actual[selected], base.predicted[selected], base.denominator
                            )
                            - _series_nrmse(
                                trans.actual[selected], trans.predicted[selected], trans.denominator
                            )
                        )
                    sampled_fold_values.append(float(np.mean(channel_values)))
                sampled_event_values.append(float(np.mean(sampled_fold_values)))
            draws[replicate] = float(np.mean(sampled_event_values))

        atomic = np.asarray(event_channel_differences)
        event_array = np.asarray(event_differences)
        rows.append({
            "comparison": f"validation-selected TransPINN vs {baseline}",
            "baseline": baseline,
            "baseline_macro_nrmse": baseline_macro,
            "transpinn_macro_nrmse": trans_macro,
            "baseline_minus_transpinn_nrmse": point_difference,
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "bootstrap_probability_baseline_worse": float(np.mean(draws > 0.0)),
            "relative_reduction_percent": 100.0 * point_difference / max(baseline_macro, 1e-12),
            "event_channel_wins": int(np.sum(atomic > 0.0)),
            "event_channel_ties": int(np.sum(np.isclose(atomic, 0.0))),
            "event_channel_losses": int(np.sum(atomic < 0.0)),
            "event_wins": int(np.sum(event_array > 0.0)),
            "event_ties": int(np.sum(np.isclose(event_array, 0.0))),
            "event_losses": int(np.sum(event_array < 0.0)),
            "events": len(index.events),
            "selection_candidates_per_fold": len(index.seeds),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": random_seed + 7919 + 1009 * comparison_index,
            "block_length_rule": "ceil(sqrt(n_test)); circular blocks",
        })
    return pd.DataFrame(rows)


def dataset_statistics(
    manifest: dict[str, object],
    index: ExperimentIndex,
    cache: dict[tuple[str, int, str, int], pd.DataFrame],
) -> pd.DataFrame:
    metadata = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    reference_model = "TransPINN" if "TransPINN" in index.models else index.models[0]
    reference_seed = index.seeds[0]
    rows: list[dict[str, object]] = []
    for event in index.events:
        item = metadata[event]
        processed = pd.read_csv(PROCESSED / f"{event}.csv", parse_dates=["timestamp"])
        fold_records: list[dict[str, object]] = []
        for fold in index.folds_by_event[event]:
            frame = cache[(event, fold, reference_model, reference_seed)]
            split = frame["split"].astype(str)
            record: dict[str, object] = {
                "fold": fold,
                "train": int(split.eq("train").sum()),
                "validation": int(split.eq("validation").sum()),
                "test": int(split.eq("test").sum()),
                "prefix": len(frame),
            }
            for channel in CHANNELS:
                actual = frame[f"actual_{channel}"].to_numpy(float)
                record[f"validation_{channel}"] = float(actual[split.eq("validation")].sum())
                record[f"test_{channel}"] = float(actual[split.eq("test")].sum())
            fold_records.append(record)
        evaluated_bins = max(int(record["prefix"]) for record in fold_records)
        evaluated = processed.iloc[:evaluated_bins]
        rows.append({
            "event": event,
            "source": item["source"],
            "bin_width": item["bin_width"],
            "full_bins": int(item["bins"]),
            "evaluated_prefix_bins": evaluated_bins,
            "evaluated_start": str(evaluated["timestamp"].iloc[0]),
            "evaluated_end": str(evaluated["timestamp"].iloc[-1]),
            "full_rumor_total": float(item["rumor_total"]),
            "full_debunk_total": float(item["debunk_total"]),
            "evaluated_rumor_total": float(evaluated["rumor_activity"].sum()),
            "evaluated_debunk_total": float(evaluated["debunk_activity"].sum()),
            "folds": len(fold_records),
            "fold_boundaries_train_validation_test": " | ".join(
                f"F{r['fold']}:{r['train']}/{r['validation']}/{r['test']}" for r in fold_records
            ),
            "train_bins_min": min(int(r["train"]) for r in fold_records),
            "train_bins_max": max(int(r["train"]) for r in fold_records),
            "validation_bins_min": min(int(r["validation"]) for r in fold_records),
            "validation_bins_max": max(int(r["validation"]) for r in fold_records),
            "test_bins_min": min(int(r["test"]) for r in fold_records),
            "test_bins_max": max(int(r["test"]) for r in fold_records),
            "minimum_validation_rumor_total": min(float(r["validation_rumor"]) for r in fold_records),
            "minimum_validation_debunk_total": min(float(r["validation_debunk"]) for r in fold_records),
            "minimum_test_rumor_total": min(float(r["test_rumor"]) for r in fold_records),
            "minimum_test_debunk_total": min(float(r["test_debunk"]) for r in fold_records),
            "fold_rule": manifest.get("fold_rule", ""),
            "mapping": item["mapping"],
            "caveat": item.get("caveat", ""),
            "processed_sha256": item["sha256"],
            "source_sha256": item.get("source_sha256", ""),
        })
    statistics = pd.DataFrame(rows)
    held_out = [
        "minimum_validation_rumor_total", "minimum_validation_debunk_total",
        "minimum_test_rumor_total", "minimum_test_debunk_total",
    ]
    if (statistics[held_out] <= 0.0).any().any():
        raise ValueError("Dataset statistics reveal a degenerate validation/test channel")
    return statistics


def plot_macro_errors(macro: pd.DataFrame) -> list[Path]:
    order = macro.sort_values("macro_nrmse_mean", ascending=True).reset_index(drop=True)
    errors = order["macro_nrmse_std"].fillna(0.0).to_numpy(float)
    fig, axis = plt.subplots(figsize=(7.15, 3.45), constrained_layout=True)
    bars = axis.barh(
        order["model"], order["macro_nrmse_mean"], xerr=errors,
        color=[COLORS.get(model, "#888888") for model in order["model"]],
        alpha=0.9, capsize=3, height=0.66,
    )
    axis.set_xlabel("Macro test NRMSE (mean ± sample SD across seeds)")
    axis.set_ylabel("")
    axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    maximum = float((order["macro_nrmse_mean"] + errors).max())
    axis.set_xlim(0.0, max(0.05, maximum * 1.18))
    for bar, value, error in zip(bars, order["macro_nrmse_mean"], errors):
        axis.text(float(value) + float(error) + maximum * 0.025,
                  bar.get_y() + bar.get_height() / 2,
                  f"{float(value):.3f}", va="center", fontsize=8)
    outputs = [FIGURES / "baseline_rolling_nrmse.png", FIGURES / "baseline_rolling_nrmse.pdf"]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_representative_forecasts(
    cache: dict[tuple[str, int, str, int], pd.DataFrame], index: ExperimentIndex
) -> list[Path]:
    missing = set(FORECAST_MODELS) - set(index.models)
    if missing:
        raise ValueError(f"Forecast comparison models missing: {sorted(missing)}")
    fig, axes = plt.subplots(
        len(index.events), len(CHANNELS), figsize=(10.2, 10.1), constrained_layout=False
    )
    axes = np.atleast_2d(axes)
    for row, event in enumerate(index.events):
        folds = index.folds_by_event[event]
        fold = folds[len(folds) // 2]
        reference = cache[(event, fold, "TransPINN", index.seeds[0])]
        x = np.arange(len(reference))
        validation_start = int(reference["split"].eq("train").sum())
        test_start = validation_start + int(reference["split"].eq("validation").sum())
        for column, channel in enumerate(CHANNELS):
            axis = axes[row, column]
            axis.axvspan(validation_start - 0.5, test_start - 0.5,
                         color="#dce7f2", alpha=0.55, linewidth=0)
            axis.axvspan(test_start - 0.5, len(reference) - 0.5,
                         color="#f4dfdf", alpha=0.55, linewidth=0)
            actual = np.log1p(reference[f"actual_{channel}"].to_numpy(float))
            axis.step(x, actual, where="mid", color="#202020", linewidth=1.0,
                      alpha=0.82, label="Observed")
            for model in FORECAST_MODELS:
                values = np.stack([
                    np.log1p(np.clip(
                        cache[(event, fold, model, seed)][f"predicted_{channel}"].to_numpy(float),
                        0.0, None,
                    ))
                    for seed in index.seeds
                ])
                mean = values.mean(axis=0)
                axis.plot(x, mean, color=COLORS[model], linewidth=1.2, label=model)
                if model == "TransPINN" and len(index.seeds) > 1:
                    sd = values.std(axis=0, ddof=1)
                    axis.fill_between(x, np.maximum(mean - sd, 0.0), mean + sd,
                                      color=COLORS[model], alpha=0.12, linewidth=0)
            axis.axvline(validation_start - 0.5, color="#6384a4", linestyle="--", linewidth=0.8)
            axis.axvline(test_start - 0.5, color="#a45e5e", linestyle=":", linewidth=1.0)
            axis.set_title(
                f"{EVENT_LABELS.get(event, event)} — {channel} (fold {fold})", fontsize=9.5
            )
            axis.set_xlabel("Time-bin index")
            axis.set_ylabel(r"$\log(1+\mathrm{activity})$")
            axis.set_ylim(bottom=0.0)
            axis.grid(axis="y", color="#e5e5e5", linewidth=0.55)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.06, top=0.945,
                        hspace=0.62, wspace=0.17)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.993),
               ncol=5, frameon=False, fontsize=8)
    outputs = [
        FIGURES / "multievent_rolling_forecasts.png",
        FIGURES / "multievent_rolling_forecasts.pdf",
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_validation_selected_overview(
    event_summary: pd.DataFrame,
    macro: pd.DataFrame,
    bootstrap: pd.DataFrame,
    resources: pd.DataFrame,
    index: ExperimentIndex,
) -> list[Path]:
    """Event errors, paired effects, and compute for deployed estimators."""
    models = [model for model in PREFERRED_MODELS if model in index.models]
    categories = list(index.events) + ["macro"]
    fig = plt.figure(figsize=(10.4, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    bars_axis = fig.add_subplot(grid[0, :])
    forest_axis = fig.add_subplot(grid[1, 0])
    resource_axis = fig.add_subplot(grid[1, 1])

    x = np.arange(len(categories), dtype=float)
    width = 0.15
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * width
    for model, offset in zip(models, offsets):
        values = []
        for event in index.events:
            value = event_summary[
                event_summary.event.eq(event) & event_summary.model.eq(model)
            ]["event_nrmse_mean"]
            values.append(float(value.iloc[0]))
        values.append(float(macro.loc[macro.model.eq(model), "macro_nrmse"].iloc[0]))
        bars_axis.bar(
            x + offset, values, width=width, color=COLORS[model], alpha=0.9,
            label=model, edgecolor="white", linewidth=0.35,
        )
    bars_axis.set_xticks(x, [
        *[EVENT_LABELS.get(event, event) for event in index.events], "Macro"
    ])
    bars_axis.set_ylabel("Validation-selected test NRMSE")
    bars_axis.grid(axis="y", color="#dddddd", linewidth=0.65)
    bars_axis.set_axisbelow(True)
    bars_axis.legend(ncol=5, frameon=False, fontsize=8, loc="upper center")
    bars_axis.set_title("(a) Event-level and equal-event macro comparison", loc="left", fontsize=10)

    forest = bootstrap.sort_values(
        "baseline_minus_transpinn_nrmse", ascending=True, kind="stable"
    ).reset_index(drop=True)
    y = np.arange(len(forest))
    point = forest["baseline_minus_transpinn_nrmse"].to_numpy(float)
    lower = forest["ci95_low"].to_numpy(float)
    upper = forest["ci95_high"].to_numpy(float)
    forest_axis.errorbar(
        point, y, xerr=np.vstack([point - lower, upper - point]), fmt="o",
        color=COLORS["TransPINN"], ecolor="#555555", capsize=3, markersize=5,
    )
    forest_axis.axvline(0.0, color="#222222", linestyle="--", linewidth=0.9)
    forest_axis.set_yticks(y, forest["baseline"])
    forest_axis.set_xlabel("Baseline minus TransPINN NRMSE")
    forest_axis.grid(axis="x", color="#e1e1e1", linewidth=0.6)
    forest_axis.set_title("(b) Paired 95% block-bootstrap intervals", loc="left", fontsize=10)

    merged = macro.merge(resources, on="model", how="inner")
    merged["selection_cpu_seconds"] = (
        merged["runtime_seconds_mean"] * len(index.seeds)
    )
    for row in merged.itertuples(index=False):
        resource_axis.scatter(
            row.selection_cpu_seconds, row.macro_nrmse, s=48,
            color=COLORS.get(row.model, "#777777"), edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        resource_axis.annotate(
            row.model, (row.selection_cpu_seconds, row.macro_nrmse),
            xytext=(5, 4), textcoords="offset points", fontsize=7.5,
        )
    resource_axis.set_xlabel("CPU seconds per fold (three candidates)")
    resource_axis.set_ylabel("Macro test NRMSE")
    resource_axis.grid(color="#e1e1e1", linewidth=0.6)
    resource_axis.set_title("(c) Accuracy--compute trade-off", loc="left", fontsize=10)

    outputs = [
        FIGURES / "validation_selected_overview.png",
        FIGURES / "validation_selected_overview.pdf",
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_validation_selected_heatmap(
    event_channel: pd.DataFrame, index: ExperimentIndex
) -> list[Path]:
    models = [model for model in PREFERRED_MODELS if model in index.models]
    row_keys = [(event, channel) for event in index.events for channel in CHANNELS]
    matrix = np.empty((len(row_keys), len(models)), dtype=float)
    for row, (event, channel) in enumerate(row_keys):
        for column, model in enumerate(models):
            value = event_channel[
                event_channel.event.eq(event) & event_channel.channel.eq(channel)
                & event_channel.model.eq(model)
            ]["nrmse"]
            matrix[row, column] = float(value.iloc[0])

    fig, axis = plt.subplots(figsize=(7.35, 4.65), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(np.arange(len(models)), models, rotation=20, ha="right")
    axis.set_yticks(
        np.arange(len(row_keys)),
        [f"{EVENT_LABELS.get(event, event)} — {channel}" for event, channel in row_keys],
    )
    threshold = float((matrix.min() + matrix.max()) / 2.0)
    for row in range(matrix.shape[0]):
        best = int(np.argmin(matrix[row]))
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, f"{matrix[row, column]:.3f}",
                ha="center", va="center", fontsize=7.5,
                color="white" if matrix[row, column] > threshold else "#222222",
                fontweight="bold" if column == best else "normal",
            )
        axis.add_patch(plt.Rectangle(
            (best - 0.49, row - 0.49), 0.98, 0.98,
            fill=False, edgecolor="#16753a", linewidth=1.5,
        ))
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Test NRMSE (lower is better)")
    axis.set_title("Validation-selected event--channel comparison; green box marks row minimum")
    outputs = [
        FIGURES / "validation_selected_event_channel_heatmap.png",
        FIGURES / "validation_selected_event_channel_heatmap.pdf",
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_all_seed_error_distribution(
    runs: pd.DataFrame, index: ExperimentIndex
) -> list[Path]:
    models = [model for model in PREFERRED_MODELS if model in index.models]
    units = runs.groupby(
        ["event", "fold", "model", "seed"], as_index=False
    )["nrmse"].mean()
    data = [units.loc[units.model.eq(model), "nrmse"].to_numpy(float) for model in models]
    fig, axis = plt.subplots(figsize=(7.35, 3.75), constrained_layout=True)
    box = axis.boxplot(
        data, tick_labels=models, patch_artist=True, showfliers=False, widths=0.58,
        medianprops={"color": "#222222", "linewidth": 1.2},
    )
    for patch, model in zip(box["boxes"], models):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.58)
    rng = np.random.default_rng(20260825)
    for column, (model, values) in enumerate(zip(models, data), start=1):
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        axis.scatter(
            column + jitter, values, s=13, color=COLORS[model], alpha=0.68,
            edgecolor="white", linewidth=0.25,
        )
    axis.set_ylabel("Test NRMSE per event--fold--seed")
    axis.set_title("All prespecified initializations: error distribution before validation selection")
    axis.grid(axis="y", color="#e1e1e1", linewidth=0.6)
    axis.set_axisbelow(True)
    outputs = [
        FIGURES / "all_seed_error_distribution.png",
        FIGURES / "all_seed_error_distribution.pdf",
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_validation_selected_forecasts(
    cache: dict[tuple[str, int, str, int], pd.DataFrame],
    selected_seeds: dict[tuple[str, int, str], int],
    index: ExperimentIndex,
) -> list[Path]:
    models = [model for model in PREFERRED_MODELS if model in index.models]
    fig, axes = plt.subplots(
        len(index.events), len(CHANNELS), figsize=(10.2, 10.1), constrained_layout=False
    )
    axes = np.atleast_2d(axes)
    for row, event in enumerate(index.events):
        folds = index.folds_by_event[event]
        fold = folds[len(folds) // 2]
        trans_seed = selected_seeds[(event, fold, "TransPINN")]
        reference = cache[(event, fold, "TransPINN", trans_seed)]
        x = np.arange(len(reference))
        validation_start = int(reference["split"].eq("train").sum())
        test_start = validation_start + int(reference["split"].eq("validation").sum())
        for column, channel in enumerate(CHANNELS):
            axis = axes[row, column]
            axis.axvspan(validation_start - 0.5, test_start - 0.5,
                         color="#dce7f2", alpha=0.55, linewidth=0)
            axis.axvspan(test_start - 0.5, len(reference) - 0.5,
                         color="#f4dfdf", alpha=0.55, linewidth=0)
            actual = np.log1p(reference[f"actual_{channel}"].to_numpy(float))
            axis.step(x, actual, where="mid", color="#202020", linewidth=1.0,
                      alpha=0.82, label="Observed")
            for model in models:
                seed = selected_seeds[(event, fold, model)]
                values = np.log1p(np.clip(
                    cache[(event, fold, model, seed)][f"predicted_{channel}"].to_numpy(float),
                    0.0, None,
                ))
                axis.plot(x, values, color=COLORS[model], linewidth=1.15, label=model)
            axis.axvline(validation_start - 0.5, color="#6384a4", linestyle="--", linewidth=0.8)
            axis.axvline(test_start - 0.5, color="#a45e5e", linestyle=":", linewidth=1.0)
            axis.set_title(
                f"{EVENT_LABELS.get(event, event)} — {channel} (fold {fold})", fontsize=9.5
            )
            axis.set_xlabel("Time-bin index")
            axis.set_ylabel(r"$\log(1+\mathrm{activity})$")
            axis.set_ylim(bottom=0.0)
            axis.grid(axis="y", color="#e5e5e5", linewidth=0.55)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.06, top=0.945,
                        hspace=0.62, wspace=0.17)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.993),
               ncol=6, frameon=False, fontsize=7.5)
    outputs = [
        FIGURES / "validation_selected_forecasts.png",
        FIGURES / "validation_selected_forecasts.pdf",
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def write_analysis_manifest(
    run_manifest: dict[str, object],
    index: ExperimentIndex,
    bootstrap_reps: int,
    bootstrap_seed: int,
    validation_selection: pd.DataFrame,
    outputs: list[Path],
    audited_rows: int,
) -> Path:
    path = TABLES / "analysis_manifest.json"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": "experiments/analyze_multievent_results.py",
        "script_sha256": {
            "prepare_multievent_data.py": _sha256(ROOT / "experiments" / "prepare_multievent_data.py"),
            "run_multievent_baselines.py": _sha256(ROOT / "experiments" / "run_multievent_baselines.py"),
            "analyze_multievent_results.py": _sha256(ROOT / "experiments" / "analyze_multievent_results.py"),
        },
        "processed_metadata_sha256": _sha256(PROCESSED / "metadata.json"),
        "requirements_sha256": _sha256(ROOT / "experiments" / "requirements.txt"),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "torch": _package_version("torch"),
        },
        "source_run_table": str(RUNS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_run_table_sha256": _sha256(RUNS_PATH),
        "source_run_manifest": str(RUN_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_run_manifest_sha256": _sha256(RUN_MANIFEST_PATH),
        "prediction_pattern": "output/predictions/{event}_fold{fold}_{model_slug}_seed{seed}.csv",
        "legacy_nonfold_predictions_used": False,
        "events": list(index.events),
        "models": list(index.models),
        "seeds": list(index.seeds),
        "folds_by_event": {event: list(folds) for event, folds in index.folds_by_event.items()},
        "audited_event_fold_model_seed_channel_rows": audited_rows,
        "aggregation": (
            "equal channels within fold; equal folds within event; equal events for each seed; "
            "mean and sample SD (ddof=1) across seeds"
        ),
        "validation_selected_deployment": {
            "rule": (
                "for every event-fold-model, choose the smallest-validation-loss member "
                "of the same three prespecified seeds; break exact ties by smaller seed; "
                "never consult test predictions or errors"
            ),
            "aggregation": "equal channels within fold; equal folds within event; equal events",
            "candidate_count_per_event_fold_model": len(index.seeds),
            "selected_initializations": validation_selection.to_dict(orient="records"),
        },
        "normalization": "test RMSE or MAE divided by max(training-channel RMS, 1.0)",
        "bootstrap": {
            "replicates": bootstrap_reps,
            "base_seed": bootstrap_seed,
            "outer_unit": "event (equal-event macro)",
            "inner_resampling": "folds and seeds with replacement",
            "time_resampling": "paired circular blocks of ceil(sqrt(n_test))",
        },
        "checks": {
            "complete_cartesian_grid": True,
            "manifest_matches_grid": True,
            "fold_qualified_predictions_only": True,
            "identical_actuals_and_splits_across_models_and_seeds": True,
            "positive_validation_and_test_activity_in_both_channels": True,
            "reported_nrmse_matches_prediction_recomputation": True,
        },
        "training_fold_rule": run_manifest.get("fold_rule", ""),
        "outputs": [str(item.relative_to(ROOT)).replace("\\", "/") for item in outputs],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse completed rolling-origin baseline experiments without retraining."
    )
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs, run_manifest, index = load_inputs()
    audited, cache = validate_predictions_and_metrics(runs, index)
    event_channel, event_summary, event_seed, macro_seed, macro, resources = summaries(
        audited, index
    )
    selected_seed_map, selection_table = validation_selected_seeds(run_manifest, index)
    selected_runs, selected_event_channel, selected_event, selected_macro = (
        validation_selected_summaries(audited, selected_seed_map, index)
    )
    datasets = dataset_statistics(run_manifest, index, cache)
    bootstrap = paired_hierarchical_block_bootstrap(
        cache, index, args.bootstrap_reps, args.bootstrap_seed
    )
    selected_bootstrap = paired_validation_selected_bootstrap(
        cache, selected_seed_map, index, args.bootstrap_reps, args.bootstrap_seed
    )

    table_outputs = [
        TABLES / "baseline_runs_verified.csv",
        TABLES / "dataset_statistics.csv",
        TABLES / "baseline_event_channel_summary.csv",
        TABLES / "baseline_event_summary.csv",
        TABLES / "baseline_event_seed_summary.csv",
        TABLES / "baseline_macro_seed_summary.csv",
        TABLES / "baseline_summary.csv",
        TABLES / "model_resources.csv",
        TABLES / "baseline_paired_block_bootstrap.csv",
        TABLES / "validation_selected_initializations.csv",
        TABLES / "validation_selected_runs.csv",
        TABLES / "validation_selected_event_channel_summary.csv",
        TABLES / "validation_selected_event_summary.csv",
        TABLES / "validation_selected_summary.csv",
        TABLES / "validation_selected_paired_block_bootstrap.csv",
    ]
    frames = [
        audited, datasets, event_channel, event_summary, event_seed,
        macro_seed, macro, resources, bootstrap, selection_table, selected_runs,
        selected_event_channel, selected_event, selected_macro, selected_bootstrap,
    ]
    for frame, path in zip(frames, table_outputs):
        frame.to_csv(path, index=False)

    figure_outputs = (
        plot_macro_errors(macro)
        + plot_representative_forecasts(cache, index)
        + plot_validation_selected_overview(
            selected_event, selected_macro, selected_bootstrap, resources, index
        )
        + plot_validation_selected_heatmap(selected_event_channel, index)
        + plot_all_seed_error_distribution(audited, index)
        + plot_validation_selected_forecasts(cache, selected_seed_map, index)
    )
    analysis_manifest_path = TABLES / "analysis_manifest.json"
    write_analysis_manifest(
        run_manifest, index, args.bootstrap_reps, args.bootstrap_seed,
        selection_table, table_outputs + figure_outputs + [analysis_manifest_path],
        len(audited),
    )

    print("Detected rolling-origin design:")
    print(json.dumps({
        "events": index.events,
        "models": index.models,
        "seeds": index.seeds,
        "folds_by_event": index.folds_by_event,
    }, indent=2))
    print("\nMacro test errors (equal event/channel/fold weighting):")
    print(macro.to_string(index=False))
    print("\nPaired hierarchical moving-block bootstrap:")
    print(bootstrap.to_string(index=False))
    print("\nValidation-selected deployed estimators (test errors):")
    print(selected_macro.to_string(index=False))
    print("\nValidation-selected paired hierarchical block bootstrap:")
    print(selected_bootstrap.to_string(index=False))
    print(f"\nAnalysis manifest: {analysis_manifest_path}")


if __name__ == "__main__":
    main()
