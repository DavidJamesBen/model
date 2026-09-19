"""Evaluate identical seed-aggregation rules for every neural baseline.

This diagnostic never changes a fitted model.  It averages the three
prespecified seed predictions within each event/fold/model and recomputes the
test NRMSE from the archived prediction files.  The rule is deliberately
applied to every model, so the comparison does not grant TransPINN a special
ensemble advantage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"
PREDICTION_DIR = ROOT / "output" / "predictions"
MODELS = ("PINN", "LSTM-PINN", "TimePINN", "NoAttn-PINN", "TransPINN")
EVENTS = ("harvey", "csdc", "ottawa", "sydney")
CHANNELS = ("rumor", "debunk")


def prediction_path(event: str, fold: int, model: str, seed: int) -> Path:
    slug = model.lower().replace("-", "_")
    return PREDICTION_DIR / f"{event}_fold{fold}_{slug}_seed{seed}.csv"


def main() -> None:
    manifest = json.loads(
        (TABLE_DIR / "baseline_run_manifest.json").read_text(encoding="utf-8")
    )
    seeds = tuple(int(seed) for seed in manifest["seeds"])
    validation_scores = {
        (str(item["event"]), int(item["fold"]), str(item["model"]), int(item["seed"])):
        float(item["validation_score"])
        for item in manifest["runs"]
    }
    rows: list[dict[str, object]] = []
    for event in EVENTS:
        for fold in (1, 2, 3):
            for model in MODELS:
                frames = [
                    pd.read_csv(prediction_path(event, fold, model, seed))
                    for seed in seeds
                ]
                scores = np.array(
                    [validation_scores[(event, fold, model, seed)] for seed in seeds]
                )
                inverse_weights = 1.0 / np.maximum(scores, 1.0e-12)
                inverse_weights /= inverse_weights.sum()
                best_index = int(np.argmin(scores))
                for channel in CHANNELS:
                    actual = frames[0][f"actual_{channel}"].to_numpy(float)
                    split = frames[0]["split"].astype(str).to_numpy()
                    predicted = np.stack(
                        [frame[f"predicted_{channel}"].to_numpy(float) for frame in frames]
                    )
                    train = split == "train"
                    test = split == "test"
                    denominator = max(
                        float(np.sqrt(np.mean(actual[train] ** 2))), 1.0
                    )
                    candidates = {
                        "uniform_seed_mean": predicted.mean(axis=0),
                        "inverse_validation_weighted": np.average(
                            predicted, axis=0, weights=inverse_weights
                        ),
                        "validation_best_seed": predicted[best_index],
                    }
                    for strategy, estimate in candidates.items():
                        nrmse = float(
                            np.sqrt(np.mean((estimate[test] - actual[test]) ** 2))
                            / denominator
                        )
                        rows.append(
                            {
                                "strategy": strategy,
                                "event": event,
                                "fold": fold,
                                "model": model,
                                "channel": channel,
                                "nrmse": nrmse,
                            }
                        )

    detailed = pd.DataFrame(rows)
    detailed.to_csv(TABLE_DIR / "seed_ensemble_detailed.csv", index=False)
    summary = (
        detailed.groupby(["strategy", "event", "model"], as_index=False)["nrmse"]
        .mean()
        .groupby(["strategy", "model"], as_index=False)["nrmse"]
        .mean()
        .sort_values(["strategy", "nrmse"], kind="stable")
    )
    summary.to_csv(TABLE_DIR / "seed_ensemble_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
