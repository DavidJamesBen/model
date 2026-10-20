"""Fixed-delay comparison of PINN and TransPINN.

Each run fixes the normalized DDE delay before optimisation.  Thus the two
architectures are compared at exactly the same delay, rolling-origin split,
random seed, data, physics residual, and optimisation budget.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from run_multievent_baselines import (
    ACTIVE_ACTIVITY_FRACTION,
    Config,
    EVENTS,
    load_event_folds,
    train_one,
)


MODELS = ("PINN", "TransPINN")
# The learned-delay baseline estimates cluster around 0.07.  The central
# values resolve that empirically relevant region; the two endpoints are
# stress tests within the prespecified admissible interval.
DEFAULT_TAUS = (0.02, 0.05, 0.07, 0.10, 0.15)


def tau_label(tau: float) -> str:
    return f"{tau:.3f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PINN and TransPINN at prespecified normalized delays."
    )
    parser.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    parser.add_argument("--events", nargs="+", choices=EVENTS, default=list(EVENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--output-root", type=Path, default=Path("output") / "tau_sweep")
    parser.add_argument("--smoke", action="store_true",
                        help="Use a tiny training budget to check the pipeline only.")
    args = parser.parse_args()

    taus = tuple(sorted(set(args.taus)))
    if not taus:
        raise ValueError("at least one tau value is required")
    if any(not 0.01 <= tau <= 0.22 for tau in taus):
        raise ValueError("all normalized tau values must lie in [0.01, 0.22]")

    config = Config(epochs=args.epochs)
    if args.smoke:
        config = Config(epochs=min(args.epochs, 20), min_epochs=20, patience_checks=2,
                        collocation_points=24)

    output_root = args.output_root
    table_dir = output_root / "tables"
    prediction_dir = output_root / "predictions"
    table_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    device = torch.device("cpu")
    folds_by_event = {event: load_event_folds(event) for event in args.events}
    total = sum(len(folds_by_event[event]) for event in args.events) * len(MODELS) \
        * len(args.seeds) * len(taus)
    all_rows: list[dict[str, object]] = []
    manifest_runs: list[dict[str, object]] = []
    run_index = 0
    for tau in taus:
        for event in args.events:
            for fold, frame, split, _, _ in folds_by_event[event]:
                for model_name in MODELS:
                    for seed in args.seeds:
                        run_index += 1
                        print(
                            f"[{run_index:03d}/{total:03d}] tau={tau:.3f} | {event} | "
                            f"fold={fold} | {model_name} | seed={seed}",
                            flush=True,
                        )
                        rows, prediction, run_info = train_one(
                            event, fold, frame, split, model_name, seed, config, device,
                            fixed_tau=tau,
                        )
                        for row in rows:
                            row["fixed_tau_normalized"] = tau
                        run_info["fixed_tau_normalized"] = tau
                        all_rows.extend(rows)
                        manifest_runs.append(run_info)
                        prediction.to_csv(
                            prediction_dir / (
                                f"tau_{tau_label(tau)}_{event}_fold{fold}_"
                                f"{model_name.lower()}_seed{seed}.csv"
                            ),
                            index=False,
                        )
                        pd.DataFrame(all_rows).to_csv(
                            table_dir / "tau_sweep_runs.csv", index=False
                        )

    manifest = {
        "purpose": "Fixed-delay PINN versus TransPINN sensitivity experiment",
        "config": asdict(config),
        "models": list(MODELS),
        "normalized_tau_values": list(taus),
        "events": args.events,
        "seeds": args.seeds,
        "active_activity_fraction_per_channel": ACTIVE_ACTIVITY_FRACTION,
        "protocol": (
            "For every tau, event, expanding-window fold, and seed, PINN and "
            "TransPINN use the identical data, split, DDE, history, observation "
            "heads, collocation grid, optimizer, and epoch budget. Tau is fixed "
            "rather than selected from test data."
        ),
        "runs": manifest_runs,
    }
    (table_dir / "tau_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved {len(all_rows)} channel-level rows under {output_root}")


if __name__ == "__main__":
    main()
