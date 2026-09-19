"""Summarise the fixed-delay PINN versus TransPINN experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"PINN": "#6c757d", "TransPINN": "#c92828"}
EVENT_LABELS = {"harvey": "Harvey", "csdc": "CSDC-Rumor", "ottawa": "Ottawa", "sydney": "Sydney"}


def seed_macro(runs: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight channels, folds, and events before averaging seeds."""
    return (
        runs.groupby(["fixed_tau_normalized", "model", "seed", "event", "fold"], as_index=False)["nrmse"]
        .mean()
        .groupby(["fixed_tau_normalized", "model", "seed", "event"], as_index=False)["nrmse"]
        .mean()
        .groupby(["fixed_tau_normalized", "model", "seed"], as_index=False)["nrmse"]
        .mean()
    )


def make_summary(runs: pd.DataFrame) -> pd.DataFrame:
    macro = seed_macro(runs)
    summary = (
        macro.groupby(["fixed_tau_normalized", "model"])["nrmse"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "macro_nrmse_mean", "std": "macro_nrmse_sd", "count": "seed_count"})
    )
    return summary.sort_values(["fixed_tau_normalized", "model"]).reset_index(drop=True)


def plot_macro(summary: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
    left, right = axes
    for model in ("PINN", "TransPINN"):
        sample = summary[summary["model"] == model]
        left.errorbar(sample["fixed_tau_normalized"], sample["macro_nrmse_mean"],
                      yerr=sample["macro_nrmse_sd"].fillna(0.0), marker="o", markersize=4,
                      capsize=3, linewidth=1.6, color=COLORS[model], label=model)
    left.set_xlabel("Fixed normalized delay $\\tau$")
    left.set_ylabel("Macro test NRMSE (mean $\\pm$ SD across seeds)")
    left.grid(alpha=0.25)
    left.legend(frameon=False)

    pivot = summary.pivot(index="fixed_tau_normalized", columns="model", values="macro_nrmse_mean")
    delta = pivot["PINN"] - pivot["TransPINN"]
    right.axhline(0.0, color="#333333", linewidth=0.8)
    right.plot(delta.index, delta.to_numpy(), marker="o", markersize=4,
               linewidth=1.6, color="#3465a4")
    right.fill_between(delta.index, 0.0, delta.to_numpy(), where=delta.to_numpy() >= 0.0,
                       color="#7ac77a", alpha=0.35, label="TransPINN lower")
    right.fill_between(delta.index, 0.0, delta.to_numpy(), where=delta.to_numpy() < 0.0,
                       color="#e88989", alpha=0.35, label="PINN lower")
    right.set_xlabel("Fixed normalized delay $\\tau$")
    right.set_ylabel("PINN $-$ TransPINN NRMSE")
    right.grid(alpha=0.25)
    right.legend(frameon=False, fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"pinn_vs_transpinn_tau_sweep.{extension}", dpi=300)
    plt.close(fig)


def plot_event_grid(runs: pd.DataFrame, figure_dir: Path) -> None:
    event_seed = (
        runs.groupby(["fixed_tau_normalized", "event", "model", "seed", "fold"], as_index=False)["nrmse"]
        .mean()
        .groupby(["fixed_tau_normalized", "event", "model", "seed"], as_index=False)["nrmse"]
        .mean()
        .groupby(["fixed_tau_normalized", "event", "model"], as_index=False)["nrmse"]
        .agg(mean="mean", sd="std")
    )
    events = list(dict.fromkeys(runs["event"]))
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), constrained_layout=True)
    for axis, event in zip(axes.flat, events):
        for model in ("PINN", "TransPINN"):
            sample = event_seed[(event_seed["event"] == event) & (event_seed["model"] == model)]
            axis.errorbar(sample["fixed_tau_normalized"], sample["mean"],
                          yerr=sample["sd"].fillna(0.0), marker="o", markersize=3.5,
                          capsize=2.5, linewidth=1.3, color=COLORS[model], label=model)
        axis.set_title(EVENT_LABELS.get(event, event.title()))
        axis.set_xlabel("Fixed normalized delay $\\tau$")
        axis.set_ylabel("Test NRMSE")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(events):]:
        axis.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"pinn_vs_transpinn_tau_by_event.{extension}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("output") / "tau_sweep")
    args = parser.parse_args()
    table_dir = args.input_root / "tables"
    figure_dir = args.input_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(table_dir / "tau_sweep_runs.csv")
    required = {"PINN", "TransPINN"}
    if set(runs["model"]) != required:
        raise ValueError(f"expected exactly {required}, found {set(runs['model'])}")
    if runs.empty:
        raise ValueError("tau sweep table is empty")
    summary = make_summary(runs)
    summary.to_csv(table_dir / "tau_sweep_macro_summary.csv", index=False)
    plot_macro(summary, figure_dir)
    plot_event_grid(runs, figure_dir)
    (table_dir / "tau_sweep_analysis_manifest.json").write_text(
        json.dumps({
            "metric": "test NRMSE",
            "aggregation": "equal weighting of channels, folds, and events; seed SD shown",
            "comparison": "positive PINN-minus-TransPINN values favour TransPINN",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nFigures written to {figure_dir}")


if __name__ == "__main__":
    main()
