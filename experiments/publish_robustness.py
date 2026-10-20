"""Publish complete, audited robustness results without selecting favorable cells.

Inputs are the per-group CSV files written by analyze_robustness.py. A requested
stage must be complete before any of its figures or tables are generated. This
script never trains models, changes an analysis, or supplies placeholder data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/robustness/publication"
MODELS = ("TransPINN", "Matched-PINN", "NoAttn-PINN", "Data-Transformer", "Data-MLP")
SYNTHETIC_MODELS = (*MODELS, "DDE-LS")
PRIMARY = ("Data-Transformer", "Data-MLP")
SCENARIOS = ("consistent", "decay_mismatch")
SCENARIO_LABELS = {"consistent": "Model-consistent data", "decay_mismatch": "Media-decay mismatch"}
EVALUATIONS = ("all_seeds", "validation_selected")
EVALUATION_LABELS = {"all_seeds": "All runs", "validation_selected": "Validation-selected runs"}
COLORS = {"TransPINN": "#245a81", "Matched-PINN": "#d18134", "NoAttn-PINN": "#8a5791",
          "Data-Transformer": "#3d8772", "Data-MLP": "#b64b4e", "DDE-LS": "#404040"}


def long_path(path):
    """Support Windows extended paths without changing artifact file names."""
    if os.name != "nt":
        return Path(path)
    absolute = os.path.abspath(os.fspath(path))
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def configure_fonts():
    found = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.serif": ["Times New Roman"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold", "mathtext.bfit": "Times New Roman:italic:bold",
        "mathtext.sf": "Times New Roman", "mathtext.tt": "Times New Roman",
        "mathtext.cal": "Times New Roman:italic", "mathtext.fallback": None,
        "axes.spines.top": False, "axes.spines.right": False})
    return found


def load_summary(path, models, expected_fits):
    frame = pd.read_csv(path)
    required = {"model", "nrmse_mean", "nrmse_std", "selected_nrmse", "fit_count"}
    if not required <= set(frame):
        raise ValueError(f"Required summary fields absent: {path}")
    if frame.model.duplicated().any() or set(frame.model) != set(models):
        raise ValueError(f"Incomplete or duplicated model grid: {path}")
    if not np.isfinite(frame[["nrmse_mean", "nrmse_std", "selected_nrmse"]]).all().all():
        raise ValueError(f"Nonfinite result: {path}")
    if (frame.fit_count != expected_fits).any():
        raise ValueError(f"Expected {expected_fits} fits/model: {path}")
    if (frame[["nrmse_mean", "nrmse_std", "selected_nrmse"]] < 0).any().any():
        raise ValueError(f"Negative error or deviation: {path}")
    return frame.set_index("model").loc[list(models)]


def number(value, digits=4):
    """Compact LaTeX number, retaining tiny numerical-solver errors."""
    value = float(value)
    if value == 0:
        return "0"
    if 1e-3 <= abs(value) < 1e3:
        return f"{value:.{digits}f}"
    exponent = math.floor(math.log10(abs(value)))
    return rf"{value / 10 ** exponent:.2f}\times10^{{{exponent}}}"


def mean_sd(mean, sd):
    if 0 < abs(mean) < 1e-3:
        exponent = math.floor(math.log10(abs(mean)))
        return rf"({mean / 10 ** exponent:.2f}\pm{sd / 10 ** exponent:.2f})\times10^{{{exponent}}}"
    return rf"{number(mean)}\pm{number(sd)}"


def save_figure(fig, destination, basename):
    fig.savefig(destination / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(destination / f"{basename}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def table_wrapper(caption, label, columns, body, note):
    body = body.rstrip()
    if body.endswith(r"\hline"):
        body = body[:-len(r"\hline")].rstrip()
    body = body.replace(r"\hline", r"\midrule")
    return ("% Generated from complete audited results by experiments/publish_robustness.py\n"
        "\\begin{table*}[t]\n\\centering\n"
        rf"\caption{{{caption}}}" + "\n" + rf"\label{{{label}}}" + "\n"
        "\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.08}\n"
        rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{columns}@{{}}}}" + "\n\\toprule\n" + body +
        "\n\\bottomrule\n\\end{tabular*}\n\\par\\smallskip\n"
        rf"\begin{{minipage}}{{\textwidth}}\footnotesize {note}\end{{minipage}}" +
        "\n\\end{table*}\n")


def publish_stage1(source, destination):
    paths = [source / f"stage1_b{budget}" / "summary.csv" for budget in (260, 1040)]
    tables = {budget: load_summary(path, MODELS, 36) for budget, path in zip((260, 1040), paths)}
    lines = [r"& \multicolumn{3}{c}{260 updates} & \multicolumn{3}{c}{1040 updates} \\",
             r"Model & Mean $\pm$ SD & Selected & Time (s) & Mean $\pm$ SD & Selected & Time (s) \\", r"\hline"]
    rows = []
    for model in MODELS:
        cells = []
        for budget in (260, 1040):
            row = tables[budget].loc[model]
            cells += [f"${mean_sd(row.nrmse_mean, row.nrmse_std)}$", f"${number(row.selected_nrmse)}$",
                      f"{row.runtime_seconds_mean:.2f}"]
        line = model + " & " + " & ".join(cells) + r" \\"
        lines.append(line)
        rows.append(line)
    caption = "Metric-aligned forecasting under two prespecified optimization budgets."
    note = ("Macro test NRMSE is averaged equally over channels, folds and events. Mean and sample SD span three seeds. "
        "Selected uses validation-based seed selection per event/fold. Each model has 18,619 parameters.")
    timing_note = (" Time averages optimization and validation cost per fit on an NVIDIA GeForce RTX 5070, "
        "using GPU batches of three. It measures amortized cost, not isolated-run latency.")
    (destination / "stage1_table.tex").write_text(table_wrapper(caption, "tab:aligned_budgets", "lcccccc", "\n".join(lines), note + timing_note), encoding="utf-8")
    (destination / "stage1_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), layout="constrained", sharey=True)
    x = np.arange(len(MODELS))
    for ax, metric, title in zip(axes, ("nrmse_mean", "selected_nrmse"), ("All runs", "Validation-selected runs")):
        for offset, budget, color in ((-.17, 260, "#67879e"), (.17, 1040, "#c58757")):
            frame = tables[budget]
            error = frame.nrmse_std.to_numpy() if metric == "nrmse_mean" else None
            ax.bar(x + offset, frame[metric], width=.32, color=color, yerr=error,
                   error_kw={"elinewidth": .75, "capsize": 2}, label=f"{budget} updates")
        ax.set_xticks(x, MODELS, rotation=27, ha="right")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test NRMSE")
    axes[1].legend(frameon=False)
    save_figure(fig, destination, "stage1_budgets")
    (destination / "stage1_budgets_caption.txt").write_text(caption + " " + note + " Error bars show sample SD, not confidence intervals.\n", encoding="utf-8")
    return paths


def publish_stage2(source, destination):
    matrices, summaries, paths = {}, {}, []
    retentions, noises = (100, 50, 25), (0, 5, 10)
    rows = []
    for retention in retentions:
        for noise in noises:
            path = source / f"stage2_r{retention}_n{noise}" / "paired_intervals.csv"
            summary_path = path.with_name("summary.csv")
            summaries[(retention, noise)] = load_summary(summary_path, MODELS, 120)
            paths += [path, summary_path]
            frame = pd.read_csv(path)
            for evaluation in EVALUATIONS:
                for baseline in PRIMARY:
                    found = frame[(frame.evaluation == evaluation) & (frame.baseline == baseline)]
                    if len(found) != 1:
                        raise ValueError(f"Missing/duplicate paired comparison: {path}/{evaluation}/{baseline}")
                    row = found.iloc[0]
                    fields = ["baseline_minus_transpinn_nrmse", "simultaneous_low", "simultaneous_high"]
                    if int(row.family_size) != 36 or not np.isfinite(row[fields].to_numpy(dtype=float)).all():
                        raise ValueError(f"Incomplete familywise interval: {path}")
                    matrices[(evaluation, baseline, retention, noise)] = row
                    rows.append({"retention": retention, "noise": noise, **row.to_dict()})
    combined = pd.DataFrame(rows)
    combined.to_csv(destination / "stage2_display_values.csv", index=False)
    metric_rows = ["Retained / Noise & " + " & ".join(MODELS) + r" \\", r"\hline"]
    for retention in retentions:
        for noise in noises:
            cells = []
            for model in MODELS:
                row = summaries[(retention, noise)].loc[model]
                cells.append(rf"${mean_sd(row.nrmse_mean, row.nrmse_std)}\;[{number(row.selected_nrmse)}]$")
            metric_rows.append(rf"{retention}\% / {noise}\% & " + " & ".join(cells) + r" \\")
    metric_note = ("Each entry gives macro test NRMSE as mean and sample SD across ten paired replicates, "
        "followed by the validation-selected result in brackets. A replicate combines initialization, observation-mask "
        "and noise seeds. Selection chooses the complete replicate per event/fold. Channels, folds and events receive "
        "equal weight. Each replicate uses its retained clean training observations for normalization. "
        "All nine conditions use 1040 updates.")
    metrics_tex = table_wrapper(
        "Forecasting accuracy for all models across the complete perturbation grid.",
        "tab:robustness_metrics", "lccccc", "\n".join(metric_rows), metric_note)
    # The five long model headings and nine mean--SD cells need a slightly
    # tighter setting to avoid an overfull box in the two-column manuscript.
    metrics_tex = metrics_tex.replace("\\footnotesize\n\\setlength{\\tabcolsep}{4pt}",
                                      "\\scriptsize\n\\setlength{\\tabcolsep}{2pt}", 1)
    (destination / "stage2_metrics_table.tex").write_text(metrics_tex, encoding="utf-8")
    maximum = max(float(combined.baseline_minus_transpinn_nrmse.abs().max()), 1e-12)
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), layout="constrained")
    for r, evaluation in enumerate(EVALUATIONS):
        for c, baseline in enumerate(PRIMARY):
            ax = axes[r, c]
            grid = np.array([[matrices[(evaluation, baseline, retention, noise)].baseline_minus_transpinn_nrmse
                              for noise in noises] for retention in retentions])
            plotted = ax.imshow(grid, cmap="RdBu", norm=norm, aspect="auto")
            for i, retention in enumerate(retentions):
                for j, noise in enumerate(noises):
                    row = matrices[(evaluation, baseline, retention, noise)]
                    significant = row.simultaneous_low > 0 or row.simultaneous_high < 0
                    text = f"{grid[i, j]:+.3g}" + (" *" if significant else "")
                    ax.text(j, i, text, ha="center", va="center", fontsize=9,
                            color="white" if abs(grid[i, j]) / maximum > .65 else "#111111")
            ax.set_xticks(range(3), noises)
            ax.set_yticks(range(3), retentions)
            ax.set_xlabel("Training noise (%)")
            if c == 0:
                ax.set_ylabel("Observations retained (%)")
            ax.set_title(f"{EVALUATION_LABELS[evaluation]}\n{baseline}")
            ax.tick_params(length=0)
    colorbar = fig.colorbar(plotted, ax=axes, shrink=.9, pad=.025)
    colorbar.set_label("Baseline NRMSE − TransPINN NRMSE")
    save_figure(fig, destination, "stage2_full_grid")
    caption = ("Complete observation-retention and training-noise grid. Positive differences favor TransPINN. "
        "Negative differences favor the named data-only baseline. Every cell uses ten paired initialization and "
        "observation-perturbation seeds. Asterisks mark simultaneous intervals excluding zero in either direction. "
        "Bonferroni adjustment covers 36 comparisons: nine cells, two data-only baselines, and both evaluation rules. "
        "Normalization uses the retained clean training observations separately for each replicate. "
        "Validation and test labels are unperturbed.")
    (destination / "stage2_full_grid_caption.txt").write_text(caption + "\n", encoding="utf-8")
    header = [r"& \multicolumn{2}{c}{All runs} & \multicolumn{2}{c}{Validation-selected runs} \\",
        r"Retained / Noise & Data-Transformer & Data-MLP & Data-Transformer & Data-MLP \\", r"\hline"]
    latex_rows = []
    for retention in retentions:
        for noise in noises:
            cells = []
            for evaluation in EVALUATIONS:
                for baseline in PRIMARY:
                    row = matrices[(evaluation, baseline, retention, noise)]
                    cells.append(rf"${number(row.baseline_minus_transpinn_nrmse)}\;[{number(row.simultaneous_low)},\,{number(row.simultaneous_high)}]$")
            latex_rows.append(rf"{retention}\% / {noise}\% & " + " & ".join(cells) + r" \\")
    note = ("Each entry gives baseline minus TransPINN macro NRMSE with its 95\\% familywise simultaneous interval. "
        "Positive values favor TransPINN. Adjustment covers all 36 displayed comparisons. "
        "All nine conditions and both evaluation rules are retained.")
    (destination / "stage2_intervals_table.tex").write_text(table_wrapper(
        "Paired forecasting differences across the complete perturbation grid.",
        "tab:robustness_full_grid", "lcccc", "\n".join(header + latex_rows), note), encoding="utf-8")
    return paths


def synthetic_recovery_table(source):
    """Construct an audited 80-bin test recovery table without forecast cherry-picking."""
    models = ("TransPINN", "Matched-PINN", "NoAttn-PINN", "DDE-LS")
    quantities = ("parameter_log_rmse", "latent_E_H_U_M1_M2_nrmse")
    fields = [f"{quantity}_{stat}" for quantity in quantities for stat in ("mean", "std")]
    fields += [f"selected_{quantity}" for quantity in quantities]
    paths, frames = [], {}
    for scenario in SCENARIOS:
        path = source / f"test_{scenario}_h80" / "recovery_summary.csv"
        frame = pd.read_csv(path)
        if not {"model", *fields} <= set(frame):
            raise ValueError(f"Missing parameter/state recovery columns: {path}")
        if frame.model.duplicated().any() or set(frame.model) != set(SYNTHETIC_MODELS):
            raise ValueError(f"Incomplete model recovery grid: {path}")
        frame = frame.set_index("model").loc[list(models)]
        if not np.isfinite(frame[fields]).all().all() or (frame[fields] < 0).any().any():
            raise ValueError(f"Missing or invalid physical-model recovery result: {path}")
        paths.append(path)
        frames[scenario] = frame
    rows = [r"Model & Parameter log-RMSE & Latent-state NRMSE \\", r"\hline"]
    for scenario in SCENARIOS:
        rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{SCENARIO_LABELS[scenario]}}}}} \\")
        for model in models:
            row = frames[scenario].loc[model]
            cells = [rf"${mean_sd(row[f'{quantity}_mean'], row[f'{quantity}_std'])}\;[{number(row[f'selected_{quantity}'])}]$"
                     for quantity in quantities]
            rows.append(model + " & " + " & ".join(cells) + r" \\")
        rows.append(r"\hline")
    note = ("Mean and sample SD span three seeds after averaging ten test trajectories. Brackets show validation-selected results. "
        "Parameter log-RMSE averages $\\log^2(\\widehat\\theta_j/\\theta_j)$ over eight coefficients before taking the square root. "
        "Reporting gains use raw units. Latent NRMSE averages $E,H,U,M_1,M_2$ errors over 80 test-bin right endpoints, "
        "each normalized by its true training RMS. History and nuisance coefficients are fixed. "
        "Data-only models do not estimate these quantities.")
    content = table_wrapper("Conditional parameter and latent-state recovery on independent synthetic test trajectories.",
                            "tab:synthetic_recovery", "lcc", "\n".join(rows), note)
    return content, paths


def synthetic_runtime_table(tables):
    rows = [r"Model & Execution & Model-consistent & Media-decay mismatch \\", r"\hline"]
    for model in SYNTHETIC_MODELS:
        cells = []
        for scenario in SCENARIOS:
            row = tables[(scenario, 80)].loc[model]
            mean, sd = float(row.runtime_seconds_mean), float(row.runtime_seconds_std)
            if not np.isfinite([mean, sd]).all() or mean <= 0 or sd < 0:
                raise ValueError(f"Invalid fitting runtime: {model}/{scenario}")
            cells.append(rf"${mean:.2f}\pm{sd:.2f}$")
        execution = "CPU, serial" if model == "DDE-LS" else "GPU, amortized"
        rows.append(model + " & " + execution + " & " + " & ".join(cells) + r" \\")
    note = ("Seconds per fit are summarized by mean and sample SD over individual test-trajectory fits. "
        "Neural times amortize GPU optimization and validation within each batch. DDE-LS uses serial CPU optimization "
        "and excludes one-time compilation. Budgets are 1040 Adam updates for neural models and at most 400 residual "
        "evaluations plus numerical Jacobians for each DDE start. Hardware and optimization units differ, so no speedup is inferred.")
    return table_wrapper("Computational cost of the independent synthetic test fits.",
                         "tab:synthetic_runtime", "llcc", "\n".join(rows), note)


def publish_synthetic(source, destination, partition):
    tables, paths = {}, []
    for scenario in SCENARIOS:
        for horizon in (20, 40, 80):
            path = source / f"{partition}_{scenario}_h{horizon}" / "summary.csv"
            tables[(scenario, horizon)] = load_summary(path, SYNTHETIC_MODELS,
                                                       30 if partition == "test" else 9)
            paths.append(path)
    if partition == "test":
        recovery_content, recovery_paths = synthetic_recovery_table(source)
        paths += recovery_paths
    rows = [r"Model & 20 bins & 40 bins & 80 bins \\", r"\hline"]
    for scenario in SCENARIOS:
        rows.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{SCENARIO_LABELS[scenario]}}}}} \\")
        for model in SYNTHETIC_MODELS:
            cells = []
            for horizon in (20, 40, 80):
                row = tables[(scenario, horizon)].loc[model]
                cells.append(rf"${mean_sd(row.nrmse_mean, row.nrmse_std)}\;[{number(row.selected_nrmse)}]$")
            rows.append(model + " & " + " & ".join(cells) + r" \\")
        rows.append(r"\hline")
    description = "held-out test" if partition == "test" else "development"
    count = 10 if partition == "test" else 3
    caption = f"Independent synthetic {description} forecasting with integrated-bin observations."
    note = (f"NRMSE is averaged over {count} trajectories per scenario. Mean and sample SD span three seeds. "
        "Brackets show validation-selected results. Paired scenarios share trajectory parameters. Media decay is 1.2 "
        "in the mismatch generator and fixed at 1 in the physics-informed and numerical DDE fits. DDE-LS independently estimates eight coefficients "
        "with known history and fixed nuisance terms. Numerical and neural optimization budgets differ.")
    if partition == "development":
        note += " Development trajectories are reported separately and do not enter held-out test inference."
    label = "tab:synthetic_forecast" if partition == "test" else "tab:synthetic_development"
    (destination / f"synthetic_{partition}_table.tex").write_text(table_wrapper(
        caption, label, "lccc", "\n".join(rows), note), encoding="utf-8")
    if partition == "test":
        (destination / "synthetic_recovery_table.tex").write_text(recovery_content, encoding="utf-8")
        (destination / "synthetic_runtime_table.tex").write_text(synthetic_runtime_table(tables), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), layout="constrained", sharex=True)
    for r, metric in enumerate(("nrmse_mean", "selected_nrmse")):
        for c, scenario in enumerate(SCENARIOS):
            ax = axes[r, c]
            for k, model in enumerate(SYNTHETIC_MODELS):
                values = np.array([tables[(scenario, h)].loc[model, metric] for h in (20, 40, 80)])
                if np.any(values <= 0):
                    raise ValueError("Logarithmic synthetic figure requires strictly positive errors")
                ax.plot((20, 40, 80), values, color=COLORS[model], marker=("o", "s", "^", "D", "v", "P")[k],
                        ms=3.5, lw=1, label=model)
            ax.set_yscale("log")
            ax.grid(alpha=.16)
            ax.set_xticks((20, 40, 80))
            ax.set_title(SCENARIO_LABELS[scenario] + "\n" + ("All-run mean" if r == 0 else "Validation-selected runs"))
            if c == 0:
                ax.set_ylabel("Test NRMSE")
            if r == 1:
                ax.set_xlabel("Forecast horizon (bins)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, destination, f"synthetic_{partition}_horizons")
    (destination / f"synthetic_{partition}_horizons_caption.txt").write_text(caption + " " + note +
        " The logarithmic axis shows differences in error magnitude. Lines connect forecast horizons and are not fitted trends.\n", encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empirical-analysis", type=Path, default=ROOT / "output/robustness/empirical/analysis")
    parser.add_argument("--synthetic-analysis", type=Path, default=ROOT / "output/robustness/synthetic/comparison")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("stage1", "stage2", "synthetic", "all"), default="all")
    parser.add_argument("--synthetic-partition", choices=("test", "development", "all"), default="all")
    args = parser.parse_args()
    args.empirical_analysis = long_path(args.empirical_analysis)
    args.synthetic_analysis = long_path(args.synthetic_analysis)
    args.output = long_path(args.output)
    stages = ("stage1", "stage2", "synthetic") if args.stage == "all" else (args.stage,)
    font = configure_fonts()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage in stages:
        if stage == "stage1":
            paths += publish_stage1(args.empirical_analysis, args.output)
        elif stage == "stage2":
            paths += publish_stage2(args.empirical_analysis, args.output)
        else:
            partitions = ("test", "development") if args.synthetic_partition == "all" else (args.synthetic_partition,)
            for partition in partitions:
                paths += publish_synthetic(args.synthetic_analysis, args.output, partition)
    manifest = {"stage": args.stage, "font": "Times New Roman", "font_file": font,
        "source_sha256": {str(path.relative_to(long_path(ROOT)) if path.is_relative_to(long_path(ROOT)) else path):
                           hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "publication_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Only complete audited groups are rendered. No training or analysis results are modified."}
    (args.output / f"publication_manifest_{args.stage}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Published {args.stage} from {len(paths)} audited input files to {args.output}")


if __name__ == "__main__":
    main()
