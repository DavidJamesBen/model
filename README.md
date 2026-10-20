# TransPINN source code and supplementary materials

Repository: https://github.com/DavidJamesBen/model

This package contains the code, processed data and archived results for the
TransPINN manuscript. Run all commands from this directory. Use a short path
when extracting on Windows, for example D:/TransPINN.

## Contents

- `paper.pdf`: manuscript with black revision text.
- `TransPINN-key-changes.tex`, `bib.bib` and image files: manuscript sources.
- `TransPINN-*-supplement.pdf` and `.tex`: parameter identifiability and Lyapunov supplements.
- `experiments/`: training, numerical analysis, figure generation and dependency files.
- `data/processed/`: Harvey, CSDC-Rumor, Ottawa and Sydney activity tables with provenance metadata.
- `output/`: archived settings, seed records, fitted checkpoints, predictions, tables and figures.
- `SHA256SUMS.txt`: checksums for every packaged file except this checksum list.

## Environments

Use Python 3.12. Install the original CPU benchmark environment with:

```sh
python -m pip install -r experiments/requirements-identifiability.txt
```

Use a separate Python 3.12.10 environment for the extended robustness study:

```sh
python -m pip install -r experiments/requirements-robustness.txt
```

The latter environment uses CUDA 12.8 PyTorch wheels. Training requires a
compatible NVIDIA driver. Exact runtime details are in
`output/robustness/neural_protocol.json`. Archived source files and protocol
documents retain their original bytes so that experiment hashes remain valid.
Optional imports from local `tmp` dependency directories are harmless when
packages are installed normally. No such local dependencies are required.

## Recompute tables and figures from archived predictions

Original rolling evaluation:

```sh
python experiments/analyze_multievent_results.py --bootstrap-reps 5000
python experiments/analyze_seed_ensembles.py
python experiments/analyze_fair_benchmark.py
python experiments/analyze_mechanism_ablation.py
python experiments/publish_delay_ablation.py
python experiments/publish_identifiability.py
python experiments/publish_lyapunov_analysis.py
```

Extended robustness and synthetic recovery, in the robustness environment:

```sh
python experiments/robustness_status.py
python experiments/analyze_robustness.py --empirical-root output/robustness/empirical --scope all
python experiments/analyze_robustness.py --synthetic-root output/robustness
python experiments/publish_robustness.py --stage all
```

Analysis commands can replace derived tables and figures. Retain an untouched
copy of the archive when comparing regenerated results. Bootstrap analysis
can take substantial time. Hardware and library versions affect numerical results.

## Training and numerical studies

The following entry points use the packaged processed data:

```sh
python experiments/run_multievent_baselines.py --seeds 11 22 33
python experiments/run_revision_checks.py
python experiments/run_fair_benchmark.py
python experiments/select_baseline.py delay
python experiments/run_delay_ablation.py
python experiments/select_baseline.py frozen
python experiments/run_identifiability.py --audit --event harvey
python experiments/run_identifiability.py --event csdc
python experiments/run_identifiability.py --event ottawa
python experiments/run_identifiability.py --event sydney
python experiments/run_lyapunov_analysis.py
```

Extended training uses `run_robustness.py --stage all --device cuda --batch-size 32`.
Synthetic observations and the independent numerical baseline use
`robustness_synthetic.py` and `robustness_dde_baseline.py --partition all`.
Run these scripts as `python experiments/<script>`. Existing completed runs
can be reused only when their frozen configuration hashes match.

## Archived baseline versions

The default `run_multievent_baselines.py` matches the frozen hashes in the
matched benchmark and robustness protocols. The later delay version adds
acceptance of a fixed zero delay. Both original files are in
`experiments/versions/`. Use `select_baseline.py delay` for the zero-delay
training experiment, then `select_baseline.py frozen` for other experiments.
This switch changes two validation lines. It does not change archived results.

## Result map

| Study | Location |
|---|---|
| Original rolling forecasts and seed selection | `output/tables`, `output/predictions`, `output/figures` |
| Matched model size comparison | `output/fair_benchmark` |
| Mechanism comparison | `output/rigorous_revision`, `output/mechanism_ablation` |
| Zero delay comparison | `output/delay_ablation` |
| Conditional parameter intervals | `output/identifiability` |
| Lyapunov estimates | `output/lyapunov` |
| Extended empirical fits | `output/robustness/empirical` |
| Synthetic data and numerical DDE fits | `output/robustness/synthetic` |
| Synthetic neural fits | `output/robustness/synthetic_neural` |
| Extended publication tables and figures | `output/robustness/publication` |

Additional forecast and delay sensitivity directories contain supporting
diagnostics. They do not replace the primary comparisons. Design documents
record the protocol at the time of specification. Current completion counts
are in `output/robustness/execution_status.json`. Runtime notes explain the
recorded hardware conditions. Seeds and bounds are stored with each study.

## Data provenance

The original CSDC-Rumor and RumourEval records remain subject to their source
terms and are not included. Processed activity tables are sufficient for the
training and analysis commands above. `data/processed/metadata.json` records
the sources, mappings and hashes. Rebuilding processed tables with
`prepare_multievent_data.py` requires the original datasets at the paths
specified in that script.

The Harvey workbook is an author-provided aggregation. Tweet identifiers,
labeling provenance and the derivation of noninteger counts are unavailable.
The package retains that limitation. Conditional parameter intervals depend
on the fixed settings and the specified observation model.

## Compile the manuscript and supplements

Use a TeX installation with IEEEtran and the packages listed in the sources:

```sh
pdflatex TransPINN-key-changes
bibtex TransPINN-key-changes
pdflatex TransPINN-key-changes
pdflatex TransPINN-key-changes
pdflatex TransPINN-identifiability-supplement
bibtex TransPINN-identifiability-supplement
pdflatex TransPINN-identifiability-supplement
pdflatex TransPINN-identifiability-supplement
pdflatex TransPINN-lyapunov-supplement
pdflatex TransPINN-lyapunov-supplement
```

The packaged results are existing experiment records. Packaging does not
constitute a new training run. No additional license grant is implied for
third-party data or dependencies.
