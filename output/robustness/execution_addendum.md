# Execution details fixed before the full neural runs

This document supplements `output/benchmark_design/experiment_protocol.md`.
The experiment is an exploratory follow-up to the previous real-data benchmark.
The model rankings are not a stopping rule or an exclusion criterion.

## Sparse and perturbed training observations

The retained training indices are sampled uniformly without replacement.
For a given event, fold and seed, the 25% set is a subset of the 50% set.
The 50% set is a subset of the full training set. All models use the same sets.
Missing training observations contribute no data loss.

The channel RMS denominator uses the retained original training values only.
The reporting normalization estimates the total training activity as
`n_train / n_retained * sum(retained original training values)`.
Each quantity is bounded below by one. This avoids changing the effective
output amplitude solely because fewer observations are retained. Neither
normalization uses omitted observations, validation labels or test labels.

Artificial Gaussian measurement noise is added after these scales are fixed.
Thus these are controlled perturbation experiments with calibration scales
from the original retained observations. They do not simulate an application
where every calibration quantity is itself measured with unknown noise.
Negative perturbed counts are clipped to zero. Validation and test values
retain their original values. Initialization, mask and noise seeds are stored
separately. The latter two are deterministic SHA-256-derived integers.

## Independent vectorized training

Vectorization stacks independent network parameters. No weights, optimizer
moments, clipping norms or validation choices are shared across fits.
The backward pass uses the sum of the per-model objectives. A separate norm
clips each model's gradient to five. Every model completes its full budget.
The checkpoint is selected by validation macro per-bin NRMSE every ten updates.
Test predictions are generated only after selecting the checkpoint.

The CUDA runtime and source hashes are recorded in `neural_protocol.json`.
Float32 is used with TF32 and fused attention disabled. Scalar and batched
implementations are checked before the full runs. Small floating-point
differences are possible because arithmetic is grouped differently.

Batch wall-clock time is measured after device synchronization. The reported
per-fit time is batch time divided by the number of independent fits. It is
amortized throughput, not isolated single-model latency. The batch size and
total time are saved with every fit. These times must not be pooled with the
earlier CPU benchmark or presented as hardware-independent speed ratios.
The maximum independent batch size is 64. A preflight run at the largest
query length used 4.62 GB of allocated GPU memory. A batch of 90 was slower
and close to the device capacity, so it is not used for the full runs.

## Synthetic observation units and numerical baseline

Each neural observation integrates its nonnegative activity head using
three-point Gauss-Legendre quadrature inside the same bins as the generator.
For a raw reporting coefficient `g`, the coefficient in the normalized neural
head is `g * 10000 * bin_width / training_activity_total`. Its parameter bounds
are transformed by the same factor. Recovery errors undo this conversion.
The nine latent states are not training targets. The prescribed initial
history and fixed background rates are common modeling assumptions.

The synthetic neural budget is 1,040 Adam updates for each of seeds 11, 22 and
33. The numerical DDE fit uses three independent initializations with these
same seed identifiers. It has eight adjustable coefficients and a limit of
400 nonlinear least-squares function evaluations per initialization. Its
solver tolerances, convergence status and CPU runtime are reported separately.
Reaching this limit does not exclude a fit. This computational budget is not
equated with an Adam update count.

All ten test trajectories are retained in each structural condition.
The three development trajectories are summarized separately. Forecast
horizons are 20, 40 and 80 bins from the common forecast origin at bin 120.
Parameter recovery is conditional on the known history and fixed coefficients.
It is not evidence that the full rumor model is structurally identifiable.
