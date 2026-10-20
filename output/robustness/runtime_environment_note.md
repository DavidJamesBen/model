# Observed GPU timing conditions

During stage 2 on 18 September 2026 at about 13:52 UTC, NVIDIA driver 576.52
reported `SW Thermal Slowdown: Active` on the RTX 5070 Laptop GPU. The SM
clock was about 225 MHz, the reported GPU temperature was 60 C, and the
power draw was about 30 W. The current power limit was 105 W. These readings
are observations of the driver state, not a diagnosis of the underlying cause.
The computer uses external power and ventilation during execution.

No power limit, clock, thermal protection, or global system setting was
changed. The first new stage-2 TransPINN batch completed all 1,040 updates
for 64 independent fits in 427.2 seconds, with a peak allocated memory of
3.96 GB. Its predictions, masks, perturbations and validation traces passed
the independent numerical audit.

Neural runtime is the measured amortized throughput of this execution,
including the observed hardware variability. It is not an intrinsic
algorithm speed or isolated inference latency. Numerical DDE CPU timings
remain a separate measurement. Forecasting comparisons retain the frozen
parameter counts, optimizer-update budgets and validation rules.

This note records an execution condition discovered during the frozen run.
It does not change the training protocol or exclude any results.

At 22:45:33 local time (UTC+08:00), the unfinished first Harvey fold-3 batch
was stopped after about 16 minutes 40 seconds. It had produced no completion
JSON. GPU memory was then about 7,674/8,151 MiB, with 100% reported utilization,
225 MHz SM clock and about 11 W. Memory or driver pressure is suspected, but
the exact cause was not established. All 1,230 completed unique fits were
preserved. Their JSON hashes are recorded in
`recovery/resource_rescheduling_record.json`.

The remaining independent fits resume through `experiments/resume_robustness.py`
in batches of at most 32, below the frozen maximum of 64. The original
training code, data, seeds, per-fit update budgets and selection rules are
unchanged. Only uncommitted computation is repeated from its original seed.
The scheduler verifies frozen hashes and records each actual batch size.

The resource change passed a separate 32-versus-64 batch check on fixed
artificial inputs, without reading experimental outcomes. Initial predictions
and losses agreed exactly. The largest prediction difference after one Adam
step was 3.58e-7. Details, tolerances and the attention key-bias invariance
check are in `batch32_vs64_parity.json`. This supports short-step numerical
agreement, not bitwise identity throughout long optimization trajectories.
