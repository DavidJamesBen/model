"""Independent integrated-flux DDE data for the frozen robustness protocol.

The method of steps uses RK4 and piecewise-linear delay interpolation. Two
cumulative observable states are integrated at the same RK4 stages as the
human/media states. Differences of cumulative fluxes are the observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit

from revision_dde import HISTORY, LOWER, UPPER, NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/robustness/synthetic"
BIN_EDGES = np.linspace(0.0, 1.0, 201)
DEVELOPMENT_SEEDS = (20260921, 20260922, 20260923)
TEST_SEEDS = tuple(range(20261001, 20261011))
COUNT_SCALE = 10000.0


@njit(cache=True)
def _rhs(x, xd, p, media_decay):
    su, se, e, h, r, d, u, m1, m2 = x[:9]
    load = p[5] + m1 + m2
    gr = p[3] * m2 * p[5] / ((1 + m1) * load)
    gd = p[4] * m1 * p[5] / ((1 + m2) * load)
    gu = .3 * (m1 + m2) / load
    inc = (p[0] * r + p[1] * d) * (su + .55 * se)
    past = (p[0] * xd[4] + p[1] * xd[5]) * (xd[0] + .55 * xd[1])
    gate = 5 * xd[7] * xd[8] / (1 + 5 * xd[7] * xd[8])
    q = gate * past * np.exp(-1.4 * p[2])
    return np.array([
        -.4 * m1 * su + .25 * m2 * se - (p[0] * r + p[1] * d) * su,
        .4 * m1 * su - .25 * m2 * se - .55 * (p[0] * r + p[1] * d) * se,
        inc - 1.4 * e - q, q - (gr + gd + gu) * h,
        .8 * e + gr * h - (.25 + .5 * m1) * r + .25 * m2 * d,
        .4 * e + gd * h + .5 * m1 * r - (.25 + .25 * m2) * d,
        .2 * e + .25 * r + .25 * d + gu * h,
        .6 * d + .2 - media_decay * m1,
        .6 * r + .2 - media_decay * m2,
        p[6] * (.8 * e + gr * h + .25 * m2 * d),
        p[7] * (.4 * e + gd * h + .5 * m1 * r),
    ])


@njit(cache=True)
def _lookup(trajectory, t, step):
    if t <= 0:
        state = np.zeros(11)
        state[:9] = HISTORY
        return state
    position = t / step
    i = int(position)
    fraction = position - i
    return trajectory[i] * (1 - fraction) + trajectory[i + 1] * fraction


@njit(cache=True)
def _integrate(p, edges, mesh, media_decay):
    step = 1.0 / mesh
    n = int(np.ceil(edges[-1] / step)) + 1
    trajectory = np.empty((n + 1, 11))
    trajectory[0, :9] = HISTORY
    trajectory[0, 9:] = 0.0
    for i in range(n):
        x = trajectory[i]
        t = i * step
        k1 = _rhs(x, _lookup(trajectory, t - p[2], step), p, media_decay)
        xd = _lookup(trajectory, t + step / 2 - p[2], step)
        k2 = _rhs(x + step * k1 / 2, xd, p, media_decay)
        k3 = _rhs(x + step * k2 / 2, xd, p, media_decay)
        k4 = _rhs(x + step * k3, _lookup(trajectory, t + step - p[2], step), p, media_decay)
        trajectory[i + 1] = x + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    result = np.empty((len(edges), 11))
    for i in range(len(edges)):
        result[i] = _lookup(trajectory, edges[i], step)
    return result


def history_margin(theta):
    """Cohort-compatible E-history margin for the constant prescribed past."""
    past = (theta[0] * HISTORY[4] + theta[1] * HISTORY[5]) * (HISTORY[0] + .55 * HISTORY[1])
    gate = 5 * HISTORY[7] * HISTORY[8] / (1 + 5 * HISTORY[7] * HISTORY[8])
    needed = gate * past * (-np.expm1(-1.4 * theta[2])) / 1.4
    return float(HISTORY[2] - needed)


def solve_integrated(theta, bin_edges=BIN_EDGES, mesh=4000, media_decay=1.0,
                     count_scale=COUNT_SCALE):
    theta = np.asarray(theta, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if theta.shape != (8,) or np.any(theta <= 0):
        raise ValueError("Eight strictly positive conditional parameters are required")
    if 1.0 / mesh >= theta[2]:
        raise ValueError("Numerical step must be shorter than delay")
    if history_margin(theta) < -1e-12:
        raise ValueError("Cohort-incompatible constant history")
    solution = _integrate(theta, edges, int(mesh), float(media_decay))
    states = solution[:, :9]
    counts = np.diff(solution[:, 9:], axis=0) * count_scale
    return states, counts


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    out = OUT / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": 1, "rng": "NumPy PCG64", "parameter_names": NAMES,
        "lower": LOWER.tolist(), "upper": UPPER.tolist(),
        "parameter_draw": "exp(log(lower) + Uniform(0.2,0.8)*(log(upper)-log(lower)))",
        "development_seeds": DEVELOPMENT_SEEDS, "test_seeds": TEST_SEEDS,
        "history": HISTORY.tolist(), "count_scale": COUNT_SCALE,
        "observations": "200 exact-in-model binned integrated rumor/debunking-entry fluxes on [0,1]",
        "split": {"train": [0, 100], "validation": [100, 120], "test": [120, 200]},
        "horizons": [20, 40, 80], "generator_mesh": 8000,
        "refinement_mesh": 4000, "maximum_relative_count_refinement": 1e-6,
        "exclusion_rule": "Only history incompatibility, negative states, or failed refinement. No resampling or performance-based exclusions.",
        "media_decay": {"consistent": 1.0, "decay_mismatch": 1.2},
        "source_sha256": _hash(Path(__file__)),
    }
    # Written before test trajectories are generated or any model forecasts inspected.
    protocol_path = OUT / "generation_protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if old != json.loads(json.dumps(protocol)):
            raise RuntimeError("Frozen synthetic protocol differs from current source/settings")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    manifest = []
    for partition, seeds in (("development", DEVELOPMENT_SEEDS), ("test", TEST_SEEDS)):
        for seed in seeds:
            rng = np.random.Generator(np.random.PCG64(seed))
            theta = np.exp(np.log(LOWER) + rng.uniform(.2, .8, 8) * np.log(UPPER / LOWER))
            for scenario, decay in (("consistent", 1.0), ("decay_mismatch", 1.2)):
                states, y = solve_integrated(theta, mesh=8000, media_decay=decay)
                coarse_states, coarse_y = solve_integrated(theta, mesh=4000, media_decay=decay)
                relative = float(np.max(np.abs(y - coarse_y) / np.maximum(np.abs(y), 1e-12)))
                mass_error = float(np.max(np.abs(states[:, :7].sum(1) - 1)))
                if relative > 1e-6 or states.min() < -1e-10 or y.min() < 0 or mass_error > 1e-10:
                    raise AssertionError(f"Generator validity check failed: {seed}/{scenario}")
                split = np.array(["train"] * 100 + ["validation"] * 20 + ["test"] * 80)
                path = out / f"{partition}_{seed}_{scenario}.npz"
                np.savez_compressed(path, bin_edges=BIN_EDGES, times=BIN_EDGES[1:], y=y,
                    states=states, theta=theta, history=HISTORY, split=split,
                    count_scale=COUNT_SCALE, media_decay=decay, generator_seed=seed,
                    parameter_names=np.array(NAMES), partition=partition, scenario=scenario)
                manifest.append({"dataset": path.stem, "partition": partition,
                    "generator_seed": seed, "scenario": scenario, "file": str(path.relative_to(ROOT)),
                    "sha256": _hash(path), "history_margin": history_margin(theta),
                    "minimum_state": float(states.min()), "minimum_count": float(y.min()),
                    "mass_error": mass_error, "relative_count_refinement": relative,
                    "maximum_state_refinement": float(np.max(abs(states - coarse_states)))})
                print("GENERATED", path.stem, f"refinement={relative:.3g}", flush=True)
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    generate()
