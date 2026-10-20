"""Compare independent batches of32/64 without reading any experimental outcomes.

The shared first32 initializations and fixed artificial inputs receive the same
masked data loss, full DDE/history loss where applicable, independent clipping,
and one Adam update. Run only after the production GPU process has exited.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

import robustness_models as models
import run_multievent_baselines as base
import run_robustness as frozen
from analyze_robustness import long_path
from check_robustness_models import clip_independent, key_bias_slice


def objective(template, parameters, buffers, times, targets, observed):
    batch, n = times.shape[:2]
    device = times.device
    if template.physical:
        collocation = torch.linspace(0., 1., 96, device=device)[None, :, None].expand(batch, -1, -1)
        plus, minus = (collocation + .0025).clamp(0, 1), (collocation - .0025).clamp(0, 1)
        tau = .01 + .21 * torch.sigmoid(parameters["core.tau.raw"]).reshape(batch, 1, 1)
        delayed_times = collocation - tau
        all_times = torch.cat((times, collocation, plus, minus, delayed_times.clamp(0, 1),
                               times.new_zeros(batch, 1, 1)), dim=1)
        states, theta = models.batched_forward(template, parameters, buffers, all_times)
        predicted = states[:, :n, 9:11] * theta[:, None, 6:8]
        center, sp, sm, delayed, initial = torch.split(states[:, n:], (96, 96, 96, 96, 1), dim=1)
        history = times.new_tensor(frozen.HISTORY)[None, None, :]
        delayed = torch.where(delayed_times >= 0, delayed, history.expand(batch, 96, -1))
        derivative = (sp[:, :, :9] - sm[:, :, :9]) / (plus - minus)
        expected = frozen.rhs(center, delayed, theta)
        residual = torch.cat((derivative - expected[:, :, :9], center[:, :, 9:11] - expected[:, :, 9:11]), dim=-1)
        physics_loss = residual.square().mean((1, 2))
        history_loss = (initial[:, :, :9] - history[:, :, :9]).square().mean((1, 2))
    else:
        states, theta = models.batched_forward(template, parameters, buffers, times)
        predicted = states
        physics_loss = history_loss = times.new_zeros(batch)
    # Fixed raw-count scale100 and RMS denominator1 are artificial inputs only.
    data_loss = (((predicted * 100. - targets).square() * observed[:, :, None]).sum((1, 2))
                 / (2 * observed.sum(1)))
    # Full physical weight exercises the eventual objective, not just the tiny
    # first-update ramp. Both batch sizes use exactly these same weights.
    total = 12. * data_loss + .2 * physics_loss + 4. * history_loss
    return total, states, theta, torch.stack((data_loss, physics_loss, history_loss), dim=1)


def snapshot(model_name: str, batch: int, device, train_points: int = 100) -> dict:
    individuals = []
    for index in range(batch):
        base.seed_everything(1001 + index)
        individuals.append(models.make_model(model_name))
    template, parameters, buffers = models.stack_models(individuals, device)
    axis = torch.linspace(.005, .5, train_points, device=device)
    times = axis[None, :, None].expand(batch, -1, -1).clone()
    replica = torch.arange(batch, device=device, dtype=torch.float32)[:, None, None]
    targets = torch.stack((.8 + .2 * torch.sin(6. * axis), .4 + .1 * torch.cos(5. * axis)), dim=-1)[None]
    targets = targets * (1 + .002 * replica)
    observed = ((torch.arange(train_points, device=device)[None, :] +
                 torch.arange(batch, device=device)[:, None]) % 4 != 0).to(torch.float32)
    optimizer = torch.optim.Adam(list(parameters.values()), lr=.002, foreach=False)
    total, state, theta, losses = objective(template, parameters, buffers, times, targets, observed)
    initial = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    total.sum().backward()
    if any(value.grad is None for value in parameters.values()):
        raise AssertionError("At least one parameter is disconnected from the objective")
    gradients = {name: value.grad.detach().cpu().clone() for name, value in parameters.items()}
    norms = clip_independent(parameters)
    optimizer.step()
    with torch.no_grad():
        _, updated_state, updated_theta, updated_losses = objective(template, parameters, buffers, times, targets, observed)
        preserved = {}
        for name, value in parameters.items():
            selection = key_bias_slice(name, value[0])
            if selection is not None:
                preserved[name] = value[:, selection].clone()
                value[:, selection].add_(.125)
        _, shifted_state, _, _ = objective(template, parameters, buffers, times, targets, observed)
        for name, value in preserved.items():
            parameters[name][:, key_bias_slice(name, parameters[name][0])].copy_(value)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {
        "initial": initial, "gradients": gradients,
        "updated": {name: value.detach().cpu().clone() for name, value in parameters.items()},
        "state": state.detach().cpu(), "theta": theta.detach().cpu(), "losses": losses.detach().cpu(),
        "updated_state": updated_state.detach().cpu(), "updated_theta": updated_theta.detach().cpu(),
        "updated_losses": updated_losses.detach().cpu(), "shifted_state": shifted_state.detach().cpu(),
        "preclip_norm": norms.detach().cpu(),
    }
    return result


def compare(small: dict, large: dict) -> dict:
    failures, report = [], {}
    def close(label, a, b, rtol=3e-4, atol=3e-6):
        b = b[:len(a)]
        difference = float((a - b).abs().max())
        try:
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        except AssertionError as error:
            failures.append({"quantity": label, "reason": str(error)[:700]})
        return difference
    for key in ("state", "theta", "losses", "updated_state", "updated_theta", "updated_losses", "preclip_norm"):
        report[f"max_{key}_absolute_difference"] = close(key, small[key], large[key])
    report["max_initial_parameter_difference"] = 0.
    report["max_gradient_absolute_difference"] = 0.
    report["max_adam_parameter_difference_all"] = 0.
    report["max_adam_parameter_difference_excluding_invariant_key_bias"] = 0.
    invariant, differences = {}, {}
    for name in small["initial"]:
        initial_small, initial_large = small["initial"][name], large["initial"][name]
        report["max_initial_parameter_difference"] = max(report["max_initial_parameter_difference"],
            close(f"initial:{name}", initial_small, initial_large, rtol=0., atol=0.))
        ga, gb = small["gradients"][name], large["gradients"][name][:len(initial_small)]
        report["max_gradient_absolute_difference"] = max(report["max_gradient_absolute_difference"], close(f"gradient:{name}", ga, gb))
        ua, ub = small["updated"][name], large["updated"][name][:len(initial_small)]
        difference = float((ua - ub).abs().max())
        report["max_adam_parameter_difference_all"] = max(report["max_adam_parameter_difference_all"], difference)
        if difference > 1e-7:
            differences[name] = difference
        selection = key_bias_slice(name, initial_small[0])
        mask = torch.ones_like(initial_small[0], dtype=torch.bool)
        if selection is not None:
            mask[selection] = False
            maxima = torch.maximum(ga[:, selection].abs().amax(1), gb[:, selection].abs().amax(1))
            scale = torch.maximum(ga.abs().amax(1), gb.abs().amax(1)).clamp_min(1e-3)
            bound = 32 * torch.finfo(ga.dtype).eps * scale
            invariant[name] = {"max_key_gradient_noise": float(maxima.max()),
                               "max_noise_to_roundoff_bound": float((maxima / bound).max())}
            if bool((maxima > bound).any()):
                failures.append({"quantity": f"key_bias_gradient:{name}", "reason": "Exceeds inherited floating-point invariance bound"})
        effective_difference = close(f"updated_effective:{name}", ua[:, mask], ub[:, mask])
        report["max_adam_parameter_difference_excluding_invariant_key_bias"] = max(
            report["max_adam_parameter_difference_excluding_invariant_key_bias"], effective_difference)
    for label, record in (("batch32", small), ("batch64", large)):
        report[f"max_{label}_key_bias_perturbation_output_error"] = close(
            f"key_bias_invariance:{label}", record["updated_state"], record["shifted_state"], rtol=3e-5, atol=3e-7)
    report.update({"attention_key_bias_gradient_noise": invariant,
                   "named_adam_parameter_differences_over_1e-7": differences,
                   "failures": failures, "passed": not failures})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-points", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("output/robustness/batch32_vs64_parity.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    models.configure_attention()
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    report = {
        "diagnostic": "Independent computational batch-size check, no experimental outcomes read",
        "device": str(device), "torch": torch.__version__, "batch_sizes": [32, 64],
        "shared_initialization_seeds": list(range(1001, 1033)), "training_points": args.train_points,
        "input": "Fixed artificial positive training labels and75% observation masks; no dataset files or test metrics",
        "objective": "12*normalized masked data MSE +0.2*full DDE residual MSE +4*history MSE, where applicable",
        "optimizer": "One Adam update at0.002 after per-model norm clipping5; foreach=False",
        "comparison_tolerance": {"rtol": 3e-4, "atol": 3e-6},
        "key_bias_exception": "Only analytically invariant MHA key-bias update slices are excluded. Their gradients and+0.125 output perturbation are checked separately.",
        "claim_scope": "Numerical agreement within stated tolerances, not bitwise identity or equivalence of all long optimization trajectories",
        "models": {},
        "source_sha256": {p.name: hashlib.sha256(long_path(p).read_bytes()).hexdigest() for p in (
            Path(__file__), Path(frozen.__file__), Path(models.__file__), Path(base.__file__))},
    }
    output = long_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for model_name in ("TransPINN", "Data-Transformer"):
        snapshots = []
        for batch in (32, 64):
            snapshots.append(snapshot(model_name, batch, device, args.train_points))
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["models"][model_name] = compare(*snapshots)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(model_name, json.dumps(report["models"][model_name]), flush=True)
        del snapshots
    report["passed"] = all(item["passed"] for item in report["models"].values())
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("Numerical checks exceeded declared tolerances; inspect the saved differences before changing execution.")


if __name__ == "__main__":
    main()
