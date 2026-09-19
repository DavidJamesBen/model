"""Verify serial/vectorized outputs, gradients and independent Adam updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import robustness_models as models
import run_multievent_baselines as base


def objective(state, theta):
    weights = torch.linspace(0.1, 1.1, state.shape[-1], device=state.device)
    target = torch.linspace(0.05, 0.5, state.shape[-2], device=state.device).unsqueeze(-1)
    # The coefficient term checks that every physical coefficient remains a
    # distinct differentiable value under vectorization.
    return (((state - target) * weights) ** 2).mean(dim=(-1, -2)) * 80.0 + \
        (theta ** 2).sum(dim=-1) * 2.0


def clip_independent(params, max_norm=5.0):
    batch = next(iter(params.values())).shape[0]
    norm2 = sum(parameter.grad.reshape(batch, -1).square().sum(dim=1)
                for parameter in params.values())
    factors = (max_norm / (norm2.sqrt() + 1e-6)).clamp(max=1.0)
    for parameter in params.values():
        parameter.grad.mul_(factors.reshape(batch, *([1] * (parameter.ndim - 1))))
    return norm2.sqrt()


def key_bias_slice(name, tensor):
    if name.endswith("attention.in_proj_bias"):
        assert tensor.ndim == 1 and tensor.numel() % 3 == 0
        width = tensor.numel() // 3
        return slice(width, 2 * width)
    return None


def run(device):
    models.configure_attention()
    torch.set_num_threads(2)
    report = {
        "device": str(device), "torch": torch.__version__, "models": {},
        "adam_parameter_tolerance": {"rtol": 3e-4, "atol": 3e-6},
        "key_bias_exception": (
            "Only the key slice of MultiheadAttention.in_proj_bias is excluded "
            "from parameter-update equality. It adds a source-independent logit "
            "shift, which cancels in softmax. Its near-zero gradient is checked "
            "against floating-point precision, a +0.125 perturbation must leave "
            "outputs unchanged, and post-update outputs must still match. Adam "
            "epsilon can amplify tiny roundoff gradients in this invariant slice."
        ),
    }
    for name in models.MODEL_NAMES:
        individual = []
        for seed, bounds in ((11, [[0.25, 12.0], [0.25, 12.0]]),
                             (22, [[0.01, 0.9], [0.02, 1.7]])):
            base.seed_everything(seed)
            individual.append(models.make_model(name, bounds).to(device))
        template, parameters, buffers = models.stack_models(individual, device)
        times = torch.stack((torch.linspace(0.0, 1.0, 13, device=device),
                             torch.linspace(0.03, 0.97, 13, device=device))).unsqueeze(-1)
        batched_state, batched_theta = models.batched_forward(template, parameters, buffers, times)
        serial = [model(times[index]) for index, model in enumerate(individual)]
        serial_state = torch.stack([result[0] for result in serial])
        serial_theta = torch.stack([result[1] for result in serial])
        torch.testing.assert_close(batched_state, serial_state, rtol=2e-4, atol=2e-7)
        torch.testing.assert_close(batched_theta, serial_theta, rtol=2e-6, atol=2e-7)
        batched_optimizer = torch.optim.Adam(parameters.values(), lr=0.002)
        serial_optimizers = [torch.optim.Adam(model.parameters(), lr=0.002) for model in individual]
        objective(batched_state, batched_theta).sum().backward()
        for state, theta in serial:
            objective(state, theta).backward()
        max_gradient_error = 0.0
        invariant_gradients = {}
        for index, model in enumerate(individual):
            for key, parameter in model.named_parameters():
                assert parameter.grad is not None, (name, key)
                torch.testing.assert_close(parameters[key].grad[index], parameter.grad,
                                           rtol=3e-4, atol=3e-6)
                max_gradient_error = max(max_gradient_error, float(
                    (parameters[key].grad[index] - parameter.grad).abs().max().detach()))
                key_slice = key_bias_slice(key, parameter)
                if key_slice is not None:
                    gradient_max = max(float(parameter.grad[key_slice].abs().max()),
                                       float(parameters[key].grad[index, key_slice].abs().max()))
                    # Key bias adds the same q dot b to every source logit. Its
                    # derivative is zero analytically. Test that measured noise
                    # is at floating-point precision relative to the tensor.
                    gradient_scale = max(float(parameter.grad.abs().max()), 1e-3)
                    assert gradient_max <= 32 * torch.finfo(parameter.dtype).eps * gradient_scale, (name, key, gradient_max, gradient_scale)
                    invariant_gradients[f"run{index}:{key}"] = gradient_max
        norm = clip_independent(parameters)
        for model in individual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        batched_optimizer.step()
        for optimizer in serial_optimizers:
            optimizer.step()
        max_update_error = 0.0
        named_update_errors = {}
        max_noninvariant_update_error = 0.0
        for index, model in enumerate(individual):
            for key, parameter in model.named_parameters():
                key_slice = key_bias_slice(key, parameter)
                mask = torch.ones_like(parameter, dtype=torch.bool)
                if key_slice is not None:
                    mask[key_slice] = False
                torch.testing.assert_close(parameters[key][index][mask], parameter[mask],
                                           rtol=3e-4, atol=3e-6,
                                           msg=lambda message: f"{name} run{index} {key}: {message}")
                difference = (parameters[key][index] - parameter).abs().detach()
                maximum = float(difference.max())
                max_update_error = max(max_update_error, maximum)
                max_noninvariant_update_error = max(max_noninvariant_update_error, float(difference[mask].max()))
                if maximum > 1e-7:
                    named_update_errors[f"run{index}:{key}"] = maximum
        with torch.no_grad():
            updated_state, updated_theta = models.batched_forward(template, parameters, buffers, times)
            updated_serial = [model(times[index]) for index, model in enumerate(individual)]
            updated_serial_state = torch.stack([result[0] for result in updated_serial])
            updated_serial_theta = torch.stack([result[1] for result in updated_serial])
            torch.testing.assert_close(updated_state, updated_serial_state, rtol=3e-4, atol=3e-7)
            torch.testing.assert_close(updated_theta, updated_serial_theta, rtol=3e-4, atol=3e-7)
            # Verify the analytic key-bias invariance numerically using a large
            # perturbation, rather than hiding a discrepant effective parameter.
            preserved = {}
            for key, parameter in parameters.items():
                key_slice = key_bias_slice(key, parameter[0])
                if key_slice is not None:
                    preserved[key] = parameter[:, key_slice].clone()
                    parameter[:, key_slice].add_(0.125)
            shifted_state, _ = models.batched_forward(template, parameters, buffers, times)
            for key, original in preserved.items():
                parameters[key][:, key_bias_slice(key, parameters[key][0])].copy_(original)
            torch.testing.assert_close(shifted_state, updated_state, rtol=3e-5, atol=3e-7)
        report["models"][name] = {
            "parameters_per_model": sum(p.numel() for p in individual[0].parameters()),
            "max_output_absolute_error": float((batched_state - serial_state).abs().max().detach()),
            "max_gradient_absolute_error": max_gradient_error,
            "max_adam_update_absolute_error": max_update_error,
            "max_noninvariant_adam_update_absolute_error": max_noninvariant_update_error,
            "named_adam_update_errors_over_1e-7": named_update_errors,
            "attention_key_bias_gradient_noise": invariant_gradients,
            "max_post_update_output_absolute_error": float((updated_state - updated_serial_state).abs().max()),
            "max_key_bias_perturbation_output_error": float((shifted_state - updated_state).abs().max()),
            "preclip_gradient_norms": norm.detach().cpu().tolist(),
            "all_parameters_gradient_connected": True,
            "distinct_reporting_bounds_preserved": True,
        }
        print(name, json.dumps(report["models"][name]), flush=True)
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(torch.device(args.device))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
