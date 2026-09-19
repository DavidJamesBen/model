"""Check the concatenated batched DDE objective against the original serial RHS.

The reference evaluates data, center, finite-difference, delayed and history
queries separately. Float64 separates algebraic errors from float32 roundoff.
The audit also checks the learned-delay gradient by central finite differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import robustness_models as models
import run_multievent_baselines as base
import run_robustness as runner
from check_robustness_models import clip_independent


def evaluate_batched(template, parameters, buffers, times, scales, denom,
                     labels, observed, quadrature, detach_delayed=False):
    batch = times.shape[0]
    train = labels.shape[1]
    qn = len(quadrature)
    colloc = torch.linspace(0, 1, 96, dtype=times.dtype, device=times.device)[None, :, None].expand(batch, -1, -1)
    plus = (colloc + 0.0025).clamp(0, 1)
    minus = (colloc - 0.0025).clamp(0, 1)
    tau = 0.01 + 0.21 * torch.sigmoid(parameters["core.tau.raw"]).reshape(batch, 1, 1)
    delayed = colloc - tau
    query_delayed = delayed.detach() if detach_delayed else delayed
    all_times = torch.cat((times, colloc, plus, minus, query_delayed.clamp(0, 1),
                           times.new_zeros(batch, 1, 1)), dim=1)
    all_states, theta = models.batched_forward(template, parameters, buffers, all_times)
    predictions = runner.observations(all_states[:, :train * qn], theta, True, train, quadrature)
    center, sp, sm, sd, sh = torch.split(all_states[:, train * qn:], (96, 96, 96, 96, 1), dim=1)
    history = times.new_tensor(runner.HISTORY)[None, None, :]
    sd = torch.where(delayed >= 0, sd, history.expand(batch, 96, -1))
    expected = runner.rhs(center, sd, theta)
    derivative = (sp[:, :, :9] - sm[:, :, :9]) / (plus - minus)
    residual = torch.cat((derivative - expected[:, :, :9],
                          center[:, :, 9:11] - expected[:, :, 9:11]), dim=-1)
    physics_loss = residual.square().mean((1, 2))
    history_loss = (sh[:, :, :9] - history[:, :, :9]).square().mean((1, 2))
    error = (predictions * scales[:, None, :] - labels) / denom[:, None, :]
    data_loss = (error.square() * observed[:, :, None]).sum((1, 2)) / (2 * observed.sum(1))
    return 12 * data_loss + 0.12 * physics_loss + 4 * history_loss, residual, predictions


def evaluate_serial(wrapper, times, scales, denom, labels, observed, quadrature):
    core = wrapper.core
    n = labels.shape[0]
    prediction = (core.observation(times).reshape(n, len(quadrature), 2) * quadrature[None, :, None]).sum(1)
    colloc = torch.linspace(0, 1, 96, dtype=times.dtype, device=times.device)[:, None]
    residual = base.residual_matrix(core, colloc, 0.0025)
    history = times.new_tensor(runner.HISTORY)[None, :]
    history_loss = (core(times.new_zeros(1, 1))[:, :9] - history[:, :9]).square().mean()
    error = (prediction * scales[None, :] - labels) / denom[None, :]
    data_loss = (error.square() * observed[:, None]).sum() / (2 * observed.sum())
    return 12 * data_loss + 0.12 * residual.square().mean() + 4 * history_loss, residual, prediction


def run(device):
    torch.set_num_threads(2)
    models.configure_attention()
    dtype = torch.float64
    def tensor(value):
        return torch.as_tensor(value, dtype=dtype, device=device)
    times = torch.stack((torch.linspace(0.02, 0.5, 21, dtype=dtype, device=device),
                         torch.linspace(0.03, 0.54, 21, dtype=dtype, device=device))).unsqueeze(-1)
    scales = tensor([[30, 70], [22, 95]])
    denom = tensor([[2, 3], [1, 5]])
    labels = tensor([[[1, 4], [2, 3], [3, 2], [5, 1], [2, 5], [1, 8], [3, 2]],
                     [[3, 5], [1, 5], [2, 1], [6, 3], [1, 4], [5, 1], [2, 2]]])
    observed = tensor([[1, 0, 1, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1]])
    quadrature = tensor([5 / 18, 4 / 9, 5 / 18])
    report = {"device": str(device), "dtype": str(dtype), "models": {}}
    for name in ("TransPINN", "Matched-PINN", "NoAttn-PINN"):
        individual = []
        for index, seed in enumerate((11, 22)):
            base.seed_everything(seed)
            bounds = [[0.25, 12], [0.25, 12]] if index == 0 else [[0.02, 0.7], [0.04, 1.2]]
            individual.append(models.make_model(name, bounds).to(device=device, dtype=dtype))
        template, parameters, buffers = models.stack_models(individual, device)
        arguments = (times, scales, denom, labels, observed, quadrature)
        loss, residual, predictions = evaluate_batched(template, parameters, buffers, *arguments)
        serial = [evaluate_serial(model, *(argument[i] if argument is not quadrature else argument
                                           for argument in arguments))
                  for i, model in enumerate(individual)]
        serial_loss = torch.stack([item[0] for item in serial])
        serial_residual = torch.stack([item[1] for item in serial])
        serial_predictions = torch.stack([item[2] for item in serial])
        # Original history is float32, whereas this audit's history is float64.
        # Its maximum constant rounding (about 7e-9) bounds the RHS discrepancy.
        torch.testing.assert_close(loss, serial_loss, rtol=1e-9, atol=2e-9)
        torch.testing.assert_close(residual, serial_residual, rtol=2e-6, atol=2e-8)
        torch.testing.assert_close(predictions, serial_predictions, rtol=1e-12, atol=1e-12)
        loss.sum().backward()
        for item in serial:
            item[0].backward()
        max_grad = 0.0
        for index, model in enumerate(individual):
            for key, parameter in model.named_parameters():
                assert parameter.grad is not None
                torch.testing.assert_close(parameters[key].grad[index], parameter.grad, rtol=2e-6, atol=2e-8)
                max_grad = max(max_grad, float((parameters[key].grad[index] - parameter.grad).abs().max().detach()))
        delay_gradient = parameters["core.tau.raw"].grad.detach().clone()
        eps = 1e-5
        finite_difference = []
        for index in range(2):
            with torch.no_grad():
                parameters["core.tau.raw"][index] += eps
                positive = evaluate_batched(template, parameters, buffers, *arguments)[0][index]
                parameters["core.tau.raw"][index] -= 2 * eps
                negative = evaluate_batched(template, parameters, buffers, *arguments)[0][index]
                parameters["core.tau.raw"][index] += eps
                finite_difference.append((positive - negative) / (2 * eps))
        finite_difference = torch.stack(finite_difference)
        torch.testing.assert_close(delay_gradient, finite_difference, rtol=2e-5, atol=2e-9)
        detached_loss = evaluate_batched(template, parameters, buffers, *arguments, detach_delayed=True)[0]
        detached_gradient = torch.autograd.grad(detached_loss.sum(), parameters["core.tau.raw"])[0]
        delay_query_contribution = delay_gradient - detached_gradient.detach()
        assert bool((delay_query_contribution.abs() > 1e-8).all())
        batch_optimizer = torch.optim.Adam(parameters.values(), lr=0.002, foreach=False)
        serial_optimizers = [torch.optim.Adam(model.parameters(), lr=0.002, foreach=False) for model in individual]
        clip_independent(parameters)
        for model in individual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        batch_optimizer.step()
        for optimizer in serial_optimizers:
            optimizer.step()
        max_update = 0.0
        for index, model in enumerate(individual):
            for key, parameter in model.named_parameters():
                torch.testing.assert_close(parameters[key][index], parameter, rtol=2e-6, atol=2e-8)
                max_update = max(max_update, float((parameters[key][index] - parameter).abs().max().detach()))
        record = {
            "max_loss_absolute_error": float((loss - serial_loss).abs().max().detach()),
            "max_residual_absolute_error": float((residual - serial_residual).abs().max().detach()),
            "max_gradient_absolute_error": max_grad,
            "max_adam_update_absolute_error": max_update,
            "delay_raw_autograd": delay_gradient.cpu().tolist(),
            "delay_raw_finite_difference": finite_difference.cpu().tolist(),
            "delay_query_gradient_contribution": delay_query_contribution.cpu().tolist(),
            "all_checks_passed": True,
        }
        report["models"][name] = record
        print(name, json.dumps(record), flush=True)
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
