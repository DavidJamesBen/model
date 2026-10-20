"""Parameter-matched models and independent-run vectorization.

Stacking accelerates independent optimization runs. The leading parameter axis
belongs to separate models, not to a shared-weight ensemble. Reporting bounds
are buffers so distinct synthetic scales remain distinct under functional_call.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.func import functional_call, stack_module_state, vmap

import run_multievent_baselines as base
import run_fair_benchmark as fair


MODEL_NAMES = ("TransPINN", "Matched-PINN", "NoAttn-PINN",
               "Data-Transformer", "Data-MLP")
PARAMETER_COUNT = 18619


class DataMLP(nn.Module):
    """A conventional data-only MLP with 18,619 trainable parameters."""

    data_only = True

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 95), nn.Tanh(), nn.Linear(95, 92), nn.Tanh(),
            nn.Linear(92, 93), nn.Tanh(), nn.Linear(93, 2),
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(-4.5)

    def observation(self, times: Tensor) -> Tensor:
        return F.softplus(self.net(base.time_features(times)))


class BenchmarkModel(nn.Module):
    """One functional-call entry point for both trajectories and coefficients."""

    def __init__(self, name: str, reporting_bounds=None) -> None:
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown benchmark model: {name}")
        self.name = name
        self.physical = not name.startswith("Data-")
        self.core = DataMLP() if name == "Data-MLP" else fair.factory(name, 16)
        bounds = torch.as_tensor(
            [[0.25, 12.0], [0.25, 12.0]] if reporting_bounds is None else reporting_bounds,
            dtype=torch.float32,
        ).clone()
        if bounds.shape != (2, 2) or not bool(torch.isfinite(bounds).all()):
            raise ValueError("reporting_bounds must contain two finite [lower, upper] rows")
        if not bool(((bounds[:, 0] >= 0) & (bounds[:, 1] > bounds[:, 0])).all()):
            raise ValueError("Reporting bounds require 0 <= lower < upper")
        self.register_buffer("reporting_bounds", bounds)
        if self.physical:
            for parameter, row in zip((self.core.reporting_r, self.core.reporting_d), bounds):
                parameter.lower, parameter.upper = float(row[0]), float(row[1])
        count = sum(parameter.numel() for parameter in self.parameters())
        if count != PARAMETER_COUNT:
            raise AssertionError(f"{name}: expected {PARAMETER_COUNT}, found {count}")

    def forward(self, times: Tensor) -> tuple[Tensor, Tensor]:
        if not self.physical:
            observations = self.core.observation(times)
            return observations, observations.new_zeros(8)
        state = self.core(times)
        core = self.core
        raw_gains = torch.stack((core.reporting_r.raw, core.reporting_d.raw))
        bounds = self.reporting_bounds
        gains = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.sigmoid(raw_gains)
        theta = torch.stack((core.beta_1(), core.beta_2(), core.delay(),
                             core.q_r(), core.q_d(), core.chi(), gains[0], gains[1]))
        return state, theta


def make_model(name: str, reporting_bounds=None) -> BenchmarkModel:
    return BenchmarkModel(name, reporting_bounds)


def configure_attention() -> None:
    """Use the same differentiable math attention kernel in serial and vmap."""
    # Fused SDP has backend-dependent vmap support. No dropout or other stochastic
    # attention is used, so disabling fusion changes neither model nor objective.
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)


def stack_models(models: Sequence[BenchmarkModel], device) -> tuple[nn.Module, dict, dict]:
    """Return a stateless template and leaf parameters for independent optimizers."""
    if not models:
        raise ValueError("At least one model is required")
    if len({model.name for model in models}) != 1:
        raise ValueError("Each vectorized batch must contain one model architecture")
    configure_attention()
    params, buffers = stack_module_state(list(models))
    params = {name: tensor.detach().to(device).requires_grad_(True)
              for name, tensor in params.items()}
    buffers = {name: tensor.detach().to(device) for name, tensor in buffers.items()}
    template = deepcopy(models[0]).to("meta")
    return template, params, buffers


def batched_forward(template: BenchmarkModel, params: dict, buffers: dict,
                    times: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate separate models on times with shape [models, samples, 1]."""
    def call_one(parameters, state_buffers, model_times):
        return functional_call(template, (parameters, state_buffers), (model_times,))
    return vmap(call_one, in_dims=(0, 0, 0), randomness="error")(params, buffers, times)
