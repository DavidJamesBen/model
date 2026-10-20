"""Fair multi-event comparison for the revised rumor-delay model.

The five neural solvers share the same eleven-output state head, revised DDE,
history, observation loss, collocation grid, optimizer, chronological split,
and seed list.  They differ only in the temporal function approximator (or, in
TimePINN, the causal weighting of otherwise identical MLP residuals).

This is a compact CPU reference experiment intended to make every manuscript
number traceable.  It uses a differentiable centered-difference residual;
there is no interpolation or label generation outside the observed bins.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "output" / "tables"
FIGURE_DIR = ROOT / "output" / "figures"
PREDICTION_DIR = ROOT / "output" / "predictions"

EVENTS = ("harvey", "csdc", "ottawa", "sydney")
MODELS = ("PINN", "LSTM-PINN", "TimePINN", "NoAttn-PINN", "TransPINN")
ACTIVE_ACTIVITY_FRACTION = 0.99
COLORS = {
    "PINN": "#7f7f7f",
    "LSTM-PINN": "#1f77b4",
    "TimePINN": "#2ca02c",
    "NoAttn-PINN": "#9467bd",
    "TransPINN": "#d62728",
}


@dataclass(frozen=True)
class Config:
    epochs: int = 260
    min_epochs: int = 220
    patience_checks: int = 18
    check_every: int = 10
    learning_rate: float = 2.0e-3
    collocation_points: int = 96
    sequence_length: int = 16
    finite_difference_step: float = 2.5e-3
    data_weight: float = 12.0
    physics_weight: float = 0.20
    history_weight: float = 4.0
    gradient_clip: float = 5.0
    causal_epsilon: float = 2.0
    causal_chunks: int = 8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def time_features(t: Tensor) -> Tensor:
    """Continuous time plus four Fourier pairs, shape [..., 9]."""
    features = [t]
    for frequency in (1.0, 2.0, 4.0, 8.0):
        features.extend(
            [torch.sin(2.0 * math.pi * frequency * t),
             torch.cos(2.0 * math.pi * frequency * t)]
        )
    return torch.cat(features, dim=-1)


def pseudo_sequence(t: Tensor, length: int) -> Tensor:
    offsets = torch.linspace(0.18, 0.0, length, device=t.device, dtype=t.dtype)
    sequence_t = torch.clamp(t.unsqueeze(1) - offsets.view(1, -1, 1), 0.0, 1.0)
    return time_features(sequence_t)


class MLPBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 11),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.net(time_features(t))


class LSTMBackbone(nn.Module):
    def __init__(self, sequence_length: int) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(
            input_size=9,
            hidden_size=36,
            num_layers=2,
            batch_first=True,
        )
        self.head = nn.Linear(36, 11)

    def forward(self, t: Tensor) -> Tensor:
        encoded, _ = self.lstm(pseudo_sequence(t, self.sequence_length))
        return self.head(encoded[:, -1, :])


class WaveAct(nn.Module):
    """Trainable sine/cosine activation used by PINNsFormer."""

    def __init__(self) -> None:
        super().__init__()
        self.sine_weight = nn.Parameter(torch.ones(1))
        self.cosine_weight = nn.Parameter(torch.ones(1))

    def forward(self, value: Tensor) -> Tensor:
        return self.sine_weight * torch.sin(value) \
               + self.cosine_weight * torch.cos(value)


class WaveFeedForward(nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden),
            WaveAct(),
            nn.Linear(hidden, hidden),
            WaveAct(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class MatchedNoAttention(nn.Module):
    """Same projection count as MHA but no mixing between time tokens."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        combined = (self.query(query) + self.key(key) + self.value(value)) / 3.0
        return self.output(combined)


class WaveEncoderLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, hidden: int,
                 use_attention: bool) -> None:
        super().__init__()
        self.attention: nn.Module
        if use_attention:
            self.attention = nn.MultiheadAttention(
                dimension, heads, dropout=0.0, batch_first=True
            )
        else:
            self.attention = MatchedNoAttention(dimension)
        self.feed_forward = WaveFeedForward(dimension, hidden)
        self.wave_1 = WaveAct()
        self.wave_2 = WaveAct()
        self.use_attention = use_attention

    def forward(self, value: Tensor) -> Tensor:
        transformed = self.wave_1(value)
        if self.use_attention:
            mixed = self.attention(transformed, transformed, transformed, need_weights=False)[0]
        else:
            mixed = self.attention(transformed, transformed, transformed)
        value = value + mixed
        return value + self.feed_forward(self.wave_2(value))


class WaveDecoderLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, hidden: int,
                 use_attention: bool) -> None:
        super().__init__()
        if use_attention:
            self.attention: nn.Module = nn.MultiheadAttention(
                dimension, heads, dropout=0.0, batch_first=True
            )
        else:
            self.attention = MatchedNoAttention(dimension)
        self.feed_forward = WaveFeedForward(dimension, hidden)
        self.wave_1 = WaveAct()
        self.wave_2 = WaveAct()
        self.use_attention = use_attention

    def forward(self, value: Tensor, encoded: Tensor) -> Tensor:
        transformed = self.wave_1(value)
        if self.use_attention:
            mixed = self.attention(transformed, encoded, encoded, need_weights=False)[0]
        else:
            mixed = self.attention(transformed, encoded, encoded)
        value = value + mixed
        return value + self.feed_forward(self.wave_2(value))


class PINNsFormerBackbone(nn.Module):
    """Compact encoder-decoder PINNsFormer with a strict attention ablation."""

    def __init__(self, sequence_length: int, use_attention: bool) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        dimension, hidden, heads = 24, 48, 4
        self.input_projection = nn.Linear(9, dimension)
        self.encoder = WaveEncoderLayer(dimension, heads, hidden, use_attention)
        self.decoder = WaveDecoderLayer(dimension, heads, hidden, use_attention)
        self.encoder_wave = WaveAct()
        self.decoder_wave = WaveAct()
        self.head = nn.Sequential(
            nn.Linear(dimension, hidden),
            WaveAct(),
            nn.Linear(hidden, hidden),
            WaveAct(),
            nn.Linear(hidden, 11),
        )

    def forward(self, t: Tensor) -> Tensor:
        source = self.input_projection(pseudo_sequence(t, self.sequence_length))
        encoded = self.encoder_wave(self.encoder(source))
        decoded = self.decoder_wave(self.decoder(source, encoded))
        return self.head(decoded[:, -1, :])


class BoundedParameter(nn.Module):
    def __init__(self, lower: float, upper: float, initial_fraction: float = 0.5):
        super().__init__()
        self.lower = lower
        self.upper = upper
        initial_fraction = min(max(initial_fraction, 1.0e-4), 1.0 - 1.0e-4)
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.raw = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    def forward(self) -> Tensor:
        return self.lower + (self.upper - self.lower) * torch.sigmoid(self.raw)


class RevisedRumorPINN(nn.Module):
    """Neural state map plus six structural and two reporting-scale groups."""

    def __init__(self, model_name: str, sequence_length: int,
                 fixed_tau: float | None = None):
        super().__init__()
        if model_name in {"PINN", "TimePINN"}:
            self.backbone = MLPBackbone()
        elif model_name == "LSTM-PINN":
            self.backbone = LSTMBackbone(sequence_length)
        elif model_name == "NoAttn-PINN":
            self.backbone = PINNsFormerBackbone(sequence_length, use_attention=False)
        elif model_name == "TransPINN":
            self.backbone = PINNsFormerBackbone(sequence_length, use_attention=True)
        else:
            raise ValueError(model_name)
        if isinstance(self.backbone, MLPBackbone):
            output_layer = self.backbone.net[-1]
        elif isinstance(self.backbone, LSTMBackbone):
            output_layer = self.backbone.head
        else:
            output_layer = self.backbone.head[-1]
        with torch.no_grad():
            # Normalized bin fluxes are typically O(1/n_train), so starting the
            # two softplus logits below zero avoids an architecture-dependent
            # warm-up dominated by implausibly large positive activity.
            output_layer.bias[-2:].fill_(-4.5)
        self.model_name = model_name
        self.beta_1 = BoundedParameter(0.10, 8.0, 0.30)
        self.beta_2 = BoundedParameter(0.05, 5.0, 0.25)
        if fixed_tau is not None and not 0.01 <= fixed_tau <= 0.22:
            raise ValueError("fixed_tau must lie in the normalized interval [0.01, 0.22]")
        self.fixed_tau = fixed_tau
        self.tau = None if fixed_tau is not None else BoundedParameter(0.01, 0.22, 0.30)
        self.q_r = BoundedParameter(0.05, 4.0, 0.25)
        self.q_d = BoundedParameter(0.05, 4.0, 0.25)
        self.chi = BoundedParameter(0.05, 3.0, 0.35)
        # Reporting scales map physical transition fluxes to normalized counts.
        self.reporting_r = BoundedParameter(0.25, 12.0, 0.05)
        self.reporting_d = BoundedParameter(0.25, 12.0, 0.05)

    def forward(self, t: Tensor) -> Tensor:
        raw = self.backbone(t)
        human = torch.softmax(raw[:, :7], dim=1)
        media = F.softplus(raw[:, 7:9])
        # The two observation heads are nonnegative transition fluxes, not
        # unconstrained cumulative stocks.  Cumulative activity is formed only
        # by summing these scored fluxes, so the two reported quantities cannot
        # contradict one another.
        activity_flux = F.softplus(raw[:, 9:11])
        return torch.cat([human, media, activity_flux], dim=1)

    def observation(self, t: Tensor) -> Tensor:
        state = self(t)
        gains = torch.stack([self.reporting_r(), self.reporting_d()])
        return state[:, 9:11] * gains

    def delay(self) -> Tensor:
        """Return the learned delay, or the prespecified delay in a sweep."""
        if self.fixed_tau is None:
            assert self.tau is not None
            return self.tau()
        return torch.tensor(self.fixed_tau, dtype=torch.float32,
                            device=next(self.parameters()).device)

    def parameter_values(self) -> dict[str, float]:
        return {
            "beta_1": float(self.beta_1().detach()),
            "beta_2": float(self.beta_2().detach()),
            "tau_normalized": float(self.delay().detach()),
            "tau_is_fixed": self.fixed_tau is not None,
            "q_R": float(self.q_r().detach()),
            "q_D": float(self.q_d().detach()),
            "chi": float(self.chi().detach()),
            "reporting_R": float(self.reporting_r().detach()),
            "reporting_D": float(self.reporting_d().detach()),
        }


def initial_history(device: torch.device) -> Tensor:
    # Seven human compartments sum exactly to one.  The final two entries are
    # unused placeholders because activity fluxes are algebraic observations,
    # not history states.
    values = [0.93, 0.03, 0.015, 0.005, 0.010, 0.005, 0.005,
              0.20, 0.20, 0.0, 0.0]
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1)


def dde_rhs(model: RevisedRumorPINN, t: Tensor, state: Tensor) -> Tensor:
    """Right-hand side of the revised survival-weighted cohort DDE."""
    su, se, exposed, hesitant, spreader, debunker, uninterested = state[:, :7].T
    media_1, media_2 = state[:, 7], state[:, 8]
    beta_1, beta_2 = model.beta_1(), model.beta_2()
    tau = model.delay()
    q_r, q_d, chi = model.q_r(), model.q_d(), model.chi()

    epsilon = 0.55
    kappa_1, kappa_2 = 0.40, 0.25
    xi_e, sigma_e, gamma_e = 0.80, 0.40, 0.20
    a_e = xi_e + sigma_e + gamma_e
    rho = 5.0
    mu, alpha = 0.25, 0.25
    delta_1, delta_2 = 0.50, 0.25
    eta_1, eta_2 = 0.60, 0.60
    v_1, v_2 = 0.20, 0.20
    decay_1, decay_2 = 1.0, 1.0
    varpi_1, varpi_2 = 1.0, 1.0
    q_u = 0.30

    incidence = (beta_1 * spreader + beta_2 * debunker) * (su + epsilon * se)

    delayed_t = t - tau
    delayed_network = model(torch.clamp(delayed_t, 0.0, 1.0))
    history = initial_history(t.device).expand_as(delayed_network)
    delayed_state = torch.where((delayed_t >= 0.0).expand_as(delayed_network),
                                delayed_network, history)
    d_su, d_se, _, _, d_r, d_d, _, d_m1, d_m2 = delayed_state[:, :9].T
    delayed_incidence = (beta_1 * d_r + beta_2 * d_d) * (d_su + epsilon * d_se)
    path_probability = rho * d_m1 * d_m2 / (1.0 + rho * d_m1 * d_m2)
    completed = path_probability * delayed_incidence * torch.exp(-a_e * tau)

    load = chi + media_1 + media_2
    gamma_r = q_r * media_2 * chi / ((1.0 + varpi_1 * media_1) * load)
    gamma_d = q_d * media_1 * chi / ((1.0 + varpi_2 * media_2) * load)
    gamma_u = q_u * (media_1 + media_2) / load

    # Optional, explicitly named mechanisms for the supplemental audit.
    # Default models and all archived baseline results are unchanged.
    if getattr(model, "mechanism_variant", "full") == "no_fatigue":
        gamma_r = q_r * media_2 / (1.0 + varpi_1 * media_1)
        gamma_d = q_d * media_1 / (1.0 + varpi_2 * media_2)
        gamma_u = torch.zeros_like(gamma_u)
    elif getattr(model, "mechanism_variant", "full") == "no_inhibition":
        gamma_r = q_r * media_2 * chi / load
        gamma_d = q_d * media_1 * chi / load

    dsu = -kappa_1 * media_1 * su + kappa_2 * media_2 * se \
          - beta_1 * spreader * su - beta_2 * debunker * su
    dse = kappa_1 * media_1 * su - kappa_2 * media_2 * se \
          - epsilon * beta_1 * spreader * se - epsilon * beta_2 * debunker * se
    de = incidence - a_e * exposed - completed
    dh = completed - (gamma_r + gamma_d + gamma_u) * hesitant
    dr = xi_e * exposed + gamma_r * hesitant \
         - (mu + delta_1 * media_1) * spreader + delta_2 * media_2 * debunker
    dd = sigma_e * exposed + gamma_d * hesitant + delta_1 * media_1 * spreader \
         - (alpha + delta_2 * media_2) * debunker
    du = gamma_e * exposed + mu * spreader + alpha * debunker + gamma_u * hesitant
    dm1 = eta_1 * debunker + v_1 - decay_1 * media_1
    dm2 = eta_2 * spreader + v_2 - decay_2 * media_2
    dcr = xi_e * exposed + gamma_r * hesitant + delta_2 * media_2 * debunker
    dcd = sigma_e * exposed + gamma_d * hesitant + delta_1 * media_1 * spreader
    return torch.stack([dsu, dse, de, dh, dr, dd, du, dm1, dm2, dcr, dcd], dim=1)


def residual_matrix(model: RevisedRumorPINN, t: Tensor, step: float) -> Tensor:
    t_plus = torch.clamp(t + step, 0.0, 1.0)
    t_minus = torch.clamp(t - step, 0.0, 1.0)
    state = model(t)
    derivative = (model(t_plus)[:, :9] - model(t_minus)[:, :9]) / (t_plus - t_minus)
    right_hand_side = dde_rhs(model, t, state)
    state_residual = derivative - right_hand_side[:, :9]
    flux_residual = state[:, 9:11] - right_hand_side[:, 9:11]
    return torch.cat([state_residual, flux_residual], dim=1)


def increment_from_cumulative(cumulative: Tensor) -> Tensor:
    first = cumulative[:1]
    return torch.cat([first, cumulative[1:] - cumulative[:-1]], dim=0)


def causal_physics_loss(residual: Tensor, config: Config) -> Tensor:
    """Causal time-weighted residual from Wang et al. (CMAME, 2024).

    The stop-gradient cumulative loss prevents later time chunks from being
    emphasized before earlier chunks have converged.  The outer mean follows
    the published causal-PINN objective; normalizing by ``weights.sum()`` would
    define a different, self-normalized loss.
    """
    point_loss = residual.square().mean(dim=1)
    chunks = torch.tensor_split(point_loss, config.causal_chunks)
    losses = torch.stack([chunk.mean() for chunk in chunks])
    prior = torch.cumsum(losses.detach(), dim=0) - losses.detach()
    weights = torch.exp(-config.causal_epsilon * prior)
    return (weights * losses).mean()


def active_core_end(frame: pd.DataFrame) -> int:
    """Shortest prefix containing 99% of each activity channel."""
    ends = []
    for channel in ("rumor_activity", "debunk_activity"):
        values = frame[channel].to_numpy(float)
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError(f"{channel}: no observed activity")
        ends.append(
            int(np.searchsorted(np.cumsum(values), ACTIVE_ACTIVITY_FRACTION * total) + 1)
        )
    return max(ends)


def load_event_folds(event: str) -> list[tuple[int, pd.DataFrame, np.ndarray, int, int]]:
    """Create up to three expanding-window folds inside the active cascade.

    The rule is fixed before fitting: L is the 99%-per-channel active core,
    h=max(2,floor(0.1L)), and origins start at ceil(0.5L) and advance by h.
    A candidate is retained only when both validation and test contain positive
    activity in both channels.  No model output is consulted.
    """
    full = pd.read_csv(DATA_DIR / f"{event}.csv", parse_dates=["timestamp"])
    core_end = active_core_end(full)
    core = full.iloc[:core_end].copy().reset_index(drop=True)
    horizon = max(2, int(math.floor(0.10 * core_end)))
    origin = int(math.ceil(0.50 * core_end))
    folds: list[tuple[int, pd.DataFrame, np.ndarray, int, int]] = []
    candidate = 0
    while origin + 2 * horizon <= core_end and len(folds) < 3:
        candidate += 1
        validation_end = origin + horizon
        test_end = origin + 2 * horizon
        activity = core[["rumor_activity", "debunk_activity"]]
        validation_positive = (activity.iloc[origin:validation_end].sum(axis=0) > 0).all()
        test_positive = (activity.iloc[validation_end:test_end].sum(axis=0) > 0).all()
        if validation_positive and test_positive:
            folds.append(
                (
                    len(folds) + 1,
                    core.iloc[:test_end].copy().reset_index(drop=True),
                    np.array([origin, validation_end, test_end], dtype=int),
                    core_end,
                    horizon,
                )
            )
        origin += horizon
    if not folds:
        raise ValueError(f"{event}: no nondegenerate expanding-window fold")
    return folds


def validation_score(model: RevisedRumorPINN, times: Tensor, increment: Tensor,
                     cumulative: Tensor, train_end: int, validation_end: int) -> float:
    with torch.no_grad():
        predicted_increment = model.observation(times)
        predicted_cumulative = torch.cumsum(predicted_increment, dim=0)
        region = slice(train_end, validation_end)
        cumulative_error = F.mse_loss(predicted_cumulative[region], cumulative[region])
        increment_error = F.mse_loss(predicted_increment[region], increment[region])
        return float(cumulative_error + increment_error)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray,
                       train_reference: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    train_rms = float(np.sqrt(np.mean(train_reference ** 2)))
    nrmse = rmse / max(train_rms, 1.0)
    nmae = mae / max(train_rms, 1.0)
    wape = float(np.sum(np.abs(error)) / max(np.sum(np.abs(actual)), 1.0))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error ** 2)) / denominator if denominator > 0 else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "nmae": nmae,
        "wape": wape,
        "r2": r2,
    }


def train_one(event: str, fold: int, frame: pd.DataFrame, split: np.ndarray,
              model_name: str, seed: int, config: Config,
              device: torch.device, fixed_tau: float | None = None) \
        -> tuple[list[dict[str, object]], pd.DataFrame, dict[str, object]]:
    seed_everything(seed)
    activity = frame[["rumor_activity", "debunk_activity"]].to_numpy(np.float32)
    train_end, validation_end, n = split.tolist()
    scales = np.maximum(activity[:train_end].sum(axis=0), 1.0).astype(np.float32)
    increment_np = activity / scales
    cumulative_np = np.cumsum(activity, axis=0) / scales
    # Observations are bin integrals placed at bin right endpoints.  The point
    # t=0 is reserved for the pre-event history and is not simultaneously used
    # as a positive first-bin observation.
    times = (torch.arange(1, n + 1, device=device, dtype=torch.float32) / n).view(-1, 1)
    increment = torch.tensor(increment_np, device=device)
    cumulative = torch.tensor(cumulative_np, device=device)
    collocation = torch.linspace(0.0, 1.0, config.collocation_points,
                                 device=device).view(-1, 1)
    model = RevisedRumorPINN(model_name, config.sequence_length, fixed_tau=fixed_tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate / 20.0
    )

    best_state: dict[str, Tensor] | None = None
    best_score = float("inf")
    best_epoch = 0
    checks_without_improvement = 0
    start = time.perf_counter()
    history_target = initial_history(device)

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted_increment = model.observation(times[:train_end])
        predicted_cumulative = torch.cumsum(predicted_increment, dim=0)
        data_loss = F.mse_loss(predicted_cumulative, cumulative[:train_end]) \
                    + F.mse_loss(predicted_increment, increment[:train_end])
        if getattr(model, "data_only", False):
            physics_loss = torch.zeros((), device=device)
            history_loss = torch.zeros((), device=device)
        else:
            residual = residual_matrix(model, collocation, config.finite_difference_step)
            if model_name == "TimePINN":
                physics_loss = causal_physics_loss(residual, config)
            else:
                physics_loss = residual.square().mean()
            history_loss = F.mse_loss(
                model(torch.zeros((1, 1), device=device))[:, :9], history_target[:, :9]
            )
        ramp = min(1.0, epoch / 120.0)
        loss = config.data_weight * data_loss \
               + config.physics_weight * ramp * physics_loss \
               + config.history_weight * history_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{event}/{model_name}/{seed}: nonfinite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()

        if epoch % config.check_every == 0:
            score = validation_score(
                model, times, increment, cumulative, train_end, validation_end
            )
            if score < best_score - 1.0e-8:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            if (epoch >= config.min_epochs
                    and checks_without_improvement >= config.patience_checks):
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
    model.load_state_dict(best_state)
    runtime = time.perf_counter() - start

    with torch.no_grad():
        predicted_norm_increment = model.observation(times).cpu().numpy()
        predicted_norm_cumulative = np.cumsum(predicted_norm_increment, axis=0)
        final_residual = float("nan") if getattr(model, "data_only", False) else residual_matrix(
            model, collocation, config.finite_difference_step
        ).square().mean().sqrt().item()
    raw_activity = frame[["rumor_activity", "debunk_activity"]].to_numpy(float)
    scales = np.maximum(raw_activity[:train_end].sum(axis=0), 1.0)
    predicted_activity = predicted_norm_increment * scales
    predicted_cumulative = predicted_norm_cumulative * scales
    actual_cumulative = np.cumsum(raw_activity, axis=0)

    rows: list[dict[str, object]] = []
    for channel_index, channel in enumerate(("rumor", "debunk")):
        test = slice(validation_end, n)
        metrics = regression_metrics(
            raw_activity[test, channel_index],
            predicted_activity[test, channel_index],
            raw_activity[:train_end, channel_index],
        )
        cumulative_metrics = regression_metrics(
            actual_cumulative[test, channel_index],
            predicted_cumulative[test, channel_index],
            actual_cumulative[:train_end, channel_index],
        )
        row: dict[str, object] = {
            "event": event,
            "fold": fold,
            "model": model_name,
            "seed": seed,
            "channel": channel,
            "train_bins": train_end,
            "validation_bins": validation_end - train_end,
            "test_bins": n - validation_end,
            "best_epoch": best_epoch,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "runtime_seconds": runtime,
            "physics_residual_rms": final_residual,
            **metrics,
            "cumulative_rmse": cumulative_metrics["rmse"],
            "cumulative_mae": cumulative_metrics["mae"],
            "cumulative_r2": cumulative_metrics["r2"],
        }
        row.update(model.parameter_values())
        rows.append(row)

    prediction_frame = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "split": np.where(
                np.arange(n) < train_end,
                "train",
                np.where(np.arange(n) < validation_end, "validation", "test"),
            ),
            "actual_rumor": raw_activity[:, 0],
            "actual_debunk": raw_activity[:, 1],
            "predicted_rumor": predicted_activity[:, 0],
            "predicted_debunk": predicted_activity[:, 1],
            "actual_cumulative_rumor": actual_cumulative[:, 0],
            "actual_cumulative_debunk": actual_cumulative[:, 1],
            "predicted_cumulative_rumor": predicted_cumulative[:, 0],
            "predicted_cumulative_debunk": predicted_cumulative[:, 1],
        }
    )
    run_info = {
        "epochs_completed": epoch,
        "event": event,
        "fold": fold,
        "model": model_name,
        "seed": seed,
        "fixed_tau_normalized": fixed_tau,
        "best_epoch": best_epoch,
        "validation_score": best_score,
        "runtime_seconds": runtime,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameters": model.parameter_values(),
    }
    return rows, prediction_frame, run_info


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["nrmse", "nmae", "mae", "wape", "r2", "cumulative_r2",
               "physics_residual_rms", "runtime_seconds"]
    grouped = runs.groupby(["event", "model", "channel"])[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    event_summary = grouped.reset_index()
    # Equal weighting: channels within fold, folds within event, then events.
    macro = (
        runs.groupby(["model", "seed", "event", "fold"])[
            ["nrmse", "nmae", "wape", "mae"]
        ].mean().reset_index()
        .groupby(["model", "seed", "event"])[
            ["nrmse", "nmae", "wape", "mae"]
        ].mean().reset_index()
        .groupby(["model", "seed"])[
            ["nrmse", "nmae", "wape", "mae"]
        ].mean().reset_index()
    )
    summary = macro.groupby("model")[["nrmse", "nmae", "wape", "mae"]].agg(["mean", "std"])
    summary.columns = [f"macro_{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("macro_nrmse_mean")
    return event_summary, summary


def paired_bootstrap(runs: pd.DataFrame, seeds: list[int], replicates: int = 5000) -> pd.DataFrame:
    pivot = runs.pivot_table(
        index=["event", "channel", "seed"], columns="model", values="nrmse"
    ).dropna().groupby(["event", "channel"]).mean()
    rng = np.random.default_rng(20260821)
    rows: list[dict[str, object]] = []
    trans = pivot["TransPINN"].to_numpy()
    for baseline in ("PINN", "LSTM-PINN", "TimePINN", "NoAttn-PINN"):
        if baseline not in pivot.columns:
            continue
        difference = pivot[baseline].to_numpy() - trans
        draws = np.empty(replicates)
        for index in range(replicates):
            sampled = rng.integers(0, len(difference), len(difference))
            draws[index] = difference[sampled].mean()
        rows.append(
            {
                "comparison": f"TransPINN vs {baseline}",
                "baseline_minus_transpinn_nrmse": float(difference.mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "relative_reduction_percent": float(
                    100.0 * difference.mean() / max(pivot[baseline].to_numpy().mean(), 1.0e-12)
                ),
                "wins": int(np.sum(difference > 0)),
                "ties": int(np.sum(np.isclose(difference, 0.0))),
                "losses": int(np.sum(difference < 0)),
                "paired_event_channel_units": int(len(difference)),
            }
        )
    return pd.DataFrame(rows)


def plot_forecasts(predictions: dict[tuple[str, str, int], pd.DataFrame],
                   representative_seed: int) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10.0, 10.0), constrained_layout=True)
    for row_index, event in enumerate(EVENTS):
        for channel_index, channel in enumerate(("rumor", "debunk")):
            axis = axes[row_index, channel_index]
            sample = predictions[(event, "TransPINN", representative_seed)]
            x = np.arange(len(sample))
            actual = sample[f"actual_{channel}"].to_numpy()
            axis.bar(x, actual, color="#c8c8c8", width=1.0, label="observed")
            for model in ("LSTM-PINN", "TimePINN", "TransPINN"):
                pred = predictions[(event, model, representative_seed)]
                axis.plot(x, pred[f"predicted_{channel}"], color=COLORS[model],
                          linewidth=1.2, label=model)
            validation_start = int((sample["split"] == "train").sum())
            test_start = validation_start + int((sample["split"] == "validation").sum())
            axis.axvline(validation_start - 0.5, color="#555555", linestyle="--", linewidth=0.8)
            axis.axvline(test_start - 0.5, color="#111111", linestyle=":", linewidth=1.0)
            axis.set_title(f"{event.title()} -- {channel}")
            axis.set_xlabel("time bin")
            axis.set_ylabel("activity")
            if row_index == 0 and channel_index == 0:
                axis.legend(fontsize=7, ncol=2)
    for extension in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"multievent_forecasts.{extension}", dpi=300)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("macro_nrmse_mean", ascending=True)
    fig, axis = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    axis.barh(
        ordered["model"], ordered["macro_nrmse_mean"],
        xerr=ordered["macro_nrmse_std"].fillna(0.0),
        color=[COLORS[name] for name in ordered["model"]], alpha=0.88,
        capsize=3,
    )
    axis.set_xlabel("Macro test NRMSE (mean +/- SD across seeds)")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.25)
    for extension in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"baseline_macro_nrmse.{extension}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[11, 22, 33],
    )
    parser.add_argument("--events", nargs="+", choices=EVENTS, default=list(EVENTS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = Config(epochs=args.epochs)
    if args.smoke:
        config = Config(epochs=min(args.epochs, 20), min_epochs=20, patience_checks=2,
                        collocation_points=24)

    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    device = torch.device("cpu")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    run_manifest: list[dict[str, object]] = []
    folds_by_event = {event: load_event_folds(event) for event in args.events}
    total = sum(len(folds_by_event[event]) for event in args.events) \
            * len(args.models) * len(args.seeds)
    run_index = 0
    for event in args.events:
        for fold, frame, split, _, _ in folds_by_event[event]:
            for model_name in args.models:
                for seed in args.seeds:
                    run_index += 1
                    print(
                        f"[{run_index:03d}/{total:03d}] {event} | fold={fold} | "
                        f"{model_name} | seed={seed}",
                        flush=True,
                    )
                    rows, prediction, run_info = train_one(
                        event, fold, frame, split, model_name, seed, config, device
                    )
                    all_rows.extend(rows)
                    run_manifest.append(run_info)
                    prediction.to_csv(
                        PREDICTION_DIR
                        / f"{event}_fold{fold}_{model_name.lower().replace('-', '_')}_seed{seed}.csv",
                        index=False,
                    )
                    # Checkpoint the tabular results after every run.
                    pd.DataFrame(all_rows).to_csv(TABLE_DIR / "baseline_runs.csv", index=False)

    runs = pd.DataFrame(all_rows)
    event_summary, summary = summarize(runs)
    event_summary.to_csv(TABLE_DIR / "baseline_event_summary.csv", index=False)
    summary.to_csv(TABLE_DIR / "baseline_summary.csv", index=False)
    (TABLE_DIR / "baseline_run_manifest.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "active_activity_fraction_per_channel": ACTIVE_ACTIVITY_FRACTION,
                "fold_rule": (
                    "L=max(channel-specific 99% cumulative-activity prefix); "
                    "h=max(2,floor(0.1L)); expanding origins start at ceil(0.5L) "
                    "and advance by h; retain first three candidates with positive "
                    "validation and test totals in both channels"
                ),
                "seeds": args.seeds,
                "runs": run_manifest,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    plot_summary(summary)
    print("\nMacro summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
