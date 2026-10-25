from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[2]
for _path in (_ROOT, _ROOT / "tools"):
    if str(_path) not in _sys.path:
        _sys.path.insert(0, str(_path))

from experiment_utils import training_output_folder
from run_io import metadata_for_save, metadata_from_disk, prepare_run_directory, validate_checkpoint_args, check_checkpoint_paths
from experiment_utils import FONT_SIZE, LEGEND_FONT_SIZE, configure_plot_style, configure_policy_output
"""
W&B online logging (recommended for web dashboard):

Run with environment variables set in your shell (NOT as Python code):

  export WANDB_API_KEY="your_key"
  export WANDB_MODE=online

Tip: keep keys out of source control; prefer exporting in tmux/session or using a secrets manager.
"""

"""Standard PPO procedure:

Phase 1 (rollout): simulate a full trajectory over k=0..N-1. Store states,
actions, old log probabilities, advantages and entropy in a buffer. Keep
policy parameters fixed during sampling.

Phase 2 (PPO update): reuse the buffer for ppo_epochs passes (default 3).
At each time step, recompute log probabilities under the current policy,
form ratios against the rollout probabilities, and optimize the clipped
surrogate objective."""

import numpy as np
import time
import torch
import psutil
import json
import re
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import math
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm, gaussian_kde

from torch.distributions import Normal, TransformedDistribution
from torch.distributions import transforms as dist_transforms
import argparse
import shutil
import json
import glob 
import sys
import copy
from experiment_utils import plot_trajectory_combined_abc, check_nan, to_scalar, _as_1d_numpy, plot_extendability_test
from martingale import corrected_td_residual
from tools_pinn0201_optimizer import plot as pinn_plot
from tools_pinn0201_optimizer import exact_values_calculation as exact_calc
from merton_policy_plots import (
    plot_panel_c_conditional_distribution,
    plot_panel_c_evolved_x_distribution,
    plot_gibbs_policy_by_pinn,
    plot_H_by_pinn,
)

RESIDUAL_VERSION = "martingale_subtracted_v3"

try:
    import wandb
    WANDB_AVAILABLE = True
    print("[wandb] wandb is available")
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not available. Install with: pip install wandb")


def _wandb_run_active(model):
    """Only log after this model has initialized a live W&B run."""
    return bool(
        WANDB_AVAILABLE
        and getattr(model, "_wandb_enabled", False)
        and getattr(wandb, "run", None) is not None
    )

"""
Training and checkpoint evaluation use the same three-panel flow-policy figure:
(a) conditional density p(u|t,x) at t=T/4, (b) value, (c) state trajectories.
Use --policy_plot_time_fraction 0.5 for t=T/2, or --panel_a_fixed_x 12
for a fixed conditioning state. The default state is mean X_t at that time.

Train: python3 scripts/Merton/merton_pinn_resnet_SAC_opt.py
Evaluate: python3 scripts/Merton/merton_pinn_resnet_SAC_opt.py --actor_path actor.pth --critic_path critic.pth
"""


class CosineAnnealingWarmUpLR(optim.lr_scheduler._LRScheduler):
    """Warmup + Cosine Annealing LR scheduler.

    - [0, warm_up_epochs): linear warmup from 0 to base_lr
    - [warm_up_epochs, total_epochs): cosine decay from base_lr to eta_min
    """
    def __init__(self, optimizer, warm_up_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warm_up_epochs = warm_up_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_up_epochs:
            if self.warm_up_epochs == 0:
                return list(self.base_lrs)
            return [base_lr * (self.last_epoch / self.warm_up_epochs)
                    for base_lr in self.base_lrs]
        else:
            remaining = self.total_epochs - self.warm_up_epochs
            if remaining <= 0:
                return [self.eta_min for _ in self.base_lrs]
            frac = (self.last_epoch - self.warm_up_epochs) / remaining
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * frac)) / 2
                    for base_lr in self.base_lrs]


class Block(nn.Module):
    def __init__(self, inputs: int, params: dict, activation="Tanh",
                 dropout_probability: float = 0.0):
        super(Block, self).__init__()
        self.L1 = nn.Linear(inputs, inputs)
        self.L2 = nn.Linear(inputs, inputs)
        self.activation = activation
        if activation in params:
            self.act = params[activation]
        self.dropout = nn.Dropout(p=dropout_probability)
        # Near-identity init: L1 zero, L2 (last layer of residual branch) uniform[-1e-3, 1e-3].
        # Uniform breaks symmetry while keeping Block(x) ≈ x at init.
        # nn.init.zeros_(self.L1.weight)
        # nn.init.zeros_(self.L1.bias)
        # nn.init.uniform_(self.L2.weight, -1e-3, 1e-3)
        # nn.init.uniform_(self.L2.bias, -1e-3, 1e-3)

        nn.init.kaiming_uniform_(self.L1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.L1.bias)
        nn.init.zeros_(self.L2.weight)   # zero last layer of residual branch
        nn.init.zeros_(self.L2.bias)

    def forward(self, x):
        if self.activation == "Sin" or self.activation == "sin":
            a = torch.sin(self.L2(self.dropout(torch.sin(self.L1(x))))) + x
        else:
            # L2 is the zero-initialized last layer, so the block starts as identity.
            a = self.L2(self.dropout(self.act(self.L1(x)))) + x
        return a

# ----------------------------
# Small utilities: activation
# ----------------------------
def _make_act(name: str):
    name = str(name).lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("sin",):
        # Sin is not a built-in Module; wrap it to avoid inconsistent behavior.
        class _Sin(nn.Module):
            def forward(self, x):
                return torch.sin(x)
        return _Sin()
    # fallback
    return nn.Tanh()

# -----------------------------------------
# [COMMENTED: Minimal MLP "Network"] ResNet-style: first -> Block*depth -> last
# - has .last attribute
# -----------------------------------------
class Network(nn.Module):
    # ---------- previously: MLP (commented) ----------
    # def __init__(self, inputs: int, width: int, depth: int, output: int,
    #              params: dict = None, activation="Tanh", penalty=None):
    #     super().__init__()
    #     act = _make_act(activation)
    #     layers = []
    #     if depth <= 0:
    #         self.body = nn.Identity()
    #         self.last = nn.Linear(inputs, output)
    #     else:
    #         layers.append(nn.Linear(inputs, width))
    #         layers.append(act)
    #         for _ in range(depth - 1):
    #             layers.append(nn.Linear(width, width))
    #             layers.append(act)
    #         self.body = nn.Sequential(*layers)
    #         self.last = nn.Linear(width, output)
    #     self.penalty = penalty
    # def forward(self, x):
    #     h = self.body(x)
    #     y = self.last(h)
    #     return y

    def __init__(self, inputs: int, width: int, depth: int, output: int,
                 params: dict = None, activation="Tanh", penalty=None,
                 dropout_probability: float = 0.0):
        super().__init__()
        params_act = dict(params) if params else {}
        if activation not in params_act:
            params_act[activation] = _make_act(activation)
        if depth <= 0:
            self.body = nn.Identity()
            self.last = nn.Linear(inputs, output)
        else:
            self.body = nn.Sequential(
                nn.Linear(inputs, width),
                _make_act(activation),
                *[Block(width, params_act, activation, dropout_probability=dropout_probability) for _ in range(depth)]
            )
            self.last = nn.Linear(width, output)
        self.penalty = penalty

    def forward(self, x):
        h = self.body(x)
        y = self.last(h)
        return y

# ------------------------------------------------------------
# [COMMENTED: Base Gaussian head MLP] ResNet-style hidden: first -> Block*depth
# ------------------------------------------------------------
class Network_mu_sigma_split(nn.Module):
    def __init__(self, inputs: int, width: int, depth: int, output: int,
                 params: dict = None, activation="Tanh", penalty=None,
                 dropout_probability: float = 0.0):
        super().__init__()
        self.output = int(output)
        self.activation = activation
        self.penalty = penalty
        self.bound = nn.Parameter(torch.tensor(1.0))

        # ---------- previously: MLP hidden (commented) ----------
        # act = _make_act(activation)
        # layers = [nn.Linear(inputs, width), act]
        # for _ in range(max(depth - 1, 0)):
        #     layers += [nn.Linear(width, width), act]
        # self.hidden = nn.Sequential(*layers)
        params_act = dict(params) if params else {}
        if activation not in params_act:
            params_act[activation] = _make_act(activation)
        if depth <= 0:
            self.hidden = nn.Sequential(nn.Linear(inputs, width))
        else:
            self.hidden = nn.Sequential(
                nn.Linear(inputs, width),
                _make_act(activation),
                *[Block(width, params_act, activation, dropout_probability=dropout_probability) for _ in range(depth)]
            )

        self.mean_layer = nn.Linear(width, output)
        self.log_std_layer = nn.Linear(width, output)

        # default clamps (you can override outside if needed)
        self.log_std_min = -10.0
        self.log_std_max = 2.0 # 0.0 #-1.0

    def forward(self, x):
        feat = self.hidden(x)                    # [B, width]
        mean = self.mean_layer(feat)             # [B, output]
        log_std = self.log_std_layer(feat)       # [B, output]

        if self.penalty is not None:
            if self.penalty == "Tanh":
                mean = torch.tanh(mean) * self.bound
            elif self.penalty == "ScaledTanh":
                mean = (torch.tanh(mean) + 1.0) * 0.5 * self.bound

        return mean, log_std, feat

# -------------------------------------------------------------------
# Bijective conditional monotone RQ-spline flow on R with identity tails
# - Works for D-dim actions independently (per-dim parameters)
# - Forward returns (u, logabsdet_per_dim)
# -------------------------------------------------------------------
class MonotoneSplineFlowNDLinearTails(nn.Module):
    """
    Conditional monotone rational-quadratic spline on R with identity tails.
    Inside [-B, B]: RQ spline.
    Outside: identity (y=x), so mapping is bijective on R.

    param_net outputs per-dim parameters: D * (3K + 1)
      widths:  K
      heights: K
      derivs:  K+1
    """
    def __init__(self, context_dim: int, output_dim: int, hidden: int = 128,
                 K: int = 8, B: float = 5.0, params: dict = None, activation="Tanh",
                 dropout_probability: float = 0.0):
        super().__init__()
        assert K >= 2
        self.K = int(K)
        self.B = float(B)
        self.output_dim = int(output_dim)

        self.param_net = Network(
            inputs=context_dim,
            width=hidden,
            depth=2,
            output=self.output_dim * (3 * self.K + 1),
            params=params or {},
            activation=activation,
            penalty=None,
            dropout_probability=dropout_probability,
        )

    def forward(self, u0: torch.Tensor, context: torch.Tensor):
        """
        u0:      [B, D]
        context: [B, C]
        returns:
          u:        [B, D]
          logabsdet:[B, D]
        """
        Bnd = self.B
        K = self.K
        D = self.output_dim
        assert u0.size(-1) == D, f"u0 last dim {u0.size(-1)} != output_dim {D}"

        # [B, D*(3K+1)] -> [B, D, 3K+1]
        p = self.param_net(context).view(-1, D, 3 * K + 1)

        widths_unnorm  = p[:, :, :K]            # [B,D,K]
        heights_unnorm = p[:, :, K:2*K]         # [B,D,K]
        derivs_unnorm  = p[:, :, 2*K:]          # [B,D,K+1]

        # Add min_bin_width and min_bin_height to prevent degenerate bins
        # This ensures each bin has at least minimum width/height, preventing numerical issues
        min_w = 1e-3
        min_h = 1e-3
        widths  = min_w + (2.0 * Bnd - K * min_w) * F.softmax(widths_unnorm, dim=-1)
        heights = min_h + (2.0 * Bnd - K * min_h) * F.softmax(heights_unnorm, dim=-1)
        # ---- derivatives (K+1) ----
        # NOTE: don't do in-place assignment like derivs[...,0]=1.0 because it hard-cuts
        # the gradient path w.r.t. derivs_unnorm at the endpoints.
        # Instead, construct by stitching: [1, softplus(mid)+eps, 1]
        # Enforce C1 "identity tails" at boundaries by pinning end-derivatives to 1.
        # This makes the inside-spline stitch smoothly to y=x outside [-B, B]
        # (prevents slope jumps and stabilizes logdet near boundaries).
        derivs_mid = F.softplus(derivs_unnorm[..., 1:-1]) + 1e-3  # [B,D,K-1]
        ones_l = torch.ones_like(derivs_unnorm[..., :1])          # [B,D,1]
        ones_r = torch.ones_like(derivs_unnorm[..., :1])          # [B,D,1]
        derivs = torch.cat([ones_l, derivs_mid, ones_r], dim=-1)  # [B,D,K+1]

        # knots in [-B,B]
        xk = torch.cumsum(widths,  dim=-1)                   # [B,D,K]
        yk = torch.cumsum(heights, dim=-1)
        xk = F.pad(xk, (1, 0), value=0.0) - Bnd              # [B,D,K+1]
        yk = F.pad(yk, (1, 0), value=0.0) - Bnd

        x = u0                                               # [B,D]
        left  = (x <= -Bnd)
        right = (x >=  Bnd)
        mid   = (~left) & (~right)

        # identity tails
        y = x.clone()
        logabsdet = torch.zeros_like(x)

        if mid.any():
            # safe x only for mid, to avoid touching tails
            x_mid = x.clamp(-Bnd + 1e-6, Bnd - 1e-6)

            # bin index per (B,D)
            idx = (x_mid.unsqueeze(-1) >= xk[:, :, 1:]).sum(dim=-1).clamp(0, K - 1)  # [B,D]

            def gather(v, j):
                return torch.gather(v, -1, j.unsqueeze(-1)).squeeze(-1)

            x0 = gather(xk, idx)
            x1 = gather(xk, idx + 1)
            y0 = gather(yk, idx)
            y1 = gather(yk, idx + 1)
            d0 = gather(derivs, idx)
            d1 = gather(derivs, idx + 1)

            w = (x1 - x0).clamp_min(1e-6)
            h = (y1 - y0).clamp_min(1e-6)
            s = h / w

            theta = ((x_mid - x0) / w).clamp(0.0, 1.0)
            omt = 1.0 - theta

            numerator = h * (s * theta.pow(2) + d0 * theta * omt)
            denom = s + (d0 + d1 - 2.0 * s) * theta * omt
            denom = denom + 1e-8

            y_mid = y0 + numerator / denom

            denom2 = denom.pow(2)
            num_der = s.pow(2) * (d1 * theta.pow(2) + 2.0 * s * theta * omt + d0 * omt.pow(2))
            dy_dx = (num_der / denom2).clamp_min(1e-12)

            y = torch.where(mid, y_mid, y)
            logabsdet = torch.where(mid, torch.log(dy_dx), logabsdet)

        return y, logabsdet

    def inverse(self, y: torch.Tensor, context: torch.Tensor):
        """Invert the conditional RQ spline and return (x, log|dy/dx|)."""
        Bnd, K, D = self.B, self.K, self.output_dim
        if y.size(-1) != D:
            raise ValueError(f"y last dim {y.size(-1)} != output_dim {D}")
        p = self.param_net(context).view(-1, D, 3 * K + 1)
        min_w, min_h = 1e-3, 1e-3
        widths = min_w + (2.0 * Bnd - K * min_w) * F.softmax(p[..., :K], dim=-1)
        heights = min_h + (2.0 * Bnd - K * min_h) * F.softmax(p[..., K:2*K], dim=-1)
        derivs_raw = p[..., 2*K:]
        derivs = torch.cat([
            torch.ones_like(derivs_raw[..., :1]),
            F.softplus(derivs_raw[..., 1:-1]) + 1e-3,
            torch.ones_like(derivs_raw[..., :1]),
        ], dim=-1)
        xk = F.pad(torch.cumsum(widths, dim=-1), (1, 0), value=0.0) - Bnd
        yk = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) - Bnd
        mid = (y > -Bnd) & (y < Bnd)
        y_safe = y.clamp(-Bnd + 1e-6, Bnd - 1e-6)
        idx = (y_safe.unsqueeze(-1) >= yk[..., 1:]).sum(dim=-1).clamp(0, K - 1)

        def gather(v, j):
            return torch.gather(v, -1, j.unsqueeze(-1)).squeeze(-1)

        x0, x1 = gather(xk, idx), gather(xk, idx + 1)
        y0, y1 = gather(yk, idx), gather(yk, idx + 1)
        d0, d1 = gather(derivs, idx), gather(derivs, idx + 1)
        w, h = x1 - x0, y1 - y0
        s = h / w
        y_delta = y_safe - y0
        common = d0 + d1 - 2.0 * s
        qa = y_delta * common + h * (s - d0)
        qb = h * d0 - y_delta * common
        qc = -s * y_delta
        discriminant = qb.square() - 4.0 * qa * qc
        if not torch.isfinite(discriminant).all() or (discriminant < -1e-6).any():
            raise FloatingPointError("non-finite/negative RQ-spline inverse discriminant")
        sqrt_disc = torch.sqrt(discriminant.clamp_min(0.0))
        root_denom = -qb - sqrt_disc
        root_denom = torch.where(root_denom.abs() < 1e-12, root_denom.sign() * 1e-12 - (root_denom == 0).to(root_denom) * 1e-12, root_denom)
        theta = (2.0 * qc / root_denom).clamp(0.0, 1.0)
        omt = 1.0 - theta
        x_mid = x0 + theta * w
        denom = s + common * theta * omt
        num_der = s.square() * (d1 * theta.square() + 2.0 * s * theta * omt + d0 * omt.square())
        dy_dx = num_der / denom.square()
        if not torch.isfinite(x_mid).all() or not torch.isfinite(dy_dx).all() or (dy_dx <= 0).any():
            raise FloatingPointError("invalid RQ-spline inverse or Jacobian")
        return (torch.where(mid, x_mid, y),
                torch.where(mid, torch.log(dy_dx), torch.zeros_like(y)))

# ------------------------------------------------------------
# Actor: base Gaussian -> spline flow -> optional squash to [0,bound]
# ------------------------------------------------------------
class TwoStageGaussianActor(nn.Module):
    def __init__(self, inputs: int, width: int, depth: int, output: int,
                 params: dict, activation="Tanh", bound_init=1.0,
                 post_width: int = None,
                 use_squash: bool = True, squash_mode: str = "scaled_tanh",
                 flow_K: int = 6, flow_B: float = 2.5,  # Spline works in [-B, B] = [-6.5, 6.5], covers action range [-2.0, 6.0]
                 flow_num_layers: int = 2,              # Number of stacked spline flow layers
                 # For 1D actions, conditioning the flow on an additional encoded state is often redundant
                 # because feat already comes from state. For higher-dim / more complex policies, turning
                 # this ON can help express richer conditional shapes.
                 use_state_conditioning: bool = False,
                 flow_init_scale: float = 1e-3,          # Step 2: shrink init -> near-identity
                 flow_step_scale_init: float = 0.1,      # Step 2 (legacy): constant residual mixing scale (kept for ckpt compatibility)
                 # ---- Flow mode ----
                 # "residual": legacy zK = z0 + s*(f(z0)-z0) with mix schedule
                 # "direct":   zK = f(z0) directly after warmup (no residual mixing)
                 flow_mode: str = "direct",
                 # ---- Flow mix schedule (only used when flow_mode="residual") ----
                 # If flow_mix_schedule="ramp": s(step)=0 for warmup, then linearly ramps to flow_mix_final.
                 # If flow_mix_schedule="constant": s = flow_step_scale (loaded from ckpt if provided).
                 flow_mix_schedule: str = "ramp",    # "constant" | "ramp"
                 flow_mix_warmup_steps: int = 100,      # default: freeze_flow_steps
                 flow_mix_ramp_steps: int = 100,         # ramp length (actor updates)
                 flow_mix_final: float = 1.0,           # default: flow_step_scale (legacy)
                 freeze_flow_steps: int = 100,           # Step 3: freeze *updates* of flow params for first N actor updates (forward still uses flow)
                 delay_squash_steps: int = 200,          # Step 3: warmup squash (via temperature schedule) - NOT USED for scaled_tanh
                 squash_temp_init: float = 3.0,          # Step 3: NOT USED for scaled_tanh without temperature
                 squash_temp_final: float = 1.0,         # Step 3: NOT USED for scaled_tanh without temperature
                 action_min: float = -2.0,               # NEW: minimum action value
                 action_max: float = 3.0,                # NEW: maximum action value
                 # zK_l2_coeff: float = 1e-4,            # Step 4: penalize large |zK| (unused)
                 sat_coeff: float = 0,                # Step 4: penalize saturation near boundaries
                 sat_margin: float = 0.02,               # Step 4: saturation band
                 # Break strict "context-independent identity init" slightly to make flow easier to deviate from identity
                 flow_init_last_weight_noise_std: float = 1e-4,
                 dropout_probability: float = 0.0):
        super().__init__()
        self.inputs = int(inputs)
        self.output = int(output)
        self.dropout_probability = float(dropout_probability)
        if self.dropout_probability != 0.0:
            raise ValueError(
                "Actor dropout must be 0: a random dropout mask makes fixed-action policy densities "
                "and PPO likelihood ratios ill-defined."
            )

        self.activation = activation
        # Fixed bound value (not trainable, registered as buffer)
        # Action range parameters
        self.action_min = float(action_min)
        self.action_max = float(action_max)
        self.action_range = self.action_max - self.action_min
        # bound now represents the range, not the maximum
        self.register_buffer('_bound_fixed', torch.tensor(float(self.action_range)))

        self.log_std_min = -8.0
        self.log_std_max =  0.0 #-1.0
        self.smooth_log_std_bounds = True

        # Step 3 schedule: squash can be delayed (training-time toggle)
        self.use_squash_default = bool(use_squash)
        self.squash_mode = str(squash_mode).lower()
        assert self.squash_mode in ("sigmoid", "scaled_tanh")

        self.use_state_conditioning = bool(use_state_conditioning)
        self.actor_type = "gaussian_two_stage"

        # ---- Step 3: training schedule state (updated externally) ----
        self.freeze_flow_steps = int(freeze_flow_steps)
        self.delay_squash_steps = int(delay_squash_steps)
        self._update_step = 0  # call set_update_step(step) from training loop

        # Flow mode: "direct" (no residual mixing) or "residual" (legacy)
        self.flow_mode = str(flow_mode).lower()
        assert self.flow_mode in ("direct", "residual"), f"flow_mode must be 'direct' or 'residual', got {self.flow_mode}"

        # Flow mix schedule (non-trainable, only used when flow_mode="residual")
        self.flow_mix_schedule = str(flow_mix_schedule).lower()
        assert self.flow_mix_schedule in ("constant", "ramp")
        self.flow_mix_warmup_steps = int(self.freeze_flow_steps if flow_mix_warmup_steps is None else flow_mix_warmup_steps)
        self.flow_mix_ramp_steps = int(flow_mix_ramp_steps)
        self.flow_mix_final = None if flow_mix_final is None else float(flow_mix_final)
        self.flow_init_last_weight_noise_std = float(flow_init_last_weight_noise_std)

        # Step 3: squash temperature schedule (keeps action valid while avoiding early saturation)
        self.squash_temp_init = float(squash_temp_init)
        self.squash_temp_final = float(squash_temp_final)

        # ---- Step 4: soft brakes hyperparams ----
        # NOTE: zK_l2_coeff is currently unused; keep commented to avoid ambiguity.
        # self.zK_l2_coeff = float(zK_l2_coeff)
        self.sat_coeff = float(sat_coeff)
        self.sat_margin = float(sat_margin)

        # base Gaussian (returns feat)
        self.base = Network_mu_sigma_split(
            inputs=inputs, width=width, depth=depth, output=output,
            params=params, activation=activation, penalty=None,
            dropout_probability=dropout_probability,
        )

        # flow context: feat (+ optional encoded state)
        post_width = post_width or width
        if self.use_state_conditioning:
            enc_dim = max(1, width // 2)
            self.state_encoder = Network(
                inputs=inputs, width=enc_dim, depth=1, output=enc_dim,
                params=params, activation=activation, penalty=None,
                dropout_probability=dropout_probability,
            )
            self._state_enc_dim = enc_dim
        else:
            self.state_encoder = None
            self._state_enc_dim = 0

        flow_context_dim = width + self._state_enc_dim
        self.flow_num_layers = int(flow_num_layers)

        # Stacked spline flow layers, each with its own param_net
        self.flow_layers = nn.ModuleList([
            MonotoneSplineFlowNDLinearTails(
                context_dim=flow_context_dim,
                output_dim=output,
                hidden=post_width,
                K=flow_K,
                B=flow_B,
                params=params,
                activation=activation,
                dropout_probability=dropout_probability,
            )
            for _ in range(self.flow_num_layers)
        ])
        # Backward compat: old checkpoints saved flow params under "flow.*".
        # We no longer register self.flow as a submodule to avoid duplicate parameter names.
        # Instead, _load_state_dict_compat() remaps old keys. See below.

        # Step 2: (a) near-identity init by shrinking flow params (architecture-agnostic)
        self._init_flow_near_identity(flow_init_scale)

        # Step 2 (legacy): keep flow_step_scale in state_dict for ckpt compatibility, but DO NOT train it.
        # New behavior should be controlled by flow_mix_schedule (recommended: "ramp").
        # zK := z0 + s*(flow(z0)-z0)
        # NOTE: explicitly a buffer (NOT a Parameter) so it can never be optimized by mistake.
        # Kept only for checkpoint compatibility and for "constant" schedule mode.
        self.register_buffer("flow_step_scale", torch.tensor(float(flow_step_scale_init)))

    # -------------------
    # schedule controls (Step 3)
    # -------------------
    def set_update_step(self, step: int):
        """Call this from your training loop every actor optimizer update."""
        self._update_step = int(step)

    @property
    def flow_enabled(self) -> bool:
        # Flow is always "enabled" in forward. Freezing is handled by the training loop
        # (by not updating flow parameters) to avoid train/eval mismatch and sudden graph changes.
        return True

    def _flow_mix_scale(self) -> float:
        """
        Non-trainable residual mixing scale s in zK = z0 + s * (z_flow - z0).
        - constant: s = flow_step_scale (loaded from ckpt if any)
        - ramp:     s = 0 for warmup, then linearly ramps to flow_mix_final
        """
        if self.flow_mix_schedule == "constant":
            return float(self.flow_step_scale.detach().cpu().item())

        # ramp schedule
        start = int(self.flow_mix_warmup_steps)
        ramp = int(self.flow_mix_ramp_steps)
        final = float(self.flow_step_scale.detach().cpu().item()) if (self.flow_mix_final is None) else float(self.flow_mix_final)

        step = int(self._update_step)
        if step < start:
            return 0.0
        if ramp <= 0:
            return final
        frac = min(max((step - start) / float(ramp), 0.0), 1.0)
        return float(frac * final)

    @property
    def squash_enabled(self) -> bool:
        # We keep squash always ON when use_squash_default=True, but modulate its temperature.
        return self.use_squash_default

    def _squash_temperature(self) -> torch.Tensor:
        """
        Step 3: temperature schedule:
          - for first delay_squash_steps: temp anneals from init -> final
          - after that: temp = final
        """
        if self.delay_squash_steps <= 0:
            t = self.squash_temp_final
        else:
            frac = min(max(self._update_step / float(self.delay_squash_steps), 0.0), 1.0)
            t = self.squash_temp_init + (self.squash_temp_final - self.squash_temp_init) * frac
        # make tensor on correct device/dtype at call site
        return torch.tensor(float(t))

    # -------------------
    # Step 2 helper: near-identity init for flow parameters
    # -------------------
    def _init_flow_near_identity(self, scale: float):
        """
        True identity init for all flow layers: Initialize each param_net's last layer
        to output values that make the spline transformation y=x initially.

        For derivs_mid = softplus(x) + 1e-3 to equal 1:
          softplus(x) = 1 - 1e-3 ≈ 0.999
          x = log(exp(0.999) - 1) ≈ 0.541

        Initialization per layer:
        - widths_unnorm bias = 0 (softmax → uniform)
        - heights_unnorm bias = 0 (softmax → uniform)
        - derivs_unnorm[..., 1:-1] bias = 0.541 (softplus(0.541) + 1e-3 ≈ 1)
        - derivs_unnorm[..., 0] bias = 0 (endpoints, will be set to 1)
        - derivs_unnorm[..., -1] bias = 0 (endpoints, will be set to 1)
        - Last layer weight = 0 (ensures initialization is independent of context)
        """
        with torch.no_grad():
            for flow_layer in self.flow_layers:
                self._init_single_flow_layer_near_identity(flow_layer, scale)

    def _init_single_flow_layer_near_identity(self, flow_layer, scale: float):
        """Initialize a single flow layer's param_net to identity."""
        param_net = flow_layer.param_net
        if not hasattr(param_net, 'last'):
            s = float(scale)
            if s > 0:
                for p in flow_layer.parameters():
                    if p is not None:
                        p.mul_(s)
            return

        D = flow_layer.output_dim
        K = flow_layer.K

        if param_net.last.weight is not None:
            param_net.last.weight.zero_()
            noise_std = float(getattr(self, "flow_init_last_weight_noise_std", 0.0) or 0.0)
            if noise_std > 0:
                param_net.last.weight.normal_(mean=0.0, std=noise_std)

        if param_net.last.bias is not None:
            bias = param_net.last.bias
            derivs_mid_init = math.log(math.exp(1.0 - 1e-3) - 1.0)  # ≈ 0.541

            for d in range(D):
                start_idx = d * (3 * K + 1)
                bias[start_idx:start_idx + K].zero_()
                bias[start_idx + K:start_idx + 2*K].zero_()
                bias[start_idx + 2*K] = 0.0
                bias[start_idx + 2*K + 1:start_idx + 3*K] = derivs_mid_init
                bias[start_idx + 3*K] = 0.0

    # -------------------
    # Backward-compat state_dict loading: remap old "flow.*" keys to "flow_layers.0.*"
    # -------------------
    def load_state_dict(self, state_dict, strict=True):
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("flow.") and not k.startswith("flow_layers."):
                new_k = k.replace("flow.", "flow_layers.0.", 1)
                remapped[new_k] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)

    # -------------
    # squash helpers
    # -------------
    @property
    def bound_pos(self):
        """
        Fixed bound parameter (always 1.0, not trainable).
        This ensures bound > 0, preventing NaN in log(bound) computations.
        """
        return self._bound_fixed

    def _squash(self, z):
        if not self.squash_enabled:
            return z
        # For scaled_tanh: map z to [action_min, action_max] using tanh
        # tanh maps R -> (-1, 1), so (tanh(z) + 1) / 2 maps to [0, 1]
        # Then scale to [action_min, action_max]
        if self.squash_mode == "scaled_tanh":
            normalized = (torch.tanh(z) + 1.0) * 0.5  # [0, 1]
            return normalized * self.action_range + self.action_min  # [action_min, action_max]
        # sigmoid (legacy, for backward compatibility)
        b = self.bound_pos
        temp = self._squash_temperature().to(device=z.device, dtype=z.dtype).clamp_min(1e-4)
        zt = z / temp
        normalized = torch.sigmoid(zt)  # [0, 1]
        return normalized * self.action_range + self.action_min  # [action_min, action_max]

    def _unsquash(self, a):
        if not self.squash_enabled:
            return a
        # Map from [action_min, action_max] back to [0, 1]
        eps = 1e-6
        normalized = (a - self.action_min) / (self.action_range + eps)
        normalized = torch.clamp(normalized, eps, 1.0 - eps)

        if self.squash_mode == "scaled_tanh":
            # (tanh(z)+1)/2 = normalized -> z = atanh(2*normalized-1)
            t = torch.clamp(2.0 * normalized - 1.0, -1.0 + eps, 1.0 - eps)
            z = 0.5 * (torch.log1p(t) - torch.log1p(-t))
            return z
        # sigmoid inverse: logit (legacy)
        z = torch.log(normalized) - torch.log1p(-normalized)
        # IMPORTANT: sigmoid mode uses temperature in _squash(): sigmoid(z/temp).
        # So inverse must multiply the logit by the same temp to stay consistent.
        temp = self._squash_temperature().to(device=a.device, dtype=a.dtype).clamp_min(1e-4)
        return z * temp

    def _log_det_squash(self, zK):
        """
        Returns [B,1] log|det(d a / d zK)|
        Uses numerically stable formulas to avoid NaN/Inf issues.
        
        For scaled_tanh: a = action_range * 0.5 * (tanh(z) + 1) + action_min
        For sigmoid: a = action_range * sigmoid(z/temp) + action_min
        """
        if not self.squash_enabled:
            return torch.zeros(zK.size(0), 1, device=zK.device, dtype=zK.dtype)

        if self.squash_mode == "scaled_tanh":
            # a = action_range * 0.5 * (tanh(z) + 1) + action_min
            # da/dz = action_range * 0.5 * (1 - tanh(z)^2) = action_range * 0.5 * sech²(z)
            # log(da/dz) = log(action_range * 0.5) + log(sech²(z))
            # log(sech²(z)) = log(1 - tanh²(z)) = -2*log(cosh(z))

            # Stable computation using logaddexp:
            # log(cosh(z)) = log((exp(z) + exp(-z))/2) = logaddexp(z, -z) - log(2)
            logcosh = torch.logaddexp(zK, -zK) - math.log(2.0)  # log(cosh(zK))
            # da/dz = (action_range * 0.5) * sech^2(zK)
            # Convert action_range to tensor with same device/dtype as zK
            action_range_tensor = torch.tensor(self.action_range, device=zK.device, dtype=zK.dtype)
            log_det = torch.log(action_range_tensor * 0.5) - 2.0 * logcosh
        else:
            # sigmoid mode (legacy, with temperature)
            temp = self._squash_temperature().to(device=zK.device, dtype=zK.dtype).clamp_min(1e-4)
            zt = zK / temp
            log_temp = torch.log(temp)
            # a = action_range * sigmoid(z/temp) + action_min
            # da/dz = action_range * sigmoid(z/temp)*(1-sigmoid(z/temp)) * (1/temp)
            # log(da/dz) = log(action_range) + log(σ(z)(1-σ(z))) - log(temp)
            # log(σ(z)(1-σ(z))) = -softplus(-z) - softplus(z)
            # Convert action_range to tensor with same device/dtype as zK
            action_range_tensor = torch.tensor(self.action_range, device=zK.device, dtype=zK.dtype)
            log_det = torch.log(action_range_tensor) - F.softplus(-zt) - F.softplus(zt) - log_temp

        # Sum over action dimensions
        return log_det.sum(dim=-1, keepdim=True)

    # -------------------
    # base forward (mu,std)
    # -------------------
    def forward(self, state):
        mu, raw_log_std, feat = self.base(state)
        if not (math.isfinite(self.log_std_min) and math.isfinite(self.log_std_max)
                and self.log_std_min < self.log_std_max):
            raise RuntimeError(
                f"smooth log_std requires finite ordered bounds, got "
                f"[{self.log_std_min}, {self.log_std_max}]"
            )
        # A shifted sigmoid keeps the range smooth while mapping raw=0 to
        # log_std=-1 (rather than the [-8, 0] midpoint, which would initialize
        # the exploration standard deviation far too small).
        initial_log_std = -1.0
        initial_fraction = ((initial_log_std - self.log_std_min) /
                            (self.log_std_max - self.log_std_min))
        logit_offset = math.log(initial_fraction / (1.0 - initial_fraction))
        fraction = torch.sigmoid(raw_log_std + logit_offset)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * fraction
        return mu, log_std, feat

    # -------------------
    # flow forward
    # -------------------
    def _make_flow_context(self, feat, state):
        # Flow is conditioned on:
        #   - feat: learned features from the base Gaussian network (already a function of state)
        #   - optionally, a separate state encoder state_enc(state)
        # When enabled, context = concat([feat, state_enc(state)]).
        if self.use_state_conditioning:
            state_enc = self.state_encoder(state)
            return torch.cat([feat, state_enc], dim=-1)
        return feat

    def flow_forward(self, z0, feat, state):
        """
        z0:   [B,D]
        feat: [B,width]
        state:[B,inputs]
        returns zK [B,D], log_det_flow [B,1]
        """
        if self.flow_mode == "direct":
            return self._flow_forward_direct(z0, feat, state)
        else:
            return self._flow_forward_residual(z0, feat, state)

    def flow_inverse(self, zK, feat, state):
        """Return z0 and the forward log-Jacobian for a fixed post-flow value."""
        if self._update_step < self.freeze_flow_steps:
            return zK, torch.zeros(zK.size(0), 1, device=zK.device, dtype=zK.dtype)
        if self.flow_mode != "direct":
            raise NotImplementedError(
                "fixed-action log_prob requires flow_mode='direct'; residual mixing has no analytic inverse"
            )
        ctx = self._make_flow_context(feat, state)
        z = zK
        total_logabsdet = torch.zeros_like(zK)
        for layer in reversed(self.flow_layers):
            z, logabsdet = layer.inverse(z, ctx)
            total_logabsdet = total_logabsdet + logabsdet
        return z, total_logabsdet.sum(dim=-1, keepdim=True)

    def _flow_forward_direct(self, z0, feat, state):
        """
        Direct flow: zK = flow_L(...flow_1(z0)) after warmup, zK = z0 during warmup.
        No residual mixing — identity init + freeze_flow_steps ensures stability.
        Chains all layers, accumulating log-det.
        """
        # During warmup: bypass flow entirely (flow params are frozen anyway)
        if self._update_step < self.freeze_flow_steps:
            zK = z0
            log_det_flow = torch.zeros(z0.size(0), 1, device=z0.device, dtype=z0.dtype)
            return zK, log_det_flow

        ctx = self._make_flow_context(feat, state)
        z = z0
        total_logabsdet = torch.zeros(z0.size(0), z0.size(1), device=z0.device, dtype=z0.dtype)
        for layer in self.flow_layers:
            z, logabsdet = layer(z, ctx)  # [B,D], [B,D]
            total_logabsdet = total_logabsdet + logabsdet

        zK = z
        log_det_flow = total_logabsdet.sum(dim=-1, keepdim=True)  # [B,1]
        return zK, log_det_flow

    def _flow_forward_residual(self, z0, feat, state):
        """
        Legacy residual mixing: zK = z0 + s * (flow_chain(z0) - z0).
        Chains all flow layers first, then applies residual mixing on the composite.
        """
        # Residual mixing scale (schedule)
        s_val = float(self._flow_mix_scale())
        s = torch.tensor(s_val, device=z0.device, dtype=z0.dtype).clamp(0.0, 1.0)
        if s_val <= 0.0:
            zK = z0
            log_det_flow = torch.zeros(z0.size(0), 1, device=z0.device, dtype=z0.dtype)
            return zK, log_det_flow

        ctx = self._make_flow_context(feat, state)

        # Chain all flow layers: z_flow = f_L(...f_1(z0))
        z = z0
        total_logabsdet = torch.zeros(z0.size(0), z0.size(1), device=z0.device, dtype=z0.dtype)
        for layer in self.flow_layers:
            z, logabsdet = layer(z, ctx)  # [B,D], [B,D]
            total_logabsdet = total_logabsdet + logabsdet
        z_flow = z

        # Residual mixing
        zK = z0 + s * (z_flow - z0)

        # logdet for residual mixing on the composite:
        # zK = (1-s) z0 + s f(z0), where f = f_L ∘ ... ∘ f_1
        # J_composite is diagonal (each layer is dim-wise monotone spline)
        # ∂f/∂z0 = prod of per-layer derivatives = exp(sum of logabsdet)
        dy_dx = torch.exp(total_logabsdet)  # [B,D], positive
        mix_diag = (1.0 - s) + s * dy_dx
        mix_diag = torch.clamp(mix_diag, min=1e-12)
        log_det_flow = torch.log(mix_diag).sum(dim=-1, keepdim=True)  # [B,1]
        return zK, log_det_flow

    # # -------------------
    # # Step 4: soft brakes (regularizers you add into actor loss)
    # # -------------------
    # def regularization_terms(self, zK: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    #     """
    #     Return a scalar regularizer tensor to ADD to actor loss.
    #     """
    #     reg = torch.zeros((), device=zK.device, dtype=zK.dtype)

    #     # (b) penalize saturation near action_min or action_max (prevents sticking to boundaries)
    #     if self.sat_coeff > 0 and self.squash_enabled:
    #         # Normalize action to [0, 1] for saturation check
    #         normalized = (action - self.action_min) / (self.action_range + 1e-8)
    #         normalized = torch.clamp(normalized, 0.0, 1.0)
    #         m = float(self.sat_margin)
    #         low = F.relu(m - normalized)  # Near action_min
    #         high = F.relu(normalized - (1.0 - m))  # Near action_max
    #         sat_pen = (low + high).mean()
    #         reg = reg + self.sat_coeff * sat_pen

    #     return reg

    # -------------------
    # sampling
    # -------------------
    def sample(self, state, deterministic=False, noise=None):
        """
        Returns:
          action:       [B,D]
          z0:           [B,D]  (Gaussian pre-flow)
          zK:           [B,D]  (post-flow, pre-squash)
          log_det_flow: [B,1]
          eps:          [B,D]  (Gaussian noise, None if deterministic)
          mu, log_std, feat for convenience
        """
        mu, log_std, feat = self.forward(state)

        if deterministic:
            z0 = mu
            zK, log_det_flow = self.flow_forward(z0, feat, state)
            action = self._squash(zK)
            return action, z0, zK, log_det_flow, None, mu, log_std, feat

        if noise is None:
            eps = torch.randn_like(mu)
        else:
            eps = noise
        z0 = mu + torch.exp(log_std) * eps
        zK, log_det_flow = self.flow_forward(z0, feat, state)
        action = self._squash(zK)
        return action, z0, zK, log_det_flow, eps, mu, log_std, feat

    # -------------------
    # log_prob (TRAINING PATH)
    # -------------------
    def log_prob_from_cache(self, mu, log_std, z0, zK, log_det_flow):
        """
        Stable training log_prob:
          log p(a) = log p(z0) - log|det d(flow)/dz0| - log|det d(squash)/dzK|
        IMPORTANT: Use z0,zK,log_det_flow returned by sample () (no inverse).
        Returns: [B,1]
        """
        values = {"mu": mu, "log_std": log_std, "z0": z0, "zK": zK,
                  "log_det_flow": log_det_flow}
        bad = [name for name, value in values.items() if not torch.isfinite(value).all()]
        if bad:
            raise FloatingPointError(f"non-finite actor log-prob inputs: {', '.join(bad)}")
        std = torch.exp(log_std)

        # Compute each component separately and check for NaN
        log_p_z0 = Normal(mu, std).log_prob(z0).sum(dim=-1, keepdim=True)  # [B,1]
        log_det_squash = self._log_det_squash(zK)                          # [B,1]

        result = log_p_z0 - log_det_flow - log_det_squash
        if not torch.isfinite(result).all():
            raise FloatingPointError("non-finite actor log probability")
        return result

    def log_prob(self, state, action):
        """Log density of a fixed action, for score-function and PPO updates."""
        mu, log_std, feat = self.forward(state)
        zK = self._unsquash(action)
        z0, log_det_flow = self.flow_inverse(zK, feat, state)
        return self.log_prob_from_cache(mu, log_std, z0, zK, log_det_flow)

    # Optional: for diagnostics only (not recommended for training)
    def action_to_zK(self, action):
        return self._unsquash(action)

    def pack_params(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """
        Pack distribution params into a single tensor for convenience.

        Args:
            mean:    [..., D]
            log_std: [..., D]

        Returns:
            packed: [..., 2D] where packed[..., :D]=mean and packed[..., D:]=log_std
        """
        if mean.shape != log_std.shape:
            raise ValueError(f"mean and log_std must have same shape, got {mean.shape} vs {log_std.shape}")
        return torch.cat([mean, log_std], dim=-1)

    def unpack_params(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of pack_params.

        Args:
            packed: [..., 2D]

        Returns:
            mean:    [..., D]
            log_std: [..., D]
        """
        if packed.size(-1) != 2 * self.output:
            raise ValueError(f"packed last dim must be 2*output={2*self.output}, got {packed.size(-1)}")
        mean, log_std = torch.split(packed, self.output, dim=-1)
        return mean, log_std

    @torch.no_grad()
    def deterministic_proxy_mean(self, state):
        """
        Cheap proxy: a_hat(s) = squash(flow(mu(s))).
        NOTE: This is NOT the true mean E[a|s] unless transform is linear-ish
              and variance is tiny. Use mixture_mean() for the true mean.
        """
        mu, log_std, feat = self.forward(state)
        zK, _log_det_flow = self.flow_forward(mu, feat, state)
        a = self._squash(zK)
        return a

    def mixture_mean(
        self,
        state,
        n_samples: int = 256,
        antithetic: bool = True,
        chunk: int = None,
        return_std: bool = False,
        seed: int = None,
    ):
        """
        Estimate the true mean of the final action distribution after flow + squash:

            E[a|s] where a = squash(flow(z0)), z0 ~ N(mu(s), std(s)^2)

        Args:
            state:      [B, inputs]
            n_samples:  number of MC samples
            antithetic: use eps and -eps pairs to reduce variance
            chunk:      optional, process samples in chunks to save memory
            return_std: if True, also return MC std of actions (diagnostic)

        Returns:
            mean_a: [B, D]
            (optional) std_a: [B, D]

        Reproducibility:
            If seed is not None, eps ~ N(0,1) used for Monte-Carlo is drawn under a
            locally-forked RNG context (CPU + the relevant CUDA device), so the global
            RNG state outside this call is preserved. This makes mixture_mean() results
            independent of any random draws that happened earlier in the program.
        """
        mu, log_std, feat = self.forward(state)         # mu/log_std: [B,D], feat: [B,width]
        std = torch.exp(log_std)

        B, D = mu.shape
        device, dtype = mu.device, mu.dtype

        def _make_eps():
            if antithetic:
                half = (n_samples + 1) // 2
                e = torch.randn(half, B, D, device=device, dtype=dtype)
                e = torch.cat([e, -e], dim=0)[:n_samples]
                return e
            return torch.randn(n_samples, B, D, device=device, dtype=dtype)

        if seed is not None:
            seed_int = int(seed)
            cuda_devices = []
            if device.type == "cuda":
                idx = device.index
                if idx is None:
                    idx = torch.cuda.current_device()
                cuda_devices = [int(idx)]
            # fork_rng saves/restores RNG state so this call does NOT advance global RNG.
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                torch.manual_seed(seed_int)
                if device.type == "cuda":
                    # Seed all CUDA RNGs to avoid "current_device mismatch" pitfalls.
                    torch.cuda.manual_seed_all(seed_int)
                eps = _make_eps()
        else:
            eps = _make_eps()

        # optional chunking over samples to reduce peak memory
        if chunk is None or chunk >= n_samples:
            z0 = mu.unsqueeze(0) + std.unsqueeze(0) * eps          # [S,B,D]
            # flatten to [S*B, D]
            z0_flat = z0.reshape(-1, D)
            feat_flat = feat.unsqueeze(0).expand(n_samples, B, feat.size(-1)).reshape(-1, feat.size(-1))
            state_flat = state.unsqueeze(0).expand(n_samples, B, state.size(-1)).reshape(-1, state.size(-1))

            zK_flat, _ = self.flow_forward(z0_flat, feat_flat, state_flat)   # [S*B,D]
            a_flat = self._squash(zK_flat)                                   # [S*B,D]
            a = a_flat.view(n_samples, B, D)                                 # [S,B,D]

            mean_a = a.mean(dim=0)                                           # [B,D]
            if return_std:
                std_a = a.std(dim=0, unbiased=False)                         # [B,D]
                return mean_a, std_a
            return mean_a

        # chunked version
        sums = torch.zeros(B, D, device=device, dtype=dtype)
        if return_std:
            sums2 = torch.zeros(B, D, device=device, dtype=dtype)

        start = 0
        while start < n_samples:
            end = min(start + chunk, n_samples)
            S = end - start
            eps_c = eps[start:end]                                           # [S,B,D]
            z0 = mu.unsqueeze(0) + std.unsqueeze(0) * eps_c                  # [S,B,D]
            z0_flat = z0.reshape(-1, D)

            feat_flat = feat.unsqueeze(0).expand(S, B, feat.size(-1)).reshape(-1, feat.size(-1))
            state_flat = state.unsqueeze(0).expand(S, B, state.size(-1)).reshape(-1, state.size(-1))

            zK_flat, _ = self.flow_forward(z0_flat, feat_flat, state_flat)
            a_flat = self._squash(zK_flat)
            a = a_flat.view(S, B, D)

            sums += a.sum(dim=0)
            if return_std:
                sums2 += (a * a).sum(dim=0)
            start = end

        mean_a = sums / float(n_samples)
        if return_std:
            var = sums2 / float(n_samples) - mean_a * mean_a
            var = torch.clamp(var, min=0.0)
            std_a = torch.sqrt(var)
            return mean_a, std_a
        return mean_a


class FBSNN(nn.Module): # Forward-Backward Stochastic Neural Network
    def __init__(self, net_value, net_control, params, actor_path: str = None, critic_path: str = None):
        super().__init__()
        self.params_equ = params["equation"]
        self.params_train = params["train"]
        configure_policy_output(net_control, self.params_train.get("entropy_alpha", 0.0))
        # ------------------------------
        # Outer warmup master switch (avoid duplicated knobs)
        # ------------------------------
        # If warmup_steps is provided, it overrides all three related schedules so they are consistent:
        #   warmup_actor_steps = freeze_flow_steps = delay_squash_steps = warmup_steps
        # This prevents "brain split" between outer warmup and actor-internal schedules.
        warmup_steps = self.params_train.get("warmup_steps", None)
        if warmup_steps is not None:
            w = int(warmup_steps)
            self.params_train["warmup_actor_steps"] = w
            self.params_train["freeze_flow_steps"] = w
            self.params_train["delay_squash_steps"] = w

        self.device = self.params_train["device"]
        self.Xi = self.params_equ["Xi"].to(self.device).float()    # initial point, ensure float32
        self.T = self.params_equ["T"]              # terminal time
        self.M = self.params_equ["M"]             # number of trajectories
        self.N = self.params_equ["N"]              # number of time snapshots
        self.D = self.params_equ["D"]              # number of dimensions
        self.p = self.params_equ["p"]
        self.net_critic = net_value.to(self.device)
        self.net_actor = net_control.to(self.device)

        # ---- Target Critic (Polyak-averaged copy, not trained by optimizer) ----
        self.net_critic_target = copy.deepcopy(self.net_critic)
        for p in self.net_critic_target.parameters():
            p.requires_grad = False
        self.target_tau = float(self.params_train.get("target_tau", 0.005))

        self.optimizer_critic = optim.Adam(self.net_critic.parameters(), lr=self.params_train["lr_critic"])

        # Use single optimizer with parameter groups for actor (base and flow)
        # This ensures consistent zero_grad() and step() calls
        lr_base = self.params_train["lr_actor"]
        # Default: keep flow lr <= base for stability (spline + squash is sensitive)
        lr_flow = self.params_train.get("lr_flow", lr_base)

        # Collect parameters for different groups
        base_params = []
        flow_params = []
        state_encoder_params = []

        for name, param in self.net_actor.named_parameters():
            # Exclude non-trainable parameters (e.g. legacy flow_step_scale kept only for ckpt compatibility)
            if not getattr(param, "requires_grad", True):
                continue
            if name.startswith("flow_layers."):
                flow_params.append(param)
            elif name.startswith("state_encoder."):
                state_encoder_params.append(param)
            else:
                base_params.append(param)

        # Create parameter groups
        param_groups = [
            {"params": base_params, "lr": lr_base, "group_name": "actor_base"},
        ]
        if state_encoder_params:
            param_groups.append({"params": state_encoder_params, "lr": lr_base, "group_name": "actor_state_encoder"})
        if flow_params:
            param_groups.append({"params": flow_params, "lr": lr_flow, "group_name": "actor_flow"})

        self.optimizer_actor = optim.Adam(param_groups, 
                                          betas=(0.9, 0.999),
                                          weight_decay=self.params_train["weight_decay"])

        # Keep backward compatibility aliases (but they point to the same optimizer)
        self.optimizer_actor_base = self.optimizer_actor
        self.optimizer_flow = self.optimizer_actor
        # ---- LR Scheduler: Cosine Annealing with Warmup ----
        total_iters = int(self.params_train.get("iteration", 2000))
        lr_eta_min = float(self.params_train.get("lr_eta_min", 1e-6))
        freeze_flow_steps = int(self.params_train.get("freeze_flow_steps", 30))
        critic_warmup_steps = int(self.params_train.get("critic_warmup_steps", freeze_flow_steps))

        # Actor: warmup during freeze_flow_steps, then cosine decay
        self.scheduler_actor = CosineAnnealingWarmUpLR(
            self.optimizer_actor,
            warm_up_epochs=freeze_flow_steps,
            total_epochs=total_iters,
            eta_min=lr_eta_min,
        )
        # Critic: warmup for critic_warmup_steps, then cosine decay
        self.scheduler_critic = CosineAnnealingWarmUpLR(
            self.optimizer_critic,
            warm_up_epochs=critic_warmup_steps,
            total_epochs=total_iters,
            eta_min=lr_eta_min,
        )
        # NOTE: Do not create a separate flow scheduler pointing to the same optimizer.
        # Stepping two schedulers would decay lrs twice. Use scheduler_actor only.
        self.scheduler_flow = None
        self.best_critic_loss, self.best_actor_loss = 10000, 10000
        self.actor_path = actor_path
        self.critic_path = critic_path
        self.Y_error_list, self.control_error_list, self.bound_list = [], [], []
        self.CriticLoss_list, self.ActorLoss_list, self.J_list, self.control_list = [], [], [], []
        self.KLLoss_list, self.mu_of_policy_error_list = [], []
        self.dt = self.T / self.N

        # One-time log for reproducibility
        try:
            ws = self.params_train.get("warmup_steps", None)
            print(
                "[warmup-config] "
                f"warmup_strategy={self.params_train.get('warmup_strategy','affine_flow')} "
                f"warmup_steps={ws} "
                f"(warmup_actor_steps={self.params_train.get('warmup_actor_steps', None)}, "
                f"freeze_flow_steps={self.params_train.get('freeze_flow_steps', None)}, "
                f"delay_squash_steps={self.params_train.get('delay_squash_steps', None)})"
            )
        except Exception:
            pass

        mu     = torch.as_tensor(self.params_equ["mu"],     device=self.device)
        r      = torch.as_tensor(self.params_equ["r"],      device=self.device)
        sigma  = torch.as_tensor(self.params_equ["sigma"],  device=self.device)
        lam    = torch.as_tensor(self.params_equ["lambda"], device=self.device)
        z      = torch.as_tensor(self.params_equ["z"],      device=self.device)
        beta   = torch.as_tensor(self.params_equ["beta"],   device=self.device)
        gamma_entropy = torch.as_tensor(self.params_train["entropy_alpha"], device=self.device)
        B = sigma**2 + lam * z**2
        self.h_t = 1 / (beta - self.p * r - (self.p * (mu-r)**2) / (2*B*(1-self.p)))
        # print("self.h_t =", self.h_t) # tensor(1.0128, device='cuda:0')
        self.a_t = gamma_entropy * self.p / (2*beta)
        # print("self.a_t =", self.a_t) # tensor(0.0125, device='cuda:0')
        self.g_t = (1/beta) * ( r * self.a_t + gamma_entropy/2 * (torch.log(2*torch.pi*gamma_entropy) - torch.log(B) - torch.log((1-self.p)*self.h_t)) )
        # print("self.g_t =", self.g_t) # tensor(0.0323, device='cuda:0')

        self.eval_every = self.params_train["eval_every"]  # run evaluation every XX steps training iterations
        self._eval_fixed_ready = False  # whether we have built the fixed eval batch
        # Fixed (t,x) point set for Top-K evaluation (functional comparison on same states)
        self._eval_points_ready = False
        self.eval_tx = None  # Tensor [B, 1 + D] = (t, x)
        self._eval_points_last_refresh_it = None  # last iteration when eval_tx was (re)built
        self._eval_iteration = None  # set by training loop before calling eval
        self.plot_traj_mode = self.params_train.get("plot_traj_mode", "ab")

        # Cache for exact solution computation (to avoid recomputing Picard iteration)
        self._exact_solution_cache = None
        self._exact_solution_params_hash = None
        # Cache for PINN value function solver
        self._exact_solution_cache_nn = None
        self._exact_solution_params_hash_nn = None

        # X0 initialization type: "point", "uniform", or "gaussian"
        self.X0_init_type = self.params_train.get("X0_init_type", "point")
        self.X0_init_radius = self.params_train.get("X0_init_radius", 0.1)  # radius for uniform/gaussian distribution

        # Top-3 eval ckpts: list of dicts like {"iter": int, "score": float, "actor": path, "critic": path}
        self.topk_eval = []
        # Top-3 by PINN/Taylor composite errors (value+control)
        # self.topk_pinn = []
        # self.topk_taylor = []

        # [UPDATED 2026-02-10] ckpt selection:
        # Previously we saved both PINN and Taylor Top-3 using a single weighted score
        # (default ~ average of value/control errors).

        # Now we ONLY maintain PINN Top-3 ckpts under THREE different weightings to
        # balance "value err goes down then mildly rises" vs "control keeps decreasing":
        #   A) score_avg      = (control + value) / 2
        #   B) score_u1_v02  = control + value / 5
        #   C) score_u02_v1  = control / 5 + value

        # Each score maintains its own Top-3 list and writes its own alias files via tag:
        #   - tag="pinn"               -> A)
        #   - tag="pinn_u1_v0p2"      -> B)
        #   - tag="pinn_u0p2_v1"      -> C)
        self.topk_pinn = []              # A) (u+v)/2   (keeps historical tag="pinn")
        self.topk_pinn_u1_v0p2 = []     # B) u + v/5
        self.topk_pinn_u0p2_v1 = []     # C) u/5 + v
        # Keep attribute for backward-compat / debugging (no longer updated by default)
        self.topk_taylor = []

        def _absdir_or_none(p):
            return os.path.dirname(os.path.abspath(p)) if (p is not None) else None

        if self.actor_path is not None and self.critic_path is not None:
            out_from_actor = _absdir_or_none(self.actor_path)
            out_from_critic = _absdir_or_none(self.critic_path)

            if out_from_actor:
                self.output_folder = out_from_actor
            elif out_from_critic:
                self.output_folder = out_from_critic
        else:
            self.output_folder = self._default_output_folder()

        check_checkpoint_paths(self.actor_path, self.critic_path, multiplayer=False)
        if self.actor_path is None and self.critic_path is None:
            prepare_run_directory(self.output_folder, overwrite=self.params_train.get("_overwrite", False))
        print("output_folder================", self.output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        if self.actor_path and os.path.isfile(self.actor_path):
            try:
                self.net_actor.load_state_dict(torch.load(self.actor_path, map_location=self.device))
                self.net_actor.eval()
                print(f"[load] actor from given path: {self.actor_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load actor checkpoint: {self.actor_path}") from e

        if self.critic_path and os.path.isfile(self.critic_path):
            try:
                self.net_critic.load_state_dict(torch.load(self.critic_path, map_location=self.device))
                self.net_critic.eval()
                # Sync target critic with loaded online critic
                self.net_critic_target.load_state_dict(self.net_critic.state_dict())
                print(f"[load] critic from given path: {self.critic_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load critic checkpoint: {self.critic_path}") from e

        if (self.actor_path and os.path.isfile(self.actor_path)) or \
           (self.critic_path and os.path.isfile(self.critic_path)):
            loaded_params = self._load_params(self.output_folder)
            if loaded_params is not None:
                self._update_params_from_dict(loaded_params)
                try:
                    self._sync_actor_action_bounds_from_train_params(loaded_params.get("train", {}))
                except Exception:
                    pass
                # Backward-compatible: if old params.json doesn't have the new flow-mix schedule keys,
                # keep actor behavior in "constant" mode using the loaded flow_step_scale tensor.
                try:
                    self._sync_actor_flow_schedule_from_train_params(loaded_params.get("train", {}))
                except Exception:
                    pass

    # ------------------------------
    # State preprocessing (recommended for Merton: x>0 with large dynamic range)
    # ------------------------------
    def _preprocess_state(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Build the input features for actor/critic.

        Default: t/T and log(x) (optionally scaled), which is substantially more stable than raw (t, x)
        when x spans orders of magnitude.

        Shapes supported:
        - t: [B,1], x: [B,1]  -> out: [B,2]
        - t: [B,N,1], x: [B,N,1] -> out: [B,N,2]
        """
        mode = self.params_train.get("state_preprocess", "log")
        # allow booleans / None
        if mode is None or mode is False:
            mode = "raw"
        mode = str(mode).lower()

        eps = float(self.params_train.get("state_preprocess_eps", 1e-8))
        T = float(self.T) if float(self.T) != 0.0 else 1.0

        # normalize time to [0,1]
        t_n = t / (T + eps)

        if mode in ("raw", "none", "identity"):
            x_feat = x
        elif mode in ("log", "logx"):
            # x should be >0 for Merton; clamp only for numerical safety
            x_safe = torch.clamp(x, min=eps)
            x_feat = torch.log(x_safe)
            # optional scaling: divide by log(x_scale) so feature magnitudes stay O(1)
            x_scale = self.params_train.get("state_log_x_scale", None)
            if x_scale is not None:
                denom = math.log(float(x_scale) + eps)
                if abs(denom) > 0:
                    x_feat = x_feat / float(denom)
        elif mode in ("div", "scale"):
            x_ref = float(self.params_train.get("state_x_ref", 50.0))
            x_feat = x / (x_ref + eps)
        else:
            raise ValueError(f"Unknown state_preprocess mode: {mode}")

        return torch.cat([t_n, x_feat], dim=-1)

    # =========================
    # FLOW FREEZE (TRAINING-SIDE)
    # =========================
    def _sync_actor_action_bounds_from_train_params(self, train_params: dict):
        """
        Keep net_actor's action_min/action_max consistent with params.json.
        This is critical for eval-only runs, because actor is constructed before
        params.json is loaded, and action bounds are not in state_dict.
        """
        a = getattr(self, "net_actor", None)
        if a is None:
            return
        tp = train_params if isinstance(train_params, dict) else {}
        if ("action_min" not in tp) and ("action_max" not in tp):
            return
        try:
            amin = float(tp.get("action_min", getattr(a, "action_min", float("nan"))))
            amax = float(tp.get("action_max", getattr(a, "action_max", float("nan"))))
            if not (math.isfinite(amin) and math.isfinite(amax) and amax > amin):
                return
            prev_min = float(getattr(a, "action_min", amin))
            prev_max = float(getattr(a, "action_max", amax))
            if (abs(prev_min - amin) > 1e-12) or (abs(prev_max - amax) > 1e-12):
                a.action_min = amin
                a.action_max = amax
                a.action_range = amax - amin
                print(
                    f"[sync] actor action bounds updated from params.json: "
                    f"[{prev_min:.6g},{prev_max:.6g}] -> [{amin:.6g},{amax:.6g}]"
                )
        except Exception as e:
            print(f"[sync] WARNING: failed to update actor action bounds: {e}")

    def _sync_actor_flow_schedule_from_train_params(self, train_params: dict):
        """
        Keep net_actor's flow-mix schedule consistent with params.json.
        If params.json doesn't define the new keys (older runs), default to constant mode.
        """
        a = getattr(self, "net_actor", None)
        if a is None or not hasattr(a, "flow_mix_schedule"):
            return

        tp = train_params if isinstance(train_params, dict) else {}
        if "flow_mix_schedule" not in tp:
            # Older runs: preserve legacy behavior
            a.flow_mix_schedule = "constant"
            a.flow_mix_final = None
            return

        # Sync flow_mode (default "residual" for backward compat with old checkpoints)
        if hasattr(a, "flow_mode"):
            a.flow_mode = str(tp.get("flow_mode", getattr(a, "flow_mode", "residual"))).lower()

        a.flow_mix_schedule = str(tp.get("flow_mix_schedule", a.flow_mix_schedule)).lower()
        if hasattr(a, "flow_mix_warmup_steps") and ("flow_mix_warmup_steps" in tp):
            a.flow_mix_warmup_steps = int(tp.get("flow_mix_warmup_steps"))
        if hasattr(a, "flow_mix_ramp_steps") and ("flow_mix_ramp_steps" in tp):
            a.flow_mix_ramp_steps = int(tp.get("flow_mix_ramp_steps"))
        if hasattr(a, "flow_mix_final"):
            v = tp.get("flow_mix_final", None)
            a.flow_mix_final = None if v is None else float(v)
        if hasattr(a, "flow_init_last_weight_noise_std") and ("flow_init_last_weight_noise_std" in tp):
            a.flow_init_last_weight_noise_std = float(tp.get("flow_init_last_weight_noise_std"))

    def _apply_flow_param_freeze(self):
        """
        Freeze/unfreeze flow parameters by toggling requires_grad.
        This keeps the *forward path* consistent (flow always enabled) while avoiding updates early on.
        """
        freeze_steps = int(self.params_train.get("freeze_flow_steps", 0) or 0)
        curr = int(getattr(self, "_global_actor_update_step", 0) or 0)
        trainable = (curr >= freeze_steps)
        if getattr(self, "_flow_params_trainable", None) == trainable:
            return
        self._flow_params_trainable = trainable
        for name, p in self.net_actor.named_parameters():
            if name.startswith("flow_layers."):
                if getattr(p, "requires_grad", False) != trainable and p is not None:
                    p.requires_grad_(trainable)

    # =========================
    # PARAMS SAVE/LOAD HELPERS
    # =========================

    def _save_params(self, save_dir: str):
        from tools_pinn0201_optimizer import params_load_helpers as pio
        return pio.save_params(self, save_dir)

    def _save_run_script(self, save_dir: str):
        from tools_pinn0201_optimizer import params_load_helpers as pio
        return pio.save_run_script(__file__, save_dir)

    def _load_params(self, load_dir: str):
        """
        Load params from a specified directory and convert serialized entries back to torch.tensor.
        Return the loaded params dict, or None on failure.
        """
        from tools_pinn0201_optimizer import params_load_helpers as pio
        return pio.load_params(load_dir)

    def _update_params_from_dict(self, params_dict: dict):
        from tools_pinn0201_optimizer import params_load_helpers as pio
        return pio.update_params_from_dict(self, params_dict)

    # =========================
    # EVALUATION HELPERS
    # =========================

    def _ensure_eval_set(self, N_eval=None):
        """
        [ADDED] Build a fixed evaluation batch once for all future evaluations.
        Using a fixed RNG seed ensures comparability across iterations.
        Also generates a fixed X0_eval for consistent comparison between X_pred and X_test.
        
        Args:
            N_eval: Number of time intervals for evaluation. If None, uses self.N or increases if N <= 200.
        """
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.ensure_eval_set(self, N_eval=N_eval)

    def _init_X0(self, M=None):
        """
        Initialize X0 based on X0_init_type.
        
        Args:
            M: Number of trajectories. If None, uses self.M
        
        Returns:
            X0: Tensor of shape (M, D) with initial values
        """
        if M is None:
            M = self.M

        # Ensure Xi is in shape (1, D) for consistent processing
        Xi_reshaped = self.Xi.view(1, -1).float()  # (1, D), ensure float32

        if self.X0_init_type == "point":
            # Original single point initialization
            X0 = Xi_reshaped.expand(M, -1).clone()
        elif self.X0_init_type == "uniform":
            # Uniform distribution: [Xi - radius, Xi + radius]
            # Generate uniform random values in [-radius, radius]
            uniform_noise = torch.rand(M, self.D, device=self.device, dtype=torch.float32) * 2 * self.X0_init_radius - self.X0_init_radius
            X0 = Xi_reshaped.expand(M, -1) + uniform_noise
        elif self.X0_init_type == "gaussian":
            # Gaussian distribution: N(Xi, radius^2)
            # Generate Gaussian noise with std = radius
            gaussian_noise = torch.randn(M, self.D, device=self.device, dtype=torch.float32) * self.X0_init_radius
            X0 = Xi_reshaped.expand(M, -1) + gaussian_noise
        else:
            raise ValueError(f"Unknown X0_init_type: {self.X0_init_type}. Must be 'point', 'uniform', or 'gaussian'")

        return X0

    @torch.no_grad()
    def _evaluate_on_fixed(self):
        """
        [ADDED] Run one forward-only evaluation on the fixed batch and return a scalar score.
        Score definition: Y_error by default; if control_error exists, use average(Y_error, control_error).
        Lower is better.
        """
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.evaluate_on_fixed(self)

    def _relative_l2_error(self, pred, exact, eps: float = 1e-12):
        """
        Compute relative L2 error over all entries: sqrt(E[(pred-exact)^2] / (E[exact^2] + eps)).
        """
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.relative_l2_error(pred, exact, eps=eps)

    @torch.no_grad()
    def _evaluate_pinn_taylor_errors(self):
        """
        Evaluate PINN/Taylor value & control errors on a fixed set of (t,x) points.
        This is a "homogeneous yardstick": the (t,x) points are identical across iterations,
        so Top-K selection does NOT depend on changing closed-loop trajectories.
        Returns a dict with keys: pinn_value_error, pinn_control_error,
        taylor_value_error, taylor_control_error.
        
        IMPORTANT: Uses MC mean of policy (mixture_mean) for control comparison,
        not sampled actions, to get accurate mean policy error for checkpoint selection.
        """
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.evaluate_pinn_taylor_errors(self)

    @torch.no_grad()
    def _evaluate_on_fixed_grid(self, itr: int, x_min: float = 0.0, x_max: float = 20.0, n_x: int = 200):
        """
        Evaluate policy mean error at t=0 over a uniform x grid.
        This diagnostic provides numerical comparisons on fixed states.
        """
        device = self.device

        # Create the fixed x grid for numerical evaluation
        x_grid = torch.linspace(x_min, x_max, n_x, device=device, dtype=torch.float32).view(-1, 1)
        x_safe = torch.clamp(x_grid, min=1e-8)
        t_grid = torch.zeros(n_x, 1, device=device, dtype=torch.float32)

        self.net_actor.eval()
        with torch.no_grad():
            # Compute network prediction (MC mean)
            actor_type = getattr(self.net_actor, "actor_type", "")
            if actor_type in ("hyper_gaussian_split", "gaussian_two_stage"):
                state_input = self._preprocess_state(t_grid, x_safe)
                K_mean_eval = int(self.params_train["K_mean_eval"])
                mean_mixture_pred = self.net_actor.mixture_mean(
                    state_input, n_samples=K_mean_eval, antithetic=True
                )
            else:
                mu_pred, _, _ = self.net_actor(self._preprocess_state(t_grid, x_safe))
                mean_mixture_pred = mu_pred

            pi_approx = mean_mixture_pred.squeeze()
            if pi_approx.dim() > 1:
                pi_approx = pi_approx.squeeze()

            # Compute exact (PINN)
            u_mean_pinn, _, _, _pol = self.compute_exact_entropy_with_cache(x_safe)
            if u_mean_pinn.dim() > 1:
                pi_pinn = u_mean_pinn.squeeze()
            else:
                pi_pinn = u_mean_pinn
            if pi_pinn.dim() > 1:
                pi_pinn = pi_pinn.squeeze()

            # Compute relative L2 error
            error_approx_vs_pinn = torch.sqrt(
                torch.mean((pi_approx - pi_pinn) ** 2) / (torch.mean(pi_pinn ** 2) + 1e-12)
            )

            # Compute mean difference
            mean_diff = (pi_approx.mean() - pi_pinn.mean()).item()

            print(f"[fixed_grid_eval] it={itr}, x in [{x_min:.1f}, {x_max:.1f}], n_x={n_x}:")
            print(f"  relative L2 error (approx vs PINN) = {error_approx_vs_pinn.item():.6f} ({error_approx_vs_pinn.item()*100:.3f}%)")
            print(f"  mean difference (approx - PINN) = {mean_diff:.6f}")
            print(f"  PINN mean = {pi_pinn.mean().item():.4f}, approx mean = {pi_approx.mean().item():.4f}")

            # Log to wandb if available
            if _wandb_run_active(self):
                wandb.log({
                    # Clarify units:
                    # - error: percent (%)
                    # - mean_diff: action units (raw)
                    # Keep legacy keys, but add explicit *_pct / *_raw keys to avoid confusion in W&B charts.
                    "fixed_grid_error": error_approx_vs_pinn.item() * 100.0,  # legacy (pct)
                    "fixed_grid_error_pct": error_approx_vs_pinn.item() * 100.0,
                    "fixed_grid_error_raw": error_approx_vs_pinn.item(),
                    "fixed_grid_mean_diff": mean_diff,  # legacy (raw units)
                    "fixed_grid_mean_diff_raw": mean_diff,
                }, step=itr)

    def _append_pinn_taylor_errors(self, it: int, elapsed: float, errors: dict):
        """
        Append PINN/Taylor errors to CSV in output folder.
        """
        eval_log_path = os.path.join(self.output_folder, "pinn_taylor_errors_every20.csv")
        if not os.path.isfile(eval_log_path):
            with open(eval_log_path, "w", encoding="utf-8") as f:
                f.write("iter,elapsed,pinn_value_error,pinn_control_error,taylor_value_error,taylor_control_error\n")
        with open(eval_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{int(it)},{to_scalar(elapsed):.6f},"
                f"{errors['pinn_value_error']},{errors['pinn_control_error']},"
                f"{errors['taylor_value_error']},{errors['taylor_control_error']}\n"
            )

    def _append_pinn_taylor_ranges(self, it: int, elapsed: float, errors: dict):
        """
        Append PINN/Taylor x/u ranges to CSV in output folder.
        """
        range_log_path = os.path.join(self.output_folder, "pinn_taylor_ranges_every20.csv")
        if not os.path.isfile(range_log_path):
            with open(range_log_path, "w", encoding="utf-8") as f:
                f.write("iter,elapsed,x_min,x_max,u_pinn_min,u_pinn_max,u_taylor_min,u_taylor_max\n")
        with open(range_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{int(it)},{to_scalar(elapsed):.6f},"
                f"{errors['x_min']},{errors['x_max']},"
                f"{errors['u_pinn_min']},{errors['u_pinn_max']},"
                f"{errors['u_taylor_min']},{errors['u_taylor_max']}\n"
            )

    @staticmethod
    def _to_float(x):
        """
        [ADDED] Safe float conversion for Tensors/ndarrays/scalars.
        """
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.to_float(x)

    def _eval_ckpt_unique_paths(self, it: int, tag: str = "eval"):
        """
        [ADDED] Build unique file names for the current iteration's eval ckpt.
        We save actor/critic separately, allowing later aliasing to top1/2/3.
        """
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.eval_ckpt_unique_paths(self, it=it, tag=tag)

    def _persist_topN_alias(self, topk_list, tag: str = "eval"):
        """
        Refresh stable aliases actor_{tag}_top1/2/3.pth and critic_{tag}_top1/2/3.pth.
        Remove any old alias files, then write exactly top1/top2/top3.
        Refresh alias files with iteration numbers in their filenames.
        based on current topk_list (sorted by ascending score).
        Also dump a small JSON meta file for bookkeeping.
        """
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.persist_topN_alias(self, topk_list=topk_list, tag=tag)

    def _maybe_update_top3_eval(self, it: int, score: float, tag: str = "eval", topk_list=None):
        """
        [ADDED] If the new score is good enough, save current weights as an eval ckpt
        and maintain a Top-3 list (evict the worst if buffer is full).
        """
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.maybe_update_top3_eval(self, it=it, score=score, tag=tag, topk_list=topk_list)

    def _save_periodic_ckpt(self, it: int, every: int = 50):
        """
        Save actor/critic checkpoints every `every` steps into output_folder/every_{every}_ckpts.
        """
        if every is None or every <= 0:
            return
        step = int(it) + 1
        if step % every != 0:
            return

        ckpt_dir = os.path.join(self.output_folder, f"every_{int(every)}_ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        a_path = os.path.join(ckpt_dir, f"actor_it{step}.pth")
        c_path = os.path.join(ckpt_dir, f"critic_it{step}.pth")
        try:
            torch.save(self.net_actor.state_dict(), a_path)
            torch.save(self.net_critic.state_dict(), c_path)
            # Save params/run-script once for later evaluation
            if not os.path.isfile(os.path.join(ckpt_dir, "params.json")):
                self._save_params(ckpt_dir)
            if not os.path.isfile(os.path.join(ckpt_dir, os.path.basename(__file__))):
                self._save_run_script(ckpt_dir)
            # Save training state (optimizer/scheduler/global_step) for exact resume if available later.
            try:
                from tools_pinn0201_optimizer import resume_tools as _rt
                _rt.save_training_state(self, ckpt_dir=ckpt_dir, it=int(it))
            except Exception as _e:
                print(f"[warn] periodic training-state save failed at it={step}: {_e}")
            print(f"[save] periodic ckpt @it={step} -> {ckpt_dir}")
        except Exception as e:
            print(f"[warn] periodic ckpt save failed at it={step}: {e}")

    def _load_best_eval_for_plot(self, tag: str = "pinn"):
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.load_best_eval_for_plot(self, tag=tag)

    ###############
    ## Problem Functions
    ##############

    def reward_func(self, x, dt):
        if self.p == 1/2:
            x = torch.clamp(x, min=1e-8)
        X = x**self.p / self.p
        return X * dt

    def compute_exact_exploratory_taylor_and_ansatz(self, x):
        return exact_calc.compute_exact_exploratory_taylor_and_ansatz(self, x)

    # === PINN helpers (active) ===
    def _get_pinn_cache_paths(self):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.get_pinn_cache_paths(self)

    def _save_pinn_to_disk(self, net, pinn_params, params_hash, cache):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.save_pinn_to_disk(self, net, pinn_params, params_hash, cache)

    def _save_pinn_dense_grid(self, net, cache, pinn_params):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.save_pinn_dense_grid(self, net, cache, pinn_params)

    def _load_pinn_from_disk(self, pinn_params, params_hash, device, dtype, width, depth):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.load_pinn_from_disk(self, pinn_params, params_hash, device, dtype, width, depth)

    def _load_pinn_dense_grid(self):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.load_pinn_dense_grid(self)

    def _interpolate_v_from_dense_grid(self, x_eval, dense_grid_cache, device, dtype):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.interpolate_v_from_dense_grid(x_eval, dense_grid_cache, device, dtype)

    def _value_net_forward(self, net, x, x_min, x_max):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.value_net_forward(net, x, x_min, x_max)

    def _value_net_forward_chunked(self, net, x, x_min, x_max, chunk=None, use_no_grad=True):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.value_net_forward_chunked(self, net, x, x_min, x_max, chunk=chunk, use_no_grad=use_no_grad)

    def _value_net_with_derivatives(self, net, x, x_min, x_max):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.value_net_with_derivatives(net, x, x_min, x_max)

    # Unused: wrapper for pinn_calculator.solve_value_function_pinn (no callers in this script).
    def _solve_value_function_pinn(
            self, mu, r, sigma, lam, z, beta, gamma, device, dtype,
            x_min=1e-5, x_max=80.0, n_x=500, n_u=100,
            u_min=-1.0, u_max=10.0,
            x_eval_max=None,
            n_iter=3000, lr=5e-4, width=256, depth=5,
            use_cache=True,
        ):
        from tools_pinn0201_optimizer import pinn_calculator as pc
        return pc.solve_value_function_pinn(
            self, mu, r, sigma, lam, z, beta, gamma, device, dtype,
            x_min=x_min, x_max=x_max, x_eval_max=x_eval_max, n_x=n_x, n_u=n_u,
            u_min=u_min, u_max=u_max,
            n_iter=n_iter, lr=lr, width=width, depth=depth,
            use_cache=use_cache,
        )

    def compute_exact_entropy_with_cache(
        self,
        x,
        return_cache: bool = False,
        return_u_hard: bool = False,
        return_policy: bool = False,
    ):
        # Use params_train so PINN u grid matches action/pinn_u_min/max (avoids half-Gaussian when range is wrong)
        _pt = self.params_train
        pinn_cfg_defaults = {
            "x_min": _pt.get("pinn_x_min", 1e-5),
            "x_max": _pt.get("pinn_x_max", 80.0),
            "n_x": _pt.get("pinn_n_x", 200),
            "n_u": _pt.get("pinn_n_u", 150),
            "u_min": _pt.get("pinn_u_min", -1.0),
            "u_max": _pt.get("pinn_u_max", 2.0),
        }
        return exact_calc.compute_exact_entropy_with_cache(
            self,
            x,
            return_cache=return_cache,
            return_u_hard=return_u_hard,
            return_policy=return_policy,
            pinn_cfg_defaults=pinn_cfg_defaults,
        )

    DiscretePolicy = exact_calc.DiscretePolicy

    def compute_exact_entropy(self, x):
        _pt = self.params_train
        pinn_cfg_defaults = {
            "x_min": _pt.get("pinn_x_min", 1e-5),
            "x_max": _pt.get("pinn_x_max", 80.0),
            "n_x": _pt.get("pinn_n_x", 200),
            "n_u": _pt.get("pinn_n_u", 150),
            "u_min": _pt.get("pinn_u_min", -1.0),
            "u_max": _pt.get("pinn_u_max", 2.0),
        }
        policy, value = exact_calc.compute_exact_entropy(
            self,
            x,
            pinn_cfg_defaults=pinn_cfg_defaults,
        )
        # If discrete policy is explicitly requested, do not silently fall back to Normal.
        force_discrete = bool(self.params_train.get("predict_sample_discrete_policy", True))
        if force_discrete and self.params_train.get("entropy_alpha", 0.0) > 0:
            if not isinstance(policy, self.DiscretePolicy):
                raise RuntimeError(
                    "Discrete policy requested (predict_sample_discrete_policy=True), "
                    "but no discrete policy was built. Check pinn_u grid settings or policy construction."
                )
        return policy, value

    def compute_exact_no_exploratory(self, x):
        return exact_calc.compute_exact_no_exploratory(self, x)

    def mu_torch(self, t, X, control=None):       
        return ((self.params_equ["mu"]-self.params_equ["r"]) * control + self.params_equ["r"]) * X

    def sigma_torch(self, t, X, control=None):
        return torch.diag_embed(control * X) * self.params_equ["sigma"]

    def xi(self, t, X, control=None):
        return control * X * self.params_equ["z"]

    def net_u_Du(self, t, X, compute_Du=True):
        # Ensure inputs are float32 to match network dtype
        t = t.float() if t.dtype != torch.float32 else t
        X = X.float() if X.dtype != torch.float32 else X
        inputs = self._preprocess_state(t, X)
        u = self.net_critic(inputs) # Q-function
        Du = torch.autograd.grad(torch.sum(u), X, retain_graph=True, create_graph=True)[0] if compute_Du else 0
        return u, Du

    @torch.no_grad()
    def _polyak_update_target_critic(self):
        """Soft-update target critic: θ_target ← (1-τ)θ_target + τ θ_online"""
        tau = self.target_tau
        for p, p_target in zip(self.net_critic.parameters(), self.net_critic_target.parameters()):
            p_target.data.mul_(1.0 - tau).add_(tau * p.data)

    def Dg_torch(self, X):
        return torch.autograd.grad(torch.sum(self.g_torch(X)), X, retain_graph=True)[0]  # M x D

    def get_memory(self, type):
        process = psutil.Process()
        mem_info = process.memory_info()
        if type == "MB":
            print("Memory of physical: %.1f MB, virtual: %.1f MB, allocated: %.1f MB" % (
                mem_info.rss / 1024 / 1024, mem_info.vms / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))
        if type == "GB":
            print("Memory of physical: %.3f GB, virtual: %.3f GB, allocated: %.3f GB" % (
                mem_info.rss / 1024 / 1024 / 1024, mem_info.vms / 1024 / 1024 / 1024,
                torch.cuda.memory_allocated() / 1024 / 1024 / 1024))

    def fetch_minibatch(self):
        T, M, N, D = self.T, self.M, self.N, self.D
        dt = T / N
        t_grid = torch.linspace(0, T, N + 1, dtype=torch.float32).unsqueeze(0).repeat(M, 1).unsqueeze(-1).to(self.device)  # [M, N+1, 1]
        dW = torch.randn(M, N, D, device=self.device, dtype=torch.float32) * torch.tensor(np.sqrt(dt), dtype=torch.float32, device=self.device)  # [M, N, D]
        W = torch.cat([torch.zeros(M, 1, D, device=self.device, dtype=torch.float32), torch.cumsum(dW, dim=1)], dim=1)  # [M, N+1, D]

        # Generate jump times (same for both uniform and non-uniform)
        max_jump = int(3 * self.params_equ["lambda"] * T) + 1 
        if max_jump > 0:
            jump_times = torch.cumsum(torch.distributions.Exponential(self.params_equ["lambda"]).sample((M, max_jump)), dim=1)  # [M, max_jump]
            jump_times[jump_times > T] = 0  
            jump_times, _ = torch.sort(jump_times, dim=1)  
            jump_times = jump_times.unsqueeze(-1).to(self.device).float()  # [M, max_jump, 1]
            # jump_times is already a tensor on device; avoid torch.tensor(tensor) copy-construction warning.
            self.jump_index = torch.unique(torch.nonzero(jump_times)[:, 0])
            # print("jump_index", self.jump_index)
        else:
            jump_times = 0

        return t_grid, W, jump_times

    def kl_divergence_gaussian(self, mu_pred, std_pred, mu_exact, std_exact):
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.kl_divergence_gaussian(mu_pred, std_pred, mu_exact, std_exact)

    def wasserstein_distance_gaussian(self, mu_pred, std_pred, mu_exact, std_exact):
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.wasserstein_distance_gaussian(mu_pred, std_pred, mu_exact, std_exact)

    def cvm_error_from_samples(self, control, policy_exact):
        """Compute scalar CVM error (smaller is better).
        control: [M,1], [B,M,1], [M] or [B,M].
        policy_exact: Normal(mu_u, sigma_u), matching or broadcastable to control."""
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.cvm_error_from_samples(control, policy_exact)

    def _policy_cdf(self, policy_exact, u):
        """
        Robust CDF for:
        - Normal
        - TransformedDistribution with Softplus + Affine
        - DiscretePolicy
        """
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.policy_cdf(policy_exact, u)

    ##############
    ## Train
    ##############
    def train_loop(self, itr, traj_only=False, _control_exact_=False, predict=False, deterministic_eval=False):
        if predict and not traj_only:
            raise ValueError("Evaluation batches must use traj_only=True")
        if _control_exact_ and not traj_only:
            raise ValueError("Reference-policy rollouts are evaluation-only")

        lr_critic_now = None
        try:
            lr_critic_now = float(self.optimizer_critic.param_groups[0].get("lr", None))
        except Exception:
            lr_critic_now = None

        diag_log_std_max = -float("inf")
        diag_std_max = -float("inf")
        diag_abs_zK_max = -float("inf")
        diag_log_det_squash_min = float("inf")
        diag_seen = False

        def _diag_update_from_actor(log_sigma_tensor: torch.Tensor, zK_tensor: torch.Tensor):
            """Update per-itr diagnostic stats from actor outputs (no grad)."""
            nonlocal diag_log_std_max, diag_std_max, diag_abs_zK_max, diag_log_det_squash_min, diag_seen
            if (log_sigma_tensor is None) or (zK_tensor is None):
                return
            try:
                with torch.no_grad():
                    ls = log_sigma_tensor.detach()
                    zk = zK_tensor.detach()
                    if ls.numel() > 0:
                        diag_log_std_max = max(diag_log_std_max, float(ls.max().item()))
                        diag_std_max = max(diag_std_max, float(torch.exp(ls).max().item()))
                    if zk.numel() > 0:
                        diag_abs_zK_max = max(diag_abs_zK_max, float(zk.abs().max().item()))
                        if hasattr(self.net_actor, "_log_det_squash"):
                            lds = self.net_actor._log_det_squash(zk)  # [B,1]
                            if lds is not None and torch.is_tensor(lds) and lds.numel() > 0:
                                diag_log_det_squash_min = min(diag_log_det_squash_min, float(lds.min().item()))
                diag_seen = True
            except Exception:
                # Diagnostics must never break training
                return

        if not predict:
            t, W, P = self.fetch_minibatch()
        else:
            t, W, P = self.t_predict_set, self.W_predict_set, self.P_predict_set
        dt = self.dt 
        beta = self.params_equ["beta"]
        gamma_entropy = self.params_train["entropy_alpha"] 
        M = self.M

        # ------------------------------------------------------------
        # Stability/perf: PINN "exact" solves are VERY expensive.
        # Reference diagnostics belong to forward-only evaluation, never optimizer updates.
        # ------------------------------------------------------------
        compute_exact_this_itr = bool(traj_only or predict)

        X_buffer, Y_buffer, A_buffer, logpi_buffer, action_buffer, mu_logsigma_buffer = [], [], [], [], [], []
        logpi_tsallis_buffer = []
        # Keep these as tensors so warmup (actor_update_count==0) won't break .device usage later
        total_loss_critic_sum = torch.tensor(0.0, device=self.device)
        total_loss_actor_sum = torch.tensor(0.0, device=self.device)

        K_actor  = self.params_train.get("K_actor", 20)
        K_critic = self.params_train.get("K_critic", 5)
        actor_loss_accum = 0.0
        actor_chunk_count = 0
        critic_loss_accum = 0.0
        critic_chunk_count = 0
        critic_update_count = 0  
        actor_update_count = 0   
        # PPO-Clip Phase 2 buffers: collect rollout data, then update actor after full trajectory
        ppo_state_buf, ppo_action_buf, ppo_logprob_old = [], [], []
        ppo_advantage_buf, ppo_entropy_buf = [], []
        mu_of_policy_error_accum = torch.tensor(0.0, device=self.device)
        mu_of_policy_error_count = 0
        self.optimizer_actor.zero_grad(set_to_none=True)
        self.optimizer_critic.zero_grad(set_to_none=True)
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        # Use fixed X0_eval when predict=True to ensure X_pred and X_test use same initial conditions
        if predict and hasattr(self, 'X0_eval'):
            X0 = self.X0_eval.clone()
            # Align batch size to t: X0_eval may have been built with smaller M (e.g. eval_total_state_cap)
            M_t = t.shape[0]
            if X0.shape[0] != M_t:
                if X0.shape[0] < M_t:
                    # Expand by repeating rows to match t
                    n_rep = (M_t + X0.shape[0] - 1) // X0.shape[0]
                    X0 = X0.repeat(n_rep, 1)[:M_t]
                else:
                    X0 = X0[:M_t]
        else:
            X0 = self._init_X0()
        X0 = X0.detach().clone().requires_grad_(True)
        Y0, _ = self.net_u_Du(t0, X0, compute_Du=not traj_only)
        X_buffer.append(X0.detach().clone())
        Y_buffer.append(Y0.detach().clone())

        state0 = self._preprocess_state(t[:, 0, :], X0)
        # Only use mean (deterministic) when plotting or deterministic eval
        if traj_only and predict and (gamma_entropy == 0 or deterministic_eval):
            a, z0_init, zK_init, log_det_flow_init, _, mu, log_sigma, feat0 = self.net_actor.sample(state0, deterministic=True)
        else:
            if predict and hasattr(self, 'eps_predict') and len(self.eps_predict) > 0:
                noise = self.eps_predict[0] if isinstance(self.eps_predict, list) else self.eps_predict
                if isinstance(noise, dict) and "noise" in noise:
                    noise = noise["noise"]
            else:
                noise = None
            a, z0_init, zK_init, log_det_flow_init, eps, mu, log_sigma, feat0 = self.net_actor.sample(state0, deterministic=False, noise=noise)
            if predict and not _control_exact_:
                if not hasattr(self, 'eps_predict'):
                    self.eps_predict = []
                self.eps_predict.append({"noise": eps.detach().clone()})
        action_buffer.append(a)
        mu_logsigma_buffer.append(self.net_actor.pack_params(mu, log_sigma).detach().clone())
        # Update diagnostics using the initial actor sample
        _diag_update_from_actor(log_sigma, zK_init)

        # Initialize error accumulators for integration over time
        control_error_accum = []  # List to store errors at each time step
        Y_error_accum = []

        # Equally-spaced linspace grid: number of intervals
        N_actual = t.shape[1] - 1  # N

        for k in range(N_actual):
            t1 = t[:, k + 1, :]
            W1 = W[:, k + 1, :]
            dW = (W1 - W0).unsqueeze(-1) # M, 1
            # NOTE: non-uniform dt disabled; use constant dt everywhere.
            # dt_actual = (t1 - t0).squeeze(-1)  # [M]
            mask  = (P > t0.unsqueeze(1)) & (P <= t1.unsqueeze(1))
            Njump = mask.sum(dim=1, keepdim=True).to(torch.float32)
            # dt_avg = dt_actual.mean().item() if dt_actual.numel() > 0 else dt
            dMt   = Njump.squeeze(-1) - self.params_equ["lambda"] * dt  # [M]

            # ---- Default values to avoid branch-dependent UnboundLocalError ----
            # These are needed by critic TD / KL terms even when _control_exact_=True.
            entropy_bonus_dt = torch.zeros((X0.size(0), 1), device=X0.device, dtype=X0.dtype)  # already dt-scaled
            neglogp = torch.zeros_like(entropy_bonus_dt)
            neglogp_mc = neglogp
            log_prob = torch.zeros_like(entropy_bonus_dt)

            control_pre = None
            control_samples = None
            mu_gaussian_exact = None  # action-space target mean from exact (PINN/Taylor) policy, if computed
            if not _control_exact_: 
                if hasattr(self.net_actor, "set_update_step"):
                    if not hasattr(self, "_global_actor_update_step"):
                        self._global_actor_update_step = 0
                    self.net_actor.set_update_step(self._global_actor_update_step)

                state_k = self._preprocess_state(t0, X0)
                if traj_only and predict and (gamma_entropy == 0 or deterministic_eval):
                    control, z0, zK, log_det_flow, _, mu_, log_sigma, feat_ = self.net_actor.sample(
                        state_k, deterministic=True
                    )
                else:
                    if predict and hasattr(self, 'eps_predict') and len(self.eps_predict) > k + 1:
                        noise = self.eps_predict[k + 1]
                        if isinstance(noise, dict) and "noise" in noise:
                            noise = noise["noise"]
                    else:
                        noise = None
                    control, z0, zK, log_det_flow, eps, mu_, log_sigma, feat_ = self.net_actor.sample(
                        state_k, deterministic=False, noise=noise
                    )
                    if predict and not _control_exact_:
                        if not hasattr(self, 'eps_predict'):
                            self.eps_predict = []
                        while len(self.eps_predict) <= k + 1:
                            self.eps_predict.append(None)
                        self.eps_predict[k + 1] = {"noise": eps.detach().clone()}
                # Per-itr diagnostics from main actor path (avoid entropy-loop extra samples)
                _diag_update_from_actor(log_sigma, zK)

                # Use deterministic action as control "mean" for error comparisons
                control_det, z0_det, zK_det, log_det_flow_det, _, _, _, _ = self.net_actor.sample(
                    state_k, deterministic=True
                )
                # IMPORTANT:
                # - deterministic=True is a single-point proxy (often z=0), NOT the true mean of flow+squash policy.
                # - for any "mean vs mean" metric (vs exact Gaussian loc), use MC estimate E[a|s].
                # Keep control_det for debugging/visualization only.
                # NOTE: mixture_mean() is MC-based and can be very expensive if called at every time-step.
                # Use `K_mean_train` to decouple training speed from eval-quality defaults.
                #     bool(self.params_train.get("compute_mu_of_policy_error", True)) or
                # )
                need_control_mean = compute_exact_this_itr and bool(self.params_train.get("compute_mu_of_policy_error", True))
                if need_control_mean:
                    K_mean_train = int(self.params_train.get("K_mean_train", self.params_train.get("K_mean_eval", 1)))
                    K_mean_train = max(1, K_mean_train)
                    with torch.no_grad():
                        control_mean_mc = self.net_actor.mixture_mean(
                            state_k, n_samples=K_mean_train, antithetic=True
                        )
                    control_pre = control_mean_mc
                else:
                    control_pre = control_det
                if compute_exact_this_itr and (k == (N_actual - 1)) and (itr % 10 == 0):
                    # Compute exact mean control in action space ONLY when we are about to print it.
                    try:
                        if gamma_entropy == 0:
                            mu_gaussian_exact, _ = self.compute_exact_no_exploratory(X0)
                        else:
                            optimal_policy, _ = self.compute_exact_entropy(X0)
                            mu_gaussian_exact = optimal_policy.loc
                    except Exception:
                        mu_gaussian_exact = None
                    # Logging: `control_pre` is either MC mean (if computed) or deterministic proxy.
                    # Avoid referencing branch-local variables like K_mean_eval/control_mean_mc.
                    if need_control_mean:
                        k_mean_used = int(self.params_train.get("K_mean_train", self.params_train.get("K_mean_eval", 1)))
                        k_mean_used = max(1, k_mean_used)
                        control_mean_label = f"control_mean_mc({k_mean_used})"
                    else:
                        control_mean_label = "control_mean_det"
                    print(f"[actor_output] control_sample_mean={control.mean().item():.6e}, "
                          f"control_det_mean={control_det.mean().item():.6e}, "
                          f"{control_mean_label}={control_pre.mean().item():.6e}")
                    print(f"[actor_output] mu_mean={mu_.mean().item():.6e}, log_sigma_mean={log_sigma.mean().item():.6e}, sigma_mean={torch.exp(log_sigma).mean().item():.6e}")

                    # ========== Check mu_ state-dependency ==========
                    mu_std = mu_.std().item()
                    mu_range = mu_.max().item() - mu_.min().item()
                    mu_state_dependency_ratio = mu_std / (torch.exp(log_sigma).mean().item() + 1e-10)
                    print(f"[mu_diagnosis] mu_std={mu_std:.6e}, mu_range={mu_range:.6e}, state_dependency_ratio={mu_state_dependency_ratio:.6f}")
                    if mu_std < 0.01:
                        print(f"[WARNING] mu_ has very small std ({mu_std:.6e}) - network output is nearly constant!")
                        print(f"  This means the network is not learning state-dependent policy!")
                        print(f"  mu_range={mu_range:.6e} is very small, suggesting mu_ is almost constant across states")
                        print(f"  SOLUTIONS:")
                        print(f"    a) Check if network architecture is too simple or has dead neurons")
                        print(f"    b) Increase network capacity (width/depth) to learn state-dependency")
                        print(f"    c) Check if input features (state) are being properly normalized/scaled")
                        print(f"    d) Verify that state information is actually reaching the network")
                        print(f"    e) Add state-dependent regularization to encourage mu_ to vary with state")
                        print(f"    f) Check if actor learning rate is too small or gradients are vanishing")
                    elif mu_state_dependency_ratio < 0.1:
                        print(f"[WARNING] mu_ state-dependency is weak")
                        print(f"  mu_state_dependency_ratio={mu_state_dependency_ratio:.6f} (< 0.1 means mu varies <10% of sigma)")
                        print(f"  Network may be learning a nearly constant policy")

                    if gamma_entropy > 0 and mu_gaussian_exact is not None:
                        print(f"[actor_output] mu_gaussian_exact_mean={mu_gaussian_exact.mean().item():.6e}")
                        # IMPORTANT: Compare values in the same space!
                        # mu_ is in z0-space (pre-flow, pre-squash), while mu_gaussian_exact is in a-space (action space)
                        # The correct comparison should be in action space: control_det vs mu_gaussian_exact
                        # Action generation chain: z0 (mu_) --flow--> zK --squash--> a (control_det)
                        control_mean_for_cmp = control_pre if control_pre is not None else control_det
                        control_sign_product = control_mean_for_cmp.mean().item() * mu_gaussian_exact.mean().item()
                        print(f"[actor_output] control_sign_product(mean_mc)={control_sign_product:.6e} (negative => opposite signs in action space)")
                        # Also show z0-space comparison for reference (but this can have opposite signs due to flow+squash)
                        mu_sign_product = mu_.mean().item() * mu_gaussian_exact.mean().item()
                        print(f"[actor_output] mu_sign_product={mu_sign_product:.6e} (z0-space vs a-space, opposite signs are normal due to flow+squash)")
                        control_error_abs = abs(control_mean_for_cmp.mean().item() - mu_gaussian_exact.mean().item())
                        print(f"[actor_output] control_error_abs(mean_mc)={control_error_abs:.6e}, relative_error={control_error_abs/abs(mu_gaussian_exact.mean().item())*100:.2f}%")
                        # Check if control_det and mu_gaussian_exact have opposite signs (both in action space)
                        if control_sign_product < 0:
                            print(f"[actor_output] WARNING: control_mean_mc and mu_gaussian_exact have opposite signs in action space!")
                        # Note: mu_ and mu_gaussian_exact having opposite signs is NOT necessarily a problem
                        # because they are in different spaces (z0-space vs a-space)

                # Fail fast: silently replacing invalid values produces zero or misleading gradients.
                actor_values = {"mu": mu_, "log_sigma": log_sigma, "z0": z0,
                                "zK": zK, "log_det_flow": log_det_flow}
                bad_actor_values = [name for name, value in actor_values.items()
                                    if not torch.isfinite(value).all()]
                if bad_actor_values:
                    raise FloatingPointError(
                        f"non-finite actor values at itr={itr}, k={k}: {', '.join(bad_actor_values)}"
                    )

                # Check for extreme values in log_det_flow that might cause issues
                if torch.isinf(log_det_flow).any() or (log_det_flow.abs() > 50).any():
                    print(f"[WARNING] Extreme values in log_det_flow at itr={itr}, k={k}:")
                    print(f"  inf_count={torch.isinf(log_det_flow).sum().item()}, abs>50 count={(log_det_flow.abs() > 50).sum().item()}")
                    print(f"  range=[{log_det_flow.min().item():.6e}, {log_det_flow.max().item():.6e}]")

                log_prob = self.net_actor.log_prob_from_cache(
                    mu_,
                    log_sigma,
                    z0=z0,
                    zK=zK,
                    log_det_flow=log_det_flow,
                )
                # Score-function log-probability must hold the sampled action fixed.
                log_prob_score = self.net_actor.log_prob(state_k, control.detach())
                # ---- entropy bookkeeping (naming to avoid self-contradiction) ----
                # neglogp: single-sample estimate of -log pi(a|s)
                neglogp = -log_prob  # [M,1]
                # neglogp_mc: MC estimate of E[-log pi] (Shannon entropy estimator)
                # NOTE: This can be extremely expensive because it triggers extra actor samples per time-step.
                # Use `K_entropy_train` to decouple training speed from eval-quality defaults.
                K_entropy = max(1, int(self.params_train.get("K_entropy_train", self.params_train.get("K_entropy", 1))))
                if K_entropy == 1:
                    neglogp_mc = neglogp
                else:
                    logp_list = [log_prob]
                    for _ in range(K_entropy - 1):
                        a_ent, z0_ent, zK_ent, log_det_flow_ent, _, mu_ent, log_sigma_ent, feat_ent = self.net_actor.sample(
                            state_k, deterministic=False
                        )
                        logp_ent = self.net_actor.log_prob_from_cache(
                            mu_ent,
                            log_sigma_ent,
                            z0=z0_ent,
                            zK=zK_ent,
                            log_det_flow=log_det_flow_ent,
                        )
                        logp_list.append(logp_ent)
                    logp_stack = torch.stack(logp_list, dim=0)  # [K,M,1]
                    neglogp_mc = -(logp_stack.mean(dim=0))

                entropy_bonus = gamma_entropy * neglogp_mc  # [M,1]
                # - critic / TD uses dt-scaled, detached neglogp to avoid backprop through critic targets
                entropy_bonus_dt = gamma_entropy * neglogp.detach() * dt  # [M,1]

                detach_control = self.params_train.get("detach_control_for_env", True)
                if detach_control:
                    control_env = control.detach()
                else:
                    control_env = control  # WARNING: This may make training less stable, but allows gradients through dynamics

                mu_func = self.mu_torch(t0, X0, control_env) 
                diffusion = torch.bmm(self.sigma_torch(t0, X0, control_env), dW).squeeze(2)  
                xi =  self.xi(t0, X0, control_env)
                X1 = X0 + mu_func * dt + diffusion + xi * dMt

                action_buffer.append(control)
                mu_logsigma_buffer.append(self.net_actor.pack_params(mu_, log_sigma).detach().clone())
                # std_loss = std_lambda  * sigma.pow(2)
                # Optional: compute mu-of-policy error vs exact (also expensive).
                # To keep "exact" solves truly rare, gate this metric to the same outer-iteration schedule
                # and compute it only at the last time-step.
                compute_mu_err = bool(self.params_train.get("compute_mu_of_policy_error", True))
                if compute_mu_err and compute_exact_this_itr and (k == (N_actual - 1)):
                    if gamma_entropy == 0:
                        mu_gaussian_exact, _ = self.compute_exact_no_exploratory(X0)
                    else:
                        optimal_policy, _ = self.compute_exact_entropy(X0)
                        mu_gaussian_exact = optimal_policy.loc
                    # Use action-space mean E[a|s] (MC) to compare with exact Gaussian loc.
                    mu_control = control_pre if control_pre is not None else control
                    # Safety: if exact solve returns NaN/Inf (e.g. out-of-domain), do not propagate into logs.
                    try:
                        mu_ctl_ok = torch.isfinite(mu_control).all()
                        mu_ex_ok = (mu_gaussian_exact is not None) and torch.isfinite(mu_gaussian_exact).all()
                    except Exception:
                        mu_ctl_ok, mu_ex_ok = True, (mu_gaussian_exact is not None)
                    if (not mu_ctl_ok) or (not mu_ex_ok):
                        n_bad_ctl = int((~torch.isfinite(mu_control)).sum().item()) if torch.is_tensor(mu_control) else -1
                        n_bad_ex = int((~torch.isfinite(mu_gaussian_exact)).sum().item()) if torch.is_tensor(mu_gaussian_exact) else -1
                        print(
                            f"[warn] skip mu_of_policy_error at itr={itr}, k={k}: "
                            f"mu_control_bad={n_bad_ctl}, mu_exact_bad={n_bad_ex}"
                        )
                    else:
                        mu_of_policy_error_step = torch.mean(
                            (mu_control - mu_gaussian_exact) ** 2 / (mu_gaussian_exact ** 2 + 1e-8)
                        )
                        if torch.isfinite(mu_of_policy_error_step).all():
                            mu_of_policy_error_accum += mu_of_policy_error_step.detach().clone()
                            mu_of_policy_error_count += 1
                        else:
                            print(
                                f"[warn] skip mu_of_policy_error at itr={itr}, k={k}: "
                                f"step is not finite (min={mu_of_policy_error_step.min().item():.4g}, "
                                f"max={mu_of_policy_error_step.max().item():.4g})"
                            )

            else:
                if gamma_entropy == 0:
                    control_exact, _ = self.compute_exact_no_exploratory(X0)
                    # print("control_exact", control_exact.shape) # torch.Size([])
                    control_exact = control_exact * torch.ones_like(X0)
                    # print("control_exact.shape after", control_exact.shape) # torch.Size([500, 1])
                    control_pre = control_exact
                else:
                    # IMPORTANT: Use PINN (exact) policy, not learned actor network!
                    # When _control_exact_=True, we should use the true PINN solution
                    optimal_policy, _ = self.compute_exact_entropy(X0)

                    # Handle both DiscretePolicy and Normal distributions
                    if isinstance(optimal_policy, self.DiscretePolicy):
                        # Discrete policy: sample from discrete distribution
                        sample_discrete_in_predict = bool(
                            self.params_train.get("predict_sample_discrete_policy", True)
                        )
                        if predict and not sample_discrete_in_predict:
                            # In predict mode, allow deterministic mean (loc) for stability.
                            control_exact = optimal_policy.loc  # [M, 1]
                        elif predict:
                            # In predict mode, reuse the same eps as learned policy by inverse-CDF sampling:
                            # eps ~ N(0,1) -> q=Phi(eps) -> u = icdf(q)
                            if hasattr(self, 'eps_predict') and len(self.eps_predict) > k + 1:
                                eps_item = self.eps_predict[k + 1]
                                if isinstance(eps_item, dict) and "noise" in eps_item:
                                    eps_exact = eps_item["noise"]
                                else:
                                    eps_exact = eps_item
                            else:
                                raise RuntimeError(
                                    f"In predict mode, eps_predict[{k+1}] not found. "
                                    "Make sure to call train_loop with _control_exact_=False first to generate eps."
                                )
                            # Standard normal CDF to map to Uniform(0,1)
                            q = 0.5 * (1.0 + torch.erf(eps_exact / math.sqrt(2.0)))
                            control_exact = optimal_policy.icdf(q)  # [M,1]
                        else:
                            # Sample from discrete distribution
                            # sample(n=1) returns [M, 1], which matches X0 shape
                            control_exact = optimal_policy.sample(n=1)  # [M, 1]
                        control_pre = optimal_policy.loc  # [M, 1]
                    else:
                        # Normal distribution: can use loc + scale * eps
                        if predict:
                            # In predict mode, use the same random noise as learned policy path for fair comparison
                            if hasattr(self, 'eps_predict') and len(self.eps_predict) > k + 1:
                                eps_item = self.eps_predict[k + 1]
                                if isinstance(eps_item, dict) and "noise" in eps_item:
                                    eps_exact = eps_item["noise"]
                                else:
                                    eps_exact = eps_item
                            else:
                                # This should not happen if predict() is called correctly
                                # (first call with _control_exact_=False should generate all eps)
                                raise RuntimeError(
                                    f"In predict mode, eps_predict[{k+1}] not found. "
                                    "Make sure to call train_loop with _control_exact_=False first to generate eps."
                                )
                            # Sample from Normal policy using the same noise
                            control_exact = optimal_policy.loc + optimal_policy.scale * eps_exact
                        else:
                            # If not in predict mode, sample from Normal policy with new random noise
                            eps_exact = torch.randn_like(X0)
                            control_exact = optimal_policy.loc + optimal_policy.scale * eps_exact

                        # Normal policy mean is just optimal_policy.loc
                        control_pre = optimal_policy.loc

                mu_func = ((self.params_equ["mu"]-self.params_equ["r"]) * control_exact + self.params_equ["r"]) * X0
                # diffusion = torch.bmm((control_exact * self.params_equ["sigma"]).unsqueeze(-1), (W1 - W0).unsqueeze(-1)).squeeze(2)  # [M, D]
                diffusion = torch.bmm(torch.diag_embed(control_exact*X0) * self.params_equ["sigma"], (W1 - W0).unsqueeze(-1)).squeeze(2)  # [M, D]
                control = control_exact
                action_buffer.append(control)
                xi = control * self.params_equ["z"] * X0  # [M, D]
                X1 = X0 + mu_func * dt + diffusion + xi * dMt

            V0, partial_V0 = self.net_u_Du(t0, X0, compute_Du=not traj_only)            
            V1 = self.net_critic_target(self._preprocess_state(t1, X1))

            if not traj_only:
                with torch.no_grad():
                    # if gamma_entropy == 0:
                    #     reward = self.reward_func(X0, dt) 
                    # else:

                    #     reward = self.reward_func_entropy(X0, dt, tsallis_entropy) 

                    # Constant dt (linspace grid)
                    reward = self.reward_func(X0, dt)
                    # Always define reward_plus_entropy_dt (even when _control_exact_=True)
                    reward_plus_entropy_dt = reward + entropy_bonus_dt
                    disc = torch.exp(torch.tensor(-beta*dt, device=V0.device))
                    V_x_alphaxu, _ = self.net_u_Du(t0, X0 + xi, compute_Du=False)
                delta_TD = corrected_td_residual(
                    reward_plus_entropy_dt, V0, V1, partial_V0, diffusion,
                    V_x_alphaxu, dMt, disc,
                )
                loss_critic_step = torch.mean(delta_TD ** 2)

                critic_loss_accum = critic_loss_accum + loss_critic_step
                critic_chunk_count += 1
                if ((k + 1) % K_critic == 0) or (k == N_actual - 1):
                    loss_critic_chunk = critic_loss_accum / float(critic_chunk_count)
                    self.optimizer_critic.zero_grad(set_to_none=True)
                    loss_critic_chunk.backward()
                    # Critic gradient clipping (prevents single-step gradient explosions)
                    grad_clip_max = self.params_train.get("grad_clip_max", 10.0)
                    torch.nn.utils.clip_grad_norm_(self.net_critic.parameters(), grad_clip_max)
                    self.optimizer_critic.step()
                    # Polyak-average update of target critic
                    self._polyak_update_target_critic()
                    # Only step scheduler at the end of each iteration, not every chunk
                    # self.scheduler_critic.step()
                    total_loss_critic_sum += loss_critic_chunk.detach()
                    critic_update_count += 1  
                    critic_loss_accum = 0.0
                    critic_chunk_count = 0

                with torch.no_grad():
                    advantage = (delta_TD.detach() - entropy_bonus_dt) / (dt + 1e-8)

                A_buffer.append(advantage)
                # store neglogp for diagnostics (old name kept to avoid touching downstream plotting code)
                logpi_tsallis_buffer.append(neglogp)
                logpi_buffer.append(log_prob)
                # Historical diagnostic name: squared corrected TD residual.
                loss_KL = delta_TD.detach().square().sum(dim=-1)

                #### loss 1: actor policy gradient
                ppo_clip_eps = self.params_train.get("ppo_clip_eps", None)
                _ppo_clip_defer = (ppo_clip_eps is not None and float(ppo_clip_eps) > 0)

                if _ppo_clip_defer:
                    # PPO-Clip: buffer rollout data for Phase 2 update after full trajectory
                    ppo_state_buf.append(state_k.detach().clone())
                    ppo_action_buf.append(control.detach().clone())
                    ppo_logprob_old.append(log_prob_score.detach().clone())
                    ppo_advantage_buf.append(advantage.detach().clone())
                    ppo_entropy_buf.append(entropy_bonus.detach().clone())
                else:
                    # REINFORCE actor loss (no clip)
                    _actor_loss_type = int(self.params_train.get("actor_loss_type", 1))
                    if _actor_loss_type == 2:
                        #### loss 2: entropy folded into advantage
                        adv_ent = advantage + gamma_entropy * neglogp_mc.detach()  # [M, 1]
                        loss_actor_step = -(1 / beta) * (log_prob_score * adv_ent)  # [M, 1]
                        loss_actor_step = torch.mean(loss_actor_step)
                    else:
                        #### loss 1: separate policy gradient + entropy bonus
                        policy_loss = torch.sum(log_prob_score * advantage, dim=-1)  # [M]
                        entropy_loss = entropy_bonus.squeeze(-1)  # [M]
                        loss_actor_step = -(1/ beta) * (policy_loss + entropy_loss)  # [M]
                        loss_actor_step = torch.mean(loss_actor_step)

                if not _ppo_clip_defer:
                    # ---- KL(π_actor || π_PINN) regularization ----
                    # KL = E_{a~π_actor}[log π_actor(a) - log π_PINN(a)]
                    # Accumulate the policy-gradient and entropy objective only.
                    actor_loss_accum = actor_loss_accum + loss_actor_step
                    actor_chunk_count += 1

                if not _ppo_clip_defer and (((k + 1) % K_actor == 0) or (k == N_actual - 1)):
                    loss_actor_chunk = actor_loss_accum / float(actor_chunk_count)
                    self.optimizer_actor.zero_grad(set_to_none=True)

                    # Optional warmup: critic-only (skip actor optimization for first warmup_actor_steps actor-updates)
                    warmup_strategy = str(self.params_train.get("warmup_strategy", "affine_flow")).lower()
                    warmup_actor_steps = int(self.params_train.get("warmup_actor_steps", self.params_train.get("freeze_flow_steps", 0)))
                    curr_actor_updates = int(getattr(self, "_global_actor_update_step", 0))
                    train_actor_now = not (warmup_strategy in ("critic_only", "critic") and (curr_actor_updates < warmup_actor_steps))

                    # Step 3: freeze flow *updates* for first freeze_flow_steps actor-updates
                    # (forward path still uses flow; this avoids train/eval mismatch and sudden graph changes)
                    if hasattr(self, "_apply_flow_param_freeze"):
                        self._apply_flow_param_freeze()

                    did_actor_step = False
                    if train_actor_now:
                        loss_actor_chunk.backward()
                        grad_clip_max = self.params_train.get("grad_clip_max", 10.0)
                        # Clip gradients for all actor parameters
                        g = torch.nn.utils.clip_grad_norm_(self.net_actor.parameters(), grad_clip_max)
                        # Convert to float for printing and comparison
                        g = float(g.detach().cpu())
                    else:
                        g = 0.0
                        if itr % 10 == 0 and k == 0:
                            print(f"[warmup] warmup_strategy={warmup_strategy}: skip actor update (curr={curr_actor_updates}, warmup_actor_steps={warmup_actor_steps})")

                    # B. Check actor gradient norm after backward
                    if itr % 10 == 0 and k == 0:
                        # Calculate gradient norm manually for verification
                        grad_norm_actor = 0.0
                        param_count = 0
                        for name, param in self.net_actor.named_parameters():
                            if not getattr(param, "requires_grad", False):
                                continue
                            if param.grad is not None:
                                param_grad_norm = param.grad.data.norm(2).item()
                                grad_norm_actor += param_grad_norm ** 2
                                param_count += 1
                            else:
                                print(f"[WARNING] Parameter {name} has no gradient!")
                        grad_norm_actor = grad_norm_actor ** 0.5

                        lr_actor = self.optimizer_actor.param_groups[0]['lr']
                        # Calculate effective update magnitude
                        effective_update = g * lr_actor  # approximate step size
                        print(f"[actor] grad_norm={g:.6f}, loss_chunk={loss_actor_chunk.item():.6e}, lr={lr_actor:.6e}")
                        print(f"[actor] grad_norm_manual={grad_norm_actor:.6f} (sqrt(sum(p.grad^2)))")
                        print(f"[actor] effective_update_magnitude={effective_update:.6e} (grad_norm * lr)")
                        print(f"[actor] If grad_norm ≈ 0, actor is NOT receiving gradients!")
                        print(f"[actor] If effective_update < 1e-6, actor learning will be very slow")
                        print(f"[actor] K_actor={K_actor}, updates_per_iteration={self.N // K_actor + 1}")

                        # Check for gradient saturation (vanishing gradients due to sigmoid/tanh saturation)
                        if hasattr(control, 'abs'):
                            # Check if control is near action_min/action_max (more meaningful than comparing to action_range)
                            if hasattr(self.net_actor, 'action_min') and hasattr(self.net_actor, 'action_max'):
                                amin = float(self.net_actor.action_min)
                                amax = float(self.net_actor.action_max)
                                arng = float(amax - amin) if (amax - amin) != 0 else 1.0
                                normalized = (control - amin) / (arng + 1e-8)
                                # If a large fraction is in the saturation bands, gradients through squash may vanish
                                sat_margin = float(getattr(self.net_actor, "sat_margin", 0.02))
                                frac_low = (normalized < sat_margin).float().mean().item()
                                frac_high = (normalized > (1.0 - sat_margin)).float().mean().item()
                                if (frac_low + frac_high) > 0.2:
                                    print(f"[WARNING] control near action bounds: frac_low={frac_low:.3f}, frac_high={frac_high:.3f} (sat_margin={sat_margin})")
                                    print(f"[WARNING] This can cause vanishing gradients in sigmoid/tanh squash!")
                    if train_actor_now:
                        self.optimizer_actor.step()
                        did_actor_step = True
                    # IMPORTANT (bugfix):
                    # If warmup_strategy="critic_only", we skip actor optimizer steps for the first warmup_actor_steps.
                    # Previously, _global_actor_update_step only advanced when we actually stepped the actor optimizer,
                    # which deadlocked warmup (curr_actor_updates stayed at 0 forever) and kept Loss_actor at 0.

                    # Fix: advance the global actor update step for schedule/warmup bookkeeping at every actor-chunk,
                    # even when we skip the optimizer step.
                    self._global_actor_update_step = int(getattr(self, "_global_actor_update_step", 0) or 0) + 1
                    if hasattr(self.net_actor, "set_update_step"):
                        self.net_actor.set_update_step(int(self._global_actor_update_step))
                    if hasattr(self, "_apply_flow_param_freeze"):
                        self._apply_flow_param_freeze()
                    # Only step scheduler at the end of each iteration, not every chunk
                    # self.scheduler_actor.step()
                    if train_actor_now:
                        total_loss_actor_sum += loss_actor_chunk.detach()
                        actor_update_count += 1  
                    actor_loss_accum = 0.0
                    actor_chunk_count = 0

            X0, t0, W0 = X1.detach().clone().requires_grad_(True), t1, W1
            X_buffer.append(X1.detach().clone())
            Y_buffer.append(V1.detach().clone())

            # ----------------------------
            # Exact metrics (PINN "truth")
            # ----------------------------
            # Compute ONLY once per scheduled outer-iteration, and only at the last time-step.
            compute_exact_now = compute_exact_this_itr and (k == (N_actual - 1))

            if (not traj_only) and compute_exact_now:

                if gamma_entropy == 0:
                    control_exact, Y_exact = self.compute_exact_no_exploratory(X1)
                    control_mean_pred = control_pre  # [M, D]
                    control_mean_exact = control_exact  # [M, D]
                    control_error_step = torch.sum((control_mean_pred - control_mean_exact) ** 2, dim=-1) / (torch.sum(control_mean_exact ** 2, dim=-1) + 1e-12)  # [M]
                    control_error_accum.append(control_error_step)  # Store for integration
                    if k == (self.N-1):
                        x_min = X1.min().detach().cpu().item()
                        x_max = X1.max().detach().cpu().item()
                        y_mean = Y_exact.mean().detach().cpu().item()
                        y_max = Y_exact.max().detach().cpu().item()
                        if itr % 10 == 0 and not _control_exact_:
                            control_pred_mean = control_pre.mean().item() if control_pre is not None else 0.0
                            control_pred_std = control_pre.std().item() if control_pre is not None else 0.0
                            print(f"[control_comparison] control_exact_mean={control_exact.mean().item():.6e}, control_pred_mean={control_pred_mean:.6e}")
                            print(f"[control_comparison] control_exact_std={control_exact.std().item():.6e}, control_pred_std={control_pred_std:.6e}")
                        print("control_exact", control_exact.mean())
                        print("Y_exact", Y_exact.mean())
                        print("Y_pred", V1.mean())
                        # Also print last-step exact errors (same definitions as returned Y_error/control_error when only one sample exists).
                        # These are diagnostic-only and are typically shown as percentages.
                        # NOTE: control_error_step is [M]; Y_error_step is computed below (after Y_exact exists).
                        control_err_last = torch.sqrt(control_error_step.mean() * dt)
                        print(f"[train_exact@last] control_error={control_err_last.item() * 100.0:.3f}%")
                        print(f"[train] X1_range=[{x_min:.4f},{x_max:.4f}], Y_exact_mean={y_mean:.4f}, Y_exact_max={y_max:.4f}")
                else:
                    policy_exact, Y_exact = self.compute_exact_entropy(X1)
                    # For gamma > 0: use KL or Wasserstein distance
                    # Get predicted policy parameters
                    if not _control_exact_:
                        if getattr(self.net_actor, "actor_type", "") in ("hyper_gaussian", "hyper_gaussian_split", "gaussian_two_stage"):
                            mu_pred = control_pre  # [M, D] - action-space mean (MC) for fair comparison to exact loc
                            sigma_pred = None  # Don't force-feed a "pretend Gaussian" std for now
                        else:
                            mu_pred = control_pre  # [M, D] - action-space mean (MC)
                            sigma_pred = None
                    else:
                        mu_pred = control_exact  # [M, D]
                        sigma_pred = policy_exact.scale  # [M, D]
                    mu_exact = policy_exact.loc  # [M, D]
                    sigma_exact = policy_exact.scale  # [M, D]

                    # Use fallback strategy: KL -> Wasserstein -> CVM (with OOM handling)
                    control_error_step = self.compute_control_error_with_fallback(
                        control, policy_exact, mu_pred, sigma_pred, mu_exact, sigma_exact, M, self.D,
                        control_samples=control_samples
                    )
                    if control_error_step is not None:
                        control_error_accum.append(control_error_step)
                    if k == (self.N-1):
                        x_min = X1.min().detach().cpu().item()
                        x_max = X1.max().detach().cpu().item()
                        y_mean = Y_exact.mean().detach().cpu().item()
                        y_max = Y_exact.max().detach().cpu().item()
                        print("Y_exact", Y_exact.mean())
                        print("Y_pred", V1.mean())
                        if control_error_step is not None:
                            control_err_last = torch.sqrt(control_error_step.mean() * dt)
                            print(f"[train_exact@last] control_error={control_err_last.item() * 100.0:.3f}%")
                        print(f"[train] X1_range=[{x_min:.4f},{x_max:.4f}], Y_exact_mean={y_mean:.4f}, Y_exact_max={y_max:.4f}")

                # Value function error: |V_pred - V_exact|^2 / (|V_exact|^2 + eps)
                Y_pred = V1  # [M, 1]
                Y_error_step = (Y_pred - Y_exact) ** 2 / (Y_exact ** 2 + 1e-12)  # [M, 1]
                Y_error_accum.append(Y_error_step.squeeze(-1))  # [M]
                if k == (self.N - 1):
                    y_err_last = torch.sqrt(Y_error_step.mean() * dt / (self.T + 1e-12))
                    print(f"[train_exact@last] Y_error={y_err_last.item() * 100.0:.3f}%")
        # ============ Phase 2: PPO-Clip Actor Update (post-rollout) ============
        _ppo_clip_eps_val = self.params_train.get("ppo_clip_eps", None)
        _ppo_clip_active = (_ppo_clip_eps_val is not None and float(_ppo_clip_eps_val) > 0)
        if _ppo_clip_active and len(ppo_state_buf) > 0 and not _control_exact_ and not traj_only:
            ppo_epochs = int(self.params_train.get("ppo_epochs", 3))
            eps_clip = float(_ppo_clip_eps_val)
            warmup_strategy = str(self.params_train.get("warmup_strategy", "affine_flow")).lower()
            warmup_actor_steps = int(self.params_train.get("warmup_actor_steps", self.params_train.get("freeze_flow_steps", 0)))
            grad_clip_max = self.params_train.get("grad_clip_max", 10.0)

            for ppo_ep in range(ppo_epochs):
                _ppo_actor_loss_accum = 0.0
                _ppo_chunk_count = 0
                for _k in range(len(ppo_state_buf)):
                    log_prob_new = self.net_actor.log_prob(
                        ppo_state_buf[_k], ppo_action_buf[_k]
                    )
                    log_prob_old = ppo_logprob_old[_k]
                    ratio = torch.exp(log_prob_new - log_prob_old)
                    ratio_clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
                    _adv = ppo_advantage_buf[_k]
                    surrogate = torch.min(ratio * _adv, ratio_clipped * _adv)
                    _actor_loss_type = int(self.params_train.get("actor_loss_type", 1))
                    if _actor_loss_type == 2:
                        adv_ent = _adv + gamma_entropy * (-log_prob_new).detach()
                        surrogate_ent = torch.min(ratio * adv_ent, ratio_clipped * adv_ent)
                        policy_loss = torch.sum(surrogate_ent, dim=-1)
                        loss_step = -(1.0 / beta) * policy_loss
                    else:
                        policy_loss = torch.sum(surrogate, dim=-1)
                        entropy_loss = gamma_entropy * (-log_prob_new).squeeze(-1)
                        loss_step = -(1.0 / beta) * (policy_loss + entropy_loss)
                    loss_step = torch.mean(loss_step)
                    _ppo_actor_loss_accum = _ppo_actor_loss_accum + loss_step
                    _ppo_chunk_count += 1
                    if ((_k + 1) % K_actor == 0) or (_k == len(ppo_state_buf) - 1):
                        loss_chunk = _ppo_actor_loss_accum / float(_ppo_chunk_count)
                        self.optimizer_actor.zero_grad(set_to_none=True)
                        curr_actor_updates = int(getattr(self, "_global_actor_update_step", 0))
                        train_actor_now = not (warmup_strategy in ("critic_only", "critic") and (curr_actor_updates < warmup_actor_steps))
                        if hasattr(self, "_apply_flow_param_freeze"):
                            self._apply_flow_param_freeze()
                        if train_actor_now:
                            loss_chunk.backward()
                            torch.nn.utils.clip_grad_norm_(self.net_actor.parameters(), grad_clip_max)
                            self.optimizer_actor.step()
                            total_loss_actor_sum += loss_chunk.detach()
                            actor_update_count += 1
                        self._global_actor_update_step = int(getattr(self, "_global_actor_update_step", 0) or 0) + 1
                        if hasattr(self.net_actor, "set_update_step"):
                            self.net_actor.set_update_step(int(self._global_actor_update_step))
                        if hasattr(self, "_apply_flow_param_freeze"):
                            self._apply_flow_param_freeze()
                        _ppo_actor_loss_accum = 0.0
                        _ppo_chunk_count = 0
                if itr % 20 == 0 and ppo_ep == 0:
                    with torch.no_grad():
                        _ratios = []
                        for _k in range(len(ppo_state_buf)):
                            lp_new = self.net_actor.log_prob(
                                ppo_state_buf[_k], ppo_action_buf[_k]
                            )
                            _ratios.append(torch.exp(lp_new - ppo_logprob_old[_k]).mean().item())
                        _r_mean = np.mean(_ratios)
                        print(f"[PPO-Clip] epoch={ppo_ep}, mean_ratio={_r_mean:.4f}, eps={eps_clip}")

        if not traj_only:
            self.control_list.append(control.detach().clone())
            self.KLLoss_list.append(loss_KL.mean().detach().clone())
            if critic_update_count > 0:
                total_loss_critic_sum = total_loss_critic_sum / critic_update_count
            else:
                total_loss_critic_sum = torch.tensor(0.0, device=self.device)
            if actor_update_count > 0:
                total_loss_actor_sum = total_loss_actor_sum / actor_update_count
            else:
                total_loss_actor_sum = torch.tensor(0.0, device=self.device)
            self.ActorLoss_list.append(total_loss_actor_sum.detach().clone())
            self.CriticLoss_list.append(total_loss_critic_sum.detach().clone())
            if mu_of_policy_error_count > 0:
                mu_of_policy_error_avg = mu_of_policy_error_accum / mu_of_policy_error_count
            else:
                # Metric not computed on this iteration (exact gated off).
                mu_of_policy_error_avg = torch.tensor(float("nan"), device=self.device)
            self.mu_of_policy_error_list.append(mu_of_policy_error_avg.detach().clone())
            if mu_of_policy_error_count > 0:
                mu_of_policy_error_pct = mu_of_policy_error_avg.mean().item() * 100.0
                print(f"mu_of_policy_error.mean = {mu_of_policy_error_avg.mean().item():.6f} ({mu_of_policy_error_pct:.3f}%)")

            # Diagnostic: print state distribution info for debugging
            if itr % 50 == 0 and len(X_buffer) > 0:
                X_all = torch.stack(X_buffer, dim=1)  # [M, N+1, D]
                D = X_all.shape[-1]  # Get D from tensor shape
                X_flat = X_all.reshape(-1, D)  # [M*(N+1), D]
                x_min_train = X_flat.min().item()
                x_max_train = X_flat.max().item()
                x_mean_train = X_flat.mean().item()
                x_std_train = X_flat.std().item()
                print(f"[state_dist] training states: min={x_min_train:.4f}, max={x_max_train:.4f}, mean={x_mean_train:.4f}, std={x_std_train:.4f}")

        # Integrate exact-metric errors over time (diagnostic only)
        # If exact metrics are disabled, return NaN to avoid misleading "0%" prints.
        if len(control_error_accum) > 0:
            control_error_t = torch.stack(control_error_accum, dim=0)  # [N, M]
            control_error = torch.sqrt(control_error_t.mean(dim=1).sum() * dt)
        else:
            control_error = torch.tensor(float("nan"), device=self.device)

        if len(Y_error_accum) > 0:
            Y_error_t = torch.stack(Y_error_accum, dim=0)  # [N, M]
            Y_error = torch.sqrt(Y_error_t.mean(dim=1).sum() * dt / (self.T + 1e-12))
        else:
            Y_error = torch.tensor(float("nan"), device=self.device)

        if (not traj_only) and (mu_of_policy_error_count > 0):
            mu_of_policy_error_final = mu_of_policy_error_accum / mu_of_policy_error_count
        else:
            # Return None when metric wasn't computed (avoids misleading "0%" logs).
            mu_of_policy_error_final = None

        # Record whether optimizers actually stepped in this outer-iteration.
        # This is used to guard lr_scheduler.step() calls to avoid
        # "Detected call of lr_scheduler.step() before optimizer.step()" warnings during warmup.
        try:
            self._did_actor_step_last_itr = bool(actor_update_count > 0)
        except Exception:
            self._did_actor_step_last_itr = False
        try:
            self._did_critic_step_last_itr = bool(critic_update_count > 0)
        except Exception:
            self._did_critic_step_last_itr = False

        # Expose per-iteration diagnostics to the caller (outer loop) for persistence.
        # Keep this robust: diagnostics must never break training.
        try:
            self._last_diagnose_stats = {
                "iteration": int(itr),
                "lr_critic": lr_critic_now,
                "log_std_max": None,
                "std_max": None,
                "|zK|_max": None,
                "log_det_squash_min": None,
            }
            if diag_seen:
                self._last_diagnose_stats["log_std_max"] = (diag_log_std_max if math.isfinite(diag_log_std_max) else None)
                self._last_diagnose_stats["std_max"] = (diag_std_max if math.isfinite(diag_std_max) else None)
                self._last_diagnose_stats["|zK|_max"] = (diag_abs_zK_max if math.isfinite(diag_abs_zK_max) else None)
                self._last_diagnose_stats["log_det_squash_min"] = (
                    diag_log_det_squash_min if math.isfinite(diag_log_det_squash_min) else None
                )
        except Exception:
            self._last_diagnose_stats = None

        return torch.stack(X_buffer, dim=1), torch.stack(Y_buffer, dim=1), Y_buffer[0], \
               torch.stack(action_buffer, dim=1), torch.stack(mu_logsigma_buffer, dim=1), \
               Y_error, control_error, \
               total_loss_critic_sum, total_loss_actor_sum, mu_of_policy_error_final 

    def train_all(self, n, N_Iter=10, start_it: int = 0):
        self.params_train["residual_version"] = RESIDUAL_VERSION
        self._save_params(self.output_folder)
        self._save_run_script(self.output_folder)
        # Track whether wandb is actually initialized/enabled for this run.
        # (WANDB_AVAILABLE only means the package is importable.)
        self._wandb_enabled = False
        if WANDB_AVAILABLE:
            # IMPORTANT: Never trigger interactive login prompts in training scripts.
            # Policy:
            # - If WANDB_DISABLED=1 or WANDB_MODE=disabled => skip
            # - If WANDB_MODE=offline/dryrun => offline
            # - Else: online only when we have credentials (env key or existing ~/.netrc); otherwise offline
            _wandb_mode_env = str(os.environ.get("WANDB_MODE", "")).lower().strip()
            _wandb_disabled = str(os.environ.get("WANDB_DISABLED", "")).lower() in ("1", "true", "yes", "y")
            api_key = (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_APIKEY") or "").strip()

            def _has_wandb_credentials() -> bool:
                # Non-interactive detection of pre-existing login.
                # wandb stores credentials in ~/.netrc on many setups.
                try:
                    netrc_path = os.path.expanduser("~/.netrc")
                    if os.path.isfile(netrc_path):
                        with open(netrc_path, "r", encoding="utf-8", errors="ignore") as f:
                            txt = f.read()
                        if ("api.wandb.ai" in txt) or ("wandb.ai" in txt):
                            return True
                except Exception:
                    pass
                return False

            if _wandb_disabled or _wandb_mode_env == "disabled":
                self._wandb_enabled = False
                print("[wandb] WANDB is disabled by env (WANDB_MODE/WANDB_DISABLED); skipping wandb init.")
            else:
                _explicit_online = (_wandb_mode_env == "online")
                have_creds = bool(api_key) or _has_wandb_credentials()

                if _wandb_mode_env in ("offline", "dryrun"):
                    _resolved_mode = "offline"
                elif _explicit_online and (not have_creds):
                    # Explicit online request but no creds => stay offline to avoid hanging on interactive login.
                    _resolved_mode = "offline"
                    print("[wandb] WANDB_MODE=online but no credentials found (WANDB_API_KEY or ~/.netrc); falling back to offline to avoid interactive login.")
                else:
                    _resolved_mode = "online" if have_creds else "offline"

                print(f"[wandb] Initializing wandb (mode={_resolved_mode})")
                # Only attempt login when key is provided; never call wandb.login() without key (can block).
                if _resolved_mode == "online" and api_key:
                    try:
                        wandb.login(key=api_key, relogin=True)
                        print("[wandb] Logged in using WANDB_API_KEY from environment")
                    except Exception as e:
                        if _explicit_online:
                            print(f"[wandb] Warning: login failed ({e}); WANDB_MODE=online so keeping online mode.")
                        else:
                            print(f"[wandb] Warning: login failed ({e}); switching to offline mode.")
                            _resolved_mode = "offline"

                # Convert parameters to wandb-compatible format
                def convert_to_wandb_value(v):
                    """Convert various types to wandb-compatible values"""
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            return v.item()
                        else:
                            return v.detach().cpu().numpy().tolist()
                    elif isinstance(v, np.ndarray):
                        return v.tolist()
                    elif isinstance(v, (list, tuple)):
                        return [convert_to_wandb_value(item) for item in v]
                    elif isinstance(v, dict):
                        return {k: convert_to_wandb_value(item) for k, item in v.items()}
                    elif isinstance(v, torch.device):
                        return str(v)
                    else:
                        # For other types (int, float, str, bool, None), return as is
                        return v

                # Prepare config with all parameters
                wandb_config = {
                    "equation": {k: convert_to_wandb_value(v) for k, v in self.params_equ.items()},
                    "train": {k: convert_to_wandb_value(v) for k, v in self.params_train.items()},
                }

                wandb_project = self.params_train.get("wandb_project", "merton_pinn_0201_optimizer_fix_spline")
                wandb_run_name = self.params_train.get("wandb_run_name", None)
                if wandb_run_name is None:
                    # wandb_run_name = os.path.basename(os.path.normpath(self.output_folder))
                    wandb_run_name = "Clip0.2_resnet_lr_actor_5e-5_lr_critic_1e-3_minlr2e-5_lrflow2e-5_xmax80_u-3_4_use_fixed_x_grid_False_flowB4_flowK8_actornet_width32_2_use_state_conditioning_False_warmup30_u_exact-3_4_preprocess_state_scale_100_N200_M200"
                try:
                    wandb.init(
                        project=wandb_project,
                        name=wandb_run_name,
                        config=wandb_config,
                        dir=self.output_folder,
                        mode=_resolved_mode,
                    )
                    self._wandb_enabled = True
                    print(f"[wandb] Initialized logging to project '{wandb_project}', run name: '{wandb_run_name}'")
                    print(f"[wandb] Recorded {len(self.params_train)} training parameters and {len(self.params_equ)} equation parameters")
                except Exception as e:
                    self._wandb_enabled = False
                    print(f"[wandb] Warning: wandb.init failed; disabling wandb for this run. err={e}")

        start_time = time.time()
        loss_log_path = os.path.join(self.output_folder, "loss_history.csv")
        if not os.path.isfile(loss_log_path):
            with open(loss_log_path, "w", encoding="utf-8") as f:
                f.write("iter,elapsed,loss_critic,loss_actor,pinn_value_error,pinn_control_error,"
                        "taylor_value_error,taylor_control_error\n")
        self.X0 = self._init_X0()
        self.bound_list.append(self.net_actor.bound_pos.detach().clone())
        for it in range(int(start_it), int(N_Iter)):
            X_pred, Y_pred, Y0, action_buffer, mu_logsigma_buffer, Y_error, control_error, loss_critic_final, loss_actor_final, mu_of_policy_error_final = self.train_loop(it)
            pinn_plot.append_training_diagnostics(
                self, it, self.KLLoss_list[-1] if self.KLLoss_list else None,
                mu_of_policy_error_final,
            )

            # ------------------------------------------------------------
            # Write diagnose JSON for it=159..599 into output_folder
            # File: diagnose_lr_log_std_log_det.json
            # Keys: lr_critic, log_std_max, std_max, |zK|_max, log_det_squash_min
            # ------------------------------------------------------------
            if 159 <= int(it) <= 599:
                diagnose_path = os.path.join(self.output_folder, "diagnose_lr_log_std_log_det.json")
                rec = getattr(self, "_last_diagnose_stats", None)
                if isinstance(rec, dict):
                    rec_to_write = dict(rec)
                    rec_to_write["iteration"] = int(it)  # enforce consistency
                    # Load existing (robustly), de-dup by iteration, append, and write back
                    existing = []
                    try:
                        if os.path.isfile(diagnose_path):
                            with open(diagnose_path, "r", encoding="utf-8") as f:
                                existing = metadata_from_disk(json.load(f), os.path.dirname(f.name))
                        if not isinstance(existing, list):
                            existing = []
                    except Exception:
                        existing = []
                    try:
                        new_existing = []
                        for r in existing:
                            if isinstance(r, dict):
                                try:
                                    if int(r.get("iteration", -1)) == int(it):
                                        continue
                                except Exception:
                                    pass
                            new_existing.append(r)
                        new_existing.append(rec_to_write)
                        with open(diagnose_path, "w", encoding="utf-8") as f:
                            json.dump(metadata_for_save(new_existing, os.path.dirname(f.name)), f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"[warn] failed to write diagnose json at it={it}: {e}")

            # Step schedulers once per iteration (not per chunk)
            # Guard: if optimizer didn't step (e.g. warmup/conditional updates), do not step scheduler.
            # This avoids PyTorch warning about scheduler.step() before optimizer.step().
            if bool(getattr(self, "_did_critic_step_last_itr", True)):
                self.scheduler_critic.step()
            # Note: single optimizer for base+flow, so step only one scheduler.
            # MultiStepLR already steps all param_groups.
            # If different schedules are needed for base vs flow, a custom scheduler would be required.
            if bool(getattr(self, "_did_actor_step_last_itr", True)):
                self.scheduler_actor.step()

            # Periodic checkpoints (also saves train_state_it*.pt for exact resume).
            periodic_every = int(self.params_train.get("periodic_ckpt_every", 50) or 0)
            if periodic_every > 0:
                self._save_periodic_ckpt(it, every=periodic_every)

            # After warmup ends, evaluate and fix x_grid
            warmup_steps = self.params_train.get("warmup_steps", 30)
            if it == warmup_steps and self.params_train.get("pinn_x_min_fixed") is None:
                print(f"[x_grid_fix] Warmup ended at itr={it}, evaluating x range...")
                # Collect x values from X_pred (current iteration's trajectory)
                if X_pred is not None and X_pred.numel() > 0:
                    X_all = X_pred.detach().reshape(-1)
                    x_min_observed = float(X_all.min().item())
                    x_max_observed = float(X_all.max().item())
                    # Set fixed x_grid with padding
                    x_min_fixed = max(0.0, x_min_observed - 10.0)
                    x_max_fixed = x_max_observed + 10.0
                    self.params_train["pinn_x_min_fixed"] = x_min_fixed
                    self.params_train["pinn_x_max_fixed"] = x_max_fixed
                    print(f"[x_grid_fix] Observed x range: [{x_min_observed:.4f}, {x_max_observed:.4f}]")
                    print(f"[x_grid_fix] Fixed x_grid: [{x_min_fixed:.4f}, {x_max_fixed:.4f}]")
                else:
                    print(f"[x_grid_fix] WARNING: No X_pred data available, keeping default range [0, 80]")

            # Use fixed x_grid if available
            if self.params_train.get("pinn_x_min_fixed") is not None:
                self.params_train["pinn_x_min"] = self.params_train["pinn_x_min_fixed"]
                self.params_train["pinn_x_max"] = self.params_train["pinn_x_max_fixed"]
            elapsed = time.time() - start_time

            # Log training losses to wandb every iteration
            if _wandb_run_active(self):
                log_dict = {
                    "iteration": it,
                    "loss_critic": to_scalar(loss_critic_final),
                    "loss_actor": to_scalar(loss_actor_final),
                    "elapsed_time": to_scalar(elapsed),
                }
                if mu_of_policy_error_final is not None:
                    # Unit clarity:
                    # - mu_of_policy_error_final is a relative MSE-like quantity in raw units
                    # - we often display it as percentage (%)
                    mu_raw = to_scalar(mu_of_policy_error_final)
                    log_dict["mu_of_policy_error"] = mu_raw * 100.0  # legacy (pct)
                    log_dict["mu_of_policy_error_pct"] = mu_raw * 100.0
                    log_dict["mu_of_policy_error_raw"] = mu_raw
                wandb.log(log_dict, step=it)

            pinn_v_err = ""
            pinn_u_err = ""
            taylor_v_err = ""
            taylor_u_err = ""

            if (it + 1) % self.eval_every == 0:
                # Provide current iteration to eval helpers (used for optional early refresh of eval_tx)
                self._eval_iteration = it
                errors = self._evaluate_pinn_taylor_errors()
                pinn_v_err = errors["pinn_value_error"]
                pinn_u_err = errors["pinn_control_error"]
                pinn_mu_err = errors.get("pinn_mu_gaussian_policy_error", None)
                # Record the reference comparison only after forward-only evaluation.
                if pinn_mu_err is not None and math.isfinite(float(pinn_mu_err)):
                    self.mu_of_policy_error_list[-1] = torch.tensor(float(pinn_mu_err), device=self.device)
                    pinn_plot.append_training_diagnostics(
                        self, it, self.KLLoss_list[-1] if self.KLLoss_list else None,
                        float(pinn_mu_err),
                    )
                taylor_v_err = errors["taylor_value_error"]
                taylor_u_err = errors["taylor_control_error"]
                taylor_mu_err = errors.get("taylor_mu_gaussian_policy_error", None)
                print(
                    f"[eval] it={it}  pinn_v={pinn_v_err}, pinn_u={pinn_u_err}, "
                    f"taylor_v={taylor_v_err}, taylor_u={taylor_u_err}, "
                    f"v_pinn_mean={errors['v_pinn_mean']}, y_pred_mean={errors['y_pred_mean']}, "
                    f"x_range=[{errors['x_min']},{errors['x_max']}], "
                    f"u_pinn_range=[{errors['u_pinn_min']},{errors['u_pinn_max']}], "
                    f"u_taylor_range=[{errors['u_taylor_min']},{errors['u_taylor_max']}]"
                )
                if pinn_mu_err is not None:
                    print(f"[eval] it={it}  pinn_mu_gaussian_policy_error={pinn_mu_err}")
                if taylor_mu_err is not None:
                    print(f"[eval] it={it}  taylor_mu_gaussian_policy_error={taylor_mu_err}")
                # Also evaluate on fixed grid (same as plotting) for fair comparison
                if (it + 1) % (self.eval_every * 5) == 0:  # Less frequent to avoid overhead
                    self._evaluate_on_fixed_grid(itr=it)
                self._append_pinn_taylor_errors(it=it, elapsed=elapsed, errors=errors)
                self._append_pinn_taylor_ranges(it=it, elapsed=elapsed, errors=errors)

                # Log evaluation metrics to wandb
                if _wandb_run_active(self):
                    eval_log_dict = {}
                    if pinn_v_err is not None and math.isfinite(pinn_v_err):
                        eval_log_dict["J_error"] = pinn_v_err
                    if pinn_u_err is not None and math.isfinite(pinn_u_err):
                        eval_log_dict["control_error"] = pinn_u_err
                    if pinn_mu_err is not None and math.isfinite(pinn_mu_err):
                        # pinn_mu_err is a raw relative error; log both raw and percent.
                        eval_log_dict["mu_gaussian_policy_error"] = pinn_mu_err * 100.0  # legacy (pct)
                        eval_log_dict["mu_gaussian_policy_error_pct"] = pinn_mu_err * 100.0
                        eval_log_dict["mu_gaussian_policy_error_raw"] = pinn_mu_err
                    if mu_of_policy_error_final is not None:
                        mu_raw = to_scalar(mu_of_policy_error_final)
                        eval_log_dict["mu_of_policy_error"] = mu_raw * 100.0  # legacy (pct)
                        eval_log_dict["mu_of_policy_error_pct"] = mu_raw * 100.0
                        eval_log_dict["mu_of_policy_error_raw"] = mu_raw
                    if eval_log_dict:
                        wandb.log(eval_log_dict, step=it)
                if (pinn_v_err is not None and pinn_u_err is not None
                        and math.isfinite(pinn_v_err) and math.isfinite(pinn_u_err)):
                    # ==========================
                    # [OLD CKPT LOGIC - commented]
                    # ==========================
                    # Weighted ckpt score: combine J_error, control_error, and mu_gaussian_policy_error
                    # w = self.params_train.get("ckpt_score_weights", None) or {}
                    # wJ = float(w.get("J_error", self.params_train.get("ckpt_w_J_error", 0.5)))
                    # wU = float(w.get("control_error", self.params_train.get("ckpt_w_control_error", 0.5)))
                    # wMu = float(w.get("mu_gaussian_policy_error", self.params_train.get("ckpt_w_mu_gaussian_policy_error", 0.0)))
                    # mu_term = (float(pinn_mu_err) if (pinn_mu_err is not None and math.isfinite(pinn_mu_err)) else 0.0)
                    # pinn_score = wJ * float(pinn_v_err) + wU * float(pinn_u_err) + wMu * mu_term
                    # self._maybe_update_top3_eval(it=it, score=pinn_score, tag="pinn", topk_list=self.topk_pinn)

                    # ==========================
                    # [NEW CKPT LOGIC - PINN only]
                    # ==========================
                    # Save Top-3 PINN checkpoints under three different value/control weightings.
                    # Lower score is better.
                    v = float(pinn_v_err)
                    u = float(pinn_u_err)
                    score_avg = 0.5 * (u + v)          # (control + value) / 2
                    score_u1_v02 = u + 0.2 * v       # control + value / 5
                    score_u0p2_v1 = 0.2 * u + v      # control / 5 + value

                    # A) keep historical tag="pinn"
                    self._maybe_update_top3_eval(it=it, score=score_avg, tag="pinn", topk_list=self.topk_pinn)
                    # B/C) separate tags => separate alias files + meta JSON
                    self._maybe_update_top3_eval(it=it, score=score_u1_v02, tag="pinn_u1_v0p2", topk_list=self.topk_pinn_u1_v0p2)
                    self._maybe_update_top3_eval(it=it, score=score_u0p2_v1, tag="pinn_u0p2_v1", topk_list=self.topk_pinn_u0p2_v1)
                if (taylor_v_err is not None and taylor_u_err is not None
                        and math.isfinite(taylor_v_err) and math.isfinite(taylor_u_err)):
                    # ==========================
                    # [OLD CKPT LOGIC - commented]
                    # ==========================
                    # w = self.params_train.get("ckpt_score_weights", None) or {}
                    # wJ = float(w.get("J_error", self.params_train.get("ckpt_w_J_error", 0.5)))
                    # wU = float(w.get("control_error", self.params_train.get("ckpt_w_control_error", 0.5)))
                    # wMu = float(w.get("mu_gaussian_policy_error", self.params_train.get("ckpt_w_mu_gaussian_policy_error", 0.0)))
                    # mu_term = (float(taylor_mu_err) if (taylor_mu_err is not None and math.isfinite(taylor_mu_err)) else 0.0)
                    # taylor_score = wJ * float(taylor_v_err) + wU * float(taylor_u_err) + wMu * mu_term
                    # self._maybe_update_top3_eval(it=it, score=taylor_score, tag="taylor", topk_list=self.topk_taylor)

                    # [NEW] Do not save Taylor ckpts by default (PINN-only selection).
                    pass
            def _csv_cell(value):
                if value is None:
                    return ""
                return value
            with open(loss_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{it},{to_scalar(elapsed):.6f},"
                    f"{to_scalar(loss_critic_final):.8e},{to_scalar(loss_actor_final):.8e},"
                    f"{_csv_cell(pinn_v_err)},{_csv_cell(pinn_u_err)},"
                    f"{_csv_cell(taylor_v_err)},{_csv_cell(taylor_u_err)}\n"
                )
            self.J_list.append(Y_pred.detach().clone())
            # NOTE: these are training-time diagnostics; they may be NaN if exact metrics are disabled.
            self.Y_error_list.append(Y_error.detach())
            self.control_error_list.append(control_error.detach())

            if it % 2 == 0:
                print("action_buffer.mean() = ", action_buffer.mean())

                yerr = to_scalar(Y_error)
                uerr = to_scalar(control_error)
                muerr = to_scalar(mu_of_policy_error_final) if (mu_of_policy_error_final is not None) else None
                if (not math.isfinite(yerr)) or (not math.isfinite(uerr)):
                    # Exact metrics disabled in-train; rely on eval metrics printed every `eval_every`.
                    if muerr is None or (not math.isfinite(float(muerr))):
                        mu_msg = "not evaluated"
                    else:
                        mu_msg = f"{100.0 * float(muerr):.3f}%"
                    print(
                        "It: %d, Time: %.2f, Loss_critic: %.3e, Loss_actor: %.3e, "
                        "J error: not evaluated, control error: not evaluated, mu_of_policy_error: %s" % (
                            int(it),
                            to_scalar(elapsed),
                            to_scalar(loss_critic_final),
                            to_scalar(loss_actor_final),
                            mu_msg,
                        )
                    )
                else:
                    if muerr is None or (not math.isfinite(float(muerr))):
                        mu_msg = "not evaluated"
                    else:
                        mu_msg = f"{100.0 * float(muerr):.3f}%"
                    msg = (
                        "It: %d, Time: %.2f, Loss_critic: %.3e, Loss_actor: %.3e, "
                        "J error: %.3f%%, control error: %.3f%%, mu_of_policy_error: %s" % (
                            int(it),
                            to_scalar(elapsed),
                            to_scalar(loss_critic_final),
                            to_scalar(loss_actor_final),
                            100.0 * yerr,
                            100.0 * uerr,
                            mu_msg,
                        )
                    )
                    print(msg)

            if it % 50 == 0:
                self.get_memory("GB")

        print("The training is over!")
        # ==========================
        # [OLD SUMMARY - commented]
        # ==========================
        # if len(self.topk_pinn) > 0:
        #     best = sorted(self.topk_pinn, key=lambda d: d["score"])[0]
        #     print(f"[summary] pinn top1: iter={best['iter']}, score={best['score']:.6g}")
        # else:
        #     print("[summary] no pinn Top-3 recorded.")
        # if len(self.topk_taylor) > 0:
        #     best = sorted(self.topk_taylor, key=lambda d: d["score"])[0]
        #     print(f"[summary] taylor top1: iter={best['iter']}, score={best['score']:.6g}")
        # else:
        #     print("[summary] no taylor Top-3 recorded.")

        # ==========================
        # [NEW SUMMARY - PINN only]
        # ==========================
        def _print_top1(name: str, li: list):
            if li:
                best = sorted(li, key=lambda d: d["score"])[0]
                print(f"[summary] {name} top1: iter={best['iter']}, score={best['score']:.6g}")
            else:
                print(f"[summary] no {name} Top-3 recorded.")

        _print_top1("pinn(score=(u+v)/2)", self.topk_pinn)
        _print_top1("pinn(score=u+v/5)", self.topk_pinn_u1_v0p2)
        _print_top1("pinn(score=u/5+v)", self.topk_pinn_u0p2_v1)

        # Finish wandb run
        if _wandb_run_active(self):
            wandb.finish()
            self._wandb_enabled = False
            print("[wandb] Logging finished.")

    def compute_control_error_with_fallback(
        self,
        control,
        policy_exact,
        mu_pred,
        sigma_pred,
        mu_exact,
        sigma_exact,
        M,
        D,
        control_samples=None,
    ):
        from tools_pinn0201_optimizer import evaluation as eval_tools
        return eval_tools.compute_control_error_with_fallback(
            self,
            control=control,
            policy_exact=policy_exact,
            mu_pred=mu_pred,
            sigma_pred=sigma_pred,
            mu_exact=mu_exact,
            sigma_exact=sigma_exact,
            M=M,
            D=D,
            control_samples=control_samples,
            actor_type_u_space=("hyper_gaussian", "hyper_gaussian_split", "gaussian_two_stage"),
            skip_actor_types=("mog_squash",),
        )

    @torch.no_grad()
    def _plot_trajs_for_current_weights(self, filename_suffix: str = "", device=None,
                                        downsample_step=5, smooth_window=1, plot_num=5, plot_num_u=1,
                                        plot_traj_mode: str = None, N_eval: int = None):
        # Re-apply CLI overrides to survive params reloads from ckpt dirs / predict().
        if getattr(self, "_cli_plot_traj_mode", None) is not None:
            self.params_train["plot_traj_mode"] = getattr(self, "_cli_plot_traj_mode")
        if getattr(self, "_cli_plot_control_mean_kind", None) is not None:
            self.params_train["plot_control_mean_kind"] = getattr(self, "_cli_plot_control_mean_kind")
        if getattr(self, "_cli_K_mean_eval_plot", None) is not None:
            self.params_train["K_mean_eval_plot"] = int(getattr(self, "_cli_K_mean_eval_plot"))
        if getattr(self, "_cli_plot_mixture_mean_use_fixed_seed", None) is not None:
            self.params_train["plot_mixture_mean_use_fixed_seed"] = bool(getattr(self, "_cli_plot_mixture_mean_use_fixed_seed"))
        if getattr(self, "_cli_plot_mixture_mean_fixed_seed", None) is not None:
            self.params_train["plot_mixture_mean_fixed_seed"] = int(getattr(self, "_cli_plot_mixture_mean_fixed_seed"))
        # Panel-c overrides (conditional distribution / evolved X_T distribution)
        # These can be overwritten by params.json reloads during eval; keep CLI intent sticky.
        if getattr(self, "_cli_plot_panel_c", None) is not None:
            self.params_train["plot_panel_c"] = bool(getattr(self, "_cli_plot_panel_c"))
        if getattr(self, "_cli_panel_c_mode", None) is not None:
            self.params_train["panel_c_mode"] = str(getattr(self, "_cli_panel_c_mode"))
        if getattr(self, "_cli_combined_layout", None) is not None:
            self.params_train["combined_layout"] = str(getattr(self, "_cli_combined_layout"))
        if getattr(self, "_cli_panel_a_traj_selector", None) is not None:
            self.params_train["panel_a_traj_selector"] = str(getattr(self, "_cli_panel_a_traj_selector"))
        if getattr(self, "_cli_panel_a_fixed_x", None) is not None:
            self.params_train["panel_a_fixed_x"] = float(getattr(self, "_cli_panel_a_fixed_x"))
        if getattr(self, "_cli_panel_c_n_samples_u", None) is not None:
            self.params_train["panel_c_n_samples_u"] = int(getattr(self, "_cli_panel_c_n_samples_u"))
        if getattr(self, "_cli_panel_c_plot_method", None) is not None:
            self.params_train["panel_c_plot_method"] = str(getattr(self, "_cli_panel_c_plot_method"))
        if getattr(self, "_cli_panel_c_x_value", None) is not None:
            self.params_train["panel_c_x_value"] = float(getattr(self, "_cli_panel_c_x_value"))
        if getattr(self, "_cli_panel_c_t_value", None) is not None:
            self.params_train["panel_c_t_value"] = float(getattr(self, "_cli_panel_c_t_value"))
        self._ensure_eval_set(N_eval=N_eval)
        device = device or self.device
        t_test = self.t_eval
        W_test, P_test   = self.W_eval, self.P_eval

        (
            X_pred, Y_pred, control_pred_sample,
            X_test_pinn, Y_test_pinn, control_pinn_sample,
            X_test_taylor, Y_test_taylor, control_taylor_sample,
        ) = self.predict(t_test, W_test, P_test, N_eval=N_eval)

        # Keep full-eval copies for computing stable sample-mean curves.
        # We may slice tensors to `plot_num` for memory-safe plotting below, but for `sample_mean`
        # we usually want the mean over the FULL eval batch, not just the plotted subset.
        control_pred_sample_all = control_pred_sample
        control_pinn_sample_all = control_pinn_sample
        control_taylor_sample_all = control_taylor_sample

        # predict() may reload ckpt params and overwrite runtime flags; enforce CLI overrides again.
        if getattr(self, "_cli_plot_traj_mode", None) is not None:
            self.params_train["plot_traj_mode"] = getattr(self, "_cli_plot_traj_mode")
        if getattr(self, "_cli_plot_control_mean_kind", None) is not None:
            self.params_train["plot_control_mean_kind"] = getattr(self, "_cli_plot_control_mean_kind")
        if getattr(self, "_cli_K_mean_eval_plot", None) is not None:
            self.params_train["K_mean_eval_plot"] = int(getattr(self, "_cli_K_mean_eval_plot"))
        if getattr(self, "_cli_plot_mixture_mean_use_fixed_seed", None) is not None:
            self.params_train["plot_mixture_mean_use_fixed_seed"] = bool(getattr(self, "_cli_plot_mixture_mean_use_fixed_seed"))
        if getattr(self, "_cli_plot_mixture_mean_fixed_seed", None) is not None:
            self.params_train["plot_mixture_mean_fixed_seed"] = int(getattr(self, "_cli_plot_mixture_mean_fixed_seed"))
        if getattr(self, "_cli_plot_panel_c", None) is not None:
            self.params_train["plot_panel_c"] = bool(getattr(self, "_cli_plot_panel_c"))
        if getattr(self, "_cli_panel_c_mode", None) is not None:
            self.params_train["panel_c_mode"] = str(getattr(self, "_cli_panel_c_mode"))
        if getattr(self, "_cli_combined_layout", None) is not None:
            self.params_train["combined_layout"] = str(getattr(self, "_cli_combined_layout"))
        if getattr(self, "_cli_panel_a_traj_selector", None) is not None:
            self.params_train["panel_a_traj_selector"] = str(getattr(self, "_cli_panel_a_traj_selector"))
        if getattr(self, "_cli_panel_a_fixed_x", None) is not None:
            self.params_train["panel_a_fixed_x"] = float(getattr(self, "_cli_panel_a_fixed_x"))
        if getattr(self, "_cli_panel_c_n_samples_u", None) is not None:
            self.params_train["panel_c_n_samples_u"] = int(getattr(self, "_cli_panel_c_n_samples_u"))
        if getattr(self, "_cli_panel_c_plot_method", None) is not None:
            self.params_train["panel_c_plot_method"] = str(getattr(self, "_cli_panel_c_plot_method"))
        if getattr(self, "_cli_panel_c_x_value", None) is not None:
            self.params_train["panel_c_x_value"] = float(getattr(self, "_cli_panel_c_x_value"))
        if getattr(self, "_cli_panel_c_t_value", None) is not None:
            self.params_train["panel_c_t_value"] = float(getattr(self, "_cli_panel_c_t_value"))

        # --- Memory safety for plotting ---
        # By default, only evaluate/plot on the first `plot_num` trajectories.
        # This avoids OOM when N_eval is large (e.g. 1000) and we would otherwise run mixture_mean
        # on M_eval*(N_eval+1) states at once.
        use_plot_subset = bool(self.params_train.get("plot_eval_use_plot_num_only", False))
        B_all = int(t_test.shape[0]) if (t_test is not None and hasattr(t_test, "shape")) else None

        if use_plot_subset and B_all is not None and B_all > 0:
            try:
                k = min(int(plot_num), B_all)
            except Exception:
                k = B_all
            if k is not None and k > 0 and k < B_all:
                # Slice time grids to stay consistent with sliced trajectories.
                try:
                    t_test = t_test[:k]
                    W_test = W_test[:k] if W_test is not None else W_test
                    P_test = P_test[:k] if P_test is not None else P_test
                except Exception:
                    pass

                def _slice0(x):
                    try:
                        if torch.is_tensor(x) and x.dim() >= 1 and int(x.shape[0]) == int(B_all):
                            return x[:k]
                    except Exception:
                        return x
                    return x

                X_pred = _slice0(X_pred)
                Y_pred = _slice0(Y_pred)
                control_pred_sample = _slice0(control_pred_sample)
                X_test_pinn = _slice0(X_test_pinn)
                Y_test_pinn = _slice0(Y_test_pinn)
                control_pinn_sample = _slice0(control_pinn_sample)
                X_test_taylor = _slice0(X_test_taylor)
                Y_test_taylor = _slice0(Y_test_taylor)
                control_taylor_sample = _slice0(control_taylor_sample)

        # --- Domain sanity: warn when plotted trajectories leave PINN cache grid ---
        try:
            x_pred_min = float(torch.nan_to_num(X_pred, nan=math.inf).min().detach().cpu().item())
            x_pred_max = float(torch.nan_to_num(X_pred, nan=-math.inf).max().detach().cpu().item())
            x_lo = float(self.params_train.get("pinn_x_min", -math.inf))
            # Prefer eval-domain bound if provided; otherwise fall back to training bound.
            x_hi = self.params_train.get("pinn_x_eval_max", None)
            if x_hi is None:
                x_hi = self.params_train.get("pinn_x_max", math.inf)
            x_hi = float(x_hi)
            tol = 1e-6
            if (x_pred_min < x_lo - tol) or (x_pred_max > x_hi + tol):
                print(
                    "[warn][plot] X_pred range exceeds PINN cache grid; "
                    "PINN 'exact' curves may be extrapolated/invalid.\n"
                    f"  X_pred_range=[{x_pred_min:.4g}, {x_pred_max:.4g}], pinn_x_range=[{x_lo:.4g}, {x_hi:.4g}]"
                )
        except Exception:
            pass

        plot_traj_mode = (plot_traj_mode or self.plot_traj_mode or
                          self.params_train.get("plot_traj_mode", "abc")).lower()
        if plot_traj_mode not in ("ab", "ac", "bc", "abc"):
            print(f"[warn] unknown plot_traj_mode={plot_traj_mode}, fallback to 'abc'")
            plot_traj_mode = "abc"

        gamma_entropy = self.params_train["entropy_alpha"]
        use_taylor = (plot_traj_mode in ("ab", "abc")) and (gamma_entropy > 0)

        # ------------------------------------------------------------------
        # Subplot-1 (policy) should compare functions on the SAME states.
        # BUT: PINN evaluation clamps x into [pinn_x_min, pinn_x_max] for validity, so if X_pred
        # leaves this domain, "PINN policy on X_pred" becomes "PINN policy on clamp(X_pred)".
        # To make the comparison fair and remove artificial mean gaps, optionally clamp X_pred
        # for BOTH PINN and FlowNet evaluations in subplot-1.
        # (Trajectories in subplot-3 remain unchanged; this only affects function evaluation.)
        # ------------------------------------------------------------------
        X_pred_for_policy_eval = X_pred
        try:
            if bool(self.params_train.get("plot_policy_eval_clamp_to_pinn_domain", True)):
                x_lo = float(self.params_train.get("pinn_x_min", -math.inf))
                x_hi = self.params_train.get("pinn_x_eval_max", None)
                if x_hi is None:
                    x_hi = self.params_train.get("pinn_x_max", math.inf)
                x_hi = float(x_hi)
                if np.isfinite(x_lo) or np.isfinite(x_hi):
                    X_pred_for_policy_eval = torch.clamp(X_pred, min=x_lo, max=x_hi)
        except Exception:
            X_pred_for_policy_eval = X_pred

        ###### 1. Learned policy ######
        actor_type = getattr(self.net_actor, "actor_type", "")
        if actor_type in ("gaussian_two_stage"):
            state_pred = self._preprocess_state(t_test, X_pred_for_policy_eval)
            # mu_pred, log_sigma_pred, feat_pred = self.net_actor(state_pred)
            # Reshape for sample: [batch, time, state_dim] -> [batch*time, state_dim]
            batch_size, time_steps, state_dim = state_pred.shape
            state_pred_flat = state_pred.reshape(-1, state_dim)

            # === Diagnostic: Check if state (raw x) and feature (preprocessed x) are varying correctly ===
            # NOTE: `mean_mixture_pred` is evaluated on `X_pred_for_policy_eval` (possibly clamped),
            # so diagnostics should use the same states for consistency.
            X_pred_flat = X_pred_for_policy_eval.reshape(-1, X_pred_for_policy_eval.shape[-1])  # [M*N, D]
            feat_x_component = state_pred_flat[:, 1] if state_dim > 1 else state_pred_flat[:, 0]  # Second dim is x-feature
            # print(f"[plot_diagnostic] X_pred shape: {X_pred.shape}, min/max/std: "
            #       f"{X_pred.min().item():.6f}/{X_pred.max().item():.6f}/{X_pred.std().item():.6f}")
            # print(f"[plot_diagnostic] state feature x-component (e.g. log(x)) min/max/std: "
            #       f"{feat_x_component.min().item():.6f}/{feat_x_component.max().item():.6f}/{feat_x_component.std().item():.6f}")

            # Plot-time mean-of-policy estimation:
            # - Use a dedicated (potentially larger) MC sample size for stability.
            # - Optionally fix RNG seed so the plotted mean curve is reproducible and not a noisy artifact.
            K_mean_eval_train = int(self.params_train["K_mean_eval"])
            K_mean_eval_plot = int(self.params_train.get("K_mean_eval_plot", K_mean_eval_train))
            use_fixed_seed = bool(self.params_train.get("plot_mixture_mean_use_fixed_seed",
                                                       self.params_train.get("mixture_mean_use_fixed_seed", True)))
            fixed_seed = int(self.params_train.get("plot_mixture_mean_fixed_seed",
                                                  self.params_train.get("mixture_mean_fixed_seed", 123456)))
            mm_kwargs = dict(n_samples=K_mean_eval_plot, antithetic=True)
            if use_fixed_seed:
                mm_kwargs["seed"] = fixed_seed
            # OOM guard: compute mixture_mean in chunks when evaluating many states.
            # Default chunk size scales inversely with n_samples.
            try:
                chunk_size = self.params_train.get("plot_mixture_mean_chunk_size", None)
                if chunk_size is not None:
                    chunk_size = int(chunk_size)
            except Exception:
                chunk_size = None
            if chunk_size is None or chunk_size <= 0:
                # Heuristic: keep (chunk_size * n_samples) bounded.
                # Feel free to override with params_train["plot_mixture_mean_chunk_size"].
                chunk_size = max(256, int(16384 // max(1, int(K_mean_eval_plot))))

            def _mixture_mean_one(x_chunk, kwargs):
                try:
                    return self.net_actor.mixture_mean(x_chunk, **kwargs)
                except TypeError:
                    # Backward-compatible: older `mixture_mean` may not accept `seed`
                    kwargs = dict(kwargs)
                    kwargs.pop("seed", None)
                    return self.net_actor.mixture_mean(x_chunk, **kwargs)

            if int(state_pred_flat.shape[0]) <= int(chunk_size):
                mean_mixture_pred_flat = _mixture_mean_one(state_pred_flat, mm_kwargs)
            else:
                outs = []
                base_seed = mm_kwargs.get("seed", None)
                for ci, s in enumerate(range(0, int(state_pred_flat.shape[0]), int(chunk_size))):
                    e = min(int(state_pred_flat.shape[0]), s + int(chunk_size))
                    kwargs_i = dict(mm_kwargs)
                    if base_seed is not None:
                        # Ensure different chunks don't reuse identical RNG streams.
                        kwargs_i["seed"] = int(base_seed) + int(ci)
                    outs.append(_mixture_mean_one(state_pred_flat[s:e], kwargs_i))
                mean_mixture_pred_flat = torch.cat(outs, dim=0)
            mean_mixture_pred = mean_mixture_pred_flat.reshape(batch_size, time_steps, -1)

            # === Diagnostic: Check mean_mixture_pred shape and variation ===
            print(f"[plot_diagnostic] mean_mixture_pred shape: {mean_mixture_pred.shape}")
            print(f"[plot_diagnostic] mean_mixture_pred min/max/std: "
                  f"{mean_mixture_pred.min().item():.6f}/{mean_mixture_pred.max().item():.6f}/{mean_mixture_pred.std().item():.6f}")
            # Check if mean varies across trajectories at a fixed time
            if mean_mixture_pred.shape[0] > 1:
                mean_at_t0 = mean_mixture_pred[:, 0, :]  # [M, action_dim]
                print(f"[plot_diagnostic] mean_mixture_pred at t=0 across trajectories: "
                      f"min={mean_at_t0.min().item():.6f}, max={mean_at_t0.max().item():.6f}, "
                      f"std={mean_at_t0.std().item():.6f}")
        else:
            state_pred = self._preprocess_state(t_test, X_pred_for_policy_eval)
            # === Diagnostic: Check if state (raw x) and feature (preprocessed x) are varying correctly ===
            feat_x_component = state_pred[:, :, 1] if state_pred.shape[2] > 1 else state_pred[:, :, 0]
            print(f"[plot_diagnostic] X_pred shape: {X_pred.shape}, min/max/std: "
                  f"{X_pred.min().item():.6f}/{X_pred.max().item():.6f}/{X_pred.std().item():.6f}")
            print(f"[plot_diagnostic] state feature x-component (e.g. log(x)) min/max/std: "
                  f"{feat_x_component.min().item():.6f}/{feat_x_component.max().item():.6f}/{feat_x_component.std().item():.6f}")
            # Warn based on raw x variance (feature variance can be small if x is near-constant).
            if X_pred.std().item() < 1e-6:
                print(f"[plot_diagnostic] WARNING: raw X_pred std={X_pred.std().item():.6e} is too small! "
                      f"Actor is effectively seeing near-constant state (in raw x).")

            mu_pred, log_sigma_pred, _ = self.net_actor(state_pred)
            # sigma_pred = torch.exp(log_sigma_pred)
            mean_mixture_pred = mu_pred

            # === Diagnostic: Check mean_mixture_pred shape and variation ===
            print(f"[plot_diagnostic] mean_mixture_pred shape: {mean_mixture_pred.shape}")
            print(f"[plot_diagnostic] mean_mixture_pred min/max/std: "
                  f"{mean_mixture_pred.min().item():.6f}/{mean_mixture_pred.max().item():.6f}/{mean_mixture_pred.std().item():.6f}")
            # Check if mean varies across trajectories at a fixed time
            if mean_mixture_pred.shape[0] > 1:
                mean_at_t0 = mean_mixture_pred[:, 0, :]  # [M, action_dim]
                print(f"[plot_diagnostic] mean_mixture_pred at t=0 across trajectories: "
                      f"min={mean_at_t0.min().item():.6f}, max={mean_at_t0.max().item():.6f}, "
                      f"std={mean_at_t0.std().item():.6f}")

        u_exact = None
        value_exact = None
        # Plot comparison mode:
        # - If plot_value_state_same_traj=True (default), compare VALUE/STATE on the SAME trajectory (X_pred).
        #   This makes subplot 2/3 a "function comparison" (v evaluated on the same states),
        #   rather than a closed-loop rollout comparison across different controls.
        # - If False, keep the original closed-loop behavior: PINN/Taylor values and states on their own trajectories.
        plot_value_state_same_traj = bool(self.params_train.get("plot_value_state_same_traj", True))

        # For subplot 1 (policy): evaluate PINN/Taylor on the SAME states as actor (possibly clamped).
        X_exact_for_eval = X_pred_for_policy_eval
        X_exact_for_plot = X_test_pinn  # Keep original PINN trajectory for state plot (subplot 3)
        u_taylor = None
        value_taylor = None
        value_taylor_for_plot = None  # Will be set to Y_test_taylor if available
        X_taylor_for_eval = X_pred_for_policy_eval
        X_taylor_for_plot = X_test_taylor  # Keep original Taylor trajectory for state plot (subplot 3)

        ###### 2. Exact policy by PINN ######
        if gamma_entropy == 0:
            plot_traj_mode = "ac"
            u_exact, value_exact = self.compute_exact_no_exploratory(X_exact_for_eval)
            u_exact = u_exact * torch.ones_like(X_exact_for_eval)
            # Y_test_pinn is already computed in predict() and represents PINN value on X_test_pinn trajectory
            value_exact_on_pinn_traj = Y_test_pinn
        else:
            # Compute PINN u* on X_pred (same as network) for function comparison (subplot 1)
            u_mean_star, u_std_star, value_exact, u_hard_star, _pol = self.compute_exact_entropy_with_cache(
                X_exact_for_eval, return_u_hard=True
            )
            u_exact = u_mean_star
            # Y_test_pinn is already computed in predict() and represents PINN value on X_test_pinn trajectory
            # Use it directly for closed-loop comparison in subplot 2
            value_exact_on_pinn_traj = Y_test_pinn

        ###### 3. Exact policy by Taylor ######
        if use_taylor:
            def _taylor_mean_control(x):
                if self.p == 1/2:
                    x = torch.clamp(x, min=1e-8)
                mu = torch.as_tensor(self.params_equ["mu"], device=x.device, dtype=x.dtype)
                r = torch.as_tensor(self.params_equ["r"], device=x.device, dtype=x.dtype)
                sigma = torch.as_tensor(self.params_equ["sigma"], device=x.device, dtype=x.dtype)
                lam = torch.as_tensor(self.params_equ["lambda"], device=x.device, dtype=x.dtype)
                z = torch.as_tensor(self.params_equ["z"], device=x.device, dtype=x.dtype)
                B = sigma**2 + lam * z**2
                C = self.h_t.to(x.device) * (x**self.p)
                div_term = B * ((1 - self.p) * C + self.a_t)
                div_term = torch.clamp(div_term, min=1e-8)
                return ((mu - r) * (C + self.a_t)) / div_term

            # IMPORTANT: Evaluate Taylor policy on X_pred (subplot 1) for function comparison.
            # For subplot 2, we can either:
            # - function-compare on the SAME states (X_pred) via `value_taylor`, or
            # - keep closed-loop comparison via `Y_test_taylor` (Taylor trajectory).
            policy_taylor, value_taylor = self.compute_exact_exploratory_taylor_and_ansatz(X_taylor_for_eval)
            u_taylor = policy_taylor.loc
            if plot_value_state_same_traj:
                # Same-trajectory function evaluation: v(t, X_pred)
                value_taylor_for_plot = value_taylor
            else:
                # Closed-loop comparison: v(t, X_test_taylor)
                value_taylor_for_plot = Y_test_taylor if Y_test_taylor is not None else value_taylor

        if value_exact is not None:
            v_pinn_mean = float(value_exact.mean().detach().cpu().item())
            y_pred_mean = float(Y_pred.mean().detach().cpu().item())
            x_min = float(X_pred.min().detach().cpu().item())
            x_max = float(X_pred.max().detach().cpu().item())
            v_taylor_mean = None if value_taylor is None else float(value_taylor.mean().detach().cpu().item())
            print(
                f"[plot] v_pinn_mean={v_pinn_mean:.6f}, y_pred_mean={y_pred_mean:.6f}, "
                f"v_taylor_mean={v_taylor_mean}, x_pred_range=[{x_min:.4f},{x_max:.4f}]"
            )

        # === Additional diagnostic: Check state-dependency at fixed time ===
        # As suggested in the images, check if mean varies with x at a fixed time t0
        if mean_mixture_pred.shape[0] > 1 and mean_mixture_pred.shape[1] > 1:
            t0_idx = mean_mixture_pred.shape[1] // 2  # Middle time point
            # Use the SAME X that was used for policy evaluation (possibly clamped).
            x_at_t0 = X_pred_for_policy_eval[:, t0_idx, :]  # [M, D] - different x values at time t0
            mean_at_t0 = mean_mixture_pred[:, t0_idx, :]  # [M, action_dim] - mean policy at time t0
            if x_at_t0.shape[1] == 1:
                x_at_t0_flat = x_at_t0[:, 0]
                mean_at_t0_flat = mean_at_t0[:, 0] if mean_at_t0.shape[1] == 1 else mean_at_t0
                print(f"[plot_diagnostic] At fixed time t0 (idx={t0_idx}):")
                print(f"  X values: min={x_at_t0_flat.min().item():.6f}, max={x_at_t0_flat.max().item():.6f}, "
                      f"std={x_at_t0_flat.std().item():.6f}")
                print(f"  Mean policy values: min={mean_at_t0_flat.min().item():.6f}, "
                      f"max={mean_at_t0_flat.max().item():.6f}, std={mean_at_t0_flat.std().item():.6f}")
                if x_at_t0_flat.std().item() > 1e-6 and mean_at_t0_flat.std().item() < 1e-6:
                    print(f"[plot_diagnostic] WARNING: X varies but mean policy is constant! "
                          f"This suggests actor is not using state information correctly.")
                elif x_at_t0_flat.std().item() < 1e-6:
                    print(f"[plot_diagnostic] WARNING: X is constant at t0! Cannot check state-dependency.")

        t_np = t_test.cpu().detach().numpy()
        Xp_np = X_pred.cpu().detach().numpy()
        Xt_np = X_exact_for_plot.cpu().detach().numpy()  # PINN trajectory for state plot (subplot 3)

        # === Diagnostic: Check if X_pred and X_test_pinn are the same (they should be different) ===
        X_pred_np = X_pred.cpu().detach().numpy()
        X_test_pinn_np = X_test_pinn.cpu().detach().numpy()
        if X_pred_np.shape == X_test_pinn_np.shape:
            diff = np.abs(X_pred_np - X_test_pinn_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"[plot_diagnostic] X_pred vs X_test_pinn: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            if max_diff < 1e-6:
                print(f"[plot_diagnostic] WARNING: X_pred and X_test_pinn are identical! This suggests they use the same control policy.")
            else:
                print(f"[plot_diagnostic] X_pred and X_test_pinn are different (as expected - different control policies).")
        else:
            print(f"[plot_diagnostic] X_pred shape {X_pred_np.shape} != X_test_pinn shape {X_test_pinn_np.shape}")
        Yp_np = Y_pred.cpu().detach().numpy()  # Network value on X_pred trajectory (subplot 2, b)
        # For subplot 2 (value): compare on SAME states by default (X_pred),
        # to avoid conflating value-function mismatch with different state distributions.
        if plot_value_state_same_traj:
            Y_exact_np = value_exact.cpu().detach().numpy() if value_exact is not None else None  # PINN v(t, X_pred)
        else:
            Y_exact_np = value_exact_on_pinn_traj.cpu().detach().numpy() if value_exact_on_pinn_traj is not None else None  # PINN v(t, X_test_pinn)
        # For combined plots, ALWAYS compare on the same trajectory (X_pred).
        Y_exact_same_traj_np = value_exact.cpu().detach().numpy() if value_exact is not None else None
        mu_np = mean_mixture_pred.cpu().detach().numpy()  # Network mean policy on X_pred (subplot 1, b)

        # === Diagnostic: Check mu_np shape before passing to plotting function ===
        # print(f"[plot_diagnostic] mu_np shape before plotting: {mu_np.shape}") #(100, 101, 1)
        print(f"[plot_diagnostic] mu_np should be [M, N+1, action_dim] where M=trajectories, N+1=time steps")
        if len(mu_np.shape) != 3:
            print(f"[plot_diagnostic] WARNING: mu_np has unexpected shape {mu_np.shape}, expected 3D!")
        u_exact_np = u_exact.cpu().detach().numpy() if u_exact is not None else None  # PINN u* on X_pred (subplot 1, a)
        u_taylor_np = u_taylor.cpu().detach().numpy() if u_taylor is not None else None  # Taylor u* on X_pred (subplot 1, c)
        # For subplot 2, value_taylor_for_plot has been selected based on plot_value_state_same_traj.
        if use_taylor and value_taylor_for_plot is not None:
            Y_taylor_np = value_taylor_for_plot.cpu().detach().numpy()
        elif value_taylor is not None:
            Y_taylor_np = value_taylor.cpu().detach().numpy()
        else:
            Y_taylor_np = None
        Xt_taylor_np = X_taylor_for_plot.cpu().detach().numpy()  # Taylor trajectory for state plot (subplot 3)

        # Auto-append checkpoint iteration tag (e.g., `_itr2000`) when evaluating from a given ckpt path.
        # This helps avoid overwriting plots when `filename_suffix` is empty (common in eval-only scripts).
        ckpt_itr_suffix = ""
        try:
            # If user already provided an itr tag in filename_suffix, do not add another.
            if not re.search(r"_itr\d+", str(filename_suffix or "")):
                cand_paths = []
                if getattr(self, "actor_path", None):
                    cand_paths.append(self.actor_path)
                if getattr(self, "critic_path", None):
                    cand_paths.append(self.critic_path)
                itr_match = None
                for p in cand_paths:
                    if not p:
                        continue
                    p_str = str(p)
                    itr_match = re.search(r"itr(\d+)", os.path.basename(p_str)) or re.search(r"itr(\d+)", p_str)
                    if itr_match is not None:
                        break
                if itr_match is not None:
                    ckpt_itr_suffix = f"_itr{itr_match.group(1)}"
        except Exception:
            ckpt_itr_suffix = ""

        def _fn(base):
            """Build plot filename (without extension)."""
            # Add control range (u_min/u_max) to filename for easier bookkeeping
            def _fmt_num_for_fname(x):
                if x is None:
                    return None
                try:
                    xf = float(x)
                    if not np.isfinite(xf):
                        return str(x)
                    # Compact, stable string (avoid long tails like 0.30000000004)
                    return f"{xf:.6g}"
                except Exception:
                    return str(x)

            u_min = self.params_train.get("action_min", self.params_train.get("pinn_u_min", None))
            u_max = self.params_train.get("action_max", self.params_train.get("pinn_u_max", None))
            u_min_s = _fmt_num_for_fname(u_min)
            u_max_s = _fmt_num_for_fname(u_max)
            u_suffix = f"_umin{u_min_s}_umax{u_max_s}" if (u_min_s is not None and u_max_s is not None) else ""

            # Add plot_traj_mode to filename
            mode_suffix = f"_{plot_traj_mode}" if plot_traj_mode else ""
            fraction = getattr(self, "_cli_policy_plot_time_fraction", None)
            if fraction is None:
                fraction = self.params_train.get("policy_plot_time_fraction", 0.25)
            policy_suffix = "_policy_Tdiv2" if float(fraction) == 0.5 else "_policy_Tdiv4"
            # Panel (a) at fixed x (cond_u_at_fixed_point) -> add _fixedx<value> to filename
            panel_a_sel = str(self.params_train.get("panel_a_traj_selector", ""))
            if panel_a_sel == "cond_u_at_fixed_point":
                x_fixed = float(self.params_train.get("panel_a_fixed_x", 10.0))
                panel_a_fixedx_suffix = "_fixedx{:g}".format(x_fixed)  # 10.0 -> fixedx10, 15.5 -> fixedx15.5
            else:
                panel_a_fixedx_suffix = ""
            return f"{base}{u_suffix}{mode_suffix}{policy_suffix}{panel_a_fixedx_suffix}{filename_suffix}{ckpt_itr_suffix}"

        # ===== DEFINITIONS (CONSISTENT ACROSS ALL PLOT MODES) =====
        # Comparison-mode names below: a = PINN, b = network, c = Taylor.
        # These letters select benchmarks, independently of subplot labels (a,b,c).
        # a = exact by PINN (always)
        # b = approximated by network (always)
        # c = exact by Taylor/Ansatz (always)

        # Subplot 1 (policy): Compare functions on SAME states X_pred
        #   - b: network policy π̂(t, X_pred) on X_pred trajectory
        #   - a: PINN policy u*(t, X_pred) on X_pred trajectory (function evaluation)
        #   - c: Taylor policy π*(t, X_pred) on X_pred trajectory (function evaluation)

        # Subplot 2 (value): Compare on SAME states for FlowNet vs PINN in combined plots
        #   - b: network value v̂(t, X_pred) on X_pred trajectory
        #   - a: PINN value v(t, X_pred) on X_pred trajectory
        #   - c: Taylor value follows plot_value_state_same_traj setting

        # Subplot 3 (state): Compare TRAJECTORIES
        #   - b: X_pred trajectory (network policy)
        #   - a: X_test_pinn trajectory (PINN policy)
        #   - c: X_test_taylor trajectory (Taylor policy)

        # ab = network + PINN, ac = network + Taylor, abc = all three, bc = network + Taylor
        if plot_traj_mode == "ab":
            # ab: network (b) + PINN (a)
            traj_u_test_np = u_exact_np  # PINN (a) policy on X_pred (subplot 1)
            traj_value_test_np = Y_exact_same_traj_np  # PINN (a) value on X_pred trajectory (subplot 2)
            traj_Xt_test_np = Xt_np  # PINN (a) trajectory (subplot 3)
            traj_u_third_np = None
            traj_value_third_np = None
            traj_Xt_third_np = None
            exact_u = "Mean of $u^*(t, \\hat{X}_t)$ (PINN)"  # PINN (a) on X_pred
            start_u = "Mean of $u^*(0, \\hat{X}_0)$ (PINN)"
            end_u = "Mean of $u^*(T, \\hat{X}_T)$ (PINN)"
            exact_v = "Benchmark $v(t, \\hat{X}_t)$ (PINN)"  # PINN (a) on X_pred trajectory
            start_v = "$v(0, \\hat{X}_0)$ (PINN)"
            end_v = "$v(T, \\hat{X}_T)$ (PINN)"
            exact_x = "$X_t$ by PINN $\\hat{u}$"  # PINN (a) trajectory
            third_u = third_v = third_x = None
        elif plot_traj_mode == "ac":
            # ac: network (b) + Taylor (c)
            traj_u_test_np = None  # No PINN
            traj_value_test_np = None  # No PINN
            traj_Xt_test_np = None  # No PINN
            traj_u_third_np = u_taylor_np  # Taylor (c)
            traj_value_third_np = Y_taylor_np  # Taylor (c)
            traj_Xt_third_np = Xt_taylor_np  # Taylor (c)
            exact_u = None
            start_u = None
            end_u = None
            exact_v = None
            start_v = None
            end_v = None
            exact_x = None
            third_u = "Taylor/Ansatz $\\pi^*(t, \\hat{X}_t)$"  # Taylor (c)
            third_v = "Taylor/Ansatz $v(t, \\hat{X}_t)$"  # Taylor (c)
            third_x = "$X_t$ by Taylor $\\pi^*(t, \\hat{X}_t)$"  # Taylor (c)
        elif plot_traj_mode == "abc":
            # abc: network (b) + PINN (a) + Taylor (c)
            traj_u_test_np = u_exact_np  # PINN (a) policy on X_pred (subplot 1)
            traj_value_test_np = Y_exact_same_traj_np  # PINN (a) value on X_pred trajectory (subplot 2)
            traj_Xt_test_np = Xt_np  # PINN (a) trajectory (subplot 3)
            traj_u_third_np = u_taylor_np  # Taylor (c) policy on X_pred (subplot 1)
            traj_value_third_np = Y_taylor_np  # Taylor (c) value on X_test_taylor trajectory (subplot 2, closed-loop)
            traj_Xt_third_np = Xt_taylor_np  # Taylor (c) trajectory (subplot 3)
            exact_u = "Mean of $u^*(t, \\hat{X}_t)$ (PINN)"  # PINN (a) on X_pred
            start_u = "Mean of $u^*(0, \\hat{X}_0)$ (PINN)"
            end_u = "Mean of $u^*(T, \\hat{X}_T)$ (PINN)"
            exact_v = "Benchmark $v(t, \\hat{X}_t)$ (PINN)"  # PINN (a) on X_pred trajectory
            start_v = "$v(0, \\hat{X}_0)$ (PINN)"
            end_v = "$v(T, \\hat{X}_T)$ (PINN)"
            exact_x = "$X_t$ by PINN $\\hat{u}$"  # PINN (a) trajectory
            third_u = "Taylor/Ansatz $\\pi^*(t, \\hat{X}_t)$"  # Taylor (c) on X_pred
            third_v = "Taylor/Ansatz $v(t, X_t)$"  # Taylor (c) on X_test_taylor trajectory
            third_x = "$X_t$ by Taylor $\\pi^*(t, \\hat{X}_t)$"  # Taylor (c) trajectory
        elif plot_traj_mode == "bc":
            # bc: network (b) + Taylor (c) (same as ac)
            traj_u_test_np = None  # No PINN
            traj_value_test_np = None  # No PINN
            traj_Xt_test_np = None  # No PINN
            traj_u_third_np = u_taylor_np  # Taylor (c)
            traj_value_third_np = Y_taylor_np  # Taylor (c)
            traj_Xt_third_np = Xt_taylor_np  # Taylor (c)
            exact_u = None
            start_u = None
            end_u = None
            exact_v = None
            start_v = None
            end_v = None
            exact_x = None
            third_u = "Taylor/Ansatz $\\pi^*(t, \\hat{X}_t)$"  # Taylor (c)
            third_v = "Taylor/Ansatz $v(t, \\hat{X}_t)$"  # Taylor (c)
            third_x = "$X_t$ by Taylor $\\pi^*(t, \\hat{X}_t)$"  # Taylor (c)
        else:
            # Default: only network (b), or network + PINN if available
            traj_u_test_np = u_exact_np if u_exact_np is not None else None
            traj_value_test_np = Y_exact_same_traj_np if Y_exact_same_traj_np is not None else None
            traj_Xt_test_np = Xt_np if Xt_np is not None else None
            traj_u_third_np = None
            traj_value_third_np = None
            traj_Xt_third_np = None
            exact_u = "Mean of $u^*(t, \\hat{X}_t)$" if u_exact_np is not None else None
            start_u = "Mean of $u^*(0, \\hat{X}_0)$" if u_exact_np is not None else None
            end_u = "Mean of $u^*(T, \\hat{X}_T)$" if u_exact_np is not None else None
            exact_v = "$v(t, \\hat{X}_t)$" if Y_exact_same_traj_np is not None else None
            start_v = "$v(0, \\hat{X}_0)$" if Y_exact_same_traj_np is not None else None
            end_v = "$v(T, \\hat{X}_T)$" if Y_exact_same_traj_np is not None else None
            exact_x = "$X_t$ by $\\hat{u}$" if Xt_np is not None else None
            third_u = third_v = third_x = None

        # Panel-c representative point (used by combined plot and standalone panel-c plots)
        panel_c_xt_rep = None
        panel_c_t_rep = float(self.params_train.get("panel_c_t_value", self.T))
        try:
            panel_c_t_rep = float(panel_c_t_rep)
        except Exception:
            panel_c_t_rep = float(self.T)
        panel_c_x_override = self.params_train.get("panel_c_x_value", None)
        if panel_c_x_override is not None:
            try:
                panel_c_xt_rep = float(panel_c_x_override)
            except Exception:
                panel_c_xt_rep = None
        try:
            X_pred_T_np = X_pred[:, -1, :].detach().cpu().numpy()
            X_pred_T_flat = X_pred_T_np.reshape(-1)
            X_pred_T_flat = X_pred_T_flat[np.isfinite(X_pred_T_flat) & (X_pred_T_flat > 0)]
            if panel_c_xt_rep is None and len(X_pred_T_flat) > 0:
                panel_c_xt_rep = float(X_pred_T_flat[0])
        except Exception:
            panel_c_xt_rep = None
        self._panel_c_xt_rep = panel_c_xt_rep
        self._panel_c_t_rep = panel_c_t_rep

        # Both training Top-k plots and explicit checkpoint evaluation always use
        # (a) a conditional flow-policy density, (b) value, (c) state trajectories.
        # Historical params.json layout/panel-c flags cannot restore a mean panel.
        plot_panel_c = bool(self.params_train.get("plot_panel_c", True))
        policy_plot_time_fraction = getattr(self, "_cli_policy_plot_time_fraction", None)
        if policy_plot_time_fraction is None:
            policy_plot_time_fraction = self.params_train.get("policy_plot_time_fraction", 0.25)

        # jump_index must match the (possibly sliced) number of trajectories being plotted.
        jump_index_plot = getattr(self, "jump_index_eval", None)
        if jump_index_plot is None:
            jump_index_plot = getattr(self, "jump_index", None)
        try:
            M_plot = int(t_test.shape[0]) if (t_test is not None and hasattr(t_test, "shape")) else None
        except Exception:
            M_plot = None
        if jump_index_plot is not None and M_plot is not None:
            try:
                if torch.is_tensor(jump_index_plot):
                    jump_index_plot = jump_index_plot.to(device)
                    jump_index_plot = jump_index_plot[jump_index_plot < int(M_plot)]
                else:
                    ji = torch.as_tensor(jump_index_plot, dtype=torch.long, device=device)
                    ji = ji[ji < int(M_plot)]
                    jump_index_plot = ji
            except Exception:
                jump_index_plot = None
        try:
            plot_combined_sw_xt = int(self.params_train.get("plot_combined_smooth_window_xt", 1))
        except Exception:
            plot_combined_sw_xt = 1
        plot_combined_sw_xt = max(1, plot_combined_sw_xt)
        try:
            plot_combined_ds_xt = int(self.params_train.get("plot_combined_downsample_step_xt", 1))
        except Exception:
            plot_combined_ds_xt = 1
        plot_combined_ds_xt = max(1, plot_combined_ds_xt)
        plot_trajectory_combined_abc(
            t_np,
            traj_u_mean=mu_np,
            traj_u_test=traj_u_test_np,
            traj_value=Yp_np,
            traj_value_test=traj_value_test_np,
            traj_Xt=Xp_np,
            traj_Xt_test=traj_Xt_test_np,
            approximate_u="Mean of $\\hat{\\pi}(t, \\hat{X}_t)$",
            exact_u=exact_u.replace("$u^*(t,X_t)$", "$\\pi^*(t,X_t)$").replace("Mean of $u^*(t,X_t)$", "Mean of $\\pi^*(t,X_t)$").replace("Taylor/Ansatz $u^*(t,X_t)$", "Taylor/Ansatz $\\pi^*(t,X_t)$") if exact_u else None,
            start_u=start_u.replace("$u^*", "$\\pi^*").replace("Taylor $u^*", "Taylor $\\pi^*") if start_u else None,
            end_u=end_u.replace("$u^*", "$\\pi^*").replace("Taylor $u^*", "Taylor $\\pi^*") if end_u else None,
            ylabel_u="Mean of $\\pi(t, \\hat{X}_t)$",
            approximate_v="Approx. $\\hat{v}(t, X_t)$",
            exact_v=exact_v,
            start_v=start_v,
            end_v=end_v,
            ylabel_v="$v(t, \\hat{X}_t)$",
            approximate_x="Approx. $\\hat{X}_t$ by $\\hat{\\pi}$",
            exact_x=exact_x,
            start_x="$X_0$",
            end_x="$X_T$",
            ylabel_x="$X_t$",
            traj_u_third=traj_u_third_np,
            traj_value_third=traj_value_third_np,
            traj_Xt_third=traj_Xt_third_np,
            third_u=third_u,
            third_v=third_v,
            third_x=third_x,
            filename=_fn("merton_power_utility_combined"),
            P_test=P_test,
            jump_index=jump_index_plot,
            num_samples=plot_num if plot_num is not None else 1,
            jump_num=3,
            output_folder=self.output_folder,
            downsample_step=5,
            smooth_window=smooth_window,
            model_type="lq_merton",
            model=self,
            panel_a_policy_distribution=True,
            policy_plot_time_fraction=float(policy_plot_time_fraction),
            smooth_window_xt=plot_combined_sw_xt,
            downsample_step_xt=plot_combined_ds_xt,
        )
        # Panel (c): conditional distribution p(u|t,x) and/or evolved x distribution
        panel_c_mode = self.params_train.get("panel_c_mode", "Conddist_u_T")
        if plot_panel_c and bool(self.params_train.get("panel_c_separate", False)):
            if self._panel_c_xt_rep is not None:
                print(f"[plot] panel-c representative point set: t={self._panel_c_t_rep:.6f}, x={self._panel_c_xt_rep:.6f}")
            else:
                self._panel_c_t_rep = float(self.T)

            panel_c_n_samples_u = self.params_train.get("panel_c_n_samples_u", 20000)
            panel_c_plot_method = self.params_train.get("panel_c_plot_method", "histogram")

            if panel_c_mode in ("Conddist_u_T", "both"):
                if self._panel_c_xt_rep is not None:
                    try:
                        plot_gibbs_policy_by_pinn(
                            self, self._panel_c_xt_rep, device=device, output_folder=self.output_folder, filename_suffix=filename_suffix
                        )
                        plot_H_by_pinn(
                            self, self._panel_c_xt_rep, device=device, output_folder=self.output_folder, filename_suffix=filename_suffix
                        )
                    except Exception as e:
                        print(f"[plot] WARNING: panel-c Gibbs/H debug failed: {e}")

                plot_panel_c_conditional_distribution(
                    self,
                    t_values=None,
                    x_values=None,
                    n_samples_u=panel_c_n_samples_u,
                    plot_method=panel_c_plot_method,
                    device=device,
                    filename_suffix=filename_suffix,
                    output_folder=self.output_folder,
                )

            if panel_c_mode in ("X_dist_evolve_T", "both"):
                plot_panel_c_evolved_x_distribution(
                    self,
                    device=device,
                    filename_suffix=filename_suffix,
                    output_folder=self.output_folder,
                )

            if panel_c_mode == "U_dist_on_XT":
                # Separate plot: distribution of u_T across terminal samples (no extra PINN calls).
                try:
                    # Prefer using already computed policy-mean trajectories when available
                    uT_flow = None
                    uT_pinn = None
                    try:
                        if "mu_np" in locals() and mu_np is not None:
                            uT_flow = np.asarray(mu_np)[:, -1, :].reshape(-1)
                        if "traj_u_test_np" in locals() and traj_u_test_np is not None:
                            uT_pinn = np.asarray(traj_u_test_np)[:, -1, :].reshape(-1)
                    except Exception:
                        uT_flow = None
                        uT_pinn = None

                    if uT_flow is None or uT_pinn is None:
                        print("[plot][panel_c] U_dist_on_XT requested but u trajectories not available in scope.")
                    else:
                        uT_flow = uT_flow[np.isfinite(uT_flow)]
                        uT_pinn = uT_pinn[np.isfinite(uT_pinn)]
                        fig_u, ax_u = plt.subplots(1, 1, figsize=(6.5, 4.5), dpi=150)
                        plot_method = str(self.params_train.get("panel_c_plot_method", "histogram"))
                        def _normalize_max1(y):
                            y = np.asarray(y)
                            ymax = np.nanmax(y) if y.size else 0.0
                            return y / ymax if ymax > 0 else y

                        if plot_method == "kde":
                            try:
                                kde_flow = gaussian_kde(uT_flow)
                                kde_pinn = gaussian_kde(uT_pinn)
                                u_min_plot = float(min(uT_flow.min(), uT_pinn.min()))
                                u_max_plot = float(max(uT_flow.max(), uT_pinn.max()))
                                u_plot = np.linspace(u_min_plot, u_max_plot, 200)
                                y_flow = _normalize_max1(kde_flow(u_plot))
                                y_pinn = _normalize_max1(kde_pinn(u_plot))
                                ax_u.plot(u_plot, y_flow, label="Flownet $u_T$", linewidth=2, alpha=0.7, color="darkorange")
                                ax_u.plot(u_plot, y_pinn, label="PINN $u_T$", linewidth=2, alpha=0.7, color="darkviolet")
                            except Exception as e:
                                print(f"[WARNING] KDE failed: {e}, falling back to histogram")
                                plot_method = "histogram"
                        if plot_method == "histogram":
                            bins = 60
                            hist_flow, edges = np.histogram(uT_flow, bins=bins, range=(u_min_plot, u_max_plot), density=False)
                            hist_pinn, _ = np.histogram(uT_pinn, bins=bins, range=(u_min_plot, u_max_plot), density=False)
                            centers = 0.5 * (edges[:-1] + edges[1:])
                            ax_u.plot(centers, _normalize_max1(hist_flow), drawstyle="steps-mid",
                                      label="Flownet $u_T$", linewidth=2, alpha=0.7, color="darkorange")
                            ax_u.plot(centers, _normalize_max1(hist_pinn), drawstyle="steps-mid",
                                      label="PINN $u_T$", linewidth=2, alpha=0.7, color="darkviolet")
                        ax_u.set_xlabel("$u_T$", fontsize=FONT_SIZE)
                        ax_u.set_ylabel("Normalized density (max=1)", fontsize=FONT_SIZE)
                        ax_u.legend(fontsize=LEGEND_FONT_SIZE, loc="best", framealpha=0.5, edgecolor="gray")
                        ax_u.grid(True, alpha=0.3)
                        fig_u.tight_layout()
                        save_path_u = os.path.join(self.output_folder, f"panel_c_u_dist_on_xt{filename_suffix}_div4.pdf")
                        save_path_u = os.path.join(self.output_folder, f"panel_c_u_dist_on_xt{filename_suffix}_div4.pdf")
                        fig_u.savefig(save_path_u, dpi=300, bbox_inches="tight")
                        plt.close(fig_u)
                        print(f"[panel_c] Saved u_T distribution plot to {save_path_u}")
                except Exception as e:
                    print(f"[plot][panel_c] U_dist_on_XT failed: {e}")

    def _default_output_folder(self):
        return training_output_folder(
            "Merton", self.params_equ, self.params_train, root=_ROOT / "output",
        )

    def save_and_read_model(self, type: str = "save"):
        ckpt_given = (self.actor_path is not None and self.critic_path is not None
                    and os.path.isfile(self.actor_path) and os.path.isfile(self.critic_path))
        if ckpt_given:
            dir_a = os.path.dirname(os.path.abspath(self.actor_path))
            dir_c = os.path.dirname(os.path.abspath(self.critic_path))
            print("-----Dir a is:-----", dir_a)
            try:
                common_dir = os.path.commonpath([dir_a, dir_c])
            except ValueError:
                common_dir = dir_a  
            self.output_folder = common_dir
        else:
            self.output_folder = self._default_output_folder()

        os.makedirs(self.output_folder, exist_ok=True)

        ACTOR_CKPT  = self.actor_path  if ckpt_given else os.path.join(self.output_folder, "actor.pth")
        CRITIC_CKPT = self.critic_path if ckpt_given else os.path.join(self.output_folder, "critic.pth")

        map_loc = self.device
        if type == "save":
            torch.save(self.net_actor.state_dict(),  ACTOR_CKPT)
            torch.save(self.net_critic.state_dict(), CRITIC_CKPT)
        elif type == "read":
            if os.path.isfile(ACTOR_CKPT):
                self.net_actor.load_state_dict(torch.load(ACTOR_CKPT, map_location=map_loc))
            else:
                raise FileNotFoundError(f"Actor checkpoint not found: {ACTOR_CKPT}")
            if os.path.isfile(CRITIC_CKPT):
                self.net_critic.load_state_dict(torch.load(CRITIC_CKPT, map_location=map_loc))
            else:
                raise FileNotFoundError(f"Critic checkpoint not found: {CRITIC_CKPT}")
            self.net_actor.eval()
            self.net_critic.eval()
        else:
            raise ValueError("type must be 'save' or 'read'")

    def predict(self, t_star, W_star, P_star, N_eval=None):  
        """
        Roll out learned policy (current in-memory weights) and exact baselines on a fixed batch.
        
        IMPORTANT:
        - This function historically auto-loaded `actor_pinn_top1.pth` via `_load_best_eval_for_plot()`.
        - That is convenient for standalone prediction, but it BREAKS plotting helpers that
          intentionally load a specific ckpt (top2/top3 or arbitrary paths) before calling
          `_plot_trajs_for_current_weights()`: the auto-load overwrites the requested ckpt.
        """

        # Ensure fixed eval set (including X0_eval) is ready before prediction
        # If N_eval is provided, use it; otherwise use default behavior
        self._ensure_eval_set(N_eval=N_eval)
        # Use the provided t_star, W_star, P_star directly (they should match the eval set if N_eval was used)
        self.t_predict_set, self.W_predict_set, self.P_predict_set = t_star, W_star, P_star

        # Infer actual N from t_star shape and temporarily adjust self.N and self.dt if needed
        N_original = self.N
        dt_original = self.dt
        N_actual = t_star.shape[1] - 1  # t_star is [M, N+1, 1]
        need_restore = False
        if N_actual != self.N:
            self.N = N_actual
            self.dt = self.T / self.N
            need_restore = True
            print(f"[predict] temporarily setting N={self.N}, dt={self.dt:.6f} for prediction (original N={N_original})")

        # ------------------------------------------------------------------
        # Reproducibility for eval/plot:
        # learned rollout uses stochastic policy sampling (eps), which changes X_hat(t),
        # hence also changes "exact by PINN" curves evaluated on X_hat(t).
        # Default: fix the rollout RNG seed (fork_rng so we don't perturb global RNG).
        # ------------------------------------------------------------------
        eval_use_fixed_seed = bool(self.params_train.get("eval_rollout_use_fixed_seed", True))
        eval_seed = self.params_train.get("eval_rollout_fixed_seed", 123456)
        seed_int = None
        try:
            if eval_use_fixed_seed and eval_seed is not None:
                seed_int = int(eval_seed)
        except Exception:
            seed_int = None
        cuda_devices = None
        try:
            if getattr(self.device, "type", "") == "cuda" and torch.cuda.is_available():
                cuda_devices = [self.device.index] if self.device.index is not None else list(range(torch.cuda.device_count()))
        except Exception:
            cuda_devices = None

        # Clear eps_predict to ensure fresh generation in first train_loop call.
        # The first call generates and stores eps_predict; the second call (PINN) reuses it.
        if hasattr(self, "eps_predict"):
            del self.eps_predict

        # ------------------------------------------------------------------
        # IMPORTANT (eval/plot correctness):
        # TwoStageGaussianActor uses internal schedules (flow mix scale / squash temperature)
        # controlled by `set_update_step(step)`. This step is NOT stored in the checkpoint
        # state_dict (it's a plain Python int), so after loading ckpt it resets to 0.

        # If we don't set a mature step for evaluation, mixture_mean()/sample() will run
        # under a "training-initial" schedule (e.g., flow_mix_scale=0 during warmup),
        # which can make policy mean curves look very wrong even if loss_history errors
        # during training were low.

        # Default behavior here: set actor update_step to a "final schedule" value unless
        # user explicitly provides `eval_actor_update_step` in params_train.
        # ------------------------------------------------------------------
        try:
            if hasattr(self.net_actor, "set_update_step"):
                # Allow user override
                eval_step = self.params_train.get("eval_actor_update_step", None)
                if eval_step is None:
                    # Pick a step that is safely after all warmups/ramps so schedules are at final values.
                    delay = int(self.params_train.get("delay_squash_steps", getattr(self.net_actor, "delay_squash_steps", 0)) or 0)
                    freeze = int(self.params_train.get("freeze_flow_steps", getattr(self.net_actor, "freeze_flow_steps", 0)) or 0)
                    warmup = int(self.params_train.get("flow_mix_warmup_steps", getattr(self.net_actor, "flow_mix_warmup_steps", freeze)) or 0)
                    ramp = int(self.params_train.get("flow_mix_ramp_steps", getattr(self.net_actor, "flow_mix_ramp_steps", 0)) or 0)
                    eval_step = max(delay, freeze, warmup + ramp) + 1
                eval_step = int(eval_step)

                # Only set when missing or when user requests forcing it.
                force = bool(self.params_train.get("force_eval_actor_update_step", True))
                if force or (not hasattr(self, "_global_actor_update_step")):
                    self._global_actor_update_step = eval_step
                # Apply to actor module
                self.net_actor.set_update_step(int(getattr(self, "_global_actor_update_step", eval_step)))
                if bool(self.params_train.get("verbose_eval_actor_update_step", True)):
                    print(f"[predict] eval actor update_step={int(getattr(self, '_global_actor_update_step', eval_step))}")
        except Exception as e:
            print(f"[predict] WARNING: failed to set actor update_step for eval schedules: {e}")
        try:
            try:
                # Checkpoint-only evaluation has empty in-memory error lists;
                # persisted evaluation CSVs are the authoritative training curve.
                pinn_plot.plot_training_errors(self)
                pinn_plot.plot_kl_std_loss(
                    self, self.KLLoss_list, self.mu_of_policy_error_list,
                    filename="KL_StdNormal_MuOfPolicy_error",
                )
                if len(self.CriticLoss_list) and len(self.ActorLoss_list):
                    pinn_plot.plot_loss(self, self.CriticLoss_list, self.ActorLoss_list, filename="merton_loss")
                    pinn_plot.plot_controlfunction(self, region=[9, 12], filename="merton_heatmap")
            except Exception as e:
                print(f"[warn] history plots skipped: {e}")

            with torch.random.fork_rng(devices=cuda_devices, enabled=bool(seed_int is not None)):
                if seed_int is not None:
                    torch.manual_seed(seed_int)
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.manual_seed_all(seed_int)
                        except Exception:
                            pass

                # Learned policy path (generates and stores eps_predict)
                X_star, Y_star, _, control_star, _, _, _, _, _, _ = self.train_loop(
                    itr=1, traj_only=True, predict=True
                )
                # Exact policy (PINN) path using the same eps_predict
                X_test_pinn, Y_test_pinn, _, control_test_pinn, _, _, _, _, _, _ = self.train_loop(
                    itr=1, traj_only=True, _control_exact_=True, predict=True
                )
                # Exact policy (Taylor/ansatz) path using the same dW/dM_t
                X0_taylor = self.X0_eval.clone() if hasattr(self, 'X0_eval') else self._init_X0()
                # Align batch size with passed eval rollout (t_predict_set/W_predict_set).
                # X0_eval can be built with smaller M due to eval_total_state_cap in ensure_eval_set().
                try:
                    M_taylor = int(self.t_predict_set.shape[0])
                    if hasattr(X0_taylor, "shape") and int(X0_taylor.shape[0]) != M_taylor:
                        if int(X0_taylor.shape[0]) < M_taylor:
                            n_rep = (M_taylor + int(X0_taylor.shape[0]) - 1) // int(X0_taylor.shape[0])
                            X0_taylor = X0_taylor.repeat(n_rep, 1)[:M_taylor]
                        else:
                            X0_taylor = X0_taylor[:M_taylor]
                except Exception:
                    pass
                X_test_taylor, control_test_taylor = pinn_plot.simulate_taylor_path(
                    self,
                    self.t_predict_set, self.W_predict_set, self.P_predict_set, X0_taylor
                )
                policy_taylor, Y_test_taylor = self.compute_exact_exploratory_taylor_and_ansatz(X_test_taylor)
                control_test_taylor = policy_taylor.loc

            return (
                X_star, Y_star, control_star,
                X_test_pinn, Y_test_pinn, control_test_pinn,
                X_test_taylor, Y_test_taylor, control_test_taylor,
            )
        finally:
            # Restore original N and dt if they were changed
            if need_restore:
                self.N = N_original
                self.dt = dt_original
                print(f"[predict] restored N={self.N}, dt={self.dt:.6f}")

def _plot_top1_evaluation(model, device=None, plot_num=5, N_eval=None):
    """Plot only the best PINN checkpoint, using the eval-only filename convention."""
    topk = pinn_plot.load_eval_topk_meta(model, tag="pinn")
    if not topk:
        print("[info] no pinn Top-1 checkpoint found; skipping evaluation plot.")
        return False

    best = min(topk, key=lambda rec: rec["score"])
    suffix = f"ckpt_actor_pinn_top1_itr{int(best['iter'])}"
    return pinn_plot.plot_ckpt_pair(
        model,
        best["actor"],
        best["critic"],
        device=device,
        filename_suffix=suffix,
        plot_num=plot_num,
        N_eval=N_eval,
    )

if __name__ == '__main__':
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for a fresh training run')
    parser.add_argument('--actor_path', type=str, default=None, help='path to actor.pth')
    parser.add_argument('--critic_path', type=str, default=None, help='path to critic.pth')
    parser.add_argument('--resume', action='store_true',
                    help='Resume training from the given actor/critic checkpoints')
    parser.add_argument(
        '--use_frozen_script_eval',
        action='store_true',
        help=(
            'If set, delegate eval-only runs to the frozen copy of this script stored in the '
            'checkpoint directory (same basename as this file). This ensures evaluation uses '
            'the exact training-time code.'
        ),
    )
    parser.add_argument(
        '--no_frozen_script_eval',
        action='store_true',
        help=(
            'Use current plotting code when evaluating checkpoints (the default). '
            'Use this if you intentionally want to evaluate with the current (possibly modified) code.'
        ),
    )
    parser.add_argument('--plot_v0_only', action='store_true',
                        help='Only plot V(0,x): PINN vs Taylor, then exit')
    parser.add_argument('--plot_extend_policy', action='store_true',
                        help='Plot extendability test (uses loaded ckpt), then exit')
    parser.add_argument('--extend_c', type=float, default=3.0,
                        help='Extendability factor c (T_eval = c * T_train if T_eval is None)')
    parser.add_argument('--extend_T_eval', type=float, default=None,
                        help='Explicit T_eval for extendability test (overrides c)')
    parser.add_argument('--x_min', type=float, default=None, help='min x for V(0,x) plot')
    parser.add_argument('--x_max', type=float, default=None, help='max x for V(0,x) plot')
    parser.add_argument('--n_x', type=int, default=200, help='number of x grid points')
    parser.add_argument('--pinn_iter', type=int, default=2000, help='PINN iterations for V(0,x) plot')
    parser.add_argument('--plot_num', type=int, default=5, help='Number of trajectories to plot (default: 5)')
    parser.add_argument('--N_eval', type=int, default=200, help='Number of time intervals for evaluation (default: 200)')
    parser.add_argument(
        '--eval_M',
        type=int,
        default=100,
        help='Number of trajectories to use ONLY for evaluation/plotting (default: 100).',
    )
    # default=None so we do NOT override params.json / runtime settings unless user explicitly passes this flag
    parser.add_argument(
        '--plot_traj_mode',
        type=str,
        default=None,
        choices=['ab', 'ac', 'bc', 'abc'],
        help='Trajectory plot mode override: ab | ac | bc | abc (default: use params/train setting).',
    )
    parser.add_argument(
        '--plot_control_mean_kind',
        type=str,
        default=None,
        choices=['traj', 'sample_mean', 'policy_mean'],
        help=(
            'Legacy mean diagnostic (does not change the conditional-density panel): '
            '"traj" plots mean over trajectories at each time t using X_pred[:,t]; '
            '"sample_mean" uses mean over trajectories of sampled actions from predict() '
            '(shared eps_predict for learned vs PINN exact). '
            '"policy_mean" is deprecated and treated as alias of "traj".'
        ),
    )
    # NOTE: fixed-x-grid mean mode was removed; mean curves are time series over t only.
    parser.add_argument(
        '--K_mean_eval_plot',
        type=int,
        default=None,
        help='MC sample size for FlowNet mixture_mean() during plotting (default: use params_train["K_mean_eval"]).',
    )
    parser.add_argument(
        '--plot_mixture_mean_fixed_seed',
        type=int,
        default=None,
        help='Fixed RNG seed for mixture_mean() during plotting (default: params_train["mixture_mean_fixed_seed"] or 123456).',
    )
    parser.add_argument(
        '--plot_mixture_mean_use_fixed_seed',
        action='store_true',
        help='If set, force using a fixed seed for mixture_mean() during plotting.',
    )
    parser.add_argument(
        '--no_plot_mixture_mean_use_fixed_seed',
        action='store_true',
        help='If set, disable fixed-seed mixture_mean() during plotting.',
    )
    parser.add_argument('--plot_panel_c', action='store_true', default=True,
                        help='Enable optional separate distribution plots; the combined flow-policy figure always uses density/value/state panels')
    parser.add_argument(
        '--panel_c_mode',
        type=str,
        default='Conddist_u_T',
        choices=['Conddist_u_T', 'X_dist_evolve_T', 'U_dist_on_XT'],
        help=(
            'Panel c mode: '
            '"Conddist_u_T" plots conditional p(u|t=T, x=x_rep) at a representative terminal x; '
            '"X_dist_evolve_T" plots the terminal-state density p(X_T) using precomputed rollouts; '
            '"U_dist_on_XT" plots the terminal control density of u_T = pi(T, X_T) across terminal samples; '
            '(default: Conddist_u_T)'
        ),
    )
    parser.add_argument('--panel_c_n_samples_u', type=int, default=None,
                        help='Number of actor samples for the conditional policy density (default: use params)')
    parser.add_argument('--panel_c_plot_method', type=str, default=None, choices=['histogram', 'kde', 'bar'],
                        help='Plot method for the conditional policy density (default: use params)')
    parser.add_argument('--panel_c_x_value', type=float, default=None,
                        help='Override x value for panel c conditional distribution (default: use X_T[0])')
    parser.add_argument('--panel_c_t_value', type=float, default=None,
                        help='Override t value for panel c conditional distribution (default: use T)')
    parser.add_argument('--policy_plot_time_fraction', type=float, choices=[0.25, 0.5], default=None,
                        help='Conditional policy density time: 0.25 = T/4 (default), 0.5 = T/2; used in training and evaluation.')
    parser.add_argument(
        '--combined_layout',
        type=str,
        default='policy_distribution',
        choices=['policy_distribution', 'Conddist_u_T', 'panel_a_mean_of_policy'],
        help=(
            'Combined figure layout: '
            '(a) conditional density at T/4 or T/2, (b) value, (c) state trajectories. '
            'Legacy layout names are accepted as aliases; flow policies always use the distribution layout.'
        ),
    ) 
    parser.add_argument(
        '--panel_a_traj_selector',
        type=str,
        default='Conddist_u_T_mean_Xt',
        choices=['Conddist_u_T_mean_Xt', 'Conddist_u_T_XTmax_from_plotted', 'Conddist_u_T_XTmax', 'first_valid_or_XTmax', 'cond_u_at_fixed_point'],
        help=(
            'Panel (a) conditional p(u|t,x): which state x to use. '
            '"Conddist_u_T_mean_Xt" = mean of X_t over eval trajectories at each t (default, better learned); '
            '"Conddist_u_T_XTmax_from_plotted" = max X_T among the 3-4 drawn in (b),(c); '
            '"Conddist_u_T_XTmax" = max X_T over full batch; '
            '"first_valid_or_XTmax" = first traj with all X_t>=2 else max X_T else 0; '
            '"cond_u_at_fixed_point" = fixed x at the selected T/4 or T/2 time (use --panel_a_fixed_x). '
            '(default: Conddist_u_T_mean_Xt)'
        ),
    )
    parser.add_argument('--panel_a_fixed_x', type=float, default=10.0,
                        help='When panel_a_traj_selector=cond_u_at_fixed_point, fixed x for p(u|t,x) at the selected time (default: 10.0)')
    # -------- PINN eval-grid densification + reproducibility knobs --------
    parser.add_argument('--pinn_eval_n_u', type=int, default=400,
                        help='[PINN eval] u-grid resolution for Gibbs mean/std (default: 400; evaluation-only, does not retrain PINN).')
    parser.add_argument('--pinn_eval_u_min', type=float, default=None,
                        help='[PINN eval] Override u-grid min for evaluation-only Gibbs mean/std.')
    parser.add_argument('--pinn_eval_u_max', type=float, default=None,
                        help='[PINN eval] Override u-grid max for evaluation-only Gibbs mean/std.')
    parser.add_argument('--eval_rollout_fixed_seed', type=int, default=None,
                        help='Fixed RNG seed for stochastic policy rollout in predict()/plots (so subplot curves are reproducible).')
    parser.add_argument('--no_eval_rollout_fixed_seed', action='store_true',
                        help='Disable fixed-seed stochastic rollout in predict()/plots.')
    parser.add_argument('--pinn_train_fixed_seed', type=int, default=None,
                        help='Fixed RNG seed for PINN training when cache miss triggers retraining (reproducible PINN).')
    parser.add_argument('--no_pinn_train_fixed_seed', action='store_true',
                        help='Disable fixed-seed PINN training on cache miss.')

    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run directory for fresh training")

    args = parser.parse_args()
    validate_checkpoint_args(parser, args, multiplayer=False)

    # -------------------------------------------------------------------------
    # Eval-time "frozen script" delegation (reproducibility)
    # - When evaluating checkpoints, prefer executing the frozen script copy
    #   inside the checkpoint directory, so code/ckpt/params always match.
    # - Historical-code delegation requires explicit --use_frozen_script_eval.
    # -------------------------------------------------------------------------
    def _delegate_eval_to_frozen_script_if_available(_args):
        # Priority rule: explicit disable overrides enable.
        # (If both flags are passed, --no_frozen_script_eval wins.)
        if bool(getattr(_args, "no_frozen_script_eval", False)):
            return False
        force = bool(getattr(_args, "use_frozen_script_eval", False))
        if not force:
            return False  # Current distribution plotting is the default.
        if os.environ.get("MERTON_FROZEN_EVAL_DELEGATED", "") == "1":
            return False
        if _args.resume or _args.plot_v0_only or bool(getattr(_args, "plot_extend_policy", False)):
            return False
        if not (_args.actor_path and _args.critic_path):
            return False
        a = os.path.abspath(_args.actor_path)
        c = os.path.abspath(_args.critic_path)
        if not (os.path.isfile(a) and os.path.isfile(c)):
            return False

        ckpt_dir = os.path.dirname(a)
        candidates = [
            os.path.join(ckpt_dir, os.path.basename(__file__)),
            os.path.join(ckpt_dir, "code_snapshot", os.path.basename(__file__)),
        ]
        frozen_script = next((p for p in candidates if os.path.isfile(p)), None)
        if frozen_script is None:
            return False

        # If the user is using newer flags, older frozen scripts may not know them.
        # Be conservative: only auto-delegate when the frozen script appears to support requested newer flags.
        try:
            with open(frozen_script, "r", encoding="utf-8") as f:
                frozen_text = f.read()
        except Exception:
            frozen_text = ""
        requested_flags = set([x for x in sys.argv[1:] if isinstance(x, str) and x.startswith("--")])
        newer_flags = {
            "--plot_extend_policy",
            "--extend_c",
            "--extend_T_eval",
            "--plot_traj_mode",
            "--plot_panel_c",
            "--panel_c_mode",
            "--combined_layout",
            "--panel_a_traj_selector",
            "--panel_a_fixed_x",
            "--panel_c_n_samples_u",
            "--panel_c_plot_method",
            "--pinn_eval_n_u",
            "--pinn_eval_u_min",
            "--pinn_eval_u_max",
            "--eval_rollout_fixed_seed",
            "--no_eval_rollout_fixed_seed",
            "--pinn_train_fixed_seed",
            "--no_pinn_train_fixed_seed",
        }
        if not force:
            missing = []
            for flg in sorted(requested_flags.intersection(newer_flags)):
                if flg not in frozen_text:
                    missing.append(flg)
            if missing:
                print(
                    "[frozen-eval][skip] Frozen script does not appear to support flags: "
                    + ", ".join(missing)
                    + ". Using current script instead. (Tip: remove those flags or pass "
                    "--use_frozen_script_eval to force delegation.)"
                )
                return False

        # If we are already executing the frozen script, don't delegate.
        try:
            if os.path.samefile(os.path.abspath(__file__), frozen_script):
                return False
        except Exception:
            if os.path.realpath(__file__) == os.path.realpath(frozen_script):
                return False

        # Delegate (best-effort) while removing wrapper flags.
        import subprocess
        passthrough_argv = [
            x for x in sys.argv[1:]
            if x not in ("--use_frozen_script_eval", "--no_frozen_script_eval")
        ]
        env = os.environ.copy()
        env["MERTON_FROZEN_EVAL_DELEGATED"] = "1"
        # Ensure imports work even when frozen script lives under output/...
        repo_root = str(_ROOT)
        import_paths = os.pathsep.join((repo_root, str(_ROOT / "tools")))
        env["PYTHONPATH"] = import_paths + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        print("[frozen-eval] Detected eval-only run with checkpoints.")
        print(f"[frozen-eval] Delegating to frozen script: {frozen_script}")
        cmd = [sys.executable, frozen_script] + passthrough_argv
        try:
            rc = subprocess.call(cmd, env=env, cwd=repo_root)
        except Exception as e:
            print(f"[warn] frozen-eval delegation failed, falling back to current script. err={e}")
            return False
        sys.exit(rc)

    _delegate_eval_to_frozen_script_if_available(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    parameters = {"axes.labelsize": 20, "axes.titlesize": 20,
                  "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}
    plt.rcParams.update(parameters)
    configure_plot_style()

    params = {
        "equation": {
            # [-2, 2.5]

            # [-1, 2] with entropy alpha = 0.05, and [-3, 4] with entropy alpha = 0.2
            "mu": 0.1, "r": 0.05, "sigma": 0.4, "lambda": 0.3, "z": 0.1, "beta": 1.0, "p": 0.5,

            #【-0.5， 1】
            # "mu": 0.05, "r": 0.03, "sigma": 0.60, "lambda": 0.60, "z": 0.60, "beta": 1.0, "p": 0.5,

            # "mu": 0.08, "r": 0.03, "sigma": 0.15, "lambda": 0.3, "z": 0.1, "beta": 1.0, "p": 0.5,

            # [-2, 2.5]

            "D": 1, "T": 10, "M": 200, "N": 200
        }, 
        "train": {
            # LR schedule: CosineAnnealingWarmUpLR (warmup then cosine decay to lr_eta_min). milestones/gamma are NOT used.
            "lr_critic": 8e-4, "gamma_critic": 0.5, "milestones_critic": [500, 2000, 50000, 300000],  # unused (cosine)
            "lr_actor": 4e-5, "gamma_actor": 0.5, "milestones_actor": [2000, 2500, 50000, 300000],       # unused (cosine) 
            "iteration": 2000, "device": device, "weight_decay": 0, "method": 'PO_CRL', "eval_every": 50, "K_actor": 20, "K_critic": 10,
            "entropy_alpha": 0.05, "p_tsallis": 1,
            "K_entropy": 128,
            # Training only needs an unbiased MC entropy estimate.  Falling back
            # to K_entropy=128 built 128 actor graphs at every time step and made
            # a single outer iteration take tens of seconds.  Keep the larger
            # value for evaluation, but use one sample during training.
            "K_entropy_train": 1,
            # PPO-Clip: when > 0, use clipped surrogate objective in Phase 2; when None/0, use SAC-style (log_prob * advantage + entropy)
            "ppo_clip_eps": 0.0,       # PPO clip epsilon (e.g. 0.2); set to 0 or None to disable clip

            "net_width": 32, "net_depth": 2,  # actor/critic net width/depth
            "dropout_probability": 0.0,  # policy density must be deterministic conditional on state

            # State preprocessing for actor/critic (highly recommended for Merton: x>0, dynamic range large)
            # - "log" (default): state = [t/T, log(x)] (optionally scaled by log(state_log_x_scale))
            # - "raw":          state = [t, x] (legacy)
            "state_preprocess": "log", #"scale", # 
            "state_preprocess_eps": 1e-8,
            "state_log_x_scale": 30.0,  # set None to disable scaling; otherwise x_feat := log(x)/log(scale)
            "state_x_ref": 30.0,        # only used when state_preprocess="div"
            # "state_log_x_scale": 25.0,  # set None to disable scaling; otherwise x_feat := log(x)/log(scale)
            # "state_x_ref": 25.0,        # only used when state_preprocess="div"

            "plot_traj_mode": "ab",  # "ab", "ac", or "abc"
            # Top-left panel mean curve definition:
            # - plot_control_mean_kind: "traj" | "sample_mean"
            #   traj:        mean over trajectories at each time t using X_pred[:,t]
            "plot_control_mean_kind": "traj",
            "plot_panel_c": True,
            "panel_c_mode": "Conddist_u_T",  # "Conddist_u_T" | "X_dist_evolve_T"
            "panel_c_n_samples_u": 20000,
            "panel_c_plot_method": "histogram",  # "histogram" | "kde" | "bar"
            "panel_c_2x2_fixed_bins": True,   
            "panel_c_2x2_bins": 80,          
            "panel_a_traj_selector": "Conddist_u_T_mean_Xt",  # mean_Xt | from_plotted | Conddist_u_T_XTmax | first_valid_or_XTmax | cond_u_at_fixed_point
            "panel_a_fixed_x": 10.0,  # when panel_a_traj_selector=cond_u_at_fixed_point
            "X0_init_type": "point", # "point", "uniform", or "gaussian"
            "X0_init_radius": 0.0,  # radius for uniform/gaussian distribution
            "action_bound": 1.0,
            # NOTE: "actor_penalty" removed — TwoStageGaussianActor hardcodes penalty=None, so this had no effect.
            "actor_use_squash": True,  #True,  # Step 1: Can set to False to temporarily disable squash during training
            "actor_squash_mode": "sigmoid",  # Step 1: Changed from "scaled_tanh" to "sigmoid" (easier to control)
            # Flow conditioning:
            # - 1D: usually keep False (context already contains feat(state); extra state_enc can be redundant/overfit)
            # - higher-dim / richer conditional shapes: try True
            "use_state_conditioning": False,
            "flow_B": 2.5,
            "flow_K": 6,
            "flow_num_layers": 1,  # Number of stacked spline flow layers
            "flow_init_scale": 1e-3,
            # Flow mode: "direct" = zK=f(z0) after warmup, "residual" = legacy zK=z0+s*(f(z0)-z0)
            "flow_mode": "direct",
            # Flow residual mixing (only used when flow_mode="residual"):
            "flow_step_scale_init": 0.1,
            "flow_mix_schedule": "ramp",      # "constant" | "ramp"
            "flow_mix_warmup_steps": 20,      # s=0 before this many actor updates
            "flow_mix_ramp_steps": 200,       # ramp length to reach flow_mix_final (was 400, faster activation)
            "flow_mix_final": 1.0,            # final s (was 0.9, let flow fully express)
            "flow_init_last_weight_noise_std": 1e-4,  # break strict identity symmetry slightly
            "warmup_strategy": "critic_only",    #"gaussian_only",    #"affine_flow",
            # Outer warmup master switch:
            #   warmup_actor_steps = freeze_flow_steps = delay_squash_steps = warmup_steps
            # This guarantees freeze_flow_steps/delay_squash_steps do NOT exceed warmup steps.
            "warmup_steps": 30,
            "freeze_flow_steps": 30,
            "delay_squash_steps": 30,
            "squash_temp_init": 2.0,  # Not used for scaled_tanh
            "squash_temp_final": 1.0,  # Not used for scaled_tanh

            "action_min":  -1.25, # -1.25,   # NEW: minimum action value (prefer >= 0)
            "action_max": 2.25,   # NEW: maximum action value
            "actor_loss_type": 1,
            # "zK_l2_coeff": 5e-5,  # unused
            "sat_coeff": 0,
            "sat_margin": 0.02,
            "pinn_chunk": 20000, "pinn_adaptive_grid": False,  # Set to True to enable adaptive x_grid and u_grid
            # PINN training domain (x_grid) and control discretization domain (u_grid)
            "pinn_x_min": 0, "pinn_x_max": 80,
            # Align PINN u-grid with actor action range by default
            "pinn_u_min": -1.5, "pinn_u_max": 2.5,
            # Jump/normalization evaluation domain: if None, auto:
            #   x_eval_max = (1 + z * u_max) * x_max + pinn_x_eval_pad
            "pinn_x_eval_max": None,
            "pinn_x_eval_pad": 2.0,
            "pinn_x_min_fixed": 0,  # Will be set after warmup: [x_min-10, x_max+10]
            "pinn_x_max_fixed": 80,  # Will be set after warmup
            "lr_flow": 1e-5,  # flow lr (was 1e-5, raised to match lr_actor scale for faster flow learning)
            # Deprecated/ignored: flow_step_scale is NOT trainable anymore (kept only for ckpt compatibility).
            "lr_flow_step_scale": 5e-6,
            "weight_decay_flow": 0.0,  # NEW: weight decay for flow network
            "milestones_flow": None,  # NEW: milestones for flow scheduler (default: same as actor)
            "gamma_flow": None,  # NEW: gamma for flow scheduler (default: same as actor)
            "pinn_n_x": 500,
            "pinn_n_u": 400,
            # NOTE: The following parameters are ONLY used when pinn_adaptive_grid=True
            "pinn_adapt_n_mc": 256,  # Number of MC samples for estimating u range
            "pinn_u_q_lo": 0.001,  # Lower quantile for u_min (0.1%)
            "pinn_u_q_hi": 0.999,  # Upper quantile for u_max (99.9%)
            "pinn_min_u_span": 1e-3,  # Minimum u span to use adaptive range
            "pinn_adapt_verbose": False,  # Print warnings if adaptive u estimation fails
            "detach_control_for_env": True,

            "pinn_u_chunk": 256,
            "pinn_cache_dense_grid": True,
            "pinn_use_dense_grid_interpolation": False,  
            "use_policy_mean_for_predict": False,
            "K_mean_eval": 128,
            "pinn_iter": 3000,
            # "use_nonuniform_time_sampling": False,  # (disabled) legacy non-uniform dt support (we now use uniform linspace grid)
            "eval_rollout_use_fixed_seed": True,
            "eval_rollout_fixed_seed": 123456,
            # Panel (c) X_t only: moving-average window (1 = no smoothing; jumps clearer)
            "plot_combined_smooth_window_xt": 1,
            "plot_combined_downsample_step_xt": 1,
            "pinn_train_use_fixed_seed": True,
            "pinn_train_fixed_seed": 123456,
            "pinn_eval_n_u": 400,
            "predict_sample_discrete_policy": True,  # If True, use DiscretePolicy for predict() rollouts
            "pinn_eval_u_min": -1.25,
            "pinn_eval_u_max": 2.25,
            # Use dense-grid interpolation for evaluation-time V(x) if cache exists.
            "pinn_eval_use_dense_grid_interpolation": True,
            # Denser grid for cache/interpolation (higher => smoother V(x) in eval plots).
            "pinn_dense_grid_n_x": 20000,
        }  
    }

    # Eval-only: load params from checkpoint directory so N, M, dt, etc. match the run being evaluated.
    # This ensures evaluation uses the same N/M/dt as training (e.g. N500_M500 ckpt evaluated with N=500, not current file's N=200).
    if (args.actor_path and args.critic_path and not args.resume and
            os.path.isfile(args.actor_path) and os.path.isfile(args.critic_path)):
        try:
            from tools_pinn0201_optimizer import params_load_helpers as pio
            _ckpt_dir = os.path.dirname(os.path.abspath(args.actor_path))
            _loaded = pio.load_params(_ckpt_dir)
            if _loaded is not None:
                if "equation" in _loaded:
                    for _k, _v in _loaded["equation"].items():
                        params["equation"][_k] = _v
                if "train" in _loaded:
                    for _k, _v in _loaded["train"].items():
                        params["train"][_k] = _v
                params["train"]["device"] = device
                print(
                    f"[eval] Loaded params from checkpoint directory for evaluation: "
                    f"N={params['equation'].get('N')}, M={params['equation'].get('M')}"
                )
        except Exception as _e:
            print(f"[eval] Failed to load params from checkpoint directory, using current file params: {_e}")

    params_equ, params_train = params["equation"], params["train"]
    if getattr(args, "plot_traj_mode", None) is not None:
        params_train["plot_traj_mode"] = str(args.plot_traj_mode)
    if getattr(args, "plot_control_mean_kind", None) is not None:
        params_train["plot_control_mean_kind"] = str(args.plot_control_mean_kind)
    if getattr(args, "eval_M", None) is not None:
        params_train["eval_M"] = int(args.eval_M)
    # NOTE: fixed-x-grid mean flags removed
    if getattr(args, "K_mean_eval_plot", None) is not None:
        params_train["K_mean_eval_plot"] = int(args.K_mean_eval_plot)
    if getattr(args, "plot_mixture_mean_fixed_seed", None) is not None:
        params_train["plot_mixture_mean_fixed_seed"] = int(args.plot_mixture_mean_fixed_seed)
    # Fixed-seed plotting toggle (tri-state: None means "do not override")
    _plot_mm_use_fixed = None
    if bool(getattr(args, "plot_mixture_mean_use_fixed_seed", False)):
        _plot_mm_use_fixed = True
    if bool(getattr(args, "no_plot_mixture_mean_use_fixed_seed", False)):
        _plot_mm_use_fixed = False
    if _plot_mm_use_fixed is not None:
        params_train["plot_mixture_mean_use_fixed_seed"] = bool(_plot_mm_use_fixed)

    # PINN eval u-grid densification (evaluation-only)
    if getattr(args, "pinn_eval_n_u", None) is not None:
        params_train["pinn_eval_n_u"] = int(args.pinn_eval_n_u)
    if getattr(args, "pinn_eval_u_min", None) is not None:
        params_train["pinn_eval_u_min"] = float(args.pinn_eval_u_min)
    if getattr(args, "pinn_eval_u_max", None) is not None:
        params_train["pinn_eval_u_max"] = float(args.pinn_eval_u_max)

    # Fixed-seed stochastic rollout for predict()/plots
    _rollout_use_fixed = None
    if bool(getattr(args, "no_eval_rollout_fixed_seed", False)):
        _rollout_use_fixed = False
    if getattr(args, "eval_rollout_fixed_seed", None) is not None:
        params_train["eval_rollout_fixed_seed"] = int(args.eval_rollout_fixed_seed)
        _rollout_use_fixed = True
    if _rollout_use_fixed is not None:
        params_train["eval_rollout_use_fixed_seed"] = bool(_rollout_use_fixed)

    # Fixed-seed PINN training on cache miss
    _pinn_train_use_fixed = None
    if bool(getattr(args, "no_pinn_train_fixed_seed", False)):
        _pinn_train_use_fixed = False
    if getattr(args, "pinn_train_fixed_seed", None) is not None:
        params_train["pinn_train_fixed_seed"] = int(args.pinn_train_fixed_seed)
        _pinn_train_use_fixed = True
    if _pinn_train_use_fixed is not None:
        params_train["pinn_train_use_fixed_seed"] = bool(_pinn_train_use_fixed)
    params_train["plot_panel_c"] = bool(getattr(args, "plot_panel_c", False))
    params_train["panel_c_mode"] = getattr(args, "panel_c_mode", params_train.get("panel_c_mode", "Conddist_u_T"))
    params_train["combined_layout"] = getattr(args, "combined_layout", params_train.get("combined_layout", "Conddist_u_T"))
    if args.policy_plot_time_fraction is not None:
        params_train["policy_plot_time_fraction"] = args.policy_plot_time_fraction
    params_train["panel_a_traj_selector"] = getattr(args, "panel_a_traj_selector", params_train.get("panel_a_traj_selector", "Conddist_u_T_mean_Xt"))
    params_train["panel_a_fixed_x"] = float(getattr(args, "panel_a_fixed_x", params_train.get("panel_a_fixed_x", 10.0)))
    # Convenience: if user passed --panel_a_fixed_x but not --panel_a_traj_selector, use fixed-point mode.
    # Also set args.panel_a_traj_selector so model._cli_panel_a_traj_selector is set and CLI override
    # in _plot_trajs_for_current_weights restores it after params.json load from checkpoint.
    if "--panel_a_fixed_x" in sys.argv and "--panel_a_traj_selector" not in sys.argv:
        params_train["panel_a_traj_selector"] = "cond_u_at_fixed_point"
        args.panel_a_traj_selector = "cond_u_at_fixed_point"
    if getattr(args, "panel_c_n_samples_u", None) is not None:
        params_train["panel_c_n_samples_u"] = int(args.panel_c_n_samples_u)
    if getattr(args, "panel_c_plot_method", None) is not None:
        params_train["panel_c_plot_method"] = getattr(args, "panel_c_plot_method")
    params["equation"]["Xi"] = torch.ones(1, params_equ["D"]) * 10
    # params["equation"]["terminal"] = torch.ones(params_equ["M"], params_equ["D"])
    params_act = {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6(), "Sigmoid": nn.Sigmoid(),
                  "LeakyReLU": nn.LeakyReLU(), "GELU": nn.GELU()}

    # -------------------------------------------------------------------------
    # Eval-only runs: build actor/critic nets using the training-time architecture
    # from ckpt_dir/params.json, even when --no_frozen_script_eval is used.

    # Important: network modules must be constructed BEFORE loading state_dict,
    # so we need to override shape-defining params (net_width/net_depth/flow_K/flow_num_layers, etc.)
    # prior to creating `net_value` / `net_control`.
    # -------------------------------------------------------------------------
    try:
        _ckpt_path = args.actor_path or args.critic_path
        _ckpt_dir = None
        if _ckpt_path and os.path.isfile(_ckpt_path):
            _ckpt_dir = os.path.dirname(os.path.abspath(_ckpt_path))
        if _ckpt_dir:
            _params_json = os.path.join(_ckpt_dir, "params.json")
            if os.path.isfile(_params_json):
                with open(_params_json, "r", encoding="utf-8") as _f:
                    _loaded = metadata_from_disk(json.load(_f), os.path.dirname(_f.name))
                _tp = _loaded.get("train", {}) if isinstance(_loaded, dict) else {}
                if isinstance(_tp, dict) and _tp:
                    _shape_keys = ("net_width", "net_depth", "flow_K", "flow_num_layers", "use_state_conditioning")
                    _updated = {}
                    for _k in _shape_keys:
                        if _k in _tp:
                            params_train[_k] = _tp[_k]
                            _updated[_k] = _tp[_k]
                    if _updated:
                        print(f"[eval][arch-sync] Loaded net architecture from params.json: {_updated}")
    except Exception as _e:
        print(f"[eval][arch-sync] Failed to sync net arch from params.json: {_e}")

    # -------------------------------------------------------------------------
    # Eval-only runs: build actor/critic nets using the training-time architecture
    # from ckpt_dir/params.json, even when --no_frozen_script_eval is used.

    # Important: network modules must be constructed BEFORE loading state_dict,
    # so we need to override shape-defining params (net_width/net_depth/flow_K/flow_num_layers, etc.)
    # prior to creating `net_value` / `net_control`.
    # -------------------------------------------------------------------------
    try:
        _ckpt_path = args.actor_path or args.critic_path
        _ckpt_dir = None
        if _ckpt_path and os.path.isfile(_ckpt_path):
            _ckpt_dir = os.path.dirname(os.path.abspath(_ckpt_path))
        if _ckpt_dir:
            _params_json = os.path.join(_ckpt_dir, "params.json")
            if os.path.isfile(_params_json):
                with open(_params_json, "r", encoding="utf-8") as _f:
                    _loaded = metadata_from_disk(json.load(_f), os.path.dirname(_f.name))
                _tp = _loaded.get("train", {}) if isinstance(_loaded, dict) else {}
                if isinstance(_tp, dict) and _tp:
                    _shape_keys = ("net_width", "net_depth", "flow_K", "flow_num_layers", "use_state_conditioning")
                    _updated = {}
                    for _k in _shape_keys:
                        if _k in _tp:
                            params_train[_k] = _tp[_k]
                            _updated[_k] = _tp[_k]
                    if _updated:
                        print(f"[eval][arch-sync] Loaded net architecture from params.json: {_updated}")
    except Exception as _e:
        print(f"[eval][arch-sync] Failed to sync net arch from params.json: {_e}")

    # net_value = Network(inputs=params_equ["D"] + 1, width=params_equ["D"] + 10, depth=3, output=1, params=params_act,
    #                     activation="Tanh")
    # For 1D Merton, width=D+10 (=11) is often underpowered; bump to a small-but-sufficient width.
    net_width = int(params_train.get("net_width", 64)) # params_equ["D"] + 1 #
    net_depth = int(params_train.get("net_depth", 2))
    net_value = Network(
        inputs=params_equ["D"] + 1,
        width=net_width,
        depth=net_depth,
        output=1,
        params=params_act,
        activation="Tanh",
    )

    # Base Gaussian actor + post-net on sampled z (mu + sigma * eps)
    net_control = TwoStageGaussianActor(
        inputs=params_equ["D"] + 1,
        width=net_width, # params_equ["D"] + 10,
        depth=net_depth, # 3,
        output=1,
        params=params_act,
        activation="Tanh",
        bound_init=params_train.get("action_bound", 1.0),  # action_range = 8.0 (from -2 to 6)
        post_width=net_width, # params_equ["D"] + 10,
        use_squash=params_train.get("actor_use_squash", True),
        squash_mode=params_train.get("actor_squash_mode", "scaled_tanh"),  # Changed to scaled_tanh
        use_state_conditioning=params_train.get("use_state_conditioning", False),
        flow_B=params_train.get("flow_B", 3),  # Spline works in [-B, B] = [-6.5, 6.5], covers action range [-2.0, 6.0]
        flow_K=params_train.get("flow_K", 6),
        flow_num_layers=params_train.get("flow_num_layers", 1),
        flow_init_scale=params_train.get("flow_init_scale", 1e-3),
        flow_mode=params_train.get("flow_mode", "direct"),
        flow_step_scale_init=params_train.get("flow_step_scale_init", 0.1),
        flow_mix_schedule=params_train.get("flow_mix_schedule", "constant"),
        flow_mix_warmup_steps=params_train.get("flow_mix_warmup_steps", None),
        flow_mix_ramp_steps=params_train.get("flow_mix_ramp_steps", 200),
        flow_mix_final=params_train.get("flow_mix_final", None),
        freeze_flow_steps=params_train.get("freeze_flow_steps", 30),
        delay_squash_steps=params_train.get("delay_squash_steps", 30),  # Not used for scaled_tanh
        squash_temp_init=params_train.get("squash_temp_init", 3.0),  # Not used for scaled_tanh
        squash_temp_final=params_train.get("squash_temp_final", 1.0),  # Not used for scaled_tanh
        flow_init_last_weight_noise_std=params_train.get("flow_init_last_weight_noise_std", 1e-4),
        action_min=params_train.get("action_min", -2.0),  # NEW
        action_max=params_train.get("action_max", 3.0),   # NEW
        # zK_l2_coeff=params_train.get("zK_l2_coeff", 1e-4),  # unused
        sat_coeff=params_train.get("sat_coeff", 0),
        sat_margin=params_train.get("sat_margin", 0.02),
        dropout_probability=params_train.get("dropout_probability", 0.0),
    )
    if args.output_dir is not None:
        params_train["output_dir"] = os.path.abspath(args.output_dir)
    params_train["_overwrite"] = args.overwrite
    model = FBSNN(net_value, net_control, params, args.actor_path, args.critic_path)
    # Persist CLI overrides so they survive ckpt param reloads inside _load_ckpt_pair()/predict().
    model._cli_plot_traj_mode = getattr(args, "plot_traj_mode", None)
    model._cli_plot_control_mean_kind = getattr(args, "plot_control_mean_kind", None)
    model._cli_K_mean_eval_plot = getattr(args, "K_mean_eval_plot", None)
    model._cli_plot_mixture_mean_fixed_seed = getattr(args, "plot_mixture_mean_fixed_seed", None)
    model._cli_plot_mixture_mean_use_fixed_seed = _plot_mm_use_fixed
    # Only make panel-c CLI overrides sticky when user explicitly asked for panel-c.
    _cli_plot_panel_c = True if bool(getattr(args, "plot_panel_c", False)) else None
    model._cli_plot_panel_c = _cli_plot_panel_c
    model._cli_panel_c_mode = getattr(args, "panel_c_mode", None) if _cli_plot_panel_c else None
    model._cli_combined_layout = getattr(args, "combined_layout", None)
    model._cli_policy_plot_time_fraction = args.policy_plot_time_fraction
    model._cli_panel_a_traj_selector = getattr(args, "panel_a_traj_selector", None)
    model._cli_panel_a_fixed_x = getattr(args, "panel_a_fixed_x", None)
    model._cli_panel_c_n_samples_u = getattr(args, "panel_c_n_samples_u", None) if _cli_plot_panel_c else None
    model._cli_panel_c_plot_method = getattr(args, "panel_c_plot_method", None) if _cli_plot_panel_c else None
    model._cli_panel_c_x_value = getattr(args, "panel_c_x_value", None) if _cli_plot_panel_c else None
    model._cli_panel_c_t_value = getattr(args, "panel_c_t_value", None) if _cli_plot_panel_c else None
    if args.plot_v0_only:
        pinn_plot.plot_V0_xgrid_pinn_vs_taylor(
            model,
            x_min=args.x_min,
            x_max=args.x_max,
            n_x=args.n_x,
            n_iter=args.pinn_iter,
        )
        sys.exit(0)
    # model.train_all(n=10, N_Iter=params_train["iteration"])
    has_ckpt = (args.actor_path is not None and args.critic_path is not None
                and os.path.isfile(args.actor_path) and os.path.isfile(args.critic_path))

    if args.plot_extend_policy:
        if not has_ckpt:
            parser.error("--plot_extend_policy requires valid actor/critic checkpoints")
        itr = 0
        if args.actor_path:
            last_part = args.actor_path.split('_')[-1].split('.')[0]
            if last_part.startswith('itr'):
                last_part = last_part[3:]
            elif last_part.startswith('it'):
                last_part = last_part[2:]
            if last_part.isdigit():
                itr = int(last_part)
        T_train = params_equ["T"]
        plot_extendability_test(
            model=model,
            T_train=T_train,
            T_eval=args.extend_T_eval,
            c=float(args.extend_c),
            device=device,
            output_folder=model.output_folder,
            filename="extendability_test",
            downsample_step=1,
            smooth_window=1,
            itr=itr,
        )
        sys.exit(0)

    if args.resume:
        if not has_ckpt:
            raise FileNotFoundError("--resume requires valid actor/critic checkpoints")
        else:
            print(f"[resume] is given by \n  actor:  {args.actor_path}\n  critic: {args.critic_path}")
            print("=====Now training is starting from history.=====")
            sd_actor = torch.load(args.actor_path, map_location=device)
            sd_critic = torch.load(args.critic_path, map_location=device)
            model.net_actor.load_state_dict(sd_actor)
            model.net_critic.load_state_dict(sd_critic)
            # Sync target critic with loaded online critic
            model.net_critic_target.load_state_dict(sd_critic)
            model.net_actor.train(); model.net_critic.train()

            actor_dir = os.path.dirname(os.path.abspath(args.actor_path))
            resume_dir = os.path.join(actor_dir, "resume")
            prepare_run_directory(resume_dir, resume=True)
            model.output_folder = resume_dir

            print(f"[resume] output_folder set to: {model.output_folder}")
            # Resume training state:
            # - If training-state file exists: restore optimizer/scheduler/global_step exactly.
            # - Else: infer start_it and align schedules + MultiStepLR as a compatible fallback.
            from tools_pinn0201_optimizer import resume_tools as _rt
            start_it = int(_rt.prepare_resume(model, args.actor_path, args.critic_path, device=device))
            total_it = int(params_train["iteration"])
            if start_it >= total_it:
                print(f"[resume] start_it={start_it} >= total iteration={total_it}; nothing to train.")
            else:
                model.train_all(n=20, N_Iter=total_it, start_it=start_it)
            model.net_actor.eval(); model.net_critic.eval()

        _plot_top1_evaluation(model, device=device, plot_num=args.plot_num, N_eval=args.N_eval)

    else:
        if has_ckpt:
            print(f"Loading checkpoints for evaluation/plotting without further training:\n  actor:  {args.actor_path}\n  critic: {args.critic_path}")
            plot_num = args.plot_num if hasattr(args, 'plot_num') else 5
            N_eval = args.N_eval if hasattr(args, 'N_eval') and args.N_eval is not None else None
            pinn_plot.plot_ckpt_pair(
                model,
                args.actor_path,
                args.critic_path,
                device=device,
                plot_num=plot_num,
                N_eval=N_eval,
            )
            print("[eval] Checkpoint evaluation plot saved.")
            sys.exit(0)
        else:
            print("[train] No checkpoint supplied; training from scratch.")
            model.train_all(n=20, N_Iter=params_train["iteration"])
            model.net_actor.eval(); model.net_critic.eval()
            _plot_top1_evaluation(model, device=device, plot_num=args.plot_num, N_eval=args.N_eval)
