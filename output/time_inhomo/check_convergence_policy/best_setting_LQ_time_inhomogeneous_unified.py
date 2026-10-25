"""Time-inhomogeneous LQ actor-critic experiments with a matched benchmark.

Run from the repository root. One entry point supports convergence/periodic
coefficients and classical control (alpha=0)/Shannon policy (alpha>0).
Historical scripts and the source snapshots of recorded runs are in backup/.

The problem is infinite-horizon; --horizon is the rollout training window.
    dX = b(t) u dt + sigma(t) dW + jump_alpha(t) (dN - lambda dt)
    f(x,u) = -q2 |x|^2 - q1 |u|^2
The policy objective adds -entropy_alpha*log(pi). Both value targets and
benchmarks use that same entropy weight, once. Optional martingale correction
subtracts noise increments, as in manuscript0917.pdf (3.17).
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from martingale import corrected_td_residual
from experiment_utils import configure_plot_style, configure_policy_output


@dataclass
class ExperimentConfig:
    """Defaults follow the recorded time-inhomogeneous experiments."""

    coefficient_type: str = "periodic"
    entropy_alpha: float = 0.05
    actor_loss: str = "manuscript"
    fixed_std: float = 0.1
    horizon: float = 20.0
    time_steps: int = 2000
    batch_size: int = 100
    dimensions: int = 1
    iterations: int = 3000
    width: int = 16
    depth: int = 3
    initial_state: float = 10.0
    beta: float = 1.0
    q1: float = 2.0
    q2: float = 0.1
    jump_rates: tuple = (0.2,)
    mu0: float = 0.6
    mu_inf: float = 0.5
    k_mu: float = 1.0
    sigma0: float = 0.3
    sigma_inf: float = 0.2
    k_sigma: float = 1.0
    alpha0: float = 0.3
    alpha_inf: float = 0.2
    k_alpha: float = 1.0
    mu_bar: float = 0.2
    mu_amp: float = 0.02
    phi_mu: float = 0.0
    sigma_bar: float = 0.2
    sigma_amp: float = 0.1
    phi_sigma: float = 0.0
    alpha_bar: float = 0.2
    alpha_amp: float = 0.1
    phi_alpha: float = 0.0
    period: float = 10.0
    random_time: str = "none"
    exact_dt: float = 0.02
    truncation_horizon: float = 60.0
    exact_rtol: float = 1e-9
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    actor_update_every: int = 20
    critic_update_every: int = 5
    critic_warmup_iterations: int = 10
    eval_every: int = 20
    seed: int = 2025
    eval_seed: int = 789012
    grad_clip: float = 1.0
    martingale_correction: bool = True
    target_polyak: float = 0.0
    device: str = "cpu"

    @property
    def mode(self):
        return "control" if self.entropy_alpha == 0 else "policy"

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if self.coefficient_type == "convergent":
            self.coefficient_type = "convergence"
        if self.coefficient_type not in {"convergence", "periodic"}:
            raise ValueError("coefficient_type must be convergence or periodic")
        if not math.isfinite(self.entropy_alpha) or self.entropy_alpha < 0:
            raise ValueError("entropy_alpha must be finite and >= 0")
        for key in ("fixed_std", "horizon", "time_steps", "batch_size", "dimensions",
                    "width", "beta", "q1", "period", "exact_dt", "exact_rtol",
                    "actor_lr", "critic_lr", "actor_update_every", "critic_update_every",
                    "eval_every", "grad_clip"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and > 0")
        if self.q2 < 0 or self.depth < 0 or self.iterations < 0 or self.critic_warmup_iterations < 0:
            raise ValueError("q2, depth, iterations and critic_warmup_iterations must be >= 0")
        if not 0 <= self.target_polyak < 1:
            raise ValueError("target_polyak must be in [0, 1)")
        if self.actor_loss not in {"manuscript", "entropy_outside"}:
            raise ValueError("unknown actor_loss")
        if self.random_time not in {"none", "start", "end"}:
            raise ValueError("random_time must be none, start or end")
        if self.coefficient_type == "convergence":
            if self.truncation_horizon <= self.horizon:
                raise ValueError("truncation_horizon must exceed the rollout horizon")
            if min(self.k_mu, self.k_sigma, self.k_alpha) <= 0:
                raise ValueError("convergence rates must be > 0")
            if self.random_time != "none":
                raise ValueError("random-time is available for periodic coefficients")
        rates = tuple(float(v) for v in self.jump_rates)
        if len(rates) == 1:
            rates *= self.dimensions
        if len(rates) != self.dimensions or any(v < 0 or not math.isfinite(v) for v in rates):
            raise ValueError("jump_rates needs one non-negative rate per dimension")
        self.jump_rates = rates
        return self

    def to_params(self):
        """Keep equation/train metadata readable by historical tooling."""
        pe = {key: getattr(self, key) for key in (
            "beta", "q1", "q2", "mu0", "mu_inf", "k_mu", "sigma0", "sigma_inf",
            "k_sigma", "alpha0", "alpha_inf", "k_alpha", "mu_bar", "mu_amp",
            "phi_mu", "sigma_bar", "sigma_amp", "phi_sigma", "alpha_bar", "alpha_amp", "phi_alpha")}
        pe.update(T=self.horizon, N=self.time_steps, M=self.batch_size, D=self.dimensions,
                  Xi=self.initial_state, coeff_SDE_type=self.coefficient_type,
                  random_time=self.random_time, dt_exact=self.exact_dt,
                  T_trunc=self.truncation_horizon)
        pe["lambda"] = list(self.jump_rates)
        pe.update({key: self.period for key in ("P_mu", "P_sigma", "P_alpha")})
        pt = dict(entropy_alpha=self.entropy_alpha, gamma=self.entropy_alpha, p_tsallis=1,
                  actor_loss=self.actor_loss, policy_fixed_std=self.fixed_std if self.mode == "control" else None,
                  iteration=self.iterations, lr_actor=self.actor_lr, lr_critic=self.critic_lr,
                  K_actor=self.actor_update_every, K_critic=self.critic_update_every,
                  eval_every=self.eval_every, critic_warmup_iterations=self.critic_warmup_iterations,
                  net_width=self.width, net_depth=self.depth, device=self.device)
        return {"config": asdict(self), "equation": pe, "train": pt,
                "benchmark_mode": self.mode, "format_version": 1}

    @classmethod
    def from_params(cls, params):
        if "config" in params:
            allowed = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in params["config"].items() if k in allowed}).validate()
        pe, pt = params["equation"], params["train"]
        if pt.get("p_tsallis", 1) != 1:
            raise ValueError("This solver has a Shannon Gaussian benchmark; Tsallis checkpoints are unsupported")
        cfg = cls()
        for key in ("beta", "q1", "q2", "mu0", "mu_inf", "k_mu", "sigma0", "sigma_inf",
                    "k_sigma", "alpha0", "alpha_inf", "k_alpha", "mu_bar", "mu_amp", "phi_mu",
                    "sigma_bar", "sigma_amp", "phi_sigma", "alpha_bar", "alpha_amp", "phi_alpha"):
            if key in pe:
                setattr(cfg, key, float(pe[key]))
        for new, old in (("horizon", "T"), ("time_steps", "N"), ("batch_size", "M"),
                         ("dimensions", "D"), ("exact_dt", "dt_exact"), ("truncation_horizon", "T_trunc")):
            if old in pe:
                setattr(cfg, new, pe[old])
        cfg.coefficient_type = pe.get("coeff_SDE_type", "convergent")
        cfg.entropy_alpha = float(pt.get("entropy_alpha", pt.get("gamma", 0.0)))
        cfg.jump_rates = tuple(np.atleast_1d(pe.get("lambda", 0.2)).tolist())
        cfg.period = float(pe.get("P_mu", 10.0))
        if cfg.coefficient_type == "periodic" and any(
            not math.isclose(float(pe.get(k, cfg.period)), cfg.period) for k in ("P_sigma", "P_alpha")
        ):
            raise ValueError("periodic coefficients must share a common period")
        cfg.random_time = pe.get("random_time") or "none"
        xi = pe.get("Xi", 10.0)
        if isinstance(xi, dict):
            xi = xi["data"]
        cfg.initial_state = float(np.asarray(xi).reshape(-1)[0])
        for new, old in (("iterations", "iteration"), ("actor_lr", "lr_actor"),
                         ("critic_lr", "lr_critic"), ("actor_update_every", "K_actor"),
                         ("critic_update_every", "K_critic"), ("eval_every", "eval_every"),
                         ("critic_warmup_iterations", "critic_warmup_iterations"),
                         ("width", "net_width"), ("depth", "net_depth"), ("actor_loss", "actor_loss")):
            if old in pt:
                setattr(cfg, new, pt[old])
        if pt.get("policy_fixed_std") is not None:
            cfg.fixed_std = float(pt["policy_fixed_std"])
        # Historical runs used an online bootstrapped critic, not Polyak targets.
        cfg.target_polyak = 0.0
        return cfg.validate()


class LQBenchmark:
    """Negative Riccati branch and its infinite-horizon value constant.

    Integrate h' = beta*h + q2 - b(t)^2*h^2/q1 and
    g0' = beta*g0 - (D*sigma(t)^2 + sum(lambda)*alpha(t)^2)*h
    backward, where the negative branch is stable. For periodic coefficients,
    iterate a full period to enforce both boundary conditions. For convergence,
    impose the limiting algebraic solution at T_trunc. Adaptive RK45 with dense
    output avoids shifted time labels and repeated integration per time step.

    The entropy constant is added only when a policy value is requested:
        g_alpha = g0 + D*alpha/(2*beta)*log(pi*alpha/q1).
    This exact identity also keeps control and policy cache entries separate.
    """

    def __init__(self, config):
        self.config = copy.deepcopy(config).validate()
        cfg = self.config
        self.end = cfg.period if cfg.coefficient_type == "periodic" else cfg.truncation_horizon
        b = cfg.mu_bar if cfg.coefficient_type == "periodic" else cfg.mu_inf
        s = cfg.sigma_bar if cfg.coefficient_type == "periodic" else cfg.sigma_inf
        a = cfg.alpha_bar if cfg.coefficient_type == "periodic" else cfg.alpha_inf
        h = -2 * cfg.q2 / (cfg.beta + math.sqrt(cfg.beta**2 + 4*b*b*cfg.q2/cfg.q1))
        g = (cfg.dimensions*s*s + sum(cfg.jump_rates)*a*a) * h / cfg.beta
        boundary = np.array([h, g], dtype=float)
        if cfg.coefficient_type == "periodic":
            for _ in range(1000):
                sol = self._integrate(boundary)
                updated = sol.y[:, -1]
                if np.max(np.abs(updated-boundary)) < 1e-11:
                    boundary = updated
                    break
                boundary = updated
            else:
                raise RuntimeError("Periodic Riccati boundary iteration did not converge")
        self.solution = self._integrate(boundary)
        if cfg.coefficient_type == "periodic":
            mismatch = np.max(np.abs(self.solution.sol(0.0)-self.solution.sol(self.end)))
            if mismatch > 1e-8:
                raise RuntimeError(f"Periodic benchmark boundary mismatch: {mismatch}")

    def coefficients(self, time):
        cfg = self.config
        if cfg.coefficient_type == "periodic":
            phase = 2*math.pi*time/cfg.period
            return tuple(getattr(cfg, name+"_bar") + getattr(cfg, name+"_amp") *
                         np.sin(phase+getattr(cfg, "phi_"+name)) for name in ("mu", "sigma", "alpha"))
        return tuple(getattr(cfg, name+"_inf") + (getattr(cfg, name+"0")-getattr(cfg, name+"_inf")) *
                     np.exp(-getattr(cfg, "k_"+name)*time) for name in ("mu", "sigma", "alpha"))

    def _rhs(self, t, hg):
        cfg = self.config
        b, sigma, jump_alpha = self.coefficients(t)
        h, g = hg
        return [cfg.beta*h + cfg.q2 - b*b*h*h/cfg.q1,
                cfg.beta*g - (cfg.dimensions*sigma*sigma+sum(cfg.jump_rates)*jump_alpha*jump_alpha)*h]

    def _integrate(self, boundary):
        cfg = self.config
        sol = solve_ivp(self._rhs, (self.end, 0.0), boundary, dense_output=True,
                        rtol=cfg.exact_rtol, atol=cfg.exact_rtol*0.01, max_step=cfg.exact_dt)
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError(f"Riccati integration failed: {sol.message}")
        return sol

    def h_g(self, time, entropy_alpha=None):
        cfg = self.config
        alpha = cfg.entropy_alpha if entropy_alpha is None else float(entropy_alpha)
        if alpha < 0:
            raise ValueError("entropy_alpha must be >= 0")
        t = np.asarray(time, dtype=float)
        if cfg.coefficient_type == "periodic":
            t = np.mod(t, cfg.period)
        elif np.any((t < 0) | (t > self.end)):
            raise ValueError("requested time lies outside the convergent benchmark grid")
        hg = self.solution.sol(t.reshape(-1)).reshape((2,)+t.shape)
        h, g = hg[0], hg[1]
        if alpha > 0:
            g = g + cfg.dimensions*alpha/(2*cfg.beta)*math.log(math.pi*alpha/cfg.q1)
        return h, g

    def tensors(self, x, t, entropy_alpha=None):
        """Evaluate each sample at its own time, including random initial phases."""
        if torch.is_tensor(t):
            t = t.detach().cpu().numpy()
        t = np.asarray(t).squeeze(-1) if np.ndim(t) and np.shape(t)[-1] == 1 else np.asarray(t)
        h, g = self.h_g(t, entropy_alpha)
        h = torch.as_tensor(h, device=x.device, dtype=x.dtype).unsqueeze(-1)
        g = torch.as_tensor(g, device=x.device, dtype=x.dtype).unsqueeze(-1)
        b = torch.as_tensor(self.coefficients(t)[0], device=x.device, dtype=x.dtype).unsqueeze(-1)
        mean = b*h*x/self.config.q1
        value = h*x.square().sum(dim=-1, keepdim=True)+g
        return mean, value

    def control(self, x, t=0.0):
        return self.tensors(x, t, entropy_alpha=0.0)

    def policy(self, x, t=0.0):
        if self.config.entropy_alpha <= 0:
            raise ValueError("alpha=0 has a deterministic control benchmark, not a Normal policy")
        mean, value = self.tensors(x, t, self.config.entropy_alpha)
        std = math.sqrt(self.config.entropy_alpha/(2*self.config.q1))
        return Normal(mean, torch.full_like(mean, std)), value


class Block(nn.Module):
    def __init__(self, inputs, params=None, activation="Tanh"):
        super().__init__()
        self.L1 = nn.Linear(inputs, inputs)
        self.L2 = nn.Linear(inputs, inputs)
        self.act = nn.Tanh() if params is None else params[activation]

    def forward(self, x):
        return self.act(self.L1(self.act(self.L2(x))))+x


class Network(nn.Module):
    """Scalar critic with state-dict names compatible with D=1 historical runs."""

    def __init__(self, inputs, width, depth, output=1, params=None, activation="Tanh", penalty=None):
        super().__init__()
        if penalty is not None:
            raise ValueError("The value network is unconstrained")
        self.first = nn.Linear(inputs, width)
        self.last = nn.Linear(width, output)
        self.network = nn.Sequential(self.first, *[Block(width, params, activation) for _ in range(depth)], self.last)
        self.register_buffer("bound", torch.tensor(2.0))  # legacy unused checkpoint key

    def forward(self, x):
        return self.network(x)


class Network_mu_sigma_split(nn.Module):
    def __init__(self, inputs, width, depth, output=1, params=None, activation="Tanh",
                 penalty=None, gamma=None, fixed_std=None):
        super().__init__()
        if penalty is not None:
            raise ValueError("The LQ control domain is unconstrained")
        self.first = nn.Linear(inputs, width)
        self.hidden = nn.Sequential(*[Block(width, params, activation) for _ in range(depth)])
        self.mean_layer = nn.Linear(width, output)
        self.use_fixed_std = fixed_std is not None
        if self.use_fixed_std:
            if fixed_std <= 0:
                raise ValueError("fixed_std must be > 0")
            self.register_buffer("fixed_log_std", torch.tensor(math.log(fixed_std)))
        else:
            self.log_std_layer = nn.Linear(width, output)
        self.register_buffer("bound", torch.tensor(1.0))

    def forward(self, x):
        hidden = self.hidden(torch.tanh(self.first(x)))
        mean = self.mean_layer(hidden)
        log_std = (self.fixed_log_std.expand_as(mean) if self.use_fixed_std else
                   self.log_std_layer(hidden).clamp(-10.0, 2.0))
        return mean, log_std


def actor_objective(log_prob, residual_no_entropy, entropy_alpha, dt, beta,
                    shannon_entropy, mode="manuscript"):
    """The soft advantage contains -alpha*log(pi) exactly once."""
    advantage = residual_no_entropy.detach()/dt
    if mode == "manuscript":
        soft_advantage = (advantage-entropy_alpha*log_prob.detach()).detach()
        return -log_prob*soft_advantage/beta
    if mode == "entropy_outside":
        return -(log_prob*advantage+entropy_alpha*shannon_entropy)/beta
    raise ValueError(f"Unknown actor loss: {mode}")


class FBSNN(nn.Module):
    """Online score-function actor-critic with optional Polyak target critic."""

    def __init__(self, net_value, net_control, params, actor_path=None, critic_path=None, output_folder=None):
        super().__init__()
        self.config = ExperimentConfig.from_params(params)
        cfg = self.config
        self.device = torch.device(cfg.device)
        self.net_critic = net_value.to(self.device)
        self.net_actor = net_control.to(self.device)
        configure_policy_output(self.net_actor, cfg.entropy_alpha)
        self.target_critic = None
        if cfg.target_polyak > 0:
            self.target_critic = copy.deepcopy(self.net_critic).requires_grad_(False)
        self.benchmark = LQBenchmark(cfg)
        self.dt = cfg.horizon/cfg.time_steps
        self.optimizer_actor = torch.optim.Adam(self.net_actor.parameters(), lr=cfg.actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.net_critic.parameters(), lr=cfg.critic_lr)
        self.scheduler_actor = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_actor, [500, 2000, 300000], gamma=0.5)
        self.scheduler_critic = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_critic, [1000, 5000, 300000], gamma=0.5)
        self.completed_iterations = 0
        self.history = []
        self.best_score = math.inf
        self.params_equ, self.params_train = params["equation"], params["train"]
        self.output_folder = Path(output_folder) if output_folder else (
            ROOT/"output"/"time_inhomo"/cfg.coefficient_type/cfg.mode/
            datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if (actor_path is None) != (critic_path is None):
            raise ValueError("provide both actor and critic checkpoint paths")
        if actor_path:
            self.net_actor.load_state_dict(torch.load(actor_path, map_location=self.device, weights_only=True))
            self.net_critic.load_state_dict(torch.load(critic_path, map_location=self.device, weights_only=True))
            if self.target_critic is not None:
                self.target_critic.load_state_dict(self.net_critic.state_dict())

    def encode_time(self, t):
        cfg = self.config
        if cfg.coefficient_type != "periodic":
            return t
        phase = 2*math.pi*t/cfg.period
        return torch.cat([torch.sin(phase), torch.cos(phase), torch.sin(2*phase), torch.cos(2*phase)], dim=-1)

    def state_input(self, t, x):
        return torch.cat([self.encode_time(t), x], dim=-1)

    def coefficients(self, t):
        cfg = self.config
        if cfg.coefficient_type == "periodic":
            phase = 2*math.pi*t/cfg.period
            return tuple(getattr(cfg, n+"_bar")+getattr(cfg, n+"_amp")*
                         torch.sin(phase+getattr(cfg, "phi_"+n)) for n in ("mu", "sigma", "alpha"))
        return tuple(getattr(cfg, n+"_inf")+(getattr(cfg, n+"0")-getattr(cfg, n+"_inf"))*
                     torch.exp(-getattr(cfg, "k_"+n)*t) for n in ("mu", "sigma", "alpha"))

    def distribution(self, t, x):
        mean, log_std = self.net_actor(self.state_input(t, x))
        return Normal(mean, log_std.exp())

    def compute_exact_no_exploratory(self, x, t=0.0, **_):
        return self.benchmark.control(x, t)

    def compute_exact_entropy(self, x, t=0.0):
        return self.benchmark.policy(x, t)

    def net_u_Du(self, t, x, compute_Du=True):
        value = self.net_critic(self.state_input(t, x))
        gradient = None
        if compute_Du:
            gradient = torch.autograd.grad(value.sum(), x, retain_graph=True)[0].detach()
        return value, gradient

    def _initial_times(self, generator):
        cfg = self.config
        t = torch.zeros(cfg.batch_size, 1, device=self.device)
        end = torch.full_like(t, cfg.horizon)
        if cfg.random_time == "start":
            t = torch.rand(t.shape, device=self.device, generator=generator)*cfg.period
            end = t+cfg.horizon
        elif cfg.random_time == "end":
            # Keep at least one full step for every trajectory.
            end = self.dt+(cfg.horizon-self.dt)*torch.rand(t.shape, device=self.device, generator=generator)
        return t, end

    @staticmethod
    def _masked_mean(values, active):
        return (values*active).sum()/active.sum().clamp_min(1.0)

    def _step_optimizer(self, losses, optimizer, parameters):
        if not losses:
            return 0.0
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; check learning rates and rollout step size")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        losses.clear()
        return float(loss.detach())

    def train_iteration(self, actor_enabled=True):
        cfg = self.config
        self.net_actor.train()
        self.net_critic.train()
        initial_t, end = self._initial_times(None)
        t = initial_t
        x = torch.full((cfg.batch_size, cfg.dimensions), cfg.initial_state, device=self.device)
        rates = torch.tensor(cfg.jump_rates, device=self.device).reshape(1, -1)
        actor_losses, critic_losses = [], []
        actor_total = critic_total = 0.0
        actor_updates = critic_updates = 0
        discount = math.exp(-cfg.beta*self.dt)
        for k in range(cfg.time_steps):
            active = (t+self.dt <= end+1e-6).float()
            x = x.detach().requires_grad_(True)
            dist = self.distribution(t, x.detach())
            control = dist.sample()
            log_prob = dist.log_prob(control).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            b, sigma, jump_alpha = self.coefficients(t)
            diffusion = sigma*math.sqrt(self.dt)*torch.randn_like(x)
            jumps = torch.poisson((rates*self.dt).expand_as(x))
            compensated = jumps-rates*self.dt
            x_next = x.detach()+b*control*self.dt+diffusion+jump_alpha*compensated
            value, gradient = self.net_u_Du(t, x, compute_Du=cfg.martingale_correction)
            with torch.no_grad():
                bootstrap = self.net_critic if self.target_critic is None else self.target_critic
                next_value = bootstrap(self.state_input(t+self.dt, x_next))
                reward = -(cfg.q2*x.detach().square().sum(-1, keepdim=True)+
                           cfg.q1*control.square().sum(-1, keepdim=True))*self.dt
                jump_values = []
                if cfg.martingale_correction:
                    for d in range(cfg.dimensions):
                        shifted = x.detach().clone()
                        shifted[:, d:d+1] += jump_alpha
                        jump_values.append(self.net_critic(self.state_input(t, shifted)))
            if cfg.martingale_correction:
                residual = corrected_td_residual(reward, value, next_value, gradient, diffusion,
                                                 torch.cat(jump_values, dim=-1), compensated, discount)
            else:
                residual = (reward+discount*next_value).detach()-value
            # Form the actor advantage before updating any critic parameters.
            if actor_enabled:
                actor_loss = actor_objective(log_prob, residual, cfg.entropy_alpha, self.dt, cfg.beta, entropy, cfg.actor_loss)
                actor_losses.append(self._masked_mean(actor_loss, active))
            soft_residual = residual-cfg.entropy_alpha*log_prob.detach()*self.dt
            critic_losses.append(self._masked_mean(soft_residual.square(), active))
            if (k+1) % cfg.critic_update_every == 0 or k == cfg.time_steps-1:
                critic_total += self._step_optimizer(critic_losses, self.optimizer_critic, self.net_critic.parameters())
                critic_updates += 1
                if self.target_critic is not None:
                    with torch.no_grad():
                        for target, online in zip(self.target_critic.parameters(), self.net_critic.parameters()):
                            target.mul_(cfg.target_polyak).add_(online, alpha=1-cfg.target_polyak)
            if actor_enabled and ((k+1) % cfg.actor_update_every == 0 or k == cfg.time_steps-1):
                actor_total += self._step_optimizer(actor_losses, self.optimizer_actor, self.net_actor.parameters())
                actor_updates += 1
            x = torch.where(active.bool(), x_next, x.detach())
            t = initial_t+(k+1)*self.dt
        self.scheduler_critic.step()
        if actor_enabled:
            self.scheduler_actor.step()
        return {"critic_loss": critic_total/max(critic_updates, 1),
                "actor_loss": actor_total/max(actor_updates, 1)}

    @torch.no_grad()
    def evaluate(self, seed=None):
        """Fixed environment/action noise; compare references on learned states.

        Alpha=0 rolls out the learned/exact means deterministically. Alpha>0
        rolls out both stochastic policies with common action noise and reports
        KL(pi_exact || pi_learned). Policy means are plotted separately from
        sampled actions. Local generators keep the training RNG unchanged.
        """
        cfg = self.config
        seed = cfg.eval_seed if seed is None else seed
        rng = torch.Generator(device=self.device).manual_seed(seed)
        previous_actor, previous_critic = self.net_actor.training, self.net_critic.training
        self.net_actor.eval()
        self.net_critic.eval()
        try:
            initial_t, end = self._initial_times(rng)
            t = initial_t
            x = torch.full((cfg.batch_size, cfg.dimensions), cfg.initial_state, device=self.device)
            x_reference = x.clone()
            rates = torch.tensor(cfg.jump_rates, device=self.device).reshape(1, -1)
            records = {key: [] for key in ("t", "x", "x_reference", "value", "value_reference",
                                          "mean", "mean_reference", "std", "std_reference", "actions", "jumps", "mask")}
            value_num = value_den = control_num = control_den = kl_sum = count = 0.0
            for k in range(cfg.time_steps+1):
                dist = self.distribution(t, x)
                if cfg.mode == "control":
                    exact_mean, exact_value = self.compute_exact_no_exploratory(x, t)
                    ref_mean, _ = self.compute_exact_no_exploratory(x_reference, t)
                    exact_std = torch.zeros_like(exact_mean)
                else:
                    exact_dist, exact_value = self.compute_exact_entropy(x, t)
                    ref_dist, _ = self.compute_exact_entropy(x_reference, t)
                    exact_mean, exact_std, ref_mean = exact_dist.loc, exact_dist.scale, ref_dist.loc
                value = self.net_critic(self.state_input(t, x))
                active = (t <= end+1e-6).float()
                value_num += float(((value-exact_value).square()*active).sum())
                value_den += float((exact_value.square()*active).sum())
                control_num += float(((dist.loc-exact_mean).square()*active).sum())
                control_den += float((exact_mean.square()*active).sum())
                if cfg.mode == "policy":
                    kl = torch.distributions.kl_divergence(exact_dist, dist).sum(-1, keepdim=True)
                    kl_sum += float((kl*active).sum())
                count += float(active.sum())
                eps = torch.randn(x.shape, generator=rng, device=self.device)
                action = dist.loc if cfg.mode == "control" else dist.loc+dist.scale*eps
                ref_action = ref_mean if cfg.mode == "control" else ref_mean+exact_std*eps
                jumps = torch.poisson((rates*self.dt).expand_as(x), generator=rng)
                for key, tensor in dict(t=t, x=x, x_reference=x_reference, value=value, value_reference=exact_value,
                                        mean=dist.loc, mean_reference=exact_mean, std=dist.scale,
                                        std_reference=exact_std, actions=action, jumps=jumps, mask=active).items():
                    records[key].append(tensor.detach().cpu().numpy())
                if k == cfg.time_steps:
                    break
                b, sigma, jump_alpha = self.coefficients(t)
                diffusion = sigma*math.sqrt(self.dt)*torch.randn(x.shape, generator=rng, device=self.device)
                compensated = jumps-rates*self.dt
                step_active = (t+self.dt <= end+1e-6).bool()
                x = torch.where(step_active, x+b*action*self.dt+diffusion+jump_alpha*compensated, x)
                x_reference = torch.where(step_active, x_reference+b*ref_action*self.dt+diffusion+jump_alpha*compensated, x_reference)
                t = initial_t+(k+1)*self.dt
            value_error = math.sqrt(value_num/max(value_den, 1e-12))
            mean_error = math.sqrt(control_num/max(control_den, 1e-12))
            control_error = mean_error if cfg.mode == "control" else kl_sum/max(count, 1.0)
            metrics = {"benchmark_mode": cfg.mode, "value_relative_rmse": value_error,
                       "mean_relative_rmse": mean_error, "control_error": control_error,
                       "control_metric": "relative_rmse" if cfg.mode == "control" else "KL(exact||learned)",
                       "score": (value_error+control_error)/2}
            return metrics, {k: np.stack(v, axis=1) for k, v in records.items()}
        finally:
            self.net_actor.train(previous_actor)
            self.net_critic.train(previous_critic)

    def save_configuration(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        (self.output_folder/"params.json").write_text(json.dumps(self.config.to_params(), indent=2)+"\n")
        source = Path(__file__)
        shutil.copyfile(source, self.output_folder/source.name)
        (self.output_folder/"provenance.json").write_text(json.dumps({
            "script": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "residual": "martingale_subtracted" if self.config.martingale_correction else "raw_td",
            "benchmark_mode": self.config.mode,
        }, indent=2)+"\n")

    def save_checkpoint(self, tag="last"):
        torch.save(self.net_actor.state_dict(), self.output_folder/f"actor_{tag}.pth")
        torch.save(self.net_critic.state_dict(), self.output_folder/f"critic_{tag}.pth")
        if tag == "last":
            state = {"iterations": self.completed_iterations, "optimizer_actor": self.optimizer_actor.state_dict(),
                     "optimizer_critic": self.optimizer_critic.state_dict(), "scheduler_actor": self.scheduler_actor.state_dict(),
                     "scheduler_critic": self.scheduler_critic.state_dict(), "rng": torch.get_rng_state(),
                     "history": self.history, "best_score": self.best_score}
            if self.target_critic is not None:
                state["target_critic"] = self.target_critic.state_dict()
            if self.device.type == "cuda":
                state["cuda_rng"] = torch.cuda.get_rng_state_all()
            torch.save(state, self.output_folder/"training_state.pth")

    def restore_training_state(self, directory):
        path = Path(directory)/"training_state.pth"
        if not path.is_file():
            print("[resume] Historical weights have no optimizer state; starting fresh optimizers from these weights.")
            return
        state = torch.load(path, map_location=self.device, weights_only=True)
        for key in ("optimizer_actor", "optimizer_critic", "scheduler_actor", "scheduler_critic"):
            getattr(self, key).load_state_dict(state[key])
        if self.target_critic is not None and "target_critic" in state:
            self.target_critic.load_state_dict(state["target_critic"])
        self.completed_iterations = state["iterations"]
        self.history = state["history"]
        # A resumed run writes to a new directory and selects its own best pair.
        self.best_score = math.inf
        torch.set_rng_state(state["rng"].cpu())
        if "cuda_rng" in state and self.device.type == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])

    def train_all(self, n=0, N_Iter=None):
        cfg = self.config
        iterations = cfg.iterations if N_Iter is None else N_Iter
        self.save_configuration()
        for _ in range(iterations):
            itr = self.completed_iterations
            losses = self.train_iteration(actor_enabled=itr >= cfg.critic_warmup_iterations)
            self.completed_iterations += 1
            record = {"iteration": self.completed_iterations, **losses}
            if self.completed_iterations % cfg.eval_every == 0 or _ == iterations-1:
                metrics, _trajectory = self.evaluate()
                record.update(metrics)
                print(f"[eval] iteration={self.completed_iterations} value={metrics['value_relative_rmse']:.6g} "
                      f"{metrics['control_metric']}={metrics['control_error']:.6g}")
                if metrics["score"] < self.best_score:
                    self.best_score = metrics["score"]
                    self.save_checkpoint("best")
            self.history.append(record)
        self.save_checkpoint("last")
        (self.output_folder/"history.json").write_text(json.dumps(self.history, indent=2)+"\n")

    def plot_results(self, metrics, trajectory):
        import matplotlib.pyplot as plt

        configure_plot_style()
        cfg = self.config
        r = trajectory
        self.output_folder.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for i in range(min(5, cfg.batch_size)):
            valid = r["mask"][i, :, 0].astype(bool)
            t = r["t"][i, valid, 0]
            label = i == 0
            axes[0].plot(t, r["mean"][i, valid, 0], color="#6565AE", alpha=0.7, label="Learned mean" if label else None)
            axes[0].plot(t, r["mean_reference"][i, valid, 0], "--", color="#E7483D", label="Exact control" if label and cfg.mode == "control" else ("Exact policy mean" if label else None))
            axes[1].plot(t, r["value"][i, valid, 0], color="#6565AE", alpha=0.7, label="Learned value" if label else None)
            axes[1].plot(t, r["value_reference"][i, valid, 0], "--", color="#E7483D", label="Exact value on learned states" if label else None)
            axes[2].plot(t, r["x"][i, valid, 0], color="#6565AE", alpha=0.7, label="Learned rollout" if label else None)
            axes[2].plot(t, r["x_reference"][i, valid, 0], "--", color="#E7483D", label="Exact rollout" if label else None)
        for ax, ylabel in zip(axes, ("Control" if cfg.mode == "control" else "Policy mean", "Value", "State")):
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=10)
        axes[-1].set_xlabel("Time")
        fig.suptitle(f"{cfg.coefficient_type}: {cfg.mode}, entropy alpha={cfg.entropy_alpha:g}")
        fig.tight_layout()
        fig.savefig(self.output_folder/"trajectories.pdf")
        plt.close(fig)
        if cfg.mode == "policy":
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(r["t"][0, :, 0], r["std"][0, :, 0], label="Learned policy std")
            ax.plot(r["t"][0, :, 0], r["std_reference"][0, :, 0], "--", label="Exact policy std")
            ax.set(xlabel="Time", ylabel="Standard deviation")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_folder/"policy_std.pdf")
            plt.close(fig)
        measured = [v for v in self.history if "score" in v]
        if measured:
            fig, ax = plt.subplots(figsize=(8, 4))
            for key, label in (("value_relative_rmse", "Value RMSE"), ("control_error", metrics["control_metric"])):
                ax.semilogy([v["iteration"] for v in measured], [max(v[key], 1e-12) for v in measured], label=label)
            ax.set(xlabel="Iteration", ylabel="Error")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_folder/"training_errors.pdf")
            plt.close(fig)


def build_model(config, actor_path=None, critic_path=None, output_folder=None):
    cfg = config.validate()
    inputs = cfg.dimensions+(4 if cfg.coefficient_type == "periodic" else 1)
    critic = Network(inputs, cfg.width, cfg.depth, output=1)
    actor = Network_mu_sigma_split(inputs, cfg.width, cfg.depth, output=cfg.dimensions,
                                   fixed_std=cfg.fixed_std if cfg.mode == "control" else None)
    return FBSNN(critic, actor, cfg.to_params(), actor_path, critic_path, output_folder)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coefficient-type", choices=["convergence", "convergent", "periodic"])
    parser.add_argument("--entropy-alpha", "--entropy_alpha", type=float)
    parser.add_argument("--actor-loss", choices=["manuscript", "entropy_outside"])
    parser.add_argument("--fixed-std", "--fixed_std", type=float)
    for name in ("horizon", "period", "exact-dt", "truncation-horizon", "actor-lr", "critic-lr", "target-polyak", "initial-state"):
        parser.add_argument("--"+name, type=float)
    for name in ("time-steps", "batch-size", "dimensions", "iterations", "width", "depth", "actor-update-every",
                 "critic-update-every", "critic-warmup-iterations", "eval-every", "seed"):
        parser.add_argument("--"+name, type=int)
    parser.add_argument("--random-time", choices=["none", "start", "end"])
    parser.add_argument("--config", type=Path, help="params.json from a recorded run or a saved config")
    parser.add_argument("--actor-path", "--actor_path", type=Path)
    parser.add_argument("--critic-path", "--critic_path", type=Path)
    parser.add_argument("--resume", action="store_true", help="additional iterations from a saved checkpoint pair")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--no-martingale-correction", action="store_true", help="raw TD ablation")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="build networks and validate the actual benchmark without training")
    parser.add_argument("--T_eval", type=float, help="evaluation rollout horizon")
    parser.add_argument("--N_eval", type=int, help="evaluation rollout steps")
    parser.add_argument("--riccati-steps", type=int, help="legacy option: set benchmark maximum step to period/steps")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if bool(args.actor_path) != bool(args.critic_path):
        parser.error("provide both --actor-path and --critic-path")
    if args.resume and not args.actor_path:
        parser.error("--resume requires an actor/critic checkpoint pair")
    checkpoint_dir = None
    if args.actor_path:
        for path in (args.actor_path, args.critic_path):
            if not path.is_file():
                parser.error(f"checkpoint not found: {path}")
        checkpoint_dir = args.actor_path.resolve().parent
    cfg = ExperimentConfig()
    config_path = args.config or (checkpoint_dir/"params.json" if checkpoint_dir else None)
    if config_path:
        if not config_path.is_file():
            parser.error(f"configuration not found: {config_path}")
        data = json.loads(config_path.read_text())
        cfg = ExperimentConfig.from_params(data) if "equation" in data or "config" in data else ExperimentConfig(**data)
    if checkpoint_dir:
        state = torch.load(args.actor_path, map_location="cpu", weights_only=True)
        # Infer shape-defining settings before constructing the modules. No
        # partial checkpoint load or silent fallback to random weights.
        cfg.width = state["first.weight"].shape[0]
        cfg.depth = len({k.split(".")[1] for k in state if k.startswith("hidden.")})
        if "fixed_log_std" in state:
            cfg.fixed_std = math.exp(float(state["fixed_log_std"]))
        if ("log_std_layer.weight" in state) != (cfg.entropy_alpha > 0):
            parser.error("checkpoint policy architecture does not match entropy mode")
    for field in fields(ExperimentConfig):
        choice = getattr(args, field.name, None)
        if choice is not None:
            setattr(cfg, field.name, choice)
    if args.no_martingale_correction:
        cfg.martingale_correction = False
    if args.riccati_steps is not None:
        if args.riccati_steps <= 0:
            parser.error("--riccati-steps must be > 0")
        cfg.exact_dt = cfg.period/args.riccati_steps
    cfg.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.T_eval is not None:
        cfg.horizon = args.T_eval
    if args.N_eval is not None:
        cfg.time_steps = args.N_eval
    try:
        cfg.validate()
        if checkpoint_dir:
            if ("log_std_layer.weight" in state) != (cfg.entropy_alpha > 0):
                raise ValueError("checkpoint entropy mode cannot be changed when loading weights")
            expected_inputs = cfg.dimensions+(4 if cfg.coefficient_type == "periodic" else 1)
            if state["first.weight"].shape[1] != expected_inputs:
                raise ValueError("checkpoint time encoding does not match coefficient type")
            if state["mean_layer.weight"].shape[0] != cfg.dimensions:
                raise ValueError("checkpoint action dimension does not match dimensions")
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        model = build_model(cfg, args.actor_path, args.critic_path, args.output_dir)
    except (ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"[config] coefficients={cfg.coefficient_type} mode={cfg.mode} alpha={cfg.entropy_alpha:g}")
    if args.dry_run:
        x = torch.full((2, cfg.dimensions), cfg.initial_state, device=model.device)
        t = torch.tensor([[0.0], [cfg.horizon/2]], device=model.device)
        if cfg.mode == "control":
            mean, value = model.compute_exact_no_exploratory(x, t)
            print(f"[benchmark] deterministic mean={mean.flatten().tolist()} value={value.flatten().tolist()}")
        else:
            policy, value = model.compute_exact_entropy(x, t)
            print(f"[benchmark] Gaussian mean={policy.loc.flatten().tolist()} std={policy.scale.flatten().tolist()} value={value.flatten().tolist()}")
        print(f"[dry-run] learns_std={hasattr(model.net_actor, 'log_std_layer')}")
        return model
    if args.resume:
        if args.actor_path.name == "actor_last.pth" and args.critic_path.name == "critic_last.pth":
            model.restore_training_state(checkpoint_dir)
        else:
            print("[resume] This pair has no matching optimizer snapshot; starting fresh optimizers from these weights.")
        for choice, optimizer, scheduler in (
            (args.actor_lr, model.optimizer_actor, model.scheduler_actor),
            (args.critic_lr, model.optimizer_critic, model.scheduler_critic),
        ):
            if choice is not None:
                for group in optimizer.param_groups:
                    group["lr"] = choice
                    group["initial_lr"] = choice
                scheduler.base_lrs = [choice]*len(optimizer.param_groups)
                scheduler._last_lr = [choice]*len(optimizer.param_groups)
    if not args.actor_path or args.resume:
        model.train_all()
    else:
        model.save_configuration()
    metrics, trajectory = model.evaluate()
    (model.output_folder/"evaluation.json").write_text(json.dumps(metrics, indent=2)+"\n")
    np.savez_compressed(model.output_folder/"trajectories.npz", **trajectory)
    if not args.no_plots:
        model.plot_results(metrics, trajectory)
    print(json.dumps(metrics, indent=2))
    print(f"[output] {model.output_folder}")
    return model


if __name__ == "__main__":
    main()
