from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[2]
for _path in (_ROOT, _ROOT / "tools"):
    if str(_path) not in _sys.path:
        _sys.path.insert(0, str(_path))

from experiment_utils import FONT_SIZE, LEGEND_FONT_SIZE, configure_plot_style, configure_policy_output, preserve_random_state
"""Unified time-homogeneous LQ experiment.

``entropy_alpha == 0`` uses the classical feedback control and a mean-only
Gaussian actor with fixed training standard deviation.  ``entropy_alpha > 0``
uses the entropy-regularized Gaussian policy and learns both mean and std.

The default ``manuscript`` actor loss follows manuscript0917.pdf (3.22)-(3.23):
the ``-alpha*log(pi)`` term is inside the stopped advantage.  The historical
``entropy_outside`` formulation remains available as an ablation.
"""
import numpy as np
import time
import torch
import psutil
import json
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.ticker import ScalarFormatter, MultipleLocator, FormatStrFormatter
import math
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm

from torch.distributions import Normal
import argparse
import shutil
import json
import glob 
import copy
import pathlib
import re
from experiment_utils import check_nan, to_scalar
from trajectory_plots import plot_lq_trajectories
from experiment_utils import (
    plot_training_series, plot_actor_critic_loss, plot_policy_diagnostics,
    plot_control_error_heatmap, plot_J_control_function, plot_value_control_errors,
)

def _to_jsonable(x):
    """Convert various types to JSON-serializable format."""
    
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    
    
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    
    
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (torch.device, torch.dtype, torch.Size)):
        return str(x)
    
    
    if isinstance(x, pathlib.Path):
        return str(x)
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if hasattr(x, "__dict__"):
        return _to_jsonable(vars(x))
    
    
    return str(x)

class Block(nn.Module):
    def __init__(self, inputs: int, params: dict, activation="Tanh"):
        super(Block, self).__init__()
        self.L1 = nn.Linear(inputs, inputs)
        self.L2 = nn.Linear(inputs, inputs)
        self.activation = activation
        if activation in params:
            self.act = params[activation]

    def forward(self, x):
        if self.activation == "Sin" or self.activation == "sin":
            a = torch.sin(self.L2(torch.sin(self.L1(x)))) + x
        else:
            a = self.act(self.L1(self.act(self.L2(x)))) + x
        return a

class Network(nn.Module):

    def __init__(self, inputs: int, width: int, depth: int, output: int, params: dict, activation="Tanh", penalty=None):
        super(Network, self).__init__()
        self.first = nn.Linear(inputs, width)
        self.last = nn.Linear(width, output)
        self.network = nn.Sequential(*[
            self.first,
            *[Block(width, params, activation)] * depth,
            self.last
        ])
        self.penalty = penalty
        self.bound = nn.Parameter(torch.tensor(2.))

    def forward(self, x):
        if self.penalty is not None:
            if self.penalty == "Sigmoid":
                sigmoid = nn.Sigmoid()
                return sigmoid(self.network(x)) * self.bound
            elif self.penalty == "Tanh":
                tanh = nn.Tanh()
                return tanh(self.network(x)) * self.bound
            else:
                raise RuntimeError("Other penalty has not bee implemented!")
        else:
            return self.network(x)

class Network_mu_sigma_split(nn.Module):
    """log_std_mode (often used when gamma=0):
      - min_only: lower bound only; unbounded sigma can diverge and impede mean learning.
      - hard_box: clamp sigma to [exp(lo), exp(hi)]; the log_std head has zero gradient at the bounds.
      - ste_box: use hard_box in the forward pass and a straight-through gradient in the backward pass.
        std.json shows clamped values while the underlying weights can still change.
      - sigmoid_box: log_std = lo + (hi-lo)*sigmoid(raw) lies strictly inside (lo, hi).
        Gradients remain available near the bounds; a variance penalty can drive the policy toward a Dirac mass."""
    def __init__(
        self,
        inputs: int,
        width: int,
        depth: int,
        output: int,
        params: dict,
        activation="Tanh",
        penalty=None,
        fixed_std: float | None = None,
        log_std_mode: str = "min_only",
        log_std_min: float = -10.0,
        log_std_max: float = -1.0,
        input_normalization: str = "center_scale",
        state_center=None,
        state_scale=1.0,
        time_scale: float = 1.0,
    ):
        super(Network_mu_sigma_split, self).__init__()
        self.output = output
        self.activation = activation
        self.first = nn.Linear(inputs, width)
        self.hidden = nn.Sequential(*[Block(width, params, activation) for _ in range(depth)])
        
        self.mean_layer = nn.Linear(width, output)
        self.use_fixed_std = fixed_std is not None
        if self.use_fixed_std:
            fixed_std = float(fixed_std)
            if fixed_std <= 0:
                raise ValueError(f"fixed_std must be positive, got {fixed_std!r}")
            self.register_buffer("fixed_log_std", torch.tensor(math.log(fixed_std)))
        else:
            self.log_std_layer = nn.Linear(width, output)
        self.penalty = penalty
        self.bound = nn.Parameter(torch.tensor(1.))
        self.log_std_mode = log_std_mode
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.input_normalization = str(input_normalization)
        if self.input_normalization not in {"none", "x_over_x0", "center_scale"}:
            raise ValueError(f"unknown input_normalization={self.input_normalization!r}")

        state_dim = inputs - 1
        center = torch.as_tensor(
            torch.zeros(state_dim) if state_center is None else state_center,
            dtype=torch.get_default_dtype(),
        ).reshape(-1)
        if center.numel() == 1:
            center = center.expand(state_dim).clone()
        if center.numel() != state_dim:
            raise ValueError(f"state_center must have {state_dim} entries, got {center.numel()}")
        scale = torch.as_tensor(state_scale, dtype=torch.get_default_dtype()).reshape(-1)
        if scale.numel() == 1:
            scale = scale.expand(state_dim).clone()
        if scale.numel() != state_dim or torch.any(scale <= 0):
            raise ValueError(f"state_scale must be positive with 1 or {state_dim} entries")
        if float(time_scale) <= 0:
            raise ValueError("time_scale must be positive")
        # Non-persistent buffers preserve strict loading of historical checkpoints.
        self.register_buffer("state_center", center, persistent=False)
        self.register_buffer("state_scale", scale, persistent=False)
        self.register_buffer("time_scale", torch.tensor(float(time_scale)), persistent=False)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalization == "none":
            return x
        t = x[..., :1] / self.time_scale.to(device=x.device, dtype=x.dtype)
        state = x[..., 1:]
        center = self.state_center.to(device=x.device, dtype=x.dtype)
        scale = self.state_scale.to(device=x.device, dtype=x.dtype)
        if self.input_normalization == "x_over_x0":
            state = state / torch.clamp(center.abs(), min=1e-6)
        else:
            state = (state - center) / scale
        return torch.cat([t, state], dim=-1)

    def _bound_log_std(self, raw: torch.Tensor) -> torch.Tensor:
        lo, hi = self.log_std_min, self.log_std_max
        if self.log_std_mode == "min_only":
            return torch.clamp(raw, min=lo)
        if self.log_std_mode == "hard_box":
            return torch.clamp(raw, lo, hi)
        if self.log_std_mode == "ste_box":
            return raw + (torch.clamp(raw, lo, hi) - raw).detach()
        if self.log_std_mode == "sigmoid_box":
            s = torch.sigmoid(raw)
            eps = 1e-6
            s = s * (1.0 - 2.0 * eps) + eps
            return lo + (hi - lo) * s
        raise ValueError(f"unknown log_std_mode={self.log_std_mode!r}")

    def forward(self, x):
        if self.activation == "Tanh":
            self.act = nn.Tanh()
        elif self.activation == "ReLU":
            self.act = nn.ReLU()
        elif self.activation == "GELU":
            self.act = nn.GELU()
        elif self.activation == "Sin" or self.activation == "sin":
            self.act = torch.sin
        x = self._normalize_input(x)
        x = self.act(self.first(x))
        x = self.hidden(x)
        mean = self.mean_layer(x)
        log_std = self.fixed_log_std.expand_as(mean) if self.use_fixed_std else self.log_std_layer(x)

        if self.penalty is not None:
            if self.penalty == "Tanh":
                mean = torch.tanh(mean) * self.bound
            elif self.penalty == "ScaledTanh":
                mean = (torch.tanh(mean) + 1) / 2 * self.bound
        
        if not self.use_fixed_std:
            log_std = self._bound_log_std(log_std)
        return mean, log_std

class FBSNN(nn.Module): # Forward-Backward Stochastic Neural Network
    def __init__(self, net_value, net_control, params, actor_path: str = None, critic_path: str = None):
        super().__init__()
        
        self.params = copy.deepcopy(params)
        self.params_equ = params["equation"]
        self.params_train = params["train"]
        configure_policy_output(net_control, self.params_train.get("entropy_alpha", 0.0))
        self.device = self.params_train["device"]
        # Ensure Xi is a tensor (handle case where it might be None or missing)
        if "Xi" not in self.params_equ or self.params_equ["Xi"] is None:
            self.params_equ["Xi"] = torch.ones(1, self.params_equ["D"]) * 5
        if isinstance(self.params_equ["Xi"], torch.Tensor):
            self.Xi = self.params_equ["Xi"].to(self.device)
        else:
            self.Xi = torch.tensor(self.params_equ["Xi"], device=self.device)
        self.T = self.params_equ["T"]              # terminal time
        self.M = self.params_equ["M"]             # number of trajectories
        self.N = self.params_equ["N"]              # number of time snapshots
        self.D = self.params_equ["D"]              # number of dimensions
        self.net_critic = net_value.to(self.device)
        self.net_actor = net_control.to(self.device)
        self.optimizer_critic = optim.Adam(self.net_critic.parameters(), lr=self.params_train["lr_critic"])
        self.optimizer_actor = optim.Adam(self.net_actor.parameters(), lr=self.params_train["lr_actor"],
                                          weight_decay=self.params_train["weight_decay"])
        self.scheduler_critic = optim.lr_scheduler.MultiStepLR(self.optimizer_critic,
                                                               milestones=self.params_train["milestones_critic"],
                                                               gamma=self.params_train["gamma_critic"])
        self.scheduler_actor = optim.lr_scheduler.MultiStepLR(self.optimizer_actor,
                                                               milestones=self.params_train["milestones_actor"],
                                                               gamma=self.params_train["gamma_actor"])
        self.best_critic_loss, self.best_actor_loss = 10000, 10000
        # self.terminal = self.params_equ["terminal"].to(self.device) 
        # self.terminal.requires_grad = True
        self.actor_path = actor_path
        self.critic_path = critic_path
        self.Y_error_list, self.control_error_list, self.bound_list = [], [], []
        self.CriticLoss_list, self.ActorLoss_list, self.J_list, self.control_list = [], [], [], []
        self.KLLoss_list, self.StdLoss_list, self.mu_gaussian_loss_list = [], [], []
        self.dt = self.T / self.N


        self.eval_every = self.params_train["eval_every"]  # run evaluation every XX steps training iterations
        self._eval_fixed_ready = False  # whether we have built the fixed eval batch

        # Top-3 eval ckpts: list of dicts like {"iter": int, "score": float, "actor": path, "critic": path}
        self.topk_eval = []

        
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

        print("output_folder================", self.output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        
        if self.actor_path and os.path.isfile(self.actor_path):
            actor_dir = os.path.dirname(os.path.abspath(self.actor_path))
            params_json_path = os.path.join(actor_dir, "params.json")
            if os.path.isfile(params_json_path):
                try:
                    with open(params_json_path, 'r') as f:
                        loaded_params = json.load(f)
                    
                    if "equation" in loaded_params:
                        for key, value in loaded_params["equation"].items():
                            if key == "Xi" and isinstance(value, list):
                                
                                self.params_equ[key] = torch.tensor(value, device=self.device, dtype=torch.float32)
                                self.Xi = self.params_equ[key]
                            elif key in ["lambda", "z"] and isinstance(value, list):
                                
                                self.params_equ[key] = value
                            else:
                                self.params_equ[key] = value
                    if "train" in loaded_params:
                        for key, value in loaded_params["train"].items():
                            if key == "device" and isinstance(value, str):
                                
                                self.params_train[key] = value
                            else:
                                self.params_train[key] = value
                    
                    if "T" in self.params_equ:
                        self.T = self.params_equ["T"]
                    if "M" in self.params_equ:
                        self.M = self.params_equ["M"]
                    if "N" in self.params_equ:
                        self.N = self.params_equ["N"]
                    if "D" in self.params_equ:
                        self.D = self.params_equ["D"]
                    self.dt = self.T / self.N
                    
                    self.params["equation"] = self.params_equ
                    self.params["train"] = self.params_train
                    print(f"[load] params.json from {params_json_path}")
                except Exception as e:
                    print(f"[warn] failed to load params.json from {params_json_path}: {e}")

        
        if self.actor_path and os.path.isfile(self.actor_path):
            try:
                self.net_actor.load_state_dict(torch.load(self.actor_path, map_location=self.device))
                self.net_actor.eval()
                print(f"[load] actor from given path: {self.actor_path}")
            except Exception as e:
                print(f"[warn] failed to load actor from {self.actor_path}: {e}")

        if self.critic_path and os.path.isfile(self.critic_path):
            try:
                self.net_critic.load_state_dict(torch.load(self.critic_path, map_location=self.device))
                self.net_critic.eval()
                print(f"[load] critic from given path: {self.critic_path}")
            except Exception as e:
                print(f"[warn] failed to load critic from {self.critic_path}: {e}")



    # =========================
    # EVALUATION HELPERS
    # =========================

    def _ensure_eval_set(self):
        """
        [ADDED] Build a fixed evaluation batch once for all future evaluations.
        Using a fixed RNG seed ensures comparability across iterations.
        """
        if self._eval_fixed_ready:
            return
        # Backup RNG states
        torch_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        try:
            # torch.manual_seed(123456)
            # np.random.seed(123456)
            torch.manual_seed(66)
            np.random.seed(66)
            with torch.no_grad():
                t, tau, W, P = self.fetch_minibatch()
            # Persist the fixed eval batch inside the model
            self.t_eval, self.tau_eval, self.W_eval, self.P_eval = t, tau, W, P
            self._eval_fixed_ready = True
        finally:
            # Restore RNG states
            torch.random.set_rng_state(torch_state)
            np.random.set_state(np_state)

    @preserve_random_state
    @torch.no_grad()
    def _evaluate_on_fixed(self):
        """
        [ADDED] Run one forward-only evaluation on the fixed batch and return a scalar score.
        Score definition: Y_error by default; if control_error exists, use average(Y_error, control_error).
        Lower is better.
        """
        self._ensure_eval_set()
        self.t_predict_set, self.tau_predict_set = self.t_eval, self.tau_eval
        self.W_predict_set, self.P_predict_set   = self.W_eval, self.P_eval

        # Use your forward-only path (traj_only=True, predict=True) to compute errors
        _, _, _, _, _, Y_error, control_error, _, _, mean_error = self.train_loop(
            itr=-1, traj_only=True, predict=True
        )

        # Compose the scalar evaluation score
        score = self._to_float(Y_error)
        if control_error is not None:
            score = 0.5 * (score + self._to_float(control_error))

        return score, {
            "Y_error": self._to_float(Y_error),
            "control_error": (None if control_error is None else self._to_float(control_error)),
            "mu_gaussian_error": self._to_float(mean_error),
        }

    @staticmethod
    def _to_float(x):
        """
        [ADDED] Safe float conversion for Tensors/ndarrays/scalars.
        """
        try:
            if torch.is_tensor(x):
                return float(x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().mean().item())
            if isinstance(x, np.ndarray):
                return float(x.reshape(-1).mean())
            return float(x)
        except Exception:
            return math.inf

    def _extract_iter_from_ckpt_path(self, actor_path: str) -> int:
        """Extract the iteration from a checkpoint path. Supported names:
        - actor_eval_it{iter}.pth
        - actor_eval_top{rank}_itr{iter}.pth
        - actor.pth (look up eval_topk_meta.json)
        Return -1 if the iteration cannot be determined."""
        if not actor_path:
            return -1
        
        
        filename = os.path.basename(actor_path)
        
        
        match = re.search(r'_it(\d+)\.pth$', filename)
        if match:
            return int(match.group(1))
        
        
        match = re.search(r'_itr(\d+)\.pth$', filename)
        if match:
            return int(match.group(1))
        
        
        actor_dir = os.path.dirname(os.path.abspath(actor_path))
        meta_path = os.path.join(actor_dir, "eval_topk_meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta_list = json.load(f)
                
                for rec in meta_list:
                    if rec.get("actor") == actor_path or os.path.abspath(rec.get("actor", "")) == os.path.abspath(actor_path):
                        return int(rec.get("iter", -1))
            except Exception as e:
                print(f"[warn] failed to read meta to extract iter: {e}")
        
        return -1

    def _eval_ckpt_unique_paths(self, it: int):
        """
        [ADDED] Build unique file names for the current iteration's eval ckpt.
        We save actor/critic separately, allowing later aliasing to top1/2/3.
        """
        actor_path  = os.path.join(self.output_folder, f"actor_eval_it{it}.pth")
        critic_path = os.path.join(self.output_folder, f"critic_eval_it{it}.pth")
        return actor_path, critic_path

    def _persist_topN_alias(self):
        """
        Refresh stable aliases actor_eval_top1/2/3.pth and critic_eval_top1/2/3.pth
        Remove any old alias files, then write exactly top1/top2/top3.
        Refresh alias files with iteration numbers in their filenames.
        based on current self.topk_eval (sorted by ascending score).
        Also dump a small JSON meta file for bookkeeping.
        """
        self.topk_eval.sort(key=lambda d: d["score"])  # ascending

        # --- remove old alias files so we keep EXACTLY 3 pairs on disk ---
        try:
            for p in glob.glob(os.path.join(self.output_folder, "actor_eval_top*.pth")):
                os.remove(p)
            for p in glob.glob(os.path.join(self.output_folder, "critic_eval_top*.pth")):
                os.remove(p)
        except Exception as e:
            print(f"[warn] cleanup old top alias files failed: {e}")

        # --- write fresh top1/top2/top3 with iteration number in the filename ---
        for rank, rec in enumerate(self.topk_eval[:3], start=1):
            itr = rec["iter"]
            a_dst = os.path.join(self.output_folder, f"actor_eval_top{rank}_itr{itr}.pth")
            c_dst = os.path.join(self.output_folder, f"critic_eval_top{rank}_itr{itr}.pth")
            try:
                shutil.copyfile(rec["actor"], a_dst)
                shutil.copyfile(rec["critic"], c_dst)
            except Exception as e:
                print(f"[warn] copying top{rank} aliases failed: {e}")

        
        try:
            best = self.topk_eval[0]
            shutil.copyfile(best["actor"], os.path.join(self.output_folder, "actor_eval_top1.pth"))
            shutil.copyfile(best["critic"], os.path.join(self.output_folder, "critic_eval_top1.pth"))
        except Exception as e:
            print(f"[warn] write top1 quick alias failed: {e}")

        
        meta_path = os.path.join(self.output_folder, "eval_topk_meta.json")
        try:
            with open(meta_path, "w") as f:
                json.dump(self.topk_eval[:3], f, indent=2)
        except Exception as e:
            print(f"[warn] write meta failed: {e}")

    def _load_best_eval_for_plot(self):
        """Prefer weights already loaded from actor_path/critic_path in __init__,
        preserving the current output_folder. Otherwise try the eval Top-1 aliases;
        if unavailable, retain the in-memory weights and default output directory."""
        if (self.actor_path and os.path.isfile(self.actor_path)) or \
           (self.critic_path and os.path.isfile(self.critic_path)):
            print("[load] using explicitly provided actor/critic paths; skip eval top1 lookup.")
            
            self.net_actor.eval(); self.net_critic.eval()
            return

        a_top1 = os.path.join(self.output_folder, "actor_eval_top1.pth")
        c_top1 = os.path.join(self.output_folder, "critic_eval_top1.pth")

        if os.path.isfile(a_top1) and os.path.isfile(c_top1):
            try:
                self.net_actor.load_state_dict(torch.load(a_top1, map_location=self.device))
                self.net_critic.load_state_dict(torch.load(c_top1, map_location=self.device))
                self.net_actor.eval(); self.net_critic.eval()
                print("[load] using eval top1 ckpt for plotting.")
                
            except Exception as e:
                print(f"[warn] failed to load eval top1 ckpt: {e}")
        else:
            print("[info] no eval top1 ckpt found; using in-memory weights.")
            
            if (not (self.actor_path or self.critic_path)) and \
               (self.output_folder == "" or not os.path.isdir(self.output_folder)):
                self.output_folder = self._default_output_folder()
                os.makedirs(self.output_folder, exist_ok=True)

    def _save_params_json(self, directory: str):
        """Save params.json to the specified directory."""
        params_json_path = os.path.join(directory, "params.json")
        try:
            params_to_save = {
                "equation": _to_jsonable(self.params_equ),
                "train": _to_jsonable(self.params_train)
            }
            with open(params_json_path, 'w') as f:
                json.dump(params_to_save, f, ensure_ascii=False, indent=2)
            print(f"[save] params.json to {params_json_path}")
            self._save_run_script(directory)
        except Exception as e:
            print(f"[warn] failed to save params.json to {directory}: {e}")

    def _append_std_json(self, it: int, mu_logsigma_buffer: torch.Tensor):
        """
        Record Gaussian policy std statistics during training to {output_folder}/std.json.

        mu_logsigma_buffer: [N+1, M, 2*D] where last D dims are log_sigma.
        """
        try:
            if not hasattr(self, "output_folder") or not self.output_folder:
                return
            if mu_logsigma_buffer is None or (not torch.is_tensor(mu_logsigma_buffer)):
                return
            if mu_logsigma_buffer.numel() == 0:
                return

            D = int(getattr(self, "D", 1))
            if mu_logsigma_buffer.shape[-1] < 2 * D:
                return

            log_sigma = mu_logsigma_buffer[..., D:2 * D].detach()
            sigma = torch.exp(log_sigma)

            log_sigma_flat = log_sigma.reshape(-1)
            sigma_flat = sigma.reshape(-1)

            rec = {
                "it": int(it),
                "log_sigma_mean": float(log_sigma_flat.mean().cpu().item()),
                "log_sigma_min": float(log_sigma_flat.min().cpu().item()),
                "log_sigma_max": float(log_sigma_flat.max().cpu().item()),
                "sigma_mean": float(sigma_flat.mean().cpu().item()),
                "sigma_min": float(sigma_flat.min().cpu().item()),
                "sigma_max": float(sigma_flat.max().cpu().item()),
            }

            path = os.path.join(self.output_folder, "std.json")
            try:
                if os.path.isfile(path):
                    with open(path, "r") as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                else:
                    data = []
            except Exception:
                data = []

            data.append(rec)
            with open(path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[warn] failed to append std.json: {e}")

    def _save_run_script(self, save_dir: str):
        """Copy the current script to the output directory to preserve its evaluation code version."""
        try:
            src = os.path.abspath(__file__)
            dst = os.path.join(save_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            print(f"[save] run script copied to {dst}")
            return True
        except Exception as e:
            print(f"[warn] failed to copy run script: {e}")
            return False

    def _maybe_update_top3_eval(self, it: int, score: float):
        """Save current weights when the score qualifies for the Top-3 evaluation list.
        During the final two iterations, also check whether a fallback tail checkpoint is needed."""
        updated = False

        if len(self.topk_eval) < 3 or score < max(self.topk_eval, key=lambda d: d["score"])["score"]:
            a_path, c_path = self._eval_ckpt_unique_paths(it)
            try:
                torch.save(self.net_actor.state_dict(),  a_path)
                torch.save(self.net_critic.state_dict(), c_path)
                
                ckpt_dir = os.path.dirname(os.path.abspath(a_path))
                self._save_params_json(ckpt_dir)
            except Exception as e:
                print(f"[warn] saving eval ckpt failed: {e}")
                
            else:
                entry = {"iter": int(it), "score": float(score),
                         "actor": a_path, "critic": c_path}

                if len(self.topk_eval) < 3:
                    self.topk_eval.append(entry)
                else:
                    worst_idx = max(range(3), key=lambda i: self.topk_eval[i]["score"])
                    old = self.topk_eval[worst_idx]
                    self.topk_eval[worst_idx] = entry
                    try:
                        if os.path.isfile(old["actor"]):  os.remove(old["actor"])
                        if os.path.isfile(old["critic"]): os.remove(old["critic"])
                    except Exception as e:
                        print(f"[warn] cleanup old eval ckpt failed: {e}")

                self._persist_topN_alias()
                print(f"[eval] accepted to Top-3: iter={it}, score={score:.6g}")
                updated = True
        else:
            print(f"[eval] not better than current Top-3: iter={it}, score={score:.6g}")

        
        self._maybe_save_tail_ckpt(it)

    def _maybe_save_tail_ckpt(self, it: int):
        """Save an additional checkpoint during the final two iterations when:
        1) it >= self.params_train["iteration"] - 1;
        2) every existing Top-3 iteration lies in the first half of training;
        3) the corresponding tail checkpoint has not already been saved."""
        
        if it < self.params_train["iteration"] - 1:
            return
        if not self.topk_eval:
            return

        
        top_iters = [rec["iter"] for rec in self.topk_eval[:min(3, len(self.topk_eval))]]
        if not top_iters:
            return

        cutoff = self.params_train["iteration"] / 2.0

        
        if max(top_iters) <= cutoff:
            
            a_path = os.path.join(self.output_folder, f"actor_eval_tail_it{it}.pth")
            c_path = os.path.join(self.output_folder, f"critic_eval_tail_it{it}.pth")

            
            if not os.path.exists(a_path) and not os.path.exists(c_path):
                try:
                    torch.save(self.net_actor.state_dict(),  a_path)
                    torch.save(self.net_critic.state_dict(), c_path)
                    
                    self._save_params_json(self.output_folder)
                    print(f"[eval] saved tail ckpt at it={it} (Top-3 all in first half).")
                except Exception as e:
                    print(f"[warn] saving tail ckpt (tail it={it}) failed: {e}")


    ###############
    ## Problem Functions
    ##############
    # def compute_exact_entropy(self, x):
    #     """
    
    
    #     - optimal_policy: 
    
    
    
    #     """
    #     if not torch.is_tensor(x):
    #         x = torch.as_tensor(x, device=self.device, dtype=torch.get_default_dtype())
    #     device, dtype = x.device, x.dtype

    #     mu     = torch.as_tensor(self.params_equ["mu"],     device=device, dtype=dtype)
    #     sigma  = torch.as_tensor(self.params_equ["sigma"],  device=device, dtype=dtype)
    #     lam    = torch.as_tensor(self.params_equ["lambda"], device=device, dtype=dtype)
    #     z      = torch.as_tensor(self.params_equ["z"],      device=device, dtype=dtype)
    #     q1     = torch.as_tensor(self.params_equ["q1"],     device=device, dtype=dtype)
    #     q2     = torch.as_tensor(self.params_equ["q2"],     device=device, dtype=dtype)
    #     beta   = torch.as_tensor(self.params_equ["beta"],   device=device, dtype=dtype)
    #     gamma_entropy = torch.as_tensor(self.params_train["entropy_alpha"],
    #                                     device=device, dtype=dtype)
    #     p_tsallis = torch.as_tensor(self.params_train["p_tsallis"],
    #                                 device=device, dtype=dtype)

    
    #     if x.dim() == 3 and x.size(-1) == 1 and x.size(0) == x.size(1):
    #         target_shape = x.shape[:-1]     # [M, M]
    #         x_body = x.squeeze(-1)          # [M, M]
    #     else:
    
    

    
    
    #     # ============================================
    
    
    
    #     # Tr(ΣΣ^T H) = h * σ^2 * D
    #     # λα^T Hα = h * sum(λ_d * z_d^2) for d=1..D
    
    
    
    #     if isinstance(lam, torch.Tensor) and lam.ndim > 0:
    
    
    
    
    #     else:
    
    #         kappa = sigma**2 + lam * z**2
        
    #     one = torch.as_tensor(1.0, device=device, dtype=dtype)

    #     # =========================
    #     # Shannon Entropy (p = 1)
    #     # =========================
    #     if self.params_train["p_tsallis"] == 1:
    #         sqrt_term = beta**2 * q1**2 + 4.0 * q1 * q2 * mu**2
    #         h = (beta * q1 - torch.sqrt(sqrt_term)) / (2.0 * mu**2)

    
    #         C_shannon = 0.5 * gamma_entropy * (torch.log(torch.pi * gamma_entropy) - torch.log(q1))
    
    
    
    
    #     # Tr(ΣΣ^T H) = h * σ^2 * D
    #     # λα^T Hα = h * sum(λ_d * z_d^2) for d=1..D
    
    #     # ====================================================================
    #         if isinstance(kappa, torch.Tensor) and kappa.ndim > 0:
    
    #             # kappa_d = sigma^2 + lambda_d * z_d^2
    
    
    
    #         else:
    
    #             g = (kappa * h + C_shannon) / beta

    
    #         mu_u = (mu * h / q1) * x_body
    #         var_u = gamma_entropy / (2.0 * q1)
    #         scale = torch.sqrt(var_u)

    
    #         mu_u = mu_u.reshape(target_shape)
    #         scale = scale.expand_as(mu_u)
    
    
    
    
    
    #         # ====================================================================
    
    #         x_squared_sum = torch.sum(x_body**2, dim=-1, keepdim=True)  # [..., 1]
    #         optimal_value_scalar = h * x_squared_sum + g
    
    
    
    #         if target_shape != x_squared_sum.shape:
    #             optimal_value = optimal_value_scalar.expand(target_shape)
    #         else:
    #             optimal_value = optimal_value_scalar.reshape(target_shape)

    #         optimal_policy = Normal(loc=mu_u, scale=scale)

    #     # =========================
    #     # Tsallis Entropy (p > 1)
    #     # =========================
    #     else:
    #         p = p_tsallis
    #         # α = 1 / (p - 1)
    #         alpha = one / (p - one)

    
    #         # term = ((p-1)/(p γ))^(1/(p-1)) * sqrt(pi/q1) * Γ(α+1)/Γ(α+3/2)
    #         log_term = (
    #             (one / (p - one)) * (torch.log(p - one) - torch.log(p) - torch.log(gamma_entropy))
    #             + 0.5 * (torch.log(torch.tensor(np.pi)) - torch.log(q1))
    #             + (torch.lgamma(alpha + one) - torch.lgamma(alpha + 1.5))
    #         )
    #         # B_p = term^(-2(p-1)/(p+1))
    #         B_p = torch.exp(-2.0 * (p - one) / (p + one) * log_term)

    
    #         R2 = B_p / q1
    #         Var_p = R2 / (2.0 * alpha + 3.0)

    #         # ---- C_p(q1, γ) ----
    #         C_p = ((p - one) / p) * q1 * Var_p - (one / p) * B_p

    
    #         sqrt_term = beta**2 * q1**2 + 4.0 * q1 * q2 * mu**2
    #         h = (beta * q1 - torch.sqrt(sqrt_term)) / (2.0 * mu**2)
    
    
    #         # ====================================================================
    #         if isinstance(kappa, torch.Tensor) and kappa.ndim > 0:
    #             kappa_total = torch.sum(kappa)
    
    #         else:
    #             g = (kappa * h + C_p) / beta

    
    

    
    
    
    #         scale = torch.sqrt(Var_p)

    #         mu_u = mu_u.reshape(target_shape)
    #         scale = scale.expand_as(mu_u)
    
    
    #         # ====================================================================
    #         x_squared_sum = torch.sum(x_body**2, dim=-1, keepdim=True)  # [..., 1]
    #         optimal_value_scalar = h * x_squared_sum + g
    #         if target_shape != x_squared_sum.shape:
    #             optimal_value = optimal_value_scalar.expand(target_shape)
    #         else:
    #             optimal_value = optimal_value_scalar.reshape(target_shape)

    #         optimal_policy = Normal(loc=mu_u, scale=scale)

    #     self.h_t = h
    #     self.g_t = g
    
    
    #     # ====================================================================
    #     if hasattr(self, 'Xi') and self.Xi.shape[1] > 1:
    
    #         self.Y0 = h * torch.sum(self.Xi[0, :] ** 2) + g
    #     else:
    
    #         self.Y0 = h * (self.Xi[0, 0] ** 2) + g

    #     return optimal_policy, optimal_value

    # # def compute_exact_entropy(self, x):
    # #     """
    
    
    
    
    # #     """
    # #     if not torch.is_tensor(x):
    # #         x = torch.as_tensor(x, device=self.device, dtype=torch.get_default_dtype())
    # #     device, dtype = x.device, x.dtype
    # #     mu     = torch.as_tensor(self.params_equ["mu"],  device=device, dtype=dtype)
    # #     sigma  = torch.as_tensor(self.params_equ["sigma"],  device=device, dtype=dtype)
    # #     lam    = torch.as_tensor(self.params_equ["lambda"], device=device, dtype=dtype)
    # #     z      = torch.as_tensor(self.params_equ["z"],      device=device, dtype=dtype)
    # #     q1     = torch.as_tensor(self.params_equ["q1"],     device=device, dtype=dtype)
    # #     q2     = torch.as_tensor(self.params_equ["q2"],     device=device, dtype=dtype)
    # #     beta   = torch.as_tensor(self.params_equ["beta"],   device=device, dtype=dtype)
    # #     gamma_entropy = torch.as_tensor(self.params_train["entropy_alpha"], device=device, dtype=dtype)

    # #     if x.dim() == 3 and x.size(-1) == 1 and x.size(0) == x.size(1):
    # #         target_shape = x.shape[:-1]     # [M, M]
    # #         x_body = x.squeeze(-1)          # [M, M]
    # #     else:
    
    

    # #     kappa = sigma**2 + lam * z**2  
    # #     sqrt_term = beta**2 * q1**2 + 4 * q1*q2 * mu**2
    # #     self.h_t = (beta * q1 - torch.sqrt(sqrt_term)) / (2 * mu**2)
    # #     self.g_t = (kappa * self.h_t + (gamma_entropy/2) * (torch.log(torch.pi * gamma_entropy)- torch.log(q1))) / beta
    # #     mu_u = (mu * self.h_t / q1) * x_body
    # #     scale = torch.sqrt(gamma_entropy / (2 * q1))
    # #     optimal_policy = torch.distributions.Normal(loc=mu_u, scale=scale)
    # #     optimal_value  = self.h_t * (x_body**2) + self.g_t  
    # #     self.Y0 = self.h_t * ( (self.Xi[0, 0])**2 ) + self.g_t

    # #     return optimal_policy, optimal_value

    def reward_func_entropy(self, x, dt, control, entropy):
        """
        scalar cost with entropy: q1 ||u||^2 + q2 ||x||^2 - gamma_entropy * entropy -> return -cost*dt, shape [...,1]
        """
        device = x.device if torch.is_tensor(x) else self.device
        dtype = x.dtype if torch.is_tensor(x) else torch.get_default_dtype()

        q1 = self._to_tensor(self.params_equ["q1"], device, dtype)
        q2 = self._to_tensor(self.params_equ["q2"], device, dtype)
        gamma_entropy = self._to_tensor(self.params_train["entropy_alpha"], device, dtype)
        
        q1_s = self._scalar(q1, "q1")
        q2_s = self._scalar(q2, "q2")
        gamma_s = self._scalar(gamma_entropy, "entropy_alpha")

        x_t = x if torch.is_tensor(x) else torch.as_tensor(x, device=device, dtype=dtype)
        u_t = control if torch.is_tensor(control) else torch.as_tensor(control, device=device, dtype=dtype)
        entropy_t = entropy if torch.is_tensor(entropy) else torch.as_tensor(entropy, device=device, dtype=dtype)
        
        if len(entropy_t.shape) == 1:
            entropy_t = entropy_t.unsqueeze(-1)
        
        
        
        
        
        
        
        # ====================================================================
        # f - alpha*log(pi) = -quadratic_cost + alpha*entropy.
        # Since entropy_t = -log(pi), the corresponding cost is
        # quadratic_cost - alpha*entropy_t.
        cost = (
            q1_s * torch.sum(u_t * u_t, dim=-1, keepdim=True)
            + q2_s * torch.sum(x_t * x_t, dim=-1, keepdim=True)
            - gamma_s * entropy_t
        )
        return -cost * dt

    # def compute_exact_no_exploratory(self, x):
    #     if not torch.is_tensor(x):
    #         x = torch.as_tensor(x, device=self.device, dtype=torch.get_default_dtype())
    #     device, dtype = x.device, x.dtype
    #     mu     = torch.as_tensor(self.params_equ["mu"],  device=device, dtype=dtype)
    #     sigma  = torch.as_tensor(self.params_equ["sigma"],  device=device, dtype=dtype)
    #     lam    = torch.as_tensor(self.params_equ["lambda"], device=device, dtype=dtype)
    #     z      = torch.as_tensor(self.params_equ["z"],      device=device, dtype=dtype)
    #     q1     = torch.as_tensor(self.params_equ["q1"],     device=device, dtype=dtype)
    #     q2     = torch.as_tensor(self.params_equ["q2"],     device=device, dtype=dtype)
    #     beta   = torch.as_tensor(self.params_equ["beta"],   device=device, dtype=dtype)

    #     if x.dim() == 3 and x.size(-1) == 1 and x.size(0) == x.size(1):
    #         target_shape = x.shape[:-1]     # [M, M]
    #         x_body = x.squeeze(-1)          # [M, M]
    #     else:
    
    #         x_body = x                      
    
    #     # kappa = sigma**2 + lam * z**2  
    #     # sqrt_term = beta**2 * q1**2 + 4 * q1*q2 * mu**2
    #     # self.h_t = (beta * q1 - torch.sqrt(sqrt_term)) / (2 * mu**2)
    #     # self.g_t = kappa * self.h_t / beta
    #     # optimal_control = (mu * self.h_t / q1) * x_body
    #     # optimal_value  = self.h_t * (x_body**2) + self.g_t
    #     # self.Y0 = self.h_t * ( (self.Xi[0, 0])**2 ) + self.g_t
    #     # ============================================
    
    
    #     if isinstance(lam, torch.Tensor) and lam.ndim > 0:
    
    
    #     else:
    
    #         kappa = sigma**2 + lam * z**2
        
    #     sqrt_term = beta**2 * q1**2 + 4 * q1*q2 * mu**2
    #     self.h_t = (beta * q1 - torch.sqrt(sqrt_term)) / (2 * mu**2)
        
    
    
    #     # ====================================================================
    #     if isinstance(kappa, torch.Tensor) and kappa.ndim > 0:
    #         kappa_total = torch.sum(kappa)
    
    #     else:
    #         self.g_t = kappa * self.h_t / beta

    #     optimal_control = (mu * self.h_t / q1) * x_body
    
    
    #     # ====================================================================
    #     x_squared_sum = torch.sum(x_body**2, dim=-1, keepdim=True)  # [..., 1]
    #     optimal_value_scalar = self.h_t * x_squared_sum + self.g_t
    #     if target_shape != x_squared_sum.shape:
    #         optimal_value = optimal_value_scalar.expand(target_shape)
    #     else:
    #         optimal_value = optimal_value_scalar.reshape(target_shape)
        
    
    #     if hasattr(self, 'Xi') and self.Xi.shape[1] > 1:
    #         self.Y0 = self.h_t * torch.sum(self.Xi[0, :] ** 2) + self.g_t
    #     else:
    #         self.Y0 = self.h_t * (self.Xi[0, 0] ** 2) + self.g_t
    #     return optimal_control, optimal_value



    @staticmethod
    def _to_tensor(v, device, dtype):
        return torch.as_tensor(v, device=device, dtype=dtype)

    @staticmethod
    def _scalar(t: torch.Tensor, name: str) -> torch.Tensor:
        """Force scalar tensor (numel==1) for formulas that require scalar."""
        if not torch.is_tensor(t):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if t.numel() != 1:
            raise ValueError(f"{name} must be scalar (numel==1), got shape={tuple(t.shape)}.")
        return t.reshape(())

    @staticmethod
    def _trace_SigmaSigmaT(sigma: torch.Tensor, D: int) -> torch.Tensor:
        """
        Σ = σ I (sigma scalar) or Σ = diag(sigma) (sigma vector length D)
        Return Tr(ΣΣ^T) = D*sigma^2 or sum_i sigma_i^2
        """
        if sigma.numel() == 1:
            s = sigma.reshape(())
            return (s * s) * D
        sv = sigma.reshape(-1)
        return torch.sum(sv * sv)

    @staticmethod
    def _sum_lambda_z2(lam: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return sum_i lam_i * z_i^2; allow scalar lam/z."""
        if lam.numel() == 1 and z.numel() == 1:
            l = lam.reshape(())
            zz = z.reshape(())
            return l * (zz * zz)
        lamv = lam.reshape(-1)
        zv = z.reshape(-1)
        # broadcast if one is scalar
        if lamv.numel() == 1 and zv.numel() > 1:
            lamv = lamv.expand_as(zv)
        if zv.numel() == 1 and lamv.numel() > 1:
            zv = zv.expand_as(lamv)
        if lamv.numel() != zv.numel():
            raise ValueError(f"lambda and z length mismatch: {lamv.numel()} vs {zv.numel()}")
        return torch.sum(lamv * (zv * zv))

    @staticmethod
    def _log_det_R_from_q1(q1: torch.Tensor, m: int) -> torch.Tensor:
        """
        If R=q1 I_m (q1 scalar): log det R = m log q1
        If R=diag(q1) (q1 vector length m): log det R = sum log q1_i
        """
        if q1.numel() == 1:
            return m * torch.log(q1.reshape(()))
        qv = q1.reshape(-1)
        if qv.numel() != m:
            raise ValueError(f"q1 vector length must equal m, got {qv.numel()} vs m={m}")
        return torch.sum(torch.log(qv))

    def _infer_state_dim(self, x_body: torch.Tensor, lam: torch.Tensor = None) -> int:
        """
        Robust D inference:
          1) self.D (if set)
          2) params_equ["D"] (if set)
          3) len(lambda) (if lambda is vector)
          4) x_body.shape[-1]
        """
        if hasattr(self, "D") and isinstance(self.D, int) and self.D > 0:
            return int(self.D)

        if hasattr(self, "params_equ") and isinstance(self.params_equ, dict) and "D" in self.params_equ:
            D0 = int(self.params_equ["D"])
            if D0 > 0:
                return D0

        if lam is not None and torch.is_tensor(lam) and lam.numel() > 1:
            return int(lam.numel())

        if x_body.dim() >= 1:
            return int(x_body.shape[-1])

        return 1

    @staticmethod
    def _canonicalize_x(x: torch.Tensor) -> torch.Tensor:
        """
        Keep your old compatibility:
          if x is [M,M,1] -> squeeze last dim -> [M,M]
          else keep unchanged.
        """
        if x.dim() == 3 and x.size(-1) == 1 and x.size(0) == x.size(1):
            return x.squeeze(-1)
        return x

    # ============================================================
    # Exact solutions (entropy / no-exploratory)
    # ============================================================
    def compute_exact_entropy(self, x):
        """
        Strict theory (single-player, separable high-dim, B=mu I, R=q1 I, Q=q2 I, H=h I):
          V(x) scalar: V = h ||x||^2 + g
          beta*g = h*( Tr(ΣΣ^T) + sum_i lambda_i z_i^2 ) + c_gamma (Shannon)
          policy: N( (mu h/q1) x , (gamma/2) R^{-1} )
        Return:
          optimal_policy: Normal with batch shape == x_body shape
          optimal_value : scalar, shape == batch + (1,)
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=self.device, dtype=torch.get_default_dtype())
        device, dtype = x.device, x.dtype

        # tensors
        mu = self._to_tensor(self.params_equ["mu"], device, dtype)
        sigma = self._to_tensor(self.params_equ["sigma"], device, dtype)
        lam = self._to_tensor(self.params_equ["lambda"], device, dtype)
        z = self._to_tensor(self.params_equ["z"], device, dtype)
        q1 = self._to_tensor(self.params_equ["q1"], device, dtype)
        q2 = self._to_tensor(self.params_equ["q2"], device, dtype)
        beta = self._to_tensor(self.params_equ["beta"], device, dtype)

        gamma_entropy = self._to_tensor(self.params_train["entropy_alpha"], device, dtype)
        p_tsallis = self._to_tensor(self.params_train["p_tsallis"], device, dtype)

        # force scalars where theory assumes scalar
        mu_s = self._scalar(mu, "mu")
        q1_s = self._scalar(q1, "q1")
        q2_s = self._scalar(q2, "q2")
        beta_s = self._scalar(beta, "beta")
        gamma_s = self._scalar(gamma_entropy, "entropy_alpha")

        x_body = self._canonicalize_x(x)

        D = self._infer_state_dim(x_body, lam=lam)
        m = D  # control dim = state dim in your setup

        # Riccati: (mu^2/q1) h^2 - beta h - q2 = 0  -> use stabilizing branch
        sqrt_term = beta_s * beta_s * q1_s * q1_s + 4.0 * q1_s * q2_s * mu_s * mu_s
        h = (beta_s * q1_s - torch.sqrt(sqrt_term)) / (2.0 * mu_s * mu_s)

        # kappa_total = Tr(ΣΣ^T) + sum_i lambda_i z_i^2
        tr_SigmaSigmaT = self._trace_SigmaSigmaT(sigma, D)
        sum_lam_z2 = self._sum_lambda_z2(lam, z)
        kappa_total = tr_SigmaSigmaT + sum_lam_z2  # scalar

        # choose entropy type
        p_val = float(p_tsallis.reshape(()).item()) if p_tsallis.numel() == 1 else float(p_tsallis.reshape(-1)[0].item())

        if p_val == 1.0:
            # Shannon: c_gamma = (gamma/2) ( m log(pi gamma) - log det R )
            log_det_R = self._log_det_R_from_q1(q1_s, m)
            c_gamma = 0.5 * gamma_s * (m * torch.log(torch.pi * gamma_s) - log_det_R)
            g = (h * kappa_total + c_gamma) / beta_s

            # policy
            mu_u = (mu_s * h / q1_s) * x_body
            var_u = gamma_s / (2.0 * q1_s)  # scalar per-dim
            scale = torch.sqrt(var_u).expand_as(mu_u)
            optimal_policy = Normal(loc=mu_u, scale=scale)

        else:
            # Tsallis branch: keep your “Normal with same mean/variance” approximation,
            # and aggregate constant term by m (independent product approximation).
            one = torch.as_tensor(1.0, device=device, dtype=dtype)
            p = p_tsallis.reshape(()) if p_tsallis.numel() == 1 else p_tsallis.reshape(-1)[0]
            alpha = one / (p - one)

            # 1D constants (same as your old code), then multiply by m
            log_term_1d = (
                (one / (p - one)) * (torch.log(p - one) - torch.log(p) - torch.log(gamma_s))
                + 0.5 * (torch.log(torch.pi) - torch.log(q1_s))
                + (torch.lgamma(alpha + one) - torch.lgamma(alpha + 1.5))
            )
            B_p_1d = torch.exp(-2.0 * (p - one) / (p + one) * log_term_1d)
            R2_1d = B_p_1d / q1_s
            Var_p_1d = R2_1d / (2.0 * alpha + 3.0)
            C_p_1d = ((p - one) / p) * q1_s * Var_p_1d - (one / p) * B_p_1d
            C_p = m * C_p_1d

            g = (h * kappa_total + C_p) / beta_s

            mu_u = (mu_s * h / q1_s) * x_body
            scale = torch.sqrt(Var_p_1d).expand_as(mu_u)
            optimal_policy = Normal(loc=mu_u, scale=scale)

        # value: scalar V = h ||x||^2 + g
        # If x is vector state: sum over last dim; else treat as scalar grid
        if x_body.dim() >= 1 and x_body.shape[-1] == D:
            x_norm2 = torch.sum(x_body * x_body, dim=-1, keepdim=True)  # [...,1]
        else:
            x_norm2 = (x_body * x_body).unsqueeze(-1) if x_body.dim() >= 1 else (x_body * x_body).reshape(1, 1)

        optimal_value = h * x_norm2 + g  # [...,1]

        # store
        self.h_t = h
        self.g_t = g

        if hasattr(self, "Xi"):
            Xi0 = self.Xi[0]
            if Xi0.numel() > 1:
                self.Y0 = h * torch.sum(Xi0 * Xi0) + g
            else:
                self.Y0 = h * (Xi0.reshape(()) ** 2) + g

        return optimal_policy, optimal_value

    def compute_exact_no_exploratory(self, x):
        """
        No-entropy exact (same h):
          beta*g = h*( Tr(ΣΣ^T) + sum_i lambda_i z_i^2 )
          V = h ||x||^2 + g
          u* = (mu h/q1) x
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=self.device, dtype=torch.get_default_dtype())
        device, dtype = x.device, x.dtype

        mu = self._to_tensor(self.params_equ["mu"], device, dtype)
        sigma = self._to_tensor(self.params_equ["sigma"], device, dtype)
        lam = self._to_tensor(self.params_equ["lambda"], device, dtype)
        z = self._to_tensor(self.params_equ["z"], device, dtype)
        q1 = self._to_tensor(self.params_equ["q1"], device, dtype)
        q2 = self._to_tensor(self.params_equ["q2"], device, dtype)
        beta = self._to_tensor(self.params_equ["beta"], device, dtype)

        mu_s = self._scalar(mu, "mu")
        q1_s = self._scalar(q1, "q1")
        q2_s = self._scalar(q2, "q2")
        beta_s = self._scalar(beta, "beta")

        x_body = self._canonicalize_x(x)
        D = self._infer_state_dim(x_body, lam=lam)

        sqrt_term = beta_s * beta_s * q1_s * q1_s + 4.0 * q1_s * q2_s * mu_s * mu_s
        h = (beta_s * q1_s - torch.sqrt(sqrt_term)) / (2.0 * mu_s * mu_s)

        tr_SigmaSigmaT = self._trace_SigmaSigmaT(sigma, D)
        sum_lam_z2 = self._sum_lambda_z2(lam, z)
        kappa_total = tr_SigmaSigmaT + sum_lam_z2

        g = (h * kappa_total) / beta_s

        optimal_control = (mu_s * h / q1_s) * x_body

        if x_body.dim() >= 1 and x_body.shape[-1] == D:
            x_norm2 = torch.sum(x_body * x_body, dim=-1, keepdim=True)
        else:
            x_norm2 = (x_body * x_body).unsqueeze(-1) if x_body.dim() >= 1 else (x_body * x_body).reshape(1, 1)

        optimal_value = h * x_norm2 + g

        self.h_t = h
        self.g_t = g

        if hasattr(self, "Xi"):
            Xi0 = self.Xi[0]
            if Xi0.numel() > 1:
                self.Y0 = h * torch.sum(Xi0 * Xi0) + g
            else:
                self.Y0 = h * (Xi0.reshape(()) ** 2) + g

        return optimal_control, optimal_value

    # ============================================================
    # Reward / coefficients
    # ============================================================
    def reward_func(self, x, dt, control):
        """
        scalar cost: q1 ||u||^2 + q2 ||x||^2  -> return -cost*dt, shape [...,1]
        """
        device = x.device if torch.is_tensor(x) else self.device
        dtype = x.dtype if torch.is_tensor(x) else torch.get_default_dtype()

        q1 = self._to_tensor(self.params_equ["q1"], device, dtype)
        q2 = self._to_tensor(self.params_equ["q2"], device, dtype)
        q1_s = self._scalar(q1, "q1")
        q2_s = self._scalar(q2, "q2")

        x_t = x if torch.is_tensor(x) else torch.as_tensor(x, device=device, dtype=dtype)
        u_t = control if torch.is_tensor(control) else torch.as_tensor(control, device=device, dtype=dtype)

        cost = q1_s * torch.sum(u_t * u_t, dim=-1, keepdim=True) + q2_s * torch.sum(x_t * x_t, dim=-1, keepdim=True)
        return -cost * dt

    def mu_torch(self, t, X, control=None):
        mu = torch.as_tensor(self.params_equ["mu"], device=self.device, dtype=torch.get_default_dtype()).reshape(())
        if control is None:
            raise ValueError("mu_torch expects control != None in this model.")
        return mu * control.to(self.device)

    def sigma_torch(self, t, X):
        """
        Return Σ as batch of diagonal matrices, shape [M, D, D].
        - If sigma scalar: Σ = sigma * I
        - If sigma vector length D: Σ = diag(sigma)
        """
        device = self.device
        dtype = torch.get_default_dtype()
        sigma = torch.as_tensor(self.params_equ["sigma"], device=device, dtype=dtype)

        M = int(self.M)
        D = int(self.D)

        if sigma.numel() == 1:
            base = torch.full((M, D), fill_value=sigma.reshape(()), device=device, dtype=dtype)
        else:
            sv = sigma.reshape(-1)
            if sv.numel() != D:
                raise ValueError(f"sigma length mismatch: expected D={D}, got {sv.numel()}")
            base = sv.view(1, D).expand(M, D).contiguous()

        return torch.diag_embed(base)  # [M, D, D]

    def xi(self, t, X):
        """
        Jump amplitude per-dim z, return shape [batch, D] aligned with X batch.
        """
        device = self.device
        dtype = torch.get_default_dtype()
        z = torch.as_tensor(self.params_equ["z"], device=device, dtype=dtype).reshape(-1)
        D = int(self.D)
        if z.numel() != D:
            raise ValueError(f"z length mismatch: expected D={D}, got {z.numel()}")
        B = int(X.shape[0]) if torch.is_tensor(X) and X.dim() >= 1 else 1
        return z.view(1, D).expand(B, D)

    # ============================================================
    # Net util
    # ============================================================
    def net_u_Du(self, t, X, compute_Du=True):
        inputs = torch.cat([t, X], dim=1)
        u = self.net_critic(inputs)  # Q-function
        Du = torch.autograd.grad(torch.sum(u), X, retain_graph=True, create_graph=True)[0] if compute_Du else 0
        return u, Du

    # ============================================================
    # Memory util
    # ============================================================
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

    # ============================================================
    # Sampling minibatch (multi-dim independent Poisson clocks)
    # ============================================================
    def fetch_minibatch(self):
        T, M, N, D = float(self.T), int(self.M), int(self.N), int(self.D)
        device = self.device
        dtype = torch.get_default_dtype()
        dt = T / N

        t_grid = (
            torch.linspace(0.0, T, N + 1, device=device, dtype=dtype)
            .unsqueeze(0)           # [1, N+1]
            .repeat(M, 1)           # [M, N+1]
            .unsqueeze(-1)          # [M, N+1, 1]
        )

        # tau_grid = torch.distributions.Exponential(
        #     torch.as_tensor(self.params_equ["beta"], device=device, dtype=dtype).reshape(())
        # ).sample((M, N + 1)).to(device)

        dW = torch.randn(M, N, D, device=device, dtype=dtype) * (dt ** 0.5)
        W = torch.cat([torch.zeros(M, 1, D, device=device, dtype=dtype), torch.cumsum(dW, dim=1)], dim=1)

        # ----- jump part: lambda is list length D -----
        lambda_list = torch.tensor(self.params_equ["lambda"], dtype=dtype, device=device).reshape(-1)
        if lambda_list.numel() != D:
            raise ValueError(f"lambda length mismatch: expected D={D}, got {lambda_list.numel()}")

        max_rate = float(lambda_list.max().item())
        max_jump = int(3.0 * max_rate * T) + 1

        if max_jump <= 0:
            jump_times = torch.zeros(M, 1, D, device=device, dtype=dtype)
            self.jump_index = torch.tensor([], dtype=torch.long, device=device)
            self.jump_index_D = torch.tensor([], dtype=torch.long, device=device)
            return t_grid, None, W, jump_times

        # generate per-dim arrival times, shape [M, max_jump, D]
        jump_times_list = []
        for d in range(D):
            rate_d = lambda_list[d].reshape(())
            jt_d = torch.distributions.Exponential(rate_d).sample((M, max_jump)).to(device)
            jt_d = torch.cumsum(jt_d, dim=1)

            # mask out > T to 0 (same behavior you used)
            jt_d = torch.where(jt_d <= T, jt_d, torch.zeros_like(jt_d))
            jump_times_list.append(jt_d.unsqueeze(-1))

        jump_times = torch.cat(jump_times_list, dim=-1)  # [M, max_jump, D]

        nz = torch.nonzero(jump_times, as_tuple=False)
        if nz.numel() > 0:
            self.jump_index = torch.unique(nz[:, 0])
            nz_last = torch.nonzero(jump_times[:, :, D - 1], as_tuple=False)
            self.jump_index_D = torch.unique(nz_last[:, 0]) if nz_last.numel() > 0 else torch.tensor([], dtype=torch.long, device=device)
        else:
            self.jump_index = torch.tensor([], dtype=torch.long, device=device)
            self.jump_index_D = torch.tensor([], dtype=torch.long, device=device)

        return t_grid, None, W, jump_times

    def sample_initial_state(self, training: bool) -> torch.Tensor:
        """Use a fixed X0 for evaluation and a wider reset distribution for training."""
        x0 = self.Xi.expand(self.M, -1).clone()
        coverage_std = float(self.params_train.get("state_coverage_std", 0.0))
        if training and coverage_std > 0.0:
            x0 = x0 + coverage_std * torch.randn_like(x0)
        return x0

    def diagnose_actor_slope(self):
        """Compare d actor_mean / d physical state with the analytic LQ slope."""
        was_training = self.net_actor.training
        self.net_actor.eval()
        probe = torch.cat([
            torch.zeros(1, 1, device=self.device),
            self.Xi[:1].to(self.device),
        ], dim=1).detach().requires_grad_(True)
        with torch.enable_grad():
            mean, _ = self.net_actor(probe)
            rows = []
            for j in range(self.D):
                grad = torch.autograd.grad(mean[0, j], probe, retain_graph=True)[0][0]
                rows.append(grad)
            jac = torch.stack(rows)
        if was_training:
            self.net_actor.train()

        mu = float(self.params_equ["mu"])
        q1 = float(self.params_equ["q1"])
        q2 = float(self.params_equ["q2"])
        beta = float(self.params_equ["beta"])
        h = (beta * q1 - math.sqrt(beta**2 * q1**2 + 4.0 * q1 * q2 * mu**2)) / (2.0 * mu**2)
        target = mu * h / q1
        state_jac = jac[:, 1:]
        diagonal = torch.diagonal(state_jac)
        off_diagonal = state_jac - torch.diag_embed(diagonal)
        return {
            "target": target,
            "diag_mean": float(diagonal.mean().detach().cpu()),
            "diag_rmse": float(torch.sqrt(torch.mean((diagonal - target) ** 2)).detach().cpu()),
            "cross_rms": float(torch.sqrt(torch.mean(off_diagonal ** 2)).detach().cpu()),
            "time_rms": float(torch.sqrt(torch.mean(jac[:, 0] ** 2)).detach().cpu()),
        }

    # ============================================================
    # Other utils
    # ============================================================
    def _clamp_log_sigma(self, log_sigma, itr):
        if self.params_train.get("entropy_alpha", 0.0) > 0:
            return log_sigma
        if itr is None:
            return torch.clamp(log_sigma, min=-8, max=-2)

        iter_total = self.params_train["iteration"]
        if itr < int(iter_total / 10):
            return torch.clamp(log_sigma, min=-10, max=-1)
        elif itr < int(iter_total / 5):
            return torch.clamp(log_sigma, min=-8, max=-1)
        elif itr < int(iter_total / 3):
            return torch.clamp(log_sigma, min=-10, max=0)
        else:
            return torch.clamp(log_sigma, min=-10, max=-1)

    def kl_divergence_gaussian(self, mu_pred, sigma_pred, mu_exact, sigma_exact):
        """
        Compute KL divergence between two Gaussian distributions.
        For p = N(mu_0, Sigma_0) and q = N(mu_1, Sigma_1), with dimension d:
        KL(p||q) = (1/2) * (log(det Sigma_1 / det Sigma_0) - d + tr(Sigma_1^-1 Sigma_0) + (mu_1 - mu_0)^T Sigma_1^-1 (mu_1 - mu_0))
        
        Here we assume diagonal covariances: Sigma = diag(sigma^2)
        
        Args:
            mu_pred: [M, D] or [N, M, D] - predicted mean (M=trajectories, N=time steps, D=dimension)
            sigma_pred: [M, D] or [N, M, D] - predicted std
            mu_exact: [M, D] or [N, M, D] - exact mean
            sigma_exact: [M, D] or [N, M, D] - exact std
        Returns:
            scalar KL divergence
        """
        # Handle different input shapes
        if mu_pred.dim() == 2:
            mu_pred = mu_pred.unsqueeze(0)  # [1, M, D]
            sigma_pred = sigma_pred.unsqueeze(0)
            mu_exact = mu_exact.unsqueeze(0)
            sigma_exact = sigma_exact.unsqueeze(0)
        
        N, M, D = mu_pred.shape
        
        # For diagonal covariance: det(Sigma) = prod(sigma^2), tr(Sigma_1^-1 Sigma_0) = sum(sigma_0^2 / sigma_1^2)
        sigma_pred_sq = sigma_pred ** 2
        sigma_exact_sq = sigma_exact ** 2
        
        # log(det Sigma_1 / det Sigma_0) = sum(log(sigma_1^2)) - sum(log(sigma_0^2))
        log_det_ratio = torch.sum(torch.log(sigma_exact_sq + 1e-12) - torch.log(sigma_pred_sq + 1e-12), dim=-1)  # [N, M]
        
        # tr(Sigma_1^-1 Sigma_0) = sum(sigma_0^2 / sigma_1^2)
        tr_term = torch.sum(sigma_pred_sq / (sigma_exact_sq + 1e-12), dim=-1)  # [N, M]
        
        # (mu_1 - mu_0)^T Sigma_1^-1 (mu_1 - mu_0) = sum((mu_1 - mu_0)^2 / sigma_1^2)
        mu_diff = mu_exact - mu_pred  # [N, M, D]
        quad_term = torch.sum((mu_diff ** 2) / (sigma_exact_sq + 1e-12), dim=-1)  # [N, M]
        
        # KL = (1/2) * (log_det_ratio - D + tr_term + quad_term)
        kl = 0.5 * (log_det_ratio - D + tr_term + quad_term)  # [N, M]
        
        return kl.mean()  # Average over time and trajectories
    
    def wasserstein_distance_gaussian(self, mu_pred, sigma_pred, mu_exact, sigma_exact):
        """
        Compute squared Wasserstein-2 distance between two Gaussian distributions.
        For p = N(mu_0, Sigma_0) and q = N(mu_1, Sigma_1):
        W_2^2(p, q) = ||mu_0 - mu_1||^2 + tr(Sigma_0 + Sigma_1 - 2(Sigma_1^(1/2) Sigma_0 Sigma_1^(1/2))^(1/2))
        
        For diagonal covariances: W_2^2 = ||mu_0 - mu_1||^2 + sum((sigma_0 - sigma_1)^2)
        
        Args:
            mu_pred: [M, D] or [N, M, D] - predicted mean (M=trajectories, N=time steps, D=dimension)
            sigma_pred: [M, D] or [N, M, D] - predicted std
            mu_exact: [M, D] or [N, M, D] - exact mean
            sigma_exact: [M, D] or [N, M, D] - exact std
        Returns:
            scalar squared Wasserstein-2 distance
        """
        # Handle different input shapes
        if mu_pred.dim() == 2:
            mu_pred = mu_pred.unsqueeze(0)  # [1, M, D]
            sigma_pred = sigma_pred.unsqueeze(0)
            mu_exact = mu_exact.unsqueeze(0)
            sigma_exact = sigma_exact.unsqueeze(0)
        
        # ||mu_0 - mu_1||^2
        mu_diff = mu_exact - mu_pred  # [N, M, D]
        mu_term = torch.sum(mu_diff ** 2, dim=-1)  # [N, M]
        
        # For diagonal: sum((sigma_0 - sigma_1)^2)
        sigma_diff = sigma_exact - sigma_pred  # [N, M, D]
        sigma_term = torch.sum(sigma_diff ** 2, dim=-1)  # [N, M]
        
        w2_sq = mu_term + sigma_term  # [N, M]
        return w2_sq.mean()  # Average over time and trajectories
    
    def cvm_error_from_samples_multi(self, control, policy_exact):
        """
        CVM error (only for 1D case, use KL/Wasserstein for high-dim).
        control:
            [M, D] or [N, M, D] (optionally [..,1] tail allowed) (M=trajectories, N=time steps, D=dimension)
        policy_exact:
            Normal with loc/scale broadcastable to control
        return:
            scalar mean CVM over batch and dims
        """
        u = control
        if not torch.is_tensor(u):
            u = torch.as_tensor(u, device=self.device, dtype=torch.get_default_dtype())

        # squeeze trailing singleton dims if present
        while u.dim() >= 2 and u.size(-1) == 1:
            u = u.squeeze(-1)

        # unify to [B, M, D]
        if u.dim() == 2:
            u = u.unsqueeze(0)
        if u.dim() != 3:
            raise ValueError(f"control must be [M,D] or [N,M,D], got shape={tuple(u.shape)}")

        B, M, D = u.shape

        # Try cdf; if shape mismatch, expand distribution to u.shape
        try:
            p = policy_exact.cdf(u)
        except Exception:
            policy_exact = policy_exact.expand(u.shape)
            p = policy_exact.cdf(u)

        p_sorted, _ = torch.sort(p, dim=1)  # sort along M

        i = torch.arange(1, M + 1, device=p.device, dtype=p.dtype).view(1, M, 1)
        grid = (2 * i - 1) / (2 * M)  # [1, M, 1] -> broadcast

        cvm_per_bd = ((p_sorted - grid) ** 2).mean(dim=1) + (1.0 / (12.0 * M))  # [B, D]
        return cvm_per_bd.mean()

    ##############
    ## Train
    ##############

    def train_onlycritic(self, itr, traj_only=False, _control_exact_=False):
        if _control_exact_ and not traj_only:
            raise ValueError("Reference-policy rollouts are evaluation-only")
        evaluate_metrics = bool(traj_only)
        t, tau_sampled, W, P = self.fetch_minibatch()
        dt = self.dt 
        beta = self.params_equ["beta"]
        gamma_entropy = self.params_train["entropy_alpha"] 
        p_tsallis = self.params_train.get("p_tsallis", 1) 
        gamma_soft = self.params_train["gamma_soft"] 
        std_lambda = 1.0  
        M = self.M
        X_buffer, Y_buffer, A_buffer, action_buffer, mu_logsigma_buffer = [], [], [], [], []
        total_loss_critic_sum = 0
        critic_loss_accum = 0.0
        critic_chunk_count = 0
        self.optimizer_critic.zero_grad(set_to_none=True)

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = self.sample_initial_state(training=True)
        X0 = X0.detach().clone().requires_grad_(True)
        Y0, _ = self.net_u_Du(t0, X0, compute_Du=not traj_only)
        
        
        X_buffer.append(X0.detach().clone())
        Y_buffer.append(Y0.detach().clone())

        mu, log_sigma = self.net_actor(torch.cat([t[:, 0, :], X0], dim=1))  # [M, 2]
        # log_sigma = self._clamp_log_sigma(log_sigma, itr)
        sigma = torch.exp(log_sigma) 
        eps = torch.randn_like(mu)
        a = mu + sigma * eps  # reparameterized sample
        action_buffer.append(a)
        mu_logsigma_buffer.append(torch.cat([mu, log_sigma], dim=1))

        K_critic = max(1, self.N // 50)  
        
        # Initialize error accumulators for integration over time
        control_error_accum = []  # List to store errors at each time step
        Y_error_accum = []  # List to store value errors at each time step
  
        for k in range(self.N):
            t1 = t[:, k + 1, :]
            W1 = W[:, k + 1, :]
            dW = (W1 - W0).unsqueeze(-1) # M, 1
            
            mask  = (P > t0.unsqueeze(1)) & (P <= t1.unsqueeze(1))
            Njump = mask.sum(dim=1).to(torch.float32) # [M, D]
            
            
            lambda_tensor = torch.tensor(self.params_equ["lambda"], device=self.device)
            if lambda_tensor.dim() == 0:  
                lambda_tensor = lambda_tensor.unsqueeze(0).expand(self.D)  # [D]
            elif lambda_tensor.dim() == 1 and lambda_tensor.shape[0] == 1:  # [1]
                lambda_tensor = lambda_tensor.expand(self.D)  # [D]
            
            
            
            dMt = Njump - lambda_tensor.unsqueeze(0) * dt  # [M, D]
            
            # dMt   = Njump - torch.tensor(self.params_equ["lambda"], device=self.device).unsqueeze(0) * dt  # [M, D]
            eps = None

            if not _control_exact_: 
                mu_, log_sigma = self.net_actor(torch.cat([t0, X0], dim=1))  # [M, 2]
                # log_sigma = self._clamp_log_sigma(log_sigma, itr)
                sigma = torch.exp(log_sigma) 
                eps = torch.randn_like(mu_)
                control = mu_ + sigma * eps  
                if k == (self.N - 1):
                    print("control", control.mean())

                dist = torch.distributions.Normal(mu_, sigma)
                
                log_prob = dist.log_prob(control.detach()).sum(dim=-1, keepdim=True)
                shannon_entropy = dist.entropy().sum(dim=-1, keepdim=True)  # -log_prob #

                
                # ```python
                # dist = Normal(action_mean, action_std)
                
                
                # ```

                
                # ```python
                
                # dist = torch.distributions.Normal(mu_, sigma)
                
                # ```

                

                # ### 1. `dist.sample()` vs `mu_ + sigma * eps`

                # **`dist.sample()`**：
                
                
                

                # **`mu_ + sigma * eps`**：
                
                
                

                

                
                # ```
                
                #                 ↓
                
                # ```
                

                
                # ```
                
                #         ↓
                
                # ```
                
                # ```
                
                #         ↓                    ↓
                
                
                
                # ```
                


                
                z = torch.exp(log_prob) 
                if p_tsallis != 1:
                    tsallis_entropy = (1 / (p_tsallis - 1)) * (1 - torch.pow(z, p_tsallis - 1)) 
                else:
                    tsallis_entropy = -log_prob # shannon_entropy

                mu_func = self.mu_torch(t0, X0, control) # [M, D]
                # print("dW.shape", dW.shape)      # [100, 1]
                # diffusion = self.sigma_torch(t0, X0, control) * dW.squeeze(-1)
                # diffusion = torch.bmm(self.sigma_torch(t0, X0), (W1 - W0).unsqueeze(-1)).squeeze(2)
                diffusion = torch.bmm(self.sigma_torch(t0, X0), (W1 - W0).unsqueeze(-1)).squeeze(2)
                xi =  self.xi(t0, X0)
                X1 = X0 + mu_func * dt + diffusion + xi * dMt

                action_buffer.append(control)
                mu_logsigma_buffer.append(torch.cat([mu_, log_sigma], dim=1))
                
                
                std_loss = std_lambda * sigma.pow(2)
            else:
                if gamma_entropy == 0:
                    control_exact, _ = self.compute_exact_no_exploratory(X0)
                    control_exact = control_exact * torch.ones_like(X0)
                else:
                    check_nan(X0, "X0 for exact entropy")
                    optimal_policy, _ = self.compute_exact_entropy(X0)
                    eps = torch.randn_like(X0)
                    # print("eps.shape", eps.shape) # torch.Size([100, 1])
                    control_exact = optimal_policy.loc + optimal_policy.scale * eps

                mu_func = control_exact
                # diffusion = torch.bmm((control_exact * self.params_equ["sigma"]).unsqueeze(-1), (W1 - W0).unsqueeze(-1)).squeeze(2)  # [M, D]
                diffusion = torch.bmm(self.sigma_torch(t0, X0), (W1 - W0).unsqueeze(-1)).squeeze(2)
                # diffusion = torch.bmm((control_exact * self.params_equ["sigma"]).unsqueeze(-1), (W1 - W0).unsqueeze(-1)).squeeze(2)  # [M, D]
                control = control_exact
                # xi = self.params_equ["z"]  # [M, D]
                xi =  self.xi(t0, X0)
                X1 = X0 + mu_func * dt + diffusion + xi * dMt
                
            V0, partial_V0 = self.net_u_Du(t0, X0, compute_Du=not traj_only)
            # print("t1, X1 shape", t1.shape, X1.shape) # torch.Size([100, 1]) torch.Size([100, 100])
            V1, _ = self.net_u_Du(t1, X1, compute_Du=not traj_only)
            

            if not traj_only:
                if gamma_entropy == 0:
                    reward = self.reward_func(X0, dt, control) 
                else:
                    reward = self.reward_func_entropy(X0, dt, control, tsallis_entropy) 
                # ----------------------------------------------------------------------
                # V1 = (1 - gamma_soft) * V1 + gamma_soft * V0
                disc    = torch.exp(torch.tensor(-beta*dt, device=V0.device))
                
                
                
                
                # V_x_alphaxu, _ = self.net_u_Du(t0, X0 + xi, compute_Du=False)
                # V_diff = (V_x_alphaxu - V0).expand_as(dMt)  # [M, 1] -> [M, D]
                
                jump_term_sum = torch.zeros_like(V0)  # [M, 1]
                for j in range(self.D):
                    newX = torch.zeros_like(X0)
                    newX[:, j] = xi[:, j]  
                    V_x_alphaxu_j, _ = self.net_u_Du(t0, X0 + newX, compute_Du=False)  # [M, 1]
                    jump_term_sum = jump_term_sum + (V_x_alphaxu_j - V0) * dMt[:, j].unsqueeze(1)  # [M, 1]
                
                
                
                # V_diff = (V_x_alphaxu - V0).expand_as(dMt)  # [M, 1] -> [M, D]
                # reward_ = reward + (partial_V0 * diffusion).sum(dim=-1, keepdim=True) + (V_diff * dMt).sum(dim=-1, keepdim=True)  # [M, 1]
                
                
                #          loss = (Y0 - reward - Y1)**2
                
                
                reward_ = reward  - (partial_V0 * diffusion).sum(dim=-1, keepdim=True) - jump_term_sum  # [M, 1]
                # reward_ = reward + (partial_V0 * diffusion).sum(dim=-1, keepdim=True) + jump_term_sum  # [M, 1]
                disc = torch.exp(torch.tensor(-beta*dt, device=V0.device))
                td_target = reward_ + disc * V1.detach()  
                # td_target = reward_ + gamma_entropy * tsallis_entropy.detach() + disc * V1.detach() 
                delta = td_target - V0  
                loss_critic_step = torch.mean(delta ** 2) 
                critic_loss_accum = critic_loss_accum + loss_critic_step
                critic_chunk_count += 1

                if ((k + 1) % K_critic == 0) or (k == self.N - 1):
                    loss_critic_chunk = critic_loss_accum / float(critic_chunk_count)  
                    self.optimizer_critic.zero_grad(set_to_none=True)
                    loss_critic_chunk.backward()
                    g0 = torch.nn.utils.clip_grad_norm_(self.net_critic.parameters(), 1.0)
                    self.optimizer_critic.step()
                    
                    
                    
                    # self.scheduler_critic.step()
                    # print("grad_norm", g0)
                    total_loss_critic_sum += loss_critic_chunk.detach()
                    critic_loss_accum = 0.0
                    critic_chunk_count = 0

            if evaluate_metrics:
                with torch.no_grad():
                    # Compute errors at each time step for integration
                    if gamma_entropy == 0:
                        control_exact, Y_exact = self.compute_exact_no_exploratory(X0)

                        control_mean_pred = control
                        control_mean_exact = control_exact  # [M, D]
                        control_error_step = torch.sum((control_mean_pred - control_mean_exact) ** 2, dim=-1) / (torch.sum(control_mean_exact ** 2, dim=-1) + 1e-12)  # [M]
                        control_error_accum.append(control_error_step)  # Store for integration
                        if k == (self.N-1):
                            print("control_exact", control_exact.mean())
                            print("Y_exact", Y_exact.mean())
                            print("Y_pred", V1.mean())
                    else:
                        policy_exact, Y_exact = self.compute_exact_entropy(X0)
                        if not _control_exact_:
                            mu_pred = mu_  # [M, D]
                            sigma_pred = sigma  # [M, D]
                        else:
                            mu_pred = control_exact  # [M, D]
                            sigma_pred = policy_exact.scale  # [M, D]
                        mu_exact = policy_exact.loc  # [M, D]
                        sigma_exact = policy_exact.scale  # [M, D]

                        # Use KL divergence for high-dim, CVM for 1D
                        if self.D > 1:
                            # High-dimensional: use KL divergence
                            control_error_step = self.kl_divergence_gaussian(
                                mu_pred.unsqueeze(0), sigma_pred.unsqueeze(0),
                                mu_exact.unsqueeze(0), sigma_exact.unsqueeze(0)
                            )
                            # KL returns scalar, need to expand to [M] for consistency
                            control_error_step = control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
                        else:
                            # 1D: can use CVM
                            control_error_step = self.cvm_error_from_samples_multi(control.unsqueeze(0), policy_exact)
                            control_error_step = control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
                        control_error_accum.append(control_error_step)
                        if k == (self.N-1):
                            print("Y_exact", Y_exact.mean())
                            print("Y_pred", V1.mean())

                    # Value function error: |V_pred - V_exact|^2 / (|V_exact|^2 + eps)
                    Y_pred = V1  # [M, 1]
                    Y_error_step = (Y_pred - Y_exact) ** 2 / (Y_exact ** 2 + 1e-12)  # [M, 1]
                    Y_error_accum.append(Y_error_step.squeeze(-1))  # [M]
                
            X0, t0, W0 = X1.detach().clone().requires_grad_(True), t1, W1
            X_buffer.append(X1.detach().clone())
            
            Y_buffer.append(V1.detach().clone())

        total_loss_critic_sum = total_loss_critic_sum / (self.N // K_critic)
        
        
        if not traj_only:
            self.scheduler_critic.step()
        
        # Integrate errors over time: E = sqrt(∫_0^T error^2 dt / (∫_0^T exact^2 dt + eps))
        # For control error
        if len(control_error_accum) > 0:
            control_error_t = torch.stack(control_error_accum, dim=0)  # [N, M]
            # Integrate: sum over time steps and average over trajectories
            control_error = torch.sqrt(control_error_t.mean(dim=1).sum() * dt / (self.T + 1e-12))
        else:
            control_error = None
        
        # For value error
        if len(Y_error_accum) > 0:
            Y_error_t = torch.stack(Y_error_accum, dim=0)  # [N, M]
            # Integrate: sum over time steps and average over trajectories
            Y_error = torch.sqrt(Y_error_t.mean(dim=1).sum() * dt / (self.T + 1e-12))
        else:
            Y_error = torch.tensor(float("nan"), device=self.device)
    
        return torch.stack(X_buffer, dim=1), torch.stack(Y_buffer, dim=1), Y_buffer[0], torch.stack(action_buffer, dim=1) , torch.stack(mu_logsigma_buffer, dim=1), Y_error, control_error, total_loss_critic_sum


    def train_loop(self, itr, traj_only=False, _control_exact_=False, predict=False):
        if predict and not traj_only:
            raise ValueError("Evaluation batches must use traj_only=True")
        if _control_exact_ and not traj_only:
            raise ValueError("Reference-policy rollouts are evaluation-only")
        evaluate_metrics = bool(traj_only or predict)
        if not predict:
            t, _, W, P = self.fetch_minibatch()
        else:
            t, _, W, P = self.t_predict_set, self.tau_predict_set, self.W_predict_set, self.P_predict_set
        dt = self.dt 
        beta = self.params_equ["beta"]
        gamma_entropy = self.params_train["entropy_alpha"] 
        actor_loss_mode = self.params_train.get("actor_loss", "manuscript")
        p_tsallis = self.params_train.get("p_tsallis", 1) 
        gamma_soft = self.params_train["gamma_soft"] 
        std_lambda = 1.0  
        _pen = self.params_train.get("deterministic_sigma_penalty")
        deterministic_sigma_penalty = float(0.05 if _pen is None and gamma_entropy == 0 else (0.0 if _pen is None else _pen))
        M = self.M
        X_buffer, Y_buffer, A_buffer, logpi_buffer, action_buffer, mu_logsigma_buffer = [], [], [], [], [], []
        logpi_tsallis_buffer = []
        total_loss_critic_sum = 0
        total_loss_actor_sum = 0
        critic_chunk_total_count = 0  
        actor_chunk_total_count = 0   

        
        K_actor = 20 
        K_critic = 5 
        actor_loss_accum = 0.0
        actor_chunk_count = 0
        critic_loss_accum = 0.0
        critic_chunk_count = 0
        std_loss_accum = torch.tensor(0.0, device=self.device)  
        std_loss_count = 0       
        mu_gaussian_loss_accum = torch.tensor(0.0, device=self.device)  
        mu_gaussian_loss_count = 0  
        self.optimizer_actor.zero_grad(set_to_none=True)
        self.optimizer_critic.zero_grad(set_to_none=True)
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = self.sample_initial_state(training=not predict)
        X0 = X0.detach().clone().requires_grad_(True)
        Y0, _ = self.net_u_Du(t0, X0, compute_Du=not traj_only)
        
        
        X_buffer.append(X0.detach().clone())
        Y_buffer.append(Y0.detach().clone())

        mu, log_sigma = self.net_actor(torch.cat([t[:, 0, :], X0], dim=1))  # [M, 2]
        # log_sigma = self._clamp_log_sigma(log_sigma, itr)
        sigma = torch.exp(log_sigma) 
        eps = torch.randn_like(mu)
        a = mu + sigma * eps  # reparameterized sample
        action_buffer.append(a)
        mu_logsigma_buffer.append(torch.cat([mu, log_sigma], dim=1).detach().clone())

        # Initialize error accumulators for integration over time
        control_error_accum = []  # List to store errors at each time step
        Y_error_accum = []  # List to store value errors at each time step
             
        for k in range(self.N):
            t1 = t[:, k + 1, :]
            W1 = W[:, k + 1, :]
            dW = (W1 - W0).unsqueeze(-1) # M, 1
            mask  = (P > t0.unsqueeze(1)) & (P <= t1.unsqueeze(1))
            Njump = mask.sum(dim=1).to(torch.float32) # [M, D]
            
            
            lambda_tensor = torch.tensor(self.params_equ["lambda"], device=self.device)
            if lambda_tensor.dim() == 0:  
                lambda_tensor = lambda_tensor.unsqueeze(0).expand(self.D)  # [D]
            elif lambda_tensor.dim() == 1 and lambda_tensor.shape[0] == 1:  # [1]
                lambda_tensor = lambda_tensor.expand(self.D)  # [D]
            
            
            
            dMt = Njump - lambda_tensor.unsqueeze(0) * dt  # [M, D]
            
            # dMt   = Njump - torch.tensor(self.params_equ["lambda"], device=self.device).unsqueeze(0) * dt  
            # print("dMt.shape", dMt.shape)
            if not _control_exact_: 
                mu_, log_sigma = self.net_actor(torch.cat([t0, X0], dim=1))  # [M, 2]
                # log_sigma = self._clamp_log_sigma(log_sigma, itr)
                sigma = torch.exp(log_sigma) 
                eps = torch.randn_like(mu_)
                control = mu_ + sigma * eps  

                if k == (self.N - 1):
                    print("control", control.mean())
                dist = torch.distributions.Normal(mu_, sigma)
                log_prob = dist.log_prob(control.detach()).sum(dim=-1, keepdim=True)
                shannon_entropy = dist.entropy().sum(dim=-1, keepdim=True)  # -log_prob #

                
                if p_tsallis != 1:
                    z = torch.exp(log_prob)
                    tsallis_entropy = (1 / (p_tsallis - 1)) * (1 - torch.pow(z, p_tsallis - 1)) 
                else:
                    tsallis_entropy = - log_prob # shannon_entropy  #

                mu_func = self.mu_torch(t0, X0, control) 
                diffusion = torch.bmm(self.sigma_torch(t0, X0), (W1 - W0).unsqueeze(-1)).squeeze(2)
                xi =  self.xi(t0, X0)
                X1 = X0 + mu_func * dt + diffusion + xi * dMt

                action_buffer.append(control)
                mu_logsigma_buffer.append(torch.cat([mu_, log_sigma], dim=1).detach().clone())
                
                
                std_loss = std_lambda * sigma.pow(2)
                std_loss_accum += std_loss.mean().detach().clone()
                std_loss_count += 1
                if evaluate_metrics:
                    with torch.no_grad():
                        if gamma_entropy == 0:
                            mu_gaussian_exact, _ = self.compute_exact_no_exploratory(X0)
                        else:
                            optimal_policy, _ = self.compute_exact_entropy(X0)
                            sigma_theoretical = optimal_policy.scale
                            mu_gaussian_exact = optimal_policy.loc
                
                
                        mu_gaussian_loss_step = torch.mean((mu_ - mu_gaussian_exact)**2 / (mu_gaussian_exact**2 + 1e-8))
                
                
                
                
                        mu_gaussian_loss_accum += mu_gaussian_loss_step.detach().clone()
                        mu_gaussian_loss_count += 1

            else:
                if gamma_entropy == 0:
                    control_exact, _ = self.compute_exact_no_exploratory(X0)
                    # print("control_exact", control_exact.shape) # torch.Size([])
                    control_exact = control_exact * torch.ones_like(X0)
                    # print("control_exact.shape after", control_exact.shape) # torch.Size([500, 1])
                else:
                    optimal_policy, _ = self.compute_exact_entropy(X0)
                    eps = torch.randn_like(X0) # torch.Size([100, 1])
                    # if predict:
                    control_exact = optimal_policy.loc
                    # else:
                    #     control_exact = optimal_policy.loc + optimal_policy.scale * eps
                   
                mu_func = self.params_equ["mu"] * control_exact  # [M, D]
                diffusion = torch.bmm(self.sigma_torch(t0, X0), (W1 - W0).unsqueeze(-1)).squeeze(2)
                control = control_exact
                action_buffer.append(control)
                xi = torch.tensor(self.params_equ["z"], device=self.device).unsqueeze(0)
                
                X1 = X0 + mu_func * dt + diffusion + xi * dMt
                
            V0, partial_V0 = self.net_u_Du(t0, X0, compute_Du=not traj_only)            
            V1 = self.net_critic(torch.cat([t1, X1], dim=1))   
            V1 = (1 - gamma_soft) * V1 + gamma_soft * V0.detach()
            

            if not traj_only:
                
                
                
                
                
                # =====================================================
                
                
                
                
                with torch.no_grad():
                    reward_no_entropy = self.reward_func(X0, dt, control)
                    if gamma_entropy == 0:
                        reward = reward_no_entropy
                    else:
                        reward = self.reward_func_entropy(X0, dt, control, tsallis_entropy)
                    
                    
                    
                    
                    jump_term_sum = torch.zeros_like(V0)  # [M, 1]
                    for j in range(self.D):
                        newX = torch.zeros_like(X0)
                        newX[:, j] = xi[:, j]  
                        V_x_alphaxu_j, _ = self.net_u_Du(t0, X0 + newX, compute_Du=False)  # [M, 1]
                        jump_term_sum = jump_term_sum + (V_x_alphaxu_j - V0) * dMt[:, j].unsqueeze(1)  # [M, 1]
                
                
                
                
                
                
                
                
                #          loss = (Y0 - reward - Y1)**2
                
                
                
                
                
                reward_ = reward  - (partial_V0 * diffusion).sum(dim=-1, keepdim=True) - jump_term_sum  # [M, 1]
                reward_no_entropy_ = reward_no_entropy - (partial_V0 * diffusion).sum(dim=-1, keepdim=True) - jump_term_sum
                disc = torch.exp(torch.tensor(-beta*dt, device=V0.device))
                delta = reward_ + disc * V1.detach() - V0  
                
                loss_critic_step = torch.mean(delta ** 2)  

                critic_loss_accum = critic_loss_accum + loss_critic_step
                critic_chunk_count += 1
                if ((k + 1) % K_critic == 0) or (k == self.N - 1):
                    loss_critic_chunk = critic_loss_accum / float(critic_chunk_count)
                    self.optimizer_critic.zero_grad(set_to_none=True)
                    loss_critic_chunk.backward()
                    g0 = torch.nn.utils.clip_grad_norm_(self.net_critic.parameters(), 1.0)
                    self.optimizer_critic.step()
                    
                    
                    
                    # self.scheduler_critic.step()
                    total_loss_critic_sum += loss_critic_chunk.detach()
                    critic_chunk_total_count += 1  
                    critic_loss_accum = 0.0
                    critic_chunk_count = 0

                with torch.no_grad():
                    
                    
                    
                    
                    
                    # ===========================================================
                    
                    
                    
                    
                    advantage = (reward_ + torch.exp(torch.tensor(-beta * dt)) * V1.detach() - V0.detach()) / (dt + 1e-8)  # [M, 1]
                    advantage_no_entropy = (
                        reward_no_entropy_
                        + torch.exp(torch.tensor(-beta * dt)) * V1.detach()
                        - V0.detach()
                    ) / (dt + 1e-8)
                    # ===========================================================================================

                A_buffer.append(advantage)
                logpi_tsallis_buffer.append(tsallis_entropy)
                logpi_buffer.append(log_prob)
                # ====== loss kl ==========
                # # ref: P19 in Reinforcement Learning for Jump-Diffusions with Financial
                
                
                
                
                
                # V_diff_kl = (V_x_alphaxu - V0).expand_as(dMt)  # [M, 1] -> [M, D]
                # q_multiply_dt = V1 - V0 - beta * V0 * dt + reward + (partial_V0 * diffusion).sum(dim=-1, keepdim=True) + (V_diff_kl * dMt).sum(dim=-1, keepdim=True)  # [M, 1]
                
                jump_term_kl = torch.zeros_like(V0)  # [M, 1]
                for j in range(self.D):
                    newX = torch.zeros_like(X0)
                    newX[:, j] = xi[:, j]  
                    V_x_alphaxu_j, _ = self.net_u_Du(t0, X0 + newX, compute_Du=False)  # [M, 1]
                    jump_term_kl = jump_term_kl + (V_x_alphaxu_j - V0) * dMt[:, j].unsqueeze(1)  # [M, 1]
                q_multiply_dt = V1 - V0 - beta * V0 * dt + reward + (partial_V0 * diffusion).sum(dim=-1, keepdim=True) + jump_term_kl  # [M, 1]
                
                loss_KL = torch.sum((q_multiply_dt.detach() + gamma_entropy * tsallis_entropy * dt)**2)  
                

                
                #     tau = 1.0
                #     c   = 2.0
                #     w_t = 1.0 + c * torch.exp(-t0 / tau)           # [M,1]
                # loss_actor_step = - (1/beta) * torch.mean( w_t * torch.sum(log_prob * advantage, dim=-1, keepdim=True) )

                
                # ∇_θ J(θ) = E [∇_θ log π_θ(u | s) A(s,u)]
                
                
                #   log π = -1/2 (m log(2π) + Σ log σ_j^2 + Σ (u_j - μ_j)^2 / σ_j^2)
                
                
                # =====================================================
                
                # # log_prob: [M,1]
                
                # pg_term = - (1.0 / beta) * (log_prob * advantage.detach())  # [M,1]
                
                # loss_actor_step = (pg_term + ent_term).mean()
                
                
                
                
                # A_soft_step = advantage.detach() + gamma_entropy * tsallis_entropy
                # A_soft_step = (advantage + gamma_entropy * tsallis_entropy).detach()  # [M, 1] + [M, 1] = [M, 1]
                # loss_actor_step = - (1 / beta) * (log_prob * A_soft_step)  # [M, 1] * [M, 1] = [M, 1]
                
                if actor_loss_mode == "manuscript":
                    # manuscript0917.pdf, (3.22)-(3.23): advantage already
                    # contains -alpha*log(pi) through reward_func_entropy.
                    A_soft_step = advantage.detach()
                    loss_actor_step = -(1.0 / beta) * (log_prob * A_soft_step).mean()
                elif actor_loss_mode == "entropy_outside":
                    # Ablation: use the non-entropy advantage in the score term
                    # and differentiate the Shannon entropy separately.
                    pg_term = -(1.0 / beta) * (log_prob * advantage_no_entropy.detach())
                    ent_term = -(1.0 / beta) * gamma_entropy * shannon_entropy
                    loss_actor_step = (pg_term + ent_term).mean()
                else:
                    raise ValueError(f"Unknown actor_loss={actor_loss_mode!r}")
                # A_soft_step = advantage + gamma_entropy * tsallis_entropy.detach()
                # loss_actor_step = - (1/ beta) * torch.sum(log_prob * A_soft_step, dim=-1).mean()

                if (deterministic_sigma_penalty > 0.0 and gamma_entropy == 0
                        and not getattr(self.net_actor, "use_fixed_std", False)):
                    loss_actor_step = loss_actor_step + deterministic_sigma_penalty * sigma.pow(2).mean()

                
                
                
                # ====== std loss =========
                # loss_actor = loss_actor + std_loss
                # ====== KL loss =========
                # loss_actor_step = torch.mean(loss_actor_step + 0.01 * loss_KL)
                # ======= F loss ==========
                
                # F_loss = (- F + F.pow(2)).mean()
                # loss_actor = loss_actor + 0.1 * F_loss
                # ======== mu_gaussian loss ========
                
                
                
                
                # if gamma_entropy == 0:
                
                
                #     loss_actor_step = loss_actor_step + 0.1 * sigma_loss
                # ============================================================================

                
                
                actor_loss_accum = actor_loss_accum + loss_actor_step
                actor_chunk_count += 1

                if ((k + 1) % K_actor == 0) or (k == self.N - 1):
                    loss_actor_chunk = actor_loss_accum / float(actor_chunk_count)
                    self.optimizer_actor.zero_grad(set_to_none=True)
                    loss_actor_chunk.backward()
                    g = torch.nn.utils.clip_grad_norm_(self.net_actor.parameters(), 1.0)
                    self.optimizer_actor.step()
                    
                    
                    
                    # self.scheduler_actor.step()
                    # print("====grad_norm", g)
                    total_loss_actor_sum += loss_actor_chunk.detach()
                    actor_chunk_total_count += 1  
                    actor_loss_accum = 0.0
                    actor_chunk_count = 0

            X0, t0, W0 = X1.detach().clone().requires_grad_(True), t1, W1
            X_buffer.append(X1.detach().clone())
            
            Y_buffer.append(V1.detach().clone())

            if evaluate_metrics:
                with torch.no_grad():
                    # Compute errors at each time step for integration
                    if gamma_entropy == 0:
                        control_exact, Y_exact = self.compute_exact_no_exploratory(X1)
                        control_mean_pred = control

                        control_mean_exact = control_exact  # [M, D]
                        control_error_step = torch.sum((control_mean_pred - control_mean_exact) ** 2, dim=-1) / (torch.sum(control_mean_exact ** 2, dim=-1) + 1e-12)  # [M]
                        control_error_accum.append(control_error_step)  # Store for integration
                        if k == (self.N-1):
                            print("control_exact", control_exact.mean())
                            print("Y_exact", Y_exact.mean())
                            print("Y_pred", V1.mean())
                    else:
                        policy_exact, Y_exact = self.compute_exact_entropy(X1)
                        # For gamma > 0: use KL or Wasserstein distance
                        # Get predicted policy parameters
                        if not _control_exact_:
                            mu_pred = mu_  # [M, D]
                            sigma_pred = sigma  # [M, D]
                        else:
                            mu_pred = control_exact  # [M, D]
                            sigma_pred = policy_exact.scale  # [M, D]
                        mu_exact = policy_exact.loc  # [M, D]
                        sigma_exact = policy_exact.scale  # [M, D]

                        # Use KL divergence for high-dim, CVM for 1D
                        if self.D > 1:
                            # High-dimensional: use KL divergence
                            control_error_step = self.kl_divergence_gaussian(
                                mu_pred.unsqueeze(0), sigma_pred.unsqueeze(0),
                                mu_exact.unsqueeze(0), sigma_exact.unsqueeze(0)
                            )
                            # KL returns scalar, need to expand to [M] for consistency
                            control_error_step = control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
                        else:
                            # 1D: can use CVM
                            control_error_step = self.cvm_error_from_samples_multi(control.unsqueeze(0), policy_exact)
                            control_error_step = control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
                        control_error_accum.append(control_error_step)
                        if k == (self.N-1):
                            print("Y_exact", Y_exact.mean())
                            print("Y_pred", V1.mean())
            
                    # Value function error: |V_pred - V_exact|^2 / (|V_exact|^2 + eps)
                    Y_pred = V1  # [M, 1]
                    Y_error_step = (Y_pred - Y_exact) ** 2 / (Y_exact ** 2 + 1e-12)  # [M, 1]
                    Y_error_accum.append(Y_error_step.squeeze(-1))  # [M]
            

        if not traj_only:
            self.control_list.append(control.detach().clone())
            self.KLLoss_list.append(loss_KL.mean().detach().clone())
            
            if std_loss_count > 0:
                std_loss_avg = std_loss_accum / std_loss_count
            else:
                std_loss_avg = torch.tensor(0.0, device=self.device)
            self.StdLoss_list.append(std_loss_avg.detach().clone())
            
            if mu_gaussian_loss_count > 0:
                mu_gaussian_loss_avg = mu_gaussian_loss_accum / mu_gaussian_loss_count
            else:
                mu_gaussian_loss_avg = torch.tensor(float("nan"), device=self.device)
            
            
            if critic_chunk_total_count > 0:
                total_loss_critic_sum = total_loss_critic_sum / float(critic_chunk_total_count)
            if actor_chunk_total_count > 0:
                total_loss_actor_sum = total_loss_actor_sum / float(actor_chunk_total_count)
            self.ActorLoss_list.append(total_loss_actor_sum.detach().clone())
            self.CriticLoss_list.append(total_loss_critic_sum.detach().clone())
            self.mu_gaussian_loss_list.append(mu_gaussian_loss_avg.detach().clone())
            # print("mu_gaussian_loss.mean =", mu_gaussian_loss.mean())
            
            
            
            
            
            self.scheduler_actor.step()
            self.scheduler_critic.step()
            
            
        
        # Integrate errors over time: E = sqrt(∫_0^T error^2 dt / (∫_0^T exact^2 dt + eps))
        # For control error
        if len(control_error_accum) > 0:
            control_error_t = torch.stack(control_error_accum, dim=0)  # [N, M]
            # Integrate: sum over time steps and average over trajectories
            control_error = torch.sqrt(control_error_t.mean(dim=1).sum() * dt / (self.T + 1e-12))
        else:
            control_error = None
        
        # For value error
        if len(Y_error_accum) > 0:
            Y_error_t = torch.stack(Y_error_accum, dim=0)  # [N, M]
            # Integrate: sum over time steps and average over trajectories
            Y_error = torch.sqrt(Y_error_t.mean(dim=1).sum() * dt / (self.T + 1e-12))
        else:
            Y_error = torch.tensor(float("nan"), device=self.device)

        
        if mu_gaussian_loss_count > 0:
            mu_gaussian_loss_final = mu_gaussian_loss_accum / mu_gaussian_loss_count
        else:
            mu_gaussian_loss_final = torch.tensor(float("nan"), device=self.device)
        
        return torch.stack(X_buffer, dim=1), torch.stack(Y_buffer, dim=1), Y_buffer[0], \
               torch.stack(action_buffer, dim=1), torch.stack(mu_logsigma_buffer, dim=1), \
               Y_error, control_error, \
               total_loss_critic_sum, total_loss_actor_sum, mu_gaussian_loss_final


    def train_all(self, n, N_Iter=10, start_iter=0):
        
        self._save_params_json(self.output_folder)
        start_time = time.time()
        self.X0 = torch.cat([self.Xi] * self.M)
        if start_iter == 0:
            self.bound_list.append(self.net_actor.bound.detach().clone())
        for it_pre_onlycritic in range(n):
            X_pre, Y_pre, Y0_pre, _, _, _, _, loss_critic_pre = self.train_onlycritic(it_pre_onlycritic)
            #just for test _, _, _, _, _, _, _, loss_actor_pre = self.train_onlyactor(it_pre_onlycritic)
        for it in range(start_iter, start_iter + N_Iter):
            if it < 500:
                X_pre, Y_pre, Y0_pre, _, _, _, _, loss_critic_pre = self.train_onlycritic(it)
            X_pred, Y_pred, Y0, action_buffer, mu_logsigma_buffer, Y_error, control_error, loss_critic_final, loss_actor_final, mu_gaussian_loss_final = self.train_loop(it)
            self._append_std_json(it, mu_logsigma_buffer)
            self.J_list.append(Y_pred.detach().clone())
            self.Y_error_list.append(Y_error.detach())
            self.control_error_list.append(
                control_error.detach() if control_error is not None else torch.tensor(float("nan"))
            )
            
            if it % 2 == 0:
                elapsed = time.time() - start_time
                print("action_buffer.mean() = ", action_buffer.mean())
                
                print(
                    f"It: {it}, Time: {elapsed:.2f}, "
                    f"Loss_critic: {to_scalar(loss_critic_final):.3e}, "
                    f"Loss_actor: {to_scalar(loss_actor_final):.3e}"
                )

            if it % 50 == 0:
                self.get_memory("GB")
                
            if (it + 1) % self.eval_every == 0:
                score, detail = self._evaluate_on_fixed()
                self.Y_error_list[-1] = torch.tensor(detail["Y_error"])
                self.control_error_list[-1] = torch.tensor(detail["control_error"] if detail["control_error"] is not None else float("nan"))
                self.mu_gaussian_loss_list[-1] = torch.tensor(detail["mu_gaussian_error"])
                print(f"[eval] it={it}  score={score:.6g}  (Y_err={detail['Y_error']:.6g}, "
                      f"ctrl_err={detail['control_error'] if detail['control_error'] is not None else 'NA'})")
                self._maybe_update_top3_eval(it=it, score=score)
            slope_diag_every = int(self.params_train.get("slope_diag_every", self.eval_every))
            if slope_diag_every > 0 and ((it + 1) % slope_diag_every == 0 or it == start_iter):
                diag = self.diagnose_actor_slope()
                print(
                    "[actor-slope] "
                    f"target={diag['target']:.6f} diag_mean={diag['diag_mean']:.6f} "
                    f"diag_rmse={diag['diag_rmse']:.3e} cross_rms={diag['cross_rms']:.3e} "
                    f"time_rms={diag['time_rms']:.3e}"
                )

        print("The training is over!")
        
        self._save_params_json(self.output_folder)
        if len(self.topk_eval) > 0:
            best = sorted(self.topk_eval, key=lambda d: d["score"])[0]
            print(f"[summary] eval best: iter={best['iter']}, score={best['score']:.6g}")
        else:
            print("[summary] no eval Top-3 recorded.")


    ################
    ## Plot
    ################

    def plot_list(self, list_, ylabel, legend, color=None, title=None,
              ylabel_log=False, xticks=None, baseline=None, filename=None):
        is_control = filename is not None and "control" in filename.lower()
        return plot_training_series(
            list_, ylabel, legend, output_folder=self.output_folder, color=color,
            title=title, ylabel_log=ylabel_log, xticks=xticks, baseline=baseline, filename=filename,
            log_limits=True, ylim=(-0.2, 1) if is_control else None,
            yticks=[-0.2, 0, 0.25, 0.5, 0.75, 1] if is_control else None,
        )

    def plot_loss(self, CriticLoss, ActorLoss, filename):
        return plot_actor_critic_loss(
            CriticLoss, ActorLoss, filename, output_folder=self.output_folder, show=True,
        )


    def plot_kl_std_loss(self, kl_loss, std_loss, mu_gaussian_loss, filename=None):
        gamma_entropy = self.params_train.get("entropy_alpha", 0.0)
        q1 = self.params_equ.get("q1", 1.0)
        std_reference = np.sqrt(gamma_entropy / (2.0 * q1)) ** 2 if gamma_entropy > 0 else None
        return plot_policy_diagnostics(
            kl_loss, std_loss, mu_gaussian_loss, output_folder=self.output_folder,
            filename=filename, std_reference=std_reference,
        )


    @torch.no_grad()
    def plot_controlfunction(self, n=1001, region=(9, 11), filename=None,
                             sample_exact=False, component_idx=0):
        """Plot one control component while fixing other state coordinates at Xi.

        D is the state/action dimension. The value function remains scalar;
        component_idx selects a coordinate, not a player.
        """
        component_idx = int(component_idx)
        if not 0 <= component_idx < self.D:
            raise ValueError("component_idx must select a state/action coordinate")
        t = torch.linspace(0, self.T, n, device=self.device)
        x = torch.linspace(region[0], region[1], n, device=self.device)
        times, coordinate = torch.meshgrid(t, x, indexing="ij")
        states = self.Xi.to(self.device).reshape(1, 1, self.D).expand(n, n, self.D).clone()
        states[..., component_idx] = coordinate
        inputs = torch.cat((times.unsqueeze(-1), states), dim=-1)
        mean, _ = self.net_actor(inputs)
        approximate = mean[..., component_idx]
        if self.params_train["entropy_alpha"] == 0:
            benchmark, _ = self.compute_exact_no_exploratory(states)
            benchmark = benchmark[..., component_idx]
        else:
            policy, _ = self.compute_exact_entropy(states)
            benchmark = policy.loc[..., component_idx]
            if sample_exact:
                # Retain the same sampled reference along each fixed-x time column.
                noise = torch.randn_like(benchmark[:1])
                benchmark = benchmark + policy.scale[..., component_idx] * noise
        error = (approximate - benchmark).abs() / benchmark.abs().clamp_min(1e-12)
        return plot_control_error_heatmap(
            error.cpu().numpy(), x_region=region, horizon=self.T,
            output_folder=self.output_folder, filename=filename,
            component_idx=component_idx if self.D > 1 else None,
        )

    def _load_eval_topk_meta(self):
        """Read metadata written by _persist_topN_alias(), falling back to alias/glob scans.
        Return entries sorted by ascending score:
        [{"iter": int, "score": float, "actor": path, "critic": path}, ...]."""
        meta_path = os.path.join(self.output_folder, "eval_topk_meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    li = json.load(f)
                
                cleaned = []
                for d in li:
                    if all(k in d for k in ["iter", "score", "actor", "critic"]):
                        cleaned.append({"iter": int(d["iter"]), "score": float(d["score"]),
                                        "actor": d["actor"], "critic": d["critic"]})
                cleaned.sort(key=lambda d: d["score"])
                return cleaned
            except Exception as e:
                print(f"[warn] read meta failed: {e}")

        
        cand = []
        for a in glob.glob(os.path.join(self.output_folder, "actor_eval_top*_itr*.pth")):
            
            base = os.path.basename(a)
            
            try:
                itr = int(base.split("_itr")[-1].split(".pth")[0])
            except Exception:
                itr = -1
            c = os.path.join(self.output_folder, f"critic_eval_{base.split('actor_eval_')[-1]}")
            if os.path.isfile(c):
                cand.append({"iter": itr, "score": float('inf'), "actor": a, "critic": c})
        cand.sort(key=lambda d: (d["score"], d["iter"]))
        return cand[:3]

    def _load_ckpt_pair(self, actor_path, critic_path):
        """Load an actor/critic weight pair without changing output_folder."""
        try:
            sd_actor = torch.load(actor_path, map_location=self.device)
            sd_critic = torch.load(critic_path, map_location=self.device)
            self.net_actor.load_state_dict(sd_actor)
            self.net_critic.load_state_dict(sd_critic)
            self.net_actor.eval(); self.net_critic.eval()
            print(f"[load] actor={actor_path}\n[load] critic={critic_path}")
            return True
        except Exception as e:
            print(f"[warn] failed to load pair:\n  actor={actor_path}\n  critic={critic_path}\n  err={e}")
            return False

    @torch.no_grad()
    def _plot_trajs_for_current_weights(self, filename_suffix: str = "", device=None,
                                        downsample_step=1, smooth_window=1):
        self._ensure_eval_set()
        device = device or self.device

        
        t_test, tau_test = self.t_eval, self.tau_eval
        W_test, P_test   = self.W_eval, self.P_eval

        X0 = torch.cat([self.Xi] * self.M)
        X_pred, Y_pred, control_pred, X_test, _, _ = self.predict(t_test, tau_test, W_test, P_test)

        if self.params_train["entropy_alpha"] == 0:
            control_test, optimal_value_test = self.compute_exact_no_exploratory(X_pred)
            control_test = control_test * torch.ones_like(X_pred)
            mu_pred, log_sigma_pred = self.net_actor(torch.cat([t_test, X_pred], dim=2))
            sigma_pred = torch.exp(log_sigma_pred)
            control_pred2 = mu_pred + sigma_pred * torch.randn_like(mu_pred)
            mean_gaussian_pred = mu_pred
            mean_gaussian_test = control_test
        else:
            optimal_policy_test, optimal_value_test = self.compute_exact_entropy(X_pred)
            control_test = optimal_policy_test.loc + optimal_policy_test.scale * torch.randn_like(optimal_policy_test.loc)
            mu_pred, log_sigma_pred = self.net_actor(torch.cat([t_test, X_pred], dim=2))
            sigma_pred = torch.exp(log_sigma_pred)
            control_pred2 = mu_pred + sigma_pred * torch.randn_like(mu_pred)
            mean_gaussian_pred = mu_pred
            mean_gaussian_test = optimal_policy_test.loc

        t_np = t_test.cpu().detach().numpy()
        Xp_np = X_pred.cpu().detach().numpy()
        Xt_np = X_test.cpu().detach().numpy()
        Yp_np = Y_pred.cpu().detach().numpy()
        Yt_np = optimal_value_test.cpu().detach().numpy()
        ctrl_t_np = control_test.cpu().detach().numpy()
        ctrl_p_np = control_pred.cpu().detach().numpy()
        ctrl_p2_np = control_pred2.cpu().detach().numpy()
        mu_np = mean_gaussian_pred.cpu().detach().numpy()
        mu_star_np = mean_gaussian_test.cpu().detach().numpy()

        return plot_lq_trajectories(
            t_np, mu_np, mu_star_np, -Yp_np, -Yt_np, Xp_np, Xt_np,
            entropy_alpha=self.params_train["entropy_alpha"], output_folder=self.output_folder,
            P_test=P_test, jump_index=self.jump_index, filename_suffix=filename_suffix,
            downsample_step=downsample_step, smooth_window=smooth_window,
        )

    def plot_topk_eval(self, k: int = 3, device=None):
        """Load each Top-k evaluation checkpoint and plot it on the fixed evaluation batch.
        Add a `_top{rank}_itr{itr}_score{:.4f}` suffix to avoid overwriting figures."""
        device = device or self.device
        topk = self._load_eval_topk_meta()
        if not topk:
            print("[info] no eval Top-k found; try training first.")
            return

        self._ensure_eval_set()

        for rank, rec in enumerate(sorted(topk, key=lambda d: d["score"])[:k], start=1):
            ok = self._load_ckpt_pair(rec["actor"], rec["critic"])
            if not ok:
                continue
            suffix = f"_top{rank}_itr{int(rec['iter'])}_score{rec['score']:.4f}"
            print(f"[plot] begin {suffix}")
            self._plot_trajs_for_current_weights(filename_suffix=suffix, device=device,
                                                 downsample_step=10, smooth_window=1)
            print(f"[plot] done  {suffix}")

    def _default_output_folder(self):
        if self.params_train.get("output_dir"):
            return os.path.abspath(self.params_train["output_dir"])
        filename = os.path.basename(os.path.abspath(__file__))
        return os.path.join(
            'output', filename,
            "Network_clamp-10_nomax_net_widthD+10_3_lractor5e-5_lrcritic1e-3",
            'X0_10_lossactor_every20_losscritic_every5',
            'pre_onlycritic_itrsmaller500_and_pre10_before', 
            'fix_multidim_fix_netcritic_output_to_M_1',
            # 'fix_entropy_alpha_0_no_sigmain_control_add_std_loss',
            # 'log_prob_mean_instead_of_sum',
            # 'fix_multidim_Y0_and_kappa_but_optimal_and_reward_are_seperate_dim',
            # 'pre_onlycritic_every10itr', # 
            # 'add_0.01lossKL',
            # "logp_control_no_detach",
            # "A_soft_detach",
            # 'clamp_log_sigma_10_-1_to_-8_-1_to_-10_0_to_-10_0',
            # "seperate_logpi_each_dim", #"fix_adv_sum_and_dMt",
            # "fix_lr_decay_lr_actor_and_lr_critic_detach_V1_in_advantage",
            "schedule_step_onceiteration", #reward_add_entropy_adv_instaed_Aentropy
            # "reward_no_martingle_corrected",
            # "test",
            f"T_{self.params_equ['T']}_N{self.params_equ['N']}_M{self.params_equ['M']}_D{self.params_equ['D']}",
            f"total_itr_{self.params_train['iteration']}", f"q1_{self.params_equ['q1']}_q2_{self.params_equ['q2']}_mu_{self.params_equ['mu']}_sigma_{self.params_equ['sigma']}_beta_{self.params_equ['beta']}", 
            f"lractor_{self.params_train['lr_actor']}_lrcritic_{self.params_train['lr_critic']}",
            f"gamma_soft_{self.params_train['gamma_soft']}_entropy_alpha_{self.params_train['entropy_alpha']}_p_tsallis_{self.params_train['p_tsallis']}",
            f"actor_loss_{self.params_train.get('actor_loss', 'manuscript')}",
            f"actor_input_{self.params_train.get('actor_input_normalization', 'none')}_scale_{self.params_train.get('actor_state_scale', 1.0)}",
            f"coverage_std_{self.params_train.get('state_coverage_std', 0.0)}",
            f"policy_std_{self.params_train.get('policy_fixed_std', 'learned')}",
            "seed2026"
        )

    def _safe_has_history(self):
        return (len(getattr(self, "Y_error_list", [])) > 0 or
                len(getattr(self, "control_error_list", [])) > 0 or
                len(getattr(self, "CriticLoss_list", [])) > 0 or
                len(getattr(self, "ActorLoss_list", [])) > 0)


    def save_and_read_model(self, type: str = "save"):
        ckpt_given = (self.actor_path is not None and self.critic_path is not None
                    and os.path.isfile(self.actor_path) and os.path.isfile(self.critic_path))
        if ckpt_given:
            dir_a = os.path.dirname(os.path.abspath(self.actor_path))
            dir_c = os.path.dirname(os.path.abspath(self.critic_path))
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
            
            params_json_path = os.path.join(self.output_folder, "params.json")
            try:
                params_to_save = {
                    "equation": _to_jsonable(self.params_equ),
                    "train": _to_jsonable(self.params_train)
                }
                with open(params_json_path, 'w') as f:
                    json.dump(params_to_save, f, ensure_ascii=False, indent=2)
                print(f"[save] params.json to {params_json_path}")
            except Exception as e:
                print(f"[warn] failed to save params.json: {e}")
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

    def predict(self, t_star, tau_star=None, W_star=None, P_star=None):  
        self.t_predict_set, self.tau_predict_set, self.W_predict_set, self.P_predict_set = t_star, tau_star, W_star, P_star
        # self.save_and_read_model(type="read")
        self._load_best_eval_for_plot()
        try:
            print("self.Y_error_list len:", len(self.Y_error_list))
            print("Y error: %.3f%%, control error: %.3f%%" % (
                to_scalar(self.Y_error_list[-1]) * 100.0,
                to_scalar(self.control_error_list[-1]) * 100.0
            ))
            self.plot_list([self.Y_error_list, self.control_error_list], "Error_value/Error_control", ["J function", "control"],
                        color=["#E7483D", "#6565AE", "#8FC751"], title=None, ylabel_log=True, baseline=0.01, filename="merton_train")
            self.plot_loss(self.CriticLoss_list, self.ActorLoss_list, filename="merton_loss")
            self.plot_kl_std_loss(self.KLLoss_list, self.StdLoss_list, self.mu_gaussian_loss_list, filename="KL_StdNormal_MuGaussian_loss")
            self.plot_controlfunction(region=[9, 12], filename="merton_heatmap")
        except Exception as e:
            print(f"[warn] history plots skipped: {e}")
        
        X_star, Y_star, _, control_star, _, _, _, _, _, _ = self.train_loop(itr = 1, traj_only=True, predict=True)
        X_test, Y_test, _, control_test, _, _, _, _, _, _ = self.train_loop(itr=1, traj_only=True, _control_exact_=True, predict=True)
        return X_star, Y_star, control_star, X_test, Y_test, control_test

if __name__ == '__main__':
    # np.random.seed(2025)
    # torch.manual_seed(2025)
    # torch.cuda.manual_seed(2025)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed(2026)

    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', type=str, default=None, help='path to actor.pth')
    parser.add_argument('--critic_path', type=str, default=None, help='path to critic.pth')
    parser.add_argument('--resume', action='store_true',
                    help='Resume training from the given actor/critic checkpoints')
    parser.add_argument(
        '--entropy_alpha', '--entropy-alpha', dest='entropy_alpha', type=float, default=None,
        help='0: classical deterministic control; >0: entropy-regularized stochastic policy',
    )
    parser.add_argument(
        '--fixed_std', '--fixed-std', dest='fixed_std', type=float, default=0.1,
        help='Fixed Gaussian training std when entropy_alpha=0; only mean_theta is learned',
    )
    parser.add_argument(
        '--actor_loss', '--actor-loss', dest='actor_loss',
        choices=['manuscript', 'entropy_outside'], default=None,
        help='manuscript: entropy inside stopped advantage (Eq. 3.22-3.23); entropy_outside: ablation',
    )
    parser.add_argument('--actor-input-normalization', choices=['none', 'x_over_x0', 'center_scale'], default=None)
    parser.add_argument('--actor-state-scale', type=float, default=None)
    parser.add_argument('--state-coverage-std', type=float, default=None)
    parser.add_argument('--horizon', type=float, default=None,
                        help='Training horizon T; preserves N and all other experiment settings')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for a fresh training run')
    parser.add_argument('--slope-diag-every', type=int, default=None)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    parameters = {"axes.labelsize": 20, "axes.titlesize": 20,
                  "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}
    plt.rcParams.update(parameters)
    configure_plot_style()
    D=5
    params = {
        "equation": {
            # manuscript0917.pdf, Fig. 2 (5D entropy-regularized LQ):
            # B=0.5 I_d, Sigma=0.3 I_d, R=5 I_d, Q=0.5 I_d, T=10.
            "mu": 0.5, "sigma": 0.3, "lambda": np.linspace(0.2, 0.3, D).tolist(), "z": np.linspace(0.3, 0.2, D).tolist(), "beta": 1.0, "q2": 0.5, "q1": 5.0,
            # "mu": 0.5, "sigma": 0.3, "lambda": np.linspace(0.2, 0.3, D).tolist(), "z": np.linspace(0.3, 0.2, D).tolist(), "beta": 1.0, "q2": 0.2, "q1":2.0,
            # "mu": 1.0, "sigma": 0.3, "lambda": 0.3, "z": 0.2, "beta": 1.0, "q2": 0.5, "q1":0.5,
            "D": D, "T": 100, "M": 500, "N": 1000
        }, 
        "train": {
            "lr_critic": 1e-3, "gamma_critic": 0.2, "milestones_critic": [5000, 50000, 300000], 
            "lr_actor": 5e-5, "gamma_actor": 0.5, "milestones_actor": [3000, 50000, 300000], 
            
            
            "iteration": 1000, "device": device, "weight_decay": 0, "method": 'PO_CRL', "eval_every": 20,
            "entropy_alpha": 0.05, "p_tsallis": 1, "gamma_soft": 0.0,
            "actor_loss": "manuscript",
            "deterministic_sigma_penalty": 0.05,
            "actor_input_normalization": "center_scale",
            "actor_state_scale": 1.0,
            "state_coverage_std": 1.0,
            "slope_diag_every": 20,
            
            # "log_std_mode": "sigmoid_box", "log_std_min": -10, "log_std_max": -1,
        }  
    }
    
    
    if args.actor_path and os.path.isfile(args.actor_path):
        actor_dir = os.path.dirname(os.path.abspath(args.actor_path))
        params_json_path = os.path.join(actor_dir, "params.json")
        if os.path.isfile(params_json_path):
            try:
                with open(params_json_path, 'r') as f:
                    loaded_params = json.load(f)
                
                if "equation" in loaded_params:
                    
                    if "D" in loaded_params["equation"]:
                        D = loaded_params["equation"]["D"]
                        params["equation"]["D"] = D
                        print(f"[load] Updated D = {D} from params.json")
                    
                    for key in ["T", "M", "N"]:
                        if key in loaded_params["equation"]:
                            params["equation"][key] = loaded_params["equation"][key]
                            print(f"[load] Updated params['equation']['{key}'] = {params['equation'][key]} from params.json")
                    
                    for key, value in loaded_params["equation"].items():
                        if key not in ["Xi", "D", "T", "M", "N"]:  
                            params["equation"][key] = value
                if "train" in loaded_params:
                    for key, value in loaded_params["train"].items():
                        if key != "device":  
                            params["train"][key] = value
                    if "actor_input_normalization" not in loaded_params["train"]:
                        # Historical checkpoints were trained on raw actor inputs.
                        params["train"].update(
                            actor_input_normalization="none",
                            state_coverage_std=0.0,
                        )
                print(f"[load] params.json from {params_json_path}")
            except Exception as e:
                print(f"[warn] failed to load params.json from {params_json_path}: {e}")
                print(f"[warn] Will use default parameters (D={D})")

    # Explicit CLI mode selections override params.json; omitted flags preserve
    # the checkpoint configuration.
    if args.entropy_alpha is not None:
        if args.entropy_alpha < 0:
            parser.error('--entropy_alpha must be >= 0')
        params["train"]["entropy_alpha"] = args.entropy_alpha
    if args.actor_loss is not None:
        params["train"]["actor_loss"] = args.actor_loss
    if args.actor_input_normalization is not None:
        params["train"]["actor_input_normalization"] = args.actor_input_normalization
    if args.actor_state_scale is not None:
        params["train"]["actor_state_scale"] = args.actor_state_scale
    if args.state_coverage_std is not None:
        params["train"]["state_coverage_std"] = args.state_coverage_std
    if args.slope_diag_every is not None:
        params["train"]["slope_diag_every"] = args.slope_diag_every
    if params["train"].get("actor_state_scale", 1.0) <= 0:
        parser.error('--actor-state-scale must be > 0')
    if params["train"].get("state_coverage_std", 0.0) < 0:
        parser.error('--state-coverage-std must be >= 0')
    if args.fixed_std <= 0:
        parser.error('--fixed_std must be > 0')
    
    if args.horizon is not None:
        if not np.isfinite(args.horizon) or args.horizon <= 0:
            parser.error('--horizon must be finite and > 0')
        params["equation"]["T"] = args.horizon
    if args.output_dir is not None:
        params["train"]["output_dir"] = os.path.abspath(args.output_dir)

    params_equ, params_train = params["equation"], params["train"]
    params["equation"]["Xi"] = torch.ones(1, params_equ["D"]) * 5
    # params["equation"]["terminal"] = torch.ones(params_equ["M"], params_equ["D"])
    params_act = {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6(), "Sigmoid": nn.Sigmoid(),
                  "LeakyReLU": nn.LeakyReLU(), "GELU": nn.GELU()}

    net_value = Network(inputs=params_equ["D"] + 1, width=params_equ["D"] + 10, depth=3, output=1, params=params_act,
                        activation="Tanh")
    fixed_std_for_net = args.fixed_std if params_train["entropy_alpha"] == 0 else None
    params_train["policy_fixed_std"] = fixed_std_for_net
    if fixed_std_for_net is None:
        print(f"[policy] stochastic Gaussian: learn mean_theta and std_theta; alpha={params_train['entropy_alpha']}")
    else:
        print(f"[policy] classical control: learn mean_theta only; fixed training std={fixed_std_for_net}")
    print(f"[actor loss] {params_train.get('actor_loss', 'manuscript')}")
    net_control = Network_mu_sigma_split(
            inputs=params_equ["D"] + 1,
            width=params_equ["D"] + 10,
            depth=3,
            output=params_equ["D"],  
            params=params_act,
            activation="Tanh",
            fixed_std=fixed_std_for_net,
            log_std_mode=params_train.get("log_std_mode", "min_only"),
            log_std_min=float(params_train.get("log_std_min", -10)),
            log_std_max=float(params_train.get("log_std_max", -1)),
            input_normalization=params_train.get("actor_input_normalization", "center_scale"),
            state_center=params_equ["Xi"].reshape(-1),
            state_scale=float(params_train.get("actor_state_scale", 1.0)),
            time_scale=float(params_equ["T"]),
        )
    model = FBSNN(net_value, net_control, params, args.actor_path, args.critic_path)
    # model.train_all(n=10, N_Iter=params_train["iteration"])
    has_ckpt = (args.actor_path is not None and args.critic_path is not None
                and os.path.isfile(args.actor_path) and os.path.isfile(args.critic_path))

    if args.resume:
        if not has_ckpt:
            print("[warn] --resume is given, but it is illegal --actor_path / --critic_path; The trainning is starting from itr=0!")
            model.train_all(n=10, N_Iter=params_train["iteration"])
            model.net_actor.eval(); model.net_critic.eval()
        else:
            print(f"[resume] is given by \n  actor:  {args.actor_path}\n  critic: {args.critic_path}")
            print("=====Now training is starting from history.=====")
            
            sd_actor = torch.load(args.actor_path, map_location=device)
            sd_critic = torch.load(args.critic_path, map_location=device)
            model.net_actor.load_state_dict(sd_actor)
            model.net_critic.load_state_dict(sd_critic)
            model.net_actor.train(); model.net_critic.train()

            actor_dir = os.path.dirname(os.path.abspath(args.actor_path))
            resume_dir = os.path.join(actor_dir, "resume")
            os.makedirs(resume_dir, exist_ok=True)
            model.output_folder = resume_dir

            
            start_iter = model._extract_iter_from_ckpt_path(args.actor_path)
            if start_iter < 0:
                print(f"[warn] Cannot extract iteration from checkpoint path, starting from iter=0")
                start_iter = 0
            else:
                print(f"[resume] Detected checkpoint at iter={start_iter}")
            
            
            total_iter = params_train["iteration"]
            remaining_iter = max(0, total_iter - start_iter)
            
            if remaining_iter == 0:
                print(f"[warn] Checkpoint is already at or beyond target iteration {total_iter}, no training needed")
            else:
                print(f"[resume] Will train from iter={start_iter} to iter={total_iter} ({remaining_iter} iterations)")
                print(f"[resume] output_folder set to: {model.output_folder}")
                model.train_all(n=10, N_Iter=remaining_iter, start_iter=start_iter)
            model.net_actor.eval(); model.net_critic.eval()

        model.plot_topk_eval(k=3, device=device)

    else:
        if has_ckpt:
            print(f"Loading checkpoints for evaluation/plotting without further training:\n  actor:  {args.actor_path}\n  critic: {args.critic_path}")
            sd_actor = torch.load(args.actor_path, map_location=device)
            sd_critic = torch.load(args.critic_path, map_location=device)
            model.net_actor.load_state_dict(sd_actor); model.net_actor.eval()
            model.net_critic.load_state_dict(sd_critic); model.net_critic.eval()
            model.plot_topk_eval(k=3, device=device)
        else:
            print("[train] No checkpoint supplied; training from scratch.")
            model.train_all(n=20, N_Iter=params_train["iteration"])
            model.net_actor.eval(); model.net_critic.eval()
            model.plot_topk_eval(k=3, device=device)


    # --------------------- Below is the code for result visualization --------------------- #

    t_test, _, W_test, P_test = model.fetch_minibatch()
    t_test, W_test = t_test.to(device), W_test.to(device)
    X0 = torch.cat([params_equ["Xi"]] * params_equ["M"])
    # X_pred, Y_pred, control_pred, X_test, Y_test, control_test = model.predict(t_test, tau_test, W_test, P_test)
    X_pred, Y_pred, control_pred, X_test, _, _ = model.predict(t_test, None, W_test, P_test)

    if params_train["entropy_alpha"] == 0:
        control_test, optimal_value_test = model.compute_exact_no_exploratory(X_pred) 
        control_test = control_test * torch.ones_like(X_pred)  # torch.Size([100, 1001, 1])
        mu_pred, log_sigma_pred = model.net_actor(torch.cat([t_test, X_pred], dim=2))
        sigma_pred = torch.exp(log_sigma_pred)
        control_pred2 = mu_pred + sigma_pred * torch.randn_like(mu_pred)
        mean_gaussian_pred = mu_pred
        mean_gaussian_test = control_test
    else:
        optimal_policy_test, optimal_value_test = model.compute_exact_entropy(X_pred)
        control_test = optimal_policy_test.loc + optimal_policy_test.scale * torch.randn_like(optimal_policy_test.loc)
        # print("control_test", control_test.shape) # torch.Size([100, 1001, 1])
        mu_pred, log_sigma_pred = model.net_actor(torch.cat([t_test, X_pred], dim=2))
        sigma_pred = torch.exp(log_sigma_pred)
        control_pred2 = mu_pred + sigma_pred * torch.randn_like(mu_pred)
        mean_gaussian_pred = mu_pred
        mean_gaussian_test = optimal_policy_test.loc

    Y_test = optimal_value_test
    t_test = t_test.cpu().detach().numpy()
    X_pred = X_pred.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    Y_pred = Y_pred.cpu().detach().numpy()
    Y_test = Y_test.cpu().detach().numpy()
    control_test = control_test.cpu().detach().numpy()# torch.Size([100, 1001, 1])
    control_pred = control_pred.cpu().detach().numpy()
    control_pred2 = control_pred2.cpu().detach().numpy()
    mean_gaussian_pred = mean_gaussian_pred.cpu().detach().numpy()
    mean_gaussian_test = mean_gaussian_test.cpu().detach().numpy()


    if not has_ckpt:
        plot_J_control_function(model.J_list, model.output_folder, J=True)
        plot_J_control_function(model.control_list, model.output_folder, J=False)
    else:
        
        print("[plot] begin plotting trajectories with provided checkpoint")
        model._plot_trajs_for_current_weights(filename_suffix="", device=device,
                                             downsample_step=10, smooth_window=1)
        print("[plot] done plotting trajectories")

    Y_errors = np.sqrt(np.mean((Y_test-Y_pred)**2, 0) / np.mean(Y_test**2, 0))
    control_errors = np.sqrt(np.mean((control_test - control_pred) ** 2, 0) / np.mean(control_test ** 2, 0))
    
    if control_errors.ndim > 1:
        control_errors = np.mean(control_errors, axis=-1)  
    plot_value_control_errors(
        t_test, Y_errors, control_errors, output_folder=model.output_folder,
    )
