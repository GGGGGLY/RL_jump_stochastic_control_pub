from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[2]
for _path in (_ROOT, _ROOT / "tools"):
    if str(_path) not in _sys.path:
        _sys.path.insert(0, str(_path))

from experiment_utils import training_output_folder
from run_io import metadata_for_save, metadata_from_disk, prepare_run_directory, validate_checkpoint_args, check_checkpoint_paths
from experiment_utils import FONT_SIZE, LEGEND_FONT_SIZE, configure_plot_style, configure_policy_output, preserve_random_state
"""Multi-player CARA variant: when entropy_alpha (gamma) == 0, fix the Gaussian
policy standard deviation (default 0.1, overridden by --fixed_std) without
learning log_std. When entropy_alpha > 0, learn log_std and ignore --fixed_std."""
import numpy as np
import time
import torch
import psutil
import json
import os
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm
import multiprocessing as mp
from torch.distributions import Normal
import argparse
import shutil
import json
import glob 
import re
import copy
from experiment_utils import check_nan, to_scalar, _detect_ckpt_set, _load_best_multi_ckpt
from trajectory_plots import plot_multi_cara_trajectories
from experiment_utils import (
    plot_training_series, plot_multi_agent_errors_and_loss, plot_actor_critic_loss,
    plot_gaussian_mean_std, plot_control_error_heatmap,
)
from math import pi

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

    def __init__(self, params, penalty=None):
        super(Network, self).__init__()
        self.params = params
        self.first = nn.Linear(self.params["inputs"], self.params["width"])
        self.last = nn.Linear(self.params["width"], self.params["output"])
        self.network = nn.Sequential(*[
            self.first,
            *[Block(self.params["width"], self.params["params_act"], self.params["activation"])] * self.params["depth"],
            self.last
        ])
        self.penalty = penalty
        self.bound = nn.Parameter(torch.tensor(1.))

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
    """When params['fixed_std'] is positive, do not learn log_std: always return
    log(fixed_std) to fix Gaussian exploration noise for entropy_alpha==0.
    Otherwise learn and clamp log_std as in Multi-player-CARA.py."""
    def __init__(self, params):
        super(Network_mu_sigma_split, self).__init__()

        self.params = params
        self.activation_name = self.params.get("activation", "Tanh")
        self.penalty = self.params.get("penalty", None)

        in_dim = self.params["inputs"]
        width = self.params["width"]
        out_dim = self.params["output"]
        depth = self.params["depth"]

        self.first = nn.Linear(in_dim, width)
        self.hidden = nn.Sequential(*[
            Block(width, self.params["params_act"], self.activation_name)
            for _ in range(depth)
        ])

        self.mean_layer = nn.Linear(width, out_dim)
        fixed_std = self.params.get("fixed_std", None)
        self.use_fixed_std = fixed_std is not None
        if self.use_fixed_std:
            fs = float(fixed_std)
            if fs <= 0:
                raise ValueError(f"fixed_std must be positive, got {fixed_std!r}")
            self.register_buffer("fixed_log_std", torch.tensor(math.log(fs)))
        else:
            self.log_std_layer = nn.Linear(width, out_dim)

        self.bound = nn.Parameter(torch.tensor(1.0))

        self.log_std_min = float(self.params.get("log_std_min", -10.0))
        self.log_std_max = float(self.params.get("log_std_max", -2.0))

        if self.activation_name == "Tanh":
            self.act = nn.Tanh()
        elif self.activation_name == "ReLU":
            self.act = nn.ReLU()
        elif self.activation_name == "GELU":
            self.act = nn.GELU()
        elif self.activation_name in ["Sin", "sin"]:
            self.act = torch.sin
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    def forward(self, x):
        # backbone
        x = self.act(self.first(x))
        x = self.hidden(x)

        mean = self.mean_layer(x)

        if self.use_fixed_std:
            log_std = self.fixed_log_std.expand_as(mean)
        else:
            log_std = self.log_std_layer(x)
            if getattr(self, "entropy_alpha", 0.0) <= 0:
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        if self.penalty is not None:
            if self.penalty == "Sigmoid":
                mean = torch.sigmoid(mean) * self.bound
            elif self.penalty == "Tanh":
                mean = torch.tanh(mean) * self.bound
            elif self.penalty == "ScaledTanh":
                mean = 0.5 * (torch.tanh(mean) + 1.0) * self.bound  # [0, bound]
            else:
                raise RuntimeError("Other penalty has not been implemented!")

        return mean, log_std

def count_jumps(P_times, t0, t1):
    while P_times.dim() < 3:
        P_times = P_times.unsqueeze(-1)
    M_, J, d = P_times.shape
    t0e = t0.unsqueeze(-1)
    t1e = t1.unsqueeze(-1)
    mask = (P_times > t0e) & (P_times <= t1e)
    return mask.sum(dim=1).to(torch.float32)  # (M,d)

def _safe_numpy(x):  
    return np.asarray(x, dtype=float)

def _compute_aggregates_excluding_i(u_vec, coeff_vec, i, n):
    """Return the other-player average: \hat{u c} = (1/n) * sum_{k!=i} u_k * c_k."""
    if len(u_vec) == 1:
        return 0.0
    mask = np.ones_like(u_vec, dtype=bool)
    mask[i] = False
    return float((u_vec[mask] * coeff_vec[mask]).sum() / n)

class FBSNN(nn.Module):  # Forward-Backward Stochastic Neural Network (multi-player CARA)
    def __init__(self, net_value, net_control, params, actor_path, critic_path):
        super().__init__()
        self.params_equ = params["equation"]
        self.params_train = params["train"]
        configure_policy_output(net_control, self.params_train.get("entropy_alpha", 0.0))
        self.params_net_actor = params["net_actor"]
        self.params_net_critic = params["net_critic"]
        self.device = self.params_train["device"]

        self.Xi = self.params_equ["Xi"].to(self.device)    # initial point
        self.T = self.params_equ["T"]                     
        self.M = self.params_equ["M"]                      # number of trajectories
        self.N = self.params_equ["N"]                      # number of time snapshots
        self.D = self.params_equ["D"]                      # number of players

        self.actor_path = actor_path
        self.critic_path = critic_path
        # ===== Multi-player Networks =====
        self.dict = {}
        for i in range(self.D):
            # networks
            self.dict[f'net_critic_{i}'] = net_value[i].to(self.device)
            self.dict[f'net_actor_{i}'] = net_control[i].to(self.device)

            # optimizers with weight decay
            weight_decay_critic = self.params_train.get("weight_decay_critic", 0.0)
            weight_decay_actor = self.params_train.get("weight_decay_actor", 1e-5)
            self.dict[f'optimizer_critic_{i}'] = optim.Adam(
                self.dict[f'net_critic_{i}'].parameters(),
                lr=self.params_train["lr_critic"],
                weight_decay=weight_decay_critic
            )
            self.dict[f'optimizer_actor_{i}'] = optim.Adam(
                self.dict[f'net_actor_{i}'].parameters(),
                lr=self.params_train["lr_actor"],
                weight_decay=weight_decay_actor
            )

            # schedulers
            self.dict[f'scheduler_critic_{i}'] = optim.lr_scheduler.MultiStepLR(
                self.dict[f'optimizer_critic_{i}'],
                milestones=self.params_train["milestones_critic"],
                gamma=self.params_train["gamma_critic"],
            )
            self.dict[f'scheduler_actor_{i}'] = optim.lr_scheduler.MultiStepLR(
                self.dict[f'optimizer_actor_{i}'],
                milestones=self.params_train["milestones_actor"],
                gamma=self.params_train["gamma_actor"],
            )

        self.Y_error_list, self.control_error_list, self.bound_list = [], [], []
        self.CriticLoss_list, self.ActorLoss_list, self.J_list, self.control_list = [], [], [], []
        self.mu_gaussian_loss_list = []

        self.gaussian_mean_list = []  
        self.gaussian_std_list = []   

        self.dt = self.T / self.N
        self._eval_fixed_ready = False

        # {
        #   "iter": int,
        #   "score": float,
        #   "actor": [path_player0, ..., path_player{D-1}],
        #   "critic": [path_player0, ..., path_player{D-1}]
        # }
        self.topk_eval = []
        def _absdir_or_none(p):
            if p is None:
                return None
            p_abs = os.path.abspath(p)

            if os.path.isdir(p_abs):
                return p_abs
            else:
                return os.path.dirname(p_abs)

        if self.actor_path is not None and self.critic_path is not None:
            out_from_actor = _absdir_or_none(self.actor_path)
            out_from_critic = _absdir_or_none(self.critic_path)

            if out_from_actor:
                self.output_folder = out_from_actor
            elif out_from_critic:
                self.output_folder = out_from_critic
        else:
            self.output_folder = self._default_output_folder()

        check_checkpoint_paths(self.actor_path, self.critic_path, multiplayer=True)
        if self.actor_path is None and self.critic_path is None:
            prepare_run_directory(self.output_folder, overwrite=self.params_train.get("_overwrite", False))
        print("output_folder================", self.output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

    # =========================
    # PARAMS SAVE/LOAD HELPERS
    # =========================

    def _save_params(self, save_dir: str):
        """Save current parameters to params.json, serializing torch tensors as needed."""
        try:
            params_to_save = copy.deepcopy(self.params_equ)
            params_to_save.update(copy.deepcopy(self.params_train))
            params_to_save.update(copy.deepcopy(self.params_net_critic))
            params_to_save.update(copy.deepcopy(self.params_net_actor))

            def convert_to_serializable(obj):
                if isinstance(obj, torch.Tensor):
                    return {
                        "_type": "torch.Tensor",
                        "data": obj.cpu().detach().numpy().tolist(),
                        "shape": list(obj.shape)
                    }
                elif isinstance(obj, torch.device):

                    return str(obj)
                elif isinstance(obj, torch.dtype):

                    return str(obj)
                elif isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):

                    return f"<{type(obj).__name__}>"
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    converted = [convert_to_serializable(item) for item in obj]
                    return converted if isinstance(obj, list) else tuple(converted)
                elif isinstance(obj, (int, float, str, bool, type(None))):

                    return obj
                else:

                    try:

                        json.dumps(obj)
                        return obj
                    except (TypeError, ValueError):
                        return str(obj)

            params_serializable = convert_to_serializable(params_to_save)

            full_params = {
                "equation": convert_to_serializable(copy.deepcopy(self.params_equ)),
                "train": convert_to_serializable(copy.deepcopy(self.params_train)),
                "net_critic": copy.deepcopy(self.params_net_critic),
                "net_actor": copy.deepcopy(self.params_net_actor)
            }

            if "params_act" in full_params["net_critic"]:
                full_params["net_critic"]["params_act"] = list(full_params["net_critic"]["params_act"].keys())
            if "params_act" in full_params["net_actor"]:
                full_params["net_actor"]["params_act"] = list(full_params["net_actor"]["params_act"].keys())

            full_params["net_critic"] = convert_to_serializable(full_params["net_critic"])
            full_params["net_actor"] = convert_to_serializable(full_params["net_actor"])

            params_path = os.path.join(save_dir, "params.json")
            with open(params_path, "w") as f:
                json.dump(metadata_for_save(full_params, os.path.dirname(f.name)), f, indent=2, ensure_ascii=False)
            print(f"[save] params saved to {params_path}")
            return True
        except Exception as e:
            import traceback
            print(f"[warn] failed to save params: {e}")
            print(traceback.format_exc())
            return False

    def _load_params(self, load_dir: str):
        """Load parameters from a directory and restore torch.tensor/torch.device values.
        Return the parameter dictionary, or None on failure."""
        params_path = os.path.join(load_dir, "params.json")
        if not os.path.isfile(params_path):
            print(f"[warn] params.json not found in {load_dir}")
            return None

        try:
            with open(params_path, "r") as f:
                content = f.read()

                if content.strip().endswith('"device":'):
                    content = content.strip()[:-10].strip()
                    if content.endswith(','):
                        content = content[:-1]
                    content += '\n  }\n}'
                loaded_params = metadata_from_disk(json.loads(content), load_dir)

            def restore_objects(obj, key=None):
                if isinstance(obj, dict) and obj.get("_type") == "torch.Tensor":
                    data = np.array(obj["data"], dtype=np.float32)  
                    return torch.tensor(data, dtype=torch.float32)  
                elif isinstance(obj, str):

                    if key == "device" and (obj.startswith("cuda") or obj.startswith("cpu")):
                        return torch.device(obj)

                    elif key == "dtype" and obj.startswith("torch."):
                        dtype_map = {
                            "torch.float32": torch.float32,
                            "torch.float64": torch.float64,
                            "torch.int32": torch.int32,
                            "torch.int64": torch.int64,
                            "torch.bool": torch.bool,
                        }
                        return dtype_map.get(obj, obj)
                    else:
                        return obj
                elif isinstance(obj, dict):
                    return {k: restore_objects(v, key=k) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [restore_objects(item) for item in obj]
                else:
                    return obj

            restored_params = {}
            for key in ["equation", "train", "net_critic", "net_actor"]:
                if key in loaded_params:
                    restored_params[key] = restore_objects(loaded_params[key])

            if "net_critic" in restored_params and "params_act" in restored_params["net_critic"]:
                act_names = restored_params["net_critic"]["params_act"]
                restored_params["net_critic"]["params_act"] = {
                    "Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(),
                    "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()
                }
            if "net_actor" in restored_params and "params_act" in restored_params["net_actor"]:
                act_names = restored_params["net_actor"]["params_act"]
                restored_params["net_actor"]["params_act"] = {
                    "Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(),
                    "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()
                }

            print(f"[load] params loaded from {params_path}")
            return restored_params
        except Exception as e:
            import traceback
            print(f"[warn] failed to load params: {e}")
            print(traceback.format_exc())
            return None

    def _update_params_from_dict(self, params_dict: dict):
        """Update model parameter attributes from a loaded parameter dictionary."""
        if params_dict is None:
            return False

        try:
            if "equation" in params_dict:
                self.params_equ.update(params_dict["equation"])

                if "Xi" in params_dict["equation"]:
                    self.Xi = params_dict["equation"]["Xi"].to(self.device)
                if "T" in params_dict["equation"]:
                    self.T = params_dict["equation"]["T"]
                if "M" in params_dict["equation"]:
                    self.M = params_dict["equation"]["M"]
                if "N" in params_dict["equation"]:
                    self.N = params_dict["equation"]["N"]
                if "D" in params_dict["equation"]:
                    self.D = params_dict["equation"]["D"]
                self.dt = self.T / self.N

            if "train" in params_dict:
                self.params_train.update(params_dict["train"])
                self.params_train["device"] = self.device  # Keep the device chosen on this machine.

            if "net_critic" in params_dict:
                self.params_net_critic.update(params_dict["net_critic"])

            if "net_actor" in params_dict:
                self.params_net_actor.update(params_dict["net_actor"])

            print("[update] model params updated from checkpoint")
            return True
        except Exception as e:
            print(f"[warn] failed to update params: {e}")
            return False

    # =========================
    # EVALUATION HELPERS
    # =========================

    def _ensure_reference_solution(self):
        """Lazily prepare analytical controls for evaluation and plotting only."""
        if self.params_train["entropy_alpha"] == 0:
            if not hasattr(self, "control_exact_no_exploratory"):
                self.compute_exact_no_exploratory()
            self.control_exact = self.control_exact_no_exploratory
        else:
            if not hasattr(self, "soft_mean_control"):
                self.compute_exact_soft(max_iter=200)
            self.control_exact = self.soft_mean_control

    def _ensure_eval_set(self):
        """Create the fixed evaluation batch once, using a fixed random seed so
        checkpoint evaluations remain comparable across iterations."""
        if self._eval_fixed_ready:
            return

        torch_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        try:
            torch.manual_seed(123456)
            np.random.seed(123456)
            with torch.no_grad():
                t, tau, W, B, P, P0 = self.fetch_minibatch()
            self.t_eval, self.tau_eval, self.W_eval, self.B_eval, self.P_eval, self.P0_eval = t, tau, W, B, P, P0
            self._eval_fixed_ready = True
        finally:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(np_state)

    @preserve_random_state
    @torch.no_grad()
    def _evaluate_on_fixed(self):
        self._ensure_eval_set()
        self.t_predict_set, self.tau_predict_set = self.t_eval, self.tau_eval
        self.W_predict_set, self.B_predict_set = self.W_eval, self.B_eval
        self.P_predict_set, self.P0_predict_set = self.P_eval, self.P0_eval
        out = self.train_loop(itr=-1, traj_only=True, predict=True)

        Y_error = out[5]
        control_error = out[6]

        score = self._to_float(Y_error)
        if control_error is not None:
            score = 0.5 * (score + self._to_float(control_error))

        return score, {
            "Y_error": self._to_float(Y_error),
            "control_error": (None if control_error is None else self._to_float(control_error)),
        }

    def save_current_eval_errors(self, filename: str = "current_eval_errors.json"):
        """Save current relative value/control errors to output_folder."""
        os.makedirs(self.output_folder, exist_ok=True)
        score, detail = self._evaluate_on_fixed()
        payload = {
            "Y_error": detail["Y_error"],
            "control_error": detail["control_error"],
            "score": score,
        }
        out_path = os.path.join(self.output_folder, filename)
        with open(out_path, "w") as f:
            json.dump(metadata_for_save(payload, os.path.dirname(f.name)), f, indent=2, ensure_ascii=False)
        print(f"[eval] current errors saved to {out_path}")
        return payload

    @staticmethod
    def _to_float(x):
        """Safely convert a Tensor, ndarray or scalar to float for evaluation scores."""
        try:
            if torch.is_tensor(x):
                if x.numel() == 1:
                    return float(x.detach().cpu().item())
                return float(x.detach().cpu().mean().item())
            if isinstance(x, np.ndarray):
                return float(x.reshape(-1).mean())
            return float(x)
        except Exception:
            return math.inf

    def _eval_ckpt_unique_paths(self, it: int):
        """Build unique checkpoint paths for this iteration, one actor/critic per player.
        Return actor_paths and critic_paths, each with length D."""
        actor_paths = []
        critic_paths = []
        for i in range(self.D):
            a_path = os.path.join(self.output_folder, f"actor_player{i}_eval_it{it}.pth")
            c_path = os.path.join(self.output_folder, f"critic_player{i}_eval_it{it}.pth")
            actor_paths.append(a_path)
            critic_paths.append(c_path)
        return actor_paths, critic_paths

    def _persist_topN_alias(self):
        """Refresh ranked checkpoint files from self.topk_eval (sorted by score),
        create Top-1 aliases without iteration numbers, and write metadata JSON."""

        self.topk_eval.sort(key=lambda d: d["score"])

        try:
            for p in glob.glob(os.path.join(self.output_folder, "actor_player*_eval_top*.pth")):
                os.remove(p)
            for p in glob.glob(os.path.join(self.output_folder, "critic_player*_eval_top*.pth")):
                os.remove(p)
        except Exception as e:
            print(f"[warn] cleanup old top alias files failed: {e}")

        for rank, rec in enumerate(self.topk_eval[:3], start=1):
            itr = rec["iter"]
            actor_paths = rec["actor"]
            critic_paths = rec["critic"]
            for i in range(self.D):
                a_src = actor_paths[i]
                c_src = critic_paths[i]
                a_dst = os.path.join(self.output_folder, f"actor_player{i}_eval_top{rank}_itr{itr}.pth")
                c_dst = os.path.join(self.output_folder, f"critic_player{i}_eval_top{rank}_itr{itr}.pth")
                try:
                    shutil.copyfile(a_src, a_dst)
                    shutil.copyfile(c_src, c_dst)
                except Exception as e:
                    print(f"[warn] copying top{rank} aliases for player {i} failed: {e}")

        try:
            best = self.topk_eval[0]
            actor_paths = best["actor"]
            critic_paths = best["critic"]
            for i in range(self.D):
                a_src = actor_paths[i]
                c_src = critic_paths[i]
                a_quick = os.path.join(self.output_folder, f"actor_player{i}_eval_top1.pth")
                c_quick = os.path.join(self.output_folder, f"critic_player{i}_eval_top1.pth")
                shutil.copyfile(a_src, a_quick)
                shutil.copyfile(c_src, c_quick)
        except Exception as e:
            print(f"[warn] write top1 quick alias failed: {e}")

        meta_path = os.path.join(self.output_folder, "eval_topk_meta.json")
        try:
            with open(meta_path, "w") as f:
                json.dump(metadata_for_save(self.topk_eval[:3], os.path.dirname(f.name)), f, indent=2)
        except Exception as e:
            print(f"[warn] write meta failed: {e}")

        self._save_params(self.output_folder)

    def _maybe_update_top3_eval(self, it: int, score: float):
        """Save all player weights when the score qualifies for the Top-3 evaluation list."""

        if len(self.topk_eval) < 3 or score < max(self.topk_eval, key=lambda d: d["score"])["score"]:
            actor_paths, critic_paths = self._eval_ckpt_unique_paths(it)
            try:

                for i in range(self.D):
                    torch.save(self.dict[f'net_actor_{i}'].state_dict(), actor_paths[i])
                    torch.save(self.dict[f'net_critic_{i}'].state_dict(), critic_paths[i])

                if len(actor_paths) > 0:
                    ckpt_dir = os.path.dirname(actor_paths[0])
                    self._save_params(ckpt_dir)
            except Exception as e:
                print(f"[warn] saving eval ckpt failed: {e}")
                return

            entry = {
                "iter": int(it),
                "score": float(score),
                "actor": actor_paths,
                "critic": critic_paths,
            }

            if len(self.topk_eval) < 3:
                self.topk_eval.append(entry)
            else:

                worst_idx = max(range(3), key=lambda i: self.topk_eval[i]["score"])
                old = self.topk_eval[worst_idx]
                self.topk_eval[worst_idx] = entry
                try:
                    for p in old["actor"]:
                        if os.path.isfile(p):
                            os.remove(p)
                    for p in old["critic"]:
                        if os.path.isfile(p):
                            os.remove(p)
                except Exception as e:
                    print(f"[warn] cleanup old eval ckpt failed: {e}")

            self._persist_topN_alias()
            print(f"[eval] accepted to Top-3: iter={it}, score={score:.6g}")
        else:
            print(f"[eval] not better than current Top-3: iter={it}, score={score:.6g}")

    def _load_best_eval_for_plot(self):
        """Load the Top-1 multi-player aliases for plotting; retain in-memory weights if absent."""

        all_exist = True
        a_paths = []
        c_paths = []
        for i in range(self.D):
            a_p = os.path.join(self.output_folder, f"actor_player{i}_eval_top1.pth")
            c_p = os.path.join(self.output_folder, f"critic_player{i}_eval_top1.pth")
            if not (os.path.isfile(a_p) and os.path.isfile(c_p)):
                all_exist = False
                break
            a_paths.append(a_p)
            c_paths.append(c_p)

        if all_exist:
            try:
                for i in range(self.D):
                    self.dict[f'net_actor_{i}'].load_state_dict(
                        torch.load(a_paths[i], map_location=self.device)
                    )
                    self.dict[f'net_critic_{i}'].load_state_dict(
                        torch.load(c_paths[i], map_location=self.device)
                    )
                    self.dict[f'net_actor_{i}'].eval()
                    self.dict[f'net_critic_{i}'].eval()
                print("[load] using multi-player eval top1 ckpt for plotting.")
            except Exception as e:
                raise RuntimeError("Failed to load multi-player Top-1 checkpoints") from e
        else:
            print("[info] no complete multi-player eval top1 ckpt found; using in-memory weights.")

    def compute_exact_no_exploratory(self):
        """
        Multi-player CARA with relative performance (no entropy).

        We solve for a stationary Nash equilibrium with constant controls u_i^*.

        Value ansatz (per player i):
            V^i(x, y_i)
            = - 1 / K_i^* * exp(
                - (1/delta_i) * ((1 - theta_i/n) x_i - theta_i y_i)
            ),
            y_i = (1/n) * sum_{k != i} x_k.

        For fixed others' controls u_{-i}, the generator contribution has the form
            (L^{i,u} V^i) / V^i = Psi_i(u_i; u_{-i}) + C_i(u_{-i}),

        where Psi_i collects all terms that depend on u_i, and C_i all terms
        independent of u_i (they still depend on u_{-i}).

        The Nash equilibrium u^* solves, for all i,
            Psi_i'(u_i^*; u_{-i}^*) = 0,

        and the stationary constants are
            K_i^* = beta_disc - (Psi_i(u_i^*; u_{-i}^*) + C_i(u^*)).
        """
        import numpy as np
        from scipy.optimize import fsolve

        D = self.D
        n = D  # you are using 1/n also for "others"-average

        # ----- parameters -----
        mu    = np.array(self.params_equ["mu"],    dtype=float)  # [D]
        nu    = np.array(self.params_equ["nu"],    dtype=float)  # [D]
        sigma = np.array(self.params_equ["sigma"], dtype=float)  # [D]
        alpha = np.array(self.params_equ["alpha"], dtype=float)  # [D]
        beta  = np.array(self.params_equ["beta"],  dtype=float)  # [D]
        lam   = np.array(self.params_equ["lambda"],dtype=float)  # [D]
        lam0  = float(self.params_equ["lambda0"])
        delta = np.array(self.params_equ["delta"], dtype=float)  # [D]
        theta = np.array(self.params_equ["theta"], dtype=float)  # [D]

        beta_disc = float(self.params_equ["rho"])  # discount rate

        # effective risk aversions:
        gamma = (1.0 - theta / n) / delta   # gamma_i = (1 - theta_i/n)/delta_i
        eta   = theta / delta               # eta_i   = theta_i/delta_i

        # ---------- 1) Define F(u) = (Psi_1',...,Psi_D') ----------
        def F(u_vec):
            """
            u_vec: shape (D,)
            F_i(u) = d/du_i Psi_i(u_i; u_{-i}) with hats taken over k != i.

            This includes:
            - own drift/vol/jump contributions
            - cross terms via hat{u sigma}_i, hat{u beta}_i, which depend only on u_{-i}.
            """
            u = np.asarray(u_vec, dtype=float)
            F_val = np.zeros(D, dtype=float)

            for i in range(D):
                gi  = gamma[i]
                et  = eta[i]
                mui = mu[i]
                nui = nu[i]
                sigi = sigma[i]
                ali = alpha[i]
                beti = beta[i]
                lami = lam[i]

                # hats exclude player i -> independent of u_i for derivative
                if D > 1:
                    mask = np.ones(D, dtype=bool)
                    mask[i] = False
                    hat_u_sigma = float((u[mask] * sigma[mask]).sum() / n)
                    hat_u_beta  = float((u[mask] * beta[mask]).sum() / n)
                else:
                    hat_u_sigma = 0.0
                    hat_u_beta  = 0.0

                ui = u[i]

                # Psi_i'(u_i):
                #   -gi mu_i
                # + gi^2 (nu_i^2 + sigma_i^2) u_i
                # - gi eta_i sigma_i hat{u sigma}_i
                # + lambda_i alpha_i gi (1 - e^{-gi alpha_i u_i})
                # + lambda_0 beta_i gi (1 - e^{eta_i hat{u beta}_i} e^{-gi beta_i u_i})
                term_gauss  = -gi * mui + gi**2 * (nui**2 + sigi**2) * ui
                term_cross  = -gi * et * sigi * hat_u_sigma
                term_jump_i = lami * ali * gi * (1.0 - np.exp(-gi * ali * ui))
                term_jump0  = lam0 * beti * gi * (
                    1.0 - np.exp(et * hat_u_beta) * np.exp(-gi * beti * ui)
                )

                F_val[i] = term_gauss + term_cross + term_jump_i + term_jump0

            return F_val

        # ---------- 2) Solve F(u) = 0 for u_star (vector root) ----------
        # initial guess: decoupled Merton+jump approximation
        u0 = np.zeros(D, dtype=float)
        for i in range(D):
            denom = gamma[i] * (
                nu[i]**2 + sigma[i]**2
                + lam[i] * alpha[i]**2
                + lam0   * beta[i]**2
            )
            u0[i] = mu[i] / denom if abs(denom) > 1e-12 else 0.0

        # Solve the full system jointly so cross-terms are respected
        u_star = fsolve(F, u0)
        u_star = np.asarray(u_star, dtype=float)
                # ---------- 3) Compute Psi_i(u_i^*; u_{-i}^*) ----------
        Psi_star = np.zeros(D, dtype=float)
        for i in range(D):
            gi  = gamma[i]
            et  = eta[i]
            mui = mu[i]
            nui = nu[i]
            sigi = sigma[i]
            ali = alpha[i]
            beti = beta[i]
            lami = lam[i]
            ui  = u_star[i]

            # hats at equilibrium (exclude i)
            if D > 1:
                mask = np.ones(D, dtype=bool)
                mask[i] = False
                hat_u_sigma = float((u_star[mask] * sigma[mask]).sum() / n)
                hat_u_beta  = float((u_star[mask] * beta[mask]).sum()  / n)
            else:
                hat_u_sigma = 0.0
                hat_u_beta  = 0.0

            # Psi_i(u_i; u_{-i}) with your splitting:
            Psi_star[i] = (
                -gi * mui * ui
                + 0.5 * gi**2 * (nui**2 + sigi**2) * ui**2
                - gi * et * sigi * hat_u_sigma * ui
                + lami * (np.exp(-gi * ali * ui) - 1.0 + gi * ali * ui)
                + lam0 * ( np.exp(-gi * beti * ui + et * hat_u_beta) + gi * beti * ui )
            )
        print("Psi_star", Psi_star)

                # ---------- 4) Compute C_i(u^*) (u_i-independent constants) ----------
        C = np.zeros(D, dtype=float)
        for i in range(D):
            mask = np.ones(D, dtype=bool)
            mask[i] = False

            # hats from others at equilibrium (k != i)
            hat_u_mu   = (u_star[mask] * mu[mask]).sum()    / n
            hat_u_beta = (u_star[mask] * beta[mask]).sum()  / n

            # Brownian constants:
            #   idio: (1/n^2) * sum (u_k*nu_k)^2
            hat_u_nu2  = ((u_star[mask] * nu[mask])**2).sum() / (n**2)
            #   common: ( (1/n) * sum (u_k*sigma_k) )^2
            hat_u_sig  = (u_star[mask] * sigma[mask]).sum() / n
            hat_u_sig2 = hat_u_sig ** 2

            Ci  = (theta[i] / delta[i]) * hat_u_mu
            Ci += 0.5 * (theta[i]**2 / delta[i]**2) * (hat_u_nu2 + hat_u_sig2)

            # idiosyncratic jumps of others:
            coef = theta[i] / (delta[i] * n)
            for k in range(D):
                if k == i:
                    continue
                Jk = u_star[k] * alpha[k]
                Ci += lam[k] * (np.exp(coef * Jk) - 1.0 - coef * Jk)

            # common jump (constant part under your split)
            Ci += lam0 * (-1.0 - (theta[i] / delta[i]) * hat_u_beta)

            C[i] = Ci
        print("C=", C)

        # ---------- 5) Lambda_i^*, K_i^* ----------
        Lambda_star = Psi_star + C
        K_star = beta_disc - Lambda_star

        if np.any(K_star <= 0):
            print("[warn] beta_disc <= Lambda_i^* for some i: stationary V^i may be infinite.")

        # store as tensors
        self.control_exact_no_exploratory = torch.tensor(
            u_star, device=self.device, dtype=torch.get_default_dtype()
        )
        self.K_star = torch.tensor(
            K_star, device=self.device, dtype=torch.get_default_dtype()
        )
        self.gamma = torch.tensor(
            gamma, device=self.device, dtype=torch.get_default_dtype()
        )

        # print("u_star.shape", u_star.shape) #(5,)
        print("Exact control u*:", u_star)
        print("Psi*(u*):", Psi_star)
        print("C(u*):", C)
        print("K* = beta_disc - (Psi*+C):", K_star)

    def exact_value(self, t, x):
        """
        Stationary multi-player CARA value function.

        t: (M, N+1, 1), unused (stationary).
        x: (M, N+1, D), wealths of all players.

        Returns:
        V: (M, N+1, D), with i-th component equal to
        V^i(x,y_i) = - 1/K_i^* * exp(
            -1/delta_i * ((1 - theta_i/n) x_i - theta_i y_i)
        ),
        where y_i = (sum_{k != i} x_k)/n.
        """
        assert x.shape[-1] == self.D
        M, Tn, D = x.shape
        n = D

        device = x.device
        dtype = x.dtype
        delta = torch.as_tensor(
            self.params_equ["delta"], device=device, dtype=dtype
        ).view(1, 1, D)
        theta = torch.as_tensor(
            self.params_equ["theta"], device=device, dtype=dtype
        ).view(1, 1, D)

        K = self.K_star.to(device=device, dtype=dtype).view(1, 1, D)

        # sum over players
        sum_x = x.sum(dim=-1, keepdim=True)        # (M, N+1, 1)

        # y_i = (sum_{k != i} x_k)/n
        y = (sum_x - x) / float(n)                 # (M, N+1, D)

        coef_self = 1.0 - theta / float(n)
        exponent = - ((coef_self * x - theta * y) / delta)  # (M, N+1, D)

        V = - torch.exp(exponent) / K              # elementwise 1/K_i^*

        return V

    def compute_exact_soft(self, max_iter, tol=1e-10, damping=0.5, verbose=True):
        """Multi-player CARA with relative performance and Shannon entropy regularization.

        Second-order small-jump approximation with cross-term corrections:
            Psi_i(u; u_-i) ~= A_i*u - 1/2*Q_i*u^2
        where
            A_i = -gamma_i*mu_i - gamma_i*eta_i*sigma_i*\widehat{u sigma}_i
                  + lambda0*gamma_i*beta_i*(1-exp(eta_i*\widehat{u beta}_i))
            Q_i = gamma_i^2*(nu_i^2+sigma_i^2) + lambda_i*gamma_i^2*alpha_i^2
                  + lambda0*gamma_i^2*beta_i^2*exp(eta_i*\widehat{u beta}_i).

        Gaussian soft policy:
            u_i ~ N(mu_u_i, sigma_u_i^2), mu_u_i=A_i/Q_i, sigma_u_i^2=tau_i/Q_i.
        Stationary soft-value prefactor:
            K_i_soft = beta - C_i(u_-i*) - A_i^2/(2*Q_i)
                       - (tau_i/2)*log(2*pi*tau_i/Q_i).

        Solve a fixed point for the player policy means because A_i and Q_i
        depend on the other-player averages. Compute C_i from the exact constant
        term, excluding u_i and retaining u_-i*."""

        D = self.D
        n = D
        equ = self.params_equ
        mu    = _safe_numpy(equ["mu"])       # [D]
        nu    = _safe_numpy(equ["nu"])       # [D]
        sigma = _safe_numpy(equ["sigma"])    # [D]
        alpha = _safe_numpy(equ["alpha"])    # [D]
        betaJ = _safe_numpy(equ["beta"])     
        lam   = _safe_numpy(equ["lambda"])   # [D]
        lam0  = float(equ["lambda0"])
        delta = _safe_numpy(equ["delta"])    # [D]
        theta = _safe_numpy(equ["theta"])    # [D]
        beta_disc = float(equ["rho"])

        if "entropy_alpha" in equ:
            tau = _safe_numpy(equ["entropy_alpha"])    # [D]
        else:
            tau = np.ones(D, dtype=float) * 0.1  

        gamma = (1.0 - theta / n) / delta   # [D]
        eta   = theta / delta               # [D]

        u_mean = np.zeros(D, dtype=float)
        for i in range(D):
            Qi0 = (gamma[i]**2) * (nu[i]**2 + sigma[i]**2) \
                  + lam[i] * (gamma[i]**2) * (alpha[i]**2) \
                  + lam0    * (gamma[i]**2) * (betaJ[i]**2)  
            if Qi0 <= 1e-14:
                Qi0 = 1e-14
            Ai0 = -gamma[i] * mu[i]  
            u_mean[i] = Ai0 / Qi0

        for it in range(max_iter):
            u_old = u_mean.copy()

            new_u = np.zeros(D, dtype=float)
            new_Q = np.zeros(D, dtype=float)
            new_A = np.zeros(D, dtype=float)
            var_u = np.zeros(D, dtype=float)

            for i in range(D):

                hat_u_sigma = _compute_aggregates_excluding_i(u_old, sigma, i, n)
                hat_u_beta  = _compute_aggregates_excluding_i(u_old, betaJ, i, n)

                Ai = -gamma[i] * mu[i] \
                     - gamma[i] * eta[i] * sigma[i] * hat_u_sigma \
                     + lam0 * gamma[i] * betaJ[i] * (1.0 - np.exp(eta[i] * hat_u_beta))

                Qi = (gamma[i]**2) * (nu[i]**2 + sigma[i]**2) \
                     + lam[i] * (gamma[i]**2) * (alpha[i]**2) \
                     + lam0    * (gamma[i]**2) * (betaJ[i]**2) * np.exp(eta[i] * hat_u_beta)

                if Qi <= 1e-14:
                    Qi = 1e-14

                ui_mean = Ai / Qi
                ui_var  = tau[i] / Qi

                new_u[i] = ui_mean
                new_Q[i] = Qi
                new_A[i] = Ai
                var_u[i] = ui_var

            u_mean = damping * new_u + (1.0 - damping) * u_old

            if np.linalg.norm(u_mean - u_old, ord=np.inf) < tol:
                if verbose:
                    print(f"[soft] fixed-point converged in {it+1} iters; "
                          f"max|Δu|={np.max(np.abs(u_mean - u_old)):.3e}")
                break
        else:
            if verbose:
                print(f"[soft] fixed-point NOT converged in {max_iter} iters; "
                      f"max|Δu|={np.max(np.abs(u_mean - u_old)):.3e}")

        A = np.zeros(D, dtype=float)
        Q = np.zeros(D, dtype=float)
        for i in range(D):
            hat_u_sigma = _compute_aggregates_excluding_i(u_mean, sigma, i, n)
            hat_u_beta  = _compute_aggregates_excluding_i(u_mean, betaJ, i, n)
            Ai = -gamma[i] * mu[i] \
                 - gamma[i] * eta[i] * sigma[i] * hat_u_sigma \
                 + lam0 * gamma[i] * betaJ[i] * (1.0 - np.exp(eta[i] * hat_u_beta))
            Qi = (gamma[i]**2) * (nu[i]**2 + sigma[i]**2) \
                 + lam[i] * (gamma[i]**2) * (alpha[i]**2) \
                 + lam0    * (gamma[i]**2) * (betaJ[i]**2) * np.exp(eta[i] * hat_u_beta)
            if Qi <= 1e-14:
                Qi = 1e-14
            A[i], Q[i] = Ai, Qi

        var_u = tau / Q  

        C = np.zeros(D, dtype=float)
        for i in range(D):
            if D == 1:

                C[i] = 0.0
                continue

            mask = np.ones(D, dtype=bool); mask[i] = False

            hat_u_mu     = (u_mean[mask] * mu[mask]).sum() / n
            hat_u_sig    = (u_mean[mask] * sigma[mask]).sum() / n
            hat_u_nu2    = ((u_mean[mask] * nu[mask])**2).sum() / (n**2)
            hat_u_sig2   = (hat_u_sig**2)
            hat_u_alpha2 = ((u_mean[mask] * alpha[mask])**2).sum() / (n**2)
            hat_u_beta   = (u_mean[mask] * betaJ[mask]).sum() / n
            hat_u_beta2  = ((u_mean[mask] * betaJ[mask])**2).sum() / (n**2)

            Ci = 0.0

            Ci += (theta[i] / delta[i]) * hat_u_mu
            Ci += 0.5 * (theta[i]**2 / delta[i]**2) * (
                hat_u_nu2 + hat_u_sig2 + hat_u_alpha2 + hat_u_beta2
            )

            for k in range(D):
                if k == i:
                    continue
                coef = theta[i] / (delta[i] * n)
                Jk   = u_mean[k] * alpha[k]
                Ci  += lam[k] * (np.exp(coef * Jk) - 1.0 - coef * Jk)

            coef0 = theta[i] / delta[i]
            Ci   += lam0 * (np.exp(coef0 * hat_u_beta) - 1.0 - coef0 * hat_u_beta)

            C[i] = Ci

        # K_i^{soft} = β - C_i - A_i^2/(2Q_i) - (τ_i/2) * log(2π τ_i / Q_i)
        K_soft = beta_disc - C - (A**2) / (2.0 * Q) - 0.5 * tau * np.log(2.0 * pi * tau / Q)

        if verbose and np.any(K_soft <= 0):
            print("[soft][warn] Some K_i_soft <= 0; the value may diverge. Check the parameters and approximation.")

        dev = self.device
        dtype = torch.get_default_dtype()

        self.soft_mean_control = torch.tensor(u_mean, device=dev, dtype=dtype)   # [D]
        self.soft_var_control  = torch.tensor(var_u, device=dev, dtype=dtype)    # [D]
        self.soft_A            = torch.tensor(A, device=dev, dtype=dtype)        # [D]
        self.soft_Q            = torch.tensor(Q, device=dev, dtype=dtype)        # [D]
        self.K_soft            = torch.tensor(K_soft, device=dev, dtype=dtype)   # [D]
        self.gamma             = torch.tensor(gamma, device=dev, dtype=dtype)    # [D]
        self.eta               = torch.tensor(eta, device=dev, dtype=dtype)      # [D]

        if verbose:
            print("[soft] u_mean:", u_mean)
            print("[soft] var_u :", var_u)
            print("[soft] C     :", C)
            print("[soft] K_soft:", K_soft)

        return u_mean, var_u, K_soft

    @torch.no_grad()
    def exact_value_soft(self, t, x):
        """Stationary soft value in CARA form with K_i_soft.

        Inputs:
            t: (M,N+1,1), unused explicitly for the stationary problem.
            x: (M,N+1,D), all player wealth values.
        Returns:
            V_soft: (M,N+1,D), with
            V_i_soft = -(1/K_i_soft)*exp(-((1-theta_i/n)*x_i-theta_i*y_i)/delta_i)."""
        assert x.shape[-1] == self.D
        M, Tn, D = x.shape
        n = D
        device = x.device
        dt = x.dtype

        if not hasattr(self, "K_soft"):
            raise RuntimeError("K_soft has not been calculated, please call compute_exact_soft() first.")

        delta = torch.as_tensor(self.params_equ["delta"], device=device, dtype=dt).view(1, 1, D)
        theta = torch.as_tensor(self.params_equ["theta"], device=device, dtype=dt).view(1, 1, D)
        K     = self.K_soft.to(device=device, dtype=dt).view(1, 1, D)

        sum_x = x.sum(dim=-1, keepdim=True)       # (M, N+1, 1)
        y     = (sum_x - x) / float(n)            # (M, N+1, D)
        coef_self = 1.0 - theta / float(n)
        exponent  = - ((coef_self * x - theta * y) / delta)
        V_soft    = - torch.exp(exponent) / K
        return V_soft

    def reward_func(self, x, t, dt, delta=None, theta=None):
        """Instantaneous CARA utility vector for all players.
        x: [...,D], including (M,D) or (M,N+1,D).
        t, dt: not used explicitly; apply exp(-rho*t)*dt weighting externally.
        delta, theta: optional overrides of self.params_equ values."""

        orig_dim = x.dim()        # 2 or 3
        if orig_dim == 2:
            x_in = x.unsqueeze(1)  # (M, 1, D)
        else:
            x_in = x

        M, Tn, D = x_in.shape
        n = D
        device = x_in.device
        dtype = x_in.dtype

        if delta is None:
            delta = torch.as_tensor(self.params_equ["delta"], device=device, dtype=dtype)
        if theta is None:
            theta = torch.as_tensor(self.params_equ["theta"], device=device, dtype=dtype)

        delta = delta.view(1, 1, D)
        theta = theta.view(1, 1, D)

        sum_x = x_in.sum(dim=-1, keepdim=True)  # (M, Tn, 1)
        y = (sum_x - x_in) / float(n)           # (M, Tn, D)

        coef_self = 1.0 - theta / float(n)

        exponent = - ((coef_self * x_in - theta * y) / delta)
        reward = - torch.exp(exponent)          # (M, Tn, D)

        if orig_dim == 2:
            reward = reward.squeeze(1)          

        return reward

    # def existence_and_uniqueness(self):
    #     mu, nu, sigma, alpha = self.params_equ["mu"], self.params_equ["nu"], self.params_equ["sigma"], self.params_equ[
    #         "alpha"]
    #     beta, delta, theta = self.params_equ["beta"], self.params_equ["delta"], self.params_equ["theta"],
    #     lam, lam0 = self.params_equ["lambda"], params_equ["lambda0"]
    #     control_bound = 1
    #     condition1 = (max(nu) ** 2 + max(sigma) ** 2 + lam0 * max(beta) ** 2 * np.exp(
    #         2 * control_bound * max(beta) / min(delta)) + max(lam) * max(alpha) ** 2 * np.exp(
    #         control_bound * max(alpha) / min(delta))) / min(delta)
    #     condition2 = min(nu) ** 2 - max(sigma) * (max(sigma) - min(sigma)) - lam0 * max(beta) ** 2 * np.exp(
    #         2 * control_bound * max(beta) / min(delta))
    #     print("Existence and uniqueness condition in the paper:",
    #           "Satisfy" if condition1 < 1 and condition2 > 0 else "Dissatisfy")

    def _clamp_log_sigma(self, log_sigma, itr=None):

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

    def mu(self, t, X, control):
        return control * torch.tensor(self.params_equ["mu"], device=self.device).unsqueeze(0)

    def sigma_i(self, t, X, control):
        return control * torch.tensor(self.params_equ["nu"], device=self.device).unsqueeze(0)

    def sigma_0(self, t, X, control):
        return control * torch.tensor(self.params_equ["sigma"], device=self.device).unsqueeze(0)

    def xi_i(self, t, X, control):
        return control * torch.tensor(self.params_equ["alpha"], device=self.device).unsqueeze(0)

    def xi_0(self, t, X, control):
        return control * torch.tensor(self.params_equ["beta"], device=self.device).unsqueeze(0)

    def net_u_Du(self, net_critic, t, X, k, compute_Du=True, compute_all_critic=True):

        if compute_Du:
            u = net_critic[k](torch.cat([t, X], dim=1))
            Du = torch.autograd.grad(torch.sum(u), X, retain_graph=True, create_graph=True)[0]
            return u, Du
        elif compute_all_critic:
            u_list = [net_critic[i](torch.cat([t, X], dim=1)) for i in range(self.D)]
            return torch.cat(u_list, dim=1), 0
        else:
            return net_critic[k](torch.cat([t, X], dim=1)), 0

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
        Dt = np.zeros((self.M, self.N + 1, 1))
        DW = np.zeros((self.M, self.N + 1, self.D))
        DB = np.zeros((self.M, self.N + 1, 1))
        Dt[:, 1:, :] = self.dt
        DW[:, 1:, :] = np.sqrt(self.dt) * np.random.normal(size=(self.M, self.N, self.D))
        DB[:, 1:, :] = np.sqrt(self.dt) * np.random.normal(size=(self.M, self.N, 1))
        t = np.cumsum(Dt, axis=1)
        tau_grid = torch.distributions.Exponential(self.params_equ["rho"]).sample((self.M, self.N + 1)).to(self.device)
        W = np.cumsum(DW, axis=1)
        B = np.cumsum(DB, axis=1)
        P = np.cumsum(np.random.exponential(scale=[1 / self.params_equ["lambda"][i] for i in range(self.D)],
                                            size=(self.M, 30, self.D)), axis=1)
        P[np.where(P > self.T)] = 0
        count = np.count_nonzero(P, axis=1).max()
        P = P[:, 0:count, :]
        P0 = np.cumsum(np.random.exponential(scale=1 / self.params_equ["lambda0"], size=(self.M, 30)), axis=1)
        P0[np.where(P0 > self.T)] = 0
        count0 = np.count_nonzero(P0, axis=1).max()
        P0 = P0[:, 0:count0]
        self.jump_index = torch.unique(torch.nonzero(torch.tensor(P))[:, 0])
        self.jump_index0 = torch.unique(torch.nonzero(torch.tensor(P0))[:, 0])
        return torch.from_numpy(t).float().to(self.device), tau_grid.to(self.device), torch.from_numpy(W).float().to(
            self.device), torch.from_numpy(B).repeat(1, 1, self.D).float().to(self.device), torch.from_numpy(
            P).float().to(self.device), torch.from_numpy(P0).float().unsqueeze(-1).to(self.device)

    def train_onlycritic(self, itr, traj_only: bool = False, _control_exact_=False, predict: bool = False):
        if predict and not traj_only:
            raise ValueError("Evaluation batches must use traj_only=True")
        if _control_exact_ and not traj_only:
            raise ValueError("Reference-policy rollouts are evaluation-only")
        evaluate_metrics = bool(traj_only or predict)
        if evaluate_metrics:
            self._ensure_reference_solution()
        Y_error = torch.tensor(float("nan"), device=self.device)
        control_error = None
        if not predict:
            t, tau_sampled, W, B, P, P0 = self.fetch_minibatch()
        else:
            t, tau_sampled, W, B, P, P0 = (
                self.t_predict_set,
                self.tau_predict_set,
                self.W_predict_set,
                self.B_predict_set,
                self.P_predict_set,
                self.P0_predict_set
            )

        device = self.device
        dt = self.dt
        beta_disc    = float(self.params_equ["rho"])
        gamma_entropy= self.params_train.get("entropy_alpha", 0.0)
        gamma_soft   = self.params_train.get("gamma_soft", 0.0)
        p_tsallis    = self.params_train.get("p_tsallis", 1.0)

        M, N_plus_1, _ = t.shape
        D = self.D
        assert N_plus_1 == self.N + 1
        mu        = torch.as_tensor(self.params_equ["mu"],    device=device).view(1, 1, D)
        nu        = torch.as_tensor(self.params_equ["nu"],    device=device).view(1, 1, D)
        sigma     = torch.as_tensor(self.params_equ["sigma"], device=device).view(1, 1, D)
        alpha     = torch.as_tensor(self.params_equ["alpha"], device=device).view(1, 1, D)
        beta_jump = torch.as_tensor(self.params_equ["beta"],  device=device).view(1, 1, D)

        for i in range(D):
            self.dict[f'optimizer_actor_{i}'].zero_grad(set_to_none=True)
            self.dict[f'optimizer_critic_{i}'].zero_grad(set_to_none=True)

        # buffer
        X_buffer, V_buffer = [], []
        action_buffer, mu_logsigma_buffer = [], []
        logpi_buffer, tsallis_buffer = [], []

        total_loss_actor  = 0.0
        total_loss_critic = 0.0

        K_actor  = self.params_train.get("K_actor", 20)
        K_critic = self.params_train.get("K_critic", 5)
        critic_loss_accum = 0.0
        critic_chunk_count = 0

        t0 = t[:, 0, :]                             # (M,1)
        X0 = self.Xi.to(device).view(1, D).expand(M, D).contiguous()
        X0 = X0.detach().clone().requires_grad_(True)

        X_buffer.append(X0.detach())

        for k in range(self.N):
            t1 = t[:, k+1, :]                   # (M,1)
            W0 = W[:, k, :] 
            W1 = W[:, k+1, :]
            dW = W[:, k+1, :] - W[:, k, :]      # (M,D)
            B0 = B[:, k, :]
            B1 = B[:, k+1, :]
            dB = B[:, k+1, :] - B[:, k, :]      # (M,1)

            N_idio = count_jumps(P,  t0, t1) - torch.tensor(self.params_equ["lambda"]).unsqueeze(0).to(self.device) * dt    # (M,D)
            N0     = count_jumps(P0, t0, t1) - torch.tensor(self.params_equ["lambda0"]).unsqueeze(0).to(self.device) * dt   # (M,1)

            state = torch.cat([t0, X0], dim=-1)  # (M,1+D)
            # eps_k = torch.randn_like(X0[:, 0]).unsqueeze(-1)

            if not _control_exact_:

                mu_list, log_std_list = [], []
                u_list, logp_list, H_T_list = [], [], []
                for i in range(D):
                    net_a = self.dict[f'net_actor_{i}']
                    mu_i, log_std_i = net_a(state)          # (M,1)
                    # log_std_i = self._clamp_log_sigma(log_std_i, itr)
                    mu_i = mu_i.view(M,1)
                    log_std_i = log_std_i.view(M,1)
                    std_i = torch.exp(log_std_i)
                    eps_i = torch.randn_like(std_i)
                    u_i = mu_i + std_i * eps_i

                    dist_i = torch.distributions.Normal(mu_i, std_i)
                    # logp_i = dist_i.log_prob(u_i).sum(dim=-1, keepdim=True)
                    logp_i = dist_i.log_prob(u_i.detach()).sum(dim=-1, keepdim=True)
                    if p_tsallis != 1:

                        z_i = torch.exp(logp_i)
                        H_T_i = (1.0 / (p_tsallis - 1.0)) * (1.0 - z_i**(p_tsallis - 1.0))
                    else:

                        H_T_i = -logp_i

                    mu_list.append(mu_i)
                    log_std_list.append(log_std_i)
                    u_list.append(u_i)
                    logp_list.append(logp_i)
                    H_T_list.append(H_T_i)

                mu_all        = torch.cat(mu_list, dim=1)          # (M,D)
                log_std_all   = torch.cat(log_std_list, dim=1)     # (M,D)
                u_all         = torch.cat(u_list, dim=1)           # (M,D)
                logp_all      = torch.cat(logp_list, dim=1)        # (M,D)
                H_T_all       = torch.cat(H_T_list, dim=1)         # (M,D)

                action_buffer.append(u_all.detach())
                mu_logsigma_buffer.append(torch.cat([mu_all, log_std_all], dim=1).detach())
                logpi_buffer.append(logp_all.detach())
                tsallis_buffer.append(H_T_all.detach())

                drift       = u_all * mu.squeeze(0)    * dt        # (M,D)
                diff_idio   = u_all * nu.squeeze(0)    * dW        # (M,D)
                diff_common = u_all * sigma.squeeze(0) * dB        # (M,D)
                jump_idio   = u_all * alpha.squeeze(0) * N_idio    # (M,D)
                jump_common = u_all * beta_jump.squeeze(0) * N0    # (M,D)

            else:

                u_all = self.control_exact.unsqueeze(0)            # (1,D) -> broadcast
                drift       = u_all * mu.squeeze(0)    * dt
                diff_idio   = u_all * nu.squeeze(0)    * dW
                diff_common = u_all * sigma.squeeze(0) * dB
                jump_idio   = u_all * alpha.squeeze(0) * N_idio
                jump_common = u_all * beta_jump.squeeze(0) * N0

            X1 = X0 + drift + diff_idio + diff_common + jump_idio + jump_common

            with torch.no_grad():
                reward = self.reward_func(X0, t0, dt)        # CARA utility
                disc = torch.exp(torch.tensor(-beta_disc * dt, device=device))

            V0_players = []
            V1_players = []
            for i in range(D):
                net_c = self.dict[f'net_critic_{i}']
                V0_i = net_c(state)                     # (M,1)
                state1 = torch.cat([t1, X1], dim=-1)
                V1_i = net_c(state1)                   # (M,1)

                # V1_i = (1.0 - gamma_soft) * V1_i + gamma_soft * V0_i.detach()
                V0_players.append(V0_i.squeeze(-1))
                V1_players.append(V1_i.squeeze(-1))
            V0_players = torch.stack(V0_players, dim=-1)   # (M,D)
            V1_players = torch.stack(V1_players, dim=-1)   # (M,D)

            V_buffer.append(V1_players.detach())
            X_buffer.append(X1.detach())

            if not traj_only:

                td_target = reward * dt + disc * V1_players.detach()
                if (gamma_entropy != 0.0) and (not _control_exact_):

                    td_target = td_target + gamma_entropy * H_T_all.detach() * dt

                delta = td_target - V0_players          # (M,D)

                # critic loss: MSE over (M,D)
                loss_critic_step = (delta**2).mean()

                critic_loss_accum += loss_critic_step
                critic_chunk_count += 1

                if ((k + 1) % K_critic == 0) or (k == self.N - 1):
                    loss_critic_chunk = critic_loss_accum / float(critic_chunk_count)
                    for i in range(D):
                        self.dict[f'optimizer_critic_{i}'].zero_grad(set_to_none=True)
                    loss_critic_chunk.backward()
                    for i in range(D):
                        nn.utils.clip_grad_norm_(self.dict[f'net_critic_{i}'].parameters(), 1.0)
                        self.dict[f'optimizer_critic_{i}'].step()
                        self.dict[f'scheduler_critic_{i}'].step()
                    total_loss_critic += loss_critic_chunk.detach()
                    critic_loss_accum = 0.0
                    critic_chunk_count = 0

            X0 = X1.detach().clone().requires_grad_(True)
            t0, W0, B0 = t1, W1, B1
            if evaluate_metrics:
                with torch.no_grad():
                    X_traj = torch.stack(X_buffer, dim=1)      # (M, N+1, D)
                    t_traj = t.to(device)
                    if gamma_entropy == 0:
                        V_exact = self.exact_value(t_traj, X_traj) # (M, N+1, D)
                    else:
                        V_exact = self.exact_value_soft(t_traj, X_traj)
                    V_num = torch.stack(V_buffer, dim=1)       # (M, N, D)
                    if k == (self.N-1):
                        print("V_num.mean", V_num.mean())
                        print("V_exact[:, 1:, :].mean", V_exact[:, 1:, :].mean())
                    Y_error = torch.sqrt(((V_num - V_exact[:, 1:, :])**2 / (V_exact[:, 1:, :]**2 + 1e-12)).mean())

        if not traj_only:
            steps = float(self.N)
            total_loss_critic = total_loss_critic #/ steps

        return (
            torch.stack(X_buffer, dim=1),                 # (M, N+1, D)
            torch.stack(V_buffer, dim=1),                 # (M, N+1, D)
            V_buffer[0],                                  # V at t0
            torch.stack(action_buffer, dim=1) if action_buffer else None,
            torch.stack(mu_logsigma_buffer, dim=1) if mu_logsigma_buffer else None,
            Y_error,
            total_loss_critic
        )

    def train_loop(self, itr, traj_only: bool = False, _control_exact_=False, predict: bool = False):
        if predict and not traj_only:
            raise ValueError("Evaluation batches must use traj_only=True")
        if _control_exact_ and not traj_only:
            raise ValueError("Reference-policy rollouts are evaluation-only")
        evaluate_metrics = bool(traj_only or predict)
        if evaluate_metrics:
            self._ensure_reference_solution()
        Y_error = torch.tensor(float("nan"), device=self.device)
        control_error = None
        if not predict:
            t, tau_sampled, W, B, P, P0 = self.fetch_minibatch()
        else:
            t, tau_sampled, W, B, P, P0 = (
                self.t_predict_set,
                self.tau_predict_set,
                self.W_predict_set,
                self.B_predict_set,
                self.P_predict_set,
                self.P0_predict_set
            )

        device = self.device
        dt = self.dt
        beta_disc    = float(self.params_equ["rho"])
        gamma_entropy= self.params_train.get("entropy_alpha", 0.0)
        gamma_soft   = self.params_train.get("gamma_soft", 0.0)
        p_tsallis    = self.params_train.get("p_tsallis", 1.0)

        M, N_plus_1, _ = t.shape
        D = self.D
        assert N_plus_1 == self.N + 1
        mu        = torch.as_tensor(self.params_equ["mu"],    device=device).view(1, 1, D)
        nu        = torch.as_tensor(self.params_equ["nu"],    device=device).view(1, 1, D)
        sigma     = torch.as_tensor(self.params_equ["sigma"], device=device).view(1, 1, D)
        alpha     = torch.as_tensor(self.params_equ["alpha"], device=device).view(1, 1, D)
        beta_jump = torch.as_tensor(self.params_equ["beta"],  device=device).view(1, 1, D)

        for i in range(D):
            self.dict[f'optimizer_actor_{i}'].zero_grad(set_to_none=True)
            self.dict[f'optimizer_critic_{i}'].zero_grad(set_to_none=True)

        X_buffer, V_buffer = [], []
        action_buffer, mu_logsigma_buffer = [], []
        logpi_buffer, tsallis_buffer = [], []
        mu_buffer = [] 

        total_loss_actor  = 0.0
        total_loss_critic = 0.0

        K_actor  = self.params_train.get("K_actor", 20)
        K_critic = self.params_train.get("K_critic", 5)
        actor_loss_accum = critic_loss_accum = 0.0
        critic_chunk_count = 0
        actor_loss_chunk = [None for _ in range(D)]  
        actor_step_count = 0

        t0 = t[:, 0, :]                             # (M,1)
        X0 = self.Xi.to(device).view(1, D).expand(M, D).contiguous()
        X0 = X0.detach().clone().requires_grad_(True)
        X_buffer.append(X0.detach())

        for k in range(self.N):
            t1 = t[:, k+1, :]                   # (M,1)
            W0 = W[:, k, :] 
            W1 = W[:, k+1, :]
            dW = W[:, k+1, :] - W[:, k, :]      # (M,D)
            B0 = B[:, k, :]
            B1 = B[:, k+1, :]
            dB = B[:, k+1, :] - B[:, k, :]      # (M,1)

            N_idio = count_jumps(P,  t0, t1) - torch.tensor(self.params_equ["lambda"]).unsqueeze(0).to(self.device) * dt    # (M,D)
            N0     = count_jumps(P0, t0, t1) - torch.tensor(self.params_equ["lambda0"]).unsqueeze(0).to(self.device) * dt   # (M,1)

            state = torch.cat([t0, X0], dim=-1)  # (M,1+D)
            # eps_k = torch.randn_like(X0[:, 0]).unsqueeze(-1)

            if not _control_exact_:
                mu_list, log_std_list = [], []
                u_list, logp_list, H_T_list = [], [], []
                for i in range(D):
                    net_a = self.dict[f'net_actor_{i}']
                    mu_i, log_std_i = net_a(state)          # (M,1)
                    # log_std_i = self._clamp_log_sigma(log_std_i, itr)
                    mu_i = mu_i.view(M,1)
                    log_std_i = log_std_i.view(M,1)
                    std_i = torch.exp(log_std_i)
                    eps_i = torch.randn_like(std_i)
                    u_i = mu_i + std_i * eps_i

                    dist_i = torch.distributions.Normal(mu_i, std_i)
                    # logp_i = dist_i.log_prob(u_i)
                    logp_i = dist_i.log_prob(u_i.detach()).sum(dim=-1, keepdim=True)
                    if p_tsallis != 1:

                        z_i = torch.exp(logp_i)
                        H_T_i = (1.0 / (p_tsallis - 1.0)) * (1.0 - z_i**(p_tsallis - 1.0))
                    else:

                        H_T_i = -logp_i

                    mu_list.append(mu_i)
                    log_std_list.append(log_std_i)
                    u_list.append(u_i)
                    logp_list.append(logp_i)
                    H_T_list.append(H_T_i)

                mu_all        = torch.cat(mu_list, dim=1)          # (M,D)
                log_std_all   = torch.cat(log_std_list, dim=1)     # (M,D)
                u_all         = torch.cat(u_list, dim=1)           # (M,D)
                logp_all      = torch.cat(logp_list, dim=1)        # (M,D)
                H_T_all       = torch.cat(H_T_list, dim=1)         # (M,D)

                action_buffer.append(u_all.detach())
                mu_logsigma_buffer.append(torch.cat([mu_all, log_std_all], dim=1).detach())
                logpi_buffer.append(logp_all.detach())
                tsallis_buffer.append(H_T_all.detach())
                mu_buffer.append(mu_all.detach()) 

                drift       = u_all * mu.squeeze(0)    * dt        # (M,D)
                diff_idio   = u_all * nu.squeeze(0)    * dW        # (M,D)
                diff_common = u_all * sigma.squeeze(0) * dB        # (M,D)
                jump_idio   = u_all * alpha.squeeze(0) * N_idio    # (M,D)
                jump_common = u_all * beta_jump.squeeze(0) * N0    # (M,D)

            else:
                u_all = self.control_exact.unsqueeze(0)            # (1,D) -> broadcast
                drift       = u_all * mu.squeeze(0)    * dt
                diff_idio   = u_all * nu.squeeze(0)    * dW
                diff_common = u_all * sigma.squeeze(0) * dB
                jump_idio   = u_all * alpha.squeeze(0) * N_idio
                jump_common = u_all * beta_jump.squeeze(0) * N0

            X1 = X0 + drift + diff_idio + diff_common + jump_idio + jump_common

            with torch.no_grad():
                reward = self.reward_func(X0, t0, dt)        # CARA utility
                disc = torch.exp(torch.tensor(-beta_disc * dt, device=device))

            V0_players = []
            V1_players = []
            for i in range(D):
                net_c = self.dict[f'net_critic_{i}']
                V0_i = net_c(state)                     # (M,1)
                state1 = torch.cat([t1, X1], dim=-1)
                V1_i = net_c(state1)                   # (M,1)

                # V1_i = (1.0 - gamma_soft) * V1_i + gamma_soft * V0_i.detach()
                V0_players.append(V0_i.squeeze(-1))
                V1_players.append(V1_i.squeeze(-1))
            V0_players = torch.stack(V0_players, dim=-1)   # (M,D)
            V1_players = torch.stack(V1_players, dim=-1)   # (M,D)

            V_buffer.append(V1_players.detach())
            X_buffer.append(X1.detach())

            if not traj_only:
                td_target = reward * dt + disc * V1_players.detach()
                if (gamma_entropy != 0.0) and (not _control_exact_):
                    H_T_all_for_critic = torch.cat([h.detach() for h in H_T_list], dim=1)  # (M,D)
                    td_target = td_target + gamma_entropy * H_T_all_for_critic * dt

                delta = td_target - V0_players          # (M,D)
                loss_critic_step = (delta**2).mean()

                critic_loss_accum += loss_critic_step
                critic_chunk_count += 1

                if ((k + 1) % K_critic == 0) or (k == self.N - 1):
                    loss_critic_chunk = critic_loss_accum / float(critic_chunk_count)
                    for i in range(D):
                        self.dict[f'optimizer_critic_{i}'].zero_grad(set_to_none=True)
                    loss_critic_chunk.backward()
                    for i in range(D):
                        nn.utils.clip_grad_norm_(self.dict[f'net_critic_{i}'].parameters(), 1.0)
                        self.dict[f'optimizer_critic_{i}'].step()
                        self.dict[f'scheduler_critic_{i}'].step()
                    total_loss_critic += loss_critic_chunk.detach()
                    critic_loss_accum = 0.0
                    critic_chunk_count = 0

                # ========= Actor：per-player PG =========
                advantage = (td_target - V0_players) / (self.dt + 1e-10)     # (M,D)

                for i in range(D):
                    adv_i  = advantage[:, i]  # (M,)
                    logp_i = logp_list[i].squeeze(-1)     # (M,)
                    H_i    = H_T_list[i].squeeze(-1)      # (M,)

                    # Enable advantage normalization to reduce variance
                    adv_mean = adv_i.mean()
                    adv_std = adv_i.std()
                    # if adv_std > 1e-8:
                    #     adv_i = (adv_i - adv_mean) / (adv_std + 1e-8)
                    # else:
                    #     # If all advantages are the same, just subtract mean
                    #     adv_i = adv_i - adv_mean

                    # Policy gradient loss
                    loss_i_pg = -(logp_i * adv_i.detach())
                    if gamma_entropy != 0.0:
                        loss_i_pg = loss_i_pg + gamma_entropy * H_i

                    loss_i = loss_i_pg.mean()

                    if actor_loss_chunk[i] is None:
                        actor_loss_chunk[i] = loss_i
                    else:
                        actor_loss_chunk[i] = actor_loss_chunk[i] + loss_i

                actor_step_count += 1

                if ((k + 1) % K_actor == 0) or (k == self.N - 1):
                    for i in range(D):
                        if actor_loss_chunk[i] is None:
                            continue

                        loss_to_backward_i = actor_loss_chunk[i] / float(actor_step_count)
                        # FIXED: Accumulate the averaged loss for reporting
                        total_loss_actor += loss_to_backward_i.detach()

                        opt_a = self.dict[f'optimizer_actor_{i}']
                        sch_a = self.dict[f'scheduler_actor_{i}']
                        net_a = self.dict[f'net_actor_{i}']

                        opt_a.zero_grad(set_to_none=True)

                        retain = (i < D - 1)
                        loss_to_backward_i.backward(retain_graph=retain)
                        # Adaptive gradient clipping: use larger threshold when advantage is normalized
                        grad_clip_threshold = self.params_train.get("grad_clip_actor", 1.0)
                        g = torch.nn.utils.clip_grad_norm_(net_a.parameters(), grad_clip_threshold)
                        # print("g=======", g) #when adv_i is normalized, abs of g becomes bigger, which is we want.
                        opt_a.step()
                        sch_a.step()
                        actor_loss_chunk[i] = None

                    actor_step_count = 0

            X0 = X1.detach().clone().requires_grad_(True)
            t0, W0, B0 = t1, W1, B1

        if evaluate_metrics:
            with torch.no_grad():

                if len(action_buffer) > 0:
                    actions_all = torch.stack(action_buffer, dim=1)  # (M, N, D)
                    mu_all      = torch.stack(mu_buffer,      dim=1)  # (M,N,D)
                    u_exact     = self.control_exact.view(1,1,D)
                    ctrl_err_sample = ((actions_all - u_exact)**2).mean() / (u_exact**2 + 1e-12).mean()

                    ctrl_err_mu = ((mu_all - u_exact)**2).mean() / (u_exact**2 + 1e-12).mean()
                    control_error = ctrl_err_mu

                    # control_error = (torch.sqrt((actions_all - u_exact.unsqueeze(0).unsqueeze(0))**2).mean(dim=0)).mean()
                else:
                    control_error = None

                X_traj = torch.stack(X_buffer, dim=1)      # (M, N+1, D)
                t_traj = t.to(device)

                V_exact = (self.exact_value if gamma_entropy == 0.0 else self.exact_value_soft)(t_traj, X_traj)   # (M, N+1, D)
                V_num   = torch.stack(V_buffer, dim=1)     # (M, N, D)
                if k == (self.N-1):
                    print("V_num.mean", V_num.mean())
                    print("V_exact[:, 1:, :].mean", V_exact[:, 1:, :].mean())
                if k == self.N -1:
                    # print("u[0]_pred", actions_all[:,:,0].mean())
                    print("u[0]_pred", mu_all[:,:,0].mean())
                    print("u_exact[0]", u_exact[0][0][0])
                    # print("u[-1]_pred", actions_all[:,:,-1].mean())
                    print("u[-1]_pred", mu_all[:,:,-1].mean())
                    print("u_exact[-1]", u_exact[0][0][-1])

                Y_error = ((V_num - V_exact[:, 1:, :])**2).mean() / (V_exact[:, 1:, :]**2 + 1e-12).mean()

        if not traj_only:
            # Divide by D to get average actor loss per agent (for comparison with critic loss)
            self.ActorLoss_list.append(total_loss_actor / D)
            self.CriticLoss_list.append(total_loss_critic.detach().cpu())
            self.control_error_list.append(
                control_error.detach().cpu() if control_error is not None else torch.tensor(float("nan"))
            )
            self.Y_error_list.append(Y_error.detach().cpu())

            if mu_logsigma_buffer and len(mu_logsigma_buffer) > 0:
                mu_logsigma_all = torch.stack(mu_logsigma_buffer, dim=1)  # (M, N, 2*D)

                mu_player0 = mu_logsigma_all[:, :, 0]  # (M, N)
                log_std_player0 = mu_logsigma_all[:, :, D]  # (M, N)
                std_player0 = torch.exp(log_std_player0)  # (M, N)

                mean_over_trajs = mu_player0.mean(dim=0)  # (N,)
                std_over_trajs = std_player0.mean(dim=0)  # (N,)

                mean_final = mean_over_trajs.mean().detach().cpu()
                std_final = std_over_trajs.mean().detach().cpu()

                self.gaussian_mean_list.append(mean_final)
                self.gaussian_std_list.append(std_final)

                self._save_gaussian_mean_std()

            total_loss_actor = total_loss_actor / self.N
            total_loss_critic = total_loss_critic / self.N

        return (
            torch.stack(X_buffer, dim=1),                 # (M, N+1, D)
            torch.stack(V_buffer, dim=1),                 # (M, N+1, D)
            V_buffer[0],                                  # V at t0
            torch.stack(action_buffer, dim=1) if action_buffer else None,
            torch.stack(mu_logsigma_buffer, dim=1) if mu_logsigma_buffer else None,
            Y_error,
            control_error,
            total_loss_critic,
            total_loss_actor
        )

    def train_all(self, n: int = 0, N_Iter: int = 10):
        start_time = time.time()
        n_precritic = n

        self._read_gaussian_mean_std()

        for it_pre in range(n_precritic):
            X_pre, Y_pre, Y0_pre, _, _, _, loss_critic_pre = self.train_onlycritic(itr=it_pre, traj_only=False, predict=False)

        # Learning rate warmup
        warmup_steps = self.params_train.get("warmup_steps", 100)
        initial_lr_actor = self.params_train["lr_actor"]
        initial_lr_critic = self.params_train["lr_critic"]

        for it in range(N_Iter):
            # Learning rate warmup for both actor and critic
            if it < warmup_steps:
                warmup_factor = (it + 1) / warmup_steps
                for i in range(self.D):
                    for param_group in self.dict[f'optimizer_actor_{i}'].param_groups:
                        param_group['lr'] = initial_lr_actor * warmup_factor
                    for param_group in self.dict[f'optimizer_critic_{i}'].param_groups:
                        param_group['lr'] = initial_lr_critic * warmup_factor

            # Improved critic pre-training: more frequent and longer duration
            # This helps ensure critic is well-trained before actor learning
            critic_pretrain_duration = self.params_train.get("critic_pretrain_duration", 2000)
            critic_pretrain_freq = self.params_train.get("critic_pretrain_freq", 10)
            if it < critic_pretrain_duration and it % critic_pretrain_freq == 0:
                X_pre, Y_pre, Y0_pre, _, _, _, loss_critic_pre = self.train_onlycritic(itr=it, traj_only=False, predict=False)

            X_pred, V_pred, V0, action_buf, mu_logsigma_buf, \
            Y_err, ctrl_err, loss_c, loss_a = self.train_loop(it)

            elapsed = time.time() - start_time

            if it % 2 == 0:
                msg = (
                    f"It:{it:4d}  Time:{elapsed:7.2f}  "
                    f"Lc:{loss_c:.3e}  La:{loss_a:.3e}"
                )
                print(msg)

            if (it + 1) % self.params_train["eval_every"] == 0:
                score, detail = self._evaluate_on_fixed()
                self.Y_error_list[-1] = torch.tensor(detail["Y_error"])
                self.control_error_list[-1] = torch.tensor(detail["control_error"] if detail["control_error"] is not None else float("nan"))
                print(
                    f"[eval] it={it}  score={score:.6g}  "
                    f"(Y_err={detail['Y_error']:.6g}, "
                    f"ctrl_err={detail['control_error'] if detail['control_error'] is not None else 'NA'})"
                )
                self._maybe_update_top3_eval(it=it, score=score)

        print("The training is over!")
        self.save_and_read_list(type="save")

        self._save_params(self.output_folder)
        if len(self.topk_eval) > 0:
            best = sorted(self.topk_eval, key=lambda d: d["score"])[0]
            print(f"[summary] eval best: iter={best['iter']}, score={best['score']:.6g}")
        else:
            print("[summary] no eval Top-3 recorded.")

        self.plot_gaussian_mean_std()

    ################
    ## Plot
    ################
    def plot_list(self, list_, ylabel, legend, color=None, title=None, ylabel_log=False, xticks=None, baseline=None, filename=None):
        return plot_training_series(
            list_, ylabel, legend, output_folder=self.output_folder,
            iterations=self.params_train["iteration"], color=color, title=title,
            ylabel_log=ylabel_log, xticks=xticks, baseline=baseline, filename=filename,
        )

    def plot_controlfunction(self, player_idx=0, n=201, x_region=(-2, 2),
                         filename=None, sample_exact=False):
        """Evaluate player_idx's relative control error against the analytical u_i*.
        The actor input is [t,x_vec]; vary x_i while fixing all other state components."""
        self._ensure_reference_solution()
        device = self.device
        i = int(player_idx)
        assert 0 <= i < self.D, "player_idx out of range"

        t = torch.linspace(0.0, float(self.T), n, device=device).view(n, 1)     # (n,1)
        x_i = torch.linspace(x_region[0], x_region[1], n, device=device).view(1, n)  # (1,n)

        base_x = self.Xi.to(device).view(1, self.D).clone()
        X = base_x.repeat(n, n, 1)              # (n,n,D)
        X[:, :, i] = x_i.repeat(n, 1)           

        T_grid = t.repeat(1, n)                 # (n,n)
        Z = torch.cat([
            T_grid.unsqueeze(-1),
            X.view(n, n, self.D)
        ], dim=-1)                              # (n,n,1+D)

        net_a_i = self.dict[f'net_actor_{i}']
        with torch.no_grad():
            mu_i, log_std_i = net_a_i(Z.view(-1, 1 + self.D))
            mu_i = mu_i.view(n, n)

            u_hat = mu_i

        u_star_i = self.control_exact[i].to(device)
        if sample_exact:

            u_star = u_star_i + 0.0 * torch.randn_like(u_hat)
        else:
            u_star = torch.full_like(u_hat, float(u_star_i))

        denom = torch.clamp(torch.abs(u_star), min=1e-12)
        rel_err = (u_hat - u_star) / denom
        rel_err = rel_err.abs().detach().cpu().numpy()

        return plot_control_error_heatmap(
            rel_err, player_idx=i, x_region=x_region, horizon=float(self.T),
            output_folder=self.output_folder, filename=filename,
        )

    def _load_eval_topk_meta(self):
        """Read multi-player Top-k metadata:
        [{"iter": int, "score": float, "actor": [player paths], "critic": [player paths]}].
        If metadata is absent, attempt recovery from checkpoint filename patterns."""
        meta_path = os.path.join(self.output_folder, "eval_topk_meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    li = metadata_from_disk(json.load(f), os.path.dirname(f.name))
                cleaned = []
                for d in li:
                    if all(k in d for k in ["iter", "score", "actor", "critic"]):

                        actor = d["actor"]
                        critic = d["critic"]
                        if isinstance(actor, str):
                            actor = [actor]
                        if isinstance(critic, str):
                            critic = [critic]
                        cleaned.append({
                            "iter": int(d["iter"]),
                            "score": float(d["score"]),
                            "actor": actor,
                            "critic": critic
                        })
                cleaned.sort(key=lambda d: d["score"])
                return cleaned
            except Exception as e:
                print(f"[warn] read meta failed: {e}")

        cand = []
        pattern = os.path.join(self.output_folder, "actor_player*_eval_top*_itr*.pth")
        for a in glob.glob(pattern):
            base = os.path.basename(a)
            try:
                itr = int(base.split("_itr")[-1].split(".pth")[0])
            except Exception:
                itr = -1

            cand.append({
                "iter": itr,
                "score": float("inf"),
                "actor": [a],
                "critic": []
            })
        cand.sort(key=lambda d: (d["score"], d["iter"]))
        return cand[:3]

    def _load_ckpt_set(self, actor_paths, critic_paths, ckpt_root=None, loaded_player_indices=None):
        """Load the supplied per-player actor/critic checkpoint paths and corresponding params.json.

        Args:
            actor_paths: actor checkpoint paths.
            critic_paths: critic checkpoint paths.
            ckpt_root: preferred directory for params.json.
            loaded_player_indices: selected player indices, or all D players when None."""
        if loaded_player_indices is None:

            try:
                assert len(actor_paths) == self.D and len(critic_paths) == self.D
            except AssertionError:
                raise ValueError("A complete actor/critic checkpoint set is required for all players")
            loaded_player_indices = list(range(self.D))
        else:

            if len(actor_paths) != len(loaded_player_indices) or len(critic_paths) != len(loaded_player_indices):
                raise ValueError("Checkpoint path counts do not match the selected players")

        try:
            for idx, player_i in enumerate(loaded_player_indices):
                sd_a = torch.load(actor_paths[idx], map_location=self.device)
                sd_c = torch.load(critic_paths[idx], map_location=self.device)
                self.dict[f'net_actor_{player_i}'].load_state_dict(sd_a)
                self.dict[f'net_critic_{player_i}'].load_state_dict(sd_c)
                self.dict[f'net_actor_{player_i}'].eval()
                self.dict[f'net_critic_{player_i}'].eval()
            print(f"[load] loaded {len(loaded_player_indices)} players' ckpt set:")
            for idx, player_i in enumerate(loaded_player_indices):
                print(f"  actor_player{player_i} = {actor_paths[idx]}")
                print(f"  critic_player{player_i} = {critic_paths[idx]}")

            loaded_params = None
            if ckpt_root is not None and os.path.isdir(ckpt_root):
                loaded_params = self._load_params(ckpt_root)

            if loaded_params is None and len(actor_paths) > 0:
                ckpt_dir = os.path.dirname(actor_paths[0])
                loaded_params = self._load_params(ckpt_dir)

            if loaded_params is not None:
                self._update_params_from_dict(loaded_params)

            return True
        except Exception as e:
            raise RuntimeError("Failed to load the multi-player actor/critic checkpoint set") from e

    @torch.no_grad()
    def _plot_trajs_for_current_weights(self,
                                        filename_suffix: str = "",
                                        device=None,
                                        downsample_step=1,
                                        smooth_window=1,
                                        players_to_plot=None):
        """
            dX_t^i = π_i(μ_i dt + ν_i dW_t^i + σ_i dB_t + α_i dM_t^i + β_i dM_t^0).
        """
        self._ensure_reference_solution()
        device = device or self.device
        self._ensure_eval_set()
        t_test  = self.t_eval.to(device)     # (M, N+1, 1)
        W_test  = self.W_eval.to(device)     # (M, N+1, D)
        B_test  = self.B_eval.to(device)     # (M, N+1, 1)
        P_test  = self.P_eval.to(device)     
        P0_test = self.P0_eval.to(device)    # (M, J0, 1)

        M, N_plus_1, _ = t_test.shape
        D = self.D
        N = N_plus_1 - 1
        dt = self.dt
        mu        = torch.as_tensor(self.params_equ["mu"],    device=device).view(1, 1, D)
        nu        = torch.as_tensor(self.params_equ["nu"],    device=device).view(1, 1, D)
        sigma     = torch.as_tensor(self.params_equ["sigma"], device=device).view(1, 1, D)
        alpha     = torch.as_tensor(self.params_equ["alpha"], device=device).view(1, 1, D)
        beta_jump = torch.as_tensor(self.params_equ["beta"],  device=device).view(1, 1, D)
        gamma_entropy = torch.as_tensor(self.params_train["entropy_alpha"],  device=device)

        X0 = self.Xi.to(device).view(1, D).expand(M, D).contiguous()

        X_learn = [X0]
        U_learn = []
        V_learn = []
        if gamma_entropy == 0:
            u_star = self.control_exact.to(device).view(1, D)   # (1,D)
        else:
            u_star = self.soft_mean_control.to(device)

        X_exact = [X0]
        U_exact = []

        def count_jumps(P_times, t0, t1):
            """Count events in (t0,t1] for each path and jump channel.
            P_times: jump times, (M,J,d) or (M,N+1,d).
            t0, t1: (M,1).
            Return shape (M,d)."""
            while P_times.dim() < 3:
                P_times = P_times.unsqueeze(-1)
            # (M, J, d)
            t0e = t0.unsqueeze(-1)   # (M,1,1)
            t1e = t1.unsqueeze(-1)
            mask = (P_times > t0e) & (P_times <= t1e)
            return mask.sum(dim=1).to(torch.float32)  # (M,d)

        # ================== rollout ==================
        for k in range(N):
            t0 = t_test[:, k, :].contiguous()      # (M,1)
            t1 = t_test[:, k+1, :].contiguous()    # (M,1)

            dW = W_test[:, k+1, :] - W_test[:, k, :]    # (M,D)
            dB = B_test[:, k+1, :] - B_test[:, k, :]    # (M,1)

            N_idio = count_jumps(P_test,  t0, t1) - torch.tensor(self.params_equ["lambda"]).unsqueeze(0).to(self.device) * (t1-t0)    # (M,D)
            N0     = count_jumps(P0_test, t0, t1) - torch.tensor(self.params_equ["lambda0"]).unsqueeze(0).to(self.device) * (t1-t0)    # (M,1)

            # ----- learned -----
            X_curr = X_learn[-1]                        # (M,D)
            state  = torch.cat([t0, X_curr], dim=-1)    # (M,1+D)

            u_list = []
            v_list = []
            for i in range(D):

                if players_to_plot is not None and i not in players_to_plot:

                    net_a = self.dict[f"net_actor_{i}"]
                    net_c = self.dict[f"net_critic_{i}"]

                else:
                    net_a = self.dict[f"net_actor_{i}"]
                    net_c = self.dict[f"net_critic_{i}"]
                mu_i, log_std_i = net_a(state)
                # log_std_i = self._clamp_log_sigma(log_std_i)
                u_i = mu_i.view(M, 1)                   
                v_i = net_c(state).view(M, 1)
                u_list.append(u_i)
                v_list.append(v_i)

            u_all = torch.cat(u_list, dim=1)            # (M,D)
            v_all = torch.cat(v_list, dim=1)            # (M,D)

            drift_l       = u_all * mu.squeeze(0)    * dt      # (M,D)
            diff_idio_l   = u_all * nu.squeeze(0)    * dW      # (M,D)
            diff_common_l = u_all * sigma.squeeze(0) * dB      # (M,D)
            jump_idio_l   = u_all * alpha.squeeze(0) * N_idio  # (M,D)
            jump_common_l = u_all * beta_jump.squeeze(0) * N0  # (M,D)

            X_next_l = X_curr + drift_l + diff_idio_l + diff_common_l + jump_idio_l + jump_common_l

            X_learn.append(X_next_l)
            U_learn.append(u_all)
            V_learn.append(v_all)

            Xc = X_exact[-1]
            u_exact = u_star.expand(M, D)

            drift_e       = u_exact * mu.squeeze(0)    * dt
            diff_idio_e   = u_exact * nu.squeeze(0)    * dW
            diff_common_e = u_exact * sigma.squeeze(0) * dB
            jump_idio_e   = u_exact * alpha.squeeze(0) * N_idio
            jump_common_e = u_exact * beta_jump.squeeze(0) * N0

            X_next_e = Xc + drift_e + diff_idio_e + diff_common_e + jump_idio_e + jump_common_e

            X_exact.append(X_next_e)
            U_exact.append(u_exact)

        # -------- stack --------
        X_pred       = torch.stack(X_learn, dim=1)             # (M,N+1,D)
        X_test       = torch.stack(X_exact, dim=1)             # (M,N+1,D)
        control_pred = torch.stack(U_learn, dim=1)             # (M,N,D)
        control_test = torch.stack(U_exact, dim=1)             # (M,N,D)
        V_pred       = torch.stack(V_learn, dim=1)             # (M,N,D)
        V_pred       = torch.cat([V_pred, V_pred[:, -1:, :]], dim=1)  # -> (M,N+1,D)

        if gamma_entropy == 0:
            V_test = self.exact_value(t_test, X_pred)              # (M,N+1,D)
        else:
            V_test = self.exact_value_soft(t_test, X_pred)

        # P_test: (M, J_idio, D), P0_test: (M, J0, 1)

        return plot_multi_cara_trajectories(
            t_test, V_pred, V_test, X_pred, X_test, control_pred, control_test, P_test, P0_test,
            dimensions=D, entropy_alpha=self.params_train.get("entropy_alpha", 0.0),
            output_folder=self.output_folder, jump_index=self.jump_index,
            players_to_plot=players_to_plot, filename_suffix=filename_suffix, device=device,
            downsample_step=downsample_step, smooth_window=smooth_window,
            CriticLoss=getattr(self, 'CriticLoss_list', None),
            ActorLoss=getattr(self, 'ActorLoss_list', None),
        )

    def plot_combined_errors_and_loss(self, Y_errors, ctrl_err_pad, time_points, 
                                       CriticLoss, ActorLoss, filename_suffix=""):
        return plot_multi_agent_errors_and_loss(
            Y_errors, ctrl_err_pad, time_points, CriticLoss, ActorLoss,
            dimensions=self.D, output_folder=self.output_folder, filename_suffix=filename_suffix,
        )

    def plot_loss(self, CriticLoss, ActorLoss, filename):
        return plot_actor_critic_loss(CriticLoss, ActorLoss, filename, output_folder=self.output_folder)

    def plot_gaussian_mean_std(self, filename="gaussian_mean_std"):
        return plot_gaussian_mean_std(
            getattr(self, 'gaussian_mean_list', None), getattr(self, 'gaussian_std_list', None),
            output_folder=self.output_folder, filename=filename,
        )

    def plot_topk_eval(self, k: int = 3, device=None):
        """Load each Top-k multi-player checkpoint and plot its trajectories on the fixed evaluation batch."""
        device = device or self.device
        topk = self._load_eval_topk_meta()
        if not topk:
            print("[info] no eval Top-k found; train first.")
            return

        self._ensure_eval_set()

        for rank, rec in enumerate(sorted(topk, key=lambda d: d["score"])[:k], start=1):
            actor_paths = rec["actor"]
            critic_paths = rec["critic"]
            if isinstance(actor_paths, str):
                actor_paths = [actor_paths]
            if isinstance(critic_paths, str):
                critic_paths = [critic_paths]

            ok = self._load_ckpt_set(actor_paths, critic_paths, ckpt_root=self.output_folder)
            if not ok:
                continue

            suffix = f"_top{rank}_itr{int(rec['iter'])}_score{rec['score']:.4f}"
            print(f"[plot] begin {suffix}")
            self._plot_trajs_for_current_weights(filename_suffix=suffix, device=device,
                                                downsample_step=10, smooth_window=1)
            print(f"[plot] done  {suffix}")

    def _default_output_folder(self):
        return training_output_folder(
            "Multi-CARA_fixed_std", self.params_equ, self.params_train, root=_ROOT / "output",
        )

    def save_and_read_list(self, type="save", from_folder=None):
        """type: "save" or "read".
        from_folder: directory to read from; defaults to self.output_folder."""
        name = ["Y_error.json", "control_error.json", "CriticLoss.json", "ActorLoss.json"]
        save_list = [self.Y_error_list, self.control_error_list, self.CriticLoss_list, self.ActorLoss_list]
        folder = from_folder if from_folder is not None else self.output_folder

        if type == "save":
            os.makedirs(self.output_folder, exist_ok=True)
            for i in range(len(name)):
                with open(os.path.join(self.output_folder, name[i]), 'w') as file:
                    json.dump(metadata_for_save([tensor.tolist() for tensor in save_list[i]], os.path.dirname(file.name)), file)
        elif type == "read":
            for i in range(len(name)):
                file_path = os.path.join(folder, name[i])
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        save_list[i][:] = [torch.tensor(j) for j in metadata_from_disk(json.load(file), os.path.dirname(file.name))]
                else:

                    print(f"[info] {name[i]} not found in {folder}, starting with empty list")

            self._read_gaussian_mean_std(from_folder=folder)
        else:
            raise RuntimeError("You can only save or read the model!")

    def _save_gaussian_mean_std(self):
        """Persist Gaussian-network means and standard deviations to JSON during training."""
        if not hasattr(self, 'gaussian_mean_list') or not hasattr(self, 'gaussian_std_list'):
            return

        os.makedirs(self.output_folder, exist_ok=True)
        file_path = os.path.join(self.output_folder, "gaussian_net_mean_std.json")

        data = {
            "gaussian_mean": [float(x) for x in self.gaussian_mean_list],
            "gaussian_std": [float(x) for x in self.gaussian_std_list]
        }

        with open(file_path, 'w') as f:
            json.dump(metadata_for_save(data, os.path.dirname(f.name)), f, indent=2, ensure_ascii=False)

    def _read_gaussian_mean_std(self, from_folder=None):
        """Load Gaussian-network means and standard deviations from JSON.
        from_folder defaults to self.output_folder."""
        folder = from_folder if from_folder is not None else self.output_folder
        file_path = os.path.join(folder, "gaussian_net_mean_std.json")

        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = metadata_from_disk(json.load(f), os.path.dirname(f.name))
                if "gaussian_mean" in data and "gaussian_std" in data:
                    self.gaussian_mean_list[:] = [torch.tensor(x) for x in data["gaussian_mean"]]
                    self.gaussian_std_list[:] = [torch.tensor(x) for x in data["gaussian_std"]]
                    print(f"[info] Loaded gaussian mean/std from {file_path}, found {len(self.gaussian_mean_list)} entries")
            except Exception as e:
                print(f"[warn] Failed to load gaussian mean/std from {file_path}: {e}")
        else:
            print(f"[info] gaussian_net_mean_std.json not found in {folder}, starting with empty list")

    # def predict(self, t, tau, W, B, P, P0, X0):
    #     self.save_and_read_model(type="read")
    #     print("Y error: %.3f%%, control error: %.3f%%" % (self.Y_error_list[-1].numpy() * 100, self.control_error_list[-1].numpy() * 100))
    #     self.plot_list([self.Y_error_list, self.control_error_list], "Error_value/Error_control", ["value function", "control"],
    #                    color=["#E7483D", "#6565AE", "#8FC751"], title=None, ylabel_log=True, baseline=0.01, filename="game_train")
    #     net_critic = [self.dict[f'net_critic_{d}'] for d in range(D)]
    #     net_actor = [self.dict[f'net_actor_{d}'] for d in range(D)]
    #     _, _, X_star, Y_star, _, control_star= self.loss_function(net_critic, net_actor, t, tau, W, B, P, P0, X0, i=0, n=self.N, k=0, traj_only=True)
    #     _, _, X_test, _, _, _ = self.loss_function(net_critic, net_actor, t, tau, W, B, P, P0, X0, i=0, n=self.N, k=0, traj_only=True, control_exact=True)
    #     control_pred_mu_logsigma = torch.cat([self.dict[f'net_actor_{i}'](torch.cat([t, X_pred], dim=2)) for i in range(self.D)], dim=2)
    #     control_pred_mu = control_pred_mu_logsigma[:, :, 0]
    #     control_pred_logsigma = control_pred_mu_logsigma[:, :, 1]
    #     control_pred_sigma = torch.exp(control_pred_logsigma)
    #     eps = torch.randn_like(control_pred_mu)
    #     control_pred = control_pred_mu + control_pred_sigma * eps
    #     return X_star, Y_star, control_pred, control_star, X_test

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)
    # device = torch.device("cuda:1")
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', type=str, default=None, help='path to actor.pth')
    parser.add_argument('--critic_path', type=str, default=None, help='path to critic.pth')
    parser.add_argument('--resume', action='store_true',
                    help='Resume training from the given actor/critic checkpoints')
    parser.add_argument(
        '--fixed_std',
        type=float,
        default=0.1,
        help='When entropy_alpha==0, use this fixed Gaussian std (no learned log_std). Ignored if entropy_alpha>0.',
    )

    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run directory for fresh training")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for a fresh training run")

    args = parser.parse_args()
    validate_checkpoint_args(parser, args, multiplayer=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    parameters = {"axes.labelsize": 20, "axes.titlesize": 20,
                  "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}
    plt.rcParams.update(parameters)
    configure_plot_style()

    loaded_params_dict = None
    if args.actor_path is not None or args.critic_path is not None:
        ckpt_dir = None
        for p in [args.actor_path, args.critic_path]:
            if p is not None:
                if os.path.isdir(p):
                    ckpt_dir = p
                    break
                elif os.path.isfile(p):
                    ckpt_dir = os.path.dirname(p)
                    break

        if ckpt_dir is not None:
            params_path = os.path.join(ckpt_dir, "params.json")
            if os.path.isfile(params_path):
                try:
                    with open(params_path, "r") as f:
                        content = f.read()

                        if content.strip().endswith('"device":'):
                            content = content.strip()[:-10].strip()
                            if content.endswith(','):
                                content = content[:-1]
                            content += '\n  }\n}'
                        loaded_params_dict = metadata_from_disk(json.loads(content), ckpt_dir)
                    print(f"[init] Loaded params from checkpoint directory: {ckpt_dir}")
                except Exception as e:
                    print(f"[warn] Failed to load params.json from {ckpt_dir}: {e}")

    if loaded_params_dict is not None:
        D = loaded_params_dict["equation"]["D"]
        print(f"[init] Using D={D} from checkpoint params")
    else:
        D = 25
            ## original
            # "mu": [0.05]+[0.04]*(D-1), "nu": [0.2]+[0.1]*(D-1), "sigma": [0.35]+[0.25]*(D-1), "alpha": [0.3]+[0.2]*(D-1),
            # "beta": [0.25]+[0.15]*(D-1), "delta": [2]+[1.0]*(D-1), "theta": [0.5]+[0.2]*(D-1), "lambda": [0.3]+[0.2]*(D-1),
            # "lambda0": 0.25, "D": D, "T": 5, "M": 100, "N": 1000, "rho":1.0,
            # "Xi": torch.tensor([10.]+[9.]*(D-1)).unsqueeze(0)
            ## reasonable 1
            # "mu": [0.03]+[0.02]*(D-1), "nu": [0.2]+[0.1]*(D-1), "sigma": [0.35]+[0.25]*(D-1), "alpha": [0.3]+[0.2]*(D-1),
            # "beta": [0.25]+[0.15]*(D-1), "delta": [2.5]+[2.0]*(D-1), "theta": [0.3]+[0.2]*(D-1), "lambda": [0.3]+[0.2]*(D-1),
            # "lambda0": 0.25, "D": D, "T": 5, "M": 100, "N": 1000, "rho":1.0,
            # "Xi": torch.tensor([5.]+[4.]*(D-1)).unsqueeze(0), 
            ## players_are_similar 1
            # "mu": [0.02]+[0.02]*(D-1), "nu": [0.1]+[0.1]*(D-1), "sigma": [0.25]+[0.25]*(D-1), "alpha": [0.2]+[0.2]*(D-1),
            # "beta": [0.15]+[0.15]*(D-1), "delta": [2.0]+[2.0]*(D-1), "theta": [0.25]+[0.2]*(D-1), "lambda": [0.2]+[0.2]*(D-1),
            # "lambda0": 0.25, "D": D, "T": 5, "M": 100, "N": 1000, "rho":1.0,
            # "Xi": torch.tensor([4.]+[4.]*(D-1)).unsqueeze(0), 

    if loaded_params_dict is not None:
        params_equation = loaded_params_dict["equation"]
        if isinstance(params_equation.get("Xi"), dict) and "_type" in params_equation["Xi"]:
            params_equation["Xi"] = torch.tensor(params_equation["Xi"]["data"])
        # if isinstance(params_equation.get("terminal"), dict) and "_type" in params_equation["terminal"]:
        #     params_equation["terminal"] = torch.tensor(params_equation["terminal"]["data"])

        default_train_params = {
            "lr_critic": 1e-3, "gamma_critic": 0.5, "milestones_critic": [1000, 150000, 200000],
            "lr_actor": 5e-4, "gamma_actor": 0.5, "milestones_actor": [10000, 40000],
            "iteration": 1000, "device": device, "method": 'PO_CRL',
            "entropy_alpha": 0.0, "eval_every": 20, "K_actor": 30, "K_critic": 10,
            "p_tsallis": 1, "gamma_soft": 0.0,
            "weight_decay_actor": 1e-5,
            "weight_decay_critic": 0.0,
            "advantage_scale": "dt",
            "warmup_steps": 100,
            "critic_pretrain_duration": 2000,
            "critic_pretrain_freq": 10,
            "grad_clip_actor": 1.0,
        }
        params_train = loaded_params_dict.get("train", {})
        for k, v in default_train_params.items():
            if k not in params_train:
                params_train[k] = v
        params_train["device"] = device  # Always use current device

        params_net_critic = loaded_params_dict.get("net_critic", {
            "inputs": D+1, "width": D+10, "depth": 2, "output": 1, "activation": "Tanh", "penalty": "Tanh",
            "params_act": {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()},
        })
        params_net_actor = loaded_params_dict.get("net_actor", {
            "inputs": D+1, "width": D+10, "depth": 2, "output": 1, "activation": "Tanh", "penalty": "ScaledTanh",
            "params_act": {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()},
        })

        def check_and_fix_params_act(params_net):
            if "params_act" not in params_net:
                return False
            params_act = params_net["params_act"]

            if not isinstance(params_act, dict):
                return False
            if len(params_act) > 0:
                first_value = next(iter(params_act.values()))
                if not isinstance(first_value, nn.Module):
                    return False
            return True

        if not check_and_fix_params_act(params_net_critic):
            params_net_critic["params_act"] = {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()}
        if not check_and_fix_params_act(params_net_actor):
            params_net_actor["params_act"] = {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()}

        params = {
            "equation": params_equation,
            "train": params_train,
            "net_critic": params_net_critic,
            "net_actor": params_net_actor,
        }
        print("[init] Using loaded params from checkpoint")
    else:
        params = {
            "equation": {
                ## players_are_similar 1
                # "mu": [0.02]+[0.02]*(D-1), "nu": [0.1]+[0.1]*(D-1), "sigma": [0.2]+[0.25]*(D-1), "alpha": [0.2]+[0.2]*(D-1),
                # "beta": [0.25]+[0.15]*(D-1), "delta": [2.0]+[2.0]*(D-1), "theta": [0.25]+[0.2]*(D-1), "lambda": [0.25]+[0.2]*(D-1),
                # "lambda0": 0.25, "D": D, "T": 5, "M": 100, "N": 1000, "rho":1.0,
                # "Xi": torch.tensor([5.]+[4.]*(D-1)).unsqueeze(0)
                # "mu": [0.05]+[0.02]*(D-1), "nu": [0.08]+[0.05]*(D-1), "sigma": [0.5]+[0.4]*(D-1), "alpha": [0.2]+[0.2]*(D-1),
                # "beta": [0.2]+[0.2]*(D-1), "delta": [1.5]+[2.0]*(D-1), "theta": [0.2]+[0.2]*(D-1), "lambda": [0.2]+[0.2]*(D-1),
                # "lambda0": 0.25, "D": D, "T": 1000, "M": 100, "N": 10000, "rho":1.0,
                # "Xi": torch.tensor([5.]+[6.]*(D-1)).unsqueeze(0)
                "mu": [0.05]+[0.02]*(D-1), "nu": [0.08]+[0.05]*(D-1), "sigma": [0.5]+[0.4]*(D-1), "alpha": [0.2]+[0.2]*(D-1),
                "beta": [0.2]+[0.2]*(D-1), "delta": [1.5]+[2.0]*(D-1), "theta": [0.2]+[0.2]*(D-1), "lambda": [0.2]+[0.2]*(D-1),
                "lambda0": 0.25, "D": D, "T": 10, "M": 100, "N": 500, "rho":1.0,
                "Xi": torch.tensor([5.]+[6.]*(D-1)).unsqueeze(0)
                # "mu": [0.02]+[0.02]*(D-1), "nu": [0.1]+[0.15]*(D-1), "sigma": [0.3]+[0.4]*(D-1), "alpha": [0.2]+[0.2]*(D-1),
                # "beta": [0.2]+[0.2]*(D-1), "delta": [2.0]+[2.0]*(D-1), "theta": [0.2]+[0.2]*(D-1), "lambda": [0.2]+[0.2]*(D-1),
                # "lambda0": 0.25, "D": D, "T": 5, "M": 100, "N": 1000, "rho":1.0,
                # "Xi": torch.tensor([5.]+[6.]*(D-1)).unsqueeze(0),
                # "mu": [0.06]+[0.05]*(D-1), "nu": [0.08]+[0.05]*(D-1), "sigma": [0.5]+[0.4]*(D-1), "alpha": [0.2]+[0.2]*(D-1),
                # "beta": [0.2]+[0.2]*(D-1), "delta": [1.5]+[1.0]*(D-1), "theta": [0.25]+[0.3]*(D-1), "lambda": [0.2]+[0.2]*(D-1),
                # "lambda0": 0.25, "D": D, "T": 5, "M": 100, "N": 500, "rho":1.0,
                # "Xi": torch.tensor([5.]+[6.]*(D-1)).unsqueeze(0)
            },
            "train": {
                "lr_critic": 1e-3, "gamma_critic": 0.5, "milestones_critic": [150000, 200000],
                "lr_actor": 1e-3, "gamma_actor": 0.5, "milestones_actor": [10000, 40000],
                "iteration": 1000, "device": device, "method": 'PO_CRL', # # Methods include: AC_Learning, PO_CRL, Inf_MFC, PO_CRL_Geomt, PG, 'SPG_Geomt_ManualGrad'
                "entropy_alpha": 0.0,  "eval_every": 20, "K_actor": 30, "K_critic": 10,
                "p_tsallis": 1, "gamma_soft": 0.0,
                # New hyperparameters for improved control learning
                "weight_decay_actor": 1e-5,  # L2 regularization for actor
                "weight_decay_critic": 0.0,  # L2 regularization for critic
                "advantage_scale": "dt",  # "dt", "none", "adaptive" - how to scale advantage
                "warmup_steps": 100,  # Learning rate warmup steps
                "critic_pretrain_duration": 500, # 2000 before # How long to pre-train critic
                "critic_pretrain_freq": 10,  # Frequency of critic pre-training
                "grad_clip_actor": 1.0,  # Gradient clipping threshold for actor
            },
            "net_critic": {
                "inputs": D+1, "width": D+10, "depth": 2, "output": 1, "activation": "Tanh", "penalty": "Tanh",
                "params_act": {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()},
            },
            "net_actor": {
                "inputs": D+1, "width": D+10, "depth": 2, "output": 1, "activation": "Tanh", "penalty": "ScaledTanh",
                "params_act": {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6()},
            }
        }
    params_equ, params_train, params_net_critic, params_net_actor = params["equation"], params["train"], params["net_critic"], params["net_actor"]

    def _entropy_alpha_is_zero(train):
        a = train.get("entropy_alpha", 0.0)
        if isinstance(a, (list, tuple)):
            return all(float(x) == 0.0 for x in a)
        return float(a) == 0.0

    if _entropy_alpha_is_zero(params_train):
        fs = float(args.fixed_std)
        params_train["policy_fixed_std"] = fs
        params_net_actor["fixed_std"] = fs
        print(f"[policy] entropy_alpha=0: fixed policy std = {fs} (no learned log_std)")
    else:
        params_train["policy_fixed_std"] = None
        params_net_actor.pop("fixed_std", None)
        print("[policy] entropy_alpha>0: learned log_std (--fixed_std ignored)")

    net_value = [Network(params_net_critic).to(device) for _ in range(D)]
    net_control = [Network_mu_sigma_split(params_net_actor).to(device) for _ in range(D)]
    params_train["_overwrite"] = args.overwrite
    if args.output_dir is not None:
        params_train["output_dir"] = os.path.abspath(args.output_dir)
    model = FBSNN(net_value, net_control, params, args.actor_path, args.critic_path)

    if args.actor_path is not None:
        ckpt_root = _detect_ckpt_set(model, args)
        if ckpt_root is None:
            parser.error("No complete multi-player checkpoint set found in the supplied directory")
        if not _load_best_multi_ckpt(model, ckpt_root):
            raise RuntimeError("Failed to load the complete multi-player checkpoint set")
        model.save_and_read_list(type="read", from_folder=ckpt_root)
        if args.resume:
            model.output_folder = str(prepare_run_directory(
                args.output_dir or os.path.join(ckpt_root, "resume"), resume=True,
            ))
            print(f"[resume] output_folder: {model.output_folder}")
            model.train_all(n=10, N_Iter=params_train["iteration"])
        model.save_current_eval_errors()
    else:
        model.train_all(n=20, N_Iter=params_train["iteration"])
    model.plot_topk_eval(k=3, device=device)

    # --------------------- Below is the code for result visualization --------------------- #
    model.save_and_read_list(type="read")    

    if len(model.Y_error_list) > 0 and len(model.control_error_list) > 0:
        print("Y error: %.3f%%, control error: %.3f%%" % (model.Y_error_list[-1].numpy() * 100, model.control_error_list[-1].numpy() * 100))
        model.plot_list([model.Y_error_list, model.control_error_list], "Error_value/Error_control", ["value function", "control"],
                        color=["#E7483D", "#6565AE", "#8FC751"], title=None, ylabel_log=True, baseline=0.01, filename="game_train")
    else:
        print("[info] Y_error_list or control_error_list is empty, skipping error plot (likely eval-only mode)")

    if len(model.CriticLoss_list) > 0 and len(model.ActorLoss_list) > 0:
        model.plot_loss(model.CriticLoss_list, model.ActorLoss_list, filename="merton_loss")
    else:
        print("[info] CriticLoss_list or ActorLoss_list is empty, skipping loss plot (likely eval-only mode)")
    model.plot_controlfunction(x_region=[9, 12], filename="merton_heatmap")

    # t_test, tau_test, W_test, B_test, P_test, P0_test = model.fetch_minibatch()
    # t_test, tau_test, W_test, B_test, P_test, P0_test = t_test.to(device), tau_test.to(device), W_test.to(device), B_test.to(device), P_test.to(device), P0_test.to(device)
    # X0 = torch.cat([params_equ["Xi"]] * params_equ["M"])
    # X_pred, Y_pred, control_pred, control_star, X_test = model.predict(t_test, tau_test, W_test, B_test, P_test, P0_test, X0.to(device))
    # Y_test = model.exact_value(t_test, X_pred)
    # t_test = t_test.cpu().detach().numpy()
    # X_pred = X_pred.cpu().detach().numpy()
    # X_test = X_test.cpu().detach().numpy()
    # Y_pred = Y_pred.cpu().detach().numpy()
    # Y_test = Y_test.cpu().detach().numpy()
    # P_test = P_test.cpu().detach().numpy()
    # P0_test = P0_test.cpu().detach().numpy()
    # control_test_list = [torch.tensor(model.control_exact) for _ in range(model.M*(model.N+1))]
    # control_test = torch.stack(control_test_list, dim=0).view(model.M, model.N+1, model.D)
    # control_pred = control_pred.cpu().detach().numpy()
    # control_star = control_star.cpu().detach().numpy()
    # control_test = control_test.cpu().detach().numpy()

    # samples = min(model.M, 5)
    # jump_num = 5
    # samples_index = torch.randperm(model.M)[0:samples]
    # samples_index = torch.unique(torch.cat([samples_index, model.jump_index[0:jump_num]], dim=0))

    # Y_errors = np.sqrt(np.mean((Y_test-Y_pred)**2, 0) / np.mean(Y_test**2, 0))
    # plt.figure(dpi=300)
    # k = 0
    # plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#6565AE', label=f"agent {k+1}")
    # for k in range(model.D):
    #     if k == 1:
    #         plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#E7483D', label=f"agent 2~10")
    #     if 2 <= k <= 9:
    #         plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#E7483D')
    #     if k == 10:
    #         plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#FAC074', label=f"agent 11~20")
    #     if 11 <= k <= 19:
    #         plt.plot(t_test[0, :, 0], np.expand_dims(Y_errors[:, k], axis=1), color='#FAC074')
    # plt.xlabel('$t$')
    # plt.ylabel('$L^2$ relative error of $e_t^v$')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(model.output_folder, "game_error_value.png"), dpi=300, bbox_inches='tight')
    # plt.show()

    # control_errors = np.sqrt(np.mean((control_test - control_pred) ** 2, 0) / np.mean(control_test ** 2, 0))
    # plt.figure(dpi=300)
    # k = 0
    # plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#6565AE', label=f"agent {k + 1}")
    # for k in range(model.D):
    #     if k == 1:
    #         plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#E7483D', label=f"agent 2~10")
    #     if 2 <= k <= 9:
    #         plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#E7483D')
    #     if k == 10:
    #         plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#FAC074', label=f"agent 11~20")
    #     if 11 <= k <= 19:
    #         plt.plot(t_test[0, :, 0], np.expand_dims(control_errors[:, k], axis=1), color='#FAC074')
    # plt.xlabel('$t$')
    # plt.ylabel('$L^2$ relative error of $e_t^u$')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(model.output_folder, "game_error_control.png"), dpi=300, bbox_inches='tight')
    # plt.show()

# def train_loop(self, itr, traj_only: bool = False, _control_exact_=False, predict: bool = False):
#         if not predict:
#             t, tau_sampled, W, B, P, P0 = self.fetch_minibatch()
#         else:
#             t, tau_sampled, W, B, P, P0 = (
#                 self.t_predict_set,
#                 self.tau_predict_set,
#                 self.W_predict_set,
#                 self.B_predict_set,
#                 self.P_predict_set,
#                 self.P0_predict_set
#             )

#         device = self.device
#         dt = self.dt
#         beta_disc    = float(self.params_equ["rho"])
#         gamma_entropy= self.params_train.get("entropy_alpha", 0.0)
#         gamma_soft   = self.params_train.get("gamma_soft", 0.0)
#         p_tsallis    = self.params_train.get("p_tsallis", 1.0)

#         M, N_plus_1, _ = t.shape
#         D = self.D
#         assert N_plus_1 == self.N + 1
#         mu        = torch.as_tensor(self.params_equ["mu"],    device=device).view(1, 1, D)
#         nu        = torch.as_tensor(self.params_equ["nu"],    device=device).view(1, 1, D)
#         sigma     = torch.as_tensor(self.params_equ["sigma"], device=device).view(1, 1, D)
#         alpha     = torch.as_tensor(self.params_equ["alpha"], device=device).view(1, 1, D)
#         beta_jump = torch.as_tensor(self.params_equ["beta"],  device=device).view(1, 1, D)

#         for i in range(D):
#             self.dict[f'optimizer_actor_{i}'].zero_grad(set_to_none=True)
#             self.dict[f'optimizer_critic_{i}'].zero_grad(set_to_none=True)

#         # buffer
#         X_buffer, V_buffer = [], []
#         action_buffer, mu_logsigma_buffer = [], []
#         logpi_buffer, tsallis_buffer = [], []

#         total_loss_actor  = 0.0
#         total_loss_critic = 0.0

#         K_actor  = self.params_train.get("K_actor", 20)
#         K_critic = self.params_train.get("K_critic", 5)
#         actor_loss_accum = critic_loss_accum = 0.0
#         actor_chunk_count = critic_chunk_count = 0
#         actor_loss_count = 0

#         t0 = t[:, 0, :]                             # (M,1)
#         X0 = self.Xi.to(device).view(1, D).expand(M, D).contiguous()
#         X0 = X0.detach().clone().requires_grad_(True)

#         X_buffer.append(X0.detach())

#         for k in range(self.N):
#             t1 = t[:, k+1, :]                   # (M,1)
#             W0 = W[:, k, :] 
#             W1 = W[:, k+1, :]
#             dW = W[:, k+1, :] - W[:, k, :]      # (M,D)
#             B0 = B[:, k, :]
#             B1 = B[:, k+1, :]
#             dB = B[:, k+1, :] - B[:, k, :]      # (M,1)

            # N_idio = count_jumps(P,  t0, t1) - torch.tensor(self.params_equ["lambda"]).unsqueeze(0).to(self.device) * dt    # (M,D)
            # N0     = count_jumps(P0, t0, t1) - torch.tensor(self.params_equ["lambda0"]).unsqueeze(0).to(self.device) * dt   # (M,1)

#             state = torch.cat([t0, X0], dim=-1)  # (M,1+D)
#             eps_k = torch.randn_like(X0[:, 0]).unsqueeze(-1)
#             # print("eps_k.shape", eps_k.shape)

#             if not _control_exact_: 

#                 mu_list, log_std_list = [], []
#                 u_list, logp_list, H_T_list = [], [], []
#                 for i in range(D):
#                     net_a = self.dict[f'net_actor_{i}']
#                     mu_i, log_std_i = net_a(state)          # (M,1)
#                     mu_i = mu_i.view(M,1)
#                     log_std_i = log_std_i.view(M,1)
#                     std_i = torch.exp(log_std_i)
#                     u_i = mu_i + std_i * eps_k

#                     dist_i = torch.distributions.Normal(mu_i, std_i)
#                     logp_i = dist_i.log_prob(u_i)
#                     if p_tsallis != 1:
#                         z_i = torch.exp(logp_i)
#                         H_T_i = (1.0 / (p_tsallis - 1.0)) * (1.0 - z_i**(p_tsallis - 1.0))
#                     else:
#                         H_T_i = -logp_i

#                     mu_list.append(mu_i)
#                     log_std_list.append(log_std_i)
#                     u_list.append(u_i)
#                     logp_list.append(logp_i)
#                     H_T_list.append(H_T_i)

#                 mu_all        = torch.cat(mu_list, dim=1)          # (M,D)
#                 log_std_all   = torch.cat(log_std_list, dim=1)     # (M,D)
#                 u_all         = torch.cat(u_list, dim=1)           # (M,D)
#                 logp_all      = torch.cat(logp_list, dim=1)        # (M,D)
#                 H_T_all       = torch.cat(H_T_list, dim=1)         # (M,D)

#                 action_buffer.append(u_all.detach())
#                 mu_logsigma_buffer.append(torch.cat([mu_all, log_std_all], dim=1).detach())
#                 logpi_buffer.append(logp_all.detach())
#                 tsallis_buffer.append(H_T_all.detach())

#                 drift       = u_all * mu.squeeze(0)    * dt        # (M,D)
#                 diff_idio   = u_all * nu.squeeze(0)    * dW        # (M,D)
#                 diff_common = u_all * sigma.squeeze(0) * dB        # (M,D)
#                 jump_idio   = u_all * alpha.squeeze(0) * N_idio    # (M,D)
#                 jump_common = u_all * beta_jump.squeeze(0) * N0    # (M,D)

#             else:
#                 u_all = self.control_exact.unsqueeze(0)
#                 drift       = u_all * mu.squeeze(0)    * dt        # (M,D)
#                 diff_idio   = u_all * nu.squeeze(0)    * dW        # (M,D)
#                 diff_common = u_all * sigma.squeeze(0) * dB        # (M,D)
#                 jump_idio   = u_all * alpha.squeeze(0) * N_idio    # (M,D)
#                 jump_common = u_all * beta_jump.squeeze(0) * N0    # (M,D)

#             X1 = X0 + drift + diff_idio + diff_common + jump_idio + jump_common

#             with torch.no_grad():
#                 reward = self.reward_func(X0, t0, dt)        # CARA utility
#                 disc = torch.exp(torch.tensor(-beta_disc * dt, device=device))

#             V0_players = []
#             V1_players = []
#             for i in range(D):
#                 net_c = self.dict[f'net_critic_{i}']
#                 V0_i = net_c(state)                     # (M,1)
#                 state1 = torch.cat([t1, X1], dim=-1)
#                 V1_i = net_c(state1)                   # (M,1)
#                 # soft target
#                 # V1_i = (1.0 - gamma_soft) * V1_i + gamma_soft * V0_i.detach()
#                 V0_players.append(V0_i.squeeze(-1))
#                 V1_players.append(V1_i.squeeze(-1))
#             V0_players = torch.stack(V0_players, dim=-1)   # (M,D)
#             V1_players = torch.stack(V1_players, dim=-1)   # (M,D)

#             V_buffer.append(V1_players.detach())
#             X_buffer.append(X1.detach())

#             if not traj_only:
#                 # TD target: r dt + e^{-rho dt} V1
#                 td_target = reward * dt + disc * V1_players.detach()
#                 delta = td_target - V0_players          # (M,D)

#                 # critic loss: MSE over (M,D)
#                 loss_critic_step = (delta**2).mean()

#                 critic_loss_accum += loss_critic_step
#                 critic_chunk_count += 1

#                 if ((k + 1) % K_critic == 0) or (k == self.N - 1):
#                     loss_critic_chunk = critic_loss_accum / float(critic_chunk_count)
#                     for i in range(D):
#                         self.dict[f'optimizer_critic_{i}'].zero_grad(set_to_none=True)
#                     loss_critic_chunk.backward()
#                     for i in range(D):
#                         nn.utils.clip_grad_norm_(self.dict[f'net_critic_{i}'].parameters(), 1.0)
#                         self.dict[f'optimizer_critic_{i}'].step()
#                         self.dict[f'scheduler_critic_{i}'].step()
#                     total_loss_critic += loss_critic_chunk.detach()
#                     critic_loss_accum = 0.0
#                     critic_chunk_count = 0

#                 # ========= Actor：per-player PG =========

#                 loss_actor_step_total = 0.0

#                 for i in range(D):
#                     adv_i  = advantage[:, i]       # (M,)
#                     logp_i = logp_all[:, i]       # (M,)
#                     H_i    = H_T_all[:, i]        # (M,)

#                     # loss_i = -(logp_i * adv_i.detach())

#                     adv_mean = adv_i.mean()
#                     adv_std  = adv_i.std()
#                     if adv_std > 1e-8:
#                         adv_i_norm = (adv_i - adv_mean) / adv_std
#                     else:

#                     loss_i = -(logp_i * adv_i_norm.detach())

#                     if gamma_entropy != 0.0:
#                         loss_i = loss_i + gamma_entropy * H_i

#                     # batch mean
#                     loss_i = loss_i.mean()

#                     retain = (i < D - 1)
#                     loss_i.backward(retain_graph=retain)
#                     loss_actor_step_total += loss_i.detach()

#                 actor_loss_accum += loss_actor_step_total
#                 actor_loss_count += 1

#                 if ((k + 1) % K_actor == 0) or (k == self.N - 1):
#                     loss_actor_chunk = actor_loss_accum / float(max(actor_loss_count, 1))

#                     for i in range(D):
#                         net_a = self.dict[f'net_actor_{i}']
#                         opt_a = self.dict[f'optimizer_actor_{i}']
#                         sch_a = self.dict[f'scheduler_actor_{i}']

#                         nn.utils.clip_grad_norm_(net_a.parameters(), 1.0)
#                         opt_a.step()
#                         sch_a.step()
#                         opt_a.zero_grad(set_to_none=True)

#                     total_loss_actor += loss_actor_chunk
#                     actor_loss_accum = 0.0
#                     actor_loss_count = 0

#                 # # ===== All actors mean: policy gradient + entropy =====
#                 # advantage = td_target - V0_players      # (M,D)

#                 # # logp_all: (M,D) for actions u_i: policy gradient: sum_i E[ - (1/beta) logpi_i * advantage_i ]
#                 # pg = -(logp_all * advantage.detach()) / (dt + 1e-8)  # normalized
#                 # loss_actor_step = pg

#                 # if gamma_entropy != 0.0:

#                 #     loss_actor_step += gamma_entropy * H_T_all

#                 # actor_loss_accum += loss_actor_step.mean()
#                 # actor_chunk_count += 1

#                 # if ((k + 1) % K_actor == 0) or (k == self.N - 1):
#                 #     loss_actor_chunk = actor_loss_accum / float(actor_chunk_count)
#                 #     for i in range(D):
#                 #         self.dict[f'optimizer_actor_{i}'].zero_grad(set_to_none=True)
#                 #     loss_actor_chunk.backward()
#                 #     for i in range(D):
#                 #         nn.utils.clip_grad_norm_(self.dict[f'net_actor_{i}'].parameters(), 1.0)
#                 #         self.dict[f'optimizer_actor_{i}'].step()
#                 #         self.dict[f'scheduler_actor_{i}'].step()
#                 #     total_loss_actor += loss_actor_chunk.detach()
#                 #     actor_loss_accum = 0.0
#                 #     actor_chunk_count = 0

#             X0 = X1.detach().clone().requires_grad_(True)
#             t0, W0, B0 = t1, W1, B1

#         with torch.no_grad():
#             # control_exact: shape (D,)
#             u_exact = self.control_exact.to(device=device) #(D,)

#             # if len(action_buffer) > 0:
#             #     actions_all = torch.stack(action_buffer, dim=1)  # (M, N, D)
#             #     u_mean = actions_all.mean(dim=(0, 1))            # (D,)
#             #     control_error = torch.sqrt(
#             #         ((u_mean - u_exact)**2).mean()
#             #         / (u_exact**2 + 1e-12).mean()
#             #     )
#             #     if k == self.N -1:
#             #         print("u_pred", u_mean.mean())
#             #         print("u_exact", u_exact.mean())
#             if len(action_buffer) > 0:
#                 actions_all = torch.stack(action_buffer, dim=1)  # (M, N, D)

#                 # control_error = (torch.sqrt(
#                 #     ((actions_all - u_exact.unsqueeze(0).unsqueeze(0))**2).mean(dim=0)
#                 #     / (u_exact**2 + 1e-12).mean(dim=0)
#                 # )).mean()

#                 control_error = (torch.sqrt((actions_all - u_exact.unsqueeze(0).unsqueeze(0))**2).mean(dim=0)).mean()
#                 if k == self.N -1:
#                     print("u[0]_pred", actions_all[:,:,0].mean())
#                     print("u_exact[0]", u_exact[0])
#                     print("u[-1]_pred", actions_all[:,:,-1].mean())
#                     print("u_exact[-1]", u_exact[-1])
#             else:
#                 control_error = None

#             X_traj = torch.stack(X_buffer, dim=1)      # (M, N+1, D)
#             t_traj = t.to(device)
#             V_exact = self.exact_value(t_traj, X_traj) # (M, N+1, D)
#             V_num = torch.stack(V_buffer, dim=1)       # (M, N, D)
#             if k == self.N -1:
#                 print("V_num.mean", V_num.mean())
#                 print("V_exact[:, 1:, :].mean", V_exact[:, 1:, :].mean())
#             Y_error = torch.sqrt(((V_num - V_exact[:, 1:, :])**2 / (V_exact[:, 1:, :]**2 + 1e-12)).mean())

#         if not traj_only:
#             steps = float(self.N)
#             total_loss_actor = total_loss_actor  #/ steps
#             total_loss_critic = total_loss_critic  #/ steps
#             self.ActorLoss_list.append(total_loss_actor.detach().cpu())
#             self.CriticLoss_list.append(total_loss_critic.detach().cpu())
#             if control_error is not None:
#                 self.control_error_list.append(control_error.detach().cpu())
#             self.Y_error_list.append(Y_error.detach().cpu())

#         return (
#             torch.stack(X_buffer, dim=1),                 # (M, N+1, D)
#             torch.stack(V_buffer, dim=1),                 # (M, N+1, D)
#             V_buffer[0],                                  # V at t0
#             torch.stack(action_buffer, dim=1) if action_buffer else None,
#             torch.stack(mu_logsigma_buffer, dim=1) if mu_logsigma_buffer else None,
#             Y_error,
#             control_error,
#             total_loss_critic,
#             total_loss_actor
#         )
