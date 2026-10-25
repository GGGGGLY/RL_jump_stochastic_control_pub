"""
Parameter persistence and restoration for the Merton experiments.

Save equation and training parameters, preserve a copy of the run script,
restore tensors and device entries from JSON, and apply loaded parameters
to a supplied model instance. Metadata paths are stored relative to the
run directory and resolved when loading.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import hashlib
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from run_io import metadata_for_save, metadata_from_disk


def _sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_chmod_readonly(path: str) -> None:
    """Best-effort make a file read-only (helps prevent accidental overwrite)."""
    try:
        os.chmod(path, 0o444)
    except Exception:
        pass


def _convert_tensor(obj: Any) -> Any:
    """Convert torch.Tensor / torch.device into JSON-serializable structures."""
    if isinstance(obj, torch.Tensor):
        return {
            "_type": "torch.Tensor",
            "data": obj.cpu().detach().numpy().tolist(),
            "shape": list(obj.shape),
        }
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _convert_tensor(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_tensor(item) for item in obj]
    return obj


def _restore_tensor(obj: Any) -> Any:
    """Restore torch.Tensor / torch.device from serialized JSON structures."""
    if isinstance(obj, dict) and obj.get("_type") == "torch.Tensor":
        data = np.array(obj["data"])
        return torch.tensor(data)
    if isinstance(obj, dict):
        return {k: _restore_tensor(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_tensor(item) for item in obj]
    if isinstance(obj, str) and (obj.startswith("cuda") or obj == "cpu"):
        return torch.device(obj)
    return obj


def save_params(model: Any, save_dir: str) -> bool:
    """
    Save `model.params_equ` and `model.params_train` into save_dir/params.json.

    Keep an existing params.json. If its contents differ or cannot be read,
    save the new parameters as params__conflict__{ts}_{hash}.json.
    New files are made read-only where supported.
    """
    try:
        full_params = {
            "equation": _convert_tensor(copy.deepcopy(model.params_equ)),
            "train": _convert_tensor(copy.deepcopy(model.params_train)),
        }
        full_params = metadata_for_save(full_params, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        params_path = os.path.join(save_dir, "params.json")
        if os.path.isfile(params_path):
            # Don't overwrite; but if different, persist a conflict snapshot.
            try:
                with open(params_path, "r") as f:
                    old = json.load(f)
                if old == full_params:
                    print(f"[save] params already exist and match: {params_path} (skip)")
                    return True
            except Exception:
                # If we can't load/compare, treat as conflict.
                pass
            ts = time.strftime("%Y%m%d_%H%M%S")
            payload_hash = hashlib.sha256(
                json.dumps(full_params, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
            conflict_path = os.path.join(save_dir, f"params__conflict__{ts}_{payload_hash}.json")
            with open(conflict_path, "w") as f:
                json.dump(full_params, f, indent=2)
            _safe_chmod_readonly(conflict_path)
            print(f"[save] WARNING: params.json exists; wrote conflict snapshot to {conflict_path}")
            return True

        with open(params_path, "w") as f:
            json.dump(full_params, f, indent=2)
        _safe_chmod_readonly(params_path)
        print(f"[save] params saved (frozen) to {params_path}")
        return True
    except Exception as e:
        print(f"[warn] failed to save params: {e}")
        return False


def save_run_script(source_file: str, save_dir: str) -> bool:
    """
    Copy the running script into save_dir for reproducibility.

    Keep an existing script copy. If its contents differ or cannot be compared,
    save a versioned copy as {name}__conflict__{ts}_{hash}.py.
    New copies are made read-only where supported.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        src = os.path.abspath(source_file)
        dst = os.path.join(save_dir, os.path.basename(src))
        if os.path.isfile(dst):
            # If same content, do nothing; otherwise write a conflict snapshot.
            try:
                same = False
                if os.path.getsize(dst) == os.path.getsize(src):
                    same = (_sha256_file(dst) == _sha256_file(src))
                if same:
                    print(f"[save] run script already exists and matches: {dst} (skip)")
                    return True
            except Exception:
                pass
            ts = time.strftime("%Y%m%d_%H%M%S")
            try:
                src_hash = _sha256_file(src)[:12]
            except Exception:
                src_hash = "unknownhash"
            base = os.path.basename(src)
            conflict = os.path.join(save_dir, f"{base}__conflict__{ts}_{src_hash}.py")
            shutil.copy2(src, conflict)
            _safe_chmod_readonly(conflict)
            print(f"[save] WARNING: frozen script exists; wrote conflict snapshot to {conflict}")
            return True

        shutil.copy2(src, dst)
        _safe_chmod_readonly(dst)
        print(f"[save] run script copied (frozen) to {dst}")
        return True
    except Exception as e:
        print(f"[warn] failed to copy run script: {e}")
        return False


def load_params(load_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load params.json from a specified directory and convert serialized entries
    back to torch.Tensor / torch.device.
    """
    params_path = os.path.join(load_dir, "params.json")
    if not os.path.isfile(params_path):
        print(f"[warn] params.json not found in {load_dir}")
        return None
    try:
        with open(params_path, "r") as f:
            loaded_params = metadata_from_disk(json.load(f), load_dir)
        restored_params: Dict[str, Any] = {}
        for key in ["equation", "train"]:
            if key in loaded_params:
                restored_params[key] = _restore_tensor(loaded_params[key])
        print(f"[load] params loaded from {params_path}")
        return restored_params
    except Exception as e:
        print(f"[warn] failed to load params: {e}")
        return None


def update_params_from_dict(model: Any, params_dict: Optional[Dict[str, Any]]) -> bool:
    """
    Apply loaded equation and training parameters to the supplied model.

    When equation parameters are present, refresh Xi/T/M/N/D/p, the time step,
    and h_t/a_t/g_t using the current training parameters. Then apply the loaded
    training parameters while retaining the model's current device.
    """
    if params_dict is None:
        return False
    try:
        if "equation" in params_dict:
            model.params_equ.update(params_dict["equation"])
            if "Xi" in params_dict["equation"]:
                model.Xi = params_dict["equation"]["Xi"].to(model.device).float()
            if "T" in params_dict["equation"]:
                model.T = params_dict["equation"]["T"]
            if "M" in params_dict["equation"]:
                model.M = params_dict["equation"]["M"]
            if "N" in params_dict["equation"]:
                model.N = params_dict["equation"]["N"]
            if "D" in params_dict["equation"]:
                model.D = params_dict["equation"]["D"]
            if "p" in params_dict["equation"]:
                model.p = params_dict["equation"]["p"]
            model.dt = model.T / model.N

            # recompute dependent variables
            mu = torch.as_tensor(model.params_equ["mu"], device=model.device)
            r = torch.as_tensor(model.params_equ["r"], device=model.device)
            sigma = torch.as_tensor(model.params_equ["sigma"], device=model.device)
            lam = torch.as_tensor(model.params_equ["lambda"], device=model.device)
            z = torch.as_tensor(model.params_equ["z"], device=model.device)
            beta = torch.as_tensor(model.params_equ["beta"], device=model.device)
            gamma_entropy = torch.as_tensor(model.params_train["entropy_alpha"], device=model.device)
            B = sigma**2 + lam * z**2
            model.h_t = 1 / (beta - model.p * r - (model.p * (mu - r) ** 2) / (2 * B * (1 - model.p)))
            model.a_t = gamma_entropy * model.p / (2 * beta)
            model.g_t = (1 / beta) * (
                r * model.a_t
                + gamma_entropy / 2 * (torch.log(2 * torch.pi * gamma_entropy) - torch.log(B) - torch.log((1 - model.p) * model.h_t))
            )

        if "train" in params_dict:
            model.params_train.update(params_dict["train"])
            model.params_train["device"] = model.device  # Keep networks and data on the current device.
            if "eval_every" in params_dict["train"]:
                model.eval_every = params_dict["train"]["eval_every"]
            if "X0_init_type" in params_dict["train"]:
                model.X0_init_type = params_dict["train"]["X0_init_type"]
            if "X0_init_radius" in params_dict["train"]:
                model.X0_init_radius = params_dict["train"]["X0_init_radius"]
            if "plot_traj_mode" in params_dict["train"]:
                model.plot_traj_mode = params_dict["train"]["plot_traj_mode"]
            else:
                model.plot_traj_mode = model.params_train.get("plot_traj_mode", model.plot_traj_mode)

        print("[update] model params updated from checkpoint")
        return True
    except Exception as e:
        print(f"[warn] failed to update params: {e}")
        return False

