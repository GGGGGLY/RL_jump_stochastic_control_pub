"""
PINN benchmark solver and cache utilities for the Merton experiments.

Provide value-network evaluation and derivatives, dense-grid interpolation,
cache persistence, and the solve_value_function_pinn solver. Model-dependent
functions receive the experiment model explicitly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def get_pinn_cache_paths(model: Any) -> Tuple[Optional[str], Optional[str]]:
    """Get paths for PINN cache files in output_folder."""
    if not hasattr(model, "output_folder") or model.output_folder is None:
        return None, None
    state_dict_path = os.path.join(model.output_folder, "pinn_state_dict.pt")
    params_path = os.path.join(model.output_folder, "pinn_params.json")
    return state_dict_path, params_path


def value_net_forward(net: torch.nn.Module, x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
    x_norm = 2.0 * (x - x_min) / max(1e-12, (x_max - x_min)) - 1.0
    return net(x_norm)


def value_net_forward_chunked(
    model: Any,
    net: torch.nn.Module,
    x: torch.Tensor,
    x_min: float,
    x_max: float,
    chunk: Optional[int] = None,
    use_no_grad: bool = True,
) -> torch.Tensor:
    if chunk is None:
        chunk = int(getattr(model, "params_train", {}).get("pinn_chunk", 200000))
    x_norm = 2.0 * (x - x_min) / max(1e-12, (x_max - x_min)) - 1.0
    outs = []
    if use_no_grad:
        with torch.no_grad():
            for i in range(0, x_norm.shape[0], chunk):
                outs.append(net(x_norm[i : i + chunk]))
    else:
        for i in range(0, x_norm.shape[0], chunk):
            outs.append(net(x_norm[i : i + chunk]))
    return torch.cat(outs, dim=0)


def value_net_with_derivatives(
    net: torch.nn.Module, x: torch.Tensor, x_min: float, x_max: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        x = x.clone().detach().requires_grad_(True)
        V = value_net_forward(net, x, x_min, x_max)
        Vx = torch.autograd.grad(torch.sum(V), x, retain_graph=True, create_graph=True)[0]
        Vxx = torch.autograd.grad(torch.sum(Vx), x, retain_graph=True, create_graph=True)[0]
        return V, Vx, Vxx


def save_pinn_dense_grid(model: Any, net: torch.nn.Module, cache: dict, pinn_params: dict) -> None:
    """Save dense grid of V(x) values for interpolation-based evaluation."""
    try:
        if not hasattr(model, "output_folder") or model.output_folder is None:
            return
        x_min = cache["x_min"]
        x_max = cache["x_max"]
        n_x_dense = int(getattr(model, "params_train", {}).get("pinn_dense_grid_n_x", 10000))

        x_dense = torch.linspace(
            x_min, x_max, n_x_dense, device=net[0].weight.device, dtype=net[0].weight.dtype
        ).view(-1, 1)

        with torch.no_grad():
            V_dense = value_net_forward(net, x_dense, x_min, x_max)
            if getattr(model, "params_train", {}).get("pinn_dense_grid_save_derivatives", False):
                V_dense_tensor = x_dense.clone().detach().requires_grad_(True)
                V_dense_grad = value_net_forward(net, V_dense_tensor, x_min, x_max)
                Vx_dense = torch.autograd.grad(torch.sum(V_dense_grad), V_dense_tensor, retain_graph=True)[0]
                Vxx_dense = torch.autograd.grad(torch.sum(Vx_dense), V_dense_tensor, retain_graph=True)[0]
            else:
                Vx_dense = None
                Vxx_dense = None

        dense_grid_path = os.path.join(model.output_folder, "pinn_dense_grid.pt")
        dense_grid_data = {"x_dense": x_dense.cpu(), "V_dense": V_dense.cpu(), "x_min": x_min, "x_max": x_max}
        if Vx_dense is not None:
            dense_grid_data["Vx_dense"] = Vx_dense.cpu()
        if Vxx_dense is not None:
            dense_grid_data["Vxx_dense"] = Vxx_dense.cpu()

        torch.save(dense_grid_data, dense_grid_path)
        print(f"[pinn-cache] Saved dense grid to {dense_grid_path} (n_x={n_x_dense})")
    except Exception as e:
        print(f"[warn] Failed to save PINN dense grid: {e}")


def save_pinn_to_disk(model: Any, net: torch.nn.Module, pinn_params: dict, params_hash: int, cache: dict) -> None:
    """Save PINN model state_dict and parameters to disk (optionally dense grid)."""
    try:
        state_dict_path, params_path = get_pinn_cache_paths(model)
        if state_dict_path is None:
            return

        torch.save(net.state_dict(), state_dict_path)

        params_to_save = pinn_params.copy()
        params_to_save["params_hash"] = params_hash
        params_to_save["loss_pde_final"] = cache.get("loss_pde_final", None)
        with open(params_path, "w") as f:
            json.dump(params_to_save, f, indent=2)

        print(f"[pinn-cache] Saved PINN model to {state_dict_path}")
        print(f"[pinn-cache] Saved PINN parameters to {params_path}")

        if getattr(model, "params_train", {}).get("pinn_cache_dense_grid", False):
            save_pinn_dense_grid(model, net, cache, pinn_params)
    except Exception as e:
        print(f"[warn] Failed to save PINN cache to disk: {e}")


def load_pinn_dense_grid(model: Any) -> Optional[dict]:
    """Load dense grid cache from disk."""
    try:
        if not hasattr(model, "output_folder") or model.output_folder is None:
            return None
        dense_grid_path = os.path.join(model.output_folder, "pinn_dense_grid.pt")
        if not os.path.isfile(dense_grid_path):
            return None
        dense_grid_data = torch.load(dense_grid_path, map_location="cpu")
        print(f"[pinn-cache] Loaded dense grid from {dense_grid_path}")
        return dense_grid_data
    except Exception as e:
        print(f"[warn] Failed to load PINN dense grid: {e}")
        return None


def interpolate_v_from_dense_grid(
    x_eval: torch.Tensor, dense_grid_cache: dict, device: torch.device, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Interpolate V(x) from dense grid cache using linear interpolation (scipy if available)."""
    try:
        x_dense = dense_grid_cache["x_dense"].to(device=device, dtype=dtype).squeeze(-1)
        V_dense = dense_grid_cache["V_dense"].to(device=device, dtype=dtype).squeeze(-1)
        x_eval_flat = x_eval.squeeze(-1) if x_eval.dim() > 1 else x_eval

        x_min = float(dense_grid_cache["x_min"])
        x_max = float(dense_grid_cache["x_max"])
        x_eval_flat = torch.clamp(x_eval_flat, min=x_min, max=x_max)

        x_dense_np = x_dense.detach().cpu().numpy()
        V_dense_np = V_dense.detach().cpu().numpy()
        x_eval_np = x_eval_flat.detach().cpu().numpy()

        # Prefer scipy's interp1d if present
        from scipy.interpolate import interp1d  # type: ignore

        interp_func = interp1d(
            x_dense_np, V_dense_np, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        V_eval_np = interp_func(x_eval_np)
        V_eval = torch.from_numpy(V_eval_np).to(device=device, dtype=dtype)
        return V_eval.view_as(x_eval)
    except Exception as e:
        # If scipy isn't installed or interpolation fails, caller can fall back to NN evaluation.
        print(f"[warn] Failed to interpolate from dense grid: {e}")
        return None


def load_pinn_from_disk(
    model: Any,
    pinn_params: dict,
    params_hash: int,
    device: torch.device,
    dtype: torch.dtype,
    width: int,
    depth: int,
) -> Optional[dict]:
    """Load PINN model from disk if parameters match. Optionally loads dense grid cache."""
    try:
        print("[pinn-cache] Trying to load PINN from disk...", flush=True)
        state_dict_path, params_path = get_pinn_cache_paths(model)
        if state_dict_path is None or params_path is None:
            return None
        if not os.path.isfile(state_dict_path) or not os.path.isfile(params_path):
            return None

        with open(params_path, "r") as f:
            saved_params = json.load(f)
        saved_params.pop("params_hash", None)
        saved_params.pop("loss_pde_final", None)

        # Backward-compat: older caches used "x_max" only. Normalize to new keys.
        if "x_max_train" not in saved_params and "x_max" in saved_params:
            saved_params["x_max_train"] = saved_params["x_max"]
        if "x_max_eval" not in saved_params and "x_max" in saved_params:
            saved_params["x_max_eval"] = saved_params["x_max"]

        for key in pinn_params:
            if key not in saved_params:
                return None
            if isinstance(pinn_params[key], float):
                if abs(pinn_params[key] - saved_params[key]) > 1e-10:
                    return None
            elif pinn_params[key] != saved_params[key]:
                return None

        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        net = nn.Sequential(*layers).to(device=device, dtype=dtype)
        print("[pinn-cache] Loading state_dict...", flush=True)
        net.load_state_dict(torch.load(state_dict_path, map_location=device))
        net.eval()
        print("[pinn-cache] state_dict loaded.", flush=True)

        x_min = saved_params.get("x_min", pinn_params["x_min"])
        # New-format params store both training max and eval/normalization max.
        x_max_train = saved_params.get("x_max_train", saved_params.get("x_max", pinn_params.get("x_max_train", None)))
        if x_max_train is None:
            x_max_train = pinn_params.get("x_max_eval", pinn_params.get("x_max_train", 0.0))
        x_max_eval = saved_params.get("x_max_eval", x_max_train)
        u_min = saved_params.get("u_min", pinn_params["u_min"])
        u_max = saved_params.get("u_max", pinn_params["u_max"])
        n_u = saved_params.get("n_u", pinn_params["n_u"])

        u_grid = torch.linspace(u_min, u_max, n_u, device=device, dtype=dtype).view(1, -1)
        du = float(u_grid[0, 1].item() - u_grid[0, 0].item()) if n_u > 1 else 1.0
        u_weights = torch.ones_like(u_grid) * du
        if n_u > 1:
            u_weights[:, 0] *= 0.5
            u_weights[:, -1] *= 0.5

        _core_keys = ("mu", "r", "sigma", "lam", "z", "beta", "gamma", "p", "x_min", "x_max_train", "x_max_eval", "u_min", "u_max")
        _core_subset = {k: saved_params[k] for k in _core_keys if k in saved_params}
        _core_hash = hash(tuple(sorted(_core_subset.items()))) if _core_subset else None

        cache = {
            "net": net,
            "x_min": float(x_min),
            # Backward-compatible key used by callers for normalization/clamping:
            "x_max": float(x_max_eval),
            "x_max_train": float(x_max_train),
            "x_max_eval": float(x_max_eval),
            "u_grid": u_grid.detach(),
            "u_weights": u_weights.detach(),
            "loss_pde_final": saved_params.get("loss_pde_final", None),
            "_core_hash": _core_hash,
        }

        if getattr(model, "params_train", {}).get("pinn_cache_dense_grid", False):
            print("[pinn-cache] Loading dense grid from disk...", flush=True)
            dense_grid_cache = load_pinn_dense_grid(model)
            if dense_grid_cache is not None:
                cache["dense_grid"] = dense_grid_cache
            print("[pinn-cache] Dense grid load done.", flush=True)
        return cache
    except Exception as e:
        print(f"[warn] Failed to load PINN cache from disk: {e}")
        return None


def solve_value_function_pinn(
    model: Any,
    mu: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    lam: torch.Tensor,
    z: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    x_min: float = 1e-5,
    x_max: float = 80.0,
    x_eval_max: Optional[float] = None,
    n_x: int = 500,
    n_u: int = 100,
    u_min: float = -1.0,
    u_max: float = 10.0,
    n_iter: int = 3000,
    lr: float = 5e-4,
    width: int = 256,
    depth: int = 5,
    use_cache: bool = True,
) -> dict:
    """PINN solver used by compute_exact_entropy_with_cache. Returns a cache dict."""
    # Training grid is built on [x_min, x_max]. However, the jump term evaluates V(x*(1+z u)),
    # which can exceed x_max. To avoid NN input extrapolation (x_norm outside [-1,1]),
    # we optionally use a larger normalization/evaluation bound x_eval_max (>= x_max).
    #
    # When x_eval_max is omitted, derive a bound covering the jump-shifted grid:
    #   x_eval_max := max(x_max, max_u (1+z u) * x_max + pad)
    # where u is taken over {u_min, u_max} (covers sign flips), pad defaults to 2.0.
    try:
        pad = float((getattr(model, "params_train", {}) or {}).get("pinn_x_eval_pad", 2.0))
    except Exception:
        pad = 2.0
    if x_eval_max is None:
        try:
            z_val = float(z.detach().cpu().item())
        except Exception:
            z_val = float(z.item()) if hasattr(z, "item") else float(z)
        one_plus_lo = 1.0 + z_val * float(u_min)
        one_plus_hi = 1.0 + z_val * float(u_max)
        one_plus_max = max(one_plus_lo, one_plus_hi)
        x_eval_max_f = max(float(x_max), one_plus_max * float(x_max) + pad)
    else:
        x_eval_max_f = float(x_eval_max)
    x_eval_max_f = max(float(x_max), float(x_eval_max_f))
    params_train = getattr(model, "params_train", {}) or {}
    pinn_params = {
        "mu": float(mu.item()),
        "r": float(r.item()),
        "sigma": float(sigma.item()),
        "lam": float(lam.item()),
        "z": float(z.item()),
        "beta": float(beta.item()),
        "gamma": float(gamma.item()),
        "p": float(getattr(model, "p")),
        "x_min": float(x_min),
        "x_max_train": float(x_max),
        "x_max_eval": float(x_eval_max_f),
        "n_x": int(n_x),
        "n_u": int(n_u),
        "u_min": float(u_min),
        "u_max": float(u_max),
        "n_iter": int(n_iter),
        "width": int(width),
        "depth": int(depth),
        "lr": float(lr),
        "pinn_reg_vx_vxx_coeff": float(params_train.get("pinn_reg_vx_vxx_coeff", 0.0)),
        "device": str(device),
        "dtype": str(dtype),
    }
    params_hash = hash(tuple(sorted(pinn_params.items())))
    # Core hash: PDE and domain only (ignore device, dtype, n_x, n_u, n_iter, lr, width, depth).
    # Reuse cache when core matches so panel_c can reuse the cache from predict() even if device/hash differs.
    _core_keys = ("mu", "r", "sigma", "lam", "z", "beta", "gamma", "p", "x_min", "x_max_train", "x_max_eval", "u_min", "u_max")
    core_hash = hash(tuple(sorted((k, pinn_params[k]) for k in _core_keys if k in pinn_params)))

    if use_cache and getattr(model, "_exact_solution_cache_nn", None) is not None:
        cached_hash = getattr(model, "_exact_solution_params_hash_nn", None)
        if cached_hash == params_hash:
            print("[pinn-cache] Using in-memory PINN cache (skip solve).", flush=True)
            return model._exact_solution_cache_nn
        # Reuse cache if only device/dtype/n_x/n_u etc. differ but PDE and domain match (e.g. panel_c after predict).
        _cache = model._exact_solution_cache_nn
        cached_core = _cache.get("_core_hash", None) if isinstance(_cache, dict) else getattr(model, "_exact_solution_core_hash_nn", None)
        if cached_core is not None and cached_core == core_hash:
            print("[pinn-cache] Using in-memory PINN cache (core params match; skip solve).", flush=True)
            return model._exact_solution_cache_nn
        if cached_hash is not None:
            print("[pinn-cache] In-memory cache hash mismatch (cached=%s, current=%s); will try disk or retrain." % (cached_hash, params_hash), flush=True)

    if use_cache:
        cache_loaded = load_pinn_from_disk(model, pinn_params, params_hash, device, dtype, width, depth)
        if cache_loaded is not None:
            print("[pinn-cache] Loaded PINN model from disk cache")
            model._exact_solution_cache_nn = cache_loaded
            model._exact_solution_params_hash_nn = params_hash
            return cache_loaded

    # ------------------------------------------------------------------
    # Reproducibility: if cache miss triggers a fresh PINN training run,
    # we want the resulting V-net to be stable across evaluation runs.
    # Use fork_rng so we DON'T perturb global RNG state (training loop etc).
    # ------------------------------------------------------------------
    print(
        "[pinn-cache] Cache miss: training PINN from scratch (n_iter=%d, n_x=%d, n_u=%d). "
        "If very slow, set pinn_n_x/pinn_n_u smaller or ensure disk cache exists."
        % (n_iter, n_x, n_u),
        flush=True,
    )
    params_train = getattr(model, "params_train", {}) or {}
    use_fixed_seed = bool(params_train.get("pinn_train_use_fixed_seed", True))
    seed_val = params_train.get("pinn_train_fixed_seed", 123456)
    seed_int = int(seed_val) if (use_fixed_seed and seed_val is not None) else None

    cuda_devices = None
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            # fork_rng expects CUDA device indices
            cuda_devices = [device.index] if device.index is not None else list(range(torch.cuda.device_count()))
        except Exception:
            cuda_devices = None

    with torch.random.fork_rng(devices=cuda_devices, enabled=bool(seed_int is not None)):
        if seed_int is not None:
            torch.manual_seed(seed_int)
            if torch.cuda.is_available():
                try:
                    torch.cuda.manual_seed_all(seed_int)
                except Exception:
                    pass

        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        net = nn.Sequential(*layers).to(device=device, dtype=dtype)
        optim_v = optim.Adam(net.parameters(), lr=lr)

        x_grid = torch.linspace(x_min, x_max, n_x, device=device, dtype=dtype).view(-1, 1)
        u_grid = torch.linspace(u_min, u_max, n_u, device=device, dtype=dtype).view(1, -1)
        du = float(u_grid[0, 1].item() - u_grid[0, 0].item()) if n_u > 1 else 1.0
        u_weights = torch.ones_like(u_grid) * du
        if n_u > 1:
            u_weights[:, 0] *= 0.5
            u_weights[:, -1] *= 0.5

        mu_minus_r = mu - r
        r0 = r
        sig0 = sigma
        lam0 = lam

        n_shift = n_x * n_u
        v_shift_chunk = int(getattr(model, "params_train", {}).get("pinn_train_v_shift_chunk", 10000))
        v_shift_chunk = max(1, min(n_shift, v_shift_chunk))
        print(
            "[pinn-train] n_x=%d, n_u=%d, n_shift=%d (V_shift chunk=%d, ~%d chunks/iter)."
            % (n_x, n_u, n_shift, v_shift_chunk, (n_shift + v_shift_chunk - 1) // v_shift_chunk),
            flush=True,
        )
        t_start = time.perf_counter()

        with torch.enable_grad():
            for it in range(n_iter):
                optim_v.zero_grad(set_to_none=True)
                V, Vx, Vxx = value_net_with_derivatives(net, x_grid, x_min, x_eval_max_f)

                x_col = x_grid
                one_plus = 1.0 + z * u_grid
                raw_shift = x_col * one_plus
                x_shift = raw_shift
                if not getattr(model, "_pinn_shift_range_logged", False):
                    try:
                        shift_min = float(raw_shift.min().detach().cpu().item())
                        shift_max = float(raw_shift.max().detach().cpu().item())
                        if shift_min < x_min or shift_max > x_eval_max_f:
                            print(
                                f"[pinn-grid] WARNING: x_shift out of range. "
                                f"x_shift=[{shift_min:.4g}, {shift_max:.4g}] "
                                f"but x_eval=[{x_min:.4g}, {x_eval_max_f:.4g}] "
                                f"(x_grid=[{x_min:.4g}, {x_max:.4g}]). "
                                "Consider enlarging pinn_x_eval_max (or reducing u_max / z)."
                            )
                    except Exception:
                        pass
                    model._pinn_shift_range_logged = True

                x_shift_flat = x_shift.reshape(-1, 1)
                V_shift_parts = []
                for start in range(0, n_shift, v_shift_chunk):
                    end = min(start + v_shift_chunk, n_shift)
                    v_c = value_net_forward(net, x_shift_flat[start:end], x_min, x_eval_max_f)
                    V_shift_parts.append(v_c)
                V_shift = torch.cat(V_shift_parts, dim=0).view(n_x, n_u)

                reward = torch.pow(torch.clamp(x_col, min=1e-12), getattr(model, "p")) / getattr(model, "p")
                drift = (r0 + mu_minus_r * u_grid) * (x_col * Vx)
                diffusion = 0.5 * (sig0 * u_grid * x_col) ** 2 * Vxx
                jump = lam0 * (V_shift - V - z * u_grid * x_col * Vx)
                H = reward + drift + diffusion + jump

                if gamma.item() > 0:
                    H_max, _ = torch.max(H, dim=1, keepdim=True)
                    H_shifted = (H - H_max) / gamma
                    exp_vals = torch.exp(H_shifted) * u_weights
                    exp_sum = torch.sum(exp_vals, dim=1, keepdim=True)
                    log_integral = gamma * torch.log(exp_sum + 1e-30) + H_max
                    V_rhs = log_integral / beta
                else:
                    H_max, _ = torch.max(H, dim=1, keepdim=True)
                    V_rhs = H_max / beta

                loss_pde = torch.mean((V - V_rhs) ** 2)
                loss_total = loss_pde
                # Merton ansatz V(x)=h*x^p/p => Vx/Vxx = x/(p-1), i.e. x*Vxx - (p-1)*Vx = 0.
                # Regularizer encourages this to reduce Vx/Vxx oscillation and stabilize mean-of-pi.
                reg_coeff = float(params_train.get("pinn_reg_vx_vxx_coeff", 0.0))
                p_val = float(getattr(model, "p"))
                if reg_coeff > 0 and abs(p_val - 1.0) > 1e-6:
                    x_safe = torch.clamp(x_col, min=1e-10)
                    residual_vx_vxx = x_safe * Vxx - (p_val - 1.0) * Vx
                    loss_vx_vxx = torch.mean(residual_vx_vxx ** 2)
                    loss_total = loss_total + reg_coeff * loss_vx_vxx
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optim_v.step()

                if (it + 1) % 200 == 0 or (it + 1) == n_iter:
                    elapsed = time.perf_counter() - t_start
                    print(
                        "[pinn-train] it %d/%d (%.1fs elapsed, ~%.1fs/iter)"
                        % (it + 1, n_iter, elapsed, elapsed / (it + 1)),
                        flush=True,
                    )

        with torch.no_grad():
            V_final, Vx_final, Vxx_final = value_net_with_derivatives(net, x_grid, x_min, x_eval_max_f)
            x_col_final = x_grid
            one_plus_final = 1.0 + z * u_grid
            x_shift_final = x_col_final * one_plus_final
            x_shift_final_flat = x_shift_final.reshape(-1, 1)
            V_shift_final_parts = []
            for start in range(0, n_shift, v_shift_chunk):
                end = min(start + v_shift_chunk, n_shift)
                V_shift_final_parts.append(value_net_forward(net, x_shift_final_flat[start:end], x_min, x_eval_max_f))
            V_shift_final = torch.cat(V_shift_final_parts, dim=0).view(n_x, n_u)
            reward_final = torch.pow(torch.clamp(x_col_final, min=1e-12), getattr(model, "p")) / getattr(model, "p")
            drift_final = (r0 + mu_minus_r * u_grid) * (x_col_final * Vx_final)
            diffusion_final = 0.5 * (sig0 * u_grid * x_col_final) ** 2 * Vxx_final
            jump_final = lam0 * (V_shift_final - V_final - z * u_grid * x_col_final * Vx_final)
            H_final = reward_final + drift_final + diffusion_final + jump_final
            if gamma.item() > 0:
                H_max_final, _ = torch.max(H_final, dim=1, keepdim=True)
                H_shifted_final = (H_final - H_max_final) / gamma
                exp_vals_final = torch.exp(H_shifted_final) * u_weights
                exp_sum_final = torch.sum(exp_vals_final, dim=1, keepdim=True)
                log_integral_final = gamma * torch.log(exp_sum_final + 1e-30) + H_max_final
                V_rhs_final = log_integral_final / beta
            else:
                H_max_final, _ = torch.max(H_final, dim=1, keepdim=True)
                V_rhs_final = H_max_final / beta
            loss_pde_final = torch.mean((V_final - V_rhs_final) ** 2)

    net.eval()
    cache = {
        "net": net,
        "x_min": float(x_min),
        # Backward-compatible key:
        "x_max": float(x_eval_max_f),
        "x_max_train": float(x_max),
        "x_max_eval": float(x_eval_max_f),
        "u_grid": u_grid.detach(),
        "u_weights": u_weights.detach(),
        "loss_pde_final": float(loss_pde_final.item()),
        "_core_hash": core_hash,
    }
    model._exact_solution_cache_nn = cache
    model._exact_solution_params_hash_nn = params_hash

    if use_cache:
        save_pinn_to_disk(model, net, pinn_params, params_hash, cache)
    return cache

