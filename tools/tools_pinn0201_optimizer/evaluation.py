"""
Evaluation and checkpoint selection for the Merton experiments.

Build fixed rollout batches and (t, x) evaluation points, compute evaluation
metrics, and manage Top-K checkpoints. Functions operate on a supplied model
instance through wrappers in the experiment entrypoints.
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from run_io import metadata_for_save
from torch.distributions import TransformedDistribution
from torch.distributions import transforms as dist_transforms


def to_float(x: Any) -> float:
    """Safe float conversion for Tensors/ndarrays/scalars."""
    try:
        if torch.is_tensor(x):
            return float(x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().mean().item())
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1).mean())
        return float(x)
    except Exception:
        return math.inf


def relative_l2_error(pred: torch.Tensor, exact: torch.Tensor, eps: float = 1e-12) -> Optional[torch.Tensor]:
    """
    Relative L2 error over all entries:
      sqrt(E[(pred-exact)^2] / (E[exact^2] + eps))
    """
    if pred is None or exact is None:
        return None
    num = (pred - exact) ** 2
    den = exact ** 2
    return torch.sqrt(num.mean() / (den.mean() + eps))


def _finite_minmax(t: Optional[torch.Tensor]) -> Tuple[Optional[float], Optional[float]]:
    if t is None:
        return None, None
    t = t.detach().reshape(-1)
    mask = torch.isfinite(t)
    if mask.any():
        t = t[mask]
        return float(t.min().item()), float(t.max().item())
    return None, None


def ensure_eval_set(model: Any, N_eval: Optional[int] = None) -> None:
    """
    Build and persist:
    1) fixed rollout batch (t_eval/W_eval/P_eval) for trajectory-based plots/debug
    2) fixed point set eval_tx for Top-K functional evaluation (same (t,x) across iters)

    Supports optional early-training refresh of eval_tx, then freeze:
      - eval_points_refresh_every (e.g. 100)
      - eval_points_refresh_until (e.g. 1000)
      - eval_points_x_source: uniform_bounds | pred_traj_sample | pred_traj_bounds
    """
    def _desired_eval_N(N_eval_arg: Optional[int]) -> Optional[int]:
        """
        Decide the time discretization used only for evaluation rollouts/plots.
        Rule:
          - if user passes N_eval: use it
          - else if training-time N < 1000: N_eval = min(1000, 10 * N)
          - else: use training-time N (no upsample)
        """
        if N_eval_arg is not None:
            try:
                return int(N_eval_arg)
            except Exception:
                return None
        try:
            N_base = int(getattr(model, "N"))
        except Exception:
            return None
        if N_base < 1000:
            return min(1000, 10 * int(N_base))
        return int(N_base)

    fixed_ready = bool(getattr(model, "_eval_fixed_ready", False))
    points_ready = bool(getattr(model, "_eval_points_ready", False))

    # If a fixed eval rollout batch already exists but with a different N than desired, force regeneration.
    desired_N = _desired_eval_N(N_eval)
    if fixed_ready and desired_N is not None:
        try:
            t_eval = getattr(model, "t_eval", None)
            if t_eval is not None and hasattr(t_eval, "shape") and len(t_eval.shape) >= 2:
                N_curr = int(t_eval.shape[1] - 1)
                if N_curr != int(desired_N):
                    setattr(model, "_eval_fixed_ready", False)
                    fixed_ready = False
        except Exception:
            pass

    if fixed_ready and points_ready:
        return

    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_states = None
    np_state = np.random.get_state()
    # Optional: allow eval rollouts to use more trajectories than training.
    # This is purely for plotting/diagnostics and helps reduce Monte-Carlo noise
    # in time-step-wise mean curves (e.g. plot_control_mean_kind="traj"/"policy_mean").
    eval_M = None
    try:
        params_train = getattr(model, "params_train", {}) or {}
        eval_M_raw = params_train.get("eval_M", None)
        if eval_M_raw is not None:
            eval_M = int(eval_M_raw)
            if eval_M <= 0:
                eval_M = None
    except Exception:
        eval_M = None

    try:
        # Respect training-time reproducibility knobs if present.
        # - If eval_rollout_use_fixed_seed=True, force a deterministic eval batch.
        # - Otherwise, sample using the current RNG state (but still restore RNG afterward).
        params_train = getattr(model, "params_train", {}) or {}
        use_fixed = bool(params_train.get("eval_rollout_use_fixed_seed", True))
        seed = int(params_train.get("eval_rollout_fixed_seed", 123456))
        if use_fixed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                try:
                    torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass
            np.random.seed(seed)
        with torch.no_grad():
            # --- fixed rollout batch ---
            if not fixed_ready:
                N_original = int(getattr(model, "N"))
                M_original = None
                try:
                    M_original = int(getattr(model, "M"))
                except Exception:
                    M_original = None

                # OOM guard: cap total number of rollout states used for evaluation.
                # total_states ~= M * (N + 1).
                # This prevents accidental N upsampling (e.g. N->1000) combined with eval_M (e.g. 1000)
                # from creating million-state tensors that can OOM downstream (actor/critic/entropy eval).
                params_train = getattr(model, "params_train", {}) or {}
                eval_total_state_cap = params_train.get("eval_total_state_cap", None)
                if eval_total_state_cap is None:
                    eval_total_state_cap = params_train.get("eval_total_state_count_cap", 300_000)
                try:
                    eval_total_state_cap = int(eval_total_state_cap) if eval_total_state_cap is not None else None
                except Exception:
                    eval_total_state_cap = None
                if eval_total_state_cap is not None and eval_total_state_cap <= 0:
                    eval_total_state_cap = None

                N_build = int(desired_N) if desired_N is not None else int(N_original)
                M_build = None
                if eval_M is not None:
                    M_build = int(eval_M)
                elif M_original is not None:
                    M_build = int(M_original)

                if eval_total_state_cap is not None and M_build is not None:
                    total_states = int(M_build) * int(N_build + 1)
                    if total_states > int(eval_total_state_cap):
                        # Prefer reducing M first (keeps temporal resolution for plots).
                        M_new = max(1, int(eval_total_state_cap // int(N_build + 1)))
                        if M_new < int(M_build):
                            print(
                                f"[eval] OOM guard: shrinking eval rollout batch to respect eval_total_state_cap="
                                f"{int(eval_total_state_cap)}: M {int(M_build)} -> {int(M_new)} "
                                f"(N={int(N_build)}). Override by setting train.eval_total_state_cap."
                            )
                            M_build = int(M_new)
                        # If still too large (cap smaller than one full trajectory), reduce N as well.
                        total_states2 = int(M_build) * int(N_build + 1)
                        if total_states2 > int(eval_total_state_cap):
                            N_new = max(1, int(eval_total_state_cap // int(M_build) - 1))
                            if N_new < int(N_build):
                                print(
                                    f"[eval] OOM guard: shrinking eval N to respect eval_total_state_cap="
                                    f"{int(eval_total_state_cap)}: N {int(N_build)} -> {int(N_new)} "
                                    f"(M={int(M_build)}). Override by setting train.eval_total_state_cap."
                                )
                                N_build = int(N_new)

                # Apply eval-time overrides (restored after building eval batch below).
                if int(N_build) != int(N_original):
                    model.N = int(N_build)
                # if N_eval is not None:
                #         print(f"[eval] setting N_eval={int(desired_N)} for evaluation (original N={N_original})")
                # else:
                #     print(f"[eval] auto setting N_eval={int(desired_N)} for evaluation (original N={N_original})")

                # Temporarily override M for eval rollouts if requested (or by OOM guard).
                if M_build is not None and M_original is not None and int(M_build) != int(M_original):
                    try:
                        model.M = int(M_build)
                    except Exception:
                        pass

                t, W, P = model.fetch_minibatch()
                model.jump_index_eval = getattr(model, "jump_index", None)
                try:
                    X0_eval = model._init_X0(M=int(t.shape[0]))
                except Exception:
                    X0_eval = model._init_X0()

                # Restore training-time N/M after building the eval batch.
                model.N = N_original
                if M_original is not None:
                    try:
                        model.M = int(M_original)
                    except Exception:
                        pass
                model.t_eval, model.W_eval, model.P_eval = t, W, P
                model.X0_eval = X0_eval

                # Optional: fixed eps for predict-mode rollouts (plot comparability)
                try:
                    M_eval = int(t.shape[0])
                    D = int(getattr(model, "D"))
                    T_steps = int(t.shape[1])
                    eps_list = []
                    for _ in range(T_steps):
                        eps = torch.randn(M_eval, D, device=getattr(model, "device"), dtype=torch.float32)
                        eps_list.append({"noise": eps})
                    model.eps_eval = eps_list
                except Exception as e:
                    model.eps_eval = None
                    print(f"[warn] failed to build fixed eps_eval: {e}")

                model._eval_fixed_ready = True
                fixed_ready = True

            # --- fixed/refreshable (t,x) point set for Top-K ---
            refresh_every = int(getattr(model, "params_train", {}).get("eval_points_refresh_every", 0) or 0)
            refresh_until = int(getattr(model, "params_train", {}).get("eval_points_refresh_until", 0) or 0)
            it_now = getattr(model, "_eval_iteration", None)
            last_refresh = getattr(model, "_eval_points_last_refresh_it", None)

            need_refresh = (not points_ready)
            if (not need_refresh) and refresh_every > 0 and refresh_until > 0 and it_now is not None:
                if int(it_now) < refresh_until:
                    if last_refresh is None or (int(it_now) - int(last_refresh)) >= refresh_every:
                        need_refresh = True

            if need_refresh:
                params_train = getattr(model, "params_train", {})
                B = max(128, int(params_train.get("eval_num_points", 2048)))
                D = int(getattr(model, "D"))
                x_source = str(params_train.get("eval_points_x_source", "uniform_bounds")).lower()

                # Handle explicit None (JSON null): get() won't fall back
                x_lo_raw = params_train.get("pinn_x_min_fixed", None)
                if x_lo_raw is None:
                    x_lo_raw = params_train.get("pinn_x_min", 1e-5)
                x_hi_raw = params_train.get("pinn_x_max_fixed", None)
                if x_hi_raw is None:
                    x_hi_raw = params_train.get("pinn_x_max", 80.0)
                x_lo = float(x_lo_raw)
                x_hi = float(x_hi_raw)
                if not (math.isfinite(x_lo) and math.isfinite(x_hi) and (x_hi > x_lo)):
                    x_lo, x_hi = 1e-5, 80.0

                t_pts = torch.rand(B, 1, device=getattr(model, "device"), dtype=torch.float32) * float(getattr(model, "T"))

                x_pts = None
                # Option: tie early refresh to current predicted trajectory distribution
                if x_source in ("pred_traj_sample", "pred_traj_bounds") and it_now is not None:
                    try:
                        model.t_predict_set = model.t_eval
                        model.W_predict_set, model.P_predict_set = model.W_eval, model.P_eval
                        if getattr(model, "eps_eval", None) is not None:
                            model.eps_predict = model.eps_eval
                        X_pred, *_rest = model.train_loop(itr=-1, traj_only=True, predict=True, deterministic_eval=False)
                        if X_pred is not None and torch.is_tensor(X_pred) and X_pred.numel() > 0:
                            X_flat = X_pred.detach().reshape(-1, D)
                            mask = torch.isfinite(X_flat).all(dim=1)
                            X_flat = X_flat[mask]
                            if X_flat.numel() > 0:
                                if x_source == "pred_traj_sample":
                                    idx = torch.randint(0, X_flat.shape[0], (B,), device=getattr(model, "device"))
                                    x_pts = X_flat[idx]
                                else:
                                    x_min_obs = float(X_flat.min().detach().cpu().item())
                                    x_max_obs = float(X_flat.max().detach().cpu().item())
                                    span = max(1e-12, x_max_obs - x_min_obs)
                                    pad_frac = float(params_train.get("eval_points_pad_frac", 0.1))
                                    pad_abs = params_train.get("eval_points_pad_abs", None)
                                    pad = float(pad_abs) if pad_abs is not None else (pad_frac * span)
                                    lo = max(x_lo, x_min_obs - pad)
                                    hi = min(x_hi, x_max_obs + pad)
                                    if hi > lo:
                                        x_pts = torch.rand(B, D, device=getattr(model, "device"), dtype=torch.float32) * (hi - lo) + lo
                    except Exception as e:
                        print(f"[warn] eval_tx x_source={x_source} refresh failed: {e}")

                if x_pts is None:
                    x_pts = torch.rand(B, D, device=getattr(model, "device"), dtype=torch.float32) * (x_hi - x_lo) + x_lo

                x_pts = torch.clamp(x_pts, min=1e-8)
                model.eval_tx = torch.cat([t_pts, x_pts], dim=1)
                model._eval_points_ready = True
                model._eval_points_last_refresh_it = (None if it_now is None else int(it_now))
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass
        np.random.set_state(np_state)


@torch.no_grad()
def evaluate_on_fixed(model: Any) -> Tuple[float, Dict[str, Any]]:
    """Forward-only evaluation using the model's own traj-based `train_loop` error outputs."""
    ensure_eval_set(model)
    model.t_predict_set = model.t_eval
    model.W_predict_set, model.P_predict_set = model.W_eval, model.P_eval

    _, _, _, _, _, Y_error, control_error, _, _, _ = model.train_loop(itr=-1, traj_only=True, predict=True)
    score = to_float(Y_error)
    if control_error is not None:
        score = 0.5 * (score + to_float(control_error))
    return score, {"Y_error": to_float(Y_error), "control_error": (None if control_error is None else to_float(control_error))}


def _actor_input_dim(model: Any) -> Optional[int]:
    """Infer actor expected state dimension from the first linear layer (base.hidden[0].in_features)."""
    try:
        net = getattr(model, "net_actor", None)
        if net is None:
            return None
        base = getattr(net, "base", None)
        if base is not None and hasattr(base, "hidden"):
            first = base.hidden[0]
            if isinstance(first, torch.nn.Linear):
                return first.in_features
    except Exception:
        pass
    return None


def _state_for_actor(model: Any, state_feat: torch.Tensor, t_raw: torch.Tensor, x_only: torch.Tensor) -> torch.Tensor:
    """
    Build state tensor for net_actor so that its feature dim matches the actor's expected input.
    When time_independent_actor=True the actor expects x-only (e.g. 1 dim); otherwise (t,x) (e.g. 2 dim).
    Uses _preprocess_state_for_actor when available; otherwise infers from actor's first layer.
    """
    if hasattr(model, "_preprocess_state_for_actor"):
        try:
            out = model._preprocess_state_for_actor(t_raw, x_only)
            if out.shape[-1] == _actor_input_dim(model):
                return out
        except Exception:
            pass
    actor_in = _actor_input_dim(model)
    if actor_in is None:
        return state_feat
    feat_dim = state_feat.shape[-1]
    if feat_dim == actor_in:
        return state_feat
    if feat_dim > actor_in:
        # Actor expects x-only: drop first (t) column, keep last actor_in columns.
        return state_feat[..., -actor_in:]
    return state_feat


@torch.no_grad()
def evaluate_pinn_taylor_errors(model: Any) -> Dict[str, Any]:
    """
    Functional evaluation on fixed point set `eval_tx`:
      - value:  V_hat(t,x) vs V*(x) (PINN/Taylor)
      - policy: pi_hat(t,x) vs u*(x)  (PINN/Taylor)

    Returns a dict with at least:
      pinn_value_error, pinn_control_error, taylor_value_error, taylor_control_error,
      x_min, x_max, u_pinn_min/max, u_taylor_min/max, v_pinn_mean, y_pred_mean
    Also returns:
      pinn_mu_gaussian_policy_error, taylor_mu_gaussian_policy_error (when computable)
    """
    ensure_eval_set(model)
    if getattr(model, "eval_tx", None) is None:
        return {
            "pinn_value_error": None,
            "pinn_control_error": None,
            "pinn_mu_gaussian_policy_error": None,
            "taylor_value_error": None,
            "taylor_control_error": None,
            "taylor_mu_gaussian_policy_error": None,
            "x_min": None,
            "x_max": None,
            "u_pinn_min": None,
            "u_pinn_max": None,
            "u_taylor_min": None,
            "u_taylor_max": None,
            "v_pinn_mean": None,
            "y_pred_mean": None,
        }

    # Make reproducible: mixture_mean uses sampling
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_states = None
    np_state = np.random.get_state()
    try:
        params_train = getattr(model, "params_train", {}) or {}
        use_fixed = bool(params_train.get("eval_rollout_use_fixed_seed", True))
        seed = int(params_train.get("eval_rollout_fixed_seed", 123456))
        if use_fixed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                try:
                    torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass
            np.random.seed(seed)
        # eval_tx is stored in RAW coordinates: (t, x).
        # If the training pipeline uses state preprocessing, we must apply the SAME transform
        # before feeding actor/critic, while keeping x_only raw for PINN/Taylor comparisons.
        state_flat_raw = model.eval_tx.to(device=getattr(model, "device"), dtype=torch.float32)  # [B, 1+D]
        t_raw = state_flat_raw[:, :1]
        x_only = state_flat_raw[:, 1:]
        state_feat = state_flat_raw
        if hasattr(model, "_preprocess_state"):
            try:
                state_feat = model._preprocess_state(t_raw, x_only)
            except Exception:
                # Robust fallback: if signature/behavior differs across entrypoints, keep raw state.
                state_feat = state_flat_raw

        # Actor may expect x-only when time_independent_actor=True; use model's actor state.
        # Prefer _preprocess_state_for_actor; fallback: infer from actor's first layer in_features.
        state_feat_actor = _state_for_actor(model, state_feat, t_raw, x_only)

        V_hat = model.net_critic(state_feat)
        params_train = getattr(model, "params_train", {})
        K_mean_eval = int(params_train.get("K_mean_eval", 20))
        actor_type = getattr(getattr(model, "net_actor", None), "actor_type", "")
        if hasattr(model.net_actor, "mixture_mean") and actor_type in (
            "mog_squash",
            "hyper_gaussian",
            "hyper_gaussian_split",
            "gaussian_two_stage",
        ):
            # Force deterministic MC mean (independent of any prior RNG usage).
            params_train = getattr(model, "params_train", {})
            mm_seed = None
            if bool(params_train.get("mixture_mean_use_fixed_seed", True)):
                mm_seed = int(params_train.get("mixture_mean_fixed_seed", 123456))
            mm_kwargs = dict(n_samples=K_mean_eval, antithetic=True)
            if mm_seed is not None:
                mm_kwargs["seed"] = mm_seed
            try:
                pi_hat = model.net_actor.mixture_mean(state_feat_actor, **mm_kwargs)
            except TypeError as e:
                # Backward-compat: some actors may not accept `seed` kwarg.
                if "seed" in str(e) and "unexpected keyword argument" in str(e):
                    mm_kwargs.pop("seed", None)
                    pi_hat = model.net_actor.mixture_mean(state_feat_actor, **mm_kwargs)
                else:
                    raise
        else:
            mu_pred, _, _ = model.net_actor(state_feat_actor)
            pi_hat = mu_pred
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass
        np.random.set_state(np_state)

    x_min, x_max = _finite_minmax(x_only)

    gamma_entropy = float(getattr(model, "params_train", {}).get("entropy_alpha", 0.0))
    if gamma_entropy == 0:
        u_pinn, v_pinn = model.compute_exact_no_exploratory(x_only)
    else:
        u_mean, _, v_pinn, _u_hard, _pol = model.compute_exact_entropy_with_cache(x_only, return_u_hard=True)
        u_pinn = u_mean

    pinn_value_error = relative_l2_error(V_hat, v_pinn)
    pinn_control_error = relative_l2_error(pi_hat, u_pinn)
    pinn_mu_gaussian_policy_error = None
    try:
        pinn_mu_gaussian_policy_error = torch.mean((pi_hat - u_pinn) ** 2 / (u_pinn ** 2 + 1e-8))
    except Exception:
        pinn_mu_gaussian_policy_error = None

    v_pinn_mean = to_float(v_pinn.mean())
    y_pred_mean = to_float(V_hat.mean())
    u_pinn_min, u_pinn_max = _finite_minmax(u_pinn)

    taylor_value_error = None
    taylor_control_error = None
    taylor_mu_gaussian_policy_error = None
    u_taylor_min, u_taylor_max = None, None
    try:
        policy_taylor, v_taylor = model.compute_exact_exploratory_taylor_and_ansatz(x_only)
        u_taylor = policy_taylor.loc
        taylor_value_error = relative_l2_error(V_hat, v_taylor)
        taylor_control_error = relative_l2_error(pi_hat, u_taylor)
        try:
            taylor_mu_gaussian_policy_error = torch.mean((pi_hat - u_taylor) ** 2 / (u_taylor ** 2 + 1e-8))
        except Exception:
            taylor_mu_gaussian_policy_error = None
        u_taylor_min, u_taylor_max = _finite_minmax(u_taylor)
    except Exception as e:
        print(f"[warn] taylor eval failed: {e}")

    return {
        "pinn_value_error": (None if pinn_value_error is None else to_float(pinn_value_error)),
        "pinn_control_error": (None if pinn_control_error is None else to_float(pinn_control_error)),
        "pinn_mu_gaussian_policy_error": (None if pinn_mu_gaussian_policy_error is None else to_float(pinn_mu_gaussian_policy_error)),
        "taylor_value_error": (None if taylor_value_error is None else to_float(taylor_value_error)),
        "taylor_control_error": (None if taylor_control_error is None else to_float(taylor_control_error)),
        "taylor_mu_gaussian_policy_error": (None if taylor_mu_gaussian_policy_error is None else to_float(taylor_mu_gaussian_policy_error)),
        "x_min": x_min,
        "x_max": x_max,
        "u_pinn_min": u_pinn_min,
        "u_pinn_max": u_pinn_max,
        "u_taylor_min": u_taylor_min,
        "u_taylor_max": u_taylor_max,
        "v_pinn_mean": v_pinn_mean,
        "y_pred_mean": y_pred_mean,
    }


def eval_ckpt_unique_paths(model: Any, it: int, tag: str = "eval") -> Tuple[str, str]:
    actor_path = os.path.join(model.output_folder, f"actor_{tag}_it{it}.pth")
    critic_path = os.path.join(model.output_folder, f"critic_{tag}_it{it}.pth")
    return actor_path, critic_path


def persist_topN_alias(model: Any, topk_list: list, tag: str = "eval") -> None:
    topk_list.sort(key=lambda d: d["score"])

    try:
        for p in glob.glob(os.path.join(model.output_folder, f"actor_{tag}_top*.pth")):
            os.remove(p)
        for p in glob.glob(os.path.join(model.output_folder, f"critic_{tag}_top*.pth")):
            os.remove(p)
    except Exception as e:
        print(f"[warn] cleanup old top alias files failed: {e}")

    for rank, rec in enumerate(topk_list[:3], start=1):
        itr = rec["iter"]
        a_dst = os.path.join(model.output_folder, f"actor_{tag}_top{rank}_itr{itr}.pth")
        c_dst = os.path.join(model.output_folder, f"critic_{tag}_top{rank}_itr{itr}.pth")
        try:
            shutil.copyfile(rec["actor"], a_dst)
            shutil.copyfile(rec["critic"], c_dst)
        except Exception as e:
            print(f"[warn] copying top{rank} aliases failed: {e}")

    try:
        best = topk_list[0]
        shutil.copyfile(best["actor"], os.path.join(model.output_folder, f"actor_{tag}_top1.pth"))
        shutil.copyfile(best["critic"], os.path.join(model.output_folder, f"critic_{tag}_top1.pth"))
    except Exception as e:
        print(f"[warn] write top1 quick alias failed: {e}")

    meta_path = os.path.join(model.output_folder, f"{tag}_topk_meta.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata_for_save(topk_list[:3], model.output_folder), f, indent=2)
    except Exception as e:
        print(f"[warn] write meta failed: {e}")


def maybe_update_top3_eval(model: Any, it: int, score: float, tag: str = "eval", topk_list=None) -> None:
    if topk_list is None:
        topk_list = model.topk_eval

    if len(topk_list) < 3 or score < max(topk_list, key=lambda d: d["score"])["score"]:
        a_path, c_path = eval_ckpt_unique_paths(model, it, tag=tag)
        try:
            torch.save(model.net_actor.state_dict(), a_path)
            torch.save(model.net_critic.state_dict(), c_path)
        except Exception as e:
            print(f"[warn] saving eval ckpt failed: {e}")
            return

        entry = {"iter": int(it), "score": float(score), "actor": a_path, "critic": c_path}

        if len(topk_list) < 3:
            topk_list.append(entry)
        else:
            worst_idx = max(range(3), key=lambda i: topk_list[i]["score"])
            old = topk_list[worst_idx]
            topk_list[worst_idx] = entry
            try:
                if os.path.isfile(old["actor"]):
                    os.remove(old["actor"])
                if os.path.isfile(old["critic"]):
                    os.remove(old["critic"])
            except Exception as e:
                print(f"[warn] cleanup old eval ckpt failed: {e}")

        persist_topN_alias(model, topk_list=topk_list, tag=tag)
        print(f"[eval:{tag}] accepted to Top-3: iter={it}, score={score:.6g}")
    else:
        print(f"[eval:{tag}] not better than current Top-3: iter={it}, score={score:.6g}")


def load_best_eval_for_plot(model: Any, tag: str = "pinn") -> None:
    """Load actor/critic from actor_{tag}_top1.pth/critic_{tag}_top1.pth unless explicit paths were provided."""
    if (getattr(model, "actor_path", None) and os.path.isfile(model.actor_path)) or (
        getattr(model, "critic_path", None) and os.path.isfile(model.critic_path)
    ):
        print("[load] using explicitly provided actor/critic paths; skip eval top1 lookup.")
        model.net_actor.eval()
        model.net_critic.eval()
        try:
            loaded_params = model._load_params(model.output_folder)
            if loaded_params is not None:
                model._update_params_from_dict(loaded_params)
        except Exception:
            pass
        return


    a_top1 = os.path.join(model.output_folder, f"actor_{tag}_top1.pth")
    c_top1 = os.path.join(model.output_folder, f"critic_{tag}_top1.pth")

    if os.path.isfile(a_top1) and os.path.isfile(c_top1):
        try:
            model.net_actor.load_state_dict(torch.load(a_top1, map_location=model.device))
            model.net_critic.load_state_dict(torch.load(c_top1, map_location=model.device))
            model.net_actor.eval()
            model.net_critic.eval()
            print(f"[load] using {tag} top1 ckpt for plotting.")
            loaded_params = model._load_params(model.output_folder)
            if loaded_params is not None:
                model._update_params_from_dict(loaded_params)
        except Exception as e:
            print(f"[warn] failed to load {tag} top1 ckpt: {e}")
    else:
        print(f"[info] no {tag} top1 ckpt found; using in-memory weights.")
        try:
            if (not (getattr(model, "actor_path", None) or getattr(model, "critic_path", None))) and (
                getattr(model, "output_folder", "") == "" or not os.path.isdir(model.output_folder)
            ):
                model.output_folder = model._default_output_folder()
                os.makedirs(model.output_folder, exist_ok=True)
        except Exception:
            pass


# =========================
# Distribution / policy metrics (shared)
# =========================


def kl_divergence_gaussian(mu_pred: torch.Tensor, std_pred: torch.Tensor, mu_exact: torch.Tensor, std_exact: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between two diagonal Gaussians, averaged over all entries.
    Accepts shapes [N,L,D] or [L,D] and auto-expands to 3D.
    """
    if mu_pred.dim() == 2:
        mu_pred = mu_pred.unsqueeze(0)
        std_pred = std_pred.unsqueeze(0)
        mu_exact = mu_exact.unsqueeze(0)
        std_exact = std_exact.unsqueeze(0)

    _N, _L, D = mu_pred.shape
    var_pred = std_pred ** 2
    var_exact = std_exact ** 2

    log_det_ratio = torch.sum(torch.log(var_exact + 1e-12) - torch.log(var_pred + 1e-12), dim=-1)
    tr_term = torch.sum(var_pred / (var_exact + 1e-12), dim=-1)
    mu_diff = mu_exact - mu_pred
    quad_term = torch.sum((mu_diff ** 2) / (var_exact + 1e-12), dim=-1)

    kl = 0.5 * (log_det_ratio - D + tr_term + quad_term)
    return kl.mean()


def wasserstein_distance_gaussian(
    mu_pred: torch.Tensor, std_pred: torch.Tensor, mu_exact: torch.Tensor, std_exact: torch.Tensor
) -> torch.Tensor:
    """
    Squared 2-Wasserstein distance for diagonal Gaussians:
      ||mu1-mu2||^2 + ||sigma1-sigma2||^2
    averaged over all entries.
    """
    if mu_pred.dim() == 2:
        mu_pred = mu_pred.unsqueeze(0)
        std_pred = std_pred.unsqueeze(0)
        mu_exact = mu_exact.unsqueeze(0)
        std_exact = std_exact.unsqueeze(0)

    mu_diff = mu_exact - mu_pred
    mu_term = torch.sum(mu_diff ** 2, dim=-1)

    std_diff = std_exact - std_pred
    std_term = torch.sum(std_diff ** 2, dim=-1)

    w2_sq = mu_term + std_term
    return w2_sq.mean()


def policy_cdf(policy_exact: Any, u: torch.Tensor) -> torch.Tensor:
    """
    Robust CDF for:
    - Normal
    - TransformedDistribution with Softplus + Affine
    - DiscretePolicy-like objects exposing `.cdf`, `.support_u`, `.probs`
    """
    if hasattr(policy_exact, "cdf") and hasattr(policy_exact, "support_u") and hasattr(policy_exact, "probs"):
        return policy_exact.cdf(u)

    if isinstance(policy_exact, TransformedDistribution):
        base = policy_exact.base_dist
        transforms = policy_exact.transforms
        u_in = u
        if u_in.dim() == 1 and hasattr(base, "loc") and base.loc.dim() == 2 and base.loc.size(1) == 1:
            u_in = u_in.unsqueeze(-1)

        u_shift = 0.0
        if len(transforms) > 0 and isinstance(transforms[-1], dist_transforms.AffineTransform):
            u_shift = transforms[-1].loc

        u_adj = u_in - u_shift
        u_adj = torch.nan_to_num(u_adj, nan=0.0, posinf=0.0, neginf=0.0)
        z = torch.log(torch.expm1(torch.clamp(u_adj, min=1e-12)))
        z = torch.where(u_adj > 0.0, z, torch.tensor(float("-inf"), device=u_adj.device, dtype=u_adj.dtype))
        return base.cdf(z)

    if u.dim() == 2 and hasattr(policy_exact, "loc") and policy_exact.loc.dim() == 2 and policy_exact.loc.size(-1) == 1:
        u = u.unsqueeze(-1)
    elif u.dim() == 1 and hasattr(policy_exact, "loc") and policy_exact.loc.dim() == 2 and policy_exact.loc.size(-1) == 1:
        u = u.unsqueeze(-1)
    return policy_exact.cdf(u)


def cvm_error_from_samples(control: torch.Tensor, policy_exact: Any) -> torch.Tensor:
    """
    Cramér–von Mises error comparing empirical CDF of samples (control) vs exact policy CDF.

    Args:
      control: [M,1] or [B,M,1] or [M] or [B,M]
      policy_exact: distribution with `.cdf` or TransformedDistribution / DiscretePolicy-like

    Returns:
      scalar CVM error (lower is better)
    """
    u = control.squeeze(-1) if control.dim() > 1 and control.size(-1) == 1 else control

    if u.dim() == 1:
        p = policy_cdf(policy_exact, u)
        p_sorted, _ = torch.sort(p.reshape(-1))
        M = p_sorted.numel()
        i = torch.arange(1, M + 1, device=p.device, dtype=p.dtype)
        grid = (2 * i - 1) / (2 * M)
        cvm = ((p_sorted - grid) ** 2).mean() + (1.0 / (12.0 * M))
        return cvm

    if u.dim() != 2:
        u = u.view(-1, u.size(-1))
    p = policy_cdf(policy_exact, u)
    if p.dim() == 3 and p.size(-1) == 1:
        p = p.squeeze(-1)
    p_sorted, _ = torch.sort(p, dim=1)
    _B, M = p_sorted.size()
    i = torch.arange(1, M + 1, device=p.device, dtype=p.dtype).view(1, -1)
    grid = (2 * i - 1) / (2 * M)
    cvm_per_batch = ((p_sorted - grid) ** 2).mean(dim=1) + (1.0 / (12.0 * M))
    return cvm_per_batch.mean()


def compute_control_error_with_fallback(
    model: Any,
    control: torch.Tensor,
    policy_exact: Any,
    mu_pred: torch.Tensor,
    sigma_pred: torch.Tensor,
    mu_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
    M: int,
    D: int,
    control_samples: Optional[torch.Tensor] = None,
    *,
    actor_type_u_space: Tuple[str, ...] = ("hyper_gaussian", "hyper_gaussian_split", "gaussian_two_stage"),
    skip_actor_types: Tuple[str, ...] = ("mog_squash",),
) -> Optional[torch.Tensor]:
    """
    Compute a scalar-ish control distance with OOM-safe fallbacks.
    This is used inside training loops to define `control_error_step`.

    NOTE:
    - Returns a Tensor shaped like [M] when possible (by expanding scalar metrics)
    - Returns None if no suitable metric is available for the actor type / dimension
    """
    control_samples = control if control_samples is None else control_samples

    # Prefer distribution-based distance when possible (1D): CVM against exact policy CDF
    if D == 1:
        try:
            control_error_step = cvm_error_from_samples(control_samples.unsqueeze(0), policy_exact)
            control_error_step = control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
            return control_error_step
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                raise

    actor_type = getattr(getattr(model, "net_actor", None), "actor_type", "")
    if actor_type in skip_actor_types:
        return None

    # Some actor types: compare in u-space using MC mean/std against exact Normal(u)
    if actor_type in actor_type_u_space:
        u_mean = control_samples.mean(dim=0, keepdim=True)
        u_std = control_samples.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        u_exact_mean = mu_exact.mean(dim=0, keepdim=True)
        u_exact_std = sigma_exact.mean(dim=0, keepdim=True).clamp_min(1e-8)
        try:
            control_error_step = kl_divergence_gaussian(
                u_mean.unsqueeze(0), u_std.unsqueeze(0),
                u_exact_mean.unsqueeze(0), u_exact_std.unsqueeze(0),
            )
            control_error_step = control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
            return control_error_step
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                raise

        # Fallback to W2 in u-space
        try:
            control_error_step = wasserstein_distance_gaussian(
                u_mean.unsqueeze(0), u_std.unsqueeze(0),
                u_exact_mean.unsqueeze(0), u_exact_std.unsqueeze(0),
            )
            control_error_step = control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
            return control_error_step
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                raise
        return None

    # Strategy 1: KL divergence (all dimensions)
    try:
        control_error_step = kl_divergence_gaussian(
            mu_pred.unsqueeze(0), sigma_pred.unsqueeze(0),
            mu_exact.unsqueeze(0), sigma_exact.unsqueeze(0),
        )
        control_error_step = control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
        return control_error_step
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Strategy 2: W2 distance
            try:
                control_error_step = wasserstein_distance_gaussian(
                    mu_pred.unsqueeze(0), sigma_pred.unsqueeze(0),
                    mu_exact.unsqueeze(0), sigma_exact.unsqueeze(0),
                )
                control_error_step = control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
                return control_error_step
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # Strategy 3: for 1D, last try CVM
                    if D == 1:
                        try:
                            control_error_step = cvm_error_from_samples(control.unsqueeze(0), policy_exact)
                            control_error_step = (
                                control_error_step.expand(M) if control_error_step.numel() == 1 else control_error_step
                            )
                            return control_error_step
                        except RuntimeError as e3:
                            if "out of memory" in str(e3).lower():
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                return None
                            raise
                    return None
                raise
        raise

