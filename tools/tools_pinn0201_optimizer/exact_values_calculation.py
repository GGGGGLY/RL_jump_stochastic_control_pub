"""
Analytical, Taylor/ansatz, and PINN-based evaluation benchmarks for Merton problems.

Compute reference values and policies using the supplied model's parameters
and PINN helpers. This module does not import the training entrypoints.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import fsolve
from torch.distributions import Normal


class DiscretePolicy:
    """Discrete policy on u_grid with per-state probabilities."""

    def __init__(self, support_u: torch.Tensor, probs: torch.Tensor):
        # support_u: [n_u], probs: [B, n_u]
        self.support_u = support_u
        self.probs = probs

    @property
    def loc(self) -> torch.Tensor:
        """Compatibility: treat discrete mean as 'loc'."""
        # [B, 1]
        m = (self.probs * self.support_u.view(1, -1)).sum(dim=1, keepdim=True)
        return m

    @property
    def scale(self) -> torch.Tensor:
        """Compatibility: treat discrete std as 'scale'."""
        u = self.support_u.view(1, -1)
        m = (self.probs * u).sum(dim=1, keepdim=True)
        m2 = (self.probs * (u**2)).sum(dim=1, keepdim=True)
        var = torch.clamp(m2 - m**2, min=0.0)
        return torch.sqrt(var)

    def sample(self, n: int = 1) -> torch.Tensor:
        idx = torch.multinomial(self.probs, num_samples=n, replacement=True)  # [B, n]
        return self.support_u[idx]

    def icdf(self, q: torch.Tensor) -> torch.Tensor:
        """
        Inverse-CDF (quantile) for the discrete distribution.
        Useful for sampling with shared noise across methods:
            eps ~ N(0,1) -> q=Phi(eps) -> u = icdf(q)

        Args:
            q: [B,1] or [B] (values in [0,1])
        Returns:
            u: [B,1]
        """
        if q.dim() == 1:
            q_b = q.view(-1, 1)
        else:
            q_b = q.reshape(-1, 1)
        q_b = torch.clamp(q_b, 0.0, 1.0)

        cdf_grid = torch.cumsum(self.probs, dim=1)  # [B, n_u]
        # idx = first index where CDF >= q  (equivalently count of CDF < q)
        idx = torch.sum(cdf_grid < q_b, dim=1, keepdim=True)  # [B,1]
        idx = torch.clamp(idx, 0, self.support_u.numel() - 1)
        support = self.support_u.view(1, -1).expand(q_b.size(0), -1)  # [B, n_u]
        return torch.gather(support, 1, idx)

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        # u: [M] or [B,M]
        support = self.support_u
        cdf_grid = torch.cumsum(self.probs, dim=1)
        if u.dim() == 1:
            u_b = u.unsqueeze(0)
        else:
            u_b = u
        idx = torch.searchsorted(support, u_b, right=True) - 1
        idx = torch.clamp(idx, min=-1, max=support.numel() - 1)
        idx_safe = torch.clamp(idx, min=0)
        out = torch.gather(cdf_grid, 1, idx_safe)
        out = torch.where(idx < 0, torch.zeros_like(out), out)
        return out.squeeze(0) if u.dim() == 1 else out


def compute_exact_exploratory_taylor_and_ansatz(self, x):
    """
    Approximate exploratory solution based on the Taylor/ansatz value:
    V(x) = h * x^p / p + a * log(x) + g.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.get_default_dtype())
    device, dtype = x.device, x.dtype
    mu = torch.as_tensor(self.params_equ["mu"], device=device, dtype=dtype)
    r = torch.as_tensor(self.params_equ["r"], device=device, dtype=dtype)
    sigma = torch.as_tensor(self.params_equ["sigma"], device=device, dtype=dtype)
    lam = torch.as_tensor(self.params_equ["lambda"], device=device, dtype=dtype)
    z = torch.as_tensor(self.params_equ["z"], device=device, dtype=dtype)
    gamma_entropy = torch.as_tensor(self.params_train["entropy_alpha"], device=device, dtype=dtype)

    if x.dim() == 3 and x.size(-1) == 1 and x.size(0) == x.size(1):
        x_body = x.squeeze(-1)
    else:
        x_body = x

    B = sigma**2 + lam * z**2
    if self.p == 1 / 2:
        x_body = torch.clamp(x_body, min=1e-8)
    C = self.h_t.to(self.device) * (x_body**self.p)

    div_term = B * ((1 - self.p) * C + self.a_t)
    div_term = torch.clamp(div_term, min=1e-8)
    mu_u = ((mu - r) * (C + self.a_t)) / div_term
    sigma_u = gamma_entropy / div_term
    scale = torch.sqrt(torch.clamp(sigma_u, min=1e-8))

    optimal_policy = Normal(loc=mu_u, scale=scale)
    optimal_value = (
        (self.h_t.to(self.device) / self.p) * (x_body**self.p)
        + self.a_t.to(self.device) * torch.log(x_body + 1e-8)
        + self.g_t.to(self.device)
    )

    if x.dim() == 3 and x.size(-1) == 1 and x.size(0) == x.size(1):
        optimal_policy = Normal(loc=optimal_policy.loc.unsqueeze(-1), scale=optimal_policy.scale.unsqueeze(-1))
        optimal_value = optimal_value.unsqueeze(-1)

    return optimal_policy, optimal_value


def compute_exact_entropy_with_cache(
    self,
    x,
    return_cache: bool = False,
    return_u_hard: bool = False,
    return_policy: bool = False,
    *,
    pinn_cfg_defaults: dict | None = None,
    display_gamma_for_policy: float | None = None,
):
    """
    One-shot exact (PINN) evaluation:
    - trains/reuses PINN once
    - returns u_mean(x), u_std(x), V(x)
    Shapes follow compute_exact_entropy convention.

    When return_policy=True and display_gamma_for_policy is set (e.g. 0.5), the
    returned discrete policy probs are computed with that gamma instead of
    entropy_alpha, so the plotted PINN curve is less peaked (for visualization).
    """
    # Allow the calling wrapper to override the default PINN grid configuration.
    dflt = pinn_cfg_defaults or {}
    d_x_min = float(dflt.get("x_min", 1e-5))
    d_x_max = float(dflt.get("x_max", 80.0))
    d_n_x = int(dflt.get("n_x", 200))
    d_n_u = int(dflt.get("n_u", 150))
    d_u_min = float(dflt.get("u_min", -1.0))
    d_u_max = float(dflt.get("u_max", 10.0))

    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.get_default_dtype())
    device, dtype = x.device, x.dtype

    mu = torch.as_tensor(self.params_equ["mu"], device=device, dtype=dtype)
    r = torch.as_tensor(self.params_equ["r"], device=device, dtype=dtype)
    sigma = torch.as_tensor(self.params_equ["sigma"], device=device, dtype=dtype)
    lam = torch.as_tensor(self.params_equ["lambda"], device=device, dtype=dtype)
    alpha = torch.as_tensor(self.params_equ["z"], device=device, dtype=dtype)  # Jump size is stored under "z".
    beta = torch.as_tensor(self.params_equ["beta"], device=device, dtype=dtype)
    gamma = torch.as_tensor(self.params_train["entropy_alpha"], device=device, dtype=dtype)

    # --------- normalize shape handling ---------
    need_unsqueeze = False
    if x.dim() == 3 and x.size(-1) == 1:
        x_body = x.squeeze(-1)
        need_unsqueeze = True
    else:
        x_body = x
    orig_shape = x_body.shape

    # --------- train/reuse PINN once ---------
    pinn_cfg = dict(
        x_min=float(self.params_train.get("pinn_x_min", d_x_min)),
        x_max=float(self.params_train.get("pinn_x_max", d_x_max)),
        n_x=int(self.params_train.get("pinn_n_x", d_n_x)),
        n_u=int(self.params_train.get("pinn_n_u", d_n_u)),
        u_min=float(self.params_train.get("pinn_u_min", d_u_min)),
        u_max=float(self.params_train.get("pinn_u_max", d_u_max)),
        # Optional upper bound for PINN normalization and evaluation.
        x_eval_max=None,
        n_iter=int(self.params_train.get("pinn_iter", 3000)),
        lr=5e-4,
        width=256,
        depth=5,
        use_cache=True,
    )

    # Optional: allow user to explicitly set eval/normalization domain upper bound.
    x_eval_max = self.params_train.get("pinn_x_eval_max", None)
    pinn_cfg["x_eval_max"] = (float(x_eval_max) if x_eval_max is not None else None)

    # Optional adaptive grid based on current x range (and actor-implied u range)
    if self.params_train.get("pinn_adaptive_grid", False):
        x_min_cfg = float(self.params_train.get("pinn_x_min", pinn_cfg["x_min"]))
        x_max_cfg = float(self.params_train.get("pinn_x_max", pinn_cfg["x_max"]))
        u_min_cfg = float(self.params_train.get("pinn_u_min", pinn_cfg["u_min"]))
        u_max_cfg = float(self.params_train.get("pinn_u_max", pinn_cfg["u_max"]))
        x_pad = float(self.params_train.get("pinn_x_pad", 0.1))
        u_pad = float(self.params_train.get("pinn_u_pad", 0.05))

        x_min_cur = float(x_body.min().detach().cpu().item())
        x_max_cur = float(x_body.max().detach().cpu().item())
        x_span = x_max_cur - x_min_cur
        min_x_span = float(self.params_train.get("pinn_min_x_span", 1e-2))
        if x_span < min_x_span:
            x_min_adapt = x_min_cfg
            x_max_adapt = x_max_cfg
        else:
            x_pad_val = (x_pad * x_span) if x_pad <= 1.0 else x_pad
            x_min_adapt = max(x_min_cfg, x_min_cur - x_pad_val)
            x_max_adapt = min(x_max_cfg, x_max_cur + x_pad_val)

        u_min_adapt = u_min_cfg
        u_max_adapt = u_max_cfg
        try:
            b = x_body.reshape(-1, 1)
            if b.size(0) > 512:
                b = b[:512]
            t0 = torch.zeros_like(b)
            # Prefer the entrypoint's preprocess if available; fall back to raw concat.
            if hasattr(self, "_preprocess_state"):
                try:
                    state = self._preprocess_state(t0, b)
                except Exception:
                    state = torch.cat([t0, b], dim=1)
            else:
                state = torch.cat([t0, b], dim=1)

            n_mc = int(self.params_train.get("pinn_adapt_n_mc", 256))
            u_samples_list = []
            with torch.no_grad():
                for _ in range(n_mc):
                    action, _, _, _, _, _, _, _ = self.net_actor.sample(state, deterministic=False)
                    u_samples_list.append(action)
            u_samples = torch.stack(u_samples_list, dim=0)
            u_flat = u_samples.reshape(-1)
            q_lo = float(self.params_train.get("pinn_u_q_lo", 0.001))
            q_hi = float(self.params_train.get("pinn_u_q_hi", 0.999))
            u_min_cur = float(torch.quantile(u_flat, q_lo).detach().cpu().item())
            u_max_cur = float(torch.quantile(u_flat, q_hi).detach().cpu().item())

            u_span = u_max_cur - u_min_cur
            min_u_span = float(self.params_train.get("pinn_min_u_span", 1e-3))
            if u_span >= min_u_span:
                u_pad_val = (u_pad * u_span) if u_pad <= 1.0 else u_pad
                u_min_adapt = max(u_min_cfg, u_min_cur - u_pad_val)
                u_max_adapt = min(u_max_cfg, u_max_cur + u_pad_val)
            else:
                u_min_adapt = u_min_cfg
                u_max_adapt = u_max_cfg
        except Exception as e:
            if self.params_train.get("pinn_adapt_verbose", False):
                print(f"[warn] pinn_adaptive_grid actor sampling failed: {e}, using cfg range")

        if x_max_adapt > x_min_adapt:
            pinn_cfg["x_min"] = x_min_adapt
            pinn_cfg["x_max"] = x_max_adapt
        if u_max_adapt > u_min_adapt:
            pinn_cfg["u_min"] = u_min_adapt
            pinn_cfg["u_max"] = u_max_adapt

    pinn_device = device
    pinn_dtype = dtype
    print("[pinn-eval] Getting PINN cache (in-memory / disk / or train)...", flush=True)
    try:
        try:
            cache = self._solve_value_function_pinn(mu, r, sigma, lam, alpha, beta, gamma, pinn_device, pinn_dtype, **pinn_cfg)
        except TypeError:
            # Retry without x_eval_max if the model's solver rejects the argument.
            pinn_cfg.pop("x_eval_max", None)
            cache = self._solve_value_function_pinn(mu, r, sigma, lam, alpha, beta, gamma, pinn_device, pinn_dtype, **pinn_cfg)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pinn_device = torch.device("cpu")
        pinn_dtype = torch.float32
        small_cfg = dict(
            x_min=d_x_min,
            x_max=d_x_max,
            n_x=500,
            n_u=100,
            u_min=-1.0,
            u_max=10.0,
            n_iter=3000,
            lr=5e-4,
            width=256,
            depth=5,
            use_cache=False,
        )
        cache = self._solve_value_function_pinn(
            mu.detach().cpu(),
            r.detach().cpu(),
            sigma.detach().cpu(),
            lam.detach().cpu(),
            alpha.detach().cpu(),
            beta.detach().cpu(),
            gamma.detach().cpu(),
            pinn_device,
            pinn_dtype,
            **small_cfg,
        )

    net = cache["net"]
    # Ensure net is on pinn_device (cache may have been built on another device)
    net = net.to(pinn_device)
    x_min = float(cache["x_min"])
    x_max = float(cache["x_max"])
    # Make eval bound visible to plots/warnings/clamps in this run (best-effort).
    if self.params_train.get("pinn_x_eval_max", None) is None:
        try:
            self.params_train["pinn_x_eval_max"] = x_max
        except Exception:
            pass
    u_grid_train = cache["u_grid"]
    u_weights_train = cache["u_weights"]

    # Evaluation-only grid densification (best-effort)
    u_grid = u_grid_train
    u_weights = u_weights_train
    try:
        n_u_eval = self.params_train.get("pinn_eval_n_u", None)
        if n_u_eval is not None:
            n_u_eval = int(n_u_eval)
        if n_u_eval is not None and n_u_eval > 1:
            u_min_eval = self.params_train.get("pinn_eval_u_min", None)
            u_max_eval = self.params_train.get("pinn_eval_u_max", None)
            if u_min_eval is None:
                u_min_eval = float(u_grid_train.min().detach().cpu().item())
            if u_max_eval is None:
                u_max_eval = float(u_grid_train.max().detach().cpu().item())
            u_min_eval = float(u_min_eval)
            u_max_eval = float(u_max_eval)
            if u_max_eval > u_min_eval:
                u_grid = torch.linspace(u_min_eval, u_max_eval, n_u_eval, device=pinn_device, dtype=pinn_dtype).view(1, -1)
                du = float(u_grid[0, 1].item() - u_grid[0, 0].item())
                u_weights = torch.ones_like(u_grid) * du
                u_weights[:, 0] *= 0.5
                u_weights[:, -1] *= 0.5
    except Exception:
        u_grid = u_grid_train
        u_weights = u_weights_train

    # Log PINN grid ranges once per run
    if not getattr(self, "_pinn_range_logged", False):
        try:
            x_body_min = float(x_body.min().detach().cpu().item())
            x_body_max = float(x_body.max().detach().cpu().item())
            u_min_train = float(u_grid_train.min().detach().cpu().item())
            u_max_train = float(u_grid_train.max().detach().cpu().item())
            u_min_eval = float(u_grid.min().detach().cpu().item())
            u_max_eval = float(u_grid.max().detach().cpu().item())
            print(
                f"[pinn-grid] x_range=[{x_min:.4g}, {x_max:.4g}] "
                f"(x_batch=[{x_body_min:.4g}, {x_body_max:.4g}]), "
                f"u_train=[{u_min_train:.4g}, {u_max_train:.4g}] (n_u_train={int(u_grid_train.size(1))}), "
                f"u_eval=[{u_min_eval:.4g}, {u_max_eval:.4g}] (n_u_eval={int(u_grid.size(1))}), "
                f"n_x={int(pinn_cfg['n_x'])}"
            )
        except Exception:
            print("[pinn-grid] failed to log grid ranges.")
        self._pinn_range_logged = True

    # --------- move x to pinn_device/pinn_dtype, then CLAMP TO TRAINING DOMAIN ---------
    x_eval = x_body.reshape(-1, 1)
    if pinn_device.type == "cpu" and device.type == "cuda":
        x_eval = x_eval.detach().cpu().to(dtype=pinn_dtype)
    else:
        x_eval = x_eval.to(device=pinn_device, dtype=pinn_dtype)

    if not getattr(self, "_pinn_x_clip_logged", False):
        try:
            x_min_raw = float(x_eval.min().detach().cpu().item())
            x_max_raw = float(x_eval.max().detach().cpu().item())
            if x_min_raw < x_min or x_max_raw > x_max:
                clipped = torch.logical_or(x_eval < x_min, x_eval > x_max)
                frac = float(clipped.float().mean().detach().cpu().item())
                print(
                    f"[pinn-eval] WARNING: x outside PINN domain. "
                    f"x_eval=[{x_min_raw:.4g},{x_max_raw:.4g}] vs "
                    f"[{x_min:.4g},{x_max:.4g}], clipped_frac={frac:.3f}"
                )
        except Exception:
            pass
        self._pinn_x_clip_logged = True
    x_eval = torch.clamp(x_eval, min=x_min, max=x_max)
    x_col = x_eval

    # --------- evaluate V and derivatives on x_eval ---------
    # Derivatives are memory-heavy (create_graph=True). Support chunking to avoid OOM.
    deriv_chunk = self.params_train.get("pinn_deriv_chunk", None)
    if deriv_chunk is None:
        try:
            deriv_chunk = max(1000, min(10000, int(x_eval.size(0)) // 10))
        except Exception:
            deriv_chunk = None

    use_dense_grid = (
        (self.params_train.get("pinn_use_dense_grid_interpolation", False)
         or self.params_train.get("pinn_eval_use_dense_grid_interpolation", False))
        and cache.get("dense_grid") is not None
    )
    if use_dense_grid and hasattr(self, "_interpolate_v_from_dense_grid"):
        V_eval_interp = self._interpolate_v_from_dense_grid(x_eval, cache["dense_grid"], pinn_device, pinn_dtype)
        if V_eval_interp is not None:
            V_eval = V_eval_interp.view(-1, 1)
            # Still need derivatives from the NN; do it chunked if requested.
            if deriv_chunk is None or int(x_eval.size(0)) <= int(deriv_chunk):
                _, Vx_eval, Vxx_eval = self._value_net_with_derivatives(net, x_eval, x_min, x_max)
            else:
                Vx_parts, Vxx_parts = [], []
                for i in range(0, int(x_eval.size(0)), int(deriv_chunk)):
                    _, Vx_i, Vxx_i = self._value_net_with_derivatives(net, x_eval[i : i + int(deriv_chunk)], x_min, x_max)
                    Vx_parts.append(Vx_i)
                    Vxx_parts.append(Vxx_i)
                Vx_eval = torch.cat(Vx_parts, dim=0)
                Vxx_eval = torch.cat(Vxx_parts, dim=0)
        else:
            if deriv_chunk is None or int(x_eval.size(0)) <= int(deriv_chunk):
                V_eval, Vx_eval, Vxx_eval = self._value_net_with_derivatives(net, x_eval, x_min, x_max)
            else:
                V_parts, Vx_parts, Vxx_parts = [], [], []
                for i in range(0, int(x_eval.size(0)), int(deriv_chunk)):
                    V_i, Vx_i, Vxx_i = self._value_net_with_derivatives(net, x_eval[i : i + int(deriv_chunk)], x_min, x_max)
                    V_parts.append(V_i)
                    Vx_parts.append(Vx_i)
                    Vxx_parts.append(Vxx_i)
                V_eval = torch.cat(V_parts, dim=0)
                Vx_eval = torch.cat(Vx_parts, dim=0)
                Vxx_eval = torch.cat(Vxx_parts, dim=0)
    else:
        if deriv_chunk is None or int(x_eval.size(0)) <= int(deriv_chunk):
            V_eval, Vx_eval, Vxx_eval = self._value_net_with_derivatives(net, x_eval, x_min, x_max)
        else:
            V_parts, Vx_parts, Vxx_parts = [], [], []
            for i in range(0, int(x_eval.size(0)), int(deriv_chunk)):
                V_i, Vx_i, Vxx_i = self._value_net_with_derivatives(net, x_eval[i : i + int(deriv_chunk)], x_min, x_max)
                V_parts.append(V_i)
                Vx_parts.append(Vx_i)
                Vxx_parts.append(Vxx_i)
            V_eval = torch.cat(V_parts, dim=0)
            Vx_eval = torch.cat(Vx_parts, dim=0)
            Vxx_eval = torch.cat(Vxx_parts, dim=0)

    def _to_pinn(t):
        return t if (t.device.type == pinn_device.type and t.dtype == pinn_dtype) else t.detach().to(device=pinn_device, dtype=pinn_dtype)

    alpha_pinn = _to_pinn(alpha)
    mu_pinn = _to_pinn(mu)
    r_pinn = _to_pinn(r)
    sigma_pinn = _to_pinn(sigma)
    lam_pinn = _to_pinn(lam)
    beta_pinn = _to_pinn(beta)
    gamma_pinn = _to_pinn(gamma)

    reward = torch.pow(torch.clamp(x_col, min=1e-12), self.p) / self.p
    xVx = x_col * Vx_eval

    # --------- precompute V_shift(x*(1+alpha*u)) over u_grid (cache) ---------
    n_u = u_grid.size(1)
    u_chunk = int(self.params_train.get("pinn_u_chunk", 64))
    u_chunk = max(1, min(n_u, u_chunk))

    # Optional cache across repeated calls (plotting/debug): store per-chunk V_shift.
    use_v_shift_cache = bool(self.params_train.get("pinn_cache_v_shift", True))
    if not hasattr(self, "_v_shift_cache") or getattr(self, "_v_shift_cache") is None:
        try:
            self._v_shift_cache = {}
        except Exception:
            pass

    V_shift_cache = None
    if use_v_shift_cache and hasattr(self, "_v_shift_cache"):
        try:
            x_stats = (
                float(x_col.mean().detach().cpu().item()),
                float(x_col.std().detach().cpu().item()),
                float(x_col.min().detach().cpu().item()),
                float(x_col.max().detach().cpu().item()),
                int(x_col.numel()),
            )
            cache_key = (
                "vshift",
                int(n_u),
                int(u_chunk),
                float(alpha_pinn.detach().cpu().item()),
                float(x_min),
                float(x_max),
                str(pinn_device),
                str(pinn_dtype),
            )
            prev = self._v_shift_cache.get(cache_key, None)
            tol = float(self.params_train.get("pinn_v_shift_cache_tol", 1e-6))
            if prev is not None:
                prev_stats, prev_cache = prev
                if all(abs(a - b) <= tol for a, b in zip(prev_stats[:4], x_stats[:4])) and (prev_stats[4] == x_stats[4]):
                    V_shift_cache = prev_cache
        except Exception:
            V_shift_cache = None

    if V_shift_cache is None:
        V_shift_cache = {}
        with torch.no_grad():
            for start in range(0, n_u, u_chunk):
                end = min(start + u_chunk, n_u)
                u_chunk_grid = u_grid[:, start:end]
                one_plus = 1.0 + alpha_pinn * u_chunk_grid
                x_shift = x_col * one_plus
                # Clamp x_shift to PINN domain so V is not extrapolated; otherwise H(t,x,u)
                # can become monotonic in u and the Gibbs distribution p(u) looks non-unimodal.
                x_shift_safe = torch.clamp(x_shift, min=x_min, max=x_max)
                V_shift = self._value_net_forward_chunked(
                    net,
                    x_shift_safe.reshape(-1, 1),
                    x_min,
                    x_max,
                    chunk=self.params_train.get("pinn_chunk", 200000),
                    use_no_grad=True,
                ).view_as(x_shift)
                V_shift_cache[start] = V_shift.detach()
        if use_v_shift_cache and hasattr(self, "_v_shift_cache"):
            try:
                self._v_shift_cache[cache_key] = (x_stats, V_shift_cache)
            except Exception:
                pass

    # --------- chunked evaluation over u_grid to avoid OOM ---------
    max_H = torch.full((x_col.size(0), 1), -float("inf"), device=pinn_device, dtype=pinn_dtype)
    u_hard = torch.zeros_like(max_H)
    max_logit = torch.full_like(max_H, -float("inf")) if gamma_pinn.item() > 0 else None

    # pass 1: get max_H, u_hard (for stable softmax)
    with torch.no_grad():
        for start in range(0, n_u, u_chunk):
            end = min(start + u_chunk, n_u)
            u_chunk_grid = u_grid[:, start:end]
            w_chunk = u_weights[:, start:end]

            V_shift = V_shift_cache[start]

            drift = (r_pinn + (mu_pinn - r_pinn) * u_chunk_grid) * xVx
            diffusion = 0.5 * (sigma_pinn * u_chunk_grid * x_col) ** 2 * Vxx_eval
            jump = lam_pinn * (V_shift - V_eval - alpha_pinn * u_chunk_grid * xVx)
            H_chunk = reward + drift + diffusion + jump

            chunk_max, chunk_idx = H_chunk.max(dim=1, keepdim=True)
            u_chunk_exp = u_chunk_grid.expand(x_col.size(0), -1)
            u_chunk_argmax = u_chunk_exp.gather(1, chunk_idx)
            update_mask = chunk_max > max_H
            max_H = torch.where(update_mask, chunk_max, max_H)
            u_hard = torch.where(update_mask, u_chunk_argmax, u_hard)

            if gamma_pinn.item() > 0:
                logits = (H_chunk / gamma_pinn) + torch.log(w_chunk + 1e-30)
                max_logit = torch.maximum(max_logit, logits.max(dim=1, keepdim=True).values)

    # pass 2: stable softmax accumulation for u_mean/u_std
    if gamma_pinn.item() > 0:
        sum_exp = torch.zeros_like(max_H)
        sum_u = torch.zeros_like(max_H)
        sum_u2 = torch.zeros_like(max_H)
        with torch.no_grad():
            for start in range(0, n_u, u_chunk):
                end = min(start + u_chunk, n_u)
                u_chunk_grid = u_grid[:, start:end]
                w_chunk = u_weights[:, start:end]

                V_shift = V_shift_cache[start]

                drift = (r_pinn + (mu_pinn - r_pinn) * u_chunk_grid) * xVx
                diffusion = 0.5 * (sigma_pinn * u_chunk_grid * x_col) ** 2 * Vxx_eval
                jump = lam_pinn * (V_shift - V_eval - alpha_pinn * u_chunk_grid * xVx)
                H_chunk = reward + drift + diffusion + jump

                logits = (H_chunk / gamma_pinn) + torch.log(w_chunk + 1e-30)
                exp_chunk = torch.exp(logits - max_logit)
                sum_exp = sum_exp + exp_chunk.sum(dim=1, keepdim=True)
                sum_u = sum_u + (exp_chunk * u_chunk_grid).sum(dim=1, keepdim=True)
                sum_u2 = sum_u2 + (exp_chunk * (u_chunk_grid**2)).sum(dim=1, keepdim=True)

        u_mean = sum_u / (sum_exp + 1e-30)
        u_var = sum_u2 / (sum_exp + 1e-30) - u_mean**2
        u_std = torch.sqrt(torch.clamp(u_var, min=0.0))
    else:
        u_mean = u_hard
        u_std = torch.zeros_like(u_mean)

    policy_discrete = None
    if return_policy and gamma_pinn.item() > 0:
        # Optionally use a larger gamma for the returned policy so the plotted PINN curve is less peaked (not Dirac-like).
        use_display_gamma = (
            display_gamma_for_policy is not None
            and float(display_gamma_for_policy) > 0
            and float(display_gamma_for_policy) != gamma_pinn.item()
        )
        if use_display_gamma:
            gamma_for_probs = torch.full(
                (1,), float(display_gamma_for_policy), device=pinn_device, dtype=pinn_dtype
            )
            max_logit_p = torch.full((x_col.size(0), 1), -float("inf"), device=pinn_device, dtype=pinn_dtype)
            with torch.no_grad():
                for start in range(0, n_u, u_chunk):
                    end = min(start + u_chunk, n_u)
                    u_chunk_grid = u_grid[:, start:end]
                    w_chunk = u_weights[:, start:end]
                    V_shift = V_shift_cache[start]
                    drift = (r_pinn + (mu_pinn - r_pinn) * u_chunk_grid) * xVx
                    diffusion = 0.5 * (sigma_pinn * u_chunk_grid * x_col) ** 2 * Vxx_eval
                    jump = lam_pinn * (V_shift - V_eval - alpha_pinn * u_chunk_grid * xVx)
                    H_chunk = reward + drift + diffusion + jump
                    logits = (H_chunk / gamma_for_probs) + torch.log(w_chunk + 1e-30)
                    max_logit_p = torch.maximum(max_logit_p, logits.max(dim=1, keepdim=True).values)
            sum_exp_p = torch.zeros_like(max_H)
            with torch.no_grad():
                for start in range(0, n_u, u_chunk):
                    end = min(start + u_chunk, n_u)
                    u_chunk_grid = u_grid[:, start:end]
                    w_chunk = u_weights[:, start:end]
                    V_shift = V_shift_cache[start]
                    drift = (r_pinn + (mu_pinn - r_pinn) * u_chunk_grid) * xVx
                    diffusion = 0.5 * (sigma_pinn * u_chunk_grid * x_col) ** 2 * Vxx_eval
                    jump = lam_pinn * (V_shift - V_eval - alpha_pinn * u_chunk_grid * xVx)
                    H_chunk = reward + drift + diffusion + jump
                    logits = (H_chunk / gamma_for_probs) + torch.log(w_chunk + 1e-30)
                    exp_chunk = torch.exp(logits - max_logit_p)
                    sum_exp_p = sum_exp_p + exp_chunk.sum(dim=1, keepdim=True)
            probs = torch.zeros((x_col.size(0), n_u), device=pinn_device, dtype=pinn_dtype)
            with torch.no_grad():
                for start in range(0, n_u, u_chunk):
                    end = min(start + u_chunk, n_u)
                    u_chunk_grid = u_grid[:, start:end]
                    w_chunk = u_weights[:, start:end]
                    V_shift = V_shift_cache[start]
                    drift = (r_pinn + (mu_pinn - r_pinn) * u_chunk_grid) * xVx
                    diffusion = 0.5 * (sigma_pinn * u_chunk_grid * x_col) ** 2 * Vxx_eval
                    jump = lam_pinn * (V_shift - V_eval - alpha_pinn * u_chunk_grid * xVx)
                    H_chunk = reward + drift + diffusion + jump
                    logits = (H_chunk / gamma_for_probs) + torch.log(w_chunk + 1e-30)
                    exp_chunk = torch.exp(logits - max_logit_p)
                    p_chunk = exp_chunk / (sum_exp_p + 1e-30)
                    probs[:, start:end] = p_chunk
        else:
            probs = torch.zeros((x_col.size(0), n_u), device=pinn_device, dtype=pinn_dtype)
            with torch.no_grad():
                for start in range(0, n_u, u_chunk):
                    end = min(start + u_chunk, n_u)
                    u_chunk_grid = u_grid[:, start:end]
                    w_chunk = u_weights[:, start:end]

                    V_shift = V_shift_cache[start]

                    drift = (r_pinn + (mu_pinn - r_pinn) * u_chunk_grid) * xVx
                    diffusion = 0.5 * (sigma_pinn * u_chunk_grid * x_col) ** 2 * Vxx_eval
                    jump = lam_pinn * (V_shift - V_eval - alpha_pinn * u_chunk_grid * xVx)
                    H_chunk = reward + drift + diffusion + jump

                    logits = (H_chunk / gamma_pinn) + torch.log(w_chunk + 1e-30)
                    exp_chunk = torch.exp(logits - max_logit)
                    p_chunk = exp_chunk / (sum_exp + 1e-30)
                    probs[:, start:end] = p_chunk

        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-30)
        policy_discrete = (u_grid.detach(), probs.detach())

    # --------- reshape to original x shape ---------
    V_eval = V_eval.to(device=device, dtype=dtype)
    u_mean = u_mean.to(device=device, dtype=dtype)
    u_std = u_std.to(device=device, dtype=dtype)
    u_hard = u_hard.to(device=device, dtype=dtype)

    optimal_value = V_eval.view(orig_shape)
    u_mean_of_t = u_mean.view(orig_shape)
    u_std_of_t = u_std.view(orig_shape)
    u_hard_of_t = u_hard.view(orig_shape)

    if need_unsqueeze:
        optimal_value = optimal_value.unsqueeze(-1)
        u_mean_of_t = u_mean_of_t.unsqueeze(-1)
        u_std_of_t = u_std_of_t.unsqueeze(-1)
        u_hard_of_t = u_hard_of_t.unsqueeze(-1)

    if return_cache:
        if return_u_hard:
            return u_mean_of_t, u_std_of_t, optimal_value, u_hard_of_t, cache, policy_discrete
        return u_mean_of_t, u_std_of_t, optimal_value, cache, policy_discrete
    if return_u_hard:
        return u_mean_of_t, u_std_of_t, optimal_value, u_hard_of_t, policy_discrete
    return u_mean_of_t, u_std_of_t, optimal_value, policy_discrete


def compute_exact_entropy(self, x, *, pinn_cfg_defaults: dict | None = None):
    # Return the true discrete softmax policy on u_grid (when available).
    u_mean, u_std, optimal_value, _u_hard, pol = compute_exact_entropy_with_cache(
        self,
        x,
        return_cache=False,
        return_u_hard=True,
        return_policy=True,
        pinn_cfg_defaults=pinn_cfg_defaults,
    )
    if pol is not None:
        u_grid, probs = pol
        policy = DiscretePolicy(u_grid.squeeze(0), probs)
        return policy, optimal_value
    # Fallback (e.g., gamma==0 or if policy building disabled)
    return Normal(loc=u_mean, scale=u_std), optimal_value


def compute_exact_no_exploratory(self, x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.get_default_dtype())
    device, dtype = x.device, x.dtype
    mu = torch.as_tensor(self.params_equ["mu"], device=device, dtype=dtype)
    r = torch.as_tensor(self.params_equ["r"], device=device, dtype=dtype)
    sigma = torch.as_tensor(self.params_equ["sigma"], device=device, dtype=dtype)
    lam = torch.as_tensor(self.params_equ["lambda"], device=device, dtype=dtype)
    alpha = torch.as_tensor(self.params_equ["z"], device=device, dtype=dtype)
    beta = torch.as_tensor(self.params_equ["beta"], device=device, dtype=dtype)

    if x.dim() == 3 and x.size(-1) == 1 and x.size(0) == x.size(1):
        x_body = x.squeeze(-1)
    else:
        x_body = x

    # ===== ANALYTICAL SOLUTION FOR NON-ENTROPY CASE (γ=0) =====
    # Solve FOC: (μ-r) + (p-1)σ²u* + λα((1+αu*)^(p-1) - 1) = 0, with 1+αu* > 0
    def optimal_control_foc(a):
        return (
            (mu - r).item()
            + sigma.item() ** 2 * (self.p - 1) * a
            + lam.item() * alpha.item() * ((1 + a * alpha.item()) ** (self.p - 1) - 1)
        )

    u_star_scalar = np.squeeze(fsolve(optimal_control_foc, 1.0))
    u_star = torch.tensor(u_star_scalar, device=device, dtype=dtype)

    K = (
        self.p * (r + (mu - r) * u_star)
        + (self.p * (self.p - 1) / 2) * sigma**2 * u_star**2
        + lam * ((1 + alpha * u_star) ** self.p - 1 - self.p * alpha * u_star)
        - beta
    )
    h_star = -1.0 / K

    optimal_control = u_star * torch.ones_like(x_body)
    if not hasattr(self, "_warned_exact_control_bounds"):
        a_min = self.params_train.get("action_min", None)
        a_max = self.params_train.get("action_max", None)
        if a_min is not None and a_max is not None:
            a_min_f, a_max_f = float(a_min), float(a_max)
            if u_star.item() < a_min_f - 1e-6 or u_star.item() > a_max_f + 1e-6:
                print(f"[warn] exact u*={u_star.item():.6f} outside action bounds [{a_min_f:.6f}, {a_max_f:.6f}].")
        else:
            bound = self.params_train.get("action_bound", None)
            if bound is not None:
                if u_star.item() < 0.0 - 1e-6 or u_star.item() > float(bound) + 1e-6:
                    print(f"[warn] exact u*={u_star.item():.6f} outside action bounds [0, {float(bound):.6f}].")
        self._warned_exact_control_bounds = True

    if self.p == 1 / 2:
        x_body = torch.clamp(x_body, min=1e-8)
    optimal_value = (h_star / self.p) * (x_body**self.p)

    if x.dim() == 3 and x.size(-1) == 1 and x.size(0) == x.size(1):
        optimal_control = optimal_control.unsqueeze(-1)
        optimal_value = optimal_value.unsqueeze(-1)

    return optimal_control, optimal_value

