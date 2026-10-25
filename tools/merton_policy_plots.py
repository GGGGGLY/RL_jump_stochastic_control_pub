"""Merton-specific policy/state distributions, Gibbs policies and Hamiltonians.

This module was named panel_c_plot when distributions occupied panel (c).
The current flow-policy density is drawn in panel (a); these routines are kept
separate from the control/mean-value-state renderer used by LQ and Multi-CARA.
"""

from experiment_utils import FONT_SIZE, LEGEND_FONT_SIZE
import json
import math
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def plot_gibbs_policy_by_pinn(
    self,
    x_val: float,
    device=None,
    output_folder=None,
    filename_suffix="",
):
    """
    Plot Gibbs policy distribution p(u|x) computed by PINN.
    Shows both unnormalized weights (exp(H/gamma + log(w))) and normalized probabilities.
    """
    device = device or self.device
    output_folder = output_folder or self.output_folder
    os.makedirs(output_folder, exist_ok=True)
    if filename_suffix is None:
        filename_suffix = ""
    # Auto-append itr tag when available (avoid overwriting across checkpoints).
    try:
        itr_val = getattr(self, "current_itr", None)
        if itr_val is None:
            itr_val = self.params_train.get("current_itr", None)
        if itr_val is not None and (not re.search(r"_itr\d+", str(filename_suffix))):
            filename_suffix = f"{filename_suffix}_itr{int(itr_val)}"
    except Exception:
        pass

    # Get discrete policy from PINN
    x_tensor = torch.tensor([[x_val]], device=device, dtype=torch.float32)
    policy_pinn, _ = self.compute_exact_entropy(x_tensor)

    if not isinstance(policy_pinn, self.DiscretePolicy):
        print(f"[plot_gibbs_policy] policy_pinn is not DiscretePolicy, got {type(policy_pinn)}")
        return

    probs_normalized = policy_pinn.probs.squeeze(0).detach().cpu().numpy()
    support = policy_pinn.support_u.detach().cpu().numpy()

    probs_normalized = probs_normalized / (probs_normalized.sum() + 1e-30)

    # Get unnormalized weights by recomputing H and exp(H/gamma + log(w))
    cache = getattr(self, "_exact_solution_cache_nn", None)
    if cache is None:
        print("[plot_gibbs_policy] No PINN cache found, cannot compute unnormalized weights.")
        unnormalized_weights = None
    else:
        net = cache["net"]
        x_min = float(cache["x_min"])
        x_max = float(cache["x_max"])
        u_grid = cache["u_grid"].squeeze(0)
        u_weights = cache.get("u_weights", None)
        if u_weights is not None:
            u_weights = u_weights.squeeze(0)
        else:
            u_weights = torch.ones_like(u_grid)

        pinn_device = next(net.parameters()).device
        pinn_dtype = next(net.parameters()).dtype
        x_p = torch.tensor([[x_val]], device=pinn_device, dtype=pinn_dtype).clamp_(x_min, x_max)

        Vp, Vx, Vxx = self._value_net_with_derivatives(net, x_p, x_min, x_max)
        Vp = Vp.squeeze()
        Vx = Vx.squeeze()
        Vxx = Vxx.squeeze()

        mu = torch.as_tensor(self.params_equ["mu"], device=pinn_device, dtype=pinn_dtype)
        r = torch.as_tensor(self.params_equ["r"], device=pinn_device, dtype=pinn_dtype)
        sigma = torch.as_tensor(self.params_equ["sigma"], device=pinn_device, dtype=pinn_dtype)
        lam = torch.as_tensor(self.params_equ["lambda"], device=pinn_device, dtype=pinn_dtype)
        z = torch.as_tensor(self.params_equ["z"], device=pinn_device, dtype=pinn_dtype)
        gamma = torch.as_tensor(self.params_train["entropy_alpha"], device=pinn_device, dtype=pinn_dtype)

        u = u_grid.view(1, -1)
        x = x_p.view(1, 1)
        xVx = x * Vx
        reward = torch.pow(torch.clamp(x, min=1e-12), self.p) / self.p

        x_shift = x * (1.0 + z * u)
        V_shift = self._value_net_forward(net, x_shift.reshape(-1, 1), x_min, x_max).view_as(x_shift)

        drift = (r + (mu - r) * u) * xVx
        diffusion = 0.5 * (sigma * u * x) ** 2 * Vxx
        jump = lam * (V_shift - Vp.view(1, 1) - z * u * xVx)

        H = (reward + drift + diffusion + jump).squeeze(0)

        logits = (H / gamma) + torch.log(u_weights + 1e-30)
        max_logit = logits.max()
        unnormalized_weights = torch.exp(logits - max_logit).detach().cpu().numpy()

        weights_min = float(unnormalized_weights.min())
        weights_max = float(unnormalized_weights.max())
        weights_sum = float(unnormalized_weights.sum())
        weights_ratio = weights_max / (weights_min + 1e-30)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=150)

    ax1 = axes[0]
    if unnormalized_weights is not None:
        ax1.plot(support, unnormalized_weights, "b-", linewidth=2, alpha=0.8,
                 label=r"$\exp(H/\gamma + \log(w))$ (unnormalized)")
        ax1.set_ylabel("Unnormalized Weight", fontsize=FONT_SIZE)
        ax1.set_title(
            f"Unnormalized Gibbs Weights at $x={x_val:.4f}$\n"
            f"min={weights_min:.6e}, max={weights_max:.6e}, sum={weights_sum:.6e}, "
            f"ratio (max/min)={weights_ratio:.6f}",
            fontsize=12,
        )
        ax1.legend(fontsize=LEGEND_FONT_SIZE, loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")
    else:
        ax1.text(0.5, 0.5, "Unnormalized weights not available",
                 ha="center", va="center", transform=ax1.transAxes)
        ax1.set_ylabel("Unnormalized Weight", fontsize=FONT_SIZE)

    ax2 = axes[1]
    if len(support) > 1:
        width = float((support[1] - support[0]) * 0.8)
    else:
        width = 0.01

    ax2.bar(support, probs_normalized, width=width, alpha=0.7,
            edgecolor="black", linewidth=0.5, label="Normalized $p(u|x)$")
    ax2.plot(support, probs_normalized, "o-", markersize=3, linewidth=1.5,
             alpha=0.8, label="Normalized $p(u|x)$ (line)")

    uniform_prob = 1.0 / len(support)
    ax2.axhline(y=uniform_prob, color="r", linestyle="--", linewidth=2,
                alpha=0.7, label=f"Uniform (1/n_u={uniform_prob:.6f})")

    prob_min = float(probs_normalized.min())
    prob_max = float(probs_normalized.max())
    prob_std = float(probs_normalized.std())
    prob_ratio = prob_max / (prob_min + 1e-30)
    entropy = float(-(probs_normalized * np.log(probs_normalized + 1e-30)).sum())
    max_entropy = np.log(len(support))
    kl_uniform = float((probs_normalized * (np.log(probs_normalized + 1e-30) - np.log(uniform_prob))).sum())

    ax2.set_xlabel("$u$", fontsize=FONT_SIZE)
    ax2.set_ylabel("$p(u|x)$ (Normalized)", fontsize=FONT_SIZE)
    ax2.set_title(
        f"Normalized Gibbs Policy at $x={x_val:.4f}$\n"
        f"min={prob_min:.6f}, max={prob_max:.6f}, ratio (max/min)={prob_ratio:.6f}, std={prob_std:.6f}\n"
        f"entropy={entropy:.4f}/{max_entropy:.4f}, KL(uniform)={kl_uniform:.6f}",
        fontsize=12,
    )
    ax2.legend(fontsize=LEGEND_FONT_SIZE, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    base = "gibbs_policy_by_pinn"
    save_path = os.path.join(output_folder, f"{base}{filename_suffix}.pdf" if filename_suffix else f"{base}.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[plot_gibbs_policy] Saved Gibbs policy plot to {save_path}")
    if unnormalized_weights is not None:
        print(f"[plot_gibbs_policy] Unnormalized weights: min={weights_min:.6e}, max={weights_max:.6e}, ratio={weights_ratio:.6f}")
        print(f"[plot_gibbs_policy] Normalized probs: min={prob_min:.6f}, max={prob_max:.6f}, ratio={prob_ratio:.6f}")
        if weights_ratio > 1.1 and prob_ratio < 1.2:
            print("[plot_gibbs_policy] WARNING: unnormalized weights vary but normalized probs are nearly uniform.")
    plt.close(fig)


@torch.no_grad()
def plot_H_by_pinn(
    self,
    x_val: float,
    device=None,
    output_folder=None,
    filename_suffix="",
):
    """
    Plot Hamiltonian H(x,u) computed by PINN as a function of u.
    """
    device = device or self.device
    output_folder = output_folder or self.output_folder
    os.makedirs(output_folder, exist_ok=True)
    if filename_suffix is None:
        filename_suffix = ""
    # Auto-append itr tag when available (avoid overwriting across checkpoints).
    try:
        itr_val = getattr(self, "current_itr", None)
        if itr_val is None:
            itr_val = self.params_train.get("current_itr", None)
        if itr_val is not None and (not re.search(r"_itr\d+", str(filename_suffix))):
            filename_suffix = f"{filename_suffix}_itr{int(itr_val)}"
    except Exception:
        pass

    cache = getattr(self, "_exact_solution_cache_nn", None)
    if cache is None:
        print("[plot_H] No PINN cache found (_exact_solution_cache_nn is None).")
        return

    net = cache["net"]
    x_min = float(cache["x_min"])
    x_max = float(cache["x_max"])
    u_grid = cache["u_grid"].squeeze(0)

    pinn_device = next(net.parameters()).device
    pinn_dtype = next(net.parameters()).dtype
    x_p = torch.tensor([[x_val]], device=pinn_device, dtype=pinn_dtype).clamp_(x_min, x_max)

    Vp, Vx, Vxx = self._value_net_with_derivatives(net, x_p, x_min, x_max)
    Vp = Vp.squeeze()
    Vx = Vx.squeeze()
    Vxx = Vxx.squeeze()

    mu = torch.as_tensor(self.params_equ["mu"], device=pinn_device, dtype=pinn_dtype)
    r = torch.as_tensor(self.params_equ["r"], device=pinn_device, dtype=pinn_dtype)
    sigma = torch.as_tensor(self.params_equ["sigma"], device=pinn_device, dtype=pinn_dtype)
    lam = torch.as_tensor(self.params_equ["lambda"], device=pinn_device, dtype=pinn_dtype)
    z = torch.as_tensor(self.params_equ["z"], device=pinn_device, dtype=pinn_dtype)
    gamma = torch.as_tensor(self.params_train["entropy_alpha"], device=pinn_device, dtype=pinn_dtype)

    u = u_grid.view(1, -1)
    x = x_p.view(1, 1)
    xVx = x * Vx
    reward = torch.pow(torch.clamp(x, min=1e-12), self.p) / self.p

    x_shift = x * (1.0 + z * u)
    V_shift = self._value_net_forward(net, x_shift.reshape(-1, 1), x_min, x_max).view_as(x_shift)

    drift = (r + (mu - r) * u) * xVx
    diffusion = 0.5 * (sigma * u * x) ** 2 * Vxx
    jump = lam * (V_shift - Vp.view(1, 1) - z * u * xVx)

    H = (reward + drift + diffusion + jump).squeeze(0)

    u_np = u_grid.detach().cpu().numpy()
    H_np = H.detach().cpu().numpy()

    H_min = float(H.min().item())
    H_max = float(H.max().item())
    H_mean = float(H.mean().item())
    H_std = float(H.std().item())
    H_range = H_max - H_min
    H_rng_gamma = H_range / (float(gamma.item()) + 1e-30)

    u_opt_idx = int(H.argmax().item())
    u_opt = float(u_np[u_opt_idx])
    H_opt = float(H_np[u_opt_idx])

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=150)

    ax1 = axes[0]
    ax1.plot(u_np, H_np, "b-", linewidth=2, alpha=0.8, label="$H(x,u)$")
    ax1.axhline(y=H_opt, color="r", linestyle="--", linewidth=1, alpha=0.5,
                label=f"$H^*={H_opt:.6f}$ at $u^*={u_opt:.4f}$")
    ax1.axvline(x=u_opt, color="r", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("$u$", fontsize=FONT_SIZE)
    ax1.set_ylabel("$H(x,u)$", fontsize=FONT_SIZE)
    ax1.set_title(
        f"Hamiltonian $H(x,u)$ at $x={x_val:.4f}$\n"
        f"Range: [{H_min:.6f}, {H_max:.6f}], std={H_std:.6f}, "
        f"$H_{{rng}}/\\gamma={H_rng_gamma:.6f}$\n"
        f"$\\mu-r={float(mu.item() - r.item()):.6f}$, $V_x={float(Vx.item()):.6f}$, $V_{{xx}}={float(Vxx.item()):.6f}$",
        fontsize=12,
    )
    ax1.legend(fontsize=LEGEND_FONT_SIZE, loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    exp_H_gamma = np.exp(H_np / float(gamma.item()))
    exp_H_gamma_min = float(exp_H_gamma.min())
    exp_H_gamma_max = float(exp_H_gamma.max())
    exp_H_gamma_ratio = exp_H_gamma_max / (exp_H_gamma_min + 1e-30)

    ax2.plot(u_np, exp_H_gamma, "g-", linewidth=2, alpha=0.8, label=r"$\exp(H(x,u)/\gamma)$")
    ax2.set_xlabel("$u$", fontsize=FONT_SIZE)
    ax2.set_ylabel(r"$\exp(H(x,u)/\gamma)$", fontsize=FONT_SIZE)
    ax2.set_title(
        r"$\exp(H/\gamma)$ (determines Gibbs policy shape)" + "\n"
        f"Range: [{exp_H_gamma_min:.6f}, {exp_H_gamma_max:.6f}], "
        f"ratio (max/min)={exp_H_gamma_ratio:.6f}\n"
        "If ratio is close to 1, policy will be nearly uniform",
        fontsize=12,
    )
    ax2.legend(fontsize=LEGEND_FONT_SIZE, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    base = "H_by_pinn"
    save_path = os.path.join(output_folder, f"{base}{filename_suffix}.pdf" if filename_suffix else f"{base}.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[plot_H] Saved Hamiltonian plot to {save_path}")
    print(f"[plot_H] H range: [{H_min:.6f}, {H_max:.6f}], H_rng/gamma={H_rng_gamma:.6f}")
    print(f"[plot_H] exp(H/gamma) ratio (max/min)={exp_H_gamma_ratio:.6f}")
    print(f"[plot_H] H std: {H_std:.6f}, H mean: {H_mean:.6f}")
    print(f"[plot_H] Vx: {float(Vx.item()):.6f}, Vxx: {float(Vxx.item()):.6f}, mu-r: {float(mu.item() - r.item()):.6f}")

    if H_range < 1e-5:
        print(f"[plot_H] WARNING: H range is very small ({H_range:.6e}); H is nearly flat.")
    elif H_rng_gamma < 0.1:
        print(f"[plot_H] WARNING: H_range/gamma = {H_rng_gamma:.6f} < 0.1, exp(H/gamma) is near-constant.")
    elif exp_H_gamma_ratio < 1.1:
        print(f"[plot_H] WARNING: exp(H/gamma) ratio is very close to 1 ({exp_H_gamma_ratio:.6f}).")
    else:
        print("[plot_H] H range and exp(H/gamma) ratio look reasonable.")

    plt.close(fig)


@torch.no_grad()
def plot_panel_c_evolved_x_distribution(
    self,
    t0_values=None,
    x0_values=None,
    n_samples=5000,
    evolve_to_T=True,
    device=None,
    filename_suffix="",
    output_folder=None,
    ax=None,
):
    """
    Plot evolved x distribution: PINN and flow net sample many u then evolve x forward.
    """
    device = device or self.device
    output_folder = output_folder or self.output_folder
    os.makedirs(output_folder, exist_ok=True)
    if filename_suffix is None:
        filename_suffix = ""

    from scipy.stats import gaussian_kde

    if t0_values is None:
        t0_values = [0.0]
    if x0_values is None:
        if hasattr(self, "X0_eval") and self.X0_eval is not None:
            x0_mean = float(self.X0_eval.mean().item())
        else:
            x0_mean = 10.0
        x0_values = [x0_mean]

    if not isinstance(t0_values, (list, tuple)):
        t0_values = [t0_values]
    if not isinstance(x0_values, (list, tuple)):
        x0_values = [x0_values]

    n_t0 = len(t0_values)
    n_x0 = len(x0_values)

    single_ax = ax is not None
    if single_ax:
        t0_values = [t0_values[0]]
        x0_values = [x0_values[0]]
        n_t0 = 1
        n_x0 = 1
        axes = np.array([[ax]])
    else:
        fig, axes = plt.subplots(n_t0, n_x0, figsize=(5 * n_x0, 4 * n_t0), dpi=150)
        if n_t0 == 1 and n_x0 == 1:
            axes = np.array([[axes]])
        elif n_t0 == 1:
            axes = axes.reshape(1, -1) if hasattr(axes, "reshape") else np.array([axes])
        elif n_x0 == 1:
            axes = axes.reshape(-1, 1) if hasattr(axes, "reshape") else np.array([[ax] for ax in axes])
        else:
            axes = np.array(axes)

    dt = self.dt
    T_evolve = self.T
    N_steps = int(T_evolve / dt) if evolve_to_T else 1

    print(f"[panel_c] Plotting evolved x distribution for {n_t0 * n_x0} (t0,x0) combinations...")
    print(f"[panel_c] Will evolve {'to T' if evolve_to_T else 'one step'} (N_steps={N_steps})")

    for i, t0_val in enumerate(t0_values):
        for j, x0_val in enumerate(x0_values):
            ax = axes[i, j]
            print(f"[panel_c] Processing (t0={t0_val:.3f}, x0={x0_val:.3f})...")

            t0 = torch.full((n_samples, 1), t0_val, device=device, dtype=torch.float32)
            X0 = torch.full((n_samples, 1), x0_val, device=device, dtype=torch.float32)
            state0 = torch.cat([t0, X0], dim=1)

            dW = torch.randn(n_samples, N_steps, 1, device=device, dtype=torch.float32) * np.sqrt(dt)
            lambda_param = self.params_equ["lambda"]
            dNt = torch.bernoulli(torch.full((n_samples, N_steps, 1), lambda_param * dt, device=device, dtype=torch.float32))
            dMt = dNt - lambda_param * dt

            try:
                x0_tensor = torch.full((1, 1), x0_val, device=device, dtype=torch.float32)
                policy_pinn, _ = self.compute_exact_entropy(x0_tensor)

                if hasattr(policy_pinn, "icdf"):
                    U_samples = torch.rand(n_samples, 1, device=device, dtype=torch.float32)
                    U_samples = torch.clamp(U_samples, 1e-6, 1 - 1e-6)
                    u_pinn = policy_pinn.icdf(U_samples)
                else:
                    raise RuntimeError(f"PINN policy has no icdf method. policy type: {type(policy_pinn)}")

                X_curr = X0
                t_curr = t0
                for k in range(N_steps):
                    u_k = u_pinn
                    mu = self.params_equ["mu"]
                    r = self.params_equ["r"]
                    sigma = self.params_equ["sigma"]
                    alpha = self.params_equ["z"]

                    drift = (r + (mu - r) * u_k) * X_curr * dt
                    diffusion = sigma * u_k * X_curr * dW[:, k, :]
                    jump = alpha * u_k * X_curr * dMt[:, k, :]
                    X_curr = X_curr + drift + diffusion + jump
                    X_curr = torch.clamp(X_curr, min=1e-8)
                    t_curr = t_curr + dt

                X_pinn = X_curr.detach().cpu().numpy().flatten()
            except Exception as e:
                print(f"[WARNING] Failed to evolve PINN: {e}")
                X_pinn = None

            try:
                u_flow_list = []
                batch_size = 1000
                n_batches = (n_samples + batch_size - 1) // batch_size
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, n_samples)
                    state_batch = state0[batch_start:batch_end]

                    with torch.no_grad():
                        action, _, _, _, _, _, _, _ = self.net_actor.sample(
                            state_batch, deterministic=False
                        )
                    u_flow_list.append(action.detach())

                u_flow = torch.cat(u_flow_list, dim=0)

                X_curr = X0
                t_curr = t0
                for k in range(N_steps):
                    u_k = u_flow
                    mu = self.params_equ["mu"]
                    r = self.params_equ["r"]
                    sigma = self.params_equ["sigma"]
                    alpha = self.params_equ["z"]

                    drift = (r + (mu - r) * u_k) * X_curr * dt
                    diffusion = sigma * u_k * X_curr * dW[:, k, :]
                    jump = alpha * u_k * X_curr * dMt[:, k, :]
                    X_curr = X_curr + drift + diffusion + jump
                    X_curr = torch.clamp(X_curr, min=1e-8)
                    t_curr = t_curr + dt

                X_flow = X_curr.detach().cpu().numpy().flatten()
            except Exception as e:
                print(f"[WARNING] Failed to evolve flow net: {e}")
                X_flow = None

            if X_pinn is not None and X_flow is not None:
                valid_pinn = np.isfinite(X_pinn) & (X_pinn > 0)
                valid_flow = np.isfinite(X_flow) & (X_flow > 0)
                X_pinn_clean = X_pinn[valid_pinn]
                X_flow_clean = X_flow[valid_flow]

                if len(X_pinn_clean) > 0 and len(X_flow_clean) > 0:
                    try:
                        kde_pinn = gaussian_kde(X_pinn_clean)
                        kde_flow = gaussian_kde(X_flow_clean)

                        x_min = min(X_pinn_clean.min(), X_flow_clean.min())
                        x_max = max(X_pinn_clean.max(), X_flow_clean.max())
                        x_range = x_max - x_min
                        x_min_plot = max(0.0, x_min - 0.1 * x_range)
                        x_max_plot = x_max + 0.1 * x_range

                        x_plot = np.linspace(x_min_plot, x_max_plot, 200)
                        ax.plot(x_plot, kde_pinn(x_plot), label="PINN",
                                linewidth=2, alpha=0.7, color="darkviolet")
                        ax.plot(x_plot, kde_flow(x_plot), label="Flownet",
                                linewidth=2, alpha=0.7, color="darkorange")
                    except Exception as e:
                        print(f"[WARNING] KDE failed: {e}, falling back to histogram")
                        ax.hist(X_pinn_clean, bins=50, alpha=0.5, label="PINN",
                                density=True, color="darkviolet", edgecolor="black", linewidth=0.5)
                        ax.hist(X_flow_clean, bins=50, alpha=0.5, label="Flownet",
                                density=True, color="darkorange", edgecolor="black", linewidth=0.5)

                    ax.set_xlabel("$X_T$" if evolve_to_T else "$X_{t_0+dt}$", fontsize=FONT_SIZE)
                    ax.set_ylabel("Density", fontsize=FONT_SIZE)
                    ax.tick_params(axis="both", which="major", labelsize=16)
                    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="best", framealpha=0.5, edgecolor="gray")
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f"$t_0={t0_val:.2f}$, $x_0={x0_val:.2f}$", fontsize=FONT_SIZE)
                else:
                    ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Evolution failed", ha="center", va="center", transform=ax.transAxes)

    if not single_ax:
        plt.tight_layout()
        itr_val = getattr(self, "current_itr", None)
        if itr_val is None:
            itr_val = self.params_train.get("current_itr", None)
        if itr_val is not None:
            itr_tag = f"_itr{int(itr_val)}"
            if itr_tag not in filename_suffix:
                filename_suffix = f"{filename_suffix}{itr_tag}"
        filename = f"merton_panel_c_evolved_x_distribution{filename_suffix}"
        save_path = os.path.join(output_folder, f"{filename}.pdf")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[panel_c] Saved evolved x distribution plot to {save_path}")
        plt.close(fig)
        return None
    return axes[0, 0]


@torch.no_grad()
def _dump_panel_c_policy_debug(
    self,
    *,
    t_val: float,
    x_val: float,
    x_tensor: "torch.Tensor",
    state: "torch.Tensor",
    output_folder: str,
    policy_pinn=None,
    pinn_support=None,
    pinn_probs=None,
    u_flow_clean=None,
):
    """
    Dump key details needed to diagnose panel (c) policy mismatch.
    Writes JSON to: output_folder/policy_debug_panelc.json (appends entries).
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception:
        pass

    pinn_info = {
        "formula": "Discrete Gibbs on u_grid: p(u_i|x) ∝ w_i * exp(H(x,u_i)/gamma); logits = H/gamma + log(w)",
        "uses": "Hamiltonian H (reward + drift + diffusion + jump) from PINN value network derivatives",
        "t_used": False,
    }
    try:
        pinn_info["gamma_entropy"] = float(self.params_train.get("entropy_alpha", 0.0))
        pinn_info["z_jump_alpha"] = float(self.params_equ.get("z", 0.0))
        pinn_info["lam"] = float(self.params_equ.get("lambda", 0.0))
        pinn_info["mu"] = float(self.params_equ.get("mu", 0.0))
        pinn_info["r"] = float(self.params_equ.get("r", 0.0))
        pinn_info["sigma"] = float(self.params_equ.get("sigma", 0.0))
    except Exception:
        pass

    try:
        cache = getattr(self, "_exact_solution_cache_nn", None)
        if cache is not None:
            u_grid = cache.get("u_grid", None)
            u_weights = cache.get("u_weights", None)
            if u_grid is not None:
                ug = u_grid.detach().cpu().numpy().reshape(-1)
                pinn_info["u_grid"] = {
                    "n_u": int(ug.size),
                    "min": float(np.min(ug)),
                    "max": float(np.max(ug)),
                    "median_du": float(np.median(np.diff(ug))) if ug.size > 1 else None,
                }
            if u_weights is not None:
                ww = u_weights.detach().cpu().numpy().reshape(-1)
                pinn_info["u_weights"] = {
                    "min": float(np.min(ww)),
                    "max": float(np.max(ww)),
                    "sum": float(np.sum(ww)),
                }
            pinn_info["pinn_domain_x"] = {
                "x_min": float(cache.get("x_min", float("nan"))),
                "x_max": float(cache.get("x_max", float("nan"))),
            }
    except Exception:
        pass

    try:
        if (pinn_support is not None) and (pinn_probs is not None):
            sp = np.asarray(pinn_support).reshape(-1)
            pr = np.asarray(pinn_probs).reshape(-1)
            pr = pr / (pr.sum() + 1e-30)
            pinn_info["policy_discrete"] = {
                "support_min": float(np.min(sp)),
                "support_max": float(np.max(sp)),
                "prob_min": float(np.min(pr)),
                "prob_max": float(np.max(pr)),
                "prob_std": float(np.std(pr)),
                "entropy": float(-(pr * np.log(pr + 1e-30)).sum()),
                "mean": float((sp * pr).sum()),
                "std": float(max(float((sp ** 2 * pr).sum() - ((sp * pr).sum()) ** 2), 0.0) ** 0.5),
            }
            pinn_info["policy_discrete"]["max_over_min_prob"] = float(
                pinn_info["policy_discrete"]["prob_max"] / (pinn_info["policy_discrete"]["prob_min"] + 1e-30)
            )
            gamma = float(self.params_train.get("entropy_alpha", 0.0))
            if gamma > 0 and "u_grid" in pinn_info:
                pinn_info["heuristic_warning"] = (
                    "If H_range/gamma is small, exp(H/gamma) ~ 1 => Gibbs nearly uniform."
                )
    except Exception:
        pass

    actor_info = {
        "type": getattr(self.net_actor, "actor_type", type(self.net_actor).__name__),
        "sample_space": "unknown (depends on actor implementation; see below fields)",
    }
    try:
        actor_info["params_train_actor_use_squash"] = bool(self.params_train.get("actor_use_squash", False))
        actor_info["params_train_delay_squash_steps"] = int(self.params_train.get("delay_squash_steps", 0))
        actor_info["params_train_squash_temp_init"] = float(self.params_train.get("squash_temp_init", float("nan")))
        actor_info["params_train_squash_temp_final"] = float(self.params_train.get("squash_temp_final", float("nan")))
        actor_info["pinn_u_min"] = float(self.params_train.get("pinn_u_min", float("nan")))
        actor_info["pinn_u_max"] = float(self.params_train.get("pinn_u_max", float("nan")))
    except Exception:
        pass

    for k in (
        "use_squash",
        "actor_use_squash",
        "delay_squash_steps",
        "squash_temp",
        "squash_temp_init",
        "squash_temp_final",
        "squash_steps",
    ):
        if hasattr(self.net_actor, k):
            try:
                v = getattr(self.net_actor, k)
                actor_info[f"actor_attr_{k}"] = float(v) if isinstance(v, (int, float, np.floating)) else str(v)
            except Exception:
                actor_info[f"actor_attr_{k}"] = "unreadable"

    try:
        out = self.net_actor(state)
        if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[0]):
            mu0 = out[0].detach().cpu().numpy().reshape(-1)
            actor_info["forward_mu"] = {"min": float(mu0.min()), "max": float(mu0.max()), "mean": float(mu0.mean())}
            if torch.is_tensor(out[1]):
                ls0 = out[1].detach().cpu().numpy().reshape(-1)
                actor_info["forward_log_std"] = {"min": float(ls0.min()), "max": float(ls0.max()), "mean": float(ls0.mean())}
        elif isinstance(out, dict):
            actor_info["forward_dict_keys"] = list(out.keys())
    except Exception as e:
        actor_info["forward_inspect_error"] = str(e)

    try:
        if u_flow_clean is not None:
            uf = np.asarray(u_flow_clean).reshape(-1)
            actor_info["sample_u"] = {
                "n": int(uf.size),
                "min": float(np.min(uf)),
                "max": float(np.max(uf)),
                "mean": float(np.mean(uf)),
                "std": float(np.std(uf)),
                "p01": float(np.quantile(uf, 0.01)),
                "p99": float(np.quantile(uf, 0.99)),
            }
    except Exception:
        pass

    entry = {
        "t": float(t_val),
        "x": float(x_val),
        "note": "Panel (c) debug dump: PINN Gibbs (discrete on u_grid) vs Actor sampling.",
        "pinn": pinn_info,
        "actor": actor_info,
    }

    save_path = os.path.join(output_folder, "policy_debug_panelc.json")
    try:
        data = []
        if os.path.isfile(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        data.append(entry)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[panel_c][debug_dump] WARNING: failed to write {save_path}: {e}")

    try:
        g = pinn_info.get("gamma_entropy", None)
        pinn_prob_ratio = None
        if "policy_discrete" in pinn_info:
            pinn_prob_ratio = pinn_info["policy_discrete"].get("max_over_min_prob", None)
        flow_std = actor_info.get("sample_u", {}).get("std", None)
        print(f"[panel_c][debug_dump] wrote policy_debug_panelc.json (t={t_val:.4f}, x={x_val:.6f}, gamma={g}, pinn_prob_ratio={pinn_prob_ratio}, flow_std={flow_std})")
    except Exception:
        pass


def plot_panel_c_conditional_distribution(
    self,
    t_values=None,
    x_values=None,
    n_samples_u=20000,
    plot_method="histogram",
    device=None,
    filename_suffix="",
    output_folder=None,
    ax=None,
    legend_font_size=11,
    legend_label_pinn=None,
    legend_label_flow=None,
    legend_loc=None,
    force_density=False,
):
    """
    Plot conditional distribution p(u|t, x) for selected (t, x) points.
    PINN policy is time-independent: pi*(u|x), while Actor is time-dependent: pi(u|t,x).
    """
    device = device or self.device
    # Use same device as net_actor so flow sampling and PINN eval don't hit cuda/cpu mismatch
    if hasattr(self, "net_actor") and self.net_actor is not None:
        try:
            device = next(self.net_actor.parameters()).device
        except Exception:
            pass
    output_folder = output_folder or self.output_folder
    os.makedirs(output_folder, exist_ok=True)
    if filename_suffix is None:
        filename_suffix = ""

    if (x_values is None) or (x_values == "XT0"):
        x_rep = getattr(self, "_panel_c_xt_rep", None)
        if x_rep is None:
            print("[panel_c] WARNING: x_values is None/XT0 but self._panel_c_xt_rep is not set.")
            return
        x_values = [float(x_rep)]

    if t_values is None:
        t_rep = getattr(self, "_panel_c_t_rep", None)
        if t_rep is None:
            t_rep = float(self.T)
        t_values = [float(t_rep)]

    if not isinstance(t_values, (list, tuple)):
        t_values = [t_values]
    if not isinstance(x_values, (list, tuple)):
        x_values = [x_values]

    n_t = len(t_values)
    n_x = len(x_values)

    single_ax = ax is not None
    if single_ax:
        t_values = [t_values[0]]
        x_values = [x_values[0]]
        n_t = 1
        n_x = 1
        axes = np.array([[ax]])
    else:
        fig, axes = plt.subplots(n_t, n_x, figsize=(5 * n_x, 4 * n_t), dpi=150)
        if n_t == 1 and n_x == 1:
            axes = np.array([[axes]])
        elif n_t == 1:
            axes = axes.reshape(1, -1) if hasattr(axes, "reshape") else np.array([axes])
        elif n_x == 1:
            axes = axes.reshape(-1, 1) if hasattr(axes, "reshape") else np.array([[ax] for ax in axes])
        else:
            axes = np.array(axes)

    print(f"[panel_c] Plotting conditional p(u|t,x) for {n_t * n_x} (t,x) combos.")
    print(f"[panel_c] Using n_samples_u={n_samples_u}, plot_method={plot_method}")
    print(f"[panel_c] (If this stage appears to hang: set CUDA_VISIBLE_DEVICES=0 or use --panel_c_n_samples_u 2000 for a quicker run.)")

    if plot_method == "kde":
        try:
            from scipy.stats import gaussian_kde
        except Exception as e:
            print(f"[panel_c] WARNING: scipy gaussian_kde import failed ({e}). Falling back to histogram.")
            plot_method = "histogram"
    flow_bins = int(getattr(self, "params_train", {}).get("panel_c_flow_bins", 40))
    flow_rwidth = float(getattr(self, "params_train", {}).get("panel_c_flow_rwidth", 0.6))
    if not np.isfinite(flow_rwidth):
        flow_rwidth = 0.6
    flow_rwidth = max(0.05, min(1.0, flow_rwidth))
    # When Flownet samples are very concentrated (low policy variance), use more bins so the
    # histogram doesn't collapse to a single Dirac-like bar over the full u range.
    flow_adaptive_bins = bool(getattr(self, "params_train", {}).get("panel_c_flow_adaptive_bins", True))
    flow_min_bins = int(getattr(self, "params_train", {}).get("panel_c_flow_min_bins", 120))
    # For 2x2 combined layout (single ax per call): use fixed bin count so all four p(u|t,Xt) panels
    # have the same number of bars and look consistent (avoids t=dt having fewer/wider bins).
    use_fixed_bins_2x2 = single_ax and bool(getattr(self, "params_train", {}).get("panel_c_2x2_fixed_bins", True))
    fixed_bins_2x2 = int(getattr(self, "params_train", {}).get("panel_c_2x2_bins", 60))
    # Optional: minimum KDE bandwidth (in u units) so narrow Flownet distribution still shows a visible bump.
    flow_kde_min_bw = float(getattr(self, "params_train", {}).get("panel_c_flow_kde_min_bw", 0.0))

    for i, t_val in enumerate(t_values):
        for j, x_val in enumerate(x_values):
            ax = axes[i, j]
            print(f"[panel_c] (t={float(t_val):.6f}, x={float(x_val):.6f}) evaluating PINN probs + sampling actor...", flush=True)

            try:
                x_min_cfg = self.params_train.get("pinn_x_min", None)
                x_max_cfg = self.params_train.get("pinn_x_max", None)
                if x_min_cfg is not None and x_max_cfg is not None:
                    x_min_cfg = float(x_min_cfg)
                    x_max_cfg = float(x_max_cfg)
                    # Clamp x to PINN domain; use floor 0.01 to avoid numerical issues at very small x
                    x_floor = max(x_min_cfg, 0.01)
                    if x_val < x_floor or x_val > x_max_cfg:
                        x_val = max(x_floor, min(x_max_cfg, float(x_val)))
                        print(
                            f"[panel_c] x clamped to PINN range: [{x_floor:.4g},{x_max_cfg:.4g}] -> x={x_val:.6f}"
                        )
            except Exception:
                pass

            # Same (t, x) for both PINN and Flow: one (t_val, x_val) per subplot → fair comparison of p(u|t,Xt).
            t_tensor = torch.full((1, 1), float(t_val), device=device, dtype=torch.float32)
            x_tensor = torch.full((1, 1), float(x_val), device=device, dtype=torch.float32)
            # Use model's state preprocessing so actor gets same input format as during training (e.g. t/T, log(x))
            if hasattr(self, "_preprocess_state"):
                state = self._preprocess_state(t_tensor, x_tensor)
            else:
                state = torch.cat([t_tensor, x_tensor], dim=1)

            shared_noise = bool(getattr(self, "params_train", {}).get("panel_c_shared_noise", False))
            shared_noise_seed = int(getattr(self, "params_train", {}).get("panel_c_shared_noise_seed", 12345))
            eps_shared = None  # when panel_c_shared_noise=True, same eps for PINN and Flow
            if shared_noise:
                with torch.random.fork_rng(devices=None, enabled=True):
                    torch.manual_seed(shared_noise_seed)
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.manual_seed_all(shared_noise_seed)
                    eps_shared = torch.randn(n_samples_u, 1, device=device, dtype=torch.float32)

            pinn_probs = None
            pinn_support = None
            pinn_density = None
            u_pinn = None
            plot_pinn_as_mass = not force_density and bool(getattr(self, "params_train", {}).get("panel_c_pinn_plot_as_mass", False))

            try:
                print(f"[panel_c]   step 1/3: compute_exact_entropy (PINN forward)...", flush=True)
                policy_pinn, _ = self.compute_exact_entropy(x_tensor)
                print(f"[panel_c]   step 2/3: PINN policy done.", flush=True)

                if isinstance(policy_pinn, self.DiscretePolicy):
                    pinn_support = policy_pinn.support_u.detach().cpu().numpy().reshape(-1)
                    pinn_probs = policy_pinn.probs.detach().cpu().numpy()
                    if pinn_probs.ndim == 2:
                        pinn_probs = pinn_probs[0]
                    pinn_probs = pinn_probs.reshape(-1)
                    pinn_probs = pinn_probs / (pinn_probs.sum() + 1e-30)
                    if len(pinn_support) > 1:
                        du = float(np.median(np.diff(pinn_support)))
                    else:
                        du = 1.0
                    pinn_density = pinn_probs / max(du, 1e-12)
                    # Clip to u range where x_shift = x*(1+z*u) stays inside PINN x domain (same as combined plot)
                    try:
                        z_val = float(self.params_equ.get("z", 0.0))
                        pinn_x_max = float(self.params_train.get("pinn_x_max", 80.0))
                        if z_val > 1e-8 and float(x_val) > 1e-12 and pinn_x_max > float(x_val):
                            u_valid_max = (pinn_x_max / float(x_val) - 1.0) / z_val
                            valid = pinn_support <= (u_valid_max + 1e-8)
                            if not np.all(valid):
                                pinn_support = pinn_support[valid].copy()
                                pinn_density = pinn_density[valid].copy()
                                pinn_probs = pinn_probs[valid].copy()
                                pinn_probs = pinn_probs / (pinn_probs.sum() + 1e-30)
                                print(
                                    f"[panel_c] PINN Gibbs clipped to u<={u_valid_max:.4f} "
                                    f"(x={float(x_val):.4f}, pinn_x_max={pinn_x_max:.4f}, z={z_val:.4f})"
                                )
                    except Exception:
                        pass
                    try:
                        u_min_s = float(np.min(pinn_support))
                        u_max_s = float(np.max(pinn_support))
                        u_min_cfg = self.params_train.get("pinn_u_min", None)
                        u_max_cfg = self.params_train.get("pinn_u_max", None)
                        u_min_eval = self.params_train.get("pinn_eval_u_min", None)
                        u_max_eval = self.params_train.get("pinn_eval_u_max", None)
                        print(
                            "[panel_c] PINN support u_range="
                            f"[{u_min_s:.4g}, {u_max_s:.4g}] "
                            f"(cfg pinn_u=[{u_min_cfg},{u_max_cfg}], "
                            f"pinn_eval_u=[{u_min_eval},{u_max_eval}], n_u={len(pinn_support)})"
                        )
                    except Exception:
                        pass
                else:
                    if shared_noise and eps_shared is not None:
                        # Same eps for PINN and Flow so p(u|t,x) compares same-quantile outputs
                        eps = eps_shared
                        DiscretePolicy = getattr(self, "DiscretePolicy", None)
                        if DiscretePolicy is not None and isinstance(policy_pinn, DiscretePolicy):
                            q = 0.5 * (1.0 + torch.erf(eps / math.sqrt(2.0)))
                            probs_expand = policy_pinn.probs.expand(n_samples_u, -1)
                            policy_batch = DiscretePolicy(policy_pinn.support_u, probs_expand)
                            u_pinn = policy_batch.icdf(q).detach().cpu().numpy().reshape(-1)
                        elif hasattr(policy_pinn, "loc") and hasattr(policy_pinn, "scale"):
                            loc = getattr(policy_pinn, "loc").expand(n_samples_u, 1)
                            scale = getattr(policy_pinn, "scale").expand(n_samples_u, 1)
                            u_pinn = (loc + scale * eps).detach().cpu().numpy().reshape(-1)
                        elif hasattr(policy_pinn, "sample"):
                            u_pinn = policy_pinn.sample((n_samples_u,)).detach().cpu().numpy().reshape(-1)
                        else:
                            raise RuntimeError(f"PINN policy type {type(policy_pinn)} has no loc/scale and no sample().")
                        print(f"[panel_c]   PINN samples from shared eps (n={n_samples_u}, seed={shared_noise_seed}).", flush=True)
                    elif hasattr(policy_pinn, "sample"):
                        print(f"[panel_c]   sampling PINN policy (n={n_samples_u})...", flush=True)
                        u_pinn = policy_pinn.sample((n_samples_u,)).detach().cpu().numpy().reshape(-1)
                    else:
                        raise RuntimeError(f"PINN policy is not DiscretePolicy and has no sample(). type={type(policy_pinn)}")
            except Exception as e:
                print(f"[panel_c] WARNING: PINN policy eval failed at (t={t_val}, x={x_val}): {e}")
                import traceback
                traceback.print_exc()

            u_flow = None
            try:
                if shared_noise and eps_shared is not None:
                    u_flow_list = []
                    batch_size = 1000
                    for b in range(0, n_samples_u, batch_size):
                        end = min(b + batch_size, n_samples_u)
                        bs = end - b
                        state_batch = state.expand(bs, -1)
                        eps_batch = eps_shared[b:end]
                        action, _, _, _, _, _, _, _ = self.net_actor.sample(state_batch, deterministic=False, noise=eps_batch)
                        u_flow_list.append(action.detach().cpu())
                    u_flow = torch.cat(u_flow_list, dim=0).numpy().reshape(-1)
                    print(f"[panel_c]   Flow samples from same eps (seed={shared_noise_seed}).", flush=True)
                else:
                    u_flow_list = []
                    batch_size = 1000
                    n_batches = (n_samples_u + batch_size - 1) // batch_size
                    print(f"[panel_c]   step 3/3: sampling flow net (n_batches={n_batches}, bs={batch_size})...", flush=True)

                    for b in range(n_batches):
                        bs = min(batch_size, n_samples_u - b * batch_size)
                        state_batch = state.expand(bs, -1)
                        action, _, _, _, _, _, _, _ = self.net_actor.sample(state_batch, deterministic=False)
                        u_flow_list.append(action.detach().cpu())
                        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
                            print(f"[panel_c]     flow batch {b + 1}/{n_batches}", flush=True)

                    u_flow = torch.cat(u_flow_list, dim=0).numpy().reshape(-1)
                print(f"[panel_c]   (t,x)=({float(t_val):.4f},{float(x_val):.4f}) done.", flush=True)
            except Exception as e:
                print(f"[panel_c] WARNING: Flownet sampling failed at (t={t_val}, x={x_val}): {e}")
                import traceback
                traceback.print_exc()

            if u_flow is None:
                ax.text(0.5, 0.5, "Sampling failed", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"t={float(t_val):.2f}, x={float(x_val):.2f}")
                ax.grid(True, alpha=0.3)
                continue

            valid_flow = np.isfinite(u_flow)
            u_flow_clean = u_flow[valid_flow]
            if len(u_flow_clean) == 0:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"t={float(t_val):.2f}, x={float(x_val):.2f}")
                ax.grid(True, alpha=0.3)
                continue

            try:
                _dump_panel_c_policy_debug(
                    self,
                    t_val=float(t_val),
                    x_val=float(x_val),
                    x_tensor=x_tensor,
                    state=state,
                    output_folder=output_folder,
                    policy_pinn=policy_pinn if "policy_pinn" in locals() else None,
                    pinn_support=pinn_support,
                    pinn_probs=pinn_probs,
                    u_flow_clean=u_flow_clean,
                )
            except Exception as e:
                print(f"[panel_c] WARNING: debug dump failed: {e}")

            if pinn_support is not None:
                pinn_support_clean = pinn_support[np.isfinite(pinn_support)]
                u_min = float(min(pinn_support_clean.min(), u_flow_clean.min()))
                u_max = float(max(pinn_support_clean.max(), u_flow_clean.max()))
            elif u_pinn is not None:
                valid_pinn = np.isfinite(u_pinn)
                u_pinn_clean = u_pinn[valid_pinn]
                if len(u_pinn_clean) == 0:
                    ax.text(0.5, 0.5, "No valid PINN data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"t={float(t_val):.2f}, x={float(x_val):.2f}")
                    ax.grid(True, alpha=0.3)
                    continue
                u_min = float(min(u_pinn_clean.min(), u_flow_clean.min()))
                u_max = float(max(u_pinn_clean.max(), u_flow_clean.max()))
            else:
                # Flow-only: no PINN data, use flow range for axis
                u_min = float(u_flow_clean.min())
                u_max = float(u_flow_clean.max())
            action_min = self.params_train.get("pinn_u_min", None)
            action_max = self.params_train.get("pinn_u_max", None)
            if action_min is not None and action_max is not None:
                u_min_plot = float(action_min) - 1.0
                u_max_plot = float(action_max) + 1.0
            else:
                u_range = max(1e-12, u_max - u_min)
                u_min_plot = u_min - 0.05 * u_range
                u_max_plot = u_max + 0.05 * u_range

            # Flownet: when policy variance is small, samples sit in a narrow range → one bar looks like Dirac.
            # Use more bins so the histogram has resolution; optional min KDE bandwidth for a visible bump.
            # For 2x2 combined layout use fixed bin count so all four panels have the same number of bars.
            if use_fixed_bins_2x2:
                n_bins_flow = fixed_bins_2x2
            else:
                u_range_flow = float(np.ptp(u_flow_clean))
                if flow_adaptive_bins and u_range_flow < 0.5 * (u_max_plot - u_min_plot):
                    n_bins_flow = max(flow_bins, min(300, flow_min_bins))
                else:
                    n_bins_flow = flow_bins
            flow_std = float(np.std(u_flow_clean))
            _flow_lbl_default_kde = "Flownet (KDE)"
            _flow_lbl_default_hist = "Flownet (samples)"
            lbl_flow_kde = legend_label_flow if legend_label_flow is not None else _flow_lbl_default_kde
            lbl_flow_hist = legend_label_flow if legend_label_flow is not None else _flow_lbl_default_hist
            lbl_pinn_alt = legend_label_pinn if legend_label_pinn is not None else "PINN (sample KDE)"
            _legend_loc = legend_loc if legend_loc is not None else "best"

            if (pinn_support is not None) and (pinn_density is not None):
                # When panel_c_pinn_plot_as_mass: plot PINN as probability mass (y in [0,1])
                # to avoid Dirac-like spike (density = probs/du is huge at one point when policy is peaked).
                if plot_pinn_as_mass:
                    pinn_y = pinn_probs
                else:
                    pinn_y = pinn_density
                if legend_label_pinn is not None:
                    pinn_label = legend_label_pinn
                elif plot_pinn_as_mass:
                    pinn_label = "PINN (Gibbs probs)"
                else:
                    pinn_label = "PINN (Gibbs probs / du)"
                if plot_method == "bar":
                    if len(pinn_support) > 1:
                        width = float((pinn_support[1] - pinn_support[0]) * 0.8)
                    else:
                        width = 0.01
                    ax.bar(
                        pinn_support, pinn_y, width=width, alpha=0.7,
                        label=pinn_label, edgecolor="black", linewidth=0.5,
                    )
                else:
                    ax.plot(
                        pinn_support, pinn_y, linewidth=2.5, alpha=0.9,
                        label=pinn_label,
                        color="darkviolet",
                    )

                if plot_method == "kde" and not plot_pinn_as_mass:
                    try:
                        kde_flow = gaussian_kde(u_flow_clean)
                        if flow_kde_min_bw > 0 and flow_std < flow_kde_min_bw:
                            kde_flow.set_bandwidth(flow_kde_min_bw)
                        u_plot = np.linspace(u_min_plot, u_max_plot, 250)
                        ax.plot(u_plot, kde_flow(u_plot), label=lbl_flow_kde, linewidth=2, alpha=0.85)
                    except Exception as e:
                        print(f"[panel_c] WARNING: Actor KDE failed ({e}), fallback to histogram.")
                        plot_method = "histogram"
                        _flow_weights = (1.0 / len(u_flow_clean)) * np.ones_like(u_flow_clean) if plot_pinn_as_mass else None
                        ax.hist(
                            u_flow_clean, bins=n_bins_flow, alpha=0.55, density=not plot_pinn_as_mass,
                            weights=_flow_weights, label=lbl_flow_hist, color="#FFE0B2", edgecolor="#FFCC80", linewidth=0.5,
                            rwidth=flow_rwidth,
                        )
                else:
                    _flow_weights = (1.0 / len(u_flow_clean)) * np.ones_like(u_flow_clean) if plot_pinn_as_mass else None
                    ax.hist(
                        u_flow_clean, bins=n_bins_flow, alpha=0.55, density=not plot_pinn_as_mass,
                        weights=_flow_weights, label=lbl_flow_hist, color="#FFE0B2", edgecolor="#FFCC80", linewidth=0.5,
                        rwidth=flow_rwidth,
                    )

            elif plot_method == "kde":
                if pinn_support is None:
                    try:
                        kde_flow = gaussian_kde(u_flow_clean)
                        if flow_kde_min_bw > 0 and flow_std < flow_kde_min_bw:
                            kde_flow.set_bandwidth(flow_kde_min_bw)
                        u_plot = np.linspace(u_min_plot, u_max_plot, 250)
                        if u_pinn is not None and len(u_pinn_clean) > 0:
                            kde_pinn = gaussian_kde(u_pinn_clean)
                            ax.plot(u_plot, kde_pinn(u_plot), label=lbl_pinn_alt, linewidth=2, alpha=0.8)
                        ax.plot(u_plot, kde_flow(u_plot), label=lbl_flow_kde, linewidth=2, alpha=0.8)
                    except Exception as e:
                        print(f"[panel_c] WARNING: KDE failed ({e}), fallback to histogram.")
                        plot_method = "histogram"

            if plot_method == "histogram" and pinn_support is None:
                n_bins = 80
                if u_pinn is not None and len(u_pinn_clean) > 0:
                    ax.hist(
                        u_pinn_clean, bins=n_bins, alpha=0.6, density=True,
                        label=(legend_label_pinn if legend_label_pinn is not None else "PINN (samples)"),
                        color="darkviolet", edgecolor="black", linewidth=0.5,
                    )
                ax.hist(
                    u_flow_clean, bins=n_bins_flow, alpha=0.6, density=True,
                    label=lbl_flow_hist, color="#FFE0B2", edgecolor="#FFCC80", linewidth=0.5,
                    rwidth=flow_rwidth,
                )

            ax.set_xlim(u_min_plot, u_max_plot)
            ax.set_xlabel("$u$", fontsize=FONT_SIZE)
            ax.set_ylabel("$P(u\\mid t,x)$" if plot_pinn_as_mass else "$p(u\\mid t,x)$", fontsize=FONT_SIZE)
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.legend(fontsize=legend_font_size, loc=_legend_loc, framealpha=0.5, edgecolor="gray")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"$t={float(t_val):.2f}$, $x={float(x_val):.4f}$", fontsize=FONT_SIZE)

            try:
                flow_mean = float(np.mean(u_flow_clean))
                flow_std = float(np.std(u_flow_clean))
                if pinn_support is not None and pinn_probs is not None:
                    pinn_mean = float((pinn_support * pinn_probs).sum())
                    pinn_var = float((pinn_support ** 2 * pinn_probs).sum() - pinn_mean ** 2)
                    pinn_std = float(max(pinn_var, 0.0) ** 0.5)
                    print(
                        "[panel_c] stats: PINN mean/std(from probs)="
                        f"{pinn_mean:.6f}/{pinn_std:.6f}, Flow mean/std={flow_mean:.6f}/{flow_std:.6f}"
                    )
                else:
                    pinn_mean = float(np.mean(u_pinn_clean))
                    pinn_std = float(np.std(u_pinn_clean))
                    print(
                        "[panel_c] stats: PINN mean/std(samples)="
                        f"{pinn_mean:.6f}/{pinn_std:.6f}, Flow mean/std={flow_mean:.6f}/{flow_std:.6f}"
                    )
                if flow_std < 0.05:
                    print(
                        "[panel_c] Flownet std very small → plot can look Dirac-like; "
                        "use panel_c_flow_adaptive_bins=True (default) and/or panel_c_flow_kde_min_bw=0.05"
                    )
            except Exception:
                pass

    if not single_ax:
        plt.tight_layout()
        itr_val = getattr(self, "current_itr", None)
        if itr_val is None:
            itr_val = self.params_train.get("current_itr", None)
        if itr_val is not None:
            itr_tag = f"_itr{int(itr_val)}"
            if itr_tag not in filename_suffix:
                filename_suffix = f"{filename_suffix}{itr_tag}"
        # Tag filename so we can distinguish eval runs done with frozen script vs current script.
        _use_frozen = os.environ.get("MERTON_USE_FROZEN_SCRIPT_EVAL", "").strip()
        if _use_frozen == "1":
            _tag = "_use_frozen_script_eval"
        elif _use_frozen == "0":
            _tag = "_no_frozen_script_eval"
        else:
            _tag = ""
        if _tag and (_tag not in filename_suffix):
            filename_suffix = f"{filename_suffix}{_tag}"

        filename = f"merton_panel_c_conditional_distribution{filename_suffix}"
        save_path = os.path.join(output_folder, f"{filename}.pdf")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[panel_c] Saved conditional distribution plot to {save_path}")
        plt.close(fig)
        return None
    return axes[0, 0]
