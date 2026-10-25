import os
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

import glob
import json
import re

from experiment_utils import _as_1d_numpy




@torch.no_grad()
def plot_V0_xgrid_pinn_vs_taylor(
    self,
    x_grid=None,
    x_min=None,
    x_max=None,
    n_x=200,
    n_u=150,
    u_min=-1.0,
    u_max=10.0,
    n_iter=3000,
    lr=5e-4,
    width=64,
    depth=3,
    filename=None,
    device=None,
    output_folder=None,
):
    """
    Plot V(0,x) on x_grid using:
      1) PINN-solved value function
      2) Taylor (no entropy): h(t) x^p / p + g(t) at t=0
      3) Taylor with entropy: h(t) x^p / p + g(t) + a(t) log x at t=0
    """

    def _to_scalar(v):
        if torch.is_tensor(v):
            vv = v.detach()
            if vv.numel() == 1:
                return float(vv.item())
            return float(vv.reshape(-1)[0].item())
        return float(v)

    device = device or self.device
    output_folder = output_folder or self.output_folder
    os.makedirs(output_folder, exist_ok=True)
    if filename is None:
        gamma_val = float(self.params_train.get("entropy_alpha", 0.0))
        filename = f"merton_power_V0_xgrid_pinn_vs_taylor_gamma_{gamma_val}"

    if x_grid is None:
        if x_min is None or x_max is None:
            center = _to_scalar(self.Xi)
            radius = float(getattr(self, "X0_init_radius", 0.1))
            if x_min is None:
                x_min = max(1e-6, center - radius)
            if x_max is None:
                x_max = center + radius
            if x_min <= 0:
                x_min = 1e-6
        x_grid = torch.linspace(x_min, x_max, n_x, device=device, dtype=torch.float32).view(-1, 1)
    else:
        if not torch.is_tensor(x_grid):
            x_grid = torch.as_tensor(x_grid, device=device, dtype=torch.float32)
        if x_grid.dim() == 1:
            x_grid = x_grid.view(-1, 1)
        else:
            x_grid = x_grid.reshape(-1, 1)
        if x_min is None:
            x_min = float(x_grid.min().item())
        if x_max is None:
            x_max = float(x_grid.max().item())

    x_safe = torch.clamp(x_grid, min=1e-8)
    if x_min == x_max:
        x_max = x_min + 1.0

    mu = torch.as_tensor(self.params_equ["mu"], device=device, dtype=x_safe.dtype)
    r = torch.as_tensor(self.params_equ["r"], device=device, dtype=x_safe.dtype)
    sigma = torch.as_tensor(self.params_equ["sigma"], device=device, dtype=x_safe.dtype)
    lam = torch.as_tensor(self.params_equ["lambda"], device=device, dtype=x_safe.dtype)
    z = torch.as_tensor(self.params_equ["z"], device=device, dtype=x_safe.dtype)
    beta = torch.as_tensor(self.params_equ["beta"], device=device, dtype=x_safe.dtype)
    gamma = torch.as_tensor(self.params_train["entropy_alpha"], device=device, dtype=x_safe.dtype)

    cache = self._solve_value_function_pinn(
        mu,
        r,
        sigma,
        lam,
        z,
        beta,
        gamma,
        device,
        x_safe.dtype,
        x_min=float(x_min),
        x_max=float(x_max),
        n_x=n_x,
        n_u=n_u,
        u_min=u_min,
        u_max=u_max,
        n_iter=n_iter,
        lr=lr,
        width=width,
        depth=depth,
        use_cache=True,
    )
    net = cache["net"]
    v_pinn = self._value_net_forward_chunked(
        net,
        x_safe,
        cache["x_min"],
        cache["x_max"],
        chunk=self.params_train.get("pinn_chunk", 200000),
        use_no_grad=True,
    ).squeeze(-1)

    _, v_taylor = self.compute_exact_no_exploratory(x_safe)
    v_taylor = v_taylor.squeeze(-1)

    if float(self.params_train.get("entropy_alpha", 0.0)) == 0.0:
        v_taylor_with_at = v_taylor
    else:
        _, v_taylor_with_at = self.compute_exact_exploratory_taylor_and_ansatz(x_safe)
        v_taylor_with_at = v_taylor_with_at.squeeze(-1)

    x_np = x_safe.squeeze(-1).detach().cpu().numpy()
    v_pinn_np = v_pinn.detach().cpu().numpy()
    v_taylor_np = v_taylor.detach().cpu().numpy()
    v_taylor_with_at_np = v_taylor_with_at.detach().cpu().numpy()

    plt.figure(figsize=(6.0, 4.0), dpi=300)
    plt.plot(x_np, v_pinn_np, color="tab:blue", label=r"$V^{PINN}(0,x)$")
    plt.plot(x_np, v_taylor_np, "--", color="tab:orange", label=r"$h(0)\frac{x^p}{p}+g(0)$")
    plt.plot(
        x_np,
        v_taylor_with_at_np,
        "-.",
        color="tab:green",
        label=r"$h(0)\frac{x^p}{p}+g(0)+a(0)\log x$",
    )
    gamma_val = float(self.params_train.get("entropy_alpha", 0.0))
    plt.xlabel(rf"$x\ (\gamma={gamma_val})$")
    plt.ylabel(r"$V(0,x)$")
    plt.legend(framealpha=0.5, edgecolor="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{filename}.pdf"), dpi=300, bbox_inches="tight")
    plt.show()


def load_training_error_history(output_folder):
    """Read persisted PINN evaluation metrics, keeping the latest valid value per iteration.

    Loss rows normally have empty metric cells between evaluations. They must
    neither erase an earlier valid value nor be drawn as missing line segments.
    The dedicated evaluation log takes precedence over the loss-log copy.
    """
    metrics = ("pinn_value_error", "pinn_control_error")
    records = {}
    for filename in ("loss_history.csv", "pinn_taylor_errors_every20.csv"):
        path = os.path.join(output_folder, filename)
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    iteration = int(row["iter"])
                except (KeyError, TypeError, ValueError):
                    continue
                if iteration < 0:
                    continue
                for metric in metrics:
                    try:
                        value = float(row.get(metric, ""))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value) and value >= 0:
                        records.setdefault(iteration, {})[metric] = value
    iterations = np.asarray(sorted(records), dtype=int)
    values = [np.asarray([records[it].get(metric, np.nan) for it in iterations], dtype=float)
              for metric in metrics]
    return iterations, values


def plot_training_errors(self, filename="merton_train"):
    """Plot actual evaluation history in training and checkpoint-only processes."""
    iterations, series = load_training_error_history(self.output_folder)
    if iterations.size:
        labels = ["PINN value error", "PINN control error"]
    else:
        # Older runs may only have in-memory diagnostics. Their definitions are
        # different from the persisted PINN metrics, so keep them as a fallback.
        series = [getattr(self, "Y_error_list", []), getattr(self, "control_error_list", [])]
        iterations = None
        labels = ["Training value diagnostic", "Training control diagnostic"]
    return plot_list(
        self, series, "Relative error", labels,
        color=["#E7483D", "#6565AE", "#8FC751"],
        ylabel_log=True, baseline=0.01, filename=filename, iterations=iterations,
    )


def plot_list(
    self,
    list_,
    ylabel,
    legend,
    color=None,
    title=None,
    ylabel_log=False,
    xticks=None,
    baseline=None,
    filename=None,
    iterations=None,
):
    # Normalize data before creating a figure, so missing history cannot produce
    # a misleading PDF containing only the reference line.
    series = []
    for data in list_:
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        elif isinstance(data, (list, tuple)) and len(data) > 0 and torch.is_tensor(data[0]):
            data = np.stack([d.detach().cpu().numpy() for d in data])
        elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], np.ndarray):
            data = np.stack(data)
        else:
            data = np.asarray(data)
        if data.ndim == 3:
            data = data[:, 0, :].mean(axis=-1)
        elif data.ndim == 2:
            data = data[:, 0]
        elif data.ndim == 0:
            data = data.reshape(1)
        data = np.asarray(data, dtype=float).reshape(-1)
        x = np.arange(data.size) if iterations is None else np.asarray(iterations)
        if x.shape != data.shape:
            raise ValueError("iterations must match the length of each plotted series")
        valid = np.isfinite(x) & np.isfinite(data)
        if ylabel_log:
            valid &= data > 0
        series.append((x[valid], data[valid], np.count_nonzero(valid) < data.size))

    if not any(y.size for _, y, _ in series):
        print(f"[plot] {filename or 'history'} skipped: no finite {'positive ' if ylabel_log else ''}error history.")
        return None

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=300)
    visible_values = []
    for i, (x, y, sparse) in enumerate(series):
        if not y.size:
            continue
        ax.plot(x, y, label=legend[i] if legend is not None else None,
                color=color[i % len(color)] if color is not None else None,
                marker="." if sparse or y.size == 1 else None, markersize=4)
        visible_values.append(y)

    if baseline is not None:
        baseline = float(baseline.detach().cpu().item()) if torch.is_tensor(baseline) else float(baseline)
        if np.isfinite(baseline) and (not ylabel_log or baseline > 0):
            ax.axhline(baseline, linestyle="--", color=color[-1] if color else None,
                       label=f"Reference ({baseline:g})")
            visible_values.append(np.asarray([baseline]))
    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if iterations is not None:
        ax.set_xlim(left=0)
    if xticks is not None:
        ax.set_xticks(xticks[0], xticks[1])
    if title is not None:
        ax.set_title(title)
    if ylabel_log:
        ax.set_yscale("log")
        positive = np.concatenate(visible_values)
        ax.set_ylim(positive.min() * 0.8, positive.max() * 1.25)
    elif filename is not None and "control" in filename.lower():
        ax.set_ylim(-0.2, 1)
        ax.set_yticks([-0.2, 0, 0.25, 0.5, 0.75, 1])
    fig.tight_layout()
    if filename is not None:
        os.makedirs(self.output_folder, exist_ok=True)
        fig.savefig(os.path.join(self.output_folder, f"{filename}.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_loss(self, CriticLoss, ActorLoss, filename):
    CriticLoss = _as_1d_numpy(CriticLoss)
    ActorLoss = _as_1d_numpy(ActorLoss)
    fig, ax1 = plt.subplots(dpi=300)
    ax1.set_xlabel("iteration")
    plt.gca().set_yscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("ActorLoss", color="#6565AE")
    ax2.plot(ActorLoss, color="#6565AE", label="ActorLoss", zorder=1)
    ax2.tick_params(axis="y", labelcolor="#6565AE")
    y_max = float(np.max(ActorLoss))
    y_min = float(np.min(ActorLoss)) * 0.8
    ax2.set_ylim(y_min, y_max)
    medium = (y_max + y_min) / 2
    ax2.set_yticks([y_min, medium, y_max])

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", framealpha=0.5, edgecolor="gray")

    ax1.plot(CriticLoss, color="#E7483D", label="CriticLoss", zorder=2)
    ax1.tick_params(axis="y", labelcolor="#E7483D")

    if filename is not None:
        plt.savefig(os.path.join(self.output_folder, f"{filename}.pdf"), dpi=300, bbox_inches="tight")
    plt.show()


def append_training_diagnostics(self, iteration, kl_loss, policy_mean_error):
    """Persist diagnostics independently of optional W&B and in-memory lists."""
    path = os.path.join(self.output_folder, "training_diagnostics.csv")
    def cell(value):
        if value is None:
            return ""
        value = float(value.detach().cpu().mean()) if torch.is_tensor(value) else float(np.asarray(value).mean())
        return value if np.isfinite(value) else ""
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(("iter", "base_gaussian_kl", "policy_mean_relative_mse"))
        writer.writerow((int(iteration), cell(kl_loss), cell(policy_mean_error)))


def load_training_diagnostics(output_folder):
    path = os.path.join(output_folder, "training_diagnostics.csv")
    records = {}
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    iteration = int(row["iter"])
                except (KeyError, TypeError, ValueError):
                    continue
                for key in ("base_gaussian_kl", "policy_mean_relative_mse"):
                    try:
                        value = float(row.get(key, ""))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value) and value >= 0:
                        records.setdefault(iteration, {})[key] = value
    iterations = np.asarray(sorted(records), dtype=int)
    return iterations, [np.asarray([records[it].get(key, np.nan) for it in iterations])
                        for key in ("base_gaussian_kl", "policy_mean_relative_mse")]


def _diagnostic_series(values):
    if values is None:
        return np.array([], dtype=float)
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    elif isinstance(values, (list, tuple)):
        values = [v.detach().cpu().numpy() if torch.is_tensor(v) else v for v in values]
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1).mean(axis=1)
    return values


def _plot_diagnostic_axis(ax, iterations, values, label, color, ylabel, title):
    valid = np.isfinite(iterations) & np.isfinite(values) & (values >= 0)
    if np.any(values[valid] > 0):
        valid &= values > 0
        ax.set_yscale("log")
    if valid.any():
        # Connect measured points across NaN gaps and show isolated observations.
        ax.plot(iterations[valid], values[valid], color=color, label=label,
                marker="." if not valid.all() or valid.sum() == 1 else None, markersize=3)
        ax.legend(loc="best", framealpha=0.5, edgecolor="gray", fontsize=9)
        positive = values[valid][values[valid] > 0]
        if positive.size and positive.max()/positive.min() < 10:
            # Avoid long scientific minor-tick labels spilling into the next panel.
            from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.text(0.5, 0.5, "No recorded observations", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)


def plot_kl_std_loss(self, kl_loss=None, mu_of_policy_error=None, filename=None):
    iterations, persisted = load_training_diagnostics(self.output_folder)
    kl = _diagnostic_series(kl_loss)
    mu = _diagnostic_series(mu_of_policy_error)
    kl_x, mu_x = np.arange(kl.size), np.arange(mu.size)
    if iterations.size:
        kl_x, mu_x = iterations, iterations
        kl, mu = persisted
    if not np.any(np.isfinite(kl)):
        # Old runs did not persist this curve. Do not overwrite their existing
        # KL panel with an empty figure during checkpoint-only evaluation.
        print("[plot] KL history not recorded; keeping any existing diagnostic PDF.")
        return None
    mu_label, mu_ylabel = "Policy mean relative MSE", "Relative MSE"
    if not np.any(np.isfinite(mu)):
        eval_x, eval_series = load_training_error_history(self.output_folder)
        if eval_x.size:
            mu_x, mu = eval_x, eval_series[1]
            mu_label, mu_ylabel = "PINN mean error (evaluation)", "Relative L2 error"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4), dpi=300)
    _plot_diagnostic_axis(axes[0], kl_x, kl, "Base Gaussian KL", "#E7483D", "KL loss", "KL Loss Convergence")
    _plot_diagnostic_axis(axes[1], mu_x, mu, mu_label, "#009E73", mu_ylabel, "Policy Mean Error")
    fig.tight_layout()
    if filename is not None:
        fig.savefig(os.path.join(self.output_folder, f"{filename}.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    return fig, axes


def plot_controlfunction(self, n=1001, region=[9, 11], filename=None, sample_exact=False):
    t = torch.linspace(0, self.T, n)
    x = torch.linspace(region[0], region[1], n)
    T, X = torch.meshgrid(t, x)
    Z = torch.cat((T.unsqueeze(2), X.unsqueeze(2)), dim=2)
    Z = Z.to(self.device)

    if getattr(self.net_actor, "actor_type", "") in (
        "mog_squash",
        "hyper_gaussian",
        "hyper_gaussian_split",
        "gaussian_two_stage",
    ):
        mu, log_sigma, feat = self.net_actor(Z)
        # mixture_mean already applies squash internally, so no need to squash again
        # Use same K_mean_eval as training for consistency
        K_mean_eval = int(self.params_train["K_mean_eval"])
        ZZ1 = self.net_actor.mixture_mean(Z, n_samples=K_mean_eval, antithetic=True)
    else:
        ZZ1_mu, ZZ1_logsigma, _ = self.net_actor(Z)
        ZZ1 = ZZ1_mu
    print("ZZ1", ZZ1.shape)  # torch.Size([1001, 1001, 1])

    if self.params_train["entropy_alpha"] == 0:
        u_exact_row, _ = self.compute_exact_no_exploratory(x)
    else:
        optimal_policy, _ = self.compute_exact_entropy(x)
        mu_x = optimal_policy.loc.view(1, n)  # [1, n]
        # print("mu_x.shape", mu_x.shape) # torch.Size([1, 1001])
        if sample_exact:
            sig_x = optimal_policy.scale.view(1, n)  # [1, n]
            u_exact_row = mu_x + sig_x * torch.randn_like(mu_x)
        else:
            u_exact_row = mu_x  

    u_exact = u_exact_row.repeat(n, 1)  # [n, n]
    # print("u_exact.shape", u_exact.shape) # torch.Size([1001, 1001])

    
    denom = torch.clamp(torch.abs(u_exact), min=1e-12)
    if len(ZZ1.shape) == 3 and len(u_exact.shape) == 2:
        ZZ1 = ZZ1.squeeze(-1)
    err = torch.abs(ZZ1.to(self.device) - u_exact.to(self.device)) / denom.to(self.device)
    print("err.shape", err.shape)
    err = err.detach().cpu().numpy()

    f1 = plt.imshow(
        err,
        interpolation="nearest",
        cmap="rainbow",
        extent=[region[0], region[1], 0, self.T],
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(f1)
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    plt.title(r"$|(\hat{u}(t,x)-u^*(t,x))/u^*(t,x)|$")
    if filename is not None:
        plt.savefig(os.path.join(self.output_folder, "%s.pdf" % filename), dpi=300, bbox_inches="tight")
    plt.show()


def plot_J_control_function(J_list, save_dir, J: bool = True):
    """
    Plot (and save) either J-function sequence or control sequence over time steps.

    This utility is used by the optimizer entrypoint when training from scratch (no ckpt),
    where `model.J_list` / `model.control_list` are lists of torch tensors.
    """
    if J is True:
        J_array = np.array([j.detach().cpu().numpy() if torch.is_tensor(j) else np.asarray(j) for j in J_list])

        if J_array.ndim == 3:  # [N, A, B] → mean over last two dims → [N]
            J_array = J_array.mean(axis=(1, 2))
        elif J_array.ndim == 2:  # [N, A] → mean over second dim → [N]
            J_array = J_array.mean(axis=1)
        elif J_array.ndim == 4:  # [N, A, B, C] → mean over last three dims → [N]
            J_array = J_array.mean(axis=(1, 2, 3))
        elif J_array.ndim == 1:
            pass
        else:
            raise ValueError(f"Unsupported J_list shape: {J_array.shape}")

        t = np.arange(len(J_array))
        plt.figure(figsize=(6.0, 4.0), dpi=300)
        plt.plot(t, J_array, label="J(t, x)", color="blue")
        plt.xlabel("t (Time Step)")
        plt.ylabel("J-function")
        plt.title("J-function over Time")
        plt.grid(True)
        plt.legend()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "J-func.pdf")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved j-function plot to {save_path}")
    else:
        control_array = np.array([c.detach().cpu().numpy() if torch.is_tensor(c) else np.asarray(c) for c in J_list])

        if control_array.ndim == 3:  # [N, A, B] → mean over last two dims → [N]
            control_array = control_array.mean(axis=(1, 2))
        elif control_array.ndim == 2:  # [N, A] → mean over second dim → [N]
            control_array = control_array.mean(axis=1)
        elif control_array.ndim == 1:
            pass
        else:
            raise ValueError(f"Unsupported control_list shape: {control_array.shape}")

        t = np.arange(len(control_array))
        plt.figure(figsize=(6.0, 4.0), dpi=300)
        plt.plot(t, control_array, label="Control(t, x)", color="orange")
        plt.xlabel("t (Time Step)")
        plt.ylabel("Control Function")
        plt.title("Control Function over Time")
        plt.grid(True)
        plt.legend()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "control-func.pdf")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved control function plot to {save_path}")


def load_eval_topk_meta(self, tag: str = "pinn"):
    """
    Read meta written by _persist_topN_alias(); if missing, fall back to alias/wildcard scan.
    Return a list sorted by score ascending: [{"iter": int, "score": float, "actor": path, "critic": path}, ...]
    """
    meta_path = os.path.join(self.output_folder, f"{tag}_topk_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                from run_io import metadata_from_disk
                li = metadata_from_disk(json.load(f), self.output_folder)
            cleaned = []
            for d in li:
                if all(k in d for k in ["iter", "score", "actor", "critic"]):
                    cleaned.append(
                        {
                            "iter": int(d["iter"]),
                            "score": float(d["score"]),
                            "actor": d["actor"],
                            "critic": d["critic"],
                        }
                    )
            cleaned.sort(key=lambda d: d["score"])
            return cleaned
        except Exception as e:
            print(f"[warn] read meta failed: {e}")

    cand = []
    for a in glob.glob(os.path.join(self.output_folder, f"actor_{tag}_top*_itr*.pth")):
        base = os.path.basename(a)
        try:
            itr = int(base.split("_itr")[-1].split(".pth")[0])
        except Exception:
            itr = -1
        suffix = base.split(f"actor_{tag}_")[-1]
        c = os.path.join(self.output_folder, f"critic_{tag}_{suffix}")
        if os.path.isfile(c):
            cand.append({"iter": itr, "score": float("inf"), "actor": a, "critic": c})
    cand.sort(key=lambda d: (d["score"], d["iter"]))
    return cand[:3]


def load_ckpt_pair(self, actor_path, critic_path):
    """
    Load a pair of actor/critic weights into the current model (only switch weights, not change output_folder).
    Also try to load the corresponding params.json.
    """
    try:
        sd_actor = torch.load(actor_path, map_location=self.device)
        sd_critic = torch.load(critic_path, map_location=self.device)
        self.net_actor.load_state_dict(sd_actor)
        self.net_critic.load_state_dict(sd_critic)
        self.net_actor.eval()
        self.net_critic.eval()
        print(f"[load] actor={actor_path}\n[load] critic={critic_path}")

        # Try to load params: load from the directory of actor_path
        ckpt_dir = os.path.dirname(actor_path)
        loaded_params = self._load_params(ckpt_dir)
        if loaded_params is not None:
            self._update_params_from_dict(loaded_params)

        return True
    except Exception as e:
        raise RuntimeError(f"Failed to load actor/critic pair: {actor_path}, {critic_path}") from e


def simulate_taylor_path(self, t_grid, W_grid, P_grid, X0):
    # Non-Merton models (e.g. LQ) do not have "p"; delegate to model-specific path simulation
    p = getattr(self, "p", self.params_equ.get("p", None))
    if p is None:
        sim = getattr(self, "simulate_exact_path", None)
        if callable(sim):
            return sim(t_grid, W_grid, P_grid, X0)
        raise AttributeError(
            "Model has no attribute 'p' (Merton power utility). "
            "Implement simulate_exact_path(t_grid, W_grid, P_grid, X0) on the model for non-Merton problems."
        )

    dt = self.dt
    M = t_grid.shape[0]
    N = t_grid.shape[1] - 1
    if not torch.is_tensor(P_grid):
        P_grid = torch.zeros(M, 1, 1, device=X0.device, dtype=X0.dtype)

    def _taylor_mean_control(x):
        if self.p == 1 / 2:
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

    X_curr = X0
    X_list = [X_curr]
    u_list = []
    for k in range(N):
        t0 = t_grid[:, k, :]
        t1 = t_grid[:, k + 1, :]
        W0 = W_grid[:, k, :]
        W1 = W_grid[:, k + 1, :]
        dW = (W1 - W0).unsqueeze(-1)
        mask = (P_grid > t0.unsqueeze(1)) & (P_grid <= t1.unsqueeze(1))
        Njump = mask.sum(dim=1, keepdim=True).to(torch.float32)
        dMt = Njump.squeeze(-1) - self.params_equ["lambda"] * dt

        u_mean = _taylor_mean_control(X_curr)
        u_list.append(u_mean)
        mu_func = self.mu_torch(t0, X_curr, u_mean)
        diffusion = torch.bmm(self.sigma_torch(t0, X_curr, u_mean), dW).squeeze(2)
        xi = self.xi(t0, X_curr, u_mean)
        X_next = X_curr + mu_func * dt + diffusion + xi * dMt
        X_list.append(X_next)
        X_curr = X_next

    u_list.append(_taylor_mean_control(X_curr))
    X_traj = torch.stack(X_list, dim=1)
    u_traj = torch.stack(u_list, dim=1)
    return X_traj, u_traj


def plot_topk_eval(self, k: int = 3, device=None, tag: str = "pinn"):
    """
    Load Top-k evaluation weights sequentially, and plot each on a fixed evaluation batch.
    The output file name will have the suffix `_top{rank}_itr{itr}_score{:.4f}`, to avoid overwriting.
    """
    device = device or self.device
    topk = load_eval_topk_meta(self, tag=tag)
    if not topk:
        print(f"[info] no {tag} Top-k found; try training first.")
        return

    self._ensure_eval_set()

    for rank, rec in enumerate(sorted(topk, key=lambda d: d["score"])[:k], start=1):
        ok = load_ckpt_pair(self, rec["actor"], rec["critic"])
        if not ok:
            continue
        suffix = f"_{tag}_top{rank}_itr{int(rec['iter'])}_score{rec['score']:.4f}"
        print(f"[plot] begin {suffix}")
        _ds = int(getattr(self, "params_train", {}).get("plot_downsample_step", 2))
        _sw = int(getattr(self, "params_train", {}).get("plot_smooth_window", 1))
        self._plot_trajs_for_current_weights(
            filename_suffix=suffix,
            device=device,
            downsample_step=_ds,
            smooth_window=_sw,
            plot_num_u=1,
            plot_traj_mode=self.params_train.get("plot_traj_mode", "abc"),
        )
        print(f"[plot] done  {suffix}")


def plot_ckpt_pair(
    self,
    actor_path: str,
    critic_path: str,
    device=None,
    filename_suffix: str = None,
    plot_num: int = 5,
    N_eval: int = None,
):
    """
    Load any actor/critic checkpoint pair and plot trajectories on the fixed eval batch.

    Args:
        plot_num: Number of trajectories to plot (default: 5)
        N_eval: Number of time intervals for evaluation (default: None, uses auto)
    """
    if not (actor_path and critic_path and os.path.isfile(actor_path) and os.path.isfile(critic_path)):
        raise FileNotFoundError(f"Invalid checkpoint pair: {actor_path}, {critic_path}")

    # Ensure eval set is ready with the specified N_eval
    if N_eval is not None:
        self._eval_fixed_ready = False  # Force regeneration with new N_eval
    self._ensure_eval_set(N_eval=N_eval)

    ok = load_ckpt_pair(self, actor_path, critic_path)
    if not ok:
        return False

    if filename_suffix is None:
        base = os.path.splitext(os.path.basename(actor_path))[0]
        filename_suffix = f"ckpt_{base}"
    itr_match = re.search(r"itr(\d+)", os.path.basename(actor_path))
    if itr_match is None:
        itr_match = re.search(r"itr(\d+)", os.path.basename(critic_path))
    if itr_match is not None:
        itr_tag = f"_itr{itr_match.group(1)}"
        if itr_tag not in filename_suffix:
            filename_suffix = f"{filename_suffix}{itr_tag}"
        try:
            self.current_itr = int(itr_match.group(1))
            self.params_train["current_itr"] = int(itr_match.group(1))
        except Exception:
            pass

    device = device or self.device
    print(f"[plot] begin {filename_suffix}")
    M_eval = self.t_eval.shape[0] if hasattr(self, "t_eval") and self.t_eval is not None else "N/A"
    N_eval_actual = self.t_eval.shape[1] - 1 if hasattr(self, "t_eval") and self.t_eval is not None else "N/A"
    print(f"[plot] plotting {plot_num} trajectories (M_eval={M_eval}, N_eval={N_eval_actual})")
    _ds = int(getattr(self, "params_train", {}).get("plot_downsample_step", 2))
    _sw = int(getattr(self, "params_train", {}).get("plot_smooth_window", 1))
    self._plot_trajs_for_current_weights(
        filename_suffix=filename_suffix,
        device=device,
        downsample_step=_ds,
        smooth_window=_sw,
        plot_num=plot_num,
        plot_num_u=1,
        plot_traj_mode=self.params_train.get("plot_traj_mode", "abc"),
        N_eval=N_eval,
    )
    print(f"[plot] done  {filename_suffix}")
    return True
