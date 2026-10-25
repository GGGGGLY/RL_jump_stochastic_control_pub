"""Shared plotting, policy configuration and checkpoint utilities for LQ, Merton and CARA."""

import numpy as np
import functools
import random
import torch
from torch import nn
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde


import re

FONT_SIZE = 14
LEGEND_FONT_SIZE = 12


def training_output_folder(model_name, equation, train, *, root="output"):
    """Short parameter-based destination; full settings live in params.json."""
    if train.get("output_dir"):
        return os.path.abspath(train["output_dir"])

    def label(value):
        return format(value, ".8g") if isinstance(value, (float, np.floating)) else str(value)

    scale = "_".join(f"{name}{label(equation[key])}" for name, key in
                     (("D", "D"), ("T", "T"), ("N", "N"), ("M", "M")))
    settings = "_".join(f"{name}{label(train[key])}" for name, key in
                        (("alpha", "entropy_alpha"), ("lrA", "lr_actor"),
                         ("lrC", "lr_critic"), ("it", "iteration")))
    return os.path.abspath(os.path.join(root, model_name, scale, settings))


def preserve_random_state(function):
    """Keep diagnostic/evaluation randomness independent of subsequent training."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            return function(*args, **kwargs)
        finally:
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
    return wrapped


def configure_plot_style():
    """Use portable Matplotlib defaults without requiring TeX or local fonts."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.usetex": False,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "savefig.bbox": "tight",
    })


def configure_policy_output(actor, entropy_alpha):
    """Disable control-mode output bounds for entropy-regularized policies."""
    alpha = float(entropy_alpha)
    if alpha < 0:
        raise ValueError("entropy_alpha must be non-negative")
    if alpha == 0:
        return
    roots = [actor] if isinstance(actor, nn.Module) else list(actor)
    for root in roots:
        for module in root.modules():
            module.entropy_alpha = alpha
            if hasattr(module, "penalty"):
                module.penalty = None
            if not bool(getattr(module, "smooth_log_std_bounds", False)):
                if hasattr(module, "log_std_min"):
                    module.log_std_min = float("-inf")
                if hasattr(module, "log_std_max"):
                    module.log_std_max = float("inf")


def check_nan(var, name):
    if torch.isnan(var).any():
        print(f"NaN detected in {name}:")
        print(var)
        import sys
        sys.exit(1)

def to_scalar(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().item()
        return x.detach().mean().item()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.reshape(-1)[0])
        return float(np.nanmean(x))
    return float(x)

def _safe_min_max(data):
    """
    Safely compute min and max of data, filtering out NaN and Inf values.
    Returns (min_val, max_val) or (None, None) if no finite values exist.
    """
    if torch.is_tensor(data):
        data_np = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        data_np = data
    else:
        data_np = np.asarray(data)

    # Filter out NaN and Inf values
    finite_mask = np.isfinite(data_np)
    if not np.any(finite_mask):
        return None, None

    finite_data = data_np[finite_mask]
    return float(np.min(finite_data)), float(np.max(finite_data))

def _safe_min_max_pair(data_a, data_b):
    """
    Compute min/max across two datasets, ignoring NaN/Inf.
    Returns (min_val, max_val) or (None, None) if no finite values exist.
    """
    a_min, a_max = _safe_min_max(data_a)
    b_min, b_max = _safe_min_max(data_b)
    mins = [v for v in [a_min, b_min] if v is not None]
    maxs = [v for v in [a_max, b_max] if v is not None]
    if not mins or not maxs:
        return None, None
    return float(np.min(mins)), float(np.max(maxs))

def _as_1d_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().flatten().numpy()


    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.asarray([])
        if all(torch.is_tensor(v) for v in x):

            try:
                return torch.stack([v.detach().cpu().flatten() for v in x]).reshape(len(x), -1).squeeze(-1).numpy()
            except Exception:
                return np.asarray([v.detach().cpu().item() for v in x])

        return np.asarray([ (v.detach().cpu().item() if torch.is_tensor(v) else v) for v in x ])

    try:
        return np.asarray(x)
    except Exception:

        return np.asarray([float(x)])

def _detect_ckpt_set(model, args):
    """Infer the multi-player checkpoint directory. Check args.actor_path and
    args.critic_path first, then model.output_folder. Return a path or None."""

    cand_dirs = []
    for p in [args.actor_path, args.critic_path]:
        if p is not None:
            if os.path.isdir(p):
                cand_dirs.append(os.path.abspath(p))
            elif os.path.isfile(p):

                cand_dirs.append(os.path.dirname(os.path.abspath(p)))


    if not cand_dirs and hasattr(model, "output_folder"):
        cand_dirs.append(os.path.abspath(model.output_folder))


    for root in cand_dirs:
        meta_path = os.path.join(root, "eval_topk_meta.json")
        if os.path.isfile(meta_path):
            return root


        ok = True
        for i in range(model.D):
            a = os.path.join(root, f"actor_player{i}_eval_top1.pth")
            c = os.path.join(root, f"critic_player{i}_eval_top1.pth")
            if not (os.path.isfile(a) and os.path.isfile(c)):
                ok = False
                break
        if ok:
            return root

    return None


def _load_best_multi_ckpt(model, ckpt_root):
    """Load the best (Top-1) multi-player weights from ckpt_root using
    model._load_eval_topk_meta() and _load_ckpt_set(). Return whether loading succeeded."""
    old_output = model.output_folder
    model.output_folder = ckpt_root

    topk = model._load_eval_topk_meta()
    use_meta = False
    if topk:
        best = sorted(topk, key=lambda d: d["score"])[0]
        actor_paths = best["actor"]
        critic_paths = best["critic"]

        if len(actor_paths) == model.D and len(critic_paths) == model.D:
            use_meta = True
        else:
            print(f"[warn] meta data incomplete (actor: {len(actor_paths)}, critic: {len(critic_paths)}, expected: {model.D}), fallback to alias")

    if not use_meta:

        actor_paths = []
        critic_paths = []
        for i in range(model.D):
            a = os.path.join(ckpt_root, f"actor_player{i}_eval_top1.pth")
            c = os.path.join(ckpt_root, f"critic_player{i}_eval_top1.pth")
            if not (os.path.isfile(a) and os.path.isfile(c)):
                print(f"[warn] missing top1 alias for player {i} in {ckpt_root}")
                model.output_folder = old_output
                return False
            actor_paths.append(a)
            critic_paths.append(c)

    if isinstance(actor_paths, str):
        actor_paths = [actor_paths]
    if isinstance(critic_paths, str):
        critic_paths = [critic_paths]

    ok = model._load_ckpt_set(actor_paths, critic_paths, ckpt_root=ckpt_root)
    if not ok:
        model.output_folder = old_output
        return False


    return True




def _conditional_policy_point(model, times, states, plotted_indices, time_fraction):
    """Choose one shared (t,x) for the PINN density and flow samples."""
    if time_fraction not in (0.25, 0.5):
        raise ValueError("policy_plot_time_fraction must be 0.25 (T/4) or 0.5 (T/2)")
    def as_numpy(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    times, states = as_numpy(times), as_numpy(states)
    grid = times[0, :, 0]
    target = float(grid[0] + time_fraction * (grid[-1] - grid[0]))
    k = int(np.argmin(np.abs(grid - target)))
    params = model.params_train
    selector = params.get("panel_a_traj_selector", "Conddist_u_T_mean_Xt")
    chosen = None
    if selector == "cond_u_at_fixed_point":
        x = float(params.get("panel_a_fixed_x", 10.0))
    elif selector == "Conddist_u_T_mean_Xt":
        xs = states[:, k, 0]
        finite = xs[np.isfinite(xs)]
        if not finite.size:
            raise ValueError("No finite states for the conditional policy distribution")
        x = float(finite.mean())
    else:
        indices = (as_numpy(plotted_indices).astype(int).reshape(-1)
                   if selector == "Conddist_u_T_XTmax_from_plotted"
                   else np.arange(states.shape[0]))
        valid = indices[np.isfinite(states[indices, k, 0]) & np.isfinite(states[indices, -1, 0])]
        if not valid.size:
            raise ValueError("No finite trajectory for the conditional policy distribution")
        if selector == "first_valid_or_XTmax":
            suitable = valid[np.all(np.isfinite(states[valid, :, 0]) & (states[valid, :, 0] >= 2), axis=1)]
        else:
            suitable = np.array([], dtype=int)
        chosen = int(suitable[0] if suitable.size else valid[np.argmax(states[valid, -1, 0])])
        x = float(states[chosen, k, 0])
    if not np.isfinite(x):
        raise ValueError("Conditional policy state must be finite")
    x_min = max(float(params.get("pinn_x_min") or 0), 0.01)
    x_max = params.get("pinn_x_max")
    x = max(x_min, min(float(x_max) if x_max is not None else np.inf, x))
    label = ("T/4" if time_fraction == 0.25 else "T/2") if np.isclose(grid[k], target) else f"{grid[k]:g}"
    return float(grid[k]), x, label, chosen


def plot_trajectory_combined_abc(
    t_test,
    traj_u_mean, traj_u_test,
    traj_value, traj_value_test,
    traj_Xt, traj_Xt_test,
    approximate_u, exact_u, start_u, end_u, ylabel_u,
    approximate_v, exact_v, start_v, end_v, ylabel_v,
    approximate_x, exact_x, start_x, end_x, ylabel_x,
    traj_u_third=None, traj_value_third=None, traj_Xt_third=None,
    third_u=None, third_v=None, third_x=None,
    traj_control_exact_mean=None,  # Time-step-wise mean of exact control for 4th subplot
    traj_control_pred_mean=None,  # Time-step-wise mean of predicted control for 4th subplot
    traj_control_taylor_mean=None,  # Time-step-wise mean of Taylor control for 1st subplot
    filename="trajectory_combined_abc",
    P_test=None,
    num_samples=2,
    jump_num=3,
    jump_index=None,
    device="cpu",
    output_folder='./pred_traj',
    downsample_step=1,
    smooth_window=1,
    f_std=None,
    axis_label_size=24,
    tick_label_size=20,
    legend_font_size=15,
    subplot_label_size=20,
    model_type="multi_cara",
    coeff_SDE_type="piecewise",
    raw_labels=False,
    return_fig_axes=False,
    panel_c_mode=None,  # Options: X_traj_evalmode_sample, X_traj_evalmode_policymean, X_dist_evolve_T, Conddist_u_T
    model=None,  # Model instance for panel c plotting functions
    W_test=None,  # For evolving X distributions
    X0_eval=None,  # For evolving X distributions
    panel_c_defer=False,  # If True, leave subplot (c) empty for later rendering
    panel_c_traj_Xt=None,  # Optional override: use these trajectories for X_dist_evolve_T
    panel_c_traj_Xt_test=None,  # Optional override: use these PINN trajectories for X_dist_evolve_T
    panel_c_traj_u_mean=None,  # Optional override: use these control trajectories for U_dist_on_XT
    panel_c_traj_u_test=None,  # Optional override: use these PINN control trajectories for U_dist_on_XT
    panel_a_2x2_conddist=False,  # If True, panel (a) becomes 2x2 p(u|t,X_t[0]) at t=T/4,T/2,3T/4,T; (c) = X_t trajectory
    panel_a_tx_pairs=None,  # List of 4 (t, x) for panel (a) when panel_a_2x2_conddist: [(T/4,x1),(T/2,x2),(3T/4,x3),(T,x4)]
    panel_a_chosen_traj_index=None,  # Trajectory index used for panel (a); if set, include it in (b),(c) so the same path is visible
    panel_a_choose_from_plotted=False,  # If True, choose panel (a) trajectory with max X_T among the 3-4 plotted in (b),(c)
    smooth_window_xt=None,  # If set, sliding window for panel (c) X_t trajectories only; None = use smooth_window
    downsample_step_xt=None,  # If set, time subsampling for panel (c) only; None defaults to 1 (full dt; jump visible)
    control_y_limits=None,  # Optional explicit y limits for panel (a) control trajectories
    panel_a_policy_distribution=False,  # Flow policy: one density panel, followed by value/state paths
    policy_plot_time_fraction=0.25,
    ):
    """
    Plot the 3-panel evaluation figure (a,b,c).

    With panel_a_policy_distribution=True, (a) compares PINN density and flow
    samples at T/4 (or T/2), (b) shows value and (c) shows state trajectories.
    This explicit flow layout takes precedence over legacy mean/panel-c options.

    `smooth_window_xt`: optional moving-average window for panel (c) state trajectories only.
    If None, uses `smooth_window` (same as panels (a)/(b)). Use 1 for no smoothing (clearer jumps).
    `downsample_step_xt`: optional `downsample_step` for panel (c) only; default 1 keeps every time step so
    jump markers use the full grid (otherwise jumps look like smooth chords).

    Panel (c) is controlled by `panel_c_mode` and is the main source of confusion.
    This docstring defines *exactly* what each `panel_c_mode` means and what is plotted.

    --- panel_c_mode semantics (subplot c) ---
    1) panel_c_mode is None (default):
       - Subplot (c) shows state trajectories X_t (FlowNet rollout vs PINN rollout).

    2) panel_c_mode == "X_dist_evolve_T" (canonical; CLI `--panel_c_mode X_dist_evolve_T`):
       - Subplot (c) shows the *terminal-state distribution* comparison at time T using
         already-computed trajectories (NO extra PINN calls, no re-evolution):

           Samples:
             X_T^{flow,(m)} = X_pred[m, -1]
             X_T^{pinn,(m)} = X_test_pinn[m, -1]

           Visualization:
             Either a KDE density estimate
               \\hat p(x) = (1/(M h)) * sum_{m=1..M} K((x - X_T^{(m)})/h)
             or a histogram (fallback when KDE fails).

       - The plot is meant to match the 0129 "no extra PINN calls" evaluation.

    3) panel_c_mode == "Conddist_u_T"  (aka CLI `--panel_c_mode distribution_of_policy`):
       - Subplot (c) shows the *conditional control distribution* near terminal time:

           Actor/FlowNet conditional distribution:
             u^{(j)} ~ \\hat\\pi(\\cdot | t=T, x=x_rep)
             \\hat p_{flow}(u | T, x_rep) is shown via histogram/KDE of {u^{(j)}}.

           PINN (entropy-regularized) Gibbs policy (discrete on u_grid when gamma>0):
             p^*(u_i | x) \\propto w_i * exp(H(x,u_i)/\\gamma)
             where u_i are points on `u_grid`, w_i are quadrature weights.

           (The helper in `merton_policy_plots.py` decides whether to render bars/lines for PINN
            and histogram/KDE for the actor depending on `panel_c_plot_method`.)

    4) panel_c_mode == "U_dist_on_XT" (canonical; CLI `--panel_c_mode U_dist_on_XT`):
       - Subplot (c) shows the distribution of one control per terminal state, i.e.
         the *pushforward* distribution induced by the terminal-state samples:

           u_T^{flow,(m)} = u_pred[m, -1]  where u_pred[m, k] = \\hat\\pi(t_k, X_k^{flow,(m)})
           u_T^{pinn,(m)} = u_pinn[m, -1]  where u_pinn[m, k] = \\pi^*(t_k, X_k^{pinn,(m)})

         Then plot density of {u_T^{(m)}} via KDE/histogram (similar to X_dist_evolve_T).

       - This corresponds to the user's case (ii): "p(u|X_T)" meaning *one u per X_T*,
         i.e. the marginal distribution of u_T along terminal samples.

    5) panel_c_mode in ("X_traj_evalmode_sample", "X_traj_evalmode_policymean"):
       - Legacy names: treated the same as None here (plot X trajectories).
    """

    from merton_policy_plots import plot_panel_c_conditional_distribution

    def _infer_plot_traj_mode_from_filename(fn: str):
        """
        Infer plot_traj_mode token from filename, expecting a standalone suffix like `_ab`, `_ac`, `_bc`, `_abc`.
        Important: `_abc` must NOT be mistaken for `_ab`.
        """
        if not isinstance(fn, str) or not fn:
            return None
        m = re.search(r"_(abc|ab|ac|bc)(?:_|$)", fn.lower())
        return m.group(1) if m else None

    def _to_1d_time(x, ds_idx=None, dev=device):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(dev)

        if x.dim() == 3:
            if x.size(0) == 1:
                x = x[0]
        if x.dim() == 2 and x.size(-1) == 1:
            x = x[:, 0]
        elif x.dim() == 2 and x.size(0) == 1:
            x = x[0]

        if ds_idx is not None:
            x = x[ds_idx.to(dev)]
        return x

    def _smooth_1d(x, k):
        if k <= 1:
            return x
        pad = k // 2
        x_pad = torch.cat([x[:1].repeat(pad), x, x[-1:].repeat(pad)], dim=0)
        kernel = torch.ones(k, device=x.device) / k
        y = torch.conv1d(
            x_pad.view(1, 1, -1),
            kernel.view(1, 1, -1)
        ).view(-1)
        return y

    def _plot_single_subplot(ax, t_test, f_pred, f_test, approximate, exact, start, end,
                             ylabel, P_test, samples_index, ds_idx, first_legend_dict,
                             subplot_label, filename_hint="", f_std_param=None,
                             f_third=None, third_label=None, y_limits=None,
                             axis_label_size=18, tick_label_size=16, legend_font_size=14,
                             subplot_label_size=20, smooth_window_exact=None):
        def _series_for_plot(x, smooth_override=None):
            if x is None:
                return None, None, None, None
            x_full = _to_1d_time(x, None)
            k = smooth_window if smooth_override is None else max(1, int(smooth_override))
            x_smooth_full = _smooth_1d(x_full, k)
            if ds_idx is not None:
                idx = ds_idx.to(x_full.device)
                # Clamp indices to valid range
                max_idx = x_full.shape[0] - 1
                idx = torch.clamp(idx, min=0, max=max_idx)
                x_full_ds = x_full[idx]
                x_smooth_ds = x_smooth_full[idx]
            else:
                x_full_ds = x_full
                x_smooth_ds = x_smooth_full
            return x_full, x_smooth_full, x_full_ds, x_smooth_ds

        def _safe_index(arr, samples_idx, ref_shape=None):
            """Safely index array based on its dimensions.

            Args:
                arr: Array to index (can be numpy or torch)
                samples_idx: Sample indices to select
                ref_shape: Reference shape from t_test to help determine array structure
            """
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                arr = torch.from_numpy(arr)
            arr = arr.to(device)

            if arr.dim() == 3:
                # 3D: [samples, time, features]
                return arr[samples_idx, :, 0]
            elif arr.dim() == 2:
                # 2D: could be [samples, time] or [time, features]
                # Use reference shape if available
                if ref_shape is not None and len(ref_shape) >= 2:
                    # If reference is 3D, arrays should match that structure
                    if len(ref_shape) == 3:
                        # Reference is [samples, time, features], so arr should be [samples, time]
                        max_idx = samples_idx.max().item() if len(samples_idx) > 0 else 0
                        if arr.shape[0] > max_idx:
                            return arr[samples_idx, :]
                    elif len(ref_shape) == 2:
                        # Reference is 2D, check if first dimension matches
                        if ref_shape[0] == arr.shape[0]:
                            # Same first dimension, likely [samples, time]
                            max_idx = samples_idx.max().item() if len(samples_idx) > 0 else 0
                            if arr.shape[0] > max_idx:
                                return arr[samples_idx, :]

                # Fallback: try to determine from array shape and sample indices
                max_idx = samples_idx.max().item() if len(samples_idx) > 0 else 0
                if arr.shape[0] > max_idx:
                    # First dimension is large enough for sample indexing
                    # Check if it looks like [samples, time] vs [time, features]
                    # Typically time dimension is much larger than sample count
                    if arr.shape[0] <= 1000 and arr.shape[1] > arr.shape[0]:
                        # Likely [samples, time] - time dimension is larger
                        return arr[samples_idx, :]
                    elif arr.shape[1] <= 10:
                        # Likely [samples, time] - small feature dimension
                        return arr[samples_idx, :]

                # Likely [time, features] - take first feature if multiple features exist
                return arr[:, 0] if arr.shape[1] > 1 else arr
            else:
                # 1D or other - return as is
                return arr

        # Get reference shape from t_test
        t_test_ref = t_test if torch.is_tensor(t_test) else torch.tensor(t_test)
        t_test_shape = tuple(t_test_ref.shape) if hasattr(t_test_ref, 'shape') else None

        t0_full, _, t0, _ = _series_for_plot(_safe_index(t_test, samples_index[0:1], t_test_shape))
        f0_pred_full, f0_pred_smooth_full, f0_pred_orig, f0_pred_smooth = _series_for_plot(
            _safe_index(f_pred, samples_index[0:1], t_test_shape)
        )
        f0_test_full, f0_test_smooth_full, f0_test_orig, f0_test_smooth = _series_for_plot(
            _safe_index(f_test, samples_index[0:1], t_test_shape),
            smooth_override=smooth_window_exact
        )

        if ("control_mean" in filename_hint) and (f_std_param is not None):
            _, _, f0_std, _ = _series_for_plot(_safe_index(f_std_param, samples_index[0:1], t_test_shape))
            upper = (f0_pred_smooth + 3.0 * f0_std).cpu().numpy()
            lower = (f0_pred_smooth - 3.0 * f0_std).cpu().numpy()
            ax.fill_between(t0.cpu().numpy(), lower, upper,
                           alpha=0.15, label=r'$\pm 3\sigma$', zorder=0)



        if not torch.allclose(f0_pred_orig, f0_pred_smooth, atol=1e-6):
            ax.plot(t0.cpu(), f0_pred_orig.cpu(),
                   color='darkorange', alpha=0.3, zorder=1, linewidth=1, label=None)
        if f0_test_orig is not None and f0_test_smooth is not None:
            if not torch.allclose(f0_test_orig, f0_test_smooth, atol=1e-6):
                ax.plot(t0.cpu(), f0_test_orig.cpu(),
                       '--', color='darkviolet', alpha=0.3, zorder=1, linewidth=1, label=None)
        if f_third is not None:
            _, _, f0_third_orig, f0_third_smooth = _series_for_plot(
                _safe_index(f_third, samples_index[0:1], t_test_shape)
            )
            if not torch.allclose(f0_third_orig, f0_third_smooth, atol=1e-6):
                ax.plot(t0.cpu(), f0_third_orig.cpu(),
                       '-.', color='seagreen', alpha=0.3, zorder=1, linewidth=1, label=None)

        if raw_labels:
            pred_label = approximate
            exact_label = exact
        else:

            if approximate is not None:
                if approximate.strip().startswith('Approx.'):
                    pred_label = approximate
                else:
                    pred_label = f'Approx. {approximate}'
            else:
                pred_label = None
            if exact is not None:
                st = exact.strip()
                if st.startswith('Benchmark'):
                    exact_label = exact
                elif st.startswith('Exact'):
                    exact_label = exact.replace('Exact', 'Benchmark', 1)
                else:
                    exact_label = f'Benchmark {exact}'
            else:
                exact_label = None
        ax.plot(t0.cpu(), f0_pred_smooth.cpu(),
               color='darkorange', label=pred_label, zorder=2, linewidth=2)
        if f0_test_smooth is not None:
            ax.plot(t0.cpu(), f0_test_smooth.cpu(),
                   '--', color='darkviolet', label=exact_label, zorder=3, linewidth=2)
        if f_third is not None and third_label is not None:
            ax.plot(t0.cpu(), f0_third_smooth.cpu(),
                   '-.', color='seagreen', label=third_label, zorder=3, linewidth=2)

        first_legend = first_legend_dict.get('jump', None)

        if P_test is not None:
            if isinstance(P_test, np.ndarray):
                P_test_tensor = torch.from_numpy(P_test).to(device)
            else:
                P_test_tensor = P_test.to(device) if hasattr(P_test, 'to') else torch.tensor(P_test).to(device)

            if isinstance(t_test, np.ndarray):
                t_test_tensor = torch.from_numpy(t_test).to(device)
            else:
                t_test_tensor = t_test.to(device) if hasattr(t_test, 'to') else torch.tensor(t_test).to(device)

            if f_test is None:
                f_test_tensor = None
            elif isinstance(f_test, np.ndarray):
                f_test_tensor = torch.from_numpy(f_test).to(device)
            else:
                f_test_tensor = f_test.to(device) if hasattr(f_test, 'to') else torch.tensor(f_test).to(device)

            # Also prepare f_pred tensor for jump plotting when f_test is None
            if isinstance(f_pred, np.ndarray):
                f_pred_tensor = torch.from_numpy(f_pred).to(device)
            else:
                f_pred_tensor = f_pred.to(device) if hasattr(f_pred, 'to') else torch.tensor(f_pred).to(device)

            # Use f_test if available, otherwise use f_pred (for policy mean plots)
            f_data_tensor = f_test_tensor if f_test_tensor is not None else f_pred_tensor
            if f_data_tensor is not None:
                # Iterate through all samples and all jumps
                for i in range(len(samples_index)):
                    sample_idx = int(samples_index[i])
                    if sample_idx >= P_test_tensor.shape[0]:
                        continue
                    t_max = float(t_test_tensor[sample_idx, -1, 0].item())
                    t_min = float(t_test_tensor[sample_idx, 0, 0].item())

                    # Get this sample's displayed trajectory (after smoothing and downsampling)
                    sample_idx_tensor = torch.tensor([sample_idx], dtype=torch.long).to(device)
                    f_data_sample = _safe_index(f_data_tensor, sample_idx_tensor, t_test_shape)
                    t_sample = _safe_index(t_test_tensor, sample_idx_tensor, t_test_shape)

                    # Get displayed trajectory (after smoothing and downsampling)
                    _, _, t_displayed, _ = _series_for_plot(t_sample)
                    _, _, f_displayed, _ = _series_for_plot(f_data_sample)

                    # Now check all jumps for this sample
                    for j in range(P_test_tensor.shape[1]):
                        jump_time_val = float(P_test_tensor[sample_idx, j, 0].item())
                        if jump_time_val <= 0 or not (t_min <= jump_time_val <= t_max):
                            continue
                        jump_time = float(jump_time_val)

                        if "traj_Xt" in filename_hint:
                            # Full time grid: draw a *vertical* segment at jump_time using the two
                            # consecutive mesh states that bracket the jump interval (step includes jump+diffusion).
                            t_full_1d = t_test_tensor[sample_idx, :, 0].detach().cpu().numpy()
                            L = int(len(t_full_1d))
                            if L < 2:
                                continue
                            ir = int(np.searchsorted(t_full_1d, jump_time, side="right"))
                            idx = max(0, min(ir - 1, L - 2))

                            def _draw_xt_jump_vline(f_tensor):
                                nonlocal first_legend
                                if f_tensor is None:
                                    return
                                fj = f_tensor[sample_idx, :, 0].detach().float().cpu().numpy()
                                y0, y1 = float(fj[idx]), float(fj[idx + 1])
                                if not (np.isfinite(y0) and np.isfinite(y1)) or abs(y1 - y0) < 1e-14:
                                    return
                                if first_legend is None:
                                    ax.plot(
                                        [jump_time, jump_time], [y0, y1], color="red", linewidth=4.0,
                                        label="Jump", zorder=5, solid_capstyle="butt",
                                    )
                                    first_legend = 1
                                    first_legend_dict["jump"] = 1
                                else:
                                    ax.plot(
                                        [jump_time, jump_time], [y0, y1], color="red", linewidth=4.0,
                                        zorder=5, solid_capstyle="butt",
                                    )

                            _draw_xt_jump_vline(f_pred_tensor)
                            _draw_xt_jump_vline(f_test_tensor)
                            if f_third is not None:
                                try:
                                    ft = f_third if torch.is_tensor(f_third) else torch.from_numpy(
                                        np.asarray(f_third, dtype=np.float32)
                                    )
                                    ft = ft.to(device).float()
                                    if ft.dim() >= 2 and int(ft.shape[0]) > sample_idx:
                                        _draw_xt_jump_vline(ft)
                                except Exception:
                                    pass
                            continue

                        # Value / control panels: chord between downsampled points (legacy)
                        t_displayed_np = t_displayed.cpu().numpy() if torch.is_tensor(t_displayed) else t_displayed
                        idx_after = np.searchsorted(t_displayed_np, jump_time)
                        idx_before = max(0, idx_after - 1)
                        idx_after = min(len(t_displayed_np) - 1, idx_after)
                        t_before = float(t_displayed_np[idx_before])
                        t_after = float(t_displayed_np[idx_after])
                        f_displayed_np = f_displayed.cpu().numpy() if torch.is_tensor(f_displayed) else f_displayed
                        f_prev = float(f_displayed_np[idx_before])
                        f_curr = float(f_displayed_np[idx_after])
                        if first_legend is None:
                            ax.plot([t_before, t_after], [f_prev, f_curr], color='red',
                                    linewidth=3, label='Jump', zorder=3)
                            first_legend = 1
                            first_legend_dict['jump'] = 1
                        else:
                            ax.plot([t_before, t_after], [f_prev, f_curr], color='red',
                                    linewidth=3, zorder=3)

        if len(samples_index) > 1:
            for idx in samples_index[1:]:
                # Convert idx to tensor format expected by _safe_index
                if isinstance(idx, torch.Tensor):
                    idx_tensor = idx.unsqueeze(0) if idx.dim() == 0 else idx
                else:
                    idx_tensor = torch.tensor([idx], dtype=torch.long).to(device)
                _, _, t_i, _ = _series_for_plot(_safe_index(t_test, idx_tensor, t_test_shape))
                _, _, f_i_pred_orig, f_i_pred_smooth = _series_for_plot(_safe_index(f_pred, idx_tensor, t_test_shape))
                _, _, f_i_test_orig, f_i_test_smooth = _series_for_plot(
                    _safe_index(f_test, idx_tensor, t_test_shape),
                    smooth_override=smooth_window_exact
                )
                if ("control_mean" in filename_hint) and (f_std_param is not None):
                    _, _, f_i_std, _ = _series_for_plot(_safe_index(f_std_param, idx_tensor, t_test_shape))
                    upper = (f_i_pred_smooth + 3.0 * f_i_std).cpu().numpy()
                    lower = (f_i_pred_smooth - 3.0 * f_i_std).cpu().numpy()
                    ax.fill_between(t_i.cpu().numpy(), lower, upper, alpha=0.08, zorder=0)

                ax.plot(t_i.cpu(), f_i_pred_orig.cpu(), color='darkorange', alpha=0.3, zorder=1, linewidth=1)
                if f_i_test_orig is not None:
                    ax.plot(t_i.cpu(), f_i_test_orig.cpu(), '--', color='darkviolet', alpha=0.3, zorder=1, linewidth=1)
                if f_third is not None:
                    _, _, f_i_third_orig, f_i_third_smooth = _series_for_plot(
                        _safe_index(f_third, idx_tensor, t_test_shape)
                    )
                    ax.plot(t_i.cpu(), f_i_third_orig.cpu(), '-.', color='seagreen', alpha=0.3, zorder=1, linewidth=1)

                ax.plot(t_i.cpu(), f_i_pred_smooth.cpu(), color='darkorange', zorder=2, linewidth=2)
                if f_i_test_smooth is not None:
                    ax.plot(t_i.cpu(), f_i_test_smooth.cpu(), '--', color='darkviolet', zorder=3, linewidth=2)
                if f_third is not None:
                    ax.plot(t_i.cpu(), f_i_third_smooth.cpu(), '-.', color='seagreen', zorder=3, linewidth=2)

        samples_idx_np = samples_index.numpy() if isinstance(samples_index, torch.Tensor) else samples_index
        y_min = None
        y_max = None
        if f_test is not None:
            if isinstance(t_test, np.ndarray):
                t_end = t_test[samples_idx_np, -1, 0]
                f_end = f_test[samples_idx_np, -1, 0]
                f_start = f_test[0, 0, 0]
                if isinstance(f_start, np.ndarray):
                    f_start = float(f_start.item()) if f_start.size == 1 else f_start
                y_min = None
                y_max = None
            else:
                t_end = t_test[samples_index, -1, 0].cpu().numpy()
                f_end = f_test[samples_index, -1, 0].cpu().numpy()
                f_start_val = f_test[0, 0, 0]
                f_start = f_start_val.item() if isinstance(f_start_val, torch.Tensor) else float(f_start_val)
                y_min = None
                y_max = None

            ax.scatter(t_end, f_end, color='k', marker='o', s=15)
            ax.scatter(0, f_start, color='k', marker='s', s=15)
        ax.set_xlabel("t (Time)", fontsize=axis_label_size)
        ax.set_ylabel(ylabel, fontsize=axis_label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax.text(0.02, 0.95, subplot_label, transform=ax.transAxes, fontsize=subplot_label_size,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.legend(fontsize=legend_font_size, loc='best', framealpha=0.5, edgecolor='gray')
        if y_limits is not None:
            ax.set_ylim(y_limits[0], y_limits[1])
        elif y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

    t_test_for_shape = t_test if torch.is_tensor(t_test) else torch.tensor(t_test)
    M = t_test_for_shape.shape[0]

    # Modified logic: select 3 trajectories with jumps + 1 without jump = 4 total lines max
    # This ensures a good mix of jump and non-jump trajectories for visualization
    if jump_index is not None and jump_num > 0:
        if isinstance(jump_index, np.ndarray):
            jump_idx_all = torch.from_numpy(jump_index).to(device)
        elif isinstance(jump_index, torch.Tensor):
            jump_idx_all = jump_index.to(device)
        else:
            jump_idx_all = torch.tensor(list(jump_index), dtype=torch.long).to(device)

        # Select up to 3 trajectories with jumps
        num_jump_samples = min(3, len(jump_idx_all))
        jump_idx = jump_idx_all[:num_jump_samples]
    else:
        jump_idx = torch.tensor([], dtype=torch.long).to(device)
        jump_idx_all = torch.tensor([], dtype=torch.long).to(device)

    # Select 1 trajectory without jump (from samples that are not in jump_idx)
    all_idx = torch.arange(M, device=device, dtype=torch.long)
    if len(jump_idx_all) > 0:
        # Guard against stale/extra indices when M changes between rollouts
        jump_idx_all = jump_idx_all[(jump_idx_all >= 0) & (jump_idx_all < M)]
        mask = torch.ones(M, device=device, dtype=torch.bool)
        mask[jump_idx_all] = False
        non_jump_idx = all_idx[mask]
    else:
        non_jump_idx = all_idx

    # Randomly select 1 non-jump trajectory
    if len(non_jump_idx) > 0:
        perm = torch.randperm(len(non_jump_idx), device=device)
        non_jump_sample = non_jump_idx[perm[:1]]
    else:
        non_jump_sample = torch.tensor([], dtype=torch.long).to(device)

    # Combine: 3 jump trajectories + 1 non-jump trajectory = 4 total
    if len(jump_idx) > 0 and len(non_jump_sample) > 0:
        samples_index = torch.unique(torch.cat([jump_idx, non_jump_sample], dim=0))
    elif len(jump_idx) > 0:
        samples_index = jump_idx
    elif len(non_jump_sample) > 0:
        samples_index = non_jump_sample
    else:
        # Fallback: select up to 4 random samples
        samples_index = torch.randperm(M)[:min(4, M)].to(device)

    # So panel (b) and (c) show the same trajectory as panel (a) when panel (a) is 2x2 p(u|t,X_t)
    if panel_a_chosen_traj_index is not None and 0 <= panel_a_chosen_traj_index < M:
        chosen_t = torch.tensor([panel_a_chosen_traj_index], dtype=torch.long, device=device)
        samples_index = torch.unique(torch.cat([samples_index, chosen_t], dim=0))

    # Panel (a): choose trajectory with max X_T among the 3-4 plotted (so panel (a) matches what is visible in (b),(c))
    if panel_a_choose_from_plotted and panel_a_2x2_conddist and model is not None and traj_Xt is not None and len(samples_index) > 0:
        try:
            _traj = traj_Xt.cpu().numpy() if hasattr(traj_Xt, "cpu") else np.asarray(traj_Xt)
            _t = t_test.cpu().numpy() if hasattr(t_test, "cpu") else np.asarray(t_test)
            N_steps = int(_t.shape[1]) - 1
            # Legacy (kept for reference):
            # idx_dt, idx_T3, idx_2T3, idx_T = 1, max(1, N_steps // 3), max(2, 2 * N_steps // 3), N_steps
            # Panel-(a) conditioning times: T/4, T/2, 3T/4, T (replace legacy dt,T/3,2T/3,T)
            idx_T4 = max(1, N_steps // 4)
            idx_T2 = max(1, N_steps // 2)
            idx_3T4 = max(1, 3 * N_steps // 4)
            idx_T = N_steps
            si_np = samples_index.numpy() if hasattr(samples_index, "numpy") else np.asarray(samples_index)
            X_T_vals = np.array([float(_traj[int(i), idx_T, 0]) for i in si_np], dtype=np.float64)
            valid = np.isfinite(X_T_vals)
            if np.any(valid):
                best_local = int(np.argmax(X_T_vals))
                chosen_idx_int = int(si_np[best_local])
            else:
                chosen_idx_int = int(si_np[0])
            x_min_plot = max(float(model.params_train.get("pinn_x_min", 1e-5)), 0.01)
            x_max_plot = float(model.params_train.get("pinn_x_max", 80.0))
            t_vals = [
                float(_t[0, idx_T4, 0]), float(_t[0, idx_T2, 0]),
                float(_t[0, idx_3T4, 0]), float(_t[0, idx_T, 0]),
            ]
            x_raw = [
                float(_traj[chosen_idx_int, idx_T4, 0]), float(_traj[chosen_idx_int, idx_T2, 0]),
                float(_traj[chosen_idx_int, idx_3T4, 0]), float(_traj[chosen_idx_int, idx_T, 0]),
            ]
            x_vals = [max(x_min_plot, min(x_max_plot, x)) for x in x_raw]
            panel_a_tx_pairs = list(zip(t_vals, x_vals))
        except Exception:
            panel_a_tx_pairs = None

    policy_point = None
    if panel_a_policy_distribution:
        if model is None or traj_Xt is None:
            raise ValueError("Flow distribution plots require the model and state trajectories")
        policy_point = _conditional_policy_point(
            model, t_test, traj_Xt, samples_index, policy_plot_time_fraction,
        )
        if policy_point[3] is not None:
            chosen_t = torch.tensor([policy_point[3]], dtype=torch.long, device=device)
            samples_index = torch.unique(torch.cat([samples_index, chosen_t]))

    samples_index = samples_index[samples_index < M]
    samples_index = samples_index.cpu()

    # Layout: either 1x3 (legacy) or 2x3 when panel (a) = 2x2 p(u|t,X_t), (b)=value, (c)=X_t trajectory
    # (panel_a_tx_pairs may have been set above when panel_a_choose_from_plotted=True)
    use_2x2_panel_a = not panel_a_policy_distribution and bool(panel_a_2x2_conddist) and model is not None and panel_a_tx_pairs is not None and len(panel_a_tx_pairs) >= 4
    if use_2x2_panel_a:
        from matplotlib import gridspec
        fig = plt.figure(figsize=(18, 10), dpi=300)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        ax_a00 = fig.add_subplot(gs[0, 0])
        ax_a01 = fig.add_subplot(gs[0, 1])
        ax_b = fig.add_subplot(gs[0, 2])
        ax_a10 = fig.add_subplot(gs[1, 0])
        ax_a11 = fig.add_subplot(gs[1, 1])
        ax_c = fig.add_subplot(gs[1, 2])
        axes_a_2x2 = [ax_a00, ax_a01, ax_a10, ax_a11]
        axes = [ax_a00, ax_b, ax_c]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        if len(axes) != 3:
            axes = axes.reshape(-1)
        ax_b = axes[1]
        ax_c = axes[2]

    N = t_test_for_shape.shape[1]
    ds_idx = torch.arange(0, N, step=downsample_step)
    ds_idx = torch.clamp(ds_idx, max=N - 1)
    _dst_xt = downsample_step_xt if downsample_step_xt is not None else 1
    _dst_xt = max(1, int(_dst_xt))
    ds_idx_xt = torch.arange(0, N, step=_dst_xt)
    ds_idx_xt = torch.clamp(ds_idx_xt, max=N - 1)

    # Give the control panel stable visual breathing room. Matplotlib's default
    # margin is too tight when the deterministic benchmark is constant, making
    # small approximation errors look like large oscillations. Base the limits
    # on every displayed control curve and add 20% of the observed span, with a
    # 5% relative minimum so an exactly/near-constant result is still readable.
    u_ylim = None
    try:
        control_arrays = [x for x in (traj_u_mean, traj_u_test, traj_u_third) if x is not None]
        u_max = max(float(x.max()) for x in control_arrays)
        u_min = min(float(x.min()) for x in control_arrays)
        if np.isfinite(u_min) and np.isfinite(u_max):
            u_span = max(0.0, u_max - u_min)
            u_scale = max(abs(u_min), abs(u_max), 1e-3)
            u_pad = max(0.20 * u_span, 0.05 * u_scale, 1e-3)
            u_ylim = [u_min - u_pad, u_max + u_pad]
    except Exception:
        u_ylim = None
    if control_y_limits is not None:
        if len(control_y_limits) != 2 or float(control_y_limits[0]) >= float(control_y_limits[1]):
            raise ValueError("control_y_limits must be an increasing (ymin, ymax) pair")
        u_ylim = [float(control_y_limits[0]), float(control_y_limits[1])]

    # Subplot order: 1) time step wise mean of π or 2x2 p(u|t,X_t), 2) v(t, \hat X_t), 3) X_t
    # Subplot 1: either 2x2 conditional p(u|t,X_t) or time-step-wise mean of π
    smooth_window_exact_panel_a = max(1, int(smooth_window))
    if smooth_window_exact_panel_a < 3:
        smooth_window_exact_panel_a = 3
    if smooth_window_xt is None:
        smooth_window_xt_eff = smooth_window
    else:
        smooth_window_xt_eff = max(1, int(smooth_window_xt))
    if panel_a_policy_distribution:
        t_val, x_val, time_label, _ = policy_point
        plot_panel_c_conditional_distribution(
            model, t_values=[t_val], x_values=[x_val],
            n_samples_u=int(model.params_train.get("panel_c_n_samples_u", 20000)),
            plot_method=str(model.params_train.get("panel_c_plot_method", "histogram")),
            device=device, output_folder=output_folder, ax=axes[0],
            legend_font_size=legend_font_size,
            legend_label_pinn="PINN sol of (4.12)",
            legend_label_flow="Flownet (samples)", legend_loc="upper right",
            force_density=True,
        )
        axes[0].set_title(f"$t={time_label}$, $x={x_val:.4f}$", fontsize=axis_label_size)
        axes[0].set_xlabel("$u$", fontsize=axis_label_size)
        axes[0].set_ylabel(r"$p(u\mid t,x)$", fontsize=axis_label_size)
        axes[0].tick_params(axis='both', labelsize=tick_label_size)
        axes[0].text(0.02, 0.95, '(a)', transform=axes[0].transAxes,
                     fontsize=subplot_label_size, va='top')
    elif use_2x2_panel_a:
        # Panel (a) = 4 subplots of p(u|t, X_t[0]) at t=T/4, T/2, 3T/4, T
        n_samples_u = int(model.params_train.get("panel_c_n_samples_u", 20000))
        plot_method = str(model.params_train.get("panel_c_plot_method", "histogram"))
        t_labels = ["$t=T/4$", "$t=T/2$", "$t=3T/4$", "$t=T$"]
        for i, (t_val, x_val) in enumerate(panel_a_tx_pairs[:4]):
            try:
                plot_panel_c_conditional_distribution(
                    model,
                    t_values=[float(t_val)],
                    x_values=[float(x_val)],
                    n_samples_u=n_samples_u,
                    plot_method=plot_method,
                    device=device,
                    output_folder=getattr(model, "output_folder", None),
                    ax=axes_a_2x2[i],
                    legend_font_size=legend_font_size,
                    legend_label_pinn=r"PINN sol of (4.12)",
                    legend_label_flow="Flownet (samples)",
                    legend_loc="upper right",
                )
            except Exception as e:
                axes_a_2x2[i].text(0.5, 0.5, f"Failed: {e}", ha="center", va="center", transform=axes_a_2x2[i].transAxes, fontsize=10)
            axes_a_2x2[i].set_title(t_labels[i] + f", $x$={float(x_val):.4f}", fontsize=axis_label_size)
            axes_a_2x2[i].text(0.02, 0.95, '(a)' if i == 0 else '', transform=axes_a_2x2[i].transAxes, fontsize=subplot_label_size,
                               verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            axes_a_2x2[i].set_xlabel(axes_a_2x2[i].get_xlabel(), fontsize=axis_label_size)
            axes_a_2x2[i].set_ylabel(axes_a_2x2[i].get_ylabel(), fontsize=axis_label_size)
            axes_a_2x2[i].tick_params(axis='both', which='major', labelsize=tick_label_size)
    elif traj_control_exact_mean is not None and traj_control_pred_mean is not None:
        # In `plot_traj_mode=ab`, panel-(a) should compare FlowNet vs PINN only (no Taylor curve).
        # Callers sometimes still pass `traj_control_taylor_mean`; guard here to avoid accidental Taylor in ab plots.
        plot_traj_mode_from_name = _infer_plot_traj_mode_from_filename(filename)
        if plot_traj_mode_from_name == "ab":
            traj_control_taylor_mean = None

        first_legend_dict = {'jump': None}
        # Calculate y-axis limits: y_min = min(pinn_y_min, approx_y_min, taylor_y_min) * 0.8, y_max = max(...) * 1.2
        # Convert to numpy for min/max calculation
        if isinstance(traj_control_pred_mean, torch.Tensor):
            approx_y_min = float(traj_control_pred_mean.min().item())
            approx_y_max = float(traj_control_pred_mean.max().item())
        else:
            approx_y_min = float(np.min(traj_control_pred_mean))
            approx_y_max = float(np.max(traj_control_pred_mean))

        if isinstance(traj_control_exact_mean, torch.Tensor):
            pinn_y_min = float(traj_control_exact_mean.min().item())
            pinn_y_max = float(traj_control_exact_mean.max().item())
        else:
            pinn_y_min = float(np.min(traj_control_exact_mean))
            pinn_y_max = float(np.max(traj_control_exact_mean))

        # Include Taylor in y-axis limits if available
        y_min_list = [pinn_y_min, approx_y_min]
        y_max_list = [pinn_y_max, approx_y_max]
        if traj_control_taylor_mean is not None:
            if isinstance(traj_control_taylor_mean, torch.Tensor):
                taylor_y_min = float(traj_control_taylor_mean.min().item())
                taylor_y_max = float(traj_control_taylor_mean.max().item())
            else:
                taylor_y_min = float(np.min(traj_control_taylor_mean))
                taylor_y_max = float(np.max(traj_control_taylor_mean))
            y_min_list.append(taylor_y_min)
            y_max_list.append(taylor_y_max)

        # Increase y-axis range to leave space for legend (especially when Taylor is present)
        y_min = min(y_min_list) * 0.75
        y_max = max(y_max_list) * 1.35  # Increased from 1.2 to 1.35 to leave space for legend
        y_limits = [y_min, y_max]

        # NOTE:
        # `traj_control_*_mean` are time-step-wise MEAN curves (typically identical across the batch dim).
        # They are NOT per-trajectory realizations, so we should:
        # - plot a single representative curve (avoid drawing multiple identical lines)
        # - do NOT overlay jump markers (jumps are trajectory-specific; mean curves should not show them)
        def _take_first_sample(arr):
            if arr is None:
                return None
            try:
                if isinstance(arr, np.ndarray):
                    if arr.ndim >= 1 and arr.shape[0] >= 1:
                        return arr[0:1]
                    return arr
                if torch.is_tensor(arr):
                    if arr.dim() >= 1 and int(arr.shape[0]) >= 1:
                        return arr[0:1]
                    return arr
            except Exception:
                return arr
            return arr

        t_test_mean = _take_first_sample(t_test)
        pred_mean = _take_first_sample(traj_control_pred_mean)
        exact_mean = _take_first_sample(traj_control_exact_mean)
        taylor_mean = _take_first_sample(traj_control_taylor_mean) if (traj_control_taylor_mean is not None) else None
        samples_index_mean = torch.tensor([0], dtype=torch.long)

        # Plot a single time-step-wise mean curve: predicted (network) vs exact (PINN) (+ optional Taylor)
        _plot_single_subplot(axes[0], t_test_mean, pred_mean, exact_mean,
                            "Mean of $\\hat{\\pi}(t)$", "Mean of $\\pi^*(t)$ (PINN)",
                            None, None, "Mean of $\\pi(t, \\hat{X}_t)$",
                            None, samples_index_mean, ds_idx, first_legend_dict,  # no jumps for mean curves
                            '(a)', filename_hint="control_time_step_mean", f_std_param=None,
                            f_third=taylor_mean, third_label=("Mean of $\\pi(t)$ (Taylor)" if taylor_mean is not None else None),
                            y_limits=y_limits,
                            axis_label_size=axis_label_size, tick_label_size=tick_label_size,
                            legend_font_size=legend_font_size, subplot_label_size=subplot_label_size,
                            smooth_window_exact=smooth_window_exact_panel_a)
    else:
        # If control trajectories are not provided, use policy mean trajectories
        first_legend_dict = {'jump': None}
        # Same guard for the fallback branch (rare): do not draw the third curve in ab mode.
        plot_traj_mode_from_name = _infer_plot_traj_mode_from_filename(filename)
        if plot_traj_mode_from_name == "ab":
            traj_u_third = None
            third_u = None
        _plot_single_subplot(axes[0], t_test, traj_u_mean, traj_u_test,
                            approximate_u, exact_u, start_u, end_u, ylabel_u,
                            P_test, samples_index, ds_idx, first_legend_dict,  # P_test to show jumps
                            '(a)', filename_hint="traj_u", f_std_param=f_std,
                            f_third=traj_u_third, third_label=third_u,
                            y_limits=u_ylim,
                            axis_label_size=axis_label_size, tick_label_size=tick_label_size,
                            legend_font_size=legend_font_size, subplot_label_size=subplot_label_size,
                            smooth_window_exact=smooth_window_exact_panel_a)

    # Subplot 2: Value trajectories v(t, \hat X_t) (previously subplot 2)
    first_legend_dict = {'jump': None}
    _plot_single_subplot(axes[1], t_test, traj_value, traj_value_test,
                        approximate_v, exact_v, start_v, end_v, ylabel_v,
                        P_test, samples_index, ds_idx, first_legend_dict,
                        '(b)', filename_hint="traj_value", f_std_param=f_std,
                        f_third=traj_value_third, third_label=third_v,
                        axis_label_size=axis_label_size, tick_label_size=tick_label_size,
                        legend_font_size=legend_font_size, subplot_label_size=subplot_label_size)

    # Subplot 3: Content depends on panel_c_mode and layout
    # When use_2x2_panel_a: (c) always shows X_t trajectory (FlowNet vs PINN with sampled control).
    # Otherwise: distribution or state trajectories per panel_c_mode.
    if panel_a_policy_distribution or use_2x2_panel_a:
        # Panel (c) = X_t vs t (trajectories from net_actor sample and PINN icdf sample)
        # When traj_Xt is mean over runs it has shape (1, N+1, 1); clamp samples_index to avoid index 1 out of bounds
        first_legend_dict = {'jump': None}
        M_xt = getattr(traj_Xt, "shape", (M,))[0] if traj_Xt is not None else M
        if traj_Xt_test is not None and hasattr(traj_Xt_test, "shape") and len(traj_Xt_test.shape) > 0:
            M_xt = min(M_xt, traj_Xt_test.shape[0])
        samples_index_xt = samples_index[samples_index < M_xt] if (hasattr(samples_index, "__len__") and len(samples_index) > 0) else samples_index
        if (hasattr(samples_index_xt, "numel") and samples_index_xt.numel() == 0) or (hasattr(samples_index_xt, "__len__") and len(samples_index_xt) == 0):
            samples_index_xt = torch.tensor([0], dtype=torch.long) if torch.is_tensor(samples_index) else np.array([0], dtype=np.int64)
        _plot_single_subplot(axes[2], t_test, traj_Xt, traj_Xt_test,
                            approximate_x, exact_x, start_x, end_x, ylabel_x,
                            P_test, samples_index_xt, ds_idx_xt, first_legend_dict,
                            '(c)', filename_hint="traj_Xt", f_std_param=f_std,
                            f_third=traj_Xt_third, third_label=third_x,
                            axis_label_size=axis_label_size, tick_label_size=tick_label_size,
                            legend_font_size=legend_font_size, subplot_label_size=subplot_label_size,
                            smooth_window_exact=smooth_window_xt_eff)
    elif panel_c_mode in ("X_dist_evolve_T", "Conddist_u_T", "U_dist_on_XT"):
        # For distribution plots, render panel-c directly if model is provided.
        axes[2].set_xlabel("", fontsize=axis_label_size)
        axes[2].set_ylabel("", fontsize=axis_label_size)
        if panel_c_defer:
            axes[2].text(0.02, 0.95, '(c)', transform=axes[2].transAxes, fontsize=subplot_label_size,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        elif model is None:
            axes[2].text(0.5, 0.5, "Panel (c) needs model", ha="center", va="center", transform=axes[2].transAxes)
        else:
            mode_norm = panel_c_mode
            t_rep = getattr(model, "_panel_c_t_rep", None)
            if t_rep is None:
                t_rep = float(getattr(model, "T", 0.0))
            x_rep = getattr(model, "_panel_c_xt_rep", None)
            if x_rep is None:
                x0_eval = getattr(model, "X0_eval", None)
                if x0_eval is not None:
                    try:
                        x_rep = float(x0_eval.mean().item())
                    except Exception:
                        x_rep = float(np.mean(np.asarray(x0_eval)))
                elif hasattr(model, "Xi"):
                    try:
                        x_rep = float(to_scalar(model.Xi))
                    except Exception:
                        x_rep = None
            if x_rep is None:
                x_rep = 10.0

            if mode_norm == "Conddist_u_T":
                n_samples_u = 20000
                plot_method = "histogram"
                try:
                    n_samples_u = int(model.params_train.get("panel_c_n_samples_u", n_samples_u))
                    plot_method = str(model.params_train.get("panel_c_plot_method", plot_method))
                except Exception:
                    pass
                try:
                    plot_panel_c_conditional_distribution(
                        model,
                        t_values=[float(t_rep)],
                        x_values=[float(x_rep)],
                        n_samples_u=n_samples_u,
                        plot_method=plot_method,
                        device=device,
                        output_folder=getattr(model, "output_folder", None),
                        ax=axes[2],
                        legend_font_size=legend_font_size,
                    )
                except Exception as e:
                    axes[2].text(0.5, 0.5, f"Panel (c) failed: {e}", ha="center", va="center", transform=axes[2].transAxes)
            elif mode_norm == "X_dist_evolve_T":
                # Plot evolved x distribution using PRE-COMPUTED trajectories (no extra PINN calls).
                # This mirrors the 0129 plotting behavior:
                # - Compare FlowNet terminal states X_pred(T) vs PINN terminal states X_test_pinn(T)
                # - Prefer KDE for smooth curves; fallback to histogram when KDE fails.
                print("[plot] Plotting evolved x distribution at T using pre-computed trajectories (no extra PINN calls)...")
                try:
                    # traj_Xt and traj_Xt_test are expected to be [M, N+1, 1] (numpy or torch)
                    traj_Xt_use = panel_c_traj_Xt if panel_c_traj_Xt is not None else traj_Xt
                    traj_Xt_test_use = panel_c_traj_Xt_test if panel_c_traj_Xt_test is not None else traj_Xt_test
                    if (traj_Xt_use is None) or (traj_Xt_test_use is None):
                        axes[2].text(0.5, 0.5, "Trajectory data not available", ha="center", va="center",
                                     transform=axes[2].transAxes)
                        print("[WARNING] Trajectory data not available for distribution comparison")
                    else:
                        def _as_np(a):
                            if a is None:
                                return None
                            if torch.is_tensor(a):
                                return a.detach().cpu().numpy()
                            return np.asarray(a)

                        X_pred_np = _as_np(traj_Xt_use)
                        X_pinn_np = _as_np(traj_Xt_test_use)
                        # Take terminal time slice and flatten
                        X1_flow_np = X_pred_np[:, -1, :].reshape(-1) if X_pred_np.ndim >= 2 else X_pred_np.reshape(-1)
                        X1_pinn_np = X_pinn_np[:, -1, :].reshape(-1) if X_pinn_np.ndim >= 2 else X_pinn_np.reshape(-1)

                        # Filter invalid values
                        valid_flow = np.isfinite(X1_flow_np) & (X1_flow_np > 0)
                        valid_pinn = np.isfinite(X1_pinn_np) & (X1_pinn_np > 0)
                        X1_flow_np = X1_flow_np[valid_flow]
                        X1_pinn_np = X1_pinn_np[valid_pinn]

                        if len(X1_flow_np) > 0 and len(X1_pinn_np) > 0:
                            pinn_x_max = float(getattr(model, "params_train", {}).get("pinn_x_max", 50.0))
                            x_min = 0.0
                            x_max = min(50.0, pinn_x_max)

                            # Use KDE for smooth curves (same as compare_icdf_vs_flow_samples in 0129)
                            try:
                                kde_bw = None
                                try:
                                    kde_bw = getattr(model, "params_train", {}).get("panel_c_kde_bw", None)
                                except Exception:
                                    kde_bw = None
                                kde_pinn = gaussian_kde(X1_pinn_np, bw_method=kde_bw) if kde_bw is not None else gaussian_kde(X1_pinn_np)
                                kde_flow = gaussian_kde(X1_flow_np, bw_method=kde_bw) if kde_bw is not None else gaussian_kde(X1_flow_np)
                                x_plot = np.linspace(x_min, x_max, 200)
                                axes[2].plot(x_plot, kde_pinn(x_plot), label='PINN', linewidth=2, alpha=0.7, color='darkviolet')
                                axes[2].plot(x_plot, kde_flow(x_plot), label='Flownet', linewidth=2, alpha=0.7, color='darkorange')
                            except Exception as e:
                                print(f"[WARNING] KDE failed: {e}, falling back to histogram")
                                axes[2].hist(X1_pinn_np, bins=150, alpha=0.5, label='PINN',
                                             density=True, color='darkviolet', edgecolor='black', linewidth=0.5)
                                axes[2].hist(X1_flow_np, bins=150, alpha=0.5, label='Flownet',
                                             density=True, color='darkorange', edgecolor='black', linewidth=0.5)

                            axes[2].set_xlabel('$X_T$', fontsize=axis_label_size)
                            axes[2].set_ylabel('Density of state', fontsize=axis_label_size)
                            axes[2].tick_params(axis='both', which='major', labelsize=tick_label_size)
                            axes[2].set_xlim(x_min, x_max)
                            axes[2].legend(fontsize=legend_font_size, loc='best', framealpha=0.5, edgecolor='gray')
                            axes[2].grid(True, alpha=0.3)
                            print(f"[plot] ✓ Distribution comparison plotted using pre-computed trajectories "
                                  f"(n_flow={len(X1_flow_np)}, n_pinn={len(X1_pinn_np)}, no extra PINN calls)")
                        else:
                            axes[2].text(0.5, 0.5, 'No valid data for distribution', ha='center', va='center',
                                         transform=axes[2].transAxes)
                            print("[WARNING] No valid data for distribution comparison")
                except Exception as e:
                    axes[2].text(0.5, 0.5, f"Panel (c) failed: {e}", ha="center", va="center", transform=axes[2].transAxes)
            elif mode_norm == "U_dist_on_XT":
                # Plot terminal control distribution using PRE-COMPUTED trajectories (no extra PINN calls).
                # One u per terminal state sample: u_T^{(m)} = pi(T, X_T^{(m)}).
                print("[plot] Plotting terminal control distribution u_T using pre-computed trajectories (no extra PINN calls)...")
                try:
                    traj_u_mean_use = panel_c_traj_u_mean if panel_c_traj_u_mean is not None else traj_u_mean
                    traj_u_test_use = panel_c_traj_u_test if panel_c_traj_u_test is not None else traj_u_test
                    if (traj_u_mean_use is None) and (traj_u_test_use is None):
                        axes[2].text(0.5, 0.5, "Control trajectory data not available", ha="center", va="center",
                                     transform=axes[2].transAxes)
                        print("[WARNING] Control trajectory data not available for u_T distribution comparison")
                    else:
                        def _as_np(a):
                            if a is None:
                                return None
                            if torch.is_tensor(a):
                                return a.detach().cpu().numpy()
                            return np.asarray(a)

                        u_flow_np = _as_np(traj_u_mean_use) if traj_u_mean_use is not None else None
                        u_pinn_np = _as_np(traj_u_test_use) if traj_u_test_use is not None else None

                        # Take terminal time slice and flatten
                        uT_flow = None
                        uT_pinn = None
                        if u_flow_np is not None:
                            uT_flow = u_flow_np[:, -1, :].reshape(-1) if u_flow_np.ndim >= 2 else u_flow_np.reshape(-1)
                            uT_flow = uT_flow[np.isfinite(uT_flow)]
                        if u_pinn_np is not None:
                            uT_pinn = u_pinn_np[:, -1, :].reshape(-1) if u_pinn_np.ndim >= 2 else u_pinn_np.reshape(-1)
                            uT_pinn = uT_pinn[np.isfinite(uT_pinn)]

                        valid_flow = (uT_flow is not None) and (len(uT_flow) > 0)
                        valid_pinn = (uT_pinn is not None) and (len(uT_pinn) > 0)

                        if valid_flow or valid_pinn:
                            # Prefer plotting within training action bounds when available
                            try:
                                u_all = np.r_[uT_flow] if valid_flow and (not valid_pinn) else (
                                    np.r_[uT_pinn] if valid_pinn and (not valid_flow) else np.r_[uT_flow, uT_pinn]
                                )
                                u_min = float(getattr(model, "params_train", {}).get("pinn_u_min", np.nanmin(u_all)))
                                u_max = float(getattr(model, "params_train", {}).get("pinn_u_max", np.nanmax(u_all)))
                            except Exception:
                                u_all = np.r_[uT_flow] if valid_flow and (not valid_pinn) else (
                                    np.r_[uT_pinn] if valid_pinn and (not valid_flow) else np.r_[uT_flow, uT_pinn]
                                )
                                u_min = float(np.nanmin(u_all))
                                u_max = float(np.nanmax(u_all))
                            if not np.isfinite(u_min) or not np.isfinite(u_max) or (u_max <= u_min):
                                u_min = float(np.nanmin(u_all))
                                u_max = float(np.nanmax(u_all))

                            plot_method = "kde"
                            try:
                                plot_method = str(getattr(model, "params_train", {}).get("panel_c_plot_method", plot_method))
                            except Exception:
                                pass

                            # Smoothing controls (override via params_train if needed).
                            bins = int(getattr(model, "params_train", {}).get("panel_c_u_dist_bins", 60))
                            kde_bw = getattr(model, "params_train", {}).get("panel_c_kde_bw", 0.5)

                            if plot_method == "histogram":
                                hist_edges = None
                                if valid_pinn:
                                    hist_pinn, edges = np.histogram(
                                        uT_pinn, bins=bins, range=(u_min, u_max), density=True
                                    )
                                    hist_edges = edges
                                if valid_flow:
                                    hist_flow, edges = np.histogram(
                                        uT_flow, bins=bins, range=(u_min, u_max), density=True
                                    )
                                    hist_edges = edges if hist_edges is None else hist_edges
                                if hist_edges is not None:
                                    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
                                    if valid_pinn:
                                        axes[2].plot(centers, hist_pinn, drawstyle="steps-mid",
                                                     label='PINN', linewidth=2, alpha=0.7, color='darkviolet')
                                    if valid_flow:
                                        axes[2].plot(centers, hist_flow, drawstyle="steps-mid",
                                                     label='Flownet', linewidth=2, alpha=0.7, color='darkorange')
                            else:
                                try:
                                    u_plot = np.linspace(u_min, u_max, 200)
                                    if valid_pinn:
                                        kde_pinn = gaussian_kde(uT_pinn, bw_method=kde_bw)
                                        axes[2].plot(u_plot, kde_pinn(u_plot), label='PINN', linewidth=2, alpha=0.7, color='darkviolet')
                                    if valid_flow:
                                        kde_flow = gaussian_kde(uT_flow, bw_method=kde_bw)
                                        axes[2].plot(u_plot, kde_flow(u_plot), label='Flownet', linewidth=2, alpha=0.7, color='darkorange')
                                except Exception as e:
                                    print(f"[WARNING] KDE failed: {e}, falling back to histogram")
                                    hist_edges = None
                                    if valid_pinn:
                                        hist_pinn, edges = np.histogram(
                                            uT_pinn, bins=bins, range=(u_min, u_max), density=True
                                        )
                                        hist_edges = edges
                                    if valid_flow:
                                        hist_flow, edges = np.histogram(
                                            uT_flow, bins=bins, range=(u_min, u_max), density=True
                                        )
                                        hist_edges = edges if hist_edges is None else hist_edges
                                    if hist_edges is not None:
                                        centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
                                        if valid_pinn:
                                            axes[2].plot(centers, hist_pinn, drawstyle="steps-mid",
                                                         label='PINN', linewidth=2, alpha=0.7, color='darkviolet')
                                        if valid_flow:
                                            axes[2].plot(centers, hist_flow, drawstyle="steps-mid",
                                                         label='Flownet', linewidth=2, alpha=0.7, color='darkorange')

                            axes[2].set_xlabel('$u_T$', fontsize=axis_label_size)
                            axes[2].set_ylabel(r'Density of $u(t,\hat X_t)$', fontsize=axis_label_size)
                            axes[2].tick_params(axis='both', which='major', labelsize=tick_label_size)
                            axes[2].set_xlim(u_min, u_max)
                            axes[2].legend(fontsize=legend_font_size, loc='best', framealpha=0.5, edgecolor='gray')
                            axes[2].grid(True, alpha=0.3)
                            print(f"[plot] ✓ u_T distribution plotted using pre-computed trajectories "
                                  f"(n_flow={len(uT_flow) if valid_flow else 0}, n_pinn={len(uT_pinn) if valid_pinn else 0}, "
                                  f"no extra PINN calls)")
                        else:
                            axes[2].text(0.5, 0.5, 'No valid data for u_T distribution', ha='center', va='center',
                                         transform=axes[2].transAxes)
                            print("[WARNING] No valid data for u_T distribution comparison")
                except Exception as e:
                    axes[2].text(0.5, 0.5, f"Panel (c) failed: {e}", ha="center", va="center", transform=axes[2].transAxes)
            axes[2].text(0.02, 0.95, '(c)', transform=axes[2].transAxes, fontsize=subplot_label_size,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    else:
        # Default: Plot state trajectories X_t
        # This includes None, "X_traj_evalmode_sample", "X_traj_evalmode_policymean"
        print(f"[plot_trajectory_combined_abc] Plotting state trajectories in subplot 3 (panel_c_mode={panel_c_mode})")
        print(f"[plot_trajectory_combined_abc] traj_Xt is None: {traj_Xt is None}, traj_Xt_test is None: {traj_Xt_test is None}")
        if traj_Xt is not None:
            print(f"[plot_trajectory_combined_abc] traj_Xt shape: {traj_Xt.shape if hasattr(traj_Xt, 'shape') else 'N/A'}")
        if traj_Xt_test is not None:
            print(f"[plot_trajectory_combined_abc] traj_Xt_test shape: {traj_Xt_test.shape if hasattr(traj_Xt_test, 'shape') else 'N/A'}")
        first_legend_dict = {'jump': None}
        _plot_single_subplot(axes[2], t_test, traj_Xt, traj_Xt_test,
                            approximate_x, exact_x, start_x, end_x, ylabel_x,
                            P_test, samples_index, ds_idx_xt, first_legend_dict,
                            '(c)', filename_hint="traj_Xt", f_std_param=f_std,
                            f_third=traj_Xt_third, third_label=third_x,
                            axis_label_size=axis_label_size, tick_label_size=tick_label_size,
                            legend_font_size=legend_font_size, subplot_label_size=subplot_label_size,
                            smooth_window_exact=smooth_window_xt_eff)

    plt.tight_layout()

    if not return_fig_axes:
        # Tag filename so we can distinguish eval runs done with frozen script vs current script.
        # Delegated frozen runs set MERTON_USE_FROZEN_SCRIPT_EVAL=1 in the environment.
        _use_frozen = os.environ.get("MERTON_USE_FROZEN_SCRIPT_EVAL", "").strip()
        if _use_frozen == "1":
            _tag = "_use_frozen_script_eval"
        elif _use_frozen == "0":
            _tag = "_no_frozen_script_eval"
        else:
            _tag = ""
        if _tag and (_tag not in filename):
            filename = f"{filename}{_tag}"

        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, f'{filename}.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.show()
    else:
        return fig, axes




def plot_extendability_test(
    model,
    T_train,
    T_eval=None,
    c=3,
    device="cpu",
    output_folder='./pred_traj',
    filename="extendability_test",
    downsample_step=1,
    smooth_window=1,
    axis_label_size=24,
    tick_label_size=20,
    legend_font_size=15,
    itr=0,
):
    """
    Plot extendability test: training on [0, T_train], evaluating on [0, T_eval] where T_eval = c * T_train.

    Args:
        model: FBSNN model instance
        T_train: Training time horizon T
        T_eval: Evaluation time horizon (default: c * T_train)
        c: Multiplier for evaluation time (default: 3)
        device: Device to use
        output_folder: Output folder for saving plots
        filename: Base filename for output
        downsample_step: Downsampling step for plotting
        smooth_window: Smoothing window size
        axis_label_size: Font size for axis labels
        tick_label_size: Font size for tick labels
        legend_font_size: Font size for legend
    """
    from matplotlib import gridspec
    from matplotlib.patches import ConnectionPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if T_eval is None:
        T_eval = c * T_train

    # print(f"[plot_extendability_test] Training horizon: [0, {T_train:.2f}], Evaluation horizon: [0, {T_eval:.2f}]")

    device = device or model.device

    # IMPORTANT: Ensure fixed evaluation set is available for [0, T_train] portion
    # This ensures the states at [0, T_train] match the regular plot
    model._ensure_eval_set()

    # Adjust pinn_x_max for extended time horizon: when T becomes c*T, x_max should become c*x_max
    # Save original values
    original_pinn_x_max = model.params_train.get("pinn_x_max", None)
    original_pinn_x_min = model.params_train.get("pinn_x_min", None)

    if original_pinn_x_max is not None:
        # Scale pinn_x_max by c to accommodate extended time horizon
        scaled_pinn_x_max = c * original_pinn_x_max
        model.params_train["pinn_x_max"] = scaled_pinn_x_max
        print(f"[plot_extendability_test] Adjusted pinn_x_max: {original_pinn_x_max} -> {scaled_pinn_x_max} (c={c})")

    if original_pinn_x_min is not None and original_pinn_x_min > 0:
        # Scale pinn_x_min by c as well (if it's not 0)
        scaled_pinn_x_min = c * original_pinn_x_min
        model.params_train["pinn_x_min"] = scaled_pinn_x_min
        print(f"[plot_extendability_test] Adjusted pinn_x_min: {original_pinn_x_min} -> {scaled_pinn_x_min} (c={c})")

    try:
        # Generate trajectories for extended time horizon
        # IMPORTANT: Use the same fixed evaluation set for [0, T_train] to match regular plot
        # Only generate new random processes for [T_train, T_eval]

        # Get fixed evaluation set for [0, T_train]
        # IMPORTANT: Ensure t_train ends exactly at T_train
        # If model.t_eval ends at model.T (which should equal T_train), use it directly
        # Otherwise, we need to extract the portion [0, T_train] from model.t_eval
        t_train_full = model.t_eval  # [M, N_train_full+1, 1] - fixed evaluation set
        T_train_actual = float(t_train_full[0, -1, 0].item())  # Actual end time of t_train

        if abs(T_train_actual - T_train) < 1e-6:
            # t_train already ends at T_train, use it directly
            t_train = t_train_full
            W_train = model.W_eval  # [M, N_train+1, D] - fixed Brownian motion
            P_train = model.P_eval  # [M, max_jump_train, 1] - fixed Poisson process
            N_train = t_train.shape[1] - 1
        else:
            # Extract portion [0, T_train] from t_train_full
            # Find the index where t_train_full first exceeds T_train
            mask = t_train_full[0, :, 0] <= T_train
            idx_T_train = mask.sum().item()  # Last index <= T_train
            if idx_T_train < t_train_full.shape[1]:
                # Include the point at T_train (or closest point)
                t_train = t_train_full[:, :idx_T_train+1, :]  # [M, idx_T_train+1, 1]
                W_train = model.W_eval[:, :idx_T_train+1, :]  # [M, idx_T_train+1, D]
                # For P_train, filter jumps <= T_train
                P_train = model.P_eval.clone()  # [M, max_jump_train, 1]
                for m in range(model.M):
                    jumps = P_train[m, :, 0]
                    jumps[jumps > T_train] = 0  # Filter out jumps > T_train
                N_train = idx_T_train
            else:
                # T_train is beyond t_train_full, use all of it
                t_train = t_train_full
                W_train = model.W_eval
                P_train = model.P_eval
                N_train = t_train.shape[1] - 1

        X0_eval = model.X0_eval  # Use same X0 as regular plot

        # Calculate time step for training portion
        dt_train = model.dt  # Time step for training portion

        print(f"[plot_extendability_test] Using fixed evaluation set for [0, T_train]: "
              f"T_train={T_train:.4f}, T_train_actual={T_train_actual:.4f}, N_train={N_train}, dt_train={dt_train:.6f}")

        # Create time grid for extended portion [T_train, T_eval]
        T_extended = T_eval - T_train
        N_extended = int(T_extended / dt_train)  # Use same dt for consistency
        dt_eval = dt_train  # Keep same time step

        # Generate time grid for extended portion
        t_extended = torch.linspace(T_train, T_eval, N_extended + 1, dtype=torch.float32).unsqueeze(0).repeat(model.M, 1).unsqueeze(-1).to(device)  # [M, N_extended+1, 1]

        # Generate Brownian motion for extended portion [T_train, T_eval]
        # Start from the final value of W_train to ensure continuity
        W_train_final = W_train[:, -1:, :]  # [M, 1, D] - final value at T_train
        dW_extended = torch.randn(model.M, N_extended, model.D, device=device, dtype=torch.float32) * torch.tensor(np.sqrt(dt_eval), dtype=torch.float32, device=device)
        W_extended_cumsum = torch.cumsum(dW_extended, dim=1)  # [M, N_extended, D]
        W_extended = torch.cat([W_train_final, W_train_final + W_extended_cumsum], dim=1)  # [M, N_extended+1, D]

        # Generate Poisson process for extended portion [T_train, T_eval]
        max_jump_extended = int(3 * model.params_equ["lambda"] * T_extended) + 1
        if max_jump_extended > 0:
            jump_times_extended = torch.cumsum(torch.distributions.Exponential(model.params_equ["lambda"]).sample((model.M, max_jump_extended)).to(device), dim=1)
            jump_times_extended = jump_times_extended + T_train  # Shift to start from T_train
            jump_times_extended[jump_times_extended > T_eval] = 0
            jump_times_extended, _ = torch.sort(jump_times_extended, dim=1)
            P_extended = jump_times_extended.unsqueeze(-1).to(device)  # [M, max_jump_extended, 1]
        else:
            P_extended = torch.zeros(model.M, 1, 1, device=device, dtype=torch.float32)

        # Combine training and extended portions
        # Time grid: concatenate [0, T_train] and [T_train, T_eval] (remove duplicate at T_train)
        t_eval = torch.cat([t_train, t_extended[:, 1:, :]], dim=1)  # [M, N_train+1+N_extended, 1]

        # Brownian motion: concatenate (W_train already includes final value)
        W_eval = torch.cat([W_train, W_extended[:, 1:, :]], dim=1)  # [M, N_train+1+N_extended, D]

        # Poisson process: combine jumps from both portions
        # For each trajectory, combine jumps from [0, T_train] and [T_train, T_eval]
        # P_train: [M, max_jump_train, 1], P_extended: [M, max_jump_extended, 1]
        max_jump_train = P_train.shape[1]
        max_jump_extended = P_extended.shape[1]
        max_jump_total = max_jump_train + max_jump_extended

        P_eval = torch.zeros(model.M, max_jump_total, 1, device=device, dtype=torch.float32)
        for m in range(model.M):
            # Get jumps from training portion [0, T_train]
            jumps_train = P_train[m, :, 0]  # [max_jump_train]
            jumps_train = jumps_train[jumps_train > 0]  # Filter zeros

            # Get jumps from extended portion [T_train, T_eval]
            jumps_extended = P_extended[m, :, 0]  # [max_jump_extended]
            jumps_extended = jumps_extended[jumps_extended > 0]  # Filter zeros

            # Combine and sort
            if len(jumps_train) > 0 and len(jumps_extended) > 0:
                jumps_combined = torch.cat([jumps_train, jumps_extended], dim=0)
            elif len(jumps_train) > 0:
                jumps_combined = jumps_train
            elif len(jumps_extended) > 0:
                jumps_combined = jumps_extended
            else:
                jumps_combined = torch.tensor([], device=device, dtype=torch.float32)

            if len(jumps_combined) > 0:
                jumps_combined, _ = torch.sort(jumps_combined)
                n_jumps = min(len(jumps_combined), max_jump_total)
                P_eval[m, :n_jumps, 0] = jumps_combined[:n_jumps]

        # Calculate total number of time steps
        N_eval = t_eval.shape[1] - 1

        # Use fixed X0 from evaluation set
        model.X0_eval = X0_eval

        # Ensure model is in eval mode
        model.net_actor.eval()
        model.net_critic.eval()

        # Generate predictions for extended horizon
        # IMPORTANT: predict() uses the same random processes (t_eval, W_eval, P_eval) for both:
        #   - X_pred_eval: learned policy trajectory (Flownet)
        #   - X_test_eval: exact policy trajectory (PINN)
        # This ensures fair comparison: same random noise (dW, dMt) for both trajectories
        # In [T_train, T_eval], both use the same extended random processes, ensuring alignment
        print(f"[plot_extendability_test] Generating predictions for extended horizon...")
        print(f"[plot_extendability_test] Using same random processes (t_eval, W_eval, P_eval) for both X_pred and X_test_pinn")
        X_pred_eval, Y_pred_eval, control_pred_eval, X_test_eval, _, _, _, _, _ = model.predict(
            t_eval, W_eval, P_eval, N_eval=N_eval
        )
        print(f"[plot_extendability_test] Generated trajectories: X_pred_eval shape={X_pred_eval.shape}, X_test_eval shape={X_test_eval.shape}")
    finally:
        # Restore original pinn_x_max and pinn_x_min values
        if original_pinn_x_max is not None:
            model.params_train["pinn_x_max"] = original_pinn_x_max
        if original_pinn_x_min is not None:
            model.params_train["pinn_x_min"] = original_pinn_x_min
        print(f"[plot_extendability_test] Restored original pinn_x_max and pinn_x_min")

    # Compute mean of policy and value function using the correct methods (aligned with merton_pinn_0129.py eval_pinn):
    # IMPORTANT: Both learned and exact policy means should be computed on X_pred_eval states,
    # not X_test_eval, to ensure fair comparison on the same trajectory.
    # This applies to BOTH [0, T_train] and [T_train, T_eval] portions:
    #   - In [0, T_train]: X_pred_eval uses fixed evaluation set (same as regular plot)
    #   - In [T_train, T_eval]: X_pred_eval uses extended random processes (same as X_test_eval)
    # Both X_pred_eval and X_test_eval use the SAME random processes (t_eval, W_eval, P_eval),
    # ensuring alignment for fair comparison of V and mean of policy on the same trajectory.
    # 1. For learned policy: use mixture_mean on X_pred_eval
    # 2. For exact policy: use compute_exact_entropy_with_cache on X_pred_eval (not X_test_eval!)
    print(f"[plot_extendability_test] Computing mean of policy and value function on X_pred_eval trajectory...")
    print(f"[plot_extendability_test] This ensures both learned and exact policies are evaluated on the SAME trajectory (X_pred_eval)")
    print(f"[plot_extendability_test] Both X_pred_eval and X_test_eval use the SAME random processes, ensuring alignment in [T_train, T_eval]")

    if model.params_train["entropy_alpha"] == 0:
        # Deterministic case: no entropy, so mean = control itself
        control_pred_mean_eval = control_pred_eval
        # Exact policy on X_pred_eval (not X_test_eval) for fair comparison
        control_test_eval, optimal_value_test_eval = model.compute_exact_no_exploratory(X_pred_eval)
        control_test_eval = control_test_eval * torch.ones_like(X_pred_eval)
        control_test_mean_eval = control_test_eval
        u_pred_label = "u_pred"
        u_test_label = "u_test"
        u_ylabel = "$u(t, \\hat{X}_t)$"
    else:
        # Exploratory case: compute mean using mixture_mean and compute_exact_entropy_with_cache
        # Both computed on X_pred_eval states for fair comparison (aligned with merton_pinn_0129.py)
        M, N_plus_1, D = X_pred_eval.shape
        # Use model.t_predict_set (which predict actually used) to match X_pred_eval's time dimension
        # This ensures the time tensor matches X_pred_eval exactly, same as eval_pinn in merton_pinn_0129.py
        if hasattr(model, 't_predict_set') and model.t_predict_set is not None:
            t_eval_full = model.t_predict_set[:, :N_plus_1, :]  # [M, N_eval+1, 1] - match X_pred_eval's time dimension
        else:
            # Fallback to t_eval if t_predict_set is not available
            t_eval_full = t_eval[:, :N_plus_1, :]  # [M, N_eval+1, 1] - match X_pred_eval's time dimension

        # Reshape to [M*(N_eval+1), ...] for batch processing
        X_pred_flat = X_pred_eval.reshape(-1, D)  # [M*(N_eval+1), D]
        t_eval_flat = t_eval_full.reshape(-1, 1)  # [M*(N_eval+1), 1]
        state_flat = torch.cat([t_eval_flat, X_pred_flat], dim=1)  # [M*(N_eval+1), 2]

        # 1. Learned policy mean: use mixture_mean on X_pred_eval states
        K_mean_eval = int(model.params_train.get("K_mean_eval", 256))
        actor_type = getattr(model.net_actor, "actor_type", "")
        if actor_type in ("hyper_gaussian_split", "gaussian_two_stage"):
            control_mean_mc_flat = model.net_actor.mixture_mean(
                state_flat, n_samples=K_mean_eval, antithetic=True
            )  # [M*(N_eval+1), D]
        else:
            # For simple Gaussian policies, mu_pred is already the mean
            mu_pred, _, _ = model.net_actor(state_flat)
            control_mean_mc_flat = mu_pred

        # Reshape back to [M, N_eval+1, D]
        control_pred_mean_eval = control_mean_mc_flat.reshape(M, N_plus_1, D)

        # 2. Exact policy mean: use compute_exact_entropy_with_cache on X_pred_eval (not X_test_eval!)
        # This ensures we compare learned vs exact policy on the same trajectory (X_pred_eval)
        # IMPORTANT: Adjust pinn_x_max and pinn_x_min based on actual X_pred_eval range
        # to ensure x_grid covers all states in the trajectory
        X_pred_eval_flat = X_pred_eval.reshape(-1)  # Flatten to get all x values
        X_pred_eval_min = float(X_pred_eval_flat.min().detach().cpu().item())
        X_pred_eval_max = float(X_pred_eval_flat.max().detach().cpu().item())

        # Get current pinn_x_min and pinn_x_max (may have been scaled for extended horizon)
        current_pinn_x_min = float(model.params_train.get("pinn_x_min", 0.0))
        current_pinn_x_max = float(model.params_train.get("pinn_x_max", 100.0))

        # Adjust x_grid range: ensure it covers all X_pred_eval values
        # x_max = max(pinn_x_max, max(X_pred_eval)) to ensure coverage
        # x_min = min(pinn_x_min, min(X_pred_eval)) to ensure coverage
        # But also respect the original pinn_x_max limit (don't go beyond 3*pinn_x_max to avoid too large grid)
        original_pinn_x_max = original_pinn_x_max if 'original_pinn_x_max' in locals() else current_pinn_x_max
        original_pinn_x_min = original_pinn_x_min if 'original_pinn_x_min' in locals() else current_pinn_x_min

        # IMPORTANT: For extendability test, time extends to c*T, so states X_t can go much further
        # x_grid boundary should be at least original_pinn_x_max * c to accommodate extended time horizon
        # Use max(original_pinn_x_max * c, max(X_pred_eval)) for x_max to ensure coverage
        # Use min(original_pinn_x_min, min(X_pred_eval)) for x_min (but ensure >= 0)
        # Get original values (before scaling for extended horizon)
        original_pinn_x_max_for_grid = original_pinn_x_max if original_pinn_x_max is not None else current_pinn_x_max
        original_pinn_x_min_for_grid = original_pinn_x_min if original_pinn_x_min is not None else current_pinn_x_min

        # For extendability: x_max should be at least original_pinn_x_max * c (c is the extension factor)
        # This ensures the grid covers states that can be reached in the extended time horizon
        min_x_max_for_extended = original_pinn_x_max_for_grid * c
        adjusted_pinn_x_max = max(min_x_max_for_extended, X_pred_eval_max)
        adjusted_pinn_x_min = min(max(original_pinn_x_min_for_grid, 0.0), X_pred_eval_min) if X_pred_eval_min > 0 else max(original_pinn_x_min_for_grid, 0.0)

        # IMPORTANT: Also scale grid resolution (n_x) by c to maintain same resolution
        # If original range is [x_min, x_max] with n_x points, extended range [x_min, x_max * c] should have n_x * c points
        # However, for better accuracy in extended range, use a more aggressive scaling or minimum resolution
        original_pinn_n_x = int(model.params_train.get("pinn_n_x", 500))
        # Use max(n_x * c, n_x * sqrt(c)) to ensure finer grid for extended range
        # Or use a minimum resolution (e.g., 1000) to ensure accuracy
        min_resolution = int(model.params_train.get("pinn_n_x_min_extended", 2000))  # Minimum resolution for extended range
        adjusted_pinn_n_x = max(int(original_pinn_n_x * c), min_resolution)

        # IMPORTANT: For extended range [T_train, T_eval], we need a separate PINN training
        # with more iterations and finer grid to ensure accuracy for states outside original training range
        # Increase PINN training iterations for extended range (default: 1.5x original, or use config)
        original_pinn_n_iter = int(model.params_train.get("pinn_n_iter", 2000))
        extended_pinn_n_iter_multiplier = float(model.params_train.get("pinn_n_iter_extended_multiplier", 3.0))
        adjusted_pinn_n_iter = int(original_pinn_n_iter * extended_pinn_n_iter_multiplier)

        # Temporarily adjust pinn_x_max, pinn_x_min, pinn_n_x, and pinn_n_iter for compute_exact_entropy_with_cache
        model.params_train["pinn_x_max"] = adjusted_pinn_x_max
        model.params_train["pinn_x_min"] = adjusted_pinn_x_min
        model.params_train["pinn_n_x"] = adjusted_pinn_n_x
        model.params_train["pinn_n_iter"] = adjusted_pinn_n_iter
        print(f"[plot_extendability_test] Adjusted x_grid range and resolution for compute_exact_entropy_with_cache:")
        print(f"  X_pred_eval range: [{X_pred_eval_min:.4f}, {X_pred_eval_max:.4f}]")
        print(f"  Extension factor c={c}, original_pinn_x_max={original_pinn_x_max_for_grid:.4f}")
        print(f"  Minimum x_max for extended horizon: {min_x_max_for_extended:.4f} (= original_pinn_x_max * {c})")
        print(f"  adjusted_pinn_x_max: {current_pinn_x_max:.4f} -> {adjusted_pinn_x_max:.4f}")
        print(f"  adjusted_pinn_x_min: {current_pinn_x_min:.4f} -> {adjusted_pinn_x_min:.4f}")
        print(f"  adjusted_pinn_n_x: {original_pinn_n_x} -> {adjusted_pinn_n_x} (to maintain resolution)")
        print(f"  adjusted_pinn_n_iter: {original_pinn_n_iter} -> {adjusted_pinn_n_iter} (for extended range accuracy)")
        # Calculate grid spacing to check if it's fine enough
        grid_spacing = (adjusted_pinn_x_max - adjusted_pinn_x_min) / adjusted_pinn_n_x if adjusted_pinn_n_x > 0 else float('inf')
        print(f"  Grid spacing: {grid_spacing:.6f} (smaller is better for accuracy)")
        if grid_spacing > 0.1:
            print(f"  WARNING: Grid spacing is large ({grid_spacing:.6f}), PINN interpolation may be inaccurate!")
            print(f"  Consider increasing pinn_n_x or using a finer grid for extended range.")
        print(f"[plot_extendability_test] Training separate PINN for extended range [T_train={T_train:.2f}, T_eval={T_eval:.2f}]")
        print(f"  This ensures accurate policy evaluation for states outside original training range [0, T_train]")

        try:
            u_mean_pinn, _, optimal_value_test_eval, policy_discrete = model.compute_exact_entropy_with_cache(
                X_pred_eval, return_policy=True
            )
        finally:
            # Restore pinn_x_max, pinn_x_min, pinn_n_x, and pinn_n_iter (may have been scaled for extended horizon)
            model.params_train["pinn_x_max"] = current_pinn_x_max
            model.params_train["pinn_x_min"] = current_pinn_x_min
            model.params_train["pinn_n_x"] = original_pinn_n_x
            model.params_train["pinn_n_iter"] = original_pinn_n_iter
        # policy_discrete is a tuple (support_u, probs) where:
        #   support_u: [n_u] - grid points for u
        #   probs: [B, n_u] - probabilities for each u at each state
        #   B = M * (N_eval+1) (flattened batch)
        if isinstance(policy_discrete, tuple):
            # If it's a tuple, create DiscretePolicy object or compute loc manually
            support_u, probs = policy_discrete
            # probs: [B, n_u], support_u: [n_u]
            # loc = sum(probs * support_u) over u dimension: [B, 1]
            control_test_mean_eval_flat = (probs * support_u.view(1, -1)).sum(dim=1, keepdim=True)
            # Reshape to match X_pred_eval shape [M, N_eval+1, 1]
            control_test_mean_eval = control_test_mean_eval_flat.reshape(M, N_plus_1, -1)
        elif hasattr(policy_discrete, 'loc'):
            # If it's a DiscretePolicy object
            control_test_mean_eval = policy_discrete.loc
            # Reshape to match X_pred_eval shape [M, N_eval+1, 1]
            if control_test_mean_eval.shape != X_pred_eval.shape:
                control_test_mean_eval = control_test_mean_eval.reshape(M, N_plus_1, -1)
        else:
            # Fallback: use u_mean_pinn (which should be the same as DiscretePolicy.loc)
            control_test_mean_eval = u_mean_pinn
            if control_test_mean_eval.shape != X_pred_eval.shape:
                control_test_mean_eval = control_test_mean_eval.reshape(M, N_plus_1, -1)

        u_pred_label = "mean of u_pred"
        u_test_label = "mean of u_test"
        u_ylabel = "Mean of $\\pi(t, \\hat{X}_t)$"

    Y_test_eval = optimal_value_test_eval

    # Diagnostic: Check trajectory divergence at T_train and T_eval
    idx_T_train = int(T_train / dt_eval)
    idx_T_eval = X_pred_eval.shape[1] - 1
    if idx_T_train < X_pred_eval.shape[1]:
        X_pred_at_T = X_pred_eval[:, idx_T_train, :].mean().item()
        X_test_at_T = X_test_eval[:, idx_T_train, :].mean().item()
        X_pred_at_cT = X_pred_eval[:, idx_T_eval, :].mean().item()
        X_test_at_cT = X_test_eval[:, idx_T_eval, :].mean().item()
        print(f"[plot_extendability_test] Trajectory divergence:")
        print(f"  At T={T_train:.2f}: X_pred_mean={X_pred_at_T:.4f}, X_test_mean={X_test_at_T:.4f}, diff={abs(X_pred_at_T-X_test_at_T):.4f}")
        print(f"  At {c}*T={T_eval:.2f}: X_pred_mean={X_pred_at_cT:.4f}, X_test_mean={X_test_at_cT:.4f}, diff={abs(X_pred_at_cT-X_test_at_cT):.4f}")

    # Convert to numpy
    # Use model.t_predict_set if available (which predict actually used), otherwise fallback to t_eval
    if hasattr(model, 't_predict_set') and model.t_predict_set is not None:
        t_eval_np = model.t_predict_set.detach().cpu().numpy()
        print(f"[plot_extendability_test] Using model.t_predict_set for time axis: shape={t_eval_np.shape}")
    else:
        t_eval_np = t_eval.detach().cpu().numpy()
        print(f"[plot_extendability_test] Using t_eval for time axis: shape={t_eval_np.shape}")

    X_pred_eval_np = X_pred_eval.detach().cpu().numpy()
    X_test_eval_np = X_test_eval.detach().cpu().numpy()
    Y_pred_eval_np = Y_pred_eval.detach().cpu().numpy()
    Y_test_eval_np = Y_test_eval.detach().cpu().numpy()
    # Use mean of policy (mixture_mean and DiscretePolicy.loc) instead of sampled controls
    control_pred_eval_np = control_pred_mean_eval.detach().cpu().numpy()
    control_test_eval_np = control_test_mean_eval.detach().cpu().numpy()

    # Diagnostic: Check mean of policy at T_train
    # Calculate dt_eval from T_eval and number of time steps
    dt_eval_calc = T_eval / (X_pred_eval.shape[1] - 1) if X_pred_eval.shape[1] > 1 else T_eval
    idx_T_train = int(T_train / dt_eval_calc)
    if idx_T_train < control_pred_eval_np.shape[1]:
        u_pred_at_T = control_pred_eval_np[:, idx_T_train, :].mean()
        u_test_at_T = control_test_eval_np[:, idx_T_train, :].mean()
        print(f"[plot_extendability_test] Mean of policy at T={T_train:.2f} (idx={idx_T_train}):")
        print(f"  u_pred_mean = {u_pred_at_T:.6f}, u_test_mean = {u_test_at_T:.6f}, diff = {abs(u_pred_at_T - u_test_at_T):.6f}")
        print(f"  relative_error = {abs(u_pred_at_T - u_test_at_T) / (abs(u_test_at_T) + 1e-8) * 100:.4f}%")

    # Helper functions
    def _to_1d_time(x, ds_idx=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x_cpu = x.detach().cpu()

        if x_cpu.ndim == 3:
            x_cpu = x_cpu[0]
        if x_cpu.ndim == 2:
            if x_cpu.shape[-1] == 1:
                x_cpu = x_cpu[:, 0]
            elif x_cpu.shape[0] == 1:
                x_cpu = x_cpu[0]

        if ds_idx is None:
            return x_cpu.numpy()

        ds = ds_idx
        if torch.is_tensor(ds):
            ds = ds.detach().cpu().long()
        else:
            ds = torch.tensor(ds, dtype=torch.long)

        return x_cpu[ds].numpy()

    def _smooth_1d(x, k):
        if k <= 1:
            return x
        pad = k // 2
        x_pad = np.concatenate([np.repeat(x[0], pad), x, np.repeat(x[-1], pad)])
        kernel = np.ones(k) / k
        y = np.convolve(x_pad, kernel, mode='valid')
        return y

    # For "mean of policy", compute time-step-wise mean over all trajectories
    # This is consistent with how mean of policy is computed in the combined plot
    print(f"[plot_extendability_test] Computing time-step-wise mean over all {model.M} trajectories")

    # Compute time-step-wise mean: average over trajectory dimension (axis=0)
    # control_pred_eval_np: [M, N_eval+1, 1] -> [N_eval+1, 1]
    control_pred_mean_1d = np.mean(control_pred_eval_np, axis=0)  # [N_eval+1, 1]
    control_test_mean_1d = np.mean(control_test_eval_np, axis=0)  # [N_eval+1, 1]
    Y_pred_mean_1d = np.mean(Y_pred_eval_np, axis=0)  # [N_eval+1, 1]
    Y_test_mean_1d = np.mean(Y_test_eval_np, axis=0)  # [N_eval+1, 1]
    X_pred_mean_1d = np.mean(X_pred_eval_np, axis=0)  # [N_eval+1, 1]
    X_test_mean_1d = np.mean(X_test_eval_np, axis=0)  # [N_eval+1, 1]

    # For time axis, use mean time (should be same for all trajectories)
    t_mean_1d = np.mean(t_eval_np, axis=0)  # [N_eval+1, 1]

    # Downsample indices
    ds_idx = torch.arange(0, t_eval.shape[1], downsample_step)

    # Convert to 1D for plotting (squeeze last dimension if needed)
    def _squeeze_last(x):
        if x.ndim > 1 and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x

    t_mean_1d = _squeeze_last(t_mean_1d)
    control_pred_mean_1d = _squeeze_last(control_pred_mean_1d)
    control_test_mean_1d = _squeeze_last(control_test_mean_1d)
    Y_pred_mean_1d = _squeeze_last(Y_pred_mean_1d)
    Y_test_mean_1d = _squeeze_last(Y_test_mean_1d)
    X_pred_mean_1d = _squeeze_last(X_pred_mean_1d)
    X_test_mean_1d = _squeeze_last(X_test_mean_1d)

    # Downsample and smooth
    # _to_1d_time expects numpy array or torch tensor, and handles downsampling
    # Since we already have 1D arrays, we can directly index them
    if ds_idx is not None:
        ds_idx_np = ds_idx.cpu().numpy() if torch.is_tensor(ds_idx) else np.array(ds_idx)
        t_plot = t_mean_1d[ds_idx_np]
        u_pred_plot = _smooth_1d(control_pred_mean_1d[ds_idx_np], smooth_window)
        u_test_plot = _smooth_1d(control_test_mean_1d[ds_idx_np], smooth_window)
        v_pred_plot = _smooth_1d(Y_pred_mean_1d[ds_idx_np], smooth_window)
        v_test_plot = _smooth_1d(Y_test_mean_1d[ds_idx_np], smooth_window)
        x_pred_plot = _smooth_1d(X_pred_mean_1d[ds_idx_np], smooth_window)
        x_test_plot = _smooth_1d(X_test_mean_1d[ds_idx_np], smooth_window)
    else:
        t_plot = t_mean_1d
        u_pred_plot = _smooth_1d(control_pred_mean_1d, smooth_window)
        u_test_plot = _smooth_1d(control_test_mean_1d, smooth_window)
        v_pred_plot = _smooth_1d(Y_pred_mean_1d, smooth_window)
        v_test_plot = _smooth_1d(Y_test_mean_1d, smooth_window)
        x_pred_plot = _smooth_1d(X_pred_mean_1d, smooth_window)
        x_test_plot = _smooth_1d(X_test_mean_1d, smooth_window)

    # Plot helpers
    zoom_color = 'lightblue'
    zoom_alpha = 0.35

    def _make_inset(ax, width="35%", height="40%", x0=0.14, y0=0.58):
        """Create inset axes"""
        iax = inset_axes(
            ax,
            width=width, height=height,
            bbox_to_anchor=(x0, y0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            loc="lower left",
            borderpad=0.0,
        )
        iax.set_zorder(20)
        iax.patch.set_alpha(0.95)
        return iax

    def _connect_main_to_inset(ax, inset_ax, xyA_data):
        """Connect main plot to inset"""
        con = ConnectionPatch(
            xyA=xyA_data, coordsA=ax.transData,
            xyB=(1.0, 0.5), coordsB=inset_ax.transAxes,
            axesA=ax, axesB=inset_ax,
            color=zoom_color, linewidth=2.0, alpha=0.6,
            linestyle="--",
            connectionstyle="arc3,rad=0.0",
            zorder=9
        )
        ax.add_artist(con)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5), dpi=300)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)

    # Subplot 1: Control u(t, X_t)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axvspan(0.0, float(T_train), color=zoom_color, alpha=zoom_alpha, zorder=0)
    ax1.plot(t_plot, u_pred_plot, '-', color='darkorange', linewidth=2.5, label=u_pred_label, zorder=2)
    ax1.plot(t_plot, u_test_plot, '-', color='darkviolet', linewidth=2.5, label=u_test_label, zorder=3)
    ax1.axvline(x=T_train, color='blue', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    ax1.set_xlabel('$t$', fontsize=axis_label_size)
    ax1.set_ylabel(u_ylabel, fontsize=axis_label_size)
    ax1.set_ylim([0.35, 0.65])  # Set y-axis range for mean of policy
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax1.legend(fontsize=legend_font_size, loc='best')

    # Inset for subplot 1
    ax1_inset = _make_inset(ax1, width="35%", height="40%", x0=0.14, y0=0.58)
    mask_T = t_plot <= T_train
    if np.any(mask_T):
        t_inset = t_plot[mask_T]
        u_pred_inset = u_pred_plot[mask_T]
        u_test_inset = u_test_plot[mask_T]
        ax1_inset.plot(t_inset, u_pred_inset, '-', color='darkorange', linewidth=2, zorder=2)
        ax1_inset.plot(t_inset, u_test_inset, '-', color='darkviolet', linewidth=2, zorder=3)
        ax1_inset.set_xlim(0, T_train)
        ax1_inset.set_ylim(0.4, 0.6)
    ax1_inset.tick_params(labelsize=16)
    ax1_inset.grid(True, alpha=0.3)

    # Subplot 2: Value v(t, X_t)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axvspan(0.0, float(T_train), color=zoom_color, alpha=zoom_alpha, zorder=0)
    ax2.plot(t_plot, v_pred_plot, '-', color='darkorange', linewidth=2.5, label='V_pred', zorder=2)
    ax2.plot(t_plot, v_test_plot, '-', color='darkviolet', linewidth=2.5, label='V_test', zorder=3)
    ax2.axvline(x=T_train, color='blue', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    ax2.set_xlabel('$t$', fontsize=axis_label_size)
    ax2.set_ylabel('$v(t,X_t)$', fontsize=axis_label_size)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax2.legend(fontsize=legend_font_size, loc='best')

    # Inset for subplot 2
    ax2_inset = _make_inset(ax2, width="35%", height="40%", x0=0.16, y0=0.58)
    if np.any(mask_T):
        v_pred_inset = v_pred_plot[mask_T]
        v_test_inset = v_test_plot[mask_T]
        ax2_inset.plot(t_inset, v_pred_inset, '-', color='darkorange', linewidth=2, zorder=2)
        ax2_inset.plot(t_inset, v_test_inset, '-', color='darkviolet', linewidth=2, zorder=3)
        ax2_inset.set_xlim(0, T_train)
        v_min, v_max = min(np.min(v_pred_inset), np.min(v_test_inset)), max(np.max(v_pred_inset), np.max(v_test_inset))
        v_range = v_max - v_min
        if v_range > 0:
            ax2_inset.set_ylim(v_min - 0.1 * v_range, v_max + 0.1 * v_range)
    ax2_inset.tick_params(labelsize=16)
    ax2_inset.grid(True, alpha=0.3)

    # Subplot 3: Distribution comparison at c*T (T_eval)
    ax3 = fig.add_subplot(gs[0, 2])

    # Get final states at T_eval (c*T) for all trajectories
    # X_pred_eval: flow net sampled states [M, N_eval+1, 1]
    # X_test_eval: exact policy sampled states [M, N_eval+1, 1]
    X_pred_T_eval_np = X_pred_eval_np[:, -1, :].flatten()  # Flow net final states at c*T
    X_test_T_eval_np = X_test_eval_np[:, -1, :].flatten()  # Exact policy final states at c*T

    # Filter out invalid values
    valid_flow = np.isfinite(X_pred_T_eval_np) & (X_pred_T_eval_np > 0)
    valid_exact = np.isfinite(X_test_T_eval_np) & (X_test_T_eval_np > 0)
    X_pred_T_eval_np = X_pred_T_eval_np[valid_flow]
    X_test_T_eval_np = X_test_T_eval_np[valid_exact]

    if len(X_pred_T_eval_np) > 0 and len(X_test_T_eval_np) > 0:
        # Determine x-axis range
        x_min = 0.0
        x_max = max(float(np.max(X_pred_T_eval_np)), float(np.max(X_test_T_eval_np))) * 1.1

        # Use KDE for smooth curves
        try:
            kde_flow = gaussian_kde(X_pred_T_eval_np)
            kde_exact = gaussian_kde(X_test_T_eval_np)
            x_plot = np.linspace(x_min, x_max, 200)
            ax3.plot(x_plot, kde_flow(x_plot), label='Flownet', linewidth=2.5, alpha=0.7, color='darkorange', zorder=2)
            ax3.plot(x_plot, kde_exact(x_plot), label='Exact policy', linewidth=2.5, alpha=0.7, color='darkviolet', zorder=3)
        except Exception as e:
            print(f"[plot_extendability_test] KDE failed: {e}, falling back to histogram")
            # Fallback to histogram
            ax3.hist(X_pred_T_eval_np, bins=150, alpha=0.5, label='Flownet',
                    density=True, color='darkorange', edgecolor='black', linewidth=0.5, zorder=2)
            ax3.hist(X_test_T_eval_np, bins=150, alpha=0.5, label='Exact policy',
                    density=True, color='darkviolet', edgecolor='black', linewidth=0.5, zorder=3)

        ax3.set_xlabel(f'$X_{{{int(c)}T}}$', fontsize=axis_label_size)
        ax3.set_ylabel('Density of state', fontsize=axis_label_size)
        ax3.set_xlim(x_min, x_max)
        ax3.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax3.legend(fontsize=legend_font_size, loc='best', framealpha=0.5, edgecolor='gray')
        ax3.grid(True, alpha=0.3)
        print(f"[plot_extendability_test] ✓ Distribution comparison plotted at {c}*T (n_flow={len(X_pred_T_eval_np)}, n_exact={len(X_test_T_eval_np)})")
    else:
        ax3.text(0.5, 0.5, 'No valid data for distribution', ha='center', va='center',
                transform=ax3.transAxes, fontsize=axis_label_size)
        print(f"[plot_extendability_test] WARNING: No valid data for distribution comparison")

    # Connect main plots to insets (only for subplot 1 and 2 now)
    anchor_t = float(T_train) * 0.95
    idx = int(np.argmin(np.abs(t_plot - anchor_t)))
    _connect_main_to_inset(ax1, ax1_inset, (t_plot[idx], u_pred_plot[idx]))
    _connect_main_to_inset(ax2, ax2_inset, (t_plot[idx], v_pred_plot[idx]))

    plt.tight_layout(w_pad=2.5)
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f'{filename}_itr{itr}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plot_extendability_test] Plot saved to {save_path}")



def plot_training_series(list_, ylabel, legend, *, output_folder, iterations=None, color=None,
                         title=None, ylabel_log=False, xticks=None, baseline=None, filename=None,
                         log_limits=False, ylim=None, yticks=None, iteration_values=None):
    """Plot LQ/CARA histories, reducing stored trajectory dimensions to one series per metric."""
    plt.figure(figsize=(6.0, 4.0), dpi=300)

    series = []
    for i in range(len(list_)):
        data = list_[i]
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        elif isinstance(data, (list, tuple)) and len(data) > 0 and torch.is_tensor(data[0]):
            data = np.stack([d.detach().cpu().numpy() for d in data])
        elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], np.ndarray):
            data = np.stack(data)
        else:
            data = np.asarray(data)


        if data.ndim == 3:           # [N_iter, M, N]
            data = data[:, 0, :].mean(axis=-1)
        elif data.ndim == 2:         # [N_iter, M]
            data = data[:, 0]
        elif data.ndim == 0:
            data = data.reshape(1)

        series.append(np.asarray(data).astype(float))

        x = np.arange(len(series[-1])) if iteration_values is None else np.asarray(iteration_values)
        if x.shape != series[-1].shape:
            plt.close()
            raise ValueError("Iteration coordinates must match the plotted history")
        valid = np.isfinite(x) & np.isfinite(series[-1])
        if ylabel_log:
            valid &= series[-1] > 0
        if valid.any():
            # Evaluation records can be separated by NaNs at training-only
            # iterations. Connect the recorded points at their actual indices.
            plt.plot(x[valid], series[-1][valid], label=legend[i],
                     color=None if color is None else color[i % len(color)],
                     marker="o" if valid.sum() <= 3 or not valid.all() else None)

    if not plt.gca().lines:
        plt.close()
        raise ValueError("No finite history values are available to plot")

    if baseline is not None:
        if torch.is_tensor(baseline):
            baseline = baseline.detach().cpu().numpy().item()
        iters = iterations if iterations is not None else max((len(s) for s in series), default=0)
        base_arr = np.full(iters, float(baseline))
        base_color = (color[-1] if color is not None else None)
        plt.plot(base_arr, '--', color=base_color, label='baseline' if legend is None else None)

    if len(list_) > 1 or any(len(s) > 1 for s in series):
        plt.legend()
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])
    if title is not None:
        plt.title(title)



    if ylabel_log:
        plt.gca().set_yscale("log")
        nonempty = [s[np.isfinite(s)] for s in series if s.size > 0]
        all_vals = np.concatenate(nonempty) if nonempty else np.array([])
        all_vals = all_vals[all_vals > 0]
        if log_limits and all_vals.size > 0:
            ymin = max(all_vals.min() * 0.8, 1e-8)
            ymax = all_vals.max() * 1.25
            plt.ylim(ymin, ymax)
    if ylim is not None and not ylabel_log:
        plt.ylim(*ylim)
        if yticks is not None:
            plt.yticks(yticks)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(os.path.join(output_folder, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_multi_agent_errors_and_loss(Y_errors, ctrl_err_pad, time_points, CriticLoss, ActorLoss,
                                     *, dimensions, output_folder, filename_suffix=""):
    """Plot control error, value error and loss in a 1x3 figure,
    using the trajectory figure's font and legend sizes."""

    axis_label_size = 24
    tick_label_size = 20
    legend_font_size = 15

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5), dpi=300)
    D = dimensions


    ax1.plot(time_points, ctrl_err_pad[:, 0], color='#6565AE', label="agent 1", linewidth=2)
    if 2 < D < 10:
        for k in range(1, D):
            c = '#E7483D'
            lbl = f"agent 2~{D}" if k == 1 else None
            ax1.plot(time_points, ctrl_err_pad[:, k], color=c, label=lbl, linewidth=2)
    elif D >= 10:
        for k in range(1, 10):
            c = '#E7483D'
            lbl = "agent 2~10" if k == 1 else None
            ax1.plot(time_points, ctrl_err_pad[:, k], color=c, label=lbl, linewidth=2)
        if D > 10:
            for k in range(10, D):
                c = '#FAC074'
                lbl = f"agent 11~{D}" if k == 10 else None
                ax1.plot(time_points, ctrl_err_pad[:, k], color=c, label=lbl, linewidth=2)
    else:
        for k in range(1, D):
            c = '#E7483D'
            lbl = f"agent {k+1}"
            ax1.plot(time_points, ctrl_err_pad[:, k], color=c, label=lbl, linewidth=2)

    ax1.set_xlabel('Time Point Index', fontsize=axis_label_size)
    ax1.set_ylabel(r'$L^2$ relative error of $u_t$', fontsize=axis_label_size)
    ax1.legend(fontsize=legend_font_size, framealpha=0.9, loc='best')
    ax1.tick_params(labelsize=tick_label_size)
    ax1.grid(True, alpha=0.3)


    ax2.plot(time_points, Y_errors[:, 0], color='#6565AE', label="agent 1", linewidth=2)
    if 2 < D < 10:
        for k in range(1, D):
            c = '#E7483D'
            lbl = f"agent 2~{D}" if k == 1 else None
            ax2.plot(time_points, Y_errors[:, k], color=c, label=lbl, linewidth=2)
    elif D >= 10:
        for k in range(1, 10):
            c = '#E7483D'
            lbl = "agent 2~10" if k == 1 else None
            ax2.plot(time_points, Y_errors[:, k], color=c, label=lbl, linewidth=2)
        if D > 10:
            for k in range(10, D):
                c = '#FAC074'
                lbl = f"agent 11~{D}" if k == 10 else None
                ax2.plot(time_points, Y_errors[:, k], color=c, label=lbl, linewidth=2)
    else:
        for k in range(1, D):
            c = '#E7483D'
            lbl = f"agent {k+1}"
            ax2.plot(time_points, Y_errors[:, k], color=c, label=lbl, linewidth=2)

    ax2.set_xlabel('Time Point Index', fontsize=axis_label_size)
    ax2.set_ylabel(r'$L^2$ relative error of $V_t$', fontsize=axis_label_size)
    ax2.legend(fontsize=legend_font_size, framealpha=0.9, loc='best')
    ax2.tick_params(labelsize=tick_label_size)
    ax2.grid(True, alpha=0.3)


    CriticLoss_np = _as_1d_numpy(CriticLoss)
    ActorLoss_np = _as_1d_numpy(ActorLoss)

    has_critic = CriticLoss_np is not None and len(CriticLoss_np) > 0
    has_actor = ActorLoss_np is not None and len(ActorLoss_np) > 0

    if has_critic or has_actor:
        ax3.set_xlabel('iteration', fontsize=axis_label_size)
        ax3.set_yscale("log")

        if has_actor:
            ax3_twin = ax3.twinx()
            ax3_twin.set_ylabel('ActorLoss', color='#6565AE', fontsize=axis_label_size)
            ax3_twin.plot(ActorLoss_np, color='#6565AE', label='ActorLoss', zorder=1, linewidth=2)
            ax3_twin.tick_params(axis='y', labelcolor='#6565AE', labelsize=tick_label_size)
            y_max = float(np.max(ActorLoss_np))
            y_min = float(np.min(ActorLoss_np)) * 0.8
            ax3_twin.set_ylim(y_min, y_max)
            medium = (y_max + y_min) / 2
            ax3_twin.set_yticks([y_min, medium, y_max])
        else:
            ax3_twin = None

        if has_critic:
            ax3.set_ylabel('CriticLoss', color='#E7483D', fontsize=axis_label_size)
            ax3.plot(CriticLoss_np, color='#E7483D', label='CriticLoss', zorder=2, linewidth=2)
            ax3.tick_params(axis='y', labelcolor='#E7483D', labelsize=tick_label_size)

        lines, labels = ax3.get_legend_handles_labels()
        if ax3_twin is not None:
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines + lines2, labels + labels2, loc='best',
                      framealpha=0.9, fontsize=legend_font_size)
        else:
            ax3.legend(lines, labels, loc='best', framealpha=0.9, fontsize=legend_font_size)

        ax3.tick_params(labelsize=tick_label_size)
        ax3.grid(True, alpha=0.3)



    if has_critic:
        for label in ax3.get_yticklabels():
            label.set_color('#E7483D')

    plt.tight_layout()



    if has_critic:

        plt.draw()

        for label in ax3.get_yticklabels():
            label.set_color('#E7483D')

        ax3.yaxis.label.set_color('#E7483D')

        ax3.tick_params(axis='y', which='major', labelcolor='#E7483D', color='#E7483D')
        ax3.tick_params(axis='y', which='minor', labelcolor='#E7483D', color='#E7483D')

    path_pdf = os.path.join(output_folder, f"game_errors_and_loss_combined{filename_suffix}.pdf")

    if has_critic:
        for label in ax3.get_yticklabels(which='both'):
            label.set_color('#E7483D')
    plt.savefig(path_pdf, dpi=300, bbox_inches='tight')
    print(f"[info] Saved combined errors and loss plot to: {path_pdf}")

    plt.close(fig)


def plot_actor_critic_loss(CriticLoss, ActorLoss, filename, *, output_folder, show=False):
    """Plot supplied critic and actor histories using the existing two-axis convention."""
    CriticLoss = _as_1d_numpy(CriticLoss)
    ActorLoss = _as_1d_numpy(ActorLoss)
    fig, ax1 = plt.subplots(dpi=300)
    ax1.set_xlabel('iteration')
    plt.gca().set_yscale("log")


    has_critic = CriticLoss is not None and len(CriticLoss) > 0
    has_actor = ActorLoss is not None and len(ActorLoss) > 0

    if not has_critic and not has_actor:
        print("[warn] Both CriticLoss and ActorLoss are empty, skipping plot_loss")
        plt.close(fig)
        return

    if has_actor:
        ax2 = ax1.twinx()
        ax2.set_ylabel('ActorLoss', color='#6565AE')
        ax2.plot(ActorLoss, color='#6565AE', label='ActorLoss', zorder=1)
        ax2.tick_params(axis='y', labelcolor='#6565AE')
        y_max = float(np.max(ActorLoss))
        y_min = float(np.min(ActorLoss)) * 0.8
        ax2.set_ylim(y_min, y_max)
        medium = (y_max + y_min) / 2
        ax2.set_yticks([y_min, medium, y_max])
    else:
        ax2 = None
        print("[warn] ActorLoss is empty, skipping ActorLoss plot")

    if has_critic:
        ax1.plot(CriticLoss, color='#E7483D', label='CriticLoss', zorder=2)
        ax1.tick_params(axis='y', labelcolor='#E7483D')

        ax1.set_ylabel('CriticLoss', color='#E7483D')
        for label in ax1.get_yticklabels():
            label.set_color('#E7483D')
    else:
        print("[warn] CriticLoss is empty, skipping CriticLoss plot")


    lines, labels = ax1.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best', framealpha=0.5, edgecolor='gray')
    else:
        ax1.legend(lines, labels, loc='best', framealpha=0.5, edgecolor='gray')

    if filename is not None:
        plt.savefig(os.path.join(output_folder, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_gaussian_mean_std(gaussian_mean_list, gaussian_std_list, *, output_folder,
                           filename="gaussian_mean_std"):
    """Plot the first player's trajectory-averaged Gaussian mean and standard
    deviation against training iterations in a 1x2 figure."""
    if gaussian_mean_list is None or gaussian_std_list is None:
        print("[warn] gaussian_mean_list or gaussian_std_list not found, skipping plot")
        return

    if len(gaussian_mean_list) == 0 or len(gaussian_std_list) == 0:
        print("[warn] gaussian_mean_list or gaussian_std_list is empty, skipping plot")
        return


    mean_array = np.array([float(x) for x in gaussian_mean_list])
    std_array = np.array([float(x) for x in gaussian_std_list])
    iterations = np.arange(len(mean_array))


    axis_label_size = 24
    tick_label_size = 20
    legend_font_size = 15


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)


    ax1.plot(iterations, mean_array, color='#6565AE', linewidth=2, label='Player 0 Mean')
    ax1.set_xlabel('Iteration', fontsize=axis_label_size)
    ax1.set_ylabel('Gaussian Mean (averaged over trajectories)', fontsize=axis_label_size)
    ax1.legend(fontsize=legend_font_size, framealpha=0.9, loc='best')
    ax1.tick_params(labelsize=tick_label_size)
    ax1.grid(True, alpha=0.3)


    ax2.plot(iterations, std_array, color='#E7483D', linewidth=2, label='Player 0 Std')
    ax2.set_xlabel('Iteration', fontsize=axis_label_size)
    ax2.set_ylabel('Gaussian Std (averaged over trajectories)', fontsize=axis_label_size)
    ax2.legend(fontsize=legend_font_size, framealpha=0.9, loc='best')
    ax2.tick_params(labelsize=tick_label_size)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    path_pdf = os.path.join(output_folder, f'{filename}.pdf')
    plt.savefig(path_pdf, dpi=300, bbox_inches='tight')
    print(f"[info] Saved gaussian mean/std plot to: {path_pdf}")

    plt.close(fig)


def plot_control_error_heatmap(rel_err, *, player_idx=None, component_idx=None, x_region, horizon, output_folder,
                               filename=None):
    """Render a precomputed control error slice; the model supplies its analytical benchmark."""
    plt.figure(figsize=(5, 4), dpi=300)
    f1 = plt.imshow(
        rel_err,
        interpolation='nearest',
        cmap='rainbow',
        extent=[x_region[0], x_region[1], 0, float(horizon)],
        origin='lower',
        aspect='auto'
    )
    if player_idx is None:
        plt.colorbar(f1)
    else:
        plt.colorbar(f1, label=r"$|(\hat{u}_i - u_i^*)/u_i^*|$")
    if component_idx is not None:
        plt.xlabel(fr"$x_{{{component_idx+1}}}$")
    else:
        plt.xlabel(r"$x$" if player_idx is None else r"$x_i$")
    plt.ylabel(r"$t$")
    plt.title(r"$|(\hat{u}(t,x)-u^*(t,x))/u^*(t,x)|$" if player_idx is None
              else fr"Player {player_idx}: relative error of control")
    if component_idx is not None:
        plt.title(fr"State/control component {component_idx+1}: relative error")
    plt.tight_layout()
    if filename is not None:
        path = os.path.join(output_folder, f"{filename}.pdf" if player_idx is None else f"{filename}_player{player_idx}.pdf")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[plot] saved {path}")
    plt.show()


def plot_multi_agent_diagnostics(V_pred, V_test, control_pred, control_test, t_test,
                                 *, dimensions, output_folder, filename_suffix="",
                                 CriticLoss=None, ActorLoss=None):
    """Plot relative trajectory errors and optional loss histories from supplied NumPy arrays."""
    D = dimensions
    Y_pred_np = V_pred
    Y_test_np = V_test
    ctrl_pred_np = control_pred
    ctrl_test_np = control_test
    t_np = t_test

    # value error: (N+1,D)
    Y_errors = np.sqrt(
        np.mean((Y_test_np - Y_pred_np)**2, axis=0) /
        (np.mean(Y_test_np**2, axis=0) + 1e-12)
    )

    plt.figure(dpi=300)
    # Use time point indices instead of actual time values
    time_points = np.arange(len(t_np[0, :, 0]))
    plt.plot(time_points, Y_errors[:, 0], color='#6565AE', label="agent 1")

    if 2 < D < 10:

        for k in range(1, D):
            c = '#E7483D'
            lbl = f"agent 2~{D}" if k == 1 else None
            plt.plot(time_points, Y_errors[:, k], color=c, label=lbl)
    elif D >= 10:

        for k in range(1, 10):
            c = '#E7483D'
            lbl = "agent 2~10" if k == 1 else None
            plt.plot(time_points, Y_errors[:, k], color=c, label=lbl)
        if D > 10:

            for k in range(10, D):
                c = '#FAC074'
                lbl = f"agent 11~{D}" if k == 10 else None
                plt.plot(time_points, Y_errors[:, k], color=c, label=lbl)
    else:

        for k in range(1, D):
            c = '#E7483D'
            lbl = f"agent {k+1}" if k < D - 1 else f"agent {k+1}"
            plt.plot(time_points, Y_errors[:, k], color=c, label=lbl)
    plt.xlabel('Time Point Index')
    plt.ylabel(r'$L^2$ relative error of $V_t$')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_folder, f"game_error_value{filename_suffix}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

    # control error: (N,D)
    control_errors = np.sqrt(
        np.mean((ctrl_test_np - ctrl_pred_np)**2, axis=0) /
        (np.mean(ctrl_test_np**2, axis=0) + 1e-12)
    )
    ctrl_err_pad = np.concatenate(
        [control_errors, control_errors[-1:, :]],
        axis=0
    )

    plt.figure(dpi=300)
    plt.plot(time_points, ctrl_err_pad[:, 0], color='#6565AE', label="agent 1")

    if 2 < D < 10:

        for k in range(1, D):
            c = '#E7483D'
            lbl = f"agent 2~{D}" if k == 1 else None
            plt.plot(time_points, ctrl_err_pad[:, k], color=c, label=lbl)
    elif D >= 10:

        for k in range(1, 10):
            c = '#E7483D'
            lbl = "agent 2~10" if k == 1 else None
            plt.plot(time_points, ctrl_err_pad[:, k], color=c, label=lbl)
        if D > 10:

            for k in range(10, D):
                c = '#FAC074'
                lbl = f"agent 11~{D}" if k == 10 else None
                plt.plot(time_points, ctrl_err_pad[:, k], color=c, label=lbl)
    else:

        for k in range(1, D):
            c = '#E7483D'
            lbl = f"agent {k+1}" if k < D - 1 else f"agent {k+1}"
            plt.plot(time_points, ctrl_err_pad[:, k], color=c, label=lbl)
    plt.xlabel('Time Point Index')
    plt.ylabel(r'$L^2$ relative error of $u_t$')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_folder, f"game_error_control{filename_suffix}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


    if CriticLoss is not None and ActorLoss is not None:
        if len(CriticLoss) > 0 and len(ActorLoss) > 0:
            plot_multi_agent_errors_and_loss(
                Y_errors=Y_errors,
                ctrl_err_pad=ctrl_err_pad,
                time_points=time_points,
                CriticLoss=CriticLoss,
                ActorLoss=ActorLoss,
                filename_suffix=filename_suffix, dimensions=D, output_folder=output_folder,
            )
        else:
            print("[info] CriticLoss_list or ActorLoss_list is empty, skipping combined plot")
    else:
        print("[info] No loss data available, skipping combined plot")


def plot_policy_diagnostics(kl_loss, std_loss, mu_gaussian_loss, *, output_folder,
                            filename=None, std_reference=None):
    """Plot supplied KL, variance and mean diagnostics with an optional analytical reference."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), dpi=300)
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, (list, tuple)):

            if len(x) > 0 and isinstance(x[0], torch.Tensor):
                return np.array([t.detach().cpu().item() if t.numel() == 1 else t.detach().cpu().numpy() for t in x])
            else:
                return np.array(x)
        else:
            return np.array(x)

    kl_loss = to_numpy(kl_loss)
    std_loss = to_numpy(std_loss)
    mu_gaussian_loss = to_numpy(mu_gaussian_loss)

    if kl_loss.ndim == 2:
        kl_loss = kl_loss.mean(axis=1)
    elif kl_loss.ndim == 3:
        kl_loss = kl_loss.squeeze(-1).mean(axis=1)
    else:
        kl_loss = kl_loss

    if std_loss.ndim == 2:
        std_loss = std_loss.mean(axis=1)
    elif std_loss.ndim == 3:
        std_loss = std_loss.squeeze(-1).mean(axis=1)
    else:
        std_loss = std_loss

    if mu_gaussian_loss.ndim == 2:
        mu_gaussian_loss = mu_gaussian_loss.mean(axis=1)
    elif mu_gaussian_loss.ndim == 3:
        mu_gaussian_loss = mu_gaussian_loss.squeeze(-1).mean(axis=1)
    else:
        mu_gaussian_loss = mu_gaussian_loss

    ax1 = axes[0]
    ax1.plot(kl_loss, color='#E7483D', label='KL Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('KL Loss')
    ax1.set_yscale('log')
    ax1.set_title('KL Loss Convergence')
    ax1.legend(loc='best', framealpha=0.5, edgecolor='gray')

    ax2 = axes[1]
    ax2.plot(std_loss, color='#0072B2', label='STD Loss')

    if len(std_loss) > 0:


        if std_reference is not None:
            ax2.axhline(y=std_reference, color='r', linestyle='--',
                       label=f'Theoretical STD Loss ({std_reference:.6f})',
                       linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('STD Loss')
    ax2.set_yscale('log')
    ax2.set_title('STD Loss Convergence')
    ax2.legend(loc='best', framealpha=0.5, edgecolor='gray')
    plt.tight_layout()

    ax3 = axes[2]
    evaluated = np.flatnonzero(np.isfinite(mu_gaussian_loss))
    ax3.plot(evaluated, mu_gaussian_loss[evaluated], color='#009E73',
             label='Policy Mean Error', marker='o' if evaluated.size == 1 else None)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Policy Mean Error')
    ax3.set_yscale('log')
    ax3.set_title('Evaluated Policy Mean Error')
    ax3.legend(loc='best', framealpha=0.5, edgecolor='gray')
    plt.tight_layout()

    if filename is not None:
        save_path = os.path.join(output_folder, f'{filename}.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to {save_path}")
    plt.show()


def plot_J_control_function(J_list, save_dir, J=True, iteration_values=None):
    if J == True:
        J_array = np.array([j.detach().cpu().numpy() if torch.is_tensor(j) else j for j in J_list])

        if J_array.ndim == 3:
            J_array = J_array.mean(axis=(1, 2))
        elif J_array.ndim == 2:
            J_array = J_array.mean(axis=1)
        elif J_array.ndim == 4:
            J_array = J_array.mean(axis=(1, 2, 3))
        elif J_array.ndim == 1:
            pass
        else:
            raise ValueError(f"Unsupported J_list shape: {J_array.shape}")

        t = np.arange(len(J_array)) if iteration_values is None else np.asarray(iteration_values)
        plt.figure(figsize=(6.0, 4.0), dpi=300)
        plt.plot(t, J_array, label="J(t, x)", color='blue')
        plt.xlabel("Training iteration")
        plt.ylabel("J-function")
        plt.title("Mean predicted value during training")
        plt.grid(True)
        plt.legend()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "J-func.pdf")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved j-function plot to {save_path}")
    else:
        control_array = np.array([c.detach().cpu().numpy() if torch.is_tensor(c) else c for c in J_list])

        if control_array.ndim == 3:
            control_array = control_array.mean(axis=(1, 2))
        elif control_array.ndim == 2:
            control_array = control_array.mean(axis=1)
        elif control_array.ndim == 1:
            pass
        else:
            raise ValueError(f"Unsupported control_list shape: {control_array.shape}")
        t = np.arange(len(control_array)) if iteration_values is None else np.asarray(iteration_values)
        plt.figure(figsize=(6.0, 4.0), dpi=300)
        plt.plot(t, control_array, label="Control(t, x)", color='orange')
        plt.xlabel("Training iteration")
        plt.ylabel("Control Function")
        plt.title("Mean sampled control during training")
        plt.grid(True)
        plt.legend()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "control-func.pdf")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved control function plot to {save_path}")


def plot_value_control_errors(t_test, Y_errors, control_errors, *, output_folder,
                              filename="merton_error"):
    """Plot precomputed value/control relative errors with the existing axis scaling."""
    plt.close('all')
    fig, ax1 = plt.subplots(dpi=300)
    ax1.set_xlabel('$t$')
    # ax1.set_ylabel('$L^2$ relative errors $e^v_t$', color='#E7483D')
    ax1.plot(t_test[0, :, 0], Y_errors, color='#E7483D', label='value function $e_t^v$')
    ax1.tick_params(axis='y', labelcolor='#E7483D')
    ax2 = ax1.twinx()
    # ax2.set_ylabel('$L^2$ relative errors $e^u_t$', color='#6565AE')
    ax2.plot(t_test[0, :, 0], control_errors, color='#6565AE', label='control $e_t^u$')
    ax2.tick_params(axis='y', labelcolor='#6565AE')
    yticks1 = ax1.get_yticks()
    ax1.set_yticks(yticks1)
    ax1.set_yticklabels(['{:,.1f}'.format(y * 10000) for y in yticks1])
    ax1.text(-0.05, 1.075, '$\\times 10^{-4}$', verticalalignment='top', horizontalalignment='left',
             transform=ax1.transAxes, color='#E7483D', fontsize=18)
    yticks2 = ax2.get_yticks()
    ax2.set_yticks(yticks2)
    ax2.set_yticklabels(['{:,.2f}'.format(y * 100) for y in yticks2])
    ax2.text(0.95, 1.075, '$\\times 10^{-2}$', verticalalignment='top', horizontalalignment='left',
             transform=ax1.transAxes, color='#6565AE', fontsize=18)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best', framealpha=0.5, edgecolor='gray')
    plt.savefig(os.path.join(output_folder, f'{filename}.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_reference_series(times, approximate, benchmark, *, filename, ylabel,
                          approximate_label, benchmark_label):
    """Plot an evaluated scalar series against its analytical benchmark."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, approximate, color="darkorange", label=approximate_label)
    ax.plot(times, benchmark, "--", color="darkviolet", label=benchmark_label)
    ax.set(xlabel=r"$t$ (Time)", ylabel=ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_metric_history(records, specifications, *, filename):
    """Plot recorded metrics against their actual evaluation iteration numbers."""
    if not records:
        return
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label, color, style in specifications:
        ax.semilogy([v["iteration"] for v in records], [max(v[key], 1e-12) for v in records],
                    style, color=color, label=label)
    ax.set(xlabel="Iteration", ylabel="Error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
