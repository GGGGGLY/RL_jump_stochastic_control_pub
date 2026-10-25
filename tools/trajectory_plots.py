"""Shared three-panel trajectory figures for LQ and Multi-CARA experiments.

Models compute their own controls/policy means, values and analytical benchmarks.
This module only renders supplied arrays. Merton flow-policy distributions use
experiment_utils.plot_trajectory_combined_abc and merton_policy_plots instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiment_utils import (
    configure_plot_style, plot_multi_agent_diagnostics, plot_reference_series, plot_metric_history,
)


@dataclass
class TrajectoryPanel:
    approximate: object
    benchmark: object
    ylabel: str
    approximate_label: str
    benchmark_label: str
    start_label: str
    end_label: str
    std: object = None


@dataclass
class TrajectoryRow:
    times: object
    panels: tuple[TrajectoryPanel, TrajectoryPanel, TrajectoryPanel]
    jump_steps: object = None  # Boolean [paths, time_intervals], not jump times.
    mask: object = None
    sample_indices: object = None


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _paths(value):
    """Normalize a selected scalar component to [paths, time_points]."""
    array = _numpy(value)
    if array.ndim == 3:
        return array[:, :, 0]
    if array.ndim == 2:
        return array.T if array.shape[1] == 1 else array
    if array.ndim <= 1:
        return array.reshape(1, -1)
    raise ValueError('Trajectory arrays must have at most three dimensions')


def _smooth(values, window):
    window = max(1, int(window))
    if window == 1:
        return values
    padded = np.pad(values, (window//2, (window-1)//2), mode='edge')
    return np.convolve(padded, np.ones(window)/window, mode='valid')


def _label(text, prefix):
    if text is None:
        return None
    if text.startswith(prefix):
        return text
    if prefix == 'Benchmark' and text.startswith('Exact'):
        return text.replace('Exact', 'Benchmark', 1)
    return f'{prefix} {text}'


def jump_steps_from_times(times, jump_times):
    """Convert padded legacy jump-time arrays, including all Poisson channels."""
    times = _paths(times)
    steps = np.zeros((times.shape[0], max(0, times.shape[1]-1)), dtype=bool)
    if jump_times is None or times.shape[1] < 2:
        return steps
    jumps = _numpy(jump_times)
    for sample in range(min(times.shape[0], jumps.shape[0])):
        events = np.asarray(jumps[sample]).reshape(-1)
        events = events[np.isfinite(events) & (events > 0) &
                        (events >= times[sample, 0]) & (events <= times[sample, -1])]
        indices = np.searchsorted(times[sample], events, side='left')
        indices = np.clip(indices, 1, times.shape[1]-1)
        steps[sample, indices-1] = True
    return steps


def render_trajectory_rows(rows, filename=None, *, downsample_step=1, smooth_window=1,
                           axis_label_size=14, tick_label_size=11, legend_font_size=8.3,
                           subplot_label_size=12, close=True):
    """Render rows of (control/policy mean, value, state), with no model calls.

    No analytical formula or sign convention is inferred here. Every panel
    displays its input values unchanged, apart from optional visual smoothing.
    The same selected paths and valid-time masks are used in all three panels.
    """
    if not rows:
        raise ValueError('At least one trajectory row is required')
    downsample_step = max(1, int(downsample_step))
    configure_plot_style()
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.2, 3.3*len(rows)), squeeze=False)
    colors = dict(approximate='darkorange', benchmark='magenta', jump='red')
    for row_index, row in enumerate(rows):
        if len(row.panels) != 3:
            raise ValueError('A trajectory row must contain exactly three panels')
        times = _paths(row.times)
        shape = times.shape
        mask = np.ones(shape, dtype=bool) if row.mask is None else np.broadcast_to(_paths(row.mask).astype(bool), shape)
        jumps = (np.zeros((shape[0], shape[1]-1), dtype=bool) if row.jump_steps is None
                 else _numpy(row.jump_steps).astype(bool))
        if jumps.shape != (shape[0], shape[1]-1):
            raise ValueError('jump_steps must match the number of trajectory intervals')
        samples = (np.arange(min(3, shape[0])) if row.sample_indices is None
                   else np.asarray(row.sample_indices, dtype=int).reshape(-1))
        samples = np.asarray(list(dict.fromkeys(samples[(samples >= 0) & (samples < shape[0])])), dtype=int)
        if not samples.size:
            raise ValueError('No valid trajectory indices were selected')
        for panel_index, panel in enumerate(row.panels):
            ax = axes[row_index, panel_index]
            approximate = np.broadcast_to(_paths(panel.approximate), shape)
            benchmark = np.broadcast_to(_paths(panel.benchmark), shape)
            std = None if panel.std is None else np.broadcast_to(_paths(panel.std), shape)
            plotted = 0
            has_jump = False
            all_values = []
            for sample in samples:
                valid = (mask[sample] & np.isfinite(times[sample]) &
                         np.isfinite(approximate[sample]) & np.isfinite(benchmark[sample]))
                full_indices = np.flatnonzero(valid)
                if not full_indices.size:
                    continue
                jump_indices = np.flatnonzero(jumps[sample] & valid[:-1] & valid[1:])
                # Keep the terminal sample and both sides of jumps when thinning.
                indices = np.unique(np.concatenate((full_indices[::downsample_step], full_indices[-1:],
                                                    jump_indices, jump_indices+1)))
                t = times[sample, indices]
                pred_raw, ref_raw = approximate[sample, indices], benchmark[sample, indices]
                pred, ref = _smooth(pred_raw, smooth_window), _smooth(ref_raw, smooth_window)
                if smooth_window > 1:
                    ax.plot(t, pred_raw, color=colors['approximate'], alpha=.25, linewidth=.8)
                    ax.plot(t, ref_raw, '--', color=colors['benchmark'], alpha=.25, linewidth=.8)
                if std is not None:
                    ax.fill_between(t, pred-3*std[sample, indices], pred+3*std[sample, indices],
                                    color=colors['approximate'], alpha=.08)
                ax.plot(t, pred, color=colors['approximate'], linewidth=1.25,
                        label=_label(panel.approximate_label, 'Approx.') if plotted == 0 else None)
                ax.plot(t, ref, '--', color=colors['benchmark'], linewidth=1.15,
                        label=_label(panel.benchmark_label, 'Benchmark') if plotted == 0 else None)
                all_values.extend((pred_raw, ref_raw))
                for step in jump_indices:
                    ax.plot(times[sample, step:step+2], benchmark[sample, step:step+2],
                            color=colors['jump'], linewidth=2.5, zorder=4)
                    has_jump = True
                ax.plot(t[-1], ref_raw[-1], 'o', color='black', markersize=3.5, zorder=5)
                if plotted == 0:
                    ax.plot(t[0], ref_raw[0], 's', color='black', markersize=3.5, zorder=5)
                plotted += 1
            ax.set_xlabel(r'$t$ (Time)', fontsize=axis_label_size)
            ax.set_ylabel(panel.ylabel, fontsize=axis_label_size)
            ax.tick_params(axis='both', labelsize=tick_label_size, direction='in')
            ax.text(.015, .985, ('(a)', '(b)', '(c)')[panel_index], transform=ax.transAxes,
                    ha='left', va='top', fontsize=subplot_label_size, fontweight='bold', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.4', linewidth=.5))
            handles, labels = ax.get_legend_handles_labels()
            if has_jump:
                handles.append(Line2D([], [], color=colors['jump'], linewidth=2.5)); labels.append('Jump')
            for marker, label in [('o', panel.end_label), ('s', panel.start_label)]:
                if label:
                    handles.append(Line2D([], [], color='black', marker=marker, linestyle='None', markersize=3.5))
                    labels.append(label)
            ax.legend(handles, labels, loc='lower right' if panel_index == 0 else 'lower left',
                      fontsize=legend_font_size, framealpha=.7, edgecolor='.65', borderpad=.35,
                      handlelength=1.5, labelspacing=.25, handletextpad=.45)
            if all_values:
                values = np.concatenate(all_values)
                lo, hi = float(values.min()), float(values.max())
                span = max(hi-lo, .1*max(abs(lo), abs(hi)), 1e-3)
                lower = lo-(.7 if panel_index == 0 else .65)*span
                upper = hi+.15*span
                if panel_index == 0 and hi < 0:
                    lower, upper = lo-max(.7*span, abs(lo)), max(upper, 0.)
                ax.set_ylim(lower, upper)
            ax.margins(x=.035)
    fig.subplots_adjust(left=.065, right=.99, bottom=.18 if len(rows) == 1 else .10,
                        top=.97, wspace=.28, hspace=.40)
    if filename is not None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, bbox_inches='tight')
    if close:
        plt.close(fig)
    return fig, axes


def plot_trajectory_combined(
    t_test, traj_u_mean, traj_u_test, traj_value, traj_value_test, traj_Xt, traj_Xt_test,
    approximate_u, exact_u, start_u, end_u, ylabel_u,
    approximate_v, exact_v, start_v, end_v, ylabel_v,
    approximate_x, exact_x, start_x, end_x, ylabel_x,
    filename='trajectory_combined', P_test=None, num_samples=2, jump_num=3, jump_index=None,
    device='cpu', output_folder='./pred_traj', downsample_step=1, smooth_window=1, f_std=None,
    axis_label_size=14, tick_label_size=11, legend_font_size=8.3, subplot_label_size=12,
    model_type='multi_cara', coeff_SDE_type='piecewise',
):
    """LQ/Multi-CARA adapter; all analytical curves and value signs come from the caller.

    device/model_type/coeff_SDE_type remain accepted for existing call sites.
    They do not select analytical solutions or change the supplied data.
    """
    times = _paths(t_test)
    jumps = jump_steps_from_times(times, P_test)
    candidates = np.flatnonzero(jumps.any(axis=1)) if jump_index is None else _numpy(jump_index).astype(int).reshape(-1)
    candidates = candidates[(candidates >= 0) & (candidates < times.shape[0])]
    # Deterministic selection avoids advancing the training RNG while plotting.
    selected = list(dict.fromkeys(candidates.tolist()))[:max(0, int(jump_num))]
    non_jump = np.flatnonzero(~jumps.any(axis=1))
    selected.extend(int(i) for i in non_jump[:max(1, int(num_samples))] if int(i) not in selected)
    if not selected:
        selected = list(range(min(max(1, int(num_samples)), times.shape[0])))
    panels = (
        TrajectoryPanel(traj_u_mean, traj_u_test, ylabel_u, approximate_u, exact_u, start_u, end_u, f_std),
        TrajectoryPanel(traj_value, traj_value_test, ylabel_v, approximate_v, exact_v, start_v, end_v),
        TrajectoryPanel(traj_Xt, traj_Xt_test, ylabel_x, approximate_x, exact_x, start_x, end_x),
    )
    row = TrajectoryRow(times, panels, jump_steps=jumps, sample_indices=selected)
    destination = None if filename is None else Path(output_folder)/f'{filename}.pdf'
    fig, axes = render_trajectory_rows([row], destination, downsample_step=downsample_step,
                                      smooth_window=smooth_window, axis_label_size=axis_label_size,
                                      tick_label_size=tick_label_size, legend_font_size=legend_font_size,
                                      subplot_label_size=subplot_label_size, close=False)
    plt.show()
    return fig, axes[0]


def plot_trajectory_rows(rows, filename):
    """Time-inhomogeneous LQ adapter; retain its explicit display convention v=-V."""
    prepared = []
    if not 1 <= len(rows) <= 2:
        raise ValueError('plot_trajectory_rows expects one or two runs')
    for cfg, data in rows:
        policy = cfg.mode == 'policy'
        panels = (
            TrajectoryPanel(data['mean'], data['mean_reference'],
                r'Mean of $\pi(t,\hat X_t)$' if policy else r'$u(t,\hat X_t)$',
                r'Approx. Mean of $\hat\pi(t,\hat X_t)$' if policy else r'Approx. $\hat u(t,\hat X_t)$',
                r'Benchmark Mean of $\pi^*(t,\hat X_t)$' if policy else r'Benchmark $u^*(t,\hat X_t)$',
                r'Mean of $\pi^*(0,X_0)$' if policy else r'$u^*(0,X_0)$',
                r'Mean of $\pi^*(T,X_T)$' if policy else r'$u^*(T,X_T)$'),
            TrajectoryPanel(-_numpy(data['value']), -_numpy(data['value_reference']), r'$v(t,\hat X_t)$',
                r'Approx. $\hat v(t,\hat X_t)$', r'Benchmark $v(t,\hat X_t)$', r'$v(0,X_0)$', r'$v(T,X_T)$'),
            TrajectoryPanel(data['x'], data['x_reference'], r'$X_t$',
                r'Approx. $\hat X_t$ by $\hat\pi$' if policy else r'Approx. $\hat X_t$ by $\hat u$',
                r'Benchmark $X_t$ by $\pi^*$' if policy else r'Benchmark $X_t$ by $u^*$', r'$X_0$', r'$X_T$'),
        )
        jump_counts = _numpy(data['jumps'])
        jumps = np.any(jump_counts[:, :-1] > 0, axis=-1) if jump_counts.ndim == 3 else jump_counts[:, :-1] > 0
        prepared.append(TrajectoryRow(data['t'], panels, jumps, data['mask']))
    return render_trajectory_rows(prepared, filename)


def plot_multi_cara_trajectories(
    t_test, V_pred, V_test, X_pred, X_test, control_pred, control_test, P_test, P0_test,
    *, dimensions, entropy_alpha, output_folder, jump_index=None, players_to_plot=None,
    filename_suffix="", device="cpu", downsample_step=1, smooth_window=1,
    CriticLoss=None, ActorLoss=None,
):
    """Adapt precomputed CARA trajectories to the shared renderer, without model evaluation."""
    D = dimensions
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    V_test = _numpy(V_test)
    V_pred = _numpy(V_pred)
    X_test = _numpy(X_test)
    X_pred = _numpy(X_pred)
    t_test = _numpy(t_test)


    control_pred, control_test = _numpy(control_pred), _numpy(control_test)
    P_test = None if P_test is None else _numpy(P_test)
    P0_test = None if P0_test is None else _numpy(P0_test)

    if players_to_plot is not None:
        players_to_plot = sorted(players_to_plot)
        print(f"[plot] Selected players: {players_to_plot} (rollout uses networks for all {D} players; unloaded players may use randomly initialized networks)")
    else:
        players_to_plot = list(range(D))

    for k in players_to_plot:
        P_k = None
        if P_test is not None and P0_test is not None:
            P_idio_k = P_test[:, :, k:k+1]          # (M,J_idio,1)
            P_k = np.concatenate([P0_test, P_idio_k], axis=1)  # (M,J_tot,1)


        ctrl_pred_k = np.concatenate(
            [control_pred[:, :, k:k+1],
            control_pred[:, -1:, k:k+1]],
            axis=1
        )
        ctrl_test_k = np.concatenate(
            [control_test[:, :, k:k+1],
            control_test[:, -1:, k:k+1]],
            axis=1
        )

        # Set labels based on entropy_alpha
        if entropy_alpha == 0:
            approximate_u = f"$\\hat{{u}}_{{k+1}}(t,X_t)$"
            exact_u = f"$u_{{k+1}}^*(t,X_t)$"
            start_u = f"$u_{{k+1}}^*$"
            end_u = f"$u_{{k+1}}^*$"
            ylabel_u = f"$u_{{k+1}}(t,X_t)$"
        else:
            approximate_u = f"Mean of $\\hat{{u}}_{{k+1}}(t,X_t)$"
            exact_u = f"Mean of $u_{{k+1}}^*(t,X_t)$"
            start_u = f"Mean of $u_{{k+1}}^*$"
            end_u = f"Mean of $u_{{k+1}}^*$"
            ylabel_u = f"Mean of $u_{{k+1}}(t,X_t)$"


        plot_trajectory_combined(
            t_test,
            traj_u_mean=ctrl_pred_k,
            traj_u_test=ctrl_test_k,
            traj_value=V_pred[:, :, k:k+1],
            traj_value_test=V_test[:, :, k:k+1],
            traj_Xt=X_pred[:, :, k:k+1],
            traj_Xt_test=X_test[:, :, k:k+1],
            approximate_u=approximate_u,
            exact_u=exact_u,
            start_u=start_u,
            end_u=end_u,
            ylabel_u=ylabel_u,
            approximate_v=f"$V_{{k+1}}(t,X_t)$",
            exact_v=f"$V_{{k+1}}^*(t,X_t)$",
            start_v=f"$V_{{k+1}}(0,X_0)$",
            end_v=f"$V_{{k+1}}(T,X_T)$",
            ylabel_v=f"$V_{{k+1}}(t,X_t)$",
            approximate_x=f"$\\hat X_t^{{k+1}}$",
            exact_x=f"$X_t^{{k+1}}$",
            start_x=f"$X_0^{{k+1}}$",
            end_x=f"$X_T^{{k+1}}$",
            ylabel_x=f"$X_t^{{k+1}}$",
            filename=f"traj_combined{k+1}{filename_suffix}",
            P_test=P_k,
            num_samples=1,
            jump_num=3,
            jump_index=jump_index,
            device=device,
            output_folder=output_folder,
            downsample_step=downsample_step,
            smooth_window=smooth_window,
            model_type="multi_cara",
        )


    plot_multi_agent_diagnostics(
        V_pred, V_test, control_pred, control_test, t_test,
        dimensions=D, output_folder=output_folder, filename_suffix=filename_suffix,
        CriticLoss=CriticLoss, ActorLoss=ActorLoss,
    )


def plot_lq_trajectories(times, control_mean, control_reference, value, value_reference,
                         state, state_reference, *, entropy_alpha, output_folder,
                         P_test=None, jump_index=None, filename_suffix="",
                         downsample_step=10, smooth_window=1):
    """Adapt LQ trajectories; callers supply analytical curves and displayed value signs."""
    def _fn(base):
        return f"{base}{filename_suffix}"

    # Set labels based on entropy_alpha

    if entropy_alpha == 0:
        approximate_u = "$\\hat{u}(t,\\hat{X}_t)$"
        exact_u = "$u^*(t,\\hat{X}_t)$"
        start_u = "$u^*(0,X_0)$"
        end_u = "$u^*(T,X_T)$"
        ylabel_u = "$u(t,\\hat{X}_t)$"
    else:
        approximate_u = "Mean of $\\hat{\\pi}(t,\\hat{X}_t)$"
        exact_u = "Mean of $\\pi^*(t,\\hat{X}_t)$"
        start_u = "Mean of $\\pi^*(0,X_0)$"
        end_u = "Mean of $\\pi^*(T,X_T)$"
        ylabel_u = "Mean of $\\pi(t,\\hat{X}_t)$"

    plot_trajectory_combined(
        times,
        traj_u_mean=control_mean,
        traj_u_test=control_reference,
        traj_value=value,
        traj_value_test=value_reference,
        traj_Xt=state,
        traj_Xt_test=state_reference,
        approximate_u=approximate_u,
        exact_u=exact_u,
        start_u=start_u,
        end_u=end_u,
        ylabel_u=ylabel_u,
        approximate_v="$\\hat{v}(t,\\hat{X}_t)$",
        exact_v="$v(t,\\hat{X}_t)$",
        start_v="$v(0,X_0)$",
        end_v="$v(T,X_T)$",
        ylabel_v="$v(t,\\hat{X}_t)$",
        approximate_x="$\\hat{X}_t$ by $\\hat{u}$",
        exact_x="$X_t$ by $u^*$",
        start_x="$X_0$",
        end_x="$X_T$",
        ylabel_x="$X_t$",
        filename=_fn("merton_traj_combined"),
        P_test=P_test,
        jump_index=jump_index,
        num_samples=1,
        jump_num=3,
        output_folder=output_folder,
        downsample_step=downsample_step,
        smooth_window=smooth_window,
        model_type="lq_merton"
    )


def plot_time_inhomo_results(cfg, trajectory, history, metrics, output_folder):
    """Render precomputed time-inhomogeneous LQ results using shared plotting helpers."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    plot_trajectory_rows([(cfg, trajectory)], output_folder/"trajectories.pdf")
    if cfg.mode == "policy":
        plot_reference_series(
            trajectory["t"][0, :, 0], trajectory["std"][0, :, 0],
            trajectory["std_reference"][0, :, 0], filename=output_folder/"policy_std.pdf",
            ylabel="Standard deviation", approximate_label="Approx. policy std",
            benchmark_label="Benchmark policy std",
        )
    measured = [record for record in history if "score" in record]
    if measured:
        plot_metric_history(
            measured, (("value_relative_rmse", "Value RMSE", "darkorange", "-"),
                       ("control_error", metrics["control_metric"], "darkviolet", "--")),
            filename=output_folder/"training_errors.pdf",
        )
