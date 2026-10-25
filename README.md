# RL Jump Stochastic Control

English | [简体中文](README_zh.md)

[A Continuous-Time Actor-Critic Framework for Jump-Diffusion Controls with Normalizing Flows](https://arxiv.org/abs/2604.05398) (arXiv:2604.05398).

Numerical experiments for stochastic control with Brownian noise and compensated Poisson jumps, covering time-inhomogeneous LQ control, time-homogeneous LQ control, Merton portfolio optimization, and multi-player CARA games. The main manuscript is available in [SISC-1002-jump-main.pdf](pdf/SISC-1002-jump-main.pdf). Proofs, benchmark derivations, and additional numerical details are provided in [SISC-1002-jump-supp.pdf](pdf/SISC-1002-jump-supp.pdf).

This README describes the current Python entrypoints for training, evaluation, resuming training, and plotting. Run all commands from the repository root.

Analytical solutions and PINN benchmarks are used only for evaluation, checkpoint selection, and plotting. They do not provide supervised losses, training actions, or learning-rate feedback for the actor or critic.

Five existing result sets are stored in the default directories corresponding to their training parameters:

- [LQ, D=5, T=10, alpha=0](output/LQ/D5_T10_N1000_M500/alpha0_lrA5e-05_lrC0.001_it1000/)
- [LQ, D=5, T=10, alpha=0.05](output/LQ/D5_T10_N1000_M500/alpha0.05_lrA5e-05_lrC0.001_it1000/)
- [LQ, D=5, T=100, alpha=0](output/LQ/D5_T100_N1000_M500/alpha0_lrA5e-05_lrC0.001_it1000/)
- [LQ, D=5, T=100, alpha=0.05](output/LQ/D5_T100_N1000_M500/alpha0.05_lrA5e-05_lrC0.001_it1000/)
- [Merton, D=1, T=10, alpha=0.05](output/Merton/D1_T10_N200_M200/alpha0.05_lrA4e-05_lrC0.0008_it2000/)

The saved LQ results for T=10, alpha=0/0.05 and T=100, alpha=0 use the same semi-gradient critic update in pretraining and joint training.

These directories contain weights, evaluation figures, source snapshots, and full configurations. Paths in `params.json`, Top-k metadata, and periodic checkpoint configurations are relative to the containing JSON file, so a run directory can be moved together with its weights. The default entrypoints train actor–critic models without supervision from analytical solutions; analytical and PINN benchmarks are reserved for evaluation, checkpoint selection, and plotting.

## 1. Installation and command conventions

Python 3.11 or newer is required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The single-agent evaluation examples use the following variables. Replace them with actual weight files from the same training run:

```bash
ACTOR_CKPT="/absolute/path/to/actor_checkpoint.pth"
CRITIC_CKPT="/absolute/path/to/critic_checkpoint.pth"
```

Keep `params.json`, Top-k metadata, and the PINN cache alongside the weights. Evaluation must use the entropy mode and network architecture used during training. Missing, incomplete, or incompatible actor/critic checkpoints stop evaluation or resume with an error.

Fresh training requires an empty output directory. Use `--output-dir` for a new run; add `--resume` with valid checkpoints to continue training, or explicitly add `--overwrite` to delete and replace an existing run directory. Evaluation only loads checkpoints and generates diagnostics; it never falls back to training.

## 2. Entrypoints

- [Time-inhomogeneous LQ](scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py): a unified entrypoint for convergence/periodic coefficients and control/policy modes.
- [Time-homogeneous LQ](scripts/LQ/best_setting_LQ.py): LQ control and Gaussian policies, with actor input normalization and initial-state coverage.
- [Merton PINN/flow](scripts/Merton/merton_pinn_resnet_SAC_opt.py): a conditional spline-flow policy evaluated against PINN/Gibbs benchmarks.
- [Multi-player CARA](scripts/Multi-CARA/Multi-player-CARA.py): separate actors and critics for each player, learning Gaussian means and standard deviations.
- [Multi-player CARA with fixed std](scripts/Multi-CARA/Multi-player-CARA_fixed_std.py): fixed exploration standard deviation in classical control mode.

### Paper figures and configurations

Figure numbers follow the main manuscript. Training commands are provided in the linked sections.

| Figure | Experiment | Configuration | Training commands |
| --- | --- | --- | --- |
| Fig. 1 | LQ | D=5, T=10/100, alpha=0, N=1000, M=500; 1,000 iterations | First two commands in [Section 3.2](#32-time-homogeneous-lq). |
| Fig. 2 | LQ | D=5, T=10, alpha=0.05, N=1000, M=500; 1,000 iterations | Third command in [Section 3.2](#32-time-homogeneous-lq). |
| Fig. 3 | Time-inhomogeneous LQ | T=20, N=2000, M=100; 3,000 iterations | [Section 3.1](#31-time-inhomogeneous-lq): `--coefficient-type convergence --entropy-alpha 0` (top); `--coefficient-type periodic --entropy-alpha 0.05` (bottom). |
| Fig. 5 | Merton PINN/flow | alpha=0.05, T=10, N=200, M=200; 2,000 iterations | Training command in [Section 4](#4-merton-pinn--spline-flow-actor). |
| Figs. 6–7 | CARA | 25 players, alpha=0, T=10, N=500, M=100; 1,000 iterations | Training command in [Section 5.1](#51-standard-version). |

## 3. LQ

### 3.1 Time-inhomogeneous LQ

**File:** `scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py`

**Configuration:** This entrypoint includes the networks, coefficient functions, Riccati benchmark solver, training, evaluation, and resume support. The `convergence` coefficients approach stationary limits; the `periodic` coefficients vary periodically. `convergent` is a compatibility alias for `convergence`. Defaults: periodic coefficients, entropy alpha=0.05, T=20, N=2000, M=100, D=1, 3,000 iterations, network width 16, and depth 3. T is the finite rollout window for the infinite-horizon problem.

By default, training uses one actor and one value network. TD bootstrapping uses the same value network's next-state prediction with its gradient stopped. No target network is created, and no Polyak averaging is performed.

- `--entropy-alpha 0`: fixed exploration std, with only the policy mean learned. Evaluation uses deterministic control and the classical value/control benchmark.
- `--entropy-alpha 0.05`: learn both the Gaussian mean and std. Evaluation samples the stochastic policy and uses the entropy-regularized value/policy benchmark with the same alpha.
- Only Shannon entropy is supported. Control mode uses relative RMSE; policy mode uses `KL(pi* || pi_hat)`.

**Training:** Run the four configurations separately:

```bash
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py --coefficient-type convergence --entropy-alpha 0
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py --coefficient-type convergence --entropy-alpha 0.05
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py --coefficient-type periodic --entropy-alpha 0
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py --coefficient-type periodic --entropy-alpha 0.05
```

**Evaluation:** The configuration is loaded automatically from `params.json` next to the checkpoints. Explicit CLI options override the configuration; the entropy mode and time encoding must match the weights.

```bash
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py \
  --actor-path "$ACTOR_CKPT" --critic-path "$CRITIC_CKPT"
```

Use `--T_eval` and `--N_eval` to change the evaluation window and number of time steps. Use `--no-plots` to save numerical results only.

**Resuming:** For this entrypoint, `--iterations 100` means 100 additional iterations.

```bash
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py \
  --actor-path "$ACTOR_CKPT" --critic-path "$CRITIC_CKPT" \
  --resume --iterations 100
```

The `actor_last.pth`, `critic_last.pth`, and `training_state.pth` files from the same directory restore weights, optimizers, schedulers, and RNG states. Best or historical checkpoints without matching training states start fresh optimizers from the loaded weights.

### 3.2 Time-homogeneous LQ

**File:** `scripts/LQ/best_setting_LQ.py`

**Configuration:** Actor–critic experiments for time-homogeneous LQ control, compared with classical linear control or the Gaussian policy under Shannon entropy regularization. Defaults: D=5, T=100, N=1000, M=500, 1,000 iterations, and alpha=0.05. With alpha=0, training uses a fixed exploration std and learns only the mean.

**Training:**

```bash
python3 scripts/LQ/best_setting_LQ.py \
  --entropy-alpha 0 --fixed-std 0.1 --horizon 10 --output-dir output/paper/figure1_T10
python3 scripts/LQ/best_setting_LQ.py \
  --entropy-alpha 0 --fixed-std 0.1 --horizon 100 --output-dir output/paper/figure1_T100
python3 scripts/LQ/best_setting_LQ.py \
  --entropy-alpha 0.05 --horizon 10 --output-dir output/paper/figure2
```

Use `--actor-loss entropy_outside` for the separate-entropy ablation. Set T with `--horizon 10` or `--horizon 100`, and choose an output directory with `--output-dir`. N, M, the dimension, and the iteration count are configured in the `params` dictionary at the end of the file.

**Evaluation:**

```bash
python3 scripts/LQ/best_setting_LQ.py \
  --actor_path "$ACTOR_CKPT" --critic_path "$CRITIC_CKPT"
```

The script loads the configuration and training history next to the checkpoints and generates evaluation/Top-k plots. Use `--plot-history-only` with the checkpoint paths to redraw saved histories; runs without `history.json` are re-evaluated only at their retained checkpoints. Inspect control or distribution errors as well as value errors. The `[actor-slope]` diagnostics report the mean-policy slope, cross-dimensional coupling, and time dependence. The stored `control_error` and Top-k scores retain their historical checkpoint-selection convention. In `merton_train.pdf`, alpha=0 reports the actor-mean error against the benchmark at the same pre-step states; alpha>0 reports the square root of mean Gaussian KL at those states, rather than a percentage control error.

**Resuming:** Add `--resume` to the evaluation command to load weights and continue training. This older entrypoint does not restore the complete optimizer/RNG state.

**Outputs:** `output/LQ/<scale>/<parameters>/` by default, containing configurations, source code, actor/critic weights, Top-k metadata, error/loss histories, and trajectory PDFs.

## 4. Merton PINN / spline-flow actor

**File:** `scripts/Merton/merton_pinn_resnet_SAC_opt.py`

**Configuration:** A conditional Gaussian base distribution followed by a spline flow, evaluated against PINN/Gibbs benchmarks. Defaults: alpha=0.05, T=10, N=200, M=200, 2,000 iterations, evaluation every 50 iterations, network width 32, and depth 2. Training size, evaluation frequency, and model parameters are configured in the `params` dictionary at the end of the file.

**Training:**

```bash
WANDB_MODE=disabled python3 scripts/Merton/merton_pinn_resnet_SAC_opt.py
```

Use `--output-dir` to choose a new output directory. Running the command above starts training from scratch without loading historical weights.

**Evaluation:** Use actor/critic weights from the same training run and fixed random seeds:

```bash
WANDB_MODE=disabled python3 scripts/Merton/merton_pinn_resnet_SAC_opt.py \
  --actor_path "$ACTOR_CKPT" --critic_path "$CRITIC_CKPT" \
  --plot_traj_mode ab --policy_plot_time_fraction 0.25 \
  --eval_M 100 --N_eval 200 \
  --eval_rollout_fixed_seed 123456 --pinn_train_fixed_seed 123456
```

Training and evaluation produce horizontal three-panel figures:

- **(a) Conditional policy distribution:** compare the PINN/Gibbs density with a normalized histogram of FlowNet action samples at the same `(t,x)`, with t=T/4 by default. The axes are u and `p(u|t,x)`.
- **(b) Value trajectory:** compare network values with the PINN benchmark along the same learned-policy state trajectory.
- **(c) State trajectory:** compare learned-policy and benchmark-policy trajectories, with jumps marked.

Set `--policy_plot_time_fraction 0.5` to use t=T/2, or `--panel_a_fixed_x 12` to fix the conditioning state. Plotting uses the current source code by default.

**Outputs:** `output/Merton/<scale>/<parameters>/` by default, containing configurations, source code, actor/critic weights, Top-k records, training states, PINN caches, and evaluation PDFs. The legacy entrypoint uses `output/Merton_legacy/`.

## 5. Multi-player CARA

### 5.1 Standard version

**File:** `scripts/Multi-CARA/Multi-player-CARA.py`

**Configuration:** Multi-player CARA control with common and idiosyncratic noise and jumps. Each player has an independent actor and critic; actors learn Gaussian means and standard deviations. Defaults: 25 players, alpha=0, T=10, N=500, M=100, and 1,000 iterations.

**Training:**

```bash
python3 scripts/Multi-CARA/Multi-player-CARA.py --output-dir output/paper/figures6_7
```

The number of players, equation coefficients, alpha, and iteration count are configured in `params` at the end of the file. The CLI provides checkpoint, resume, `--output-dir`, and `--overwrite` options.

**Evaluation:** Store the complete set of player weights in one directory, alongside `params.json` and `eval_topk_meta.json`:

```bash
CARA_CKPT_DIR="/absolute/path/to/cara_run"
python3 scripts/Multi-CARA/Multi-player-CARA.py \
  --actor_path "$CARA_CKPT_DIR" --critic_path "$CARA_CKPT_DIR"
```

The script loads all players from the metadata or the `actor_player{i}_eval_top1.pth` / `critic_player{i}_eval_top1.pth` files and generates value/control errors and trajectory plots.

**Resuming:** Add `--resume` to the evaluation command to continue from the complete checkpoint set. Outputs are written to the run's `resume/` directory. The complete optimizer/RNG state is not restored.

**Outputs:** `output/Multi-CARA/<scale>/<parameters>/`, containing configurations, source code, player weights, Top-k metadata, error histories, and PDFs.

For fixed exploration variance with `alpha=0`, run `python3 scripts/Multi-CARA/Multi-player-CARA_fixed_std.py --fixed_std 0.1 --output-dir output/paper/cara_fixed_std` (fixed standard deviation 0.1).

## 6. Citation

If you use this code or these experiments in your research, please cite:

Liya Guo, Ruimeng Hu, Xu Yang, and Yi Zhu. *A Continuous-Time Actor-Critic Framework for Jump-Diffusion Controls with Normalizing Flows*. arXiv:2604.05398, 2026. [Paper and citation information](https://arxiv.org/abs/2604.05398).

```bibtex
@misc{guo2026actorcriticframeworkcontinuoustimejumpdiffusion,
      title={A Continuous-Time Actor-Critic Framework for Jump-Diffusion Controls with Normalizing Flows},
      author={Liya Guo and Ruimeng Hu and Xu Yang and Yi Zhu},
      year={2026},
      eprint={2604.05398},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2604.05398},
}
```
