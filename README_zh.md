# RL jump stochastic control

[English](README.md) | 简体中文

[A Continuous-Time Actor-Critic Framework for Jump-Diffusion Controls with Normalizing Flows](https://arxiv.org/abs/2604.05398)（arXiv:2604.05398）。

带 Brownian 噪声和补偿 Poisson 跳跃的随机控制数值实验，包含时间非齐次 LQ、时间齐次 LQ、Merton 投资及 Multi-player CARA。理论手稿正文见 [SISC-1002-jump-main.pdf](pdf/SISC-1002-jump-main.pdf)；理论证明、基准解推导及数值实验细节见补充材料 [SISC-1002-jump-supp.pdf](pdf/SISC-1002-jump-supp.pdf)。

本文只列出仓库根目录下现行脚本的训练、评估、续训和画图命令；所有命令均在仓库根目录执行。

所有入口的解析解和 PINN 数值基准仅用于评估、模型选择及绘图，不用于 actor/critic 的监督损失、训练动作或学习率调整。

现有五组结果已归入对应入口按实际训练参数生成的默认目录：

- [LQ，D=5、T=10、alpha=0](output/LQ/D5_T10_N1000_M500/alpha0_lrA5e-05_lrC0.001_it1000/)
- [LQ，D=5、T=10、alpha=0.05](output/LQ/D5_T10_N1000_M500/alpha0.05_lrA5e-05_lrC0.001_it1000/)
- [LQ，D=5、T=100、alpha=0](output/LQ/D5_T100_N1000_M500/alpha0_lrA5e-05_lrC0.001_it1000/)
- [LQ，D=5、T=100、alpha=0.05](output/LQ/D5_T100_N1000_M500/alpha0.05_lrA5e-05_lrC0.001_it1000/)
- [Merton，D=1、T=10、alpha=0.05](output/Merton/D1_T10_N200_M200/alpha0.05_lrA4e-05_lrC0.0008_it2000/)

上述 T=10、alpha=0/0.05 和 T=100、alpha=0 的 LQ 结果在 critic 预训练和联合训练中采用一致的半梯度更新。

这些目录保存权重、评估图、源码快照和完整参数；`params.json`、Top-k 元数据及周期 checkpoint 的参数文件均使用相对于所在 JSON 文件的路径，可连同权重一起搬迁运行目录。默认训练入口统一使用无解析解监督的 actor–critic 方法，解析解及 PINN 数值基准只用于评估、模型选择和绘图。

## 1. 安装与命令约定

需要 Python 3.11 或更新版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

单主体评估示例使用以下变量，请替换为同一轮训练产生的实际权重文件：

```bash
ACTOR_CKPT="/absolute/path/to/actor_checkpoint.pth"
CRITIC_CKPT="/absolute/path/to/critic_checkpoint.pth"
```

保留权重同目录的 `params.json`、Top-k 元数据及 PINN 缓存。评估应使用训练时的 entropy 模式和网络结构。actor/critic 权重缺失、不完整或与网络不兼容时，评估和续训会直接报错。

新训练要求输出目录为空。可用 `--output-dir` 指定新目录；续训须提供有效权重并添加 `--resume`；只有显式添加 `--overwrite` 才会删除并替换已有运行目录。评估只加载权重并生成诊断，不会自动转为训练。

## 2. 运行入口索引

- [时间非齐次 LQ](scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py)：统一运行 convergence/periodic 与 control/policy 四种组合。
- [时间齐次 LQ](scripts/LQ/best_setting_LQ.py)：LQ 控制及高斯策略，支持 actor 输入归一化与训练初始状态覆盖。
- [Merton PINN/flow](scripts/Merton/merton_pinn_resnet_SAC_opt.py)：条件 spline flow 策略，使用 PINN/Gibbs 数值基准。
- [Multi-player CARA](scripts/Multi-CARA/Multi-player-CARA.py)：各主体学习 Gaussian 均值与标准差。
- [Multi-player CARA fixed std](scripts/Multi-CARA/Multi-player-CARA_fixed_std.py)：经典控制模式固定训练探索标准差。

### 论文图号与运行时间

图号对应正文手稿；以下命令使用当前代码重新运行相应数值算例。

- **图 1：** LQ，D=5、T=10/100、alpha=0、N=1000、M=500、1,000 轮。运行第 3.2 节的前两条训练命令。耗时尚无记录。
- **图 2：** LQ，D=5、T=10、alpha=0.05、N=1000、M=500、1,000 轮。运行第 3.2 节第三条训练命令。耗时尚无记录。
- **图 3：** 时间非齐次 LQ，T=20、N=2000、M=100、3,000 轮。按第 3.1 节运行 `convergence --entropy-alpha 0`（上排）与 `periodic --entropy-alpha 0.05`（下排）。耗时尚无记录。
- **图 4：** 历史经典 Merton 算例，alpha=0、T=10、dt=0.01、2,000 轮。当前版本尚未提供与该配置匹配并经过核验的独立运行命令。耗时尚无记录。
- **图 5：** Merton PINN/flow，alpha=0.05、T=10、N=200、M=200、2,000 轮。运行第 4 节的训练命令。耗时尚无记录；首次评估缺少缓存时还需计算 PINN 基准。
- **图 6–7：** CARA，25 个主体、alpha=0、T=10、N=500、M=100、1,000 轮。运行第 5.1 节的训练命令。论文报告 NVIDIA RTX 4090 上耗时 1,050.64 分钟，约 17.5 小时，属于历史参考值，并非当前版本重新计时的结果。

没有实测记录的配置，可按热身后相邻训练日志的耗时估算“每轮秒数 × 总轮数”，再加上评估和绘图时间；GPU/CPU 的选择会影响耗时。

## 3. LQ

### 3.1 时间非齐次 LQ

**文件：** `scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py`

**用途与配置：** 同一入口提供网络、系数、Riccati 基准求解、训练、评估及续训。`convergence` 系数趋向稳态，`periodic` 系数周期变化；`convergent` 是前者的兼容别名。默认 periodic、entropy alpha=0.05、T=20、N=2000、M=100、D=1、3000 轮、网络宽 16/深 3。T 是无限时域问题的有限采样窗口。

默认使用一个 actor 和一个 value net。TD 自举使用同一个 value net 的下一状态预测，并停止该预测的梯度；不创建目标网络，也不进行 Polyak 平均。

- `--entropy-alpha 0`：固定 std 探索、只学习均值；评估执行确定性控制，对应经典价值与控制基准。
- `--entropy-alpha 0.05`：学习 Gaussian 均值及 std；评估随机策略，对应同一 alpha 的熵正则价值及策略基准。
- 仅支持 Shannon 熵。control 使用相对 RMSE，policy 使用 `KL(pi* || pi_hat)`。

**训练：** 四种情况分别运行：

```bash
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py --coefficient-type convergence --entropy-alpha 0
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py --coefficient-type convergence --entropy-alpha 0.05
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py --coefficient-type periodic --entropy-alpha 0
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py --coefficient-type periodic --entropy-alpha 0.05
```

**评估：** 从 checkpoint 同目录自动读取 `params.json`。显式 CLI 参数覆盖配置；entropy 模式及时间编码需与权重一致。

```bash
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py \
  --actor-path "$ACTOR_CKPT" --critic-path "$CRITIC_CKPT"
```

可用 `--T_eval`、`--N_eval` 调整评估窗口和步数，用 `--no-plots` 仅保存数值结果。

**续训：** `--iterations 100` 在此入口表示追加 100 轮。

```bash
python3 scripts/time_inhomo/best_setting_LQ_time_inhomogeneous_unified.py \
  --actor-path "$ACTOR_CKPT" --critic-path "$CRITIC_CKPT" \
  --resume --iterations 100
```

使用同目录的 `actor_last.pth`、`critic_last.pth`、`training_state.pth` 可恢复权重、优化器、scheduler 及 RNG。best 或历史权重没有匹配训练状态时，从权重启动新的优化器。

### 3.2 时间齐次 LQ

**文件：** `scripts/LQ/best_setting_LQ.py`

**用途与配置：** 时间齐次 LQ 的 actor–critic 实验，与经典线性控制或 Shannon 熵下的 Gaussian 策略基准比较。默认 D=5、T=100、N=1000、M=500、1000 轮、alpha=0.05。alpha=0 时固定训练 std，只学习均值。

**训练：**

```bash
python3 scripts/LQ/best_setting_LQ.py \
  --entropy-alpha 0 --fixed-std 0.1 --horizon 10 --output-dir output/paper/figure1_T10
python3 scripts/LQ/best_setting_LQ.py \
  --entropy-alpha 0 --fixed-std 0.1 --horizon 100 --output-dir output/paper/figure1_T100
python3 scripts/LQ/best_setting_LQ.py \
  --entropy-alpha 0.05 --horizon 10 --output-dir output/paper/figure2
```

用 `--actor-loss entropy_outside` 运行熵项分离消融。`--horizon 10` 或 `--horizon 100` 设置 T，`--output-dir` 指定新结果目录。N、M、维度和训练轮数仍在文件末尾的 `params` 中配置。

**评估：**

```bash
python3 scripts/LQ/best_setting_LQ.py \
  --actor_path "$ACTOR_CKPT" --critic_path "$CRITIC_CKPT"
```

读取 checkpoint 同目录的参数及历史记录，并生成评估/Top-k 图。 配合 checkpoint 路径使用 `--plot-history-only` 可重画历史图；缺少 `history.json` 的旧运行仅重新评估仍保留的检查点。除价值误差外，应关注控制或分布误差；`[actor-slope]` 日志记录均值斜率、跨维耦合与时间依赖诊断。已存的 `control_error` 和 Top-k 分数保留原检查点评分口径。`merton_train.pdf` 在 alpha=0 时展示同一步、同一状态下 actor 均值与基准控制的误差；alpha>0 时展示同一状态下 Gaussian KL 均值的平方根，不是百分比控制误差。

**续训：** 在上述评估命令末尾添加 `--resume`，恢复权重并继续训练。该旧入口的续训不等同于完整优化器/RNG 状态恢复。

**输出：** 默认 `output/LQ/<训练规模>/<主要参数>/`，包含参数、源码、actor/critic 权重、Top-k 元数据、误差/损失历史与轨迹 PDF。

## 4. Merton PINN / spline-flow actor

**文件：** `scripts/Merton/merton_pinn_resnet_SAC_opt.py`

**用途与配置：** 条件 Gaussian base 加 spline flow，使用 PINN/Gibbs 数值基准。默认 alpha=0.05、T=10、N=200、M=200、2000 轮，每 50 轮评估一次，网络宽 32、深 2。训练规模、评估间隔及其他模型参数在文件末尾的 `params` 中配置。

**训练：**

```bash
WANDB_MODE=disabled python3 scripts/Merton/merton_pinn_resnet_SAC_opt.py
```
可用 `--output-dir` 指定新结果目录。直接运行上述 Python 入口即从头训练，不加载旧权重。

**评估：** 使用同一次训练保存的 actor/critic 权重及固定随机种子：

```bash
WANDB_MODE=disabled python3 scripts/Merton/merton_pinn_resnet_SAC_opt.py \
  --actor_path "$ACTOR_CKPT" --critic_path "$CRITIC_CKPT" \
  --plot_traj_mode ab --policy_plot_time_fraction 0.25 \
  --eval_M 100 --N_eval 200 \
  --eval_rollout_fixed_seed 123456 --pinn_train_fixed_seed 123456
```

训练与评估均生成横向三联图：

- **(a) 条件策略分布**：默认在 `t=T/4` 的同一个 `(t,x)` 比较 PINN/Gibbs 密度与 FlowNet 动作样本的归一化直方图，横轴为 `u`，纵轴为 `p(u|t,x)`。
- **(b) 价值轨迹**：在相同的学习策略状态轨迹上比较网络价值与 PINN 基准。
- **(c) 状态轨迹**：比较学习策略与基准策略的状态轨迹，并标出跳跃。

评估命令中使用 `--policy_plot_time_fraction 0.5` 可切换到 `t=T/2`；使用 `--panel_a_fixed_x 12` 可固定条件状态。默认使用当前代码绘图。

**输出：** 默认保存到 `output/Merton/<训练规模>/<主要参数>/`，包含参数及源码、actor/critic 权重、Top-k 记录、训练状态、PINN 缓存和评估 PDF；旧版入口使用 `output/Merton_legacy/`。

## 5. Multi-player CARA

### 5.1 标准版本

**文件：** `scripts/Multi-CARA/Multi-player-CARA.py`

**用途与配置：** Multi-player CARA 控制实验，包含公共/个体噪声及跳跃。各主体使用独立 actor/critic，actor 学习 Gaussian 均值和 std。默认 25 个主体、alpha=0、T=10、N=500、M=100、1000 轮。

**训练：**

```bash
python3 scripts/Multi-CARA/Multi-player-CARA.py --output-dir output/paper/figures6_7
```

主体数量、方程系数、alpha 和轮数在文件末尾 `params` 中配置，命令行支持 checkpoint、`--resume`、`--output-dir` 和 `--overwrite` 参数。

**评估：** 多主体权重必须作为完整集合保存在同一目录。设置包含 `params.json`、`eval_topk_meta.json` 和所有主体权重的目录：

```bash
CARA_CKPT_DIR="/absolute/path/to/cara_run"
python3 scripts/Multi-CARA/Multi-player-CARA.py \
  --actor_path "$CARA_CKPT_DIR" --critic_path "$CARA_CKPT_DIR"
```

入口按元数据或 `actor_player{i}_eval_top1.pth` / `critic_player{i}_eval_top1.pth` 加载所有主体，并生成价值/控制误差和轨迹图。

**续训：** 在评估命令末尾添加 `--resume`，从整组权重继续训练，输出写入 run 的 `resume/`。不恢复完整优化器/RNG 状态。

**输出：** `output/Multi-CARA/<训练规模>/<主要参数>/`，包含参数、源码、各主体权重、Top-k 元数据、误差历史和 PDF。

当 `alpha=0` 时，固定方差版本可运行 `python3 scripts/Multi-CARA/Multi-player-CARA_fixed_std.py --fixed_std 0.1 --output-dir output/paper/cara_fixed_std`，将探索标准差固定为 0.1。

## 6. Citation

如果本仓库的代码或实验对你的研究有帮助，请引用对应论文：

Liya Guo, Ruimeng Hu, Xu Yang, and Yi Zhu. *A Continuous-Time Actor-Critic Framework for Jump-Diffusion Controls with Normalizing Flows*. arXiv:2604.05398, 2026. [arXiv 论文及引用信息](https://arxiv.org/abs/2604.05398)。

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
