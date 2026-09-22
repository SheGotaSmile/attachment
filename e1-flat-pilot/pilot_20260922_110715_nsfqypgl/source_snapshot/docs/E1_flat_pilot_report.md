# E1-flat pilot 报告

日期：2026-09-22（Asia/Shanghai）。结论：**pilot 未通过；Flat evaluator 通过；暂不建议进入正式 E1-flat。**

200 iterations 正常结束，数值有限，checkpoint 独立加载成功。但是速度指标改善与更早的接触终止同时出现，尚不能确认普通 S-small 从随机初始化获得了合理的行走学习进展。200 iterations 仅为 pilot，不是最终性能、收敛或能力上限结论。

## 1. 运行与任务定义

归档根目录 `R`：

```text
/home/xiexuhui/InstinctLab/logs/e1-flat-pilot/pilot_20260922_110715_nsfqypgl
```

实际训练目录 `T`：

```text
/home/xiexuhui/InstinctLab/logs/e1-flat-pilot/pilot_20260922_110715_nsfqypgl/train/20260922_110723_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E1_flat_pilot_s42
```

唯一一次训练使用原始 `Instinct-Locomotion-Flat-G1-v0`，S-small ActorCritic（actor `96→256→128→128→29`，critic `99→256→128→128→1`，ELU），原始 Flat reward、observation、implicit actuators、PPO/AdamW 和两个 EmpiricalNormalization。seed=42，64 envs，200 iterations，headless，cuda:0。没有 OOM，也没有降到 32。

运行时 policy `(64,96)`、critic `(64,99)`、action `(64,29)`；physics_dt=0.005 s、step_dt=0.02 s、decimation=4。rollout_length 实测 24，执行 4,800 个向量 step，update index 0–199，最终 checkpoint iter=200。

```text
actual_transitions = 64 × 24 × 200 = 307,200
```

只增加 opt-in `--e1_audit` 观测/记录、归档入口和 evaluator；没有修改原始 Flat 配置。用户指定的 seed、env 数、预算与日志位置通过现有 CLI 设置。审计读取实际张量并在原函数调用后记录统计，不改变动作、reward、梯度或 optimizer 运算。`source/` 下 204 个 Python 文件的运行前后 SHA-256 全部一致，两个外部仓库保持 clean。

| 仓库 | commit | 状态 |
| --- | --- | --- |
| InstinctLab | `ba28d3d2655b15a19b729476a630937a19610a3b` | dirty；保留已有 E0 train.py 改动、文档及诊断脚本，本轮新增 E1 审计/evaluator/报告 |
| instinct_rl | `ba45ed231ebbf0a4099cd31d607e2886814fd165` | clean |
| IsaacLab-Instinct | `f73c33173801f5f8afea4142482e47b7710c2b75` | clean |

完整初始 git 状态见 `R/manifest.json`，结束状态见 `R/repositories_final.json`，代码/文档与 tracked diff 见 `R/source_snapshot/`、`R/git_diff.patch`。diff 相对 HEAD，包含原本未提交的 E0 训练入口改动；不能把全部 diff 都归因于本轮 E1。未修改三份研究协议文档或根目录旧 `Research State.md`。

## 2. 完整命令与随机初始化证据

实际训练命令：

```bash
/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-v0 \
  --num_envs 64 --seed 42 --max_iterations 200 \
  --headless --device cuda:0 \
  --logroot /home/xiexuhui/InstinctLab/logs/e1-flat-pilot/pilot_20260922_110715_nsfqypgl/train \
  --run_name E1_flat_pilot_s42 --e1_audit
```

外层归档启动命令为：

```bash
/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/diagnostics/e1_pilot_run.py
```

最终策略独立评估命令：

```bash
/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/evaluate_flat.py \
  --checkpoint /home/xiexuhui/InstinctLab/logs/e1-flat-pilot/pilot_20260922_110715_nsfqypgl/train/20260922_110723_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E1_flat_pilot_s42/model_200.pt \
  --output-dir /home/xiexuhui/InstinctLab/logs/e1-flat-pilot/pilot_20260922_110715_nsfqypgl/evaluation_v2 \
  --num-envs 8 --episodes-per-env 4 --max-steps 5000 \
  --seed 12345 --command-mode original --headless --device cuda:0
```

本次还在新进程以相同 evaluator 参数加载本次 `model_initial.pt`，输出到 `R/evaluation_initial/`，用于诊断趋势；没有追加训练。该命令和首次失败的 evaluator 命令均完整保存于 `R/commands.sh`，并非可覆盖已有产物的批量重跑脚本。

随机初始化证据：`T/params/agent.yaml` 的 `resume: false`、`load_run: null`、`ckpt_manipulator: null`；audit 起始 iteration=0、optimizer state 为空。训练前保存本次随机网络 `model_initial.pt`：iter=0，两个 normalizer count=0；训练后 iter=200，optimizer step=4000，两个 normalizer count=307200。没有任何 E0 checkpoint 加载或 resume 命令。

| Checkpoint（均位于 T） | SHA-256 |
| --- | --- |
| `model_initial.pt`（本次随机初始模型） | `1b311022af5900ad4b56383b5be48be0dca2738329ba38090cdc9921e43e08ea` |
| `model_200.pt`（200 次更新后的模型） | `4df39669c75d9c784a414f2c64e5ea98255a561fb2d54677db1ee05786617626` |

## 3. 训练趋势与数值检查

以下原始物理指标在每步 reward 计算后、auto-reset 和 command 更新前采集。首/末列是 update 0/199；窗口列是 update 0–19/180–199 的均值。episode return 为截至该 update 的最近 100 个完整 episode 均值，首期不足 100 时使用实际已完成数。

| 指标 | 首次 update | 最后 update | 前 20 均值 | 后 20 均值 |
| --- | ---: | ---: | ---: | ---: |
| Mean episode reward / return | -4.625756 | -4.173834 | -5.138427 | -4.182996 |
| Mean reward / transition（含终止步） | -0.313000 | -0.791056 | -0.311833 | -0.744255 |
| Velocity tracking exp score | 0.306643 | 0.397865 | 0.282079 | 0.363133 |
| Base XY velocity L2 error（m/s） | 0.655415 | 0.538101 | 0.731147 | 0.579637 |
| Terminated / transition | 0.063151 | 0.190104 | 0.061068 | 0.177995 |
| Truncated / transition | 0 | 0 | 0 | 0 |
| Gaussian action std 均值 | 1.000522 | 0.792520 | 0.996427 | 0.800348 |

原 runner 的 TensorBoard 日志每 10 次更新记录一次，最后常规点是 190；`E1/*` 审计记录到 199，共 200 点。常规 `Train/mean_reward_0` 从 -4.625756 到 -4.172053（index 190）；`Train/mean_episode_length` 从 10.0 到 5.57 步。原 `Episode_Reward/track_lin_vel_xy_exp` 从 0.002600 到 0.002056；`Episode/Metrics/base_velocity/error_vel_xy` 从 0.011149 到 0.007278。后两者来自环境 episode 日志的归一化统计，随 episode 长度改变，不能直接当作上表的逐步原始 tracking/error。

数值改善不能忽略伴随退化：tracking 与原始 error 来自同一速度误差，episode reward 上升也伴随 episode 明显缩短。后 20 次更新中每步接触终止率约是前 20 次的 2.91 倍；含终止惩罚的每步 reward 变差。std 下降本身不是学习成功证据。因此这里不把两个变好的数值自动计为“至少两个合理学习改善”。这是对验收第 4 条的保守判定，不是宣称原始 Flat 任务无法在更长预算中学习。

![训练趋势](../logs/e1-flat-pilot/pilot_20260922_110715_nsfqypgl/learning_trends.png)

4,800 次实际 step 的观测、动作、reward、测量值有限；4,000 次 minibatch loss/中间量检查通过；4,000 次 optimizer 执行前真实 `.grad` 张量及 optimizer LR 检查通过；每次 update 的 gradient norm、loss、模型/optimizer、两个 normalizer 状态有限。checkpoint 全部张量及 60 个 TensorBoard scalar tags 均有限。adaptive learning rate 在首/末 update 后均约 1e-5；保持原算法，不修改 LR。

没有检测到训练数值 NaN/Inf。控制台中存在 runner 固定打印的 `NOTE: you may see ... NaN or Inf ...` 提示文本，它不是本次发现 NaN/Inf 的事件；没有以该提示为理由忽略实际数值异常。

## 4. 独立 evaluator 验收

源码：[evaluate_flat.py](../scripts/instinct_rl/evaluate_flat.py)，共享测量函数：[flat_metrics.py](../scripts/instinct_rl/flat_metrics.py)，说明：[E1_flat_evaluator.md](E1_flat_evaluator.md)。

最终 checkpoint 在独立进程 PID 602529 加载；随机初始化 checkpoint 在独立进程 PID 613357 加载。二者各 8 env × 每 env 4 episode=32 条；固定 evaluator seed=12345，使用原始 Flat command 分布。训练进程与评估进程不同。

模型、policy normalizer、critic normalizer 均与加载的 checkpoint 状态逐项一致，评估结束后三者状态完全未变。Gaussian mean action 与 actor MLP 直接输出最大差=0；全程不采样、不执行网络更新。异步 reset 实际出现；episode length 与 reset 前 buffer 一致；最终模型的 episode return 与 reward manager 各项累计和最大差 `7.15e-7`。JSON 与 CSV 的 32 行所有字段一致。

| 独立 mean-action 评估（各 32 episodes） | 随机初始模型 | 200-update 模型 |
| --- | ---: | ---: |
| Episode return | -3.944402 | -3.933082 |
| Episode length（步） | 14.5000 | 5.71875 |
| Episode length（秒） | 0.2900 | 0.114375 |
| XY velocity error（m/s） | 0.695856 | 0.523909 |
| Tracking exp score | 0.285801 | 0.405880 |
| Base contact / fall | 32/32 | 32/32 |
| Timeout | 0/32 | 0/32 |
| 初始 yaw 前向净位移（m） | 0.059725 | 0.005591 |
| 平均实际 yaw-frame 前向速度（m/s） | 0.016292 | 0.112748 |
| 平均水平速率（m/s） | 0.488494 | 0.413598 |

这些是短期、随机 command 的诊断结果，不是正式性能测试。即使同 evaluator seed，相互不同的终止时间也会改变后续 command/reset RNG 消耗，不能把后续 episode 视为严格相同轨迹配对。净位移为初始朝向投影，不能与变化中的 yaw-frame 平均速度简单互换。原 `base_contact` 包括非足部肢体的 illegal contact，并非只检查躯干。

`terminated`、`truncated`、`timeout` 从 reset 前的原 manager/term 分别读取；若接触与 timeout 同步发生，仍记录 fall=true。本轮 E1 的所有完成 episode 都是接触终止，没有实测自然 timeout 或同一步双 flags。E0 已独立验证底层 timeout/接触分支；不把它冒充 E1 evaluator 自然 timeout 的运行结果。按 quota 结束时残留的未完成 episode 长度保存在 JSON，未作为完整 episode 计入。

首次 evaluator 在仿真启动后读 `env.yaml` 时遇到 `builtins.slice` YAML tag，记录 `R/evaluation/episodes.json` 的 FAIL，未产生 episode。仅修复 evaluator 的 YAML 数据解析后重跑为 `evaluation_v2`；没有更改 checkpoint 或任务配置。Isaac Sim fast shutdown 可能返回 0，即使 evaluator 捕获异常，因此使用 JSON 状态和数据核验作为验收依据。最终只调整了退出前的状态打印顺序，不改变已验收的统计逻辑。

## 5. 显存、吞吐、产物

GPU：NVIDIA GeForce RTX 5070 Laptop GPU，总显存 8151 MiB。`gpu.csv` 每 5 s 采样，共 40 条，整卡已用显存最大采样值 2736 MiB；它包括桌面等其他 GPU 用户，且不是连续监测的绝对峰值。PyTorch allocator 峰值 allocated 25.98 MiB、reserved 32 MiB，不包含 PhysX/Kit 的全部分配。

训练（含审计）181.874 s，307200/181.874 = **1689.08 transitions/s**；完整子进程含启动关闭约 197.146 s。原 runner 控制台 Total timesteps 仅 30720，是 `log_interval=10` 下只在 log() 累加的已知计数问题；实际 307200 由每次真实 step、完整 update 序列和 normalizer count 三方核对。

| 路径（相对 R） | 内容 |
| --- | --- |
| `train/.../params/env.yaml`、`agent.yaml` | 真实训练配置 |
| `train/.../events.out.tfevents.*` | 原 runner + 每 update 审计标量 |
| `train/.../e1_training_audit.json` | 4800 steps、200 updates、4000 loss/gradient 检查 |
| `train/.../model_initial.pt`、`model_200.pt` | 本次初始和最终 checkpoint |
| `evaluation_v2/episodes.json`、`episodes.csv` | 最终 checkpoint 的有效评估结果 |
| `evaluation_initial/episodes.json`、`episodes.csv` | 本次随机模型的独立对照 |
| `evaluation/`、`evaluation.log` | 首次 YAML 解析失败记录，不纳入有效评估 |
| `train.log`、`evaluation_v2.log`、`evaluation_initial.log` | 进程 stdout/stderr |
| `gpu.csv`、`analysis.json`、`tensorboard_scalars.json` | 显存、核验与汇总、完整 scalar 导出 |
| `manifest.json`、`repositories_final.json` | 完整命令、起止/退出码、git 状态、source hashes |
| `commands.sh`、`git_diff.patch`、`source_snapshot/` | 命令档案、代码 diff 和实际源码/文档 |
| `simulator_logs/`、`SHA256SUMS`、`artifact_hashes.json` | 原始 Kit/IsaacLab 日志副本和产物 SHA-256 |

## 6. 验收与停止点

| 用户验收条件 | 结论 |
| --- | --- |
| 200 iterations 正常完成 | PASS |
| 无 NaN/Inf | PASS，按上述实测张量/梯度/日志覆盖范围 |
| Checkpoint 可以独立加载 | PASS，两个 normalizer 同时核验 |
| 至少两个训练指标合理改善 | **未确认**：存在数值改善，但接触终止增多、生存缩短，不能认定有效行走学习 |
| Evaluator 生成逐 episode 结果 | PASS，32 行 JSON/CSV，异步统计 |
| 没有使用 E0 checkpoint | PASS；本次从 random init 开始 |
| 没有扩展坡地/MoE/相机/LiDAR 或研究主线 | PASS；无 history/姿态 reward 变更 |

最终回答：E1-flat pilot **未通过**；evaluator **通过**；**暂不建议**进入正式 E1-flat。本轮完全没有加入坡地、MoE、深度相机或 LiDAR，没有启动正式三 seed 训练。

唯一最小阻塞项是：**尚未证实合理的学习改善，早期 base_contact 终止退化尚未解释。** 下一步仅应诊断原始 Flat 的早期接触终止及其与 reward/动作的关系，查清后再决定同规模 pilot 复验。当前已暂停，没有自动修改配置、扩预算或重训。
