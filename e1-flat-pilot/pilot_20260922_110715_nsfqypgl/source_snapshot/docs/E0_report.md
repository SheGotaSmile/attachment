# E0 运行时验收报告

2026-09-22 E1-flat pilot 前复核：本报告依据下述实际归档，所需 E0 验收字段已齐全。用户给出的 `e0/acceptance_20260917_6Po7jf/` 在本仓库实际位于 `logs/e0/acceptance_20260917_6Po7jf/`。E1 不加载本报告列出的任何 E0 checkpoint。

日期：2026-09-17（Asia/Shanghai）。最终结论：**PASS**。原始 Flat G1 已在真实 CUDA 仿真中完成 2 次 PPO 更新，独立进程恢复本次精确 checkpoint 后又完成 1 次更新。最终验收链训练量为 **192 + 96 = 288 transitions**；这只证明链路可运行，不证明学会行走。[两次更新][train-audit]、[独立恢复与续训][resume-audit]、[reset 实测][reset-log]。

依据原有 [事实核查](/home/xiexuhui/InstinctLab/docs/repo_fact_check.md:1) 和 [实验协议](/home/xiexuhui/InstinctLab/docs/experiment_protocol_v0.1.md:17)，以及本轮新增的运行授权。原文四份用户文档未改写；旧文档中“E0 未运行”的历史状态由本报告补充。未修改 Flat 的算法配置、奖励、观测或执行器，未实现坡地、MoE、相机或激光雷达。实际改动和原文件 hash 核验见 [运行清单][manifest]、第 7 节。

## 1. 产物与状态总表

持久化根目录：`/home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf`。下文 `R` 仅为该目录的简写，不表示自动寻找最近运行。

| 项目 | 最终状态 | 实际结果及证据 |
| --- | --- | --- |
| 三仓库、依赖、GPU、命令记录 | PASS | 完整 SHA、dirty 状态、包路径、CUDA 分配成功；[manifest:10][manifest-repos]、[manifest:93][manifest-packages]、[manifest:366][manifest-gpu] |
| 实际任务注册 | PASS | 启动 AppLauncher 后列出 14 个 Instinct 任务，含 Flat 与 Flat-Play；[registration.log:54][registration] |
| 真实 Flat 仿真 step | PASS | `cuda:0`，48 个训练向量 step；[train.log:60][train-runtime-log]、[training audit:3][train-audit] |
| 两次 PPO update | PASS | update 0、1 实际结束，rollout=24、N=4；192 transitions；[train.log:298][train-update0]、[train.log:336][train-update1] |
| 观测、动作、reward、loss、参数有限性 | PASS | 每次真实 step 检查 obs/action/reward；40 次 minibatch loss 检查、40 次 gradient_step 后参数检查；每次 update 检查模型/optimizer/normalizer；[诊断实现:123](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_training_audit.py:123)、[audit:30][train-finite] |
| runtime shape、term、joint、动作尺度、时序 | PASS | `(4,96)`、`(4,99)`、`(4,29)`；0.005/0.02 s、decimation=4，与预期一致；[audit:69][obs-terms]、[audit:397][timing] |
| 部分环境 reset | PASS | 仅 reset `[0,2]`，`[1,3]` 状态保留且下一 step 继续推进；[reset.json:23][partial-reset]、[reset.json:1453][reset-progress] |
| 动作历史与 episode 初始化 | PASS | 被 reset 环境的 action、prev_action、term raw action、episode length 及 actions 观测归零；[reset.json:1443][reset-init] |
| terminated / truncated / wrapper done | PASS | 实际 timeout 和真实接触分支均触发，6 个诊断 step 均满足 `done = terminated OR truncated`；[reset.json:1484][timeout]、[reset.json:2406][contact] |
| reset 前终止状态与新观测 | PASS | reset hook 捕获旧 episode；返回观测属于新 episode；[reset.json:1555][terminal-observation] |
| 配置、日志与训练状态保存 | PASS | 两份 YAML、TensorBoard 非空且有限、模型/optimizer/两个 normalizer/iter 均保存；[文件复核][validation]、[checkpoint 记录:32][checkpoint2] |
| 独立 checkpoint 往返 | PASS | 同一固定 batch 确定性动作最大误差 0.0；状态逐项匹配；[resume audit:411][roundtrip] |
| 从 checkpoint 再更新并保存 | PASS | update 2 完成，新增 96 transitions；`model_3.pt`，参数实际发生变化；[resume.log:302][resume-update]、[文件复核][validation] |
| 256 环境、长时训练、E1-flat 正式训练 | NOT_RUN | 本轮实际命令均为 4 环境 E0；[完整命令][commands] |
| GUI/视频、play.py 独立回放、物理/RNG 逐 bit 续跑 | NOT_RUN | 本轮独立加载通过训练入口完成，不声称 play.py 或跨平台逐 bit 验收；[完整命令][commands] |
| 坡地、MoE、新传感器、真实机器人部署 | NOT_RUN | 不属于本轮范围；[完整命令][commands] |

最终必需项没有遗留 FAIL/BLOCKED。过程中出现的失败和局部修复没有隐藏，见第 7 节。`commands.sh` 为命令档案，不是直接批量重跑脚本；重跑须新建目录，不能覆盖本次产物。

## 2. 实际运行环境

| 仓库绝对路径 | commit | 初始状态 / 结束状态 |
| --- | --- | --- |
| `/home/xiexuhui/InstinctLab` | `ba28d3d2655b15a19b729476a630937a19610a3b` | 初始无 tracked 改动，4 份用户文档 untracked；结束有本轮 train.py 修改及诊断脚本/报告。原用户文件 hash 全部不变 |
| `/home/xiexuhui/instinct_rl` | `ba45ed231ebbf0a4099cd31d607e2886814fd165` | clean / clean |
| `/home/xiexuhui/IsaacLab-Instinct` | `f73c33173801f5f8afea4142482e47b7710c2b75` | clean / clean |

依据：[manifest repositories:10][manifest-repos]；最终状态和初始用户文件 hash 位于同文件 `repositories_final`、`initial_worktree_before_diagnostics`、[user_files_unchanged:918][user-files]。注意 `repositories` 的状态采集发生在创建最早诊断脚本之后，不能把其中的诊断脚本误说成已有用户修改。[manifest][manifest]

| 组件 | 实际版本 / 路径 |
| --- | --- |
| Python | 3.11.15；`/home/xiexuhui/miniconda3/envs/instinctlab/bin/python` |
| PyTorch | 2.7.0+cu128，CUDA runtime 12.8；`/home/xiexuhui/miniconda3/envs/instinctlab/lib/python3.11/site-packages/torch/__init__.py` |
| Isaac Sim | 包元数据 5.1.0.0；`/home/xiexuhui/miniconda3/envs/instinctlab/lib/python3.11/site-packages/isaacsim/__init__.py` |
| Isaac Lab | 包元数据 0.54.4；`/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/__init__.py` |
| instinct_rl | 1.0.2；`/home/xiexuhui/instinct_rl/instinct_rl/__init__.py` |
| instinctlab | 0.1.0；`/home/xiexuhui/InstinctLab/source/instinctlab/instinctlab/__init__.py` |
| GPU / driver | NVIDIA GeForce RTX 5070 Laptop GPU / 580.173.02，实际 `cuda:0` |
| 显存 | 采集时总 8151 MiB、空闲 7545 MiB；不是训练峰值显存 |

依据：[Python:5][manifest-python]、[包版本与来源:93][manifest-packages]、[GPU/CUDA:366][manifest-gpu]。使用上述专用 Python，未升级依赖主版本；GPU 仿真在允许访问宿主 GPU 的执行环境运行，没有以 CPU 网络前向替代仿真。训练沿用 TF32=True、cuDNN deterministic=False、benchmark=False，不承诺 physics/RNG 位级可复现：[train.py:107](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:107)。

实际注册项的完整列表见 [registration.log:54][registration] 和 [training audit:51][registered-tasks]。只实例化了 Flat，不把其他任务“注册成功”等同于“可运行”。

## 3. 最小训练与张量

最终训练命令完整保存在 [train_2_updates_final.log:1][train-command]；续训命令保存在 [checkpoint_restore_and_update_final.log:1][resume-command]。两次均为原始 `Instinct-Locomotion-Flat-G1-v0`，`--num_envs 4 --seed 42 --headless --device cuda:0`，仅 update 预算、日志路径和诊断开关变化。实际导出的 [agent.yaml:1][agent-config] 确认标准 ActorCritic + PPO/AdamW、ELU、actor/critic hidden `[256,128,128]`、24 steps、5 epochs、4 minibatches，未换算法。

| 运行 | 实际向量 step | update index | transitions | 最终 iter | total_loss |
| --- | ---: | --- | ---: | ---: | --- |
| 最终初训 | 48 | 0、1 | `4*24*2=192` | 2 | 7.1062941551、4.7180690765 |
| 最终独立续训 | 24 | 2 | `4*24*1=96` | 3 | 6.0596957207 |

依据：[初训 audit:3][train-audit]、[续训 audit:3][resume-audit]。5*4=20 个 minibatch/update，初训/续训分别记录 40/20 次 loss 与更新后参数有限性检查。这里没有声称逐个读取 `.grad` 张量。独立 reset 诊断另有 6 个向量 step，即 24 个环境 transitions，不计入 PPO 训练预算。[reset.json:2433][done-merge]

修复期间另完成了一组 2+1 更新；因此本轮包含重跑的全部 PPO 工作量是 **576 transitions**，最终验收选中的连续模型链仍是 **288 transitions**，不能将重跑相加解释为一个训练模型的进度。[manifest commands:120][manifest-commands]

| 张量 / 时序 | 实际值 | 来源 |
| --- | --- | --- |
| policy observation | `(4,96)` | [audit:408][shapes] |
| critic observation | `(4,99)` | [audit:408][shapes] |
| action / num_actions | `(4,29)` / 29 | [audit:67][obs-terms]、[audit:408][shapes] |
| physics_dt | 0.005 s，即 200 Hz | [audit:397][timing] |
| step_dt / decimation | 0.02 s，即 50 Hz / 4 | [audit:397][timing] |
| 执行器 | legs、feet、waist、waist_yaw、arms 均为 ImplicitActuator | [audit:401][actuators] |

观测拼接顺序（括号为维数）：policy = `base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos(29), joint_vel(29), actions(29)`；critic 在同序列最前加 `base_lin_vel(3)`。直接读取运行时 wrapper 格式，未 reshape、删项或修改网络掩盖差异。[audit:69][obs-terms]

下表是 action term **实际解析后的顺序**，不是按 URDF 文本推测；四个环境的 scale/offset 相同。表中数值为便于阅读的舍入值，完整浮点数组分别见 [audit:145][scales]、[audit:271][offsets]。

| index | joint | scale | offset (rad) |
| ---: | --- | ---: | ---: |
| 0 | left_shoulder_pitch_joint | 0.438577324 | 0.2 |
| 1 | right_shoulder_pitch_joint | 0.438577324 | 0.2 |
| 2 | waist_pitch_joint | 0.438577324 | 0 |
| 3 | left_shoulder_roll_joint | 0.438577324 | 0.2 |
| 4 | right_shoulder_roll_joint | 0.438577324 | -0.2 |
| 5 | waist_roll_joint | 0.438577324 | 0 |
| 6 | left_shoulder_yaw_joint | 0.438577324 | 0 |
| 7 | right_shoulder_yaw_joint | 0.438577324 | 0 |
| 8 | waist_yaw_joint | 0.547546446 | 0 |
| 9 | left_elbow_joint | 0.438577324 | 0.6 |
| 10 | right_elbow_joint | 0.438577324 | 0.6 |
| 11 | left_hip_pitch_joint | 0.547546446 | -0.312 |
| 12 | right_hip_pitch_joint | 0.547546446 | -0.312 |
| 13 | left_wrist_roll_joint | 0.438577324 | 0 |
| 14 | right_wrist_roll_joint | 0.438577324 | 0 |
| 15 | left_hip_roll_joint | 0.350661457 | 0 |
| 16 | right_hip_roll_joint | 0.350661457 | 0 |
| 17 | left_wrist_pitch_joint | 0.074500874 | 0 |
| 18 | right_wrist_pitch_joint | 0.074500874 | 0 |
| 19 | left_hip_yaw_joint | 0.547546446 | 0 |
| 20 | right_hip_yaw_joint | 0.547546446 | 0 |
| 21 | left_wrist_yaw_joint | 0.074500874 | 0 |
| 22 | right_wrist_yaw_joint | 0.074500874 | 0 |
| 23 | left_knee_joint | 0.350661457 | 0.669 |
| 24 | right_knee_joint | 0.350661457 | 0.669 |
| 25 | left_ankle_pitch_joint | 0.438577324 | -0.363 |
| 26 | right_ankle_pitch_joint | 0.438577324 | -0.363 |
| 27 | left_ankle_roll_joint | 0.438577324 | 0 |
| 28 | right_ankle_roll_joint | 0.438577324 | 0 |

动作含义：`joint_position_target = raw_action * scale + offset`，动作不是直接力矩；当前 action clip 为 null。动作目标送入原有 ImplicitActuator/PhysX 控制链，不代表力矩或物理关节限制不存在。[joint_actions.py:169](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:169)、[apply_actions:199](/home/xiexuhui/IsaacLab-Instinct/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:199)、[reset runtime][reset-runtime]。

## 4. 部分 reset 与终止语义

诊断独立启动仿真，无人工 debugger，自动结束。命令见 [reset_probe.log:1][reset-command]；实现见 [e0_reset_probe.py:30](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_reset_probe.py:30)。

1. 用 0.02 的固定 action warmup 3 步，再 `reset(env_ids=[0,2])`。其他环境 `[1,3]` 的 root/joint position/joint velocity/action/prev_action/term raw action/episode_length 完全不变；被 reset 环境的动作历史与计数清零。episode 计数为 `[0,3,0,3]`，下一真实 step 为 `[1,4,1,4]`，另两个环境的 root_state 最大分量变化分别为 0.1879713 和 0.2560201，证明继续运行。[partial reset:23][partial-reset]、[初始化:1443][reset-init]、[继续推进:1453][reset-progress]
2. **人工 timeout 验收**：单独全 reset 后将 env0 的 `episode_length_buf` 设为 `max_episode_length-1=999`，随后执行一次真实 step。没有改变原始 20 s episode 配置。raw terminated=`[F,F,F,F]`、truncated=`[T,F,F,F]`、wrapper done=`[1,0,0,0]`，`extras.time_outs` 同 truncated；episode 变为 `[0,1,1,1]`。这是真实触发 timeout 分支，不是“运行时间足够长”的推断。[timeout:1484][timeout]、[构造代码:147](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_reset_probe.py:147)
3. **人工接触验收**：env1 root 放在离地 0.10 m、绕世界 Y 轴 +90 度、速度为零的位置，保留原 illegal_contact term。下一真实物理 step 即触发 terminated=`[F,T,F,F]`、truncated 全 F、done=`[0,1,0,0]`；这验证接触终止，不评价策略稳定性。[contact:2406][contact]、[构造代码:173](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_reset_probe.py:173)
4. reset hook 在 `_reset_idx` 前保存旧 episode 计数、状态、critic 与接触力；timeout 前计数 1000、action 为 0.02，返回观测中的 critic actions 已为零且 episode=0。因此终止 flags/reward 对应刚结束的 transition，而返回 observation 对应新 episode；不能把后者当 terminal observation。诊断临时 hook 已在退出时恢复。[terminal/new obs:1555][terminal-observation]、[hook:68](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_reset_probe.py:68)

6 个诊断 step 的原始 flags 与 wrapper 合并结果全部一致，obs/action/reward 均有限。[reset.json:2433][done-merge]。自然存活完整 1000 步再 timeout 为 NOT_RUN，但不影响上述明确构造的 timeout 分支验收。

## 5. Checkpoint 往返

最终选定的精确目录：

```text
初训：/home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed
续训：/home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/resume/20260917_155746_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_resume_s42_logs_fixed_from20260917_155259
```

| 文件 | SHA-256 |
| --- | --- |
| [初训 model_2.pt][model2] | `8071b7fcb85a85f4b177450b563f915f84d388714eb65868dbfb374321db928c` |
| [续训 model_3.pt][model3] | `22c11f4c48e8d959dd046dd1cc1eb8cfc313d87e507a66c32b192d49bafbf7e9` |

完整路径/hash/keys/iter 也分别写入 [初训 audit:32][checkpoint2]、[续训 audit:20][checkpoint3] 与 [manifest][manifest]。没有重命名产物，没有选择其他实验的 latest。

保存前固定输入来自本次真实环境的一个 policy observation batch `(4,96)`；在模型和 normalizer 的 eval 模式下计算确定性均值动作 `(4,29)`，原训练模式随即恢复，作为 `model_2.pt.reference.pt` 保存。独立新进程先按精确 run 路径和 `--checkpoint '^model_2[.]pt$'` 加载，然后使用完全同一输入比较。[audit 实现:73](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_training_audit.py:73)、[保存:165](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_training_audit.py:165)、[实际加载日志:295][resume-load]

结果：最大绝对误差 **0.0**，容差 `atol=1e-6, rtol=1e-5`；模型 state_dict、optimizer state_dict、policy/critic normalizer 的每个状态和 iteration 均一致。比较发生在同一 `cuda:0` 和相同数值模式下；不保证换 CPU 或 TF32 模式后位级相同。[resume audit:411][roundtrip]

初训 checkpoint：`iter=2`、optimizer 17 个参数状态均为 step=40、两个 normalizer count=192。恢复并更新一次后：`iter=3`、optimizer step=60、normalizer count=288，模型参数实际发生变化。[manifest checkpoint_artifacts:864][checkpoint-artifacts]、[参数变化:374][model-change]。恢复时 adaptive learning_rate 标量与 optimizer LR 均为 `2.2500000000000008e-5`。[roundtrip:411][roundtrip]

**迭代命名语义**：runner 先更新，周期保存发生在 iteration 自增之前，而循环结束的最终保存发生在自增之后。本轮运行的是 update 0/1，结束最终保存 model_2/iter2；从 iter2 追加一次运行 update2，最终保存 model_3/iter3。不能泛化成“任意周期文件 model_N 一定完成 N 次更新”。[runner:151](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:151)、[周期/最终保存:199](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:199)

## 6. 保存内容与完整命令

`R/runtime_manifest.json` 记录环境、三个 SHA/dirty 状态、精确 argv、工作目录、起止时间、返回码、timeout、日志路径、checkpoint hash、最终核验、未运行项和源码快照。`R/commands.sh` 收录本次执行命令；每份运行日志第 1 行也有实际完整命令。[manifest][manifest]、[命令档案][commands]

每个最终训练目录都有 `params/env.yaml`、`params/agent.yaml`、`git/*.diff`、`events.out.tfevents.*`、`e0_training_audit.json`、checkpoint 和固定输入 reference。初训/续训的 env.yaml SHA-256 相同：`3419d4896f6cbd8dd3e5128ebc11b044891170b7c82b2fdd1cefec3e886fe267`。[配置/事件文件复核][validation]

最终 TensorBoard 实际读取到初训 `E0/Loss/*` 在 transitions 96、192 的四个 loss 标量，续训在 288 的四个标量，全部有限。初训还包含原 runner 的 reward、episode、学习率和性能日志；原 `log_interval=10` 保持不变，因此仅一次的续训不会自然产生原 runner 的常规日志点，E0 的每次 update loss 由 opt-in 诊断记录。[文件复核][validation]、[agent.yaml:35][agent-config]、[audit:157](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_training_audit.py:157)

`R/simulator_logs/<label>/` 复制各实际进程的 Kit/IsaacLab 原始日志，包括原本写到 `/tmp/isaaclab/logs` 的文件；`R/source_snapshot/` 保存本轮脚本和本报告及其 hash。唯一验收产物未留在 `/tmp`。精确原路径、复制路径和 hash 见 manifest 的 `simulator_log_copies`、`source_snapshot`。[manifest][manifest]

## 7. 失败、局部修复与修改文件

| 发现 | 处理与范围 | 最终复验 |
| --- | --- | --- |
| 首次独立加载 FAIL：`model_2.pt` 被当作前缀正则，误匹配本轮诊断 sidecar `model_2.pt.reference.pt`，产生 KeyError | 本轮增加 reference 后暴露的匹配问题。改用精确 run 路径及锚定正则 `^model_2[.]pt$`；未改外部加载算法或重命名 checkpoint | [失败 log:295][load-failure]；[最终 log:295][resume-load]，PASS |
| adaptive PPO 恢复不完整：optimizer LR 恢复了，但算法 `learning_rate` 标量仍从配置初始化为 0.001 | `train.py` 在 resume load 后同步为 optimizer 已恢复的共同 LR。外部 PPO load 只加载 state_dict，后续 KL schedule 会用该标量覆写 LR，因此该修复是状态恢复修复，不是换算法/调参 | [PPO load:320](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/ppo.py:320)、[KL LR:252](/home/xiexuhui/instinct_rl/instinct_rl/algorithms/ppo.py:252)、[本地修复:213](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:213)、[最终恢复:296][lr-restore] |
| 前一组短训练/续训进程退出成功，但 TensorBoard 实际 scalar tags 为空，日志验收 FAIL | `train.py` 在 finally 中 flush/close writer；E0 诊断单独记录每次 update loss，保留默认 log_interval。随后重跑两次训练及独立一次续训 | [空文件复核:364][empty-scalars]、[最终非空复核][validation]、[修复:254](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:254)，最终 PASS |
| 原 runner 控制台 Total timesteps 少计 | `tot_timesteps` 只在 `log()` 内累加，本次 log_interval=10 时控制台只显示 96，实际是 192。未改外部 runner；用实际 step 数和 rollout 计算预算 | [train.log:331][counter-log]、[runner:235](/home/xiexuhui/instinct_rl/instinct_rl/runners/on_policy_runner.py:235)、[audit:424][train-transitions] |

外部库未修改，所以直接绕过本地训练入口调用旧 `runner.load()` 的 adaptive LR 问题仍需调用方处理；默认宽松 checkpoint 匹配也未宣称已修复。以上限制已明确记录，不影响本轮精确入口验收。

| 本轮新增/修改文件 | 原因 |
| --- | --- |
| [scripts/instinct_rl/train.py:47](/home/xiexuhui/InstinctLab/scripts/instinct_rl/train.py:47) | 增加 opt-in 验收入口；恢复 adaptive LR 标量；退出前刷写/关闭日志 |
| [scripts/instinct_rl/e0_training_audit.py:48](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_training_audit.py:48) | 实际 step/update 有限性和形状检查、运行时动作语义、checkpoint 固定输入往返、E0 loss 记录 |
| [scripts/instinct_rl/e0_reset_probe.py:30](/home/xiexuhui/InstinctLab/scripts/instinct_rl/e0_reset_probe.py:30) | 有界、独立、无需 debugger 的真实 reset/timeout/contact 验收 |
| [scripts/diagnostics/e0_runtime.py:1](/home/xiexuhui/InstinctLab/scripts/diagnostics/e0_runtime.py:1) | 环境清单、有界运行与完整日志、最终产物/hash/事件文件核验与归档 |
| [docs/E0_report.md:1](/home/xiexuhui/InstinctLab/docs/E0_report.md:1) | 汇总本轮实测、失败、修复、边界和下一阶段建议 |

未删除任何用户文件或失败日志。原有 4 份文档 SHA-256 核验不变；两个依赖仓库保持 clean。[manifest][manifest]

## 8. 未运行项与 E1-flat 建议

E0 必需项均已运行并 PASS，可以进入 **E1-flat 的容量检查和基线学习阶段**；不能据此批准 E1-slope/E2 或认为已经学会走路。未运行：256 环境容量、长时/多 seed 学习、GUI/视频、play.py 回放、真实部署、独立性能 benchmark、物理/RNG 位级续跑，以及坡地/MoE/新传感器。未自然等到 1000 步 timeout；人工触发的真实 timeout 已通过。依据为 [本次完整命令][commands] 和上表逐项结果，而不是未训练策略跌倒次数。

**以下是建议命令，NOT_RUN，本轮未启动。** 下次仍使用专用 Python 和原始 Flat，从头开始，不续训 E0 的 3-update 模型，不加 `--e0_audit`（该诊断特意限定 N=4）。先用 256 环境、seed42、100 updates 做容量/早期稳定性检查，预算 `256*24*100=614,400` transitions；8 GiB GPU 的 256 环境尚未实测，不能保证不会 OOM。[原 rollout 配置][agent-config]、[本轮 GPU][manifest-gpu]

```bash
cd /home/xiexuhui/InstinctLab
e1_flat_run=$(mktemp -d /home/xiexuhui/InstinctLab/logs/e1-flat-capacity-s42.XXXXXX)
/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-v0 \
  --num_envs 256 --seed 42 --headless --device cuda:0 \
  --max_iterations 100 --logroot "$e1_flat_run" --run_name E1_flat_capacity_s42
```

容量检查通过后，建议第一阶段从头预算 **1000 updates = 6,144,000 transitions，seed42**，观察 loss/normalizer 有限性、终止构成、速度跟踪和奖励趋势后再决定是否扩大预算与 seed43/44。每 seed 的 1000-update 预算独立计算；三 seed 为 18,432,000 transitions，不包含容量试跑。1000 updates 是学习诊断起点，不保证收敛，不能将本次 4 环境的耗时线性外推到 256 环境。

```bash
e1_flat_run=$(mktemp -d /home/xiexuhui/InstinctLab/logs/e1-flat-s42.XXXXXX)
/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-v0 \
  --num_envs 256 --seed 42 --headless --device cuda:0 \
  --max_iterations 1000 --logroot "$e1_flat_run" --run_name E1_flat_s42
```

下一阶段应继续保存精确 manifest/日志/checkpoint；不要以 runner 控制台 Total timesteps 作为预算依据。若容量检查 OOM，停止并明确记录、重新冻结环境数和 transitions 预算，不改算法或执行器来掩盖资源问题。研究协议的 20M 比较预算属于后续阶段，本次没有启动。[协议预算](/home/xiexuhui/InstinctLab/docs/experiment_protocol_v0.1.md:163)

[manifest]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/runtime_manifest.json:1
[manifest-python]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/runtime_manifest.json:5
[manifest-repos]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/runtime_manifest.json:10
[manifest-packages]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/runtime_manifest.json:93
[manifest-commands]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/runtime_manifest.json:120
[manifest-gpu]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/runtime_manifest.json:366
[checkpoint-artifacts]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/runtime_manifest.json:864
[user-files]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/runtime_manifest.json:918
[commands]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/commands.sh:1
[validation]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/artifact_validation.json:1
[empty-scalars]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/artifact_validation.json:364
[model-change]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/artifact_validation.json:374
[registration]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/registration.log:54
[train-command]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train_2_updates_final.log:1
[train-runtime-log]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train_2_updates_final.log:60
[train-update0]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train_2_updates_final.log:298
[train-update1]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train_2_updates_final.log:336
[counter-log]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train_2_updates_final.log:331
[resume-command]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/checkpoint_restore_and_update_final.log:1
[resume-load]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/checkpoint_restore_and_update_final.log:295
[lr-restore]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/checkpoint_restore_and_update_final.log:296
[resume-update]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/checkpoint_restore_and_update_final.log:302
[load-failure]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/checkpoint_restore_and_update.log:295
[train-audit]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:2
[train-finite]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:30
[checkpoint2]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:32
[registered-tasks]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:51
[obs-terms]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:69
[scales]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:145
[offsets]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:271
[timing]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:397
[actuators]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:401
[shapes]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:408
[train-transitions]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/e0_training_audit.json:424
[agent-config]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/params/agent.yaml:1
[resume-audit]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/resume/20260917_155746_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_resume_s42_logs_fixed_from20260917_155259/e0_training_audit.json:2
[checkpoint3]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/resume/20260917_155746_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_resume_s42_logs_fixed_from20260917_155259/e0_training_audit.json:20
[roundtrip]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/resume/20260917_155746_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_resume_s42_logs_fixed_from20260917_155259/e0_training_audit.json:411
[model2]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/model_2.pt
[model3]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/resume/20260917_155746_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_resume_s42_logs_fixed_from20260917_155259/model_3.pt
[reset-command]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe.log:1
[reset-log]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe.log:273
[partial-reset]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe/reset_results.json:23
[reset-init]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe/reset_results.json:1443
[reset-progress]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe/reset_results.json:1453
[timeout]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe/reset_results.json:1484
[terminal-observation]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe/reset_results.json:1555
[contact]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe/reset_results.json:2406
[done-merge]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe/reset_results.json:2433
[reset-runtime]: /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe/reset_results.json:11904
