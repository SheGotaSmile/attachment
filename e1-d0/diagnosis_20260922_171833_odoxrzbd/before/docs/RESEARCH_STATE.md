# 研究状态

更新：2026-09-22（Asia/Shanghai）。当前阶段：**E1-flat pilot 已完成运行、未通过学习验收，暂停；evaluator 已通过。** E0 运行时验收保持 PASS。

已完成：

- E0 真实 Isaac Sim/CUDA 仿真、任务注册、真实张量、两次 PPO、partial reset、checkpoint 保存和独立恢复后一次 update。详见 [E0_report.md](E0_report.md)。
- E1 原始 Flat S-small 从随机初始化完成 seed 42、64 env、200 iterations，即 307200 transitions；未使用 E0 checkpoint，无 resume。
- loss/真实梯度/学习率、观测动作奖励、模型和 normalizer 有限性检查通过；checkpoint 独立加载成功。
- Flat evaluator 使用 Gaussian mean action，加载并冻结 policy/critic normalizer；完成异步逐 episode JSON/CSV、原始任务配置一致性及统计交叉核验。
- 本次随机模型与最终模型各 32 episode 的同 seed 原始 command 分布评估；报告、命令、hash、diff、GPU/吞吐及日志归档。

当前结论：速度 tracking/error 数值改善，但每步接触终止率从前 20 updates 的 6.11% 上升到后 20 的 17.80%；独立评估平均生存从 14.5 步降至 5.72 步，最终 32/32 episode 接触终止。无法确认“合理学习改善”，不建议进入正式 E1-flat。详见 [E1_flat_pilot_report.md](E1_flat_pilot_report.md)。

当前唯一下一步：诊断原始 Flat 的早期 `base_contact` 终止退化，查清原因后再决定同规模 pilot 复验。当前没有自动开始诊断改配置、追加训练或正式三 seed 实验。

禁止事项：本阶段不实现坡地、MoE、深度相机、LiDAR、观测历史；不修改姿态奖励；不使用 E0 model_2/model_3 或 E0 resume；不把 200 iterations 写成最终性能；不以 reward 一项认定成功；未经下一阶段指令不启动正式训练。

关键运行目录：

- E0：`logs/e0/acceptance_20260917_6Po7jf/`；最终验收链为 `train/20260917_155259_..._logs_fixed` 与 `resume/20260917_155746_..._logs_fixed_from20260917_155259`，精确路径见 E0 报告。
- E1：`logs/e1-flat-pilot/pilot_20260922_110715_nsfqypgl/`。
- E1 训练：`logs/e1-flat-pilot/pilot_20260922_110715_nsfqypgl/train/20260922_110723_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E1_flat_pilot_s42/`。
- 有效最终评估：E1 归档下 `evaluation_v2/`；随机模型对照 `evaluation_initial/`；最初 `evaluation/` 是 YAML 解析失败记录。

关键 checkpoint：

- E1 训练目录 `model_200.pt`：SHA-256 `4df39669c75d9c784a414f2c64e5ea98255a561fb2d54677db1ee05786617626`。
- E1 同目录 `model_initial.pt`：SHA-256 `1b311022af5900ad4b56383b5be48be0dca2738329ba38090cdc9921e43e08ea`；这是本次随机初始模型。
- E0 `model_2.pt`：`8071b7fcb85a85f4b177450b563f915f84d388714eb65868dbfb374321db928c`；E0 `model_3.pt`：`22c11f4c48e8d959dd046dd1cc1eb8cfc313d87e507a66c32b192d49bafbf7e9`。仅保留为 E0 证据，本轮均未加载。

当前不能声称：已学会或稳定行走、E1 pilot 通过、最终收敛、跨训练 seed 可复现、坡地能力、MoE 有效、视觉/LiDAR 控制能力、真实机器人可部署。也不能由这次失败推断原始 Flat 在更长预算下必然不可学习。E0 只证明链路可运行。
