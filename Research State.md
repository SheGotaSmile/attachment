# Research State

日期：2026-09-22

## 当前阶段

E0 已通过，准备执行 E1-flat pilot。

## 已完成

- InstinctLab、instinct_rl、IsaacLab-Instinct 已完成静态核查。
- Flat G1 真实 Isaac Sim 启动成功。
- 完成 2 次 PPO update。
- policy shape 为 96，critic shape 为 99，action 为 29。
- 部分环境 reset 验收通过。
- checkpoint 保存、加载和 resume 验收通过。
- resume 从 iteration 2 开始，并生成 iteration 3 checkpoint。

## 当前结论

E0 只证明训练链路、reset 和 checkpoint 正常。
尚未证明 G1 已学会平地行走。
MoE、坡地、深度相机和 LiDAR 尚未开始。

## 当前唯一下一步

从随机初始化开始执行 E1-flat pilot：
- 原始 Flat G1；
- S-small；
- 不使用 E0 checkpoint；
- 不加入坡地；
- 不加入 MoE；
- 不加入传感器；
- 先训练 200 iterations；
- 完成独立 Flat evaluator。

## 禁止事项

- 不要直接实现 MoE；
- 不要实现连续坡度；
- 不要使用 E0 checkpoint 作为正式策略；
- 不要把 E0 结果写成行走性能结果；
- 不要开始 ±35° 或 ±40° 实验。

## 关键证据

- docs/repo_fact_check.md
- docs/experiment_protocol_v0.1.md
- e0/acceptance_20260917_6Po7jf/
- docs/E0_report.md，待补齐
