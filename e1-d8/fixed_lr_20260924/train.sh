#!/usr/bin/env bash
set -euo pipefail
cd /home/xiexuhui/InstinctLab
/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-ForwardResearch-v0 \
  --num_envs 64 --seed 42 --max_iterations 1302 --headless --device cuda:0 \
  --init_noise_std 0.2 \
  --logroot /home/xiexuhui/InstinctLab/logs/e1-d8/fixed_lr_20260924/train \
  --run_name E1_D8_ForwardResearch_fixed_lr_s42 --e1_d8_diagnostic
