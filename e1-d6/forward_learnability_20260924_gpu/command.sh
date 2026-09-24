#!/usr/bin/env bash
set -euo pipefail
/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-ForwardResearch-v0 \
  --num_envs 64 \
  --seed 42 \
  --max_iterations 200 \
  --headless \
  --device cuda:0 \
  --init_noise_std 0.2 \
  --logroot /home/xiexuhui/InstinctLab/logs/e1-d6/forward_learnability_20260924_gpu/train \
  --run_name E1_D6_ForwardResearch_s42 \
  --e1_d6_diagnostic
