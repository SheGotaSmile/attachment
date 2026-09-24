#!/usr/bin/env bash
exec /home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py \
  --task Instinct-Locomotion-Flat-G1-Research-v0 \
  --num_envs 64 --seed 42 --max_iterations 20 \
  --headless --device cuda:0 --init_noise_std 0.2 \
  --logroot /home/xiexuhui/InstinctLab/logs/e1-d4/action_distribution_20260924_091000/train \
  --run_name E1_D4_Research_s42 --e1_d4_diagnostic
