#!/usr/bin/env bash
set -euo pipefail
/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py --task Instinct-Locomotion-Flat-G1-Research-v0 --num_envs 64 --seed 42 --max_iterations 200 --headless --device cpu --init_noise_std 0.2 --logroot /home/xiexuhui/InstinctLab/logs/e1-d5/bounded_20260924_113434_cpu/train --run_name E1_D5_Research_s42_cpu --e1_d5_diagnostic
