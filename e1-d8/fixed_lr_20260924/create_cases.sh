#!/usr/bin/env bash
set -euo pipefail
cd /home/xiexuhui/InstinctLab
python_bin=/home/xiexuhui/miniconda3/envs/instinctlab/bin/python
root=/home/xiexuhui/InstinctLab/logs/e1-d8/fixed_lr_20260924
preflight=/home/xiexuhui/InstinctLab/logs/e1-d8/preflight_20260924/20260924_220945_G1ForwardResearch_E1_D8_preflight_s42
task=Instinct-Locomotion-Flat-G1-ForwardResearch-v0

"$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py \
  --task "$task" --checkpoint "$preflight/snapshot_update_000.pt" \
  --output-dir "$root/fixed_cases_create" --cases "$root/fixed_evaluation_cases.json" \
  --create-cases --case-count 8 --case-id-prefix fixed_forward_ \
  --expected-command 0.5 0.0 0.0 --max-steps 1000 \
  --seed 12345 --noise-seed 782347 --headless --device cuda:0

"$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py \
  --task "$task" --checkpoint "$preflight/snapshot_update_000.pt" \
  --output-dir "$root/independent_cases_create" --cases "$root/independent_32_cases.json" \
  --create-cases --case-count 32 --case-id-prefix independent_forward_ \
  --expected-command 0.5 0.0 0.0 --max-steps 1000 \
  --seed 92345 --noise-seed 1728347 --headless --device cuda:0
