#!/usr/bin/env bash
set -euo pipefail
cd /home/xiexuhui/InstinctLab
python_bin=/home/xiexuhui/miniconda3/envs/instinctlab/bin/python
root=/home/xiexuhui/InstinctLab/logs/e1-d8/fixed_lr_20260924
train_root="$1"
task=Instinct-Locomotion-Flat-G1-ForwardResearch-v0

for update in 000 100 300 600 900 1302; do
  "$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py \
    --task "$task" --checkpoint "$train_root/snapshot_update_${update}.pt" \
    --output-dir "$root/fixed_eval_${update}" --cases "$root/fixed_evaluation_cases.json" \
    --expected-case-count 8 --expected-command 0.5 0.0 0.0 --max-steps 1000 \
    --seed 12345 --noise-seed 782347 --headless --device cuda:0
  "$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py \
    --task "$task" --checkpoint "$train_root/snapshot_update_${update}.pt" \
    --output-dir "$root/independent_eval_${update}" --cases "$root/independent_32_cases.json" \
    --expected-case-count 32 --expected-command 0.5 0.0 0.0 --max-steps 1000 \
    --seed 92345 --noise-seed 1728347 --headless --device cuda:0
done
