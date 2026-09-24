#!/usr/bin/env bash
set -euo pipefail
cd /home/xiexuhui/InstinctLab
python_bin=/home/xiexuhui/miniconda3/envs/instinctlab/bin/python
run_root=/home/xiexuhui/InstinctLab/logs/e1-d6/forward_learnability_20260924_gpu
train_root="$run_root/train/20260924_153010_G1ForwardResearch_E1_D6_ForwardResearch_s42"
task=Instinct-Locomotion-Flat-G1-ForwardResearch-v0
fixed_cases="$run_root/fixed_evaluation_cases.json"
independent_cases="$run_root/independent_32_cases.json"

# Persist both evaluation populations before any checkpoint is evaluated.
"$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py \
  --task "$task" --checkpoint "$train_root/snapshot_update_000.pt" \
  --output-dir "$run_root/fixed_cases_create" --cases "$fixed_cases" \
  --create-cases --case-count 8 --case-id-prefix fixed_forward_ \
  --expected-command 0.5 0.0 0.0 --max-steps 1000 \
  --seed 12345 --noise-seed 782347 --headless --device cuda:0

"$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py \
  --task "$task" --checkpoint "$train_root/snapshot_update_000.pt" \
  --output-dir "$run_root/independent_cases_create" --cases "$independent_cases" \
  --create-cases --case-count 32 --case-id-prefix independent_forward_ \
  --expected-command 0.5 0.0 0.0 --max-steps 1000 \
  --seed 92345 --noise-seed 1728347 --headless --device cuda:0

for update in 000 020 050 100 150 200; do
  "$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py \
    --task "$task" --checkpoint "$train_root/snapshot_update_${update}.pt" \
    --output-dir "$run_root/eval_update_${update}" --cases "$fixed_cases" \
    --expected-case-count 8 --expected-command 0.5 0.0 0.0 --max-steps 1000 \
    --seed 12345 --noise-seed 782347 --headless --device cuda:0
done

"$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py \
  --task "$task" --checkpoint "$train_root/snapshot_update_200.pt" \
  --output-dir "$run_root/eval_final_independent_32" --cases "$independent_cases" \
  --expected-case-count 32 --expected-command 0.5 0.0 0.0 --max-steps 1000 \
  --seed 92345 --noise-seed 1728347 --headless --device cuda:0
