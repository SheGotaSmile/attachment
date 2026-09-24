#!/usr/bin/env bash
set -euo pipefail
cd /home/xiexuhui/InstinctLab
python_bin=/home/xiexuhui/miniconda3/envs/instinctlab/bin/python
run_root=/home/xiexuhui/InstinctLab/logs/e1-d5/bounded_20260924_gpu_recovery
train_root="$run_root/train/20260924_122746_G1ResearchPrimaryFall_E1_D5_Research_s42_gpu_recovery"
fixed_cases=/home/xiexuhui/InstinctLab/logs/e1-d3/diagnostic_20260924_004916__swyel2y/evaluation_cases.json

"$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py --checkpoint "$train_root/snapshot_update_000.pt" --output-dir "$run_root/independent_cases_create" --cases "$run_root/independent_cases_32.json" --create-cases --case-count 32 --seed 92345 --noise-seed 1728347 --headless --device cuda:0

for update in 000 020 050 100 150 200; do
    "$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py --checkpoint "$train_root/snapshot_update_${update}.pt" --output-dir "$run_root/eval_update_${update}" --cases "$fixed_cases" --expected-case-count 8 --seed 12345 --noise-seed 782347 --headless --device cuda:0
done

"$python_bin" -B -u scripts/instinct_rl/evaluate_research_flat.py --checkpoint "$train_root/snapshot_update_200.pt" --output-dir "$run_root/eval_final_independent_32" --cases "$run_root/independent_cases_32.json" --expected-case-count 32 --seed 92345 --noise-seed 1728347 --headless --device cuda:0
