#!/usr/bin/env bash
set -u

PY=/home/xiexuhui/miniconda3/envs/instinctlab/bin/python
ROOT=/home/xiexuhui/InstinctLab
TRAIN=$ROOT/logs/e1-flat-pilot/pilot_20260922_110715_nsfqypgl/train/20260922_110723_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E1_flat_pilot_s42
PROBE=$ROOT/scripts/diagnostics/e1_d1_probe.py

# Completed bounded matrix archive. No training is invoked by these commands.
ARCHIVE=$ROOT/logs/e1-d1/acceptance_20260923_123426_229wgskq
for controller in zero model_initial model_200; do
  for mode in legacy_raw raw_clip hard_target_clip; do
    "$PY" -B -u "$PROBE" --headless --device cuda:0 \
      --train-dir "$TRAIN" --output-dir "$ARCHIVE/${controller}_${mode}" \
      --controller "$controller" --action-mode "$mode"
  done
done

# Completed pair-confirmed model_200 and ground-positive archive.
ARCHIVE=$ROOT/logs/e1-d1/acceptance_20260923_124656_7y14lqkw
"$PY" -B -u "$PROBE" --headless --device cuda:0 \
  --train-dir "$TRAIN" --output-dir "$ARCHIVE/ground_positive" --mode ground_positive
for mode in legacy_raw raw_clip hard_target_clip; do
  "$PY" -B -u "$PROBE" --headless --device cuda:0 \
    --train-dir "$TRAIN" --output-dir "$ARCHIVE/model_200_${mode}" \
    --controller model_200 --action-mode "$mode"
done

# Completed nominal diagnostic (one deterministic initial state, max 2 s).
"$PY" -B -u "$PROBE" --headless --device cuda:0 \
  --train-dir "$TRAIN" --output-dir "$ROOT/logs/e1-d1/acceptance_20260923_100554_cscbqle0/nominal" \
  --mode nominal

# Failed environment checks retained as provenance, not as results.
/usr/bin/python3 -B -u "$PROBE" --headless --device cuda:0 \
  --train-dir "$TRAIN" --output-dir "$ROOT/logs/e1-d1/acceptance_20260923_125132_v_v877kl/model_initial_legacy_raw" \
  --controller model_initial --action-mode legacy_raw
"$PY" -B -u "$PROBE" --headless --device cuda:0 \
  --train-dir "$TRAIN" --output-dir "$ROOT/logs/e1-d1/acceptance_20260923_125153_oqxu1qsi/model_initial_legacy_raw" \
  --controller model_initial --action-mode legacy_raw

# Static checks used for handoff.
"$PY" -m py_compile "$ROOT/scripts/diagnostics/e1_d1_probe.py" "$ROOT/scripts/diagnostics/e1_d1_run.py"
git -C "$ROOT" diff --check
