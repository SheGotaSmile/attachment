# E0 executed commands (archive, not a batch replay script)

# cwd: /home/xiexuhui/InstinctLab

/home/xiexuhui/miniconda3/envs/instinctlab/bin/python scripts/diagnostics/e0_runtime.py --run-dir /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf collect

/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/list_envs.py

/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py --task Instinct-Locomotion-Flat-G1-v0 --num_envs 4 --seed 42 --max_iterations 2 --headless --device cuda:0 --logroot /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train --run_name E0_s42 --e0_audit

/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/e0_reset_probe.py --headless --device cuda:0 --task Instinct-Locomotion-Flat-G1-v0 --num_envs 4 --seed 42 --output-dir /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe

/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py --task Instinct-Locomotion-Flat-G1-v0 --num_envs 4 --seed 42 --max_iterations 1 --headless --device cuda:0 --logroot /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/resume --run_name E0_resume_s42 --resume --load_run /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_154055_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42 --checkpoint model_2.pt --e0_audit --e0_reference /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_154055_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42/model_2.pt.reference.pt

/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py --task Instinct-Locomotion-Flat-G1-v0 --num_envs 4 --seed 42 --max_iterations 1 --headless --device cuda:0 --logroot /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/resume --run_name E0_resume_s42_fixed --resume --load_run /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_154055_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42 --checkpoint '^model_2[.]pt$' --e0_audit --e0_reference /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_154055_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42/model_2.pt.reference.pt

/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py --task Instinct-Locomotion-Flat-G1-v0 --num_envs 4 --seed 42 --max_iterations 2 --headless --device cuda:0 --logroot /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train --run_name E0_s42_logs_fixed --e0_audit

/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/instinct_rl/train.py --task Instinct-Locomotion-Flat-G1-v0 --num_envs 4 --seed 42 --max_iterations 1 --headless --device cuda:0 --logroot /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/resume --run_name E0_resume_s42_logs_fixed --resume --load_run /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed --checkpoint '^model_2[.]pt$' --e0_audit --e0_reference /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed/model_2.pt.reference.pt

/home/xiexuhui/miniconda3/envs/instinctlab/bin/python -B -u scripts/diagnostics/e0_runtime.py --run-dir /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf finalize --training-dir /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/train/20260917_155259_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_s42_logs_fixed --resume-dir /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/resume/20260917_155746_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05_E0_resume_s42_logs_fixed_from20260917_155259 --reset-result /home/xiexuhui/InstinctLab/logs/e0/acceptance_20260917_6Po7jf/reset_probe/reset_results.json
