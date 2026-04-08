#!/usr/bin/env bash
set -euo pipefail

#python experiment_suite.py --preset heuristic_baseline --base-dir experiment_runs --workers 20 --seeds 20 --iterations 100
#python experiment_suite.py --preset rollout_ablation --base-dir experiment_runs --workers 20 --seeds 20 --iterations 100
#python experiment_suite.py --preset improvement_ablation --base-dir experiment_runs --workers 20 --seeds 20 --iterations 100
python experiment_suite.py --preset hyper_sweeps --base-dir experiment_runs --workers 20 --seeds 10 --iterations 50

python sweep_analysis.py experiment_runs/hyper_iterations
python sweep_analysis.py experiment_runs/hyper_rollout_depth
python sweep_analysis.py experiment_runs/hyper_exploration
python sweep_analysis.py experiment_runs/hyper_prior_bonus
