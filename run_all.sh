#!/bin/zsh
# Reproduce every manuscript result from the released data, in dependency order.
# Usage: PYTHON=/path/to/python ./run_all.sh   (default: python3)
# Takes about 35 minutes. Results are written under 05_Results/.
set -eu
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
export PYTHON="$PY" PYTHONPATH="$ROOT" MPLCONFIGDIR="$ROOT/.cache/matplotlib"
LOG="$ROOT/05_Results/run_all_logs"
mkdir -p "$LOG"
step() { local name="$1"; shift; echo "[$(date +%H:%M:%S)] $name"; "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED: $name (see $LOG/$name.log)"; tail -20 "$LOG/$name.log"; exit 1; }; }

# 1. Online replay of IR2 (Algorithm 1) and the fixed-timeout derivation inputs
step 01_online          ./run.sh online
step 02_sensitivity     ./run.sh sensitivity
step 03_static          ./run.sh static
step 04_grace           ./run.sh grace
# 2. Revision notebooks (fractional GPUs, queueing, hindsight bootstrap, March 2025 split, utilization cutoffs)
for nb in 04_FractionalGPUImpact 05_ReclaimedCapacityAndQueueing 06_ConfidenceIntervalAnalysis \
          09_WorkloadMixThresholdStability 10_UtilizationCutoffAndSamplingSensitivity; do
  step "05_nb_$nb"      ./run.sh run-notebook "04_Analysis/revision/$nb.ipynb" --timeout 3000
done
step 06_cutoff          "$PY" 04_Analysis/revision/utilization_cutoff_uniform_pp.py
# 3. Section 3 and 4.1-4.2 notebooks and tables
step 07_manuscript      ./run.sh manuscript
# 4. Adaptive IR2, frontier, rolling window, percentile trade-off
A=04_Analysis/revision/adaptive_ir2
step 08_adaptive_build  "$PY" -I "$A/adaptive_replay.py" build
step 09_adaptive_run    "$PY" -I "$A/adaptive_replay.py" run
step 10_adaptive_exp    "$PY" -I "$A/adaptive_experiments.py"
step 11_frontier        "$PY" -I "$A/subset_frontier.py"
step 12_tradeoff        "$PY" -I "$A/paper_tradeoff_figure.py"
step 13_rolling_window  "$PY" -I "$A/naive_window.py"
# 5. Confidence intervals, single-timeout comparisons, fractional bound, reuse overlap
step 14_audit           "$PY" -I 04_Analysis/revision/additional_analyses/audit_analyses.py
step 15_reuse           "$PY" -I 04_Analysis/revision/additional_analyses/reuse_overlap_online.py
# 6. Manuscript figures and the hand-aggregated numbers
step 16_harvest_s3      "$PY" 04_Analysis/figure_style/harvest_plot_data.py section3_01
step 17_harvest_s4      "$PY" 04_Analysis/figure_style/harvest_plot_data.py section4_03
step 18_figures         "$PY" 04_Analysis/figure_style/restyle_paper_figures.py
step 19_numbers         "$PY" 04_Analysis/manuscript_numbers.py
echo "[$(date +%H:%M:%S)] ALL STEPS PASSED"
