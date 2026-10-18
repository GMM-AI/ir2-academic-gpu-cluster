#!/bin/zsh
# Rebuild the processed data (03_ProcessedData, inputs/preprocessing_*) from the raw data (01_RawData).
# Usage: PYTHON=/path/to/python ./run_preprocessing.sh   (default: python3). Takes about 5 minutes.
# rebuild_time_series.py refuses to overwrite existing parquets: remove 03_ProcessedData first if it exists.
set -eu
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
export PYTHON="$PY" PYTHONPATH="$ROOT" MPLCONFIGDIR="$ROOT/.cache/matplotlib"
LOG="$ROOT/05_Results/preprocessing/logs"
mkdir -p "$LOG" "$ROOT/05_Results/preprocessing/stage1"
step() { local name="$1"; shift; echo "[$(date +%H:%M:%S)] $name"; "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED: $name (see $LOG/$name.log)"; tail -20 "$LOG/$name.log"; exit 1; }; }

# 1. Scheduler logs -> UsageLog (full period, then derivation period)
step 1_usagelog_full    ./run.sh run-notebook 02_Preprocessing/1_UsageLog_FullPeriod.ipynb --timeout 1200
step 2_usagelog_train   ./run.sh run-notebook 02_Preprocessing/2_UsageLog_DerivationPeriod.ipynb --timeout 1200
# 2. GPU utilization normalized by the allocated GPU count
step 3_gpu_normalize    "$PY" 02_Preprocessing/update_usage_logs.py
# 3. Execution time = terminated - run
step 4_execution_time   "$PY" 02_Preprocessing/used_time_from_run_start.py
# 4. Minute traces -> per-container parquet and the merged parquets
step 5_time_series      "$PY" 02_Preprocessing/rebuild_time_series.py
echo "[$(date +%H:%M:%S)] PREPROCESSING PASSED"
