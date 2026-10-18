"""Set used_time(minute) in the processed UsageLog CSVs to terminated_time - run_time
(execution start to termination; the scheduling wait is excluded).
Other columns are rewritten verbatim (read as strings), in UTF-8 with LF line endings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = [("03_ProcessedData/UsageLog20230401_20251130.csv", "utf-8", "\n"),
         ("03_ProcessedData/UsageLog20230401_20251130_20260301_20260831.csv", "utf-8", "\n")]
report = {}
for rel, enc, eol in FILES:
    p = ROOT / rel
    u = pd.read_csv(p, encoding=enc, low_memory=False, dtype=str, keep_default_na=False)
    run = pd.to_datetime(u.run_time, errors="coerce")
    term = pd.to_datetime(u.terminated_time, errors="coerce")
    created = pd.to_datetime(u.created_time, errors="coerce")
    old = pd.to_numeric(u["used_time(minute)"], errors="coerce")
    assert run.notna().all() and term.notna().all() and (term >= run).all() and (run >= created).all()
    assert np.allclose(old, (term - created).dt.total_seconds() / 60, atol=1e-6)
    new = (term - run).dt.total_seconds() / 60
    queue = (run - created).dt.total_seconds() / 60
    u["used_time(minute)"] = new.map(repr)
    u.to_csv(p, index=False, encoding=enc, lineterminator=eol)
    back = pd.read_csv(p, encoding=enc, low_memory=False)
    assert len(back) == len(u) and list(back.columns) == list(u.columns)
    assert np.allclose(pd.to_numeric(back["used_time(minute)"]), new, atol=1e-6)
    report[rel] = {"rows": int(len(u)), "queue_min_median": float(queue.median()),
                   "queue_gt_1min_pct": float(100 * (queue > 1).mean()), "queue_gt_10min_pct": float(100 * (queue > 10).mean()),
                   "queue_max_min": float(queue.max()), "used_time_lt_1min_before": int((old < 1).sum()),
                   "used_time_lt_1min_after": int((new < 1).sum())}
out = ROOT / "05_Results/preprocessing/used_time_from_run_start.json"
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=1))
