"""Reuse overlap (notebook 05, cells 1/3/13/15/17/19/21) on the online decisions.

Step 1 reruns the notebook cells unchanged with outputs redirected to a new folder and
requires an exact match with the official 05_reclaim_demand_overlap_resource_time_summary.csv.
Step 2 replaces the hindsight phase events of cell 15 with the online decisions of the
official online replay (online_time, online_kind, after_idle_min = observed idle minutes after the decision, the headline accounting) and reruns 17/19/21.
The notebook and official results are read only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NB = ROOT / "04_Analysis/revision/05_ReclaimedCapacityAndQueueing.ipynb"
OUT = ROOT / "05_Results/revision/additional_analyses/reuse_overlap"
OFFICIAL = ROOT / "05_Results/revision/05_reclaim_demand_overlap_resource_time_summary.csv"

ONLINE_15 = '''
part = linked.loc[linked.online_time.notna() & linked.after_idle_min.gt(0)].copy()
part['phase'] = part.online_kind
part['reclaim_time'] = pd.to_datetime(part.online_time)
part['potentially_reclaimable_min'] = part.after_idle_min  # observed idle minutes after the decision (headline accounting)
virtual = part.reset_index(drop=True)
virtual['population_group'] = virtual.gpu_bin.replace({'GPU-Enabled':'Integer-GPU'})
virtual['event_id'] = virtual.filename + '|' + virtual.phase
virtual['reclaim_cpu_core'] = virtual.cpu_allocated
virtual['reclaim_mem_gb'] = virtual['mem_allocated(GB)']
virtual['reclaim_gpu_ea'] = virtual.gpu_allocated
virtual['reclaim_agent'] = virtual.agent
virtual['positive_duration'] = virtual.potentially_reclaimable_min.gt(0)
virtual['lifecycle_valid'] = (virtual.reclaim_time.notna()
    & virtual.reclaim_time.ge(virtual.created_time)
    & virtual.reclaim_time.le(virtual.terminated_time))
virtual['resource_valid'] = (virtual[['reclaim_cpu_core','reclaim_mem_gb','reclaim_gpu_ea']]
    .notna().all(axis=1) & np.isfinite(virtual[['reclaim_cpu_core','reclaim_mem_gb','reclaim_gpu_ea']]).all(axis=1)
    & virtual[['reclaim_cpu_core','reclaim_mem_gb','reclaim_gpu_ea']].ge(0).all(axis=1))
virtual['scheduler_log_covered'] = (
    (virtual.population.eq('Train') & virtual.reclaim_time.ge('2023-04-01')
     & virtual.reclaim_time.lt('2025-12-01'))
    | (virtual.population.eq('Test') & virtual.reclaim_time.ge('2026-03-01')
       & virtual.reclaim_time.lt('2026-09-01')))
virtual['evaluable'] = (virtual.positive_duration & virtual.lifecycle_valid
    & virtual.resource_valid & virtual.scheduler_log_covered)
virtual['check_reason'] = np.select([
    ~virtual.lifecycle_valid, ~virtual.resource_valid, ~virtual.scheduler_log_covered],
    ['event_outside_container_lifecycle','invalid_reclaim_allocation','outside_raw_log_window'], default='')
assert virtual.event_id.is_unique
print('online events', virtual.groupby('population').size().to_dict(),
      'decisions incl. zero saved', source_events.online_time.notna().groupby(source_events.population).sum().to_dict())
'''


def cells():
    nb = json.loads(NB.read_text())
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def run(variant: str) -> pd.DataFrame:
    src = cells()
    out = OUT / variant
    out.mkdir(parents=True, exist_ok=True)
    ns: dict = {"__name__": "__nb__"}
    cwd = os.getcwd()
    os.chdir(NB.parent)
    try:
        for i in (1, 3, 13, 15, 17, 19, 21):
            code = src[i]
            if i == 3:
                code = code + "\nRESULTS = OUT_DIR\n"
                ns["OUT_DIR"] = out
            if i == 15 and variant == "online":
                code = ONLINE_15
            if i in (15, 17, 19, 21):
                ns["RESULTS"] = out
            exec(compile(code, f"cell{i}", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns["resource_time_summary"], ns["overlap_summary"], ns


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    base, base_ov, _ = run("hindsight_check")
    off = pd.read_csv(OFFICIAL)
    num = [c for c in off.columns if c not in ("period", "phase", "population")]
    import numpy as np
    same = all(np.allclose(base[c].astype(float), off[c].astype(float), equal_nan=True, rtol=0, atol=1e-6) for c in num)
    print("hindsight rerun matches official:", same)
    if not same:
        raise SystemExit("reproduction failed")
    onl, onl_ov, ns = run("online")
    cols = ["period", "phase", "population", "analysis_reclaim_events_a", "events_with_waiting_b",
            "events_with_compatible_request_d", "gpu_share_of_total_hindsight_pct", "cpu_share_of_total_hindsight_pct"]
    print("HINDSIGHT\n", base[cols].round(2).to_string(index=False))
    print("ONLINE\n", onl[cols].round(2).to_string(index=False))
