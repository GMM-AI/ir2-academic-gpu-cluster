"""Train-calibrated Pre/Post percentile sensitivity for FGCS IR2.

The input trace is read twice: once to calibrate thresholds on Train normal
containers, and once to evaluate every pre/post combination on all eligible
Train and Test containers. No source dataset is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(_PROJECT_ROOT))
from project_paths import PROCESSED_DATA, RESULTS_REVISION, install_data_io_guard
install_data_io_guard()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "online_ir2"))
from simulation import (  # noqa: E402
    POST,
    PRE,
    TRACE,
    load_usage,
    normalize_filename,
    online_decision,
    prepare_trace,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohort_contract import current_contract
CONTRACT = current_contract()

OUTPUT = RESULTS_REVISION / "_cache/threshold_sensitivity"
PRE_PERCENTILES = (80, 85, 90, 95)
POST_PERCENTILES = (60, 70, 80, 90)
REQUIRED = ["filename", "time", "used_time(minute)", "gpu_allocated(group)",
            "cpu_util(%)", "mem_util(%)", "gpu_util(%)"]


def read_containers():
    usage = load_usage()
    parquet = pq.ParquetFile(TRACE)
    if not set(REQUIRED).issubset(parquet.schema_arrow.names):
        raise ValueError("Required trace columns are missing")
    seen = set()
    for i in range(parquet.num_row_groups):
        frame = parquet.read_row_group(i, columns=REQUIRED).to_pandas()
        names = frame.filename.dropna().unique()
        if len(names) != 1:
            raise ValueError(f"Row group {i} has {len(names)} filenames")
        key = normalize_filename(names[0])
        if key in seen or key not in usage.index:
            raise ValueError(f"Duplicate or unmatched key: {key}")
        seen.add(key)
        meta = usage.loc[key]
        if meta.population not in ("Train", "Test"):
            raise ValueError(f"Unexpected population: {key}")
        gpu_bin = "Non-GPU" if meta["gpu_allocated(group)"] == "0.0_GPU" else "GPU-Enabled"
        trace, audit = prepare_trace(frame, gpu_bin, meta.terminated_time)
        if audit["duplicate_rows"] or audit["rows_after_termination"] or trace.state.eq("unknown").any():
            raise ValueError("Expected zero duplicates, post-termination rows and unknown states")
        yield key, trace, meta, gpu_bin
        if (i + 1) % 2500 == 0:
            print(f"[RESULT] processed {i + 1:,}/{parquet.num_row_groups:,} row groups", flush=True)


def calibration_row(key, trace, meta, gpu_bin):
    used = trace.loc[trace.state.eq("active")]
    if used.empty:
        return {"filename": key, "population": meta.population, "gpu_bin": gpu_bin,
                "normal": False, "pre_idle_min": np.nan, "post_idle_min": np.nan,
                "tail_idle": bool(trace.state.iloc[-1] == "idle")}
    pre_idle = max(float(used.elapsed.iloc[0] - trace.elapsed.iloc[0]), 0.0)
    post_idle = max(float(trace.elapsed.max() - used.elapsed.iloc[-1]), 0.0)
    return {"filename": key, "population": meta.population, "gpu_bin": gpu_bin,
            "normal": True, "pre_idle_min": pre_idle, "post_idle_min": post_idle,
            "tail_idle": bool(trace.state.iloc[-1] == "idle")}


def calibrate():
    rows = [calibration_row(*item) for item in read_containers()]
    cases = pd.DataFrame(rows)
    counts = cases.groupby("population").size().to_dict()
    if counts != CONTRACT["population_counts"]:
        raise AssertionError(f"Cohort mismatch: {counts}")
    normal = cases.loc[cases.population.eq("Train") & cases.normal]
    if len(normal) != CONTRACT["class_counts"]["Train"]["normal"]:
        raise AssertionError(f"Train normal count: {len(normal)} != latest audit count")
    result = []
    for gpu_bin in ("Non-GPU", "GPU-Enabled"):
        subset = normal.loc[normal.gpu_bin.eq(gpu_bin)]
        pre = subset.loc[subset.pre_idle_min.ge(1), "pre_idle_min"]
        post = subset.loc[subset.post_idle_min.ge(1) & subset.tail_idle, "post_idle_min"]
        expected = tuple(CONTRACT["calibration_counts"][gpu_bin])
        if (len(subset), len(pre), len(post)) != expected:
            raise AssertionError(f"Calibration population mismatch for {gpu_bin}: "
                                 f"{(len(subset), len(pre), len(post))} != {expected}")
        for phase, values, percentiles in (("Pre", pre, PRE_PERCENTILES),
                                           ("Post", post, POST_PERCENTILES)):
            for percentile in percentiles:
                raw = float(np.percentile(values.to_numpy(), percentile))
                result.append({"gpu_bin": gpu_bin, "phase": phase,
                               "percentile": percentile, "normal_containers": len(subset),
                               "valid_cases": len(values), "raw_min": raw,
                               "threshold_min": int(np.floor(raw + .5))})
    thresholds = pd.DataFrame(result)
    # Fresh percentile calibration is a diagnostic, not a policy change.
    reference = pd.read_csv(_PROJECT_ROOT/'inputs/calibration_thresholds_fill0.csv')
    for row in reference.itertuples(index=False):
        q=thresholds.loc[thresholds.gpu_bin.eq(row.gpu_bin)&thresholds.phase.eq(row.phase)&thresholds.percentile.eq(row.percentile)]
        if len(q)!=1 or not np.isclose(q.raw_min.iloc[0],row.raw_min,atol=1e-6,rtol=0):
            raise AssertionError(f'Independent corrected calibration mismatch: {row.gpu_bin}/{row.phase}')
    thresholds['frozen_applied_min']=[(PRE if phase=='Pre' else POST)[b] for b,phase in zip(thresholds.gpu_bin,thresholds.phase)]
    thresholds['calibration_role']='Latest corrected-data calibration; applied baseline is 85/119/146/170'
    return cases, thresholds


def first_crossing(trace, pre_limits, post_limits):
    """Find the first observed idle sample crossing each distinct limit.

    This vectorizes online_decision: active/unknown samples end an idle run;
    observation gaps alone do not reset it. An adjacent active sample
    anchors the following Post run.
    """
    states = trace.state.to_numpy()
    idle = states == "idle"
    active = states == "active"
    times = trace.time.to_numpy(dtype="datetime64[ns]").astype("int64")
    n = len(times)
    previous_idle = np.r_[False, idle[:-1]]
    previous_active = np.r_[False, active[:-1]]
    seen_active = np.maximum.accumulate(active.astype(np.int8)).astype(bool)
    run_start = idle & ~previous_idle
    base = times.copy()
    starts_after_active = run_start & previous_active & seen_active
    base[starts_after_active] = np.r_[times[0], times[:-1]][starts_after_active]
    start_idx = np.maximum.accumulate(np.where(run_start, np.arange(n), 0))
    age = (times - base[start_idx]) / 60e9
    pre_mask, post_mask = idle & ~seen_active, idle & seen_active

    def candidate(mask, limit):
        hits = np.flatnonzero(mask & (age >= limit))
        return int(hits[0]) if len(hits) else -1

    pre_hits = {p: candidate(pre_mask, limit) for p, limit in pre_limits.items()}
    post_hits = {p: candidate(post_mask, limit) for p, limit in post_limits.items()}
    support = np.minimum(np.diff(times) / 60e9, 1.0)
    active_support = support * active[:-1]
    idle_support = support * idle[:-1]
    active_tail = np.r_[np.cumsum(active_support[::-1])[::-1], 0.0]
    idle_tail = np.r_[np.cumsum(idle_support[::-1])[::-1], 0.0]
    active_idx = np.flatnonzero(active)
    return times, pre_hits, post_hits, active_tail, idle_tail, active_idx


def evaluate_container(key, trace, meta, gpu_bin, threshold_maps):
    pre_limits = threshold_maps[gpu_bin]["Pre"]
    post_limits = threshold_maps[gpu_bin]["Post"]
    times, pre_hits, post_hits, active_tail, idle_tail, active_idx = first_crossing(
        trace, pre_limits, post_limits)
    used = trace.loc[trace.state.ne("idle")]
    end_elapsed = float(trace.elapsed.max())
    first_idle = end_elapsed if used.empty else float(used.elapsed.iloc[0]) - 1.0
    last_active_elapsed = float(used.elapsed.iloc[-1]) if not used.empty else np.nan
    end = times[-1]
    rows = []
    for pre_p in PRE_PERCENTILES:
        pre_limit = pre_limits[pre_p]
        pre_saved = max(first_idle - pre_limit, 0.0)
        for post_p in POST_PERCENTILES:
            post_limit = post_limits[post_p]
            post_saved = (max(end_elapsed - last_active_elapsed - post_limit, 0.0)
                          if not used.empty else 0.0)
            pre_idx, post_idx = pre_hits[pre_p], post_hits[post_p]
            hit = min((x for x in (pre_idx, post_idx) if x >= 0), default=-1)
            kind = ("Pre" if hit == pre_idx else "Post") if hit >= 0 else "None"
            online_min = float((end - times[hit]) / 60e9) if hit >= 0 else 0.0
            active_min = float(active_tail[hit]) if hit >= 0 else 0.0
            idle_min = float(idle_tail[hit]) if hit >= 0 else 0.0
            if active_min + idle_min > online_min + 1e-6:
                raise AssertionError("Observed support exceeds post-event span")
            interrupted = bool(np.any(active_idx > hit)) if hit >= 0 else False
            rows.append({"filename": key, "population": meta.population,
                         "created_week": str(meta.created_time.to_period("W-MON")),
                         "gpu_bin": gpu_bin, "pre_percentile": pre_p,
                         "post_percentile": post_p, "pre_threshold_min": pre_limit,
                         "post_threshold_min": post_limit,
                         "cpu_allocated": float(meta["cpu_allocated(Core)"]),
                         "gpu_allocated": float(meta["gpu_allocated(EA)"]),
                         "baseline_min": end_elapsed,
                         "hindsight_pre_min": pre_saved, "hindsight_post_min": post_saved,
                         "hindsight_total_min": pre_saved + post_saved,
                         "online_kind": kind, "online_time": pd.Timestamp(times[hit]) if hit >= 0 else pd.NaT,
                         "online_gross_min": online_min,
                         "after_idle_min": idle_min, "after_active_min": active_min,
                         "after_unknown_min": max(online_min - idle_min - active_min, 0.0),
                         "activity_after_online": interrupted})
    return rows


def summarize(rows):
    details = pd.DataFrame(rows)
    summary_rows = []
    for (population, pre_p, post_p), subset in details.groupby(
            ["population", "pre_percentile", "post_percentile"], sort=True):
        for resource, alloc_col in (("CPU", "cpu_allocated"), ("GPU", "gpu_allocated")):
            group = subset if resource == "CPU" else subset.loc[subset.gpu_bin.eq("GPU-Enabled")]
            alloc = group[alloc_col]
            baseline = float((alloc * group.baseline_min).sum())
            row = {"population": population, "pre_percentile": pre_p,
                   "post_percentile": post_p, "resource": resource,
                   "containers": len(group),
                   "online_terminations": int(group.online_kind.ne("None").sum()),
                   "pre_terminations": int(group.online_kind.eq("Pre").sum()),
                   "post_terminations": int(group.online_kind.eq("Post").sum()),
                   "activity_return_containers": int(group.activity_after_online.sum()),
                   "activity_return_pct_of_terminated": (
                       100 * group.activity_after_online.sum() / group.online_kind.ne("None").sum()
                       if group.online_kind.ne("None").any() else np.nan),
                   "baseline_resource_min": baseline}
            for name, col in (("hindsight_pre", "hindsight_pre_min"),
                              ("hindsight_post", "hindsight_post_min"),
                              ("hindsight_total", "hindsight_total_min"),
                              ("online_gross", "online_gross_min"),
                              ("online_idle_observed", "after_idle_min"),
                              ("online_active_observed", "after_active_min"),
                              ("online_unknown", "after_unknown_min")):
                row[f"{name}_resource_min"] = float((alloc * group[col]).sum())
            row["hindsight_pct"] = 100 * row["hindsight_total_resource_min"] / baseline
            row["online_gross_pct"] = 100 * row["online_gross_resource_min"] / baseline
            row["online_observed_idle_pct"] = 100 * row["online_idle_observed_resource_min"] / baseline
            summary_rows.append(row)
    return details, pd.DataFrame(summary_rows)


def run(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    calibration, thresholds = calibrate()
    threshold_maps = {gpu: {phase: dict(zip(part.percentile, part.threshold_min))
                            for phase, part in subset.groupby("phase")}
                      for gpu, subset in thresholds.groupby("gpu_bin")}
    baseline = pd.read_csv(RESULTS_REVISION / "_cache/online_ir2/container_events.csv",
                           usecols=["filename", "online_time", "online_kind", "baseline_min", "hindsight_pre_saved_min", "hindsight_post_saved_min", "after_idle_min", "after_active_min", "after_unknown_min", "potential_interruption"])
    baseline["key"] = baseline.filename.map(normalize_filename)
    baseline = baseline.set_index("key", verify_integrity=True)
    rows = []
    frozen_rows = []
    for key, trace, meta, gpu_bin in read_containers():
        if key not in baseline.index:
            raise AssertionError("Missing baseline event")
        base_time, base_kind = online_decision(trace, gpu_bin)
        saved_time = pd.to_datetime(baseline.loc[key, "online_time"])
        if (pd.isna(base_time) != pd.isna(saved_time) or
                (pd.notna(base_time) and abs((base_time - saved_time).total_seconds()) > 1e-6) or
                (base_kind or "None") != (baseline.loc[key, "online_kind"]
                                          if pd.notna(baseline.loc[key, "online_kind"]) else "None")):
            raise AssertionError(f"Saved baseline event mismatch: {key}")
        container_rows = evaluate_container(key, trace, meta, gpu_bin, threshold_maps)
        frozen_maps={b:{phase:dict(values) for phase,values in part.items()} for b,part in threshold_maps.items()}
        frozen_maps[gpu_bin]['Pre'][90]=PRE[gpu_bin]
        frozen_maps[gpu_bin]['Post'][70]=POST[gpu_bin]
        frozen_candidates=evaluate_container(key,trace,meta,gpu_bin,frozen_maps)
        base_row = next(row for row in frozen_candidates
                        if row["pre_percentile"] == 90 and row["post_percentile"] == 70)
        if (pd.isna(base_time) != pd.isna(base_row["online_time"]) or
                (pd.notna(base_time) and abs((base_time - base_row["online_time"]).total_seconds()) > 1e-6) or
                (base_kind or "None") != base_row["online_kind"]):
            raise AssertionError(f"Vectorized base replay mismatch: {key}")
        for field,source in [('baseline_min','baseline_min'),('hindsight_pre_min','hindsight_pre_saved_min'),('hindsight_post_min','hindsight_post_saved_min'),('after_idle_min','after_idle_min'),('after_active_min','after_active_min'),('after_unknown_min','after_unknown_min')]:
            if not np.isclose(base_row[field],baseline.loc[key,source],atol=1e-6,rtol=0):
                raise AssertionError(f'Frozen accounting mismatch: {key}/{field}')
        if base_row['activity_after_online']!=bool(baseline.loc[key,'potential_interruption']):
            raise AssertionError(f'Frozen return mismatch: {key}')
        base_row['scenario_role']='Frozen applied policy 85/119/146/170'
        frozen_rows.append(base_row)
        rows.extend(container_rows)
    details, summary = summarize(rows)
    if len(details) != CONTRACT["row_groups"] * len(PRE_PERCENTILES) * len(POST_PERCENTILES):
        raise AssertionError("Grid coverage mismatch")
    frozen_details,frozen_summary=summarize(frozen_rows)
    for (population,resource),group in frozen_summary.groupby(['population','resource']):
        row=group.iloc[0];ref=CONTRACT['resources'][population][resource]
        for field,source in [('baseline_resource_min','baseline'),('hindsight_pre_resource_min','pre_saved'),('hindsight_post_resource_min','post_saved'),('hindsight_total_resource_min','total_saved')]:
            if not np.isclose(row[field],ref[source],atol=.1,rtol=0):
                raise AssertionError(f'Frozen total mismatch: {population}/{resource}/{field}')
    frozen_details.to_parquet(output/'frozen_container_results.parquet',index=False)
    frozen_summary.to_csv(output/'frozen_summary.csv',index=False)
    calibration.to_csv(output / "calibration_cases.csv", index=False)
    thresholds.to_csv(output / "thresholds.csv", index=False)
    details.to_parquet(output / "grid_container_results.parquet", index=False)
    summary.to_csv(output / "grid_summary.csv", index=False)
    checks = {"status": "PASS", "train_containers": CONTRACT["population_counts"]["Train"], "test_containers": CONTRACT["population_counts"]["Test"],
              "settings": len(PRE_PERCENTILES) * len(POST_PERCENTILES),
              "baseline_event_replay": "PASS for every container",
              "baseline_accounting": "PASS for Train and Test GPU/CPU",
              "source_row_groups": CONTRACT['row_groups'], "source_rows": CONTRACT['rows'],
              "frozen_policy": {'Pre': PRE, 'Post': POST},
              "diagnostic_calibration": "Train P90/P70 rounded = applied 85/119/146/170 (missing-as-zero)",
              "frozen_container_rows": len(frozen_details),
              "grid_rows": len(details), "frozen_policy_vs_diagnostic": "Separate artifacts"}
    (output / "validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print("[RESULT]", checks, flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
