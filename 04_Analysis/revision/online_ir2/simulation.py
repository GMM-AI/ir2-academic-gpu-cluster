"""FGCS IR2 observation-gated online replay and manuscript hindsight accounting.

The source Parquet is read one row group (one container) at a time. No source is
modified. All timestamps are compared as KST wall-clock values (the parquets store naive KST times).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(_PROJECT_ROOT))
from project_paths import PROCESSED_DATA, RESULTS_REVISION, install_data_io_guard
install_data_io_guard()

DATA = PROCESSED_DATA
USAGE = DATA / "UsageLog20230401_20251130_20260301_20260831.csv"
TRACE = DATA / "TimeSeries_20240926_20251130_20260301_20260831.parquet"
OUTPUT = RESULTS_REVISION / "_cache/online_ir2"
PRE = {"Non-GPU": 85.0, "GPU-Enabled": 119.0}  # GPU Pre 119 (Train P90 119.0)
POST = {"Non-GPU": 146.0, "GPU-Enabled": 170.0}
MAX_CONFIRMED_GAP_MIN = 2.0  # Intermediate-gap observation quality only; never resets the online timer.
OBSERVATION_SUPPORT_MIN = 1.0


def normalize_filename(value: str) -> str:
    name = str(value).replace("\\", "/").split("/")[-1].strip()
    for ext in (".csv", ".parquet"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def load_usage() -> pd.DataFrame:
    cols = ["filename", "created_time", "terminated_time", "gpu_allocated(group)",
            "gpu_allocated(EA)", "cpu_allocated(Core)"]
    data = pd.read_csv(USAGE, usecols=cols, low_memory=False)
    data["key"] = data["filename"].map(normalize_filename)
    if data["key"].isna().any() or data["key"].duplicated().any():
        raise ValueError("UsageLog normalized filename is missing or duplicated")
    for col in ("created_time", "terminated_time"):
        data[col] = pd.to_datetime(data[col], errors="coerce")
        if data[col].isna().any() or data[col].dt.tz is not None:
            raise ValueError(f"UsageLog {col} missing or unexpectedly timezone-aware")
    data["population"] = "Excluded"
    data.loc[data.created_time.ge("2023-03-01") & data.created_time.lt("2025-12-01"), "population"] = "Train"
    data.loc[data.created_time.ge("2026-03-01") & data.created_time.lt("2026-09-01"), "population"] = "Test"
    for col in ("gpu_allocated(EA)", "cpu_allocated(Core)"):
        data[col] = pd.to_numeric(data[col], errors="raise")
        if data[col].isna().any() or data[col].lt(0).any():
            raise ValueError(f"Invalid allocation: {col}")
    return data.set_index("key", verify_integrity=True)


def prepare_trace(frame: pd.DataFrame, gpu_bin: str, terminated: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Stable timestamp ordering; active wins conflicting duplicate measurements."""
    data = frame.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    if data.time.isna().any():
        raise ValueError("Missing trace timestamp")
    if data.time.dt.tz is not None:
        data["time"] = data.time.dt.tz_localize(None)
    data = data.sort_values("time", kind="stable")
    rows_after_termination = int(data.time.gt(terminated).sum())
    data = data.loc[data.time.le(terminated)].copy()
    if data.empty:
        raise ValueError("No trace samples on or before actual termination")
    numeric = ["cpu_util(%)", "mem_util(%)"] + (["gpu_util(%)"] if gpu_bin == "GPU-Enabled" else [])
    valid = data[numeric].notna().all(axis=1)
    idle = data["cpu_util(%)"].le(10) & data["mem_util(%)"].le(15 if gpu_bin == "Non-GPU" else 20)
    if gpu_bin == "GPU-Enabled":
        idle &= data["gpu_util(%)"].eq(0)
    data["state"] = np.select([~valid, idle], ["unknown", "idle"], default="active")
    duplicate_rows = int(data.time.duplicated().sum())
    conflicting_timestamps = int(data.groupby("time", sort=False).state.nunique().gt(1).sum())
    priority = {"unknown": 0, "idle": 1, "active": 2}
    data["priority"] = data.state.map(priority)
    data = data.sort_values(["time", "priority"], kind="stable").drop_duplicates("time", keep="last")
    data = data.sort_values("time").reset_index(drop=True)
    group_values = frame["gpu_allocated(group)"].dropna().unique().tolist()
    if len(group_values) != 1 or (group_values[0] == "0.0_GPU") != (gpu_bin == "Non-GPU"):
        raise ValueError("GPU group disagrees with UsageLog")
    elapsed = pd.to_numeric(data["used_time(minute)"], errors="coerce")
    if elapsed.isna().any() or elapsed.lt(0).any():
        raise ValueError("Invalid used_time")
    data["elapsed"] = elapsed
    if data.time.duplicated().any() or data.time.isna().any():
        raise ValueError("Trace ordering/deduplication failed")
    return data, {"duplicate_rows": duplicate_rows, "conflicting_timestamps": conflicting_timestamps,
                  "rows_after_termination": rows_after_termination}


def online_decision(trace: pd.DataFrame, gpu_bin: str) -> tuple[pd.Timestamp | pd.NaT, str | None]:
    """Only the prefix ending at the current observation affects a decision.

    State is resolved first. Active at the exact timeout wins; an unknown sample
    breaks the idle timer. Observation gaps alone do not reset it. A kill is possible only
    at an observed idle sample, with elapsed idle >= the frozen timeout.
    """
    seen_active = False
    idle_start = None
    last_active = None
    previous_state = None
    for row in trace.itertuples(index=False):
        t, state = row.time, row.state
        if state == "active":
            seen_active, idle_start, last_active = True, None, t
        elif state == "unknown":
            idle_start = None
        else:
            if idle_start is None:
                idle_start = (last_active if seen_active and previous_state == "active" else t)
            kind = "Post" if seen_active else "Pre"
            limit = POST[gpu_bin] if seen_active else PRE[gpu_bin]
            if (t - idle_start).total_seconds() / 60 >= limit:
                return t, kind
        previous_state = state
    return pd.NaT, None


def hindsight_decision(trace: pd.DataFrame, gpu_bin: str) -> dict:
    """Reproduce the submitted code's separate Pre and Post idle accounting.

    The original code counts Pre idle past its timeout even when the container
    later becomes active. Pre and Post can therefore both contribute for one
    container; their sum is an observed-trace accounting total, not the result
    of executing two terminations on that container.
    """
    end_elapsed = float(trace.elapsed.max())
    # In the submitted notebook, a missing utilization signal is classified
    # as "used" by the comparison predicates. The actual input has no such
    # samples, but retaining that rule makes this accounting faithful.
    used = trace.loc[trace.state.ne("idle")]
    first_idle = end_elapsed if used.empty else float(used.elapsed.iloc[0]) - 1.0
    pre_saved = max(first_idle - PRE[gpu_bin], 0.0)
    post_saved = (max(end_elapsed - float(used.elapsed.iloc[-1]) - POST[gpu_bin], 0.0)
                  if not used.empty else 0.0)
    pre_time = (trace.time.iloc[0] + pd.Timedelta(minutes=PRE[gpu_bin])
                if pre_saved > 0 else pd.NaT)
    post_time = (used.time.iloc[-1] + pd.Timedelta(minutes=POST[gpu_bin])
                 if post_saved > 0 else pd.NaT)
    # The overlap comparison needs one event per container. It uses the first
    # hypothetical phase, while the resource accounting retains both phases.
    first_time = pre_time if pd.notna(pre_time) else post_time
    first_kind = "Pre" if pd.notna(pre_time) else ("Post" if pd.notna(post_time) else None)
    return {"hindsight_time": first_time, "hindsight_kind": first_kind,
            "hindsight_pre_time": pre_time, "hindsight_post_time": post_time,
            "hindsight_pre_saved_min": pre_saved,
            "hindsight_post_saved_min": post_saved,
            "hindsight_saved_min": pre_saved + post_saved}


def classify_after(trace: pd.DataFrame, event: pd.Timestamp | pd.NaT) -> dict:
    """Partition observed follow-up [event,last sample] into idle/active/unknown.

    A sample supports at most the next 1 minute, ending earlier at the next
    sample. Unsampled remainder is unknown, including gaps and the final tail.
    """
    if pd.isna(event):
        return {"after_idle_min": 0.0, "after_active_min": 0.0, "after_unknown_min": 0.0}
    end = trace.time.iloc[-1]
    span = max((end - event).total_seconds() / 60, 0.0)
    result = {"idle": 0.0, "active": 0.0, "unknown": 0.0}
    times = trace.time.to_numpy(dtype="datetime64[ns]").astype("int64")
    starts = np.maximum(times[:-1], event.value)
    ends = times[1:]
    support_ends = np.minimum(times[:-1] + int(OBSERVATION_SUPPORT_MIN * 60e9), ends)
    supported = np.maximum(support_ends - starts, 0) / 60e9
    states = trace.state.to_numpy()[:-1]
    result["idle"] = float(supported[states == "idle"].sum())
    result["active"] = float(supported[states == "active"].sum())
    result["unknown"] = max(span - result["idle"] - result["active"], 0.0)
    if not np.isclose(sum(result.values()), span, atol=1e-7, rtol=0):
        raise AssertionError("Post-event interval partition does not close")
    return {f"after_{state}_min": value for state, value in result.items()}


def intermediate_gaps(trace: pd.DataFrame, gpu_bin: str) -> dict:
    active_indices = np.flatnonzero(trace.state.to_numpy() == "active")
    durations, confirmed = [], []
    longest = 0.0
    for left, right in zip(active_indices[:-1], active_indices[1:]):
        if right - left <= 1:
            continue
        between = trace.iloc[left + 1:right]
        duration = (trace.time.iloc[right] - trace.time.iloc[left]).total_seconds() / 60
        durations.append(duration)
        local_gaps = trace.time.iloc[left:right + 1].diff().dt.total_seconds().dropna() / 60
        fully_observed_idle = (between.state.eq("idle").all() and
                               local_gaps.le(MAX_CONFIRMED_GAP_MIN).all())
        if fully_observed_idle:
            confirmed.append(duration)
            longest = max(longest, duration)
    return {"intermediate_gap_count": len(durations),
            "intermediate_gap_max_min": max(durations, default=0.0),
            "confirmed_idle_gap_count": len(confirmed),
            "confirmed_idle_gap_max_min": longest,
            "confirmed_over_post_timeout": any(x >= POST[gpu_bin] for x in confirmed),
            "all_intermediate_gaps": durations,
            "confirmed_idle_gaps": confirmed}


def analyze_one(frame: pd.DataFrame, meta: pd.Series) -> dict:
    gpu_bin = "Non-GPU" if meta["gpu_allocated(group)"] == "0.0_GPU" else "GPU-Enabled"
    trace, audit = prepare_trace(frame, gpu_bin, meta.terminated_time)
    online_time, online_kind = online_decision(trace, gpu_bin)
    hindsight = hindsight_decision(trace, gpu_bin)
    t0, end = trace.time.iloc[0], trace.time.iloc[-1]
    duration = (end - t0).total_seconds() / 60
    gaps = trace.time.diff().dt.total_seconds().dropna() / 60
    middle = intermediate_gaps(trace, gpu_bin)
    active_after = trace.loc[trace.time.gt(online_time) & trace.state.eq("active")] if pd.notna(online_time) else trace.iloc[0:0]
    first_active_after = active_after.time.iloc[0] if not active_after.empty else pd.NaT
    online_saved = max((end - online_time).total_seconds() / 60, 0.0) if pd.notna(online_time) else 0.0
    after = classify_after(trace, online_time)
    baseline = float(trace.elapsed.max())
    if online_saved > baseline + 1:
        raise AssertionError("Online duration exceeds baseline by >1 min")
    result = {"filename": str(frame.filename.iloc[0]), "population": meta.population,
              "gpu_bin": gpu_bin, "gpu_group": meta["gpu_allocated(group)"],
              "cpu_allocated": float(meta["cpu_allocated(Core)"]),
              "gpu_allocated": float(meta["gpu_allocated(EA)"]),
              "created_time": meta.created_time, "terminated_time": meta.terminated_time,
              "first_observation": t0, "last_observation": end,
              "samples": len(trace), "baseline_min": baseline,
              "actual_observed_span_min": duration,
              "max_observed_gap_min": float(gaps.max()) if not gaps.empty else 0.0,
              "unknown_samples": int(trace.state.eq("unknown").sum()),
              "first_active": trace.loc[trace.state.eq("active"), "time"].min(),
              "last_active": trace.loc[trace.state.eq("active"), "time"].max(),
              **hindsight,
              "online_time": online_time, "online_kind": online_kind,
              "online_saved_min": online_saved,
              "first_active_after_online": first_active_after,
              "potential_interruption": bool(pd.notna(first_active_after)),
              "censored_before_termination": bool((meta.terminated_time - end).total_seconds() > 120),
              **audit, **after, **{k: v for k, v in middle.items() if not isinstance(v, list)}}
    if pd.notna(online_time) and online_time < t0:
        raise AssertionError("Kill before first observation")
    return result, middle


def synthetic_tests() -> list[str]:
    base = pd.Timestamp("2026-01-01 00:00:00")
    pre = int(PRE["Non-GPU"])
    cases = [
        ("all_idle", [(i, "idle") for i in range(pre+1)], "Pre", pre),
        ("pre_before_late_active", [(i, "idle") for i in range(pre+1)] + [(pre+1, "active")], "Pre", pre),
        ("active_before_pre", [(0, "idle"), (pre-1, "active")], None, None),
        ("short_middle_return", [(0, "active"), (1, "idle"), (100, "active")], None, None),
        ("long_middle_return", [(0, "active")] + [(i, "idle") for i in range(1, 148)] + [(148, "active")], "Post", 146),
        ("post_tail", [(0, "active")] + [(i, "idle") for i in range(1, 148)], "Post", 146),
        ("active_at_boundary", [(i, "idle") for i in range(pre)] + [(pre, "active")], None, None),
        ("irregular_gap", [(0, "idle"), (1, "idle"), (100, "idle")], "Pre", 100),
        ("post_gap", [(0, "active"), (1, "idle"), (180, "idle")], "Post", 180),
        ("missing_breaks_timer", [(0, "idle"), (1, "unknown"), (89, "idle")], None, None),
    ]
    passed = []
    for name, samples, expected_kind, expected_min in cases:
        f = pd.DataFrame({"time": [base + pd.Timedelta(minutes=m) for m, _ in samples],
                          "state": [s for _, s in samples]})
        event, kind = online_decision(f, "Non-GPU")
        assert kind == expected_kind, name
        assert (None if pd.isna(event) else (event - base).total_seconds()/60) == expected_min, name
        for cut in range(1, len(f) + 1):
            prefix = f.iloc[:cut]
            before = online_decision(prefix, "Non-GPU")
            mutated = pd.concat([prefix, f.iloc[cut:].assign(state="unknown")], ignore_index=True)
            after = online_decision(mutated, "Non-GPU")
            if pd.notna(before[0]):
                assert before == after, f"prefix invariance: {name}"
        passed.append(name)
    # One-minute observation support: a 10-minute silent gap is 9 minutes unknown.
    f = pd.DataFrame({"time": [base, base + pd.Timedelta(minutes=10)], "state": ["idle", "active"]})
    partition = classify_after(f, base)
    assert np.isclose(partition["after_idle_min"], 1) and np.isclose(partition["after_unknown_min"], 9)
    passed.append("gap_partition_and_future_prefix")
    duplicate = pd.DataFrame({"filename": ["case", "case", "case"],
                              "time": [base, base, base + pd.Timedelta(minutes=1)],
                              "used_time(minute)": [0., 0., 1.],
                              "gpu_allocated(group)": ["0.0_GPU"] * 3,
                              "cpu_util(%)": [0., 80., 0.],
                              "mem_util(%)": [0., 50., 0.],
                              "gpu_util(%)": [np.nan] * 3})
    prepared, audit = prepare_trace(duplicate, "Non-GPU", base + pd.Timedelta(minutes=2))
    assert prepared.state.tolist() == ["active", "idle"]
    assert audit["duplicate_rows"] == 1 and audit["conflicting_timestamps"] == 1
    passed.append("duplicate_active_priority")
    # The submitted baseline counts idle time before first use and after last
    # use for the same eventually-active container.
    phases = pd.DataFrame({"time": [base + pd.Timedelta(minutes=i) for i in range(301)],
                           "state": ["idle"] * 125 + ["active"] * 6 + ["idle"] * 170,
                           "elapsed": np.arange(301, dtype=float)})
    accounting = hindsight_decision(phases, "Non-GPU")
    assert accounting["hindsight_pre_saved_min"] == 124.0 - PRE["Non-GPU"]
    assert accounting["hindsight_post_saved_min"] == 24.0
    assert accounting["hindsight_saved_min"] == 148.0 - PRE["Non-GPU"]
    assert accounting["hindsight_kind"] == "Pre"
    passed.append("submitted_pre_and_post_accounting")
    return passed


def aggregate(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (population, gpu_bin), group in events.groupby(["population", "gpu_bin"], sort=False):
        for scope, subset in (("type", group), ("all", events.loc[events.population.eq(population)])):
            if scope == "all" and gpu_bin != sorted(events.gpu_bin.unique())[0]:
                continue
            label = gpu_bin if scope == "type" else "All"
            for resource, alloc in (("CPU", "cpu_allocated"), ("GPU", "gpu_allocated")):
                if resource == "GPU" and label == "Non-GPU":
                    continue
                if resource == "GPU" and label == "All":
                    continue  # GPU-Enabled already is the complete GPU denominator
                baseline = float((subset[alloc] * subset.baseline_min).sum())
                for policy, minutes in (("Hindsight", "hindsight_saved_min"), ("Online", "online_saved_min")):
                    saved = float((subset[alloc] * subset[minutes]).sum())
                    pre_minutes = (subset.hindsight_pre_saved_min if policy == "Hindsight"
                                   else subset.online_saved_min.where(subset.online_kind.eq("Pre"), 0.0))
                    post_minutes = (subset.hindsight_post_saved_min if policy == "Hindsight"
                                    else subset.online_saved_min.where(subset.online_kind.eq("Post"), 0.0))
                    rows.append({"population": population, "gpu_bin": label, "resource": resource,
                                 "policy": policy, "containers": len(subset),
                                 "terminated": int(subset[f"{policy.lower()}_time"].notna().sum()),
                                 "baseline_resource_min": baseline, "reclaimed_resource_min": saved,
                                 "pre_reclaimed_resource_min": float((subset[alloc] * pre_minutes).sum()),
                                 "post_reclaimed_resource_min": float((subset[alloc] * post_minutes).sum()),
                                 "reclaimed_pct": saved / baseline * 100 if baseline else np.nan})
    summary = pd.DataFrame(rows).drop_duplicates(["population", "gpu_bin", "resource", "policy"])
    overlap = []
    for (population, gpu_bin), subset in events.groupby(["population", "gpu_bin"], sort=False):
        h, o = subset.hindsight_time.notna(), subset.online_time.notna()
        both = subset.loc[h & o]
        delta = (both.online_time - both.hindsight_time).dt.total_seconds() / 60
        overlap.append({"population": population, "gpu_bin": gpu_bin, "containers": len(subset),
                        "hindsight_only": int((h & ~o).sum()), "online_only": int((o & ~h).sum()),
                        "both": int((h & o).sum()), "neither": int((~h & ~o).sum()),
                        "online_minus_hindsight_median_min": delta.median(),
                        "online_minus_hindsight_p90_min": delta.quantile(.9),
                        "pre_then_active": int((subset.online_kind.eq("Pre") & subset.potential_interruption).sum()),
                        "post_then_active": int((subset.online_kind.eq("Post") & subset.potential_interruption).sum()),
                        "potential_interruptions": int(subset.potential_interruption.sum()),
                        "potential_interruption_pct": subset.potential_interruption.mean() * 100,
                        "confirmed_long_idle_return": int(subset.confirmed_over_post_timeout.sum()),
                        "max_observed_gap_p90_min": subset.max_observed_gap_min.quantile(.9),
                        "max_observed_gap_max_min": subset.max_observed_gap_min.max(),
                        "intermediate_gap_p90_min": subset.intermediate_gap_max_min.quantile(.9),
                        "intermediate_gap_max_min": subset.intermediate_gap_max_min.max()})
    return summary, pd.DataFrame(overlap)


def run_analysis(output: Path = OUTPUT) -> dict:
    tests = synthetic_tests()
    output.mkdir(parents=True, exist_ok=True)
    usage = load_usage()
    parquet = pq.ParquetFile(TRACE)
    required = ["filename", "time", "used_time(minute)", "gpu_allocated(group)",
                "cpu_util(%)", "mem_util(%)", "gpu_util(%)"]
    if not set(required).issubset(parquet.schema_arrow.names):
        raise ValueError("TimeSeries required columns missing")
    events, gap_rows = [], []
    seen = set()
    for i in range(parquet.num_row_groups):
        frame = parquet.read_row_group(i, columns=required).to_pandas()
        names = frame.filename.dropna().unique()
        if len(names) != 1:
            raise ValueError(f"Row group {i} is not a single container")
        key = normalize_filename(names[0])
        if key in seen or key not in usage.index:
            raise ValueError(f"Duplicate/unmatched trace key: {key}")
        seen.add(key)
        meta = usage.loc[key]
        if meta.population not in ("Train", "Test"):
            raise ValueError(f"Unexpected cohort: {key}")
        event, gaps = analyze_one(frame, meta)
        events.append(event)
        for confirmed, durations in ((False, gaps["all_intermediate_gaps"]),
                                     (True, gaps["confirmed_idle_gaps"])):
            gap_rows.extend({"filename": names[0], "population": meta.population,
                             "gpu_bin": event["gpu_bin"], "confirmed_idle": confirmed,
                             "gap_min": duration} for duration in durations)
        if (i + 1) % 2500 == 0:
            print(f"[RESULT] processed {i + 1:,}/{parquet.num_row_groups:,} row groups", flush=True)
    event_df = pd.DataFrame(events)
    gaps_df = pd.DataFrame(gap_rows)
    if len(event_df) != parquet.num_row_groups or event_df.filename.duplicated().any():
        raise AssertionError("One event row per container violated")
    counts = event_df.population.value_counts().to_dict()
    # Validate against the preprocessing cohort.
    cohort_audit = pd.read_csv(_PROJECT_ROOT / "inputs/preprocessing_file_audit.csv")
    expected_counts = cohort_audit.loc[cohort_audit.status.eq("ok")].groupby("period").size().to_dict()
    if counts != expected_counts:
        raise AssertionError(f"Population mismatch: {counts}")
    if event_df[['baseline_min','online_saved_min','hindsight_saved_min']].lt(0).any().any():
        raise AssertionError("Negative resource duration")
    if not np.allclose(event_df.hindsight_pre_saved_min + event_df.hindsight_post_saved_min,
                       event_df.hindsight_saved_min, atol=1e-7, rtol=0):
        raise AssertionError("Submitted Pre/Post accounting does not reconcile")
    if event_df.online_saved_min.gt(event_df.baseline_min + 1).any():
        raise AssertionError("Online resource duration exceeds baseline")
    if event_df[['hindsight_time','online_time']].notna().sum(axis=1).gt(2).any():
        raise AssertionError("Duplicate termination")
    partition = event_df.after_idle_min + event_df.after_active_min + event_df.after_unknown_min
    if not np.allclose(partition, event_df.online_saved_min, atol=1e-6, rtol=0):
        raise AssertionError("After-termination state partition mismatch")
    summary, overlap = aggregate(event_df)
    if not np.allclose(summary.pre_reclaimed_resource_min + summary.post_reclaimed_resource_min,
                       summary.reclaimed_resource_min, atol=.01, rtol=0):
        raise AssertionError("Resource-weighted Pre/Post accounting does not reconcile")
    state_rows = []
    for (population, gpu_bin), subset in event_df.groupby(["population", "gpu_bin"], sort=False):
        for resource, alloc in (("CPU", "cpu_allocated"), ("GPU", "gpu_allocated")):
            if resource == "GPU" and gpu_bin == "Non-GPU":
                continue
            state_rows.append({"population": population, "gpu_bin": gpu_bin, "resource": resource,
                               "online_reclaimed_resource_min": float((subset[alloc] * subset.online_saved_min).sum()),
                               **{f"after_{state}_resource_min": float((subset[alloc] * subset[f"after_{state}_min"]).sum())
                                  for state in ("idle", "active", "unknown")}})
    state_summary = pd.DataFrame(state_rows)
    if not np.allclose(state_summary[[f"after_{s}_resource_min" for s in ("idle", "active", "unknown")]].sum(axis=1),
                       state_summary.online_reclaimed_resource_min, atol=.01, rtol=0):
        raise AssertionError("Resource-weighted post-kill partition mismatch")
    gap_rows_summary = []
    for (population, gpu_bin, confirmed), subset in gaps_df.groupby(["population", "gpu_bin", "confirmed_idle"], sort=False):
        gap_rows_summary.append({"population": population, "gpu_bin": gpu_bin,
                                 "confirmed_idle": bool(confirmed), "gaps": len(subset),
                                 "median_min": subset.gap_min.median(), "p90_min": subset.gap_min.quantile(.9),
                                 "max_min": subset.gap_min.max(),
                                 "at_or_above_post_timeout": int(subset.gap_min.ge(POST[gpu_bin]).sum()),
                                 "at_or_above_post_timeout_pct": subset.gap_min.ge(POST[gpu_bin]).mean() * 100})
    gap_summary = pd.DataFrame(gap_rows_summary)
    event_df.to_csv(output / "container_events.csv", index=False)
    summary.to_csv(output / "resource_summary.csv", index=False)
    overlap.to_csv(output / "event_overlap.csv", index=False)
    gaps_df.to_csv(output / "intermediate_gaps.csv", index=False)
    gap_summary.to_csv(output / "gap_summary.csv", index=False)
    state_summary.to_csv(output / "state_summary.csv", index=False)
    event_df.loc[event_df.potential_interruption].to_csv(output / "potential_interruption_cases.csv", index=False)
    checks = {"synthetic_cases": tests, "row_groups": parquet.num_row_groups,
              "population_counts": counts, "unique_events": len(event_df),
              "duplicate_timestamp_rows": int(event_df.duplicate_rows.sum()),
              "conflicting_timestamps": int(event_df.conflicting_timestamps.sum()),
              "post_termination_samples_clipped": int(event_df.rows_after_termination.sum()),
              "unknown_samples": int(event_df.unknown_samples.sum()),
              "censored_before_termination": int(event_df.censored_before_termination.sum()),
              "resource_state_partition": "PASS",
              "checks": "PASS"}
    (output / "validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print("[RESULT] validation: PASS", checks, flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run only synthetic checks")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    options = parser.parse_args()
    if options.test:
        print("[RESULT] synthetic_checks: PASS", synthetic_tests())
    else:
        run_analysis(options.output)
