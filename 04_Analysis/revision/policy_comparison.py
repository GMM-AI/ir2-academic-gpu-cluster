"""Train-only, activity-exposure-constrained IR2/static-policy comparison.

The input trace and previous results are read-only. A policy can trigger only
at an observed idle sample. "Gross post-trigger time" is NOT safe savings:
it includes later observed activity and unobserved time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from project_paths import *
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_contract import current_contract
from cache_paths import require_frozen_cache

ONLINE_CACHE = require_frozen_cache("online_ir2", ['container_events.csv'])

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WORK_ROOT = Path(str(REVISION_CODE))
sys.path.insert(0, str(WORK_ROOT / "online_ir2"))
sys.path.insert(0, str(WORK_ROOT / "threshold_sensitivity"))
from simulation import (  # noqa: E402
    POST, PRE, TRACE, load_usage, normalize_filename, prepare_trace,
)
from sensitivity import first_crossing  # noqa: E402

OUTPUT = RESULTS_REVISION
SAVED_EVENTS = ONLINE_CACHE / "container_events.csv"
POPS = ("Train", "Test")
BINS = ("Non-GPU", "GPU-Enabled")
# Specified before evaluating Test. This is a bounded discrete comparator,
# not a claim to optimize all possible static timeouts.
TIMEOUTS = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 170, 180,
                     210, 240, 300, 360, 480, 600, 720, 960, 1200, 1440, 2160], dtype=int)
MEASURES = ("terminated", "return_containers", "gross_cpu_min", "gross_gpu_min",
            "idle_cpu_min", "idle_gpu_min", "active_cpu_min", "active_gpu_min",
            "unknown_cpu_min", "unknown_gpu_min")


def first_event_indices(pre_hits: np.ndarray, post_hits: np.ndarray) -> np.ndarray:
    """Broadcast first Pre/Post hit; -1 means neither phase triggers."""
    pre = pre_hits[:, None]
    post = post_hits[None, :]
    return np.where((pre >= 0) & (post >= 0), np.minimum(pre, post),
                    np.maximum(pre, post)).astype(int)


def event_measures(times, active_tail, idle_tail, last_active_idx, hit):
    """Return post-trigger time partition, preserving unobserved remainder."""
    valid = hit >= 0
    safe_hit = np.maximum(hit, 0)
    gross = np.where(valid, (times[-1] - times[safe_hit]) / 60e9, 0.0)
    active = np.where(valid, active_tail[safe_hit], 0.0)
    idle = np.where(valid, idle_tail[safe_hit], 0.0)
    unknown = np.maximum(gross - active - idle, 0.0)
    if np.any(active + idle > gross + 1e-6):
        raise AssertionError("Observed support exceeds post-trigger span")
    return {"terminated": valid.astype(float),
            "return_containers": (valid & (last_active_idx > hit)).astype(float),
            "gross": gross, "idle": idle, "active": active, "unknown": unknown}


def evaluate_one(trace: pd.DataFrame, gpu_bin: str):
    all_pre = {int(t): int(t) for t in np.unique(np.r_[TIMEOUTS, PRE[gpu_bin]])}
    all_post = {int(t): int(t) for t in np.unique(np.r_[TIMEOUTS, POST[gpu_bin]])}
    times, pre_hits, post_hits, active_tail, idle_tail, active_idx = first_crossing(
        trace, all_pre, all_post)
    pre = np.array([pre_hits[int(t)] for t in TIMEOUTS])
    post = np.array([post_hits[int(t)] for t in TIMEOUTS])
    grid_hit = first_event_indices(pre, post)
    last_active_idx = int(active_idx[-1]) if len(active_idx) else -1
    grid = event_measures(times, active_tail, idle_tail, last_active_idx, grid_hit)
    ir2_pre = pre_hits[int(PRE[gpu_bin])]
    ir2_post = post_hits[int(POST[gpu_bin])]
    ir2_hit = first_event_indices(np.array([ir2_pre]), np.array([ir2_post]))[0, 0]
    ir2_kind = (None if ir2_hit < 0 else
                "Pre" if ir2_pre >= 0 and (ir2_post < 0 or ir2_pre <= ir2_post) else "Post")
    ir2 = event_measures(times, active_tail, idle_tail, last_active_idx,
                         np.array([[ir2_hit]]))
    return grid, ir2, (pd.Timestamp(times[ir2_hit]) if ir2_hit >= 0 else pd.NaT), ir2_kind


def synthetic_tests():
    base = pd.Timestamp("2026-01-01")
    scenarios = [
        ([(i, "idle") for i in range(16)] + [(16, "active")], 15, 999, 1, 15),
        ([(0, "active")] + [(i, "idle") for i in range(1, 16)] + [(16, "active")], 999, 15, 1, 15),
        ([(i, "idle") for i in range(15)] + [(15, "active")], 15, 999, 0, None),
        ([(0, "idle"), (1, "unknown")] + [(i, "idle") for i in range(2, 17)], 15, 999, 0, None),
        ([(0, "active"), (1, "idle"), (100, "idle")], 999, 15, 0, 100),
    ]
    for rows, pre_limit, post_limit, expected_return, expected_event in scenarios:
        f = pd.DataFrame({"time": [base + pd.Timedelta(minutes=m) for m, _ in rows],
                          "state": [s for _, s in rows]})
        times, ph, qh, at, it, ai = first_crossing(f, {0: pre_limit}, {0: post_limit})
        hit = first_event_indices(np.array([ph[0]]), np.array([qh[0]]))
        result = event_measures(times, at, it, int(ai[-1]) if len(ai) else -1, hit)
        observed = None if hit[0, 0] < 0 else (f.time.iloc[hit[0, 0]] - base).total_seconds()/60
        assert observed == expected_event, (rows, observed)
        assert result["return_containers"][0, 0] == expected_return
        assert np.isclose(result["idle"][0, 0] + result["active"][0, 0]
                          + result["unknown"][0, 0], result["gross"][0, 0])
    assert first_event_indices(np.array([-1, 3]), np.array([5, -1])).tolist() == [[5, -1], [3, 3]]
    return len(scenarios) + 1


def empty_accumulators():
    shape = (len(TIMEOUTS), len(TIMEOUTS))
    return {(p, b): {m: np.zeros(shape, dtype=float) for m in MEASURES}
            for p in POPS for b in BINS}


def add_to(acc, measured, cpu, gpu):
    for m in ("terminated", "return_containers"):
        acc[m] += measured[m]
    for m in ("gross", "idle", "active", "unknown"):
        acc[f"{m}_cpu_min"] += measured[m] * cpu
        acc[f"{m}_gpu_min"] += measured[m] * gpu


def select_policies(grid, reference):
    train_gpu = grid[("Train", "GPU-Enabled")]
    train_non = grid[("Train", "Non-GPU")]
    ref_gpu = reference[("Train", "GPU-Enabled")]["return_containers"]
    ref_non = reference[("Train", "Non-GPU")]["return_containers"]
    feasible = ((train_gpu["return_containers"] <= ref_gpu + 1e-8)
                & (train_non["return_containers"] <= ref_non + 1e-8))
    chosen = {}
    for policy, mask in (("단일 고정 제한시간 (Single static)", np.eye(len(TIMEOUTS), dtype=bool)),
                         ("단계별 고정 제한시간 (Phase static)", np.ones_like(feasible, dtype=bool))):
        candidates = np.argwhere(feasible & mask)
        if not len(candidates):
            raise AssertionError(f"No feasible Train comparator: {policy}")
        # Predeclared Train objective: highest gross GPU-min under both
        # activity-return ceilings; then CPU-min; then longer timeouts.
        score = lambda pair: (train_gpu["gross_gpu_min"][tuple(pair)],
                              train_gpu["gross_cpu_min"][tuple(pair)]
                              + train_non["gross_cpu_min"][tuple(pair)],
                              int(TIMEOUTS[pair[0]] + TIMEOUTS[pair[1]]),
                              int(TIMEOUTS[pair[0]]))
        best = max(candidates, key=score)
        chosen[policy] = (int(best[0]), int(best[1]))
    return chosen, feasible


def summary_rows(grid, reference, selected, baseline):
    rows = []
    for pop in POPS:
        for policy in ("IR2 (유형·단계별 / type + phase)", *selected):
            for group in ("GPU-Enabled", "Non-GPU", "All"):
                bins = BINS if group == "All" else (group,)
                if policy.startswith("IR2"):
                    values = {m: sum(reference[(pop, b)][m] for b in bins) for m in MEASURES}
                else:
                    pair = selected[policy]
                    values = {m: sum(grid[(pop, b)][m][pair] for b in bins) for m in MEASURES}
                row = {"population": pop, "policy": policy, "gpu_bin": group,
                       "pre_timeout_min": (np.nan if policy.startswith("IR2")
                                           else int(TIMEOUTS[selected[policy][0]])),
                       "post_timeout_min": (np.nan if policy.startswith("IR2")
                                            else int(TIMEOUTS[selected[policy][1]])),
                       "containers": int(sum(baseline[(pop, b)]["containers"] for b in bins)),
                       "baseline_cpu_min": float(sum(baseline[(pop, b)]["cpu_min"] for b in bins)),
                       "baseline_gpu_min": float(sum(baseline[(pop, b)]["gpu_min"] for b in bins)),
                       **values}
                for resource in ("cpu", "gpu"):
                    denom = row[f"baseline_{resource}_min"]
                    row[f"gross_{resource}_pct"] = (100 * row[f"gross_{resource}_min"] / denom
                                                      if denom > 0 else np.nan)
                row["return_pct_of_terminated"] = (100 * row["return_containers"] / row["terminated"]
                                                   if row["terminated"] else np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_rows(grid, feasible, selected):
    out = []
    for i, pre in enumerate(TIMEOUTS):
        for j, post in enumerate(TIMEOUTS):
            g, n = grid[("Train", "GPU-Enabled")], grid[("Train", "Non-GPU")]
            out.append({"pre_timeout_min": int(pre), "post_timeout_min": int(post),
                        "single_static": bool(i == j), "feasible": bool(feasible[i, j]),
                        "train_return_gpu": int(g["return_containers"][i, j]),
                        "train_return_nongpu": int(n["return_containers"][i, j]),
                        "train_gross_gpu_min": float(g["gross_gpu_min"][i, j]),
                        "train_gross_cpu_min": float(g["gross_cpu_min"][i, j]
                                                     + n["gross_cpu_min"][i, j]),
                        "selected_single": (i, j) == selected["단일 고정 제한시간 (Single static)"],
                        "selected_phase": (i, j) == selected["단계별 고정 제한시간 (Phase static)"]})
    return pd.DataFrame(out)


def collect_selected(trace_file, usage, selected):
    """Second read: compact paired container rows for Test uncertainty checks."""
    columns = ["filename", "time", "used_time(minute)", "gpu_allocated(group)",
               "cpu_util(%)", "mem_util(%)", "gpu_util(%)"]
    rows = []
    for k in range(trace_file.num_row_groups):
        frame = trace_file.read_row_group(k, columns=columns).to_pandas()
        key = normalize_filename(frame.filename.iloc[0])
        meta = usage.loc[key]
        gpu_bin = "Non-GPU" if meta["gpu_allocated(group)"] == "0.0_GPU" else "GPU-Enabled"
        trace, _ = prepare_trace(frame, gpu_bin, meta.terminated_time)
        grid, ir2, _, _ = evaluate_one(trace, gpu_bin)
        outcomes = {"IR2 (유형·단계별 / type + phase)":
                    {m: v[0, 0] for m, v in ir2.items()}}
        for policy, pair in selected.items():
            outcomes[policy] = {m: v[pair] for m, v in grid.items()}
        cpu, gpu = float(meta["cpu_allocated(Core)"]), float(meta["gpu_allocated(EA)"])
        for policy, outcome in outcomes.items():
            rows.append({"filename": key, "population": meta.population,
                         "created_week": str(meta.created_time.to_period("W-SUN")),
                         "gpu_bin": gpu_bin, "policy": policy,
                         "baseline_gpu_min": gpu * float(trace.elapsed.max()),
                         "baseline_cpu_min": cpu * float(trace.elapsed.max()),
                         "terminated": int(outcome["terminated"]),
                         "return_containers": int(outcome["return_containers"]),
                         **{f"{state}_{resource}_min": float(outcome[state] * alloc)
                            for state in ("gross", "idle", "active", "unknown")
                            for resource, alloc in (("gpu", gpu), ("cpu", cpu))}})
        if (k + 1) % 5000 == 0:
            print(f"[RESULT] paired rows {k+1:,}/{trace_file.num_row_groups:,}", flush=True)
    return pd.DataFrame(rows)


def paired_week_intervals(selected_rows, selected):
    """Exploratory Test creation-week bootstrap; Train selection stays frozen."""
    reference = "IR2 (유형·단계별 / type + phase)"
    test = selected_rows.loc[selected_rows.population.eq("Test")
                             & selected_rows.gpu_bin.eq("GPU-Enabled")]
    rows = []
    for policy in selected:
        a = test.loc[test.policy.eq(reference), ["filename", "created_week", "baseline_gpu_min",
                                                 "gross_gpu_min"]]
        b = test.loc[test.policy.eq(policy), ["filename", "gross_gpu_min"]]
        joined = a.merge(b, on="filename", suffixes=("_ir2", "_static"), validate="one_to_one")
        # Expected count from the cohort contract.
        expected = current_contract()["group_counts"]["Test"]["GPU-Enabled"]
        if len(joined) != expected:
            raise AssertionError("Paired Test GPU coverage changed")
        joined["difference_gpu_min"] = joined.gross_gpu_min_static - joined.gross_gpu_min_ir2
        blocks = joined.groupby("created_week")[["difference_gpu_min", "baseline_gpu_min"]].sum()
        if len(blocks) < 10:
            raise AssertionError("Too few Test creation weeks")
        rng = np.random.default_rng(20260924)
        draws = rng.integers(0, len(blocks), size=(2000, len(blocks)))
        numerator = blocks.difference_gpu_min.to_numpy()[draws].sum(axis=1)
        denominator = blocks.baseline_gpu_min.to_numpy()[draws].sum(axis=1)
        estimates = 100 * numerator / denominator
        point = 100 * joined.difference_gpu_min.sum() / joined.baseline_gpu_min.sum()
        lower, upper = np.quantile(estimates, [0.025, 0.975])
        rows.append({"policy": policy, "population": "Test", "resource": "GPU",
                     "difference_vs_ir2_pp": point, "lower_95_pp": lower, "upper_95_pp": upper,
                     "creation_week_blocks": len(blocks), "bootstrap_draws": len(estimates),
                     "scope": "조건부·탐색적 / conditional exploratory"})
    return pd.DataFrame(rows)


def run(output: Path = OUTPUT):
    tests = synthetic_tests()
    if not TRACE.is_file() or not SAVED_EVENTS.is_file():
        raise FileNotFoundError("Raw trace or validated 02 online events are missing")
    usage = load_usage()
    saved = pd.read_csv(SAVED_EVENTS, usecols=["filename", "population", "gpu_bin", "online_time",
                                             "online_kind", "potential_interruption", "baseline_min",
                                             "gpu_allocated", "cpu_allocated"])
    saved["key"] = saved.filename.map(normalize_filename)
    saved = saved.set_index("key", verify_integrity=True)
    grid, ref = empty_accumulators(), {(p, b): {m: 0.0 for m in MEASURES}
                                         for p in POPS for b in BINS}
    baseline = {(p, b): {"containers": 0, "gpu_min": 0.0, "cpu_min": 0.0}
                for p in POPS for b in BINS}
    trace_file = pq.ParquetFile(TRACE)
    required = ["filename", "time", "used_time(minute)", "gpu_allocated(group)",
                "cpu_util(%)", "mem_util(%)", "gpu_util(%)"]
    assert set(required).issubset(trace_file.schema_arrow.names)
    seen = set()
    for k in range(trace_file.num_row_groups):
        frame = trace_file.read_row_group(k, columns=required).to_pandas()
        names = frame.filename.dropna().unique()
        if len(names) != 1:
            raise AssertionError(f"Row group {k}: expected one container")
        key = normalize_filename(names[0])
        if key in seen or key not in usage.index or key not in saved.index:
            raise AssertionError(f"Duplicate or unmatched container: {key}")
        seen.add(key)
        meta = usage.loc[key]
        original = saved.loc[key]
        pop = meta.population
        gpu_bin = "Non-GPU" if meta["gpu_allocated(group)"] == "0.0_GPU" else "GPU-Enabled"
        if pop not in POPS or (pop, gpu_bin) != (original.population, original.gpu_bin):
            raise AssertionError("Population/GPU group changed")
        cpu, gpu = float(meta["cpu_allocated(Core)"]), float(meta["gpu_allocated(EA)"])
        if not np.isclose(cpu, original.cpu_allocated) or not np.isclose(gpu, original.gpu_allocated):
            raise AssertionError("Resource allocation changed")
        trace, audit = prepare_trace(frame, gpu_bin, meta.terminated_time)
        if audit["duplicate_rows"] or audit["rows_after_termination"] or trace.state.eq("unknown").any():
            raise AssertionError("Unexpected duplicate, clipped, or unknown input")
        if not np.isclose(trace.elapsed.max(), original.baseline_min):
            raise AssertionError("Original resource denominator changed")
        candidate, ir2, ir2_time, ir2_kind = evaluate_one(trace, gpu_bin)
        old_time = pd.to_datetime(original.online_time)
        old_kind = original.online_kind if pd.notna(original.online_kind) else None
        if (pd.isna(ir2_time) != pd.isna(old_time) or
                (pd.notna(ir2_time) and ir2_time != old_time) or ir2_kind != old_kind or
                bool(ir2["return_containers"][0, 0]) != bool(original.potential_interruption)):
            raise AssertionError(f"IR2 event differs from validated 02: {key}")
        bucket = (pop, gpu_bin)
        add_to(grid[bucket], candidate, cpu, gpu)
        add_to(ref[bucket], {m: v[0, 0] for m, v in ir2.items()}, cpu, gpu)
        baseline[bucket]["containers"] += 1
        baseline[bucket]["cpu_min"] += cpu * original.baseline_min
        baseline[bucket]["gpu_min"] += gpu * original.baseline_min
        if (k + 1) % 2500 == 0:
            print(f"[RESULT] processed {k+1:,}/{trace_file.num_row_groups:,} containers", flush=True)
    if len(seen) != len(saved) or len(seen) != trace_file.num_row_groups:
        raise AssertionError("Population coverage mismatch")
    counts = {p: sum(baseline[(p, b)]["containers"] for b in BINS) for p in POPS}
    if counts != saved.population.value_counts().to_dict():
        raise AssertionError(f"Expected 10,820/4,305: {counts}")
    for pop in POPS:
        for b in BINS:
            subset = saved.loc[saved.population.eq(pop) & saved.gpu_bin.eq(b)]
            if int(ref[(pop, b)]["terminated"]) != int(subset.online_time.notna().sum()):
                raise AssertionError("Original termination count mismatch")
    selected, feasible = select_policies(grid, ref)
    summary = summary_rows(grid, ref, selected, baseline)
    calibration = calibration_rows(grid, feasible, selected)
    for pop in POPS:
        for b in BINS:
            base = summary.loc[summary.population.eq(pop) & summary.gpu_bin.eq(b)]
            assert len(base) == 3 and base.containers.nunique() == 1
            for m in ("cpu", "gpu"):
                assert np.allclose(base[f"idle_{m}_min"] + base[f"active_{m}_min"]
                                   + base[f"unknown_{m}_min"], base[f"gross_{m}_min"], atol=0.05)
    paired = collect_selected(trace_file, usage, selected)
    if len(paired) != len(saved) * 3 or paired.duplicated(["filename", "policy"]).any():
        raise AssertionError("Selected paired rows are missing or duplicated")
    for (pop, group, policy), subset in paired.groupby(["population", "gpu_bin", "policy"]):
        target = summary.loc[summary.population.eq(pop) & summary.gpu_bin.eq(group)
                             & summary.policy.eq(policy)].iloc[0]
        for m in ("terminated", "return_containers", "gross_gpu_min", "gross_cpu_min",
                  "active_gpu_min", "idle_gpu_min", "unknown_gpu_min"):
            if not np.isclose(subset[m].sum(), target[m], atol=0.05, rtol=0):
                raise AssertionError(f"Selected row reconciliation failed: {pop} {group} {policy} {m}")
    intervals = paired_week_intervals(paired, selected)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "06_policy_summary.csv", index=False)
    calibration.to_csv(output / "06_train_candidate_grid.csv", index=False)
    paired.to_parquet(output / "06_selected_container_results.parquet", index=False)
    intervals.to_csv(output / "06_test_paired_intervals.csv", index=False)
    checks = {"status": "PASS", "synthetic_tests": tests, "containers": counts,
              "source_row_groups": trace_file.num_row_groups,
              "ir2_event_match": "PASS: every container time against 02",
              "candidate_timeouts_min": TIMEOUTS.tolist(),
              "risk_budget_train": {b: int(ref[("Train", b)]["return_containers"]) for b in BINS},
              "selected_train_only": {name: [int(TIMEOUTS[i]), int(TIMEOUTS[j])]
                                      for name, (i, j) in selected.items()},
              "selected_container_rows": len(paired), "paired_row_reconciliation": "PASS",
              "raw_trace": str(TRACE), "source_events": str(SAVED_EVENTS)}
    (output / "06_validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print("[RESULT]", json.dumps(checks, ensure_ascii=False), flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.test:
        print("[RESULT] synthetic_tests: PASS", synthetic_tests())
    else:
        run(args.output)
