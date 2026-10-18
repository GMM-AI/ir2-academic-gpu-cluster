"""Adaptive IR2 trace replay (adaptive timeout update, Section 4.3).

Specification (fixed before the evaluation):
  * start from the Train-derived IR2 timeouts 85/119/146/170 min;
  * Test containers are sampled with probability `rate`; samples are never
    reclaimed and, at their UsageLog termination time, contribute their
    Section 4.2 Pre/Post idle values to a per (group, phase) window;
  * the window holds the latest `window` values and is initialized with the
    last Train normal containers by termination time;
  * every `update_every` new values: candidate = floor(P(window) + 0.5)
    (Pre P90, Post P70) and tau = (1 - alpha) * tau + alpha * candidate;
  * decisions follow the official online rule (no gap reset, decision only
    at an observed idle sample with elapsed idle >= tau) using the tau in
    force at that sample time.

Samples are never reclaimed, so the tau path does not depend on any
reclamation decision and can be computed before the trace pass.

Inputs are read-only. Outputs go to 05_Results/revision/adaptive_ir2 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "04_Analysis/revision/online_ir2"))
from project_paths import RESULTS_REVISION  # noqa: E402
from simulation import POST, PRE, TRACE, load_usage, normalize_filename, prepare_trace  # noqa: E402

OUT = RESULTS_REVISION / "adaptive_ir2"
CACHE = OUT / "_cache"
EVENTS = RESULTS_REVISION / "_cache/online_ir2/container_events.csv"
CALIB = RESULTS_REVISION / "_cache/threshold_sensitivity/calibration_cases.csv"
POLICY_SUMMARY = RESULTS_REVISION / "06_policy_summary.csv"
CONTRACT = _PROJECT_ROOT / "inputs/cohort_contract.json"
CALIB_REFERENCE = _PROJECT_ROOT / "inputs/calibration_thresholds_fill0.csv"
TEST_START = pd.Timestamp("2026-03-01")
GROUPS = ("Non-GPU", "GPU-Enabled")
PHASES = ("Pre", "Post")
PCT = {"Pre": 90, "Post": 70}
STATIC = 150.0
N_SEEDS = 20
SEED_BASE = 20261013
BOOT_SEED = 20260924  # same block-bootstrap seed as policy_comparison.py
BOOT_DRAWS = 2000

MAIN = {"name": "main", "rate": 0.05, "alpha": 0.3, "window": 50, "update_every": 10}
SENSITIVITY = [
    {**MAIN, "name": "rate10", "rate": 0.10},
    {**MAIN, "name": "rate15", "rate": 0.15},
    {**MAIN, "name": "alpha0.1", "alpha": 0.1},
    {**MAIN, "name": "alpha0.5", "alpha": 0.5},
    {**MAIN, "name": "window100", "window": 100},
    {**MAIN, "name": "update20", "update_every": 20},
]
REQUIRED = ["filename", "time", "used_time(minute)", "gpu_allocated(group)",
            "cpu_util(%)", "mem_util(%)", "gpu_util(%)"]


# --------------------------------------------------------------------------
# Per-container arrays (same arithmetic as sensitivity.first_crossing)
# --------------------------------------------------------------------------
def container_arrays(trace: pd.DataFrame) -> dict:
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
    support = np.minimum(np.diff(times) / 60e9, 1.0)
    active_tail = np.r_[np.cumsum((support * active[:-1])[::-1])[::-1], 0.0]
    idle_tail = np.r_[np.cumsum((support * idle[:-1])[::-1])[::-1], 0.0]
    active_idx = np.flatnonzero(active)
    sel = np.flatnonzero(idle)
    return {"t": times[sel], "age": age[sel], "post": seen_active[sel],
            "idle_tail": idle_tail[sel], "active_tail": active_tail[sel],
            "end": int(times[-1]),
            "last_active": int(times[active_idx[-1]]) if len(active_idx) else -1}


def first_hit(arr: dict, pre: float, post: float) -> int:
    limit = np.where(arr["post"], post, pre)
    hits = np.flatnonzero(arr["age"] >= limit)
    return int(hits[0]) if len(hits) else -1


def build(output: Path = CACHE) -> dict:
    """One trace pass: verify fixed IR2 on all containers, store Test arrays."""
    t0 = time.time()
    output.mkdir(parents=True, exist_ok=True)
    usage = load_usage()
    events = pd.read_csv(EVENTS)
    events["key"] = events.filename.map(normalize_filename)
    events = events.set_index("key", verify_integrity=True)
    parquet = pq.ParquetFile(TRACE)
    containers, chunks = [], {k: [] for k in ("cid", "t", "age", "post", "idle_tail", "active_tail")}
    mismatches = []
    for i in range(parquet.num_row_groups):
        frame = parquet.read_row_group(i, columns=REQUIRED).to_pandas()
        key = normalize_filename(frame.filename.dropna().unique()[0])
        meta = usage.loc[key]
        gpu_bin = "Non-GPU" if meta["gpu_allocated(group)"] == "0.0_GPU" else "GPU-Enabled"
        trace, _ = prepare_trace(frame, gpu_bin, meta.terminated_time)
        arr = container_arrays(trace)
        # Fixed IR2 must reproduce the official online cache exactly.
        hit = first_hit(arr, PRE[gpu_bin], POST[gpu_bin])
        ev = events.loc[key]
        exp_time = pd.to_datetime(ev.online_time)
        got_time = pd.Timestamp(arr["t"][hit]) if hit >= 0 else pd.NaT
        got = {"kind": (None if hit < 0 else ("Post" if arr["post"][hit] else "Pre")),
               "gross": (arr["end"] - arr["t"][hit]) / 60e9 if hit >= 0 else 0.0,
               "idle": arr["idle_tail"][hit] if hit >= 0 else 0.0,
               "active": arr["active_tail"][hit] if hit >= 0 else 0.0,
               "ret": bool(hit >= 0 and arr["last_active"] > arr["t"][hit])}
        exp_kind = ev.online_kind if pd.notna(ev.online_kind) else None
        ok = ((pd.isna(exp_time) and pd.isna(got_time)) or exp_time == got_time) and got["kind"] == exp_kind \
            and got["ret"] == bool(ev.potential_interruption) \
            and np.isclose(got["gross"], ev.online_saved_min, atol=1e-6, rtol=0) \
            and np.isclose(got["idle"], ev.after_idle_min, atol=1e-6, rtol=0) \
            and np.isclose(got["active"], ev.after_active_min, atol=1e-6, rtol=0) \
            and np.isclose(float(trace.elapsed.max()), ev.baseline_min, atol=1e-9, rtol=0)
        if not ok:
            mismatches.append(key)
        if meta.population == "Test":
            cid = len(containers)
            containers.append({"key": key, "gpu_bin": gpu_bin,
                               "cpu": float(meta["cpu_allocated(Core)"]),
                               "gpu": float(meta["gpu_allocated(EA)"]),
                               "baseline_min": float(trace.elapsed.max()),
                               "created_time": meta.created_time,
                               "terminated_time": meta.terminated_time,
                               "end": arr["end"], "last_active": arr["last_active"],
                               "row_group": i})
            n = len(arr["t"])
            chunks["cid"].append(np.full(n, cid, dtype=np.int32))
            for k in ("t", "age", "post", "idle_tail", "active_tail"):
                chunks[k].append(arr[k])
        if (i + 1) % 2500 == 0:
            print(f"[build] {i + 1:,}/{parquet.num_row_groups:,} mismatches={len(mismatches)}", flush=True)
    if mismatches:
        raise AssertionError(f"Fixed IR2 replay differs from official cache: {len(mismatches)} e.g. {mismatches[:3]}")
    cont = pd.DataFrame(containers)
    samples = pd.DataFrame({k: np.concatenate(v) for k, v in chunks.items()})
    # Test cohort size from the cohort contract.
    expected_test = json.loads(CONTRACT.read_text())["population_counts"]["Test"]
    if len(cont) != expected_test or len(events) != parquet.num_row_groups:
        raise AssertionError("Cohort size changed")
    cont.to_parquet(output / "test_containers.parquet", index=False)
    samples.to_parquet(output / "test_idle_samples.parquet", index=False)
    info = {"status": "PASS", "row_groups": parquet.num_row_groups,
            "fixed_ir2_matches_official_cache": f"all {parquet.num_row_groups} containers",
            "test_containers": len(cont), "test_idle_samples": len(samples),
            "seconds": round(time.time() - t0, 1)}
    (output / "build_validation.json").write_text(json.dumps(info, indent=2))
    print("[build]", info, flush=True)
    return info


# --------------------------------------------------------------------------
# tau schedules
# --------------------------------------------------------------------------
def calibration_values(usage: pd.DataFrame) -> pd.DataFrame:
    """Section 4.2 Pre/Post values per normal container (long format)."""
    c = pd.read_csv(CALIB)
    c["key"] = c.filename.map(normalize_filename)
    c = c.loc[c.normal.astype(bool)].copy()
    c["terminated_time"] = usage.loc[c.key, "terminated_time"].to_numpy()
    pre = c.loc[c.pre_idle_min.ge(1), ["key", "population", "gpu_bin", "terminated_time", "pre_idle_min"]]
    post = c.loc[c.post_idle_min.ge(1) & c.tail_idle.astype(bool),
                 ["key", "population", "gpu_bin", "terminated_time", "post_idle_min"]]
    pre = pre.rename(columns={"pre_idle_min": "value"}).assign(phase="Pre")
    post = post.rename(columns={"post_idle_min": "value"}).assign(phase="Post")
    return pd.concat([pre, post], ignore_index=True)


def check_train_calibration(values: pd.DataFrame) -> dict:
    REF = pd.read_csv(CALIB_REFERENCE)
    out = {}
    for (g, ph), s in values.loc[values.population.eq("Train")].groupby(["gpu_bin", "phase"]):
        raw = float(np.percentile(s.value.to_numpy(), PCT[ph]))
        applied = (PRE if ph == "Pre" else POST)[g]
        # Raw percentile values must match the calibration reference (GPU Pre is 119.0; the
        # applied timeout is 119).
        ref = REF.loc[REF.gpu_bin.eq(g) & REF.phase.eq(ph)].iloc[0]
        if not np.isclose(raw, ref.raw_min, atol=1e-6, rtol=0) or len(s) != ref.valid_cases \
                or applied != ref.frozen_applied_min:
            raise AssertionError(f"Train calibration {g}/{ph}: {raw}/{len(s)} vs reference {ref.raw_min}/{ref.valid_cases}")
        out[f"{g}/{ph}"] = {"cases": len(s), "raw": raw, "rounded": int(np.floor(raw + 0.5)), "applied": applied}
    return out


def sample_keys(test_keys: np.ndarray, rate: float, seed: int) -> set:
    """Nested across rates for the same seed (same uniforms)."""
    u = np.random.default_rng(seed).random(len(test_keys))
    return set(test_keys[u < rate])


def build_schedule(values: pd.DataFrame, sampled: set, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Return {(group, phase): (update_times_ns, taus)} with a -inf sentinel."""
    schedule, log = {}, []
    for g in GROUPS:
        for ph in PHASES:
            tau = float((PRE if ph == "Pre" else POST)[g])
            v = values.loc[values.gpu_bin.eq(g) & values.phase.eq(ph)]
            train = v.loc[v.population.eq("Train")].sort_values(["terminated_time", "key"])
            window = list(train.value.to_numpy()[-cfg["window"]:])
            window_times = list(train.terminated_time.to_numpy()[-cfg["window"]:])
            if pd.Timestamp(max(window_times)) >= TEST_START:
                raise AssertionError("Initial window uses data after Test start")
            new = v.loc[v.population.eq("Test") & v.key.isin(sampled)].sort_values(["terminated_time", "key"])
            times, taus = [np.iinfo(np.int64).min], [tau]
            count = 0
            for _, row in new.iterrows():
                window.append(float(row.value))
                window_times.append(row.terminated_time)
                window, window_times = window[-cfg["window"]:], window_times[-cfg["window"]:]
                count += 1
                if count % cfg["update_every"]:
                    continue
                now = pd.Timestamp(row.terminated_time)
                if max(pd.Timestamp(x) for x in window_times) > now:
                    raise AssertionError("Future value in window")
                candidate = float(np.floor(np.percentile(window, PCT[ph]) + 0.5))
                tau = (1 - cfg["alpha"]) * tau + cfg["alpha"] * candidate
                times.append(now.value)
                taus.append(tau)
                log.append({"gpu_bin": g, "phase": ph, "time": now, "candidate": candidate,
                            "tau": tau, "test_values_so_far": count})
            schedule[(g, ph)] = (np.array(times, dtype=np.int64), np.array(taus))
    return schedule, pd.DataFrame(log)


def constant_schedule(pre: dict, post: dict) -> dict:
    sentinel = np.array([np.iinfo(np.int64).min], dtype=np.int64)
    return {(g, ph): (sentinel, np.array([float((pre if ph == "Pre" else post)[g])]))
            for g in GROUPS for ph in PHASES}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(cont: pd.DataFrame, s: pd.DataFrame, schedule: dict, exempt: np.ndarray) -> pd.DataFrame:
    """Per-container outcome; `exempt` containers are never reclaimed."""
    grp = cont.gpu_bin.to_numpy()[s.cid.to_numpy()]
    tau = np.empty(len(s))
    post = s.post.to_numpy()
    t = s.t.to_numpy()
    for (g, ph), (ut, uv) in schedule.items():
        m = (grp == g) & (post == (ph == "Post"))
        tau[m] = uv[np.searchsorted(ut, t[m], side="right") - 1]
    hit = (s.age.to_numpy() >= tau) & ~exempt[s.cid.to_numpy()]
    hi = np.flatnonzero(hit)
    cids, first = np.unique(s.cid.to_numpy()[hi], return_index=True)
    h = hi[first]
    out = pd.DataFrame({"terminated": np.zeros(len(cont), bool), "kind": None,
                        "t_hit": np.full(len(cont), -1, np.int64), "gross": 0.0, "idle": 0.0,
                        "active": 0.0, "ret": False})
    out.loc[cids, "terminated"] = True
    out.loc[cids, "kind"] = np.where(post[h], "Post", "Pre")
    out.loc[cids, "t_hit"] = t[h]
    out.loc[cids, "gross"] = (cont.end.to_numpy()[cids] - t[h]) / 60e9
    out.loc[cids, "idle"] = s.idle_tail.to_numpy()[h]
    out.loc[cids, "active"] = s.active_tail.to_numpy()[h]
    out.loc[cids, "ret"] = cont.last_active.to_numpy()[cids] > t[h]
    out["unknown"] = np.maximum(out.gross - out.idle - out.active, 0.0)
    return out


def aggregate(cont: pd.DataFrame, res: pd.DataFrame, include: np.ndarray) -> list[dict]:
    rows = []
    for g in (*GROUPS, "All"):
        m = include & (cont.gpu_bin.eq(g).to_numpy() if g != "All" else True)
        c, r = cont.loc[m], res.loc[m]
        row = {"gpu_bin": g, "containers": int(m.sum()), "terminated": int(r.terminated.sum()),
               "later_active": int(r.ret.sum())}
        for res_name, alloc in (("cpu", c.cpu), ("gpu", c.gpu)):
            base = float((alloc * c.baseline_min).sum())
            row[f"baseline_{res_name}_min"] = base
            for state in ("gross", "idle", "active", "unknown"):
                row[f"{state}_{res_name}_min"] = float((alloc * r[state]).sum())
            row[f"idle_{res_name}_pct"] = 100 * row[f"idle_{res_name}_min"] / base if base else np.nan
        rows.append(row)
    return rows


def week_bootstrap(cont, a, b, include, group, resource):
    """Paired creation-week block bootstrap of (a - b) in observed-idle pct points."""
    m = include & cont.gpu_bin.eq(group).to_numpy()
    alloc = cont[resource].to_numpy()[m]
    df = pd.DataFrame({"week": cont.created_time[m].dt.to_period("W-SUN").astype(str).to_numpy(),
                       "diff": alloc * (a.idle.to_numpy()[m] - b.idle.to_numpy()[m]),
                       "ret": a.ret.to_numpy()[m].astype(int) - b.ret.to_numpy()[m].astype(int),
                       "base": alloc * cont.baseline_min.to_numpy()[m]})
    blocks = df.groupby("week")[["diff", "ret", "base"]].sum()
    draws = np.random.default_rng(BOOT_SEED).integers(0, len(blocks), size=(BOOT_DRAWS, len(blocks)))
    pct = 100 * blocks["diff"].to_numpy()[draws].sum(1) / blocks.base.to_numpy()[draws].sum(1)
    ret = blocks.ret.to_numpy()[draws].sum(1)
    return {"diff_pp": 100 * df["diff"].sum() / df.base.sum(),
            "diff_pp_lo": float(np.quantile(pct, .025)), "diff_pp_hi": float(np.quantile(pct, .975)),
            "diff_later_active": int(df.ret.sum()),
            "diff_later_active_lo": float(np.quantile(ret, .025)),
            "diff_later_active_hi": float(np.quantile(ret, .975)), "weeks": len(blocks)}


def check_static_against_official(cont, s) -> dict:
    """Fixed IR2 and Static150 on all Test containers must equal 06_policy_summary."""
    official = pd.read_csv(POLICY_SUMMARY)
    official = official.loc[official.population.eq("Test")]
    none = np.zeros(len(cont), bool)
    checks = {}
    for label, sched in (("IR2", constant_schedule(PRE, POST)),
                         ("단일 고정", constant_schedule({g: STATIC for g in GROUPS}, {g: STATIC for g in GROUPS}))):
        mine = pd.DataFrame(aggregate(cont, evaluate(cont, s, sched, none), np.ones(len(cont), bool)))
        for _, row in mine.iterrows():
            ref = official.loc[official.policy.str.startswith(label) & official.gpu_bin.eq(row.gpu_bin)].iloc[0]
            for mine_col, ref_col in (("terminated", "terminated"), ("later_active", "return_containers"),
                                      ("idle_gpu_min", "idle_gpu_min"), ("idle_cpu_min", "idle_cpu_min"),
                                      ("gross_gpu_min", "gross_gpu_min"), ("gross_cpu_min", "gross_cpu_min"),
                                      ("active_gpu_min", "active_gpu_min"), ("unknown_gpu_min", "unknown_gpu_min")):
                if not np.isclose(row[mine_col], ref[ref_col], atol=0.05, rtol=0):
                    raise AssertionError(f"{label}/{row.gpu_bin}/{mine_col}: {row[mine_col]} != {ref[ref_col]}")
        checks[label] = "PASS (06_policy_summary Test, 3 groups x 8 measures)"
    return checks


def hash_tree(root: Path, skip: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_relative_to(skip):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[str(p.relative_to(root))] = h.hexdigest()
    return out


def run(configs=None, output: Path = OUT) -> dict:
    t0 = time.time()
    configs = configs or [MAIN, *SENSITIVITY]
    protected_before = hash_tree(RESULTS_REVISION, OUT)
    usage = load_usage()
    cont = pd.read_parquet(CACHE / "test_containers.parquet")
    s = pd.read_parquet(CACHE / "test_idle_samples.parquet")
    values = calibration_values(usage)
    calib = check_train_calibration(values)
    official = check_static_against_official(cont, s)
    test_keys = np.array(sorted(cont.key))
    key_pos = pd.Series(np.arange(len(cont)), index=cont.key)
    fixed = evaluate(cont, s, constant_schedule(PRE, POST), np.zeros(len(cont), bool))
    static = evaluate(cont, s, constant_schedule({g: STATIC for g in GROUPS}, {g: STATIC for g in GROUPS}),
                      np.zeros(len(cont), bool))
    seed_rows, boot_rows, tau_logs, per_container = [], [], [], []
    for cfg in configs:
        for k in range(N_SEEDS):
            seed = SEED_BASE + k
            sampled = sample_keys(test_keys, cfg["rate"], seed)
            exempt = np.zeros(len(cont), bool)
            exempt[key_pos.loc[list(sampled)].to_numpy()] = True
            schedule, log = build_schedule(values, sampled, cfg)
            tau_logs.append(log.assign(config=cfg["name"], seed=seed))
            adaptive = evaluate(cont, s, schedule, exempt)
            if cfg["name"] == "main" and k == 0:
                per_container.append((schedule, sampled))
            for scope, include in (("non_sample", ~exempt), ("all", np.ones(len(cont), bool))):
                for policy, res in (("Fixed IR2", fixed), ("Static150", static), ("Adaptive IR2", adaptive)):
                    for row in aggregate(cont, res, include):
                        seed_rows.append({"config": cfg["name"], "seed": seed, "scope": scope,
                                          "policy": policy, "samples": len(sampled), **row})
            for other_name, other in (("Fixed IR2", fixed), ("Static150", static)):
                for g, resource in (("GPU-Enabled", "gpu"), ("Non-GPU", "cpu"), ("GPU-Enabled", "cpu")):
                    boot_rows.append({"config": cfg["name"], "seed": seed, "versus": other_name,
                                      "gpu_bin": g, "resource": resource.upper(),
                                      **week_bootstrap(cont, adaptive, other, ~exempt, g, resource)})
        print(f"[run] {cfg['name']} done {time.time() - t0:.0f}s", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    seeds = pd.DataFrame(seed_rows)
    boots = pd.DataFrame(boot_rows)
    taus = pd.concat(tau_logs, ignore_index=True)
    seeds.to_csv(output / "seed_results.csv", index=False)
    boots.to_csv(output / "paired_week_bootstrap.csv", index=False)
    taus.to_csv(output / "tau_updates.csv", index=False)
    metrics = ["containers", "terminated", "later_active", "idle_gpu_pct", "idle_cpu_pct"]
    summary = (seeds.groupby(["config", "scope", "policy", "gpu_bin"])[metrics]
               .agg(["mean", "min", "max"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    independent = scalar_check(cont, *per_container[0], usage)
    protected_after = hash_tree(RESULTS_REVISION, OUT)
    if protected_before != protected_after:
        raise AssertionError("Official results under 05_Results/revision changed")
    validation = {"status": "PASS", "train_calibration": calib, "official_reproduction": official,
                  "scalar_independent_check": independent,
                  "no_future_information": "every window value terminated at or before its update; "
                                           "initial windows end before 2026-03-01",
                  "protected_files_unchanged": len(protected_before),
                  "configs": configs, "seeds": N_SEEDS, "seconds": round(time.time() - t0, 1)}
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str))
    print("[run]", json.dumps({k: v for k, v in validation.items() if k != "configs"},
                              ensure_ascii=False, default=str), flush=True)
    return validation


def scalar_check(cont, schedule, sampled, usage, n: int = 400) -> str:
    """Re-read raw traces and replay a sequential online timer with tau(t)."""
    rng = np.random.default_rng(7)
    picks = rng.choice(len(cont), size=min(n, len(cont)), replace=False)
    exempt = cont.key.isin(sampled).to_numpy()
    res = evaluate(cont, pd.read_parquet(CACHE / "test_idle_samples.parquet"), schedule, exempt)
    parquet = pq.ParquetFile(TRACE)

    def tau_at(g, ph, t):
        ut, uv = schedule[(g, ph)]
        return uv[np.searchsorted(ut, t.value, side="right") - 1]

    for cid in picks:
        row = cont.iloc[cid]
        frame = parquet.read_row_group(int(row.row_group), columns=REQUIRED).to_pandas()
        trace, _ = prepare_trace(frame, row.gpu_bin, usage.loc[row.key, "terminated_time"])
        seen, idle_start, last_active, prev, decided = False, None, None, None, pd.NaT
        if not exempt[cid]:
            for r in trace.itertuples(index=False):
                if r.state == "active":
                    seen, idle_start, last_active = True, None, r.time
                elif r.state == "unknown":
                    idle_start = None
                else:
                    if idle_start is None:
                        idle_start = last_active if seen and prev == "active" else r.time
                    limit = tau_at(row.gpu_bin, "Post" if seen else "Pre", r.time)
                    if (r.time - idle_start).total_seconds() / 60 >= limit:
                        decided = r.time
                        break
                prev = r.state
        got = pd.Timestamp(res.t_hit.iloc[cid]) if res.terminated.iloc[cid] else pd.NaT
        if not ((pd.isna(got) and pd.isna(decided)) or got == decided):
            raise AssertionError(f"Scalar replay mismatch: {row.key}")
    return f"PASS ({len(picks)} random Test containers, main seed {SEED_BASE})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["build", "run", "main-only"])
    args = parser.parse_args()
    if args.step == "build":
        build()
    elif args.step == "main-only":
        run([MAIN])
    else:
        run()
