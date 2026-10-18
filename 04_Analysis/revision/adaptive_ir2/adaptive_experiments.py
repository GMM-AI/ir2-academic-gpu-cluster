"""Adaptive IR2 follow-up experiments (Section 4.4).

W  warm start from Train IR2, initial window = last 50 Train values terminated
   before 2025-12-01; W_orig: last 50 Train values by termination time (the adaptive_replay.py run).
C  cold start: tau0 = 60 min (30/300 robustness), empty window, continuous
   replay over Train+Test with 5% samples drawn from all containers.
N  cold start without the Pre/Post split: one tau per group, pooled window,
   pooled P70/P80/P90.
O  hindsight static frontier: Pareto envelope of (later-activity rate,
   observed-idle reclaim %) over a Pre x Post fixed-timeout grid, per period
   and group. Used for comparison only, never to select a policy.

All decisions reuse adaptive_replay.evaluate (official no-gap-reset online
rule with time-varying tau). Outputs go to 05_Results/revision/adaptive_ir2/
experiments only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_replay as ar  # noqa: E402

OUT = ar.OUT / "experiments"
ALL_C = ar.CACHE / "all_containers.parquet"
ALL_S = ar.CACHE / "all_idle_samples.parquet"
CUTOFF = pd.Timestamp("2025-12-01")
POLICY_TIMEOUTS = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 170, 180, 210, 240,
                   300, 360, 480, 600, 720, 960, 1200, 1440, 2160]
GRID = sorted(set(POLICY_TIMEOUTS) | {85, 119, 146} |  # 119: GPU Pre
              set(np.unique(np.round(np.geomspace(10, 4000, 45))).astype(int).tolist()))
BIG = np.iinfo(np.int64).max

BASE = {"rate": 0.05, "alpha": 0.3, "window": 50, "update_every": 10,
        "pct": {"Pre": 90, "Post": 70}, "split": True}
WARM = {**BASE, "family": "W", "init": "ir2", "window_init": "before_cutoff", "scope": "Test"}
COLD = {**BASE, "family": "C", "init": 60.0, "window_init": "empty", "scope": "All"}
SENS = [("rate10", {"rate": 0.10}), ("rate15", {"rate": 0.15}), ("alpha0.1", {"alpha": 0.1}),
        ("alpha0.5", {"alpha": 0.5}), ("window100", {"window": 100}), ("update20", {"update_every": 20})]
CONFIGS = (
    [{**WARM, "name": "W_main"},
     {**WARM, "name": "W_orig", "window_init": "last_by_term"},
     {**WARM, "name": "W_post60", "pct": {"Pre": 90, "Post": 60}},
     {**WARM, "name": "W_post80", "pct": {"Pre": 90, "Post": 80}}]
    + [{**WARM, **chg, "name": f"W_{n}"} for n, chg in SENS]
    + [{**COLD, "name": "C_main"}, {**COLD, "name": "C_init30", "init": 30.0},
       {**COLD, "name": "C_init300", "init": 300.0}]
    + [{**COLD, **chg, "name": f"C_{n}"} for n, chg in SENS]
    + [{**COLD, "name": f"N_pooled{p}", "family": "N", "split": False, "pct": {"All": p}}
       for p in (70, 80, 90)])


# --------------------------------------------------------------------------
def build_all() -> dict:
    """All containers' idle-sample arrays; fixed IR2 re-verified."""
    usage = ar.load_usage()
    events = pd.read_csv(ar.EVENTS)
    events["key"] = events.filename.map(ar.normalize_filename)
    events = events.set_index("key", verify_integrity=True)
    pf = pq.ParquetFile(ar.TRACE)
    rows, chunks = [], {k: [] for k in ("cid", "t", "age", "post", "idle_tail", "active_tail")}
    for i in range(pf.num_row_groups):
        frame = pf.read_row_group(i, columns=ar.REQUIRED).to_pandas()
        key = ar.normalize_filename(frame.filename.dropna().unique()[0])
        meta = usage.loc[key]
        g = "Non-GPU" if meta["gpu_allocated(group)"] == "0.0_GPU" else "GPU-Enabled"
        trace, _ = ar.prepare_trace(frame, g, meta.terminated_time)
        a = ar.container_arrays(trace)
        rows.append({"key": key, "period": meta.population, "gpu_bin": g,
                     "cpu": float(meta["cpu_allocated(Core)"]), "gpu": float(meta["gpu_allocated(EA)"]),
                     "baseline_min": float(trace.elapsed.max()), "created_time": meta.created_time,
                     "terminated_time": meta.terminated_time, "end": a["end"],
                     "last_active": a["last_active"], "row_group": i})
        chunks["cid"].append(np.full(len(a["t"]), len(rows) - 1, np.int32))
        for k in ("t", "age", "post", "idle_tail", "active_tail"):
            chunks[k].append(a[k])
    cont = pd.DataFrame(rows)
    s = pd.DataFrame({k: np.concatenate(v) for k, v in chunks.items()})
    # Fixed IR2 must reproduce the official cache for every container.
    res = ar.evaluate(cont, s, ar.constant_schedule(ar.PRE, ar.POST), np.zeros(len(cont), bool))
    ev = events.loc[cont.key]
    exp_t = pd.to_datetime(ev.online_time).to_numpy(dtype="datetime64[ns]").astype("int64")
    exp_t = np.where(ev.online_time.isna().to_numpy(), -1, exp_t)
    checks = {"time": np.array_equal(np.where(res.terminated, res.t_hit, -1), exp_t),
              "kind": np.array_equal(res.kind.fillna("").replace({"Post": "Post", "Pre": "Pre"}).to_numpy(),
                                     ev.online_kind.fillna("").to_numpy()),
              "later_active": np.array_equal(res.ret.to_numpy(), ev.potential_interruption.astype(bool).to_numpy()),
              "idle": np.allclose(res.idle, ev.after_idle_min, atol=1e-6, rtol=0),
              "active": np.allclose(res.active, ev.after_active_min, atol=1e-6, rtol=0),
              "baseline": np.allclose(cont.baseline_min, ev.baseline_min, atol=1e-9, rtol=0)}
    if not all(checks.values()):
        raise AssertionError(f"Fixed IR2 replay differs from official cache: {checks}")
    cont.to_parquet(ALL_C, index=False)
    s.to_parquet(ALL_S, index=False)
    return {"containers": len(cont), "idle_samples": len(s), "fixed_ir2_vs_official": f"PASS all {len(cont)}"}


# --------------------------------------------------------------------------
def make_schedule(values: pd.DataFrame, sampled: set, cfg: dict) -> tuple[dict, pd.DataFrame]:
    units = [(g, ph) for g in ar.GROUPS for ph in ar.PHASES] if cfg["split"] else [(g, "All") for g in ar.GROUPS]
    schedule, log = {}, []
    for g, ph in units:
        v = values.loc[values.gpu_bin.eq(g) & (values.phase.eq(ph) if ph != "All" else True)]
        if cfg["init"] == "ir2":
            tau = float((ar.PRE if ph == "Pre" else ar.POST)[g])
        else:
            tau = float(cfg["init"])
        train = v.loc[v.population.eq("Train")].sort_values(["terminated_time", "key"])
        if cfg["window_init"] == "before_cutoff":
            train = train.loc[pd.to_datetime(train.terminated_time) < CUTOFF]
        if cfg["window_init"] == "empty":
            window, wtimes = [], []
        else:
            window = list(train.value.to_numpy()[-cfg["window"]:])
            wtimes = [pd.Timestamp(x) for x in train.terminated_time.to_numpy()[-cfg["window"]:]]
            if max(wtimes) >= ar.TEST_START:
                raise AssertionError("Initial window uses data after Test start")
        pool = v.loc[v.key.isin(sampled)]
        if cfg["scope"] == "Test":
            pool = pool.loc[pool.population.eq("Test")]
        pool = pool.sort_values(["terminated_time", "key"])
        times, taus, count = [np.iinfo(np.int64).min], [tau], 0
        p = cfg["pct"][ph]
        for value, term in zip(pool.value.to_numpy(), pool.terminated_time.to_numpy()):
            now = pd.Timestamp(term)
            window = (window + [float(value)])[-cfg["window"]:]
            wtimes = (wtimes + [now])[-cfg["window"]:]
            count += 1
            if count % cfg["update_every"]:
                continue
            if max(wtimes) > now:
                raise AssertionError("Future value in window")
            candidate = float(np.floor(np.percentile(window, p) + 0.5))
            tau = (1 - cfg["alpha"]) * tau + cfg["alpha"] * candidate
            times.append(now.value)
            taus.append(tau)
            log.append({"gpu_bin": g, "phase": ph, "time": now, "candidate": candidate, "tau": tau,
                        "values_so_far": count, "window_n": len(window)})
        arr = (np.array(times, dtype=np.int64), np.array(taus))
        for phase in (ar.PHASES if ph == "All" else (ph,)):
            schedule[(g, phase)] = arr
    return schedule, pd.DataFrame(log)


def metrics(cont, res, include) -> list[dict]:
    out = []
    for pop in ("Train", "Test"):
        m = include & cont["period"].eq(pop).to_numpy()
        for row in ar.aggregate(cont, res, m):
            row["period"] = pop
            row["later_active_rate"] = row["later_active"] / row["containers"] if row["containers"] else np.nan
            row["reclaim_pct"] = row["idle_gpu_pct"] if row["gpu_bin"] == "GPU-Enabled" else row["idle_cpu_pct"]
            out.append(row)
    return out


# --------------------------------------------------------------------------
def static_grid(cont, s) -> pd.DataFrame:
    cid, post, age, t = s.cid.to_numpy(), s.post.to_numpy(), s.age.to_numpy(), s.t.to_numpy()
    idle_tail = s.idle_tail.to_numpy()
    n = len(cont)
    first = {}
    for flag in (False, True):
        for tau in GRID:
            hi = np.flatnonzero((post == flag) & (age >= tau))
            u, f = np.unique(cid[hi], return_index=True)
            arr = np.full(n, BIG, dtype=np.int64)
            arr[u] = hi[f]
            first[(flag, tau)] = arr
    last_active = cont.last_active.to_numpy()
    rows = []
    keys = [(p, g) for p in ("Train", "Test") for g in ar.GROUPS]
    masks = {k: (cont["period"].eq(k[0]) & cont.gpu_bin.eq(k[1])).to_numpy() for k in keys}
    alloc = {k: (cont.gpu if k[1] == "GPU-Enabled" else cont.cpu).to_numpy()[masks[k]] for k in keys}
    base = {k: float((alloc[k] * cont.baseline_min.to_numpy()[masks[k]]).sum()) for k in keys}
    for pre in GRID:
        for po in GRID:
            h = np.minimum(first[(False, pre)], first[(True, po)])
            valid = h < BIG
            hs = np.where(valid, h, 0)
            idle = np.where(valid, idle_tail[hs], 0.0)
            ret = valid & (last_active > t[hs])
            for k in keys:
                m = masks[k]
                rows.append({"period": k[0], "gpu_bin": k[1], "pre": pre, "post": po,
                             "single": pre == po, "terminated": int(valid[m].sum()),
                             "later_active": int(ret[m].sum()),
                             "later_active_rate": ret[m].mean(),
                             "reclaim_pct": 100 * float((alloc[k] * idle[m]).sum()) / base[k]})
    return pd.DataFrame(rows)


def envelope(grid: pd.DataFrame, pop, g, single=False):
    d = grid.loc[grid["period"].eq(pop) & grid.gpu_bin.eq(g) & (grid.single if single else True)]
    d = d.sort_values(["later_active_rate", "reclaim_pct"], ascending=[True, False])
    x = d.later_active_rate.to_numpy()
    y = np.maximum.accumulate(d.reclaim_pct.to_numpy())
    return x, y


def frontier_gap(grid, pop, g, rate, reclaim, single=False):
    """reclaim minus best static reclaim at <= the same later-activity rate."""
    x, y = envelope(grid, pop, g, single)
    i = np.searchsorted(x, rate + 1e-12, side="right") - 1
    return reclaim - (y[i] if i >= 0 else np.nan)


# --------------------------------------------------------------------------
def run() -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    protected_before = ar.hash_tree(ar.RESULTS_REVISION, ar.OUT)
    info = build_all() if not (ALL_C.is_file() and ALL_S.is_file()) else {"cache": "reused"}
    cont = pd.read_parquet(ALL_C)
    s = pd.read_parquet(ALL_S)
    usage = ar.load_usage()
    values = ar.calibration_values(usage)
    ar.check_train_calibration(values)
    none = np.zeros(len(cont), bool)
    every = np.ones(len(cont), bool)
    fixed = ar.evaluate(cont, s, ar.constant_schedule(ar.PRE, ar.POST), none)
    static = ar.evaluate(cont, s, ar.constant_schedule({g: ar.STATIC for g in ar.GROUPS},
                                                       {g: ar.STATIC for g in ar.GROUPS}), none)
    grid = static_grid(cont, s)
    grid.to_csv(OUT / "static_grid.csv", index=False)
    # Grid point (150,150) and (IR2) must equal the direct evaluations and the official Static150.
    official = pd.read_csv(ar.POLICY_SUMMARY)
    for res, pre_t, post_t, label in ((static, 150, 150, "단일"), (fixed, None, None, "IR2")):
        for r in metrics(cont, res, every):
            if r["gpu_bin"] == "All":
                continue
            if pre_t is not None:
                gp = grid.loc[grid.period.eq(r["period"]) & grid.gpu_bin.eq(r["gpu_bin"])
                              & grid.pre.eq(pre_t) & grid.post.eq(post_t)].iloc[0]
            else:
                gp = grid.loc[grid.period.eq(r["period"]) & grid.gpu_bin.eq(r["gpu_bin"])
                              & grid.pre.eq(int(ar.PRE[r["gpu_bin"]])) & grid.post.eq(int(ar.POST[r["gpu_bin"]]))].iloc[0]
            if gp.later_active != r["later_active"] or not np.isclose(gp.reclaim_pct, r["reclaim_pct"], atol=1e-9):
                raise AssertionError(f"Grid point mismatch {label} {r['period']} {r['gpu_bin']}")
            o = official.loc[official.population.eq(r["period"]) & official.policy.str.startswith(label)
                             & official.gpu_bin.eq(r["gpu_bin"])].iloc[0]
            if o.return_containers != r["later_active"] or not np.isclose(o.idle_cpu_min, r["idle_cpu_min"], atol=0.05):
                raise AssertionError(f"Official mismatch {label} {r['period']} {r['gpu_bin']}")
    keys_all = np.array(sorted(cont.key))
    keys_test = np.array(sorted(cont.key[cont["period"].eq("Test")]))
    key_pos = pd.Series(np.arange(len(cont)), index=cont.key)
    rows, logs, boots = [], [], []
    for pol, res in (("Fixed IR2", fixed), ("Static150", static)):
        for scope_rows in metrics(cont, res, every):
            rows.append({"config": "reference", "seed": -1, "scope": "all", "policy": pol, **scope_rows})
    for cfg in CONFIGS:
        keys = keys_test if cfg["scope"] == "Test" else keys_all
        for k in range(ar.N_SEEDS):
            seed = ar.SEED_BASE + k
            sampled = ar.sample_keys(keys, cfg["rate"], seed)
            exempt = np.zeros(len(cont), bool)
            exempt[key_pos.loc[list(sampled)].to_numpy()] = True
            sched, log = make_schedule(values, sampled, cfg)
            logs.append(log.assign(config=cfg["name"], seed=seed))
            res = ar.evaluate(cont, s, sched, exempt)
            for scope, inc in (("non_sample", ~exempt), ("all", every)):
                for r in metrics(cont, res, inc):
                    rows.append({"config": cfg["name"], "seed": seed, "scope": scope, "policy": "Adaptive IR2", **r})
                for pol, other in (("Fixed IR2", fixed), ("Static150", static)):
                    if scope == "non_sample":
                        for r in metrics(cont, other, inc):
                            rows.append({"config": cfg["name"], "seed": seed, "scope": scope, "policy": pol, **r})
            if cfg["name"] in ("W_main", "C_main"):
                for pol, other in (("Fixed IR2", fixed), ("Static150", static)):
                    for g, rs in (("GPU-Enabled", "gpu"), ("Non-GPU", "cpu")):
                        inc = ~exempt & cont["period"].eq("Test").to_numpy()
                        boots.append({"config": cfg["name"], "seed": seed, "versus": pol, "gpu_bin": g,
                                      **ar.week_bootstrap(cont, res, other, inc, g, rs)})
        print(f"[exp] {cfg['name']} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df = df.loc[df.gpu_bin.ne("All")].copy()
    df["gap_to_frontier"] = [frontier_gap(grid, r["period"], r.gpu_bin, r.later_active_rate, r.reclaim_pct)
                             for _, r in df.iterrows()]
    df["gap_to_single_frontier"] = [frontier_gap(grid, r["period"], r.gpu_bin, r.later_active_rate, r.reclaim_pct, True)
                                    for _, r in df.iterrows()]
    df.to_csv(OUT / "results_by_seed.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(OUT / "tau_updates.csv", index=False)
    pd.DataFrame(boots).to_csv(OUT / "paired_week_bootstrap.csv", index=False)
    summ = (df.groupby(["config", "scope", "policy", "period", "gpu_bin"])
            [["reclaim_pct", "later_active", "later_active_rate", "terminated", "gap_to_frontier",
              "gap_to_single_frontier"]].agg(["mean", "min", "max"]))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / "summary.csv", index=False)
    # W_orig must reproduce the adaptive_replay.py result.
    first = pd.read_csv(ar.OUT / "seed_results.csv")
    first = first.loc[first.config.eq("main") & first.scope.eq("non_sample") & first.policy.eq("Adaptive IR2")
                      & first.gpu_bin.ne("All")]
    mine = df.loc[df.config.eq("W_orig") & df.scope.eq("non_sample") & df.policy.eq("Adaptive IR2")
                  & df.period.eq("Test")]
    m = first.merge(mine, on=["seed", "gpu_bin"], suffixes=("_a", "_b"))
    if len(m) != 40 or not (np.allclose(m.idle_gpu_min_a, m.idle_gpu_min_b) and (m.later_active_a == m.later_active_b).all()):
        raise AssertionError("W_orig does not reproduce the adaptive_replay.py result")
    figures(df, grid, values, pd.concat(logs, ignore_index=True))
    if protected_before != ar.hash_tree(ar.RESULTS_REVISION, ar.OUT):
        raise AssertionError("Official results changed")
    val = {"status": "PASS", **info, "static150_grid_and_official": "PASS",
           "w_orig_reproduces_first_run": "PASS (20 seeds x 2 groups)",
           "protected_files_unchanged": len(protected_before), "grid_timeouts": GRID,
           "configs": [{k: v for k, v in c.items()} for c in CONFIGS], "seconds": round(time.time() - t0, 1)}
    (OUT / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=2, default=str))
    print("[exp] PASS", round(time.time() - t0), "s", flush=True)
    return val


def figures(df, grid, values, logs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Frontier figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    marks = {"Fixed IR2": ("s", "#222"), "Static150": ("D", "#888")}
    for i, pop in enumerate(("Train", "Test")):
        for j, g in enumerate(ar.GROUPS):
            a = axes[i, j]
            d = grid.loc[grid["period"].eq(pop) & grid.gpu_bin.eq(g)]
            a.scatter(100 * d.later_active_rate, d.reclaim_pct, s=3, color="#ccc", label="static Pre x Post grid")
            x, y = envelope(grid, pop, g)
            a.step(100 * x, y, where="post", color="#555", lw=1.2, label="best static (hindsight)")
            x1, y1 = envelope(grid, pop, g, True)
            a.step(100 * x1, y1, where="post", color="#555", lw=1, ls=":", label="best single static")
            ref = df.loc[df.config.eq("reference") & df.period.eq(pop) & df.gpu_bin.eq(g)]
            for _, r in ref.iterrows():
                mk, col = marks[r.policy]
                a.plot(100 * r.later_active_rate, r.reclaim_pct, mk, color=col, ms=8, label=r.policy)
            for cfg, col in (("W_main", "#2a9d8f"), ("C_main", "#e76f51"), ("N_pooled80", "#6a4c93")):
                if cfg.startswith("W") and pop == "Train":
                    continue
                q = df.loc[df.config.eq(cfg) & df.scope.eq("non_sample") & df.policy.eq("Adaptive IR2")
                           & df.period.eq(pop) & df.gpu_bin.eq(g)]
                a.scatter(100 * q.later_active_rate, q.reclaim_pct, s=14, color=col, alpha=.7,
                          label={"W_main": "adaptive, warm (20 seeds)", "C_main": "adaptive, cold 60 min",
                                 "N_pooled80": "adaptive, no Pre/Post split"}[cfg])
            a.set_title(f"{pop} | {g}")
            a.set_xlabel("containers with later activity after decision (% of containers)")
            a.set_ylabel(("GPU" if g == "GPU-Enabled" else "CPU") + "-time observed idle reclaimed (%)")
            a.grid(alpha=.3)
            q = df.loc[df.period.eq(pop) & df.gpu_bin.eq(g) & df.policy.eq("Adaptive IR2")]
            lo = min(q.later_active_rate.min(), ref.later_active_rate.min()) * 100
            hi = max(q.later_active_rate.max(), ref.later_active_rate.max()) * 100
            a.set_xlim(max(lo - 6, 0), hi + 6)
            a.legend(fontsize=7, loc="lower right")
    fig.suptitle("Adaptive IR2 versus the hindsight static frontier (non-sample containers for adaptive)")
    fig.tight_layout()
    fig.savefig(OUT / "frontier.png", dpi=130)
    # tau trajectories (cold start)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    vv = values.copy()
    vv["month"] = pd.to_datetime(vv.terminated_time).dt.to_period("M").dt.to_timestamp() + pd.Timedelta(days=14)
    for idx, (g, ph) in enumerate([(g, ph) for g in ar.GROUPS for ph in ar.PHASES]):
        a = axes.flat[idx]
        for cfg, col, lw in (("C_main", "#e76f51", .6), ("C_init30", "#457b9d", .6), ("C_init300", "#2a9d8f", .6)):
            sel = logs.loc[logs.config.eq(cfg) & logs.gpu_bin.eq(g) & logs.phase.eq(ph)]
            for seed, lg in sel.groupby("seed"):
                init = {"C_main": 60, "C_init30": 30, "C_init300": 300}[cfg]
                xs = [pd.Timestamp("2024-09-26")] + list(pd.to_datetime(lg.time))
                ys = [init] + list(lg.tau)
                a.step(xs, ys, where="post", color=col, lw=lw, alpha=.5,
                       label=f"cold start {init} min" if seed == ar.SEED_BASE else None)
        m = vv.loc[vv.gpu_bin.eq(g) & vv.phase.eq(ph)].groupby("month").value
        q = m.apply(lambda x: np.percentile(x, ar.PCT[ph]) if len(x) >= 30 else np.nan)
        a.plot(q.index, q.values, "ko", ms=4, label=f"monthly P{ar.PCT[ph]}, all containers (reference)")
        a.axhline((ar.PRE if ph == "Pre" else ar.POST)[g], color="#555", ls="--", lw=1, label="fixed IR2")
        a.axvspan(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-01"), color="#eee")
        a.set_yscale("log")
        a.set_title(f"{g} | {ph} (target P{ar.PCT[ph]})")
        a.set_ylabel("timeout tau (min)")
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    fig.suptitle("Cold-start adaptive IR2: tau over time (20 seeds each; grey = no data Dec-Feb)")
    fig.tight_layout()
    fig.savefig(OUT / "tau_trajectories.png", dpi=130)


if __name__ == "__main__":
    run()
