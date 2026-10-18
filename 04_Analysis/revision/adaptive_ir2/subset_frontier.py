"""Frontier gaps on the same container subset.

The static Pre x Post grid is re-aggregated on each seed's non-sample (95%)
subset, so adaptive and comparators are measured against a frontier built from
exactly the containers they are evaluated on. Also writes the reclaim-concentration
table and the Post trade-off curve (_diag).

Read-only on inputs; writes experiments/results_by_seed_subset_gap.csv,
experiments/subset_gap_summary.csv, experiments/concentration.csv and
_diag/post_tradeoff_curve.csv/png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_experiments as ex  # noqa: E402

ar = ex.ar
OUT = ex.OUT
DIAG = ar.OUT / "_diag"


def grid_matrices(cont, s):
    """Per (pre, post) pair: per-container observed idle minutes and later activity."""
    cid, post, age, t = s.cid.to_numpy(), s.post.to_numpy(), s.age.to_numpy(), s.t.to_numpy()
    idle_tail = s.idle_tail.to_numpy()
    n = len(cont)
    first = {}
    for flag in (False, True):
        for tau in ex.GRID:
            hi = np.flatnonzero((post == flag) & (age >= tau))
            u, f = np.unique(cid[hi], return_index=True)
            arr = np.full(n, ex.BIG, dtype=np.int64)
            arr[u] = hi[f]
            first[(flag, tau)] = arr
    pairs = [(a, b) for a in ex.GRID for b in ex.GRID]
    idle = np.zeros((len(pairs), n), np.float32)
    ret = np.zeros((len(pairs), n), bool)
    last_active = cont.last_active.to_numpy()
    for k, (a, b) in enumerate(pairs):
        h = np.minimum(first[(False, a)], first[(True, b)])
        valid = h < ex.BIG
        hs = np.where(valid, h, 0)
        idle[k] = np.where(valid, idle_tail[hs], 0.0)
        ret[k] = valid & (last_active > t[hs])
    return pairs, idle, ret


def envelope_from(rate, reclaim):
    order = np.lexsort((-reclaim, rate))
    return rate[order], np.maximum.accumulate(reclaim[order])


def gap(x, y, r, v):
    i = np.searchsorted(x, r + 1e-12, side="right") - 1
    return v - (y[i] if i >= 0 else np.nan)


def main():
    cont = pd.read_parquet(ex.ALL_C)
    s = pd.read_parquet(ex.ALL_S)
    pairs, idle, ret = grid_matrices(cont, s)
    res = pd.read_csv(OUT / "results_by_seed.csv")
    keys_all = np.array(sorted(cont.key))
    keys_test = np.array(sorted(cont.key[cont.period.eq("Test")]))
    key_pos = pd.Series(np.arange(len(cont)), index=cont.key)
    cfgs = {c["name"]: c for c in ex.CONFIGS}
    rows = []
    cache = {}
    for (cfg_name, seed), part in res.loc[res.scope.eq("non_sample")].groupby(["config", "seed"]):
        cfg = cfgs[cfg_name]
        keys = keys_test if cfg["scope"] == "Test" else keys_all
        ck = (cfg["scope"], cfg["rate"], seed)
        if ck not in cache:
            sampled = ar.sample_keys(keys, cfg["rate"], seed)
            keep = np.ones(len(cont), bool)
            keep[key_pos.loc[list(sampled)].to_numpy()] = False
            env = {}
            for per in ("Train", "Test"):
                for g in ar.GROUPS:
                    m = keep & (cont.period.eq(per) & cont.gpu_bin.eq(g)).to_numpy()
                    alloc = (cont.gpu if g == "GPU-Enabled" else cont.cpu).to_numpy() * m
                    base = float((alloc * cont.baseline_min.to_numpy()).sum())
                    rec = 100 * (idle @ alloc.astype(np.float32)) / base
                    rate = ret[:, m].mean(axis=1)
                    env[(per, g)] = envelope_from(rate, rec.astype(float))
            cache[ck] = env
        env = cache[ck]
        for _, r in part.iterrows():
            x, y = env[(r.period, r.gpu_bin)]
            rows.append({**r.to_dict(), "gap_subset_frontier": gap(x, y, r.later_active_rate, r.reclaim_pct)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "results_by_seed_subset_gap.csv", index=False)
    summ = (out.groupby(["config", "policy", "period", "gpu_bin"]).gap_subset_frontier
            .agg(["mean", "min", "max"]).reset_index())
    summ.to_csv(OUT / "subset_gap_summary.csv", index=False)
    # Reclaim concentration under fixed IR2.
    fixed = ar.evaluate(cont, s, ar.constant_schedule(ar.PRE, ar.POST), np.zeros(len(cont), bool))
    conc = []
    for per in ("Train", "Test"):
        for g, rs in (("GPU-Enabled", "gpu"), ("Non-GPU", "cpu")):
            m = (cont.period.eq(per) & cont.gpu_bin.eq(g)).to_numpy()
            v = np.sort((cont[rs] * fixed.idle).to_numpy()[m])[::-1]
            row = {"period": per, "gpu_bin": g, "containers": len(v)}
            for k in (1, 5, 10):
                row[f"top{k}_share_pct"] = 100 * v[:max(1, int(round(len(v) * k / 100)))].sum() / v.sum()
            conc.append(row)
    pd.DataFrame(conc).to_csv(OUT / "concentration.csv", index=False)
    print(summ.loc[summ.config.isin(["W_main", "W_orig", "C_main", "C_init30", "C_init300", "N_pooled80"])
                   & summ.period.eq("Test")].round(2).to_string())
    print(pd.DataFrame(conc).round(1).to_string())


if __name__ == "__main__":
    main()
