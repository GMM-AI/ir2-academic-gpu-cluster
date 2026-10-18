"""Additional analyses for Sections 4.4 and 4.5.

B1  Container bootstrap (5,000, ratio of sums, same scheme as notebook 06) for the
    online headline of fixed IR2 and the paired difference IR2 - Static150.
B2  Where the savings beyond the existing 1%-for-1-hour rule come from: the IR2
    idle predicate with a single 60-minute timeout (the existing rule's duration),
    and the best single versus best Pre/Post pair at IR2's later-activity level.
B3  Cluster-wide bound for the GPU headline when non-integer allocations are
    counted in the denominator and contribute nothing.
Inputs are read only; outputs go to 05_Results/revision/additional_analyses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT  # noqa: E402

RES = ROOT / "05_Results/revision"
OUT = RES / "additional_analyses"
IR2 = "IR2 (유형·단계별 / type + phase)"
S150 = "단일 고정 제한시간 (Single static)"
N_BOOT, SEED = 5000, 20261015


def b1():
    d = pd.read_parquet(RES / "06_selected_container_results.parquet")
    rows = []
    for pop in ("Train", "Test"):
        p = d.loc[d.population.eq(pop)]
        a = p.loc[p.policy.eq(IR2)].sort_values("filename").reset_index(drop=True)
        s = p.loc[p.policy.eq(S150)].sort_values("filename").reset_index(drop=True)
        assert (a.filename.to_numpy() == s.filename.to_numpy()).all()
        rng = np.random.default_rng(SEED)
        metrics = {
            "GPU-time, GPU-enabled": ("GPU-Enabled", "idle_gpu_min", "baseline_gpu_min"),
            "CPU-time, non-GPU": ("Non-GPU", "idle_cpu_min", "baseline_cpu_min"),
            "CPU-time, all": (None, "idle_cpu_min", "baseline_cpu_min"),
        }
        for name, (g, num, den) in metrics.items():
            m = np.ones(len(a), bool) if g is None else a.gpu_bin.eq(g).to_numpy()
            xa, xs, b = a.loc[m, num].to_numpy(), s.loc[m, num].to_numpy(), a.loc[m, den].to_numpy()
            n = len(b)
            ea, es = 100 * xa.sum() / b.sum(), 100 * xs.sum() / b.sum()
            ra, rd = np.empty(N_BOOT), np.empty(N_BOOT)
            for i in range(N_BOOT):
                k = rng.integers(0, n, n)
                bb = b[k].sum()
                ra[i] = 100 * xa[k].sum() / bb
                rd[i] = 100 * (xa[k].sum() - xs[k].sum()) / bb
            rows.append({"population": pop, "metric": name, "containers": n, "ir2_pct": ea,
                         "ir2_lo": np.percentile(ra, 2.5), "ir2_hi": np.percentile(ra, 97.5),
                         "static150_pct": es, "diff_pp": ea - es,
                         "diff_lo": np.percentile(rd, 2.5), "diff_hi": np.percentile(rd, 97.5)})
        # later-activity share of containers
        for g in ("GPU-Enabled", "Non-GPU", None):
            m = np.ones(len(a), bool) if g is None else a.gpu_bin.eq(g).to_numpy()
            ya, ys = a.loc[m, "return_containers"].to_numpy(), s.loc[m, "return_containers"].to_numpy()
            n = len(ya)
            ra, rd = np.empty(N_BOOT), np.empty(N_BOOT)
            for i in range(N_BOOT):
                k = rng.integers(0, n, n)
                ra[i] = 100 * ya[k].mean()
                rd[i] = 100 * (ya[k].mean() - ys[k].mean())
            rows.append({"population": pop, "metric": f"later-active containers, {g or 'all'}", "containers": n,
                         "ir2_pct": 100 * ya.mean(), "ir2_lo": np.percentile(ra, 2.5),
                         "ir2_hi": np.percentile(ra, 97.5), "static150_pct": 100 * ys.mean(),
                         "diff_pp": 100 * (ya.mean() - ys.mean()),
                         "diff_lo": np.percentile(rd, 2.5), "diff_hi": np.percentile(rd, 97.5)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "B1_online_bootstrap_ci.csv", index=False)
    return out


def b2():
    g = pd.read_csv(RES / "adaptive_ir2/experiments/static_grid.csv")
    rows = []
    for (per, grp), d in g.groupby(["period", "gpu_bin"]):
        ref = d.loc[d.pre.eq(85 if grp == "Non-GPU" else 119) & d.post.eq(146 if grp == "Non-GPU" else 170)].iloc[0]
        s60 = d.loc[d.single & d.pre.eq(60)].iloc[0]
        ok = d.loc[d.later_active.le(ref.later_active)]
        bs, bp = ok.loc[ok.single].sort_values("reclaim_pct").iloc[-1], ok.sort_values("reclaim_pct").iloc[-1]
        rows.append({"period": per, "gpu_bin": grp,
                     "ir2_pre": ref.pre, "ir2_post": ref.post, "ir2_reclaim": ref.reclaim_pct, "ir2_later_active": ref.later_active,
                     "single60_reclaim": s60.reclaim_pct, "single60_later_active": s60.later_active,
                     "best_single_timeout": bs.pre, "best_single_reclaim": bs.reclaim_pct, "best_single_later_active": bs.later_active,
                     "best_pair": f"{bp.pre}/{bp.post}", "best_pair_reclaim": bp.reclaim_pct, "best_pair_later_active": bp.later_active})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "B2_predicate_vs_timeout.csv", index=False)
    return out


def b3():
    sc = pd.read_csv(RES / "04_fractional_allocation_scope.csv")
    t = sc.loc[sc.period.str.startswith("Test")]
    alloc_all = t.allocated_gpu_min.sum()
    nonint = t.loc[~t.gpu_class.str.startswith(("Integer", "Non-GPU")), "allocated_gpu_min"].sum()
    d = pd.read_parquet(RES / "06_selected_container_results.parquet")
    a = d.loc[d.population.eq("Test") & d.policy.eq(IR2) & d.gpu_bin.eq("GPU-Enabled")]
    out = {"test_allocated_gpu_min_all_classes": alloc_all, "test_nonint_gpu_min": nonint,
           "nonint_share_pct": 100 * nonint / alloc_all,
           "ir2_reclaimed_gpu_min": float(a.idle_gpu_min.sum()), "monitored_integer_baseline": float(a.baseline_gpu_min.sum()),
           "ir2_pct_of_monitored_integer": 100 * a.idle_gpu_min.sum() / a.baseline_gpu_min.sum(),
           "ir2_pct_of_all_allocated_gpu_time": 100 * a.idle_gpu_min.sum() / alloc_all}
    (OUT / "B3_fractional_bound.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    print(b1().round(2).to_string(index=False))
    print(b2().round(2).to_string(index=False))
    print(json.dumps(b3(), indent=1))
