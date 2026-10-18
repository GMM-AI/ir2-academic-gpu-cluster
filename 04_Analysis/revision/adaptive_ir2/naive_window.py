"""Naive rolling window without observation samples.

Same update rule as adaptive IR2 (window N=50, update every k=10 values, EWMA
alpha=0.3, initial timeouts 85/119/146/170 and initial window = last 50 Train
values), but every Test container is subject to reclamation and the window
receives what an operator would observe under the policy:
  - reclaimed in the pre phase  -> pre idle observed up to the decision (censored)
  - reclaimed in the post phase -> post idle observed up to the decision (censored),
                                   plus the complete pre value (activity was seen)
  - not reclaimed               -> the same complete values as adaptive IR2 uses
Because decisions feed back into the window, the replay is event driven: each
container's next decision is recomputed for its group whenever a timeout changes.
Compared with adaptive IR2 (5% observation samples) on the same Test containers.
"""

from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_experiments as ex  # noqa: E402
import adaptive_replay as ar  # noqa: E402

OUT = ar.OUT / "experiments" / "naive_window"
N, K, ALPHA = 50, 10, 0.3
PCT = {"Pre": 90, "Post": 70}
INF = np.iinfo(np.int64).max


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cont = pd.read_parquet(ex.ALL_C)
    s = pd.read_parquet(ex.ALL_S)
    values = ar.calibration_values(ar.load_usage())
    test = cont.period.eq("Test").to_numpy()
    tcid = np.flatnonzero(test)
    # Complete (uncensored) values per container key.
    full = {(r.key, r.phase): float(r.value) for r in values.itertuples(index=False)}
    term = pd.to_datetime(cont.terminated_time).astype("datetime64[ns]").astype("int64").to_numpy()
    # Idle samples of Test containers, grouped per container.
    s = s.loc[test[s.cid.to_numpy()]]
    order = np.argsort(s.cid.to_numpy(), kind="stable")
    cid_s, t_s, age_s, post_s = (s.cid.to_numpy()[order], s.t.to_numpy()[order],
                                 s.age.to_numpy()[order], s.post.to_numpy()[order])
    start = np.searchsorted(cid_s, np.arange(len(cont)))
    stop = np.searchsorted(cid_s, np.arange(len(cont)), side="right")

    tau, window, count, log = {}, {}, {}, []
    for g in ar.GROUPS:
        for ph in ar.PHASES:
            tau[(g, ph)] = float((ar.PRE if ph == "Pre" else ar.POST)[g])
            v = values.loc[values.gpu_bin.eq(g) & values.phase.eq(ph) & values.population.eq("Train")]
            v = v.loc[pd.to_datetime(v.terminated_time) < ex.CUTOFF].sort_values(["terminated_time", "key"])
            window[(g, ph)] = list(v.value.to_numpy()[-N:])
            count[(g, ph)] = 0
            log.append({"gpu_bin": g, "phase": ph, "time": pd.Timestamp(ar.TEST_START), "tau": tau[(g, ph)],
                        "candidate": np.nan, "censored_share": np.nan})
    censored_flags = {k: [] for k in window}
    grp = cont.gpu_bin.to_numpy()

    def next_event(c, after):
        """(time, sample index or -1) of container c's end under current timeouts, after `after`."""
        a, b = start[c], stop[c]
        if a < b:
            g = grp[c]
            lim = np.where(post_s[a:b], tau[(g, "Post")], tau[(g, "Pre")])
            hit = np.flatnonzero((age_s[a:b] >= lim) & (t_s[a:b] > after))
            if len(hit):
                return int(t_s[a + hit[0]]), a + hit[0]
        return int(term[c]), -1

    # Convention (shared with evaluate below): an update at time T applies to samples after T.
    version = np.zeros(len(cont), np.int64)
    pend = np.full(len(cont), INF, np.int64)
    done = np.zeros(len(cont), bool)
    outcome = {}
    heap = []
    for c in tcid:
        t_end, h = next_event(c, -INF)
        heapq.heappush(heap, (t_end, int(c), 0, h))
        pend[c] = t_end
    while heap:
        t_end, c, ver, h = heapq.heappop(heap)
        if done[c] or ver != version[c]:
            continue
        done[c] = True
        g, key = grp[c], cont.key.iat[c]
        obs = []
        if h >= 0:
            ph = "Post" if post_s[h] else "Pre"
            outcome[c] = (ph, t_end)
            obs.append((ph, float(age_s[h]), True))
            if ph == "Post" and (key, "Pre") in full:
                obs.append(("Pre", full[(key, "Pre")], False))
        else:
            outcome[c] = (None, t_end)
            for ph in ar.PHASES:
                if (key, ph) in full:
                    obs.append((ph, full[(key, ph)], False))
        changed = False
        for ph, val, cens in obs:
            u = (g, ph)
            window[u] = (window[u] + [val])[-N:]
            censored_flags[u] = (censored_flags[u] + [cens])[-N:]
            count[u] += 1
            if count[u] % K:
                continue
            cand = float(np.floor(np.percentile(window[u], PCT[ph]) + 0.5))
            tau[u] = (1 - ALPHA) * tau[u] + ALPHA * cand
            changed = True
            log.append({"gpu_bin": g, "phase": ph, "time": pd.Timestamp(t_end), "tau": tau[u], "candidate": cand,
                        "censored_share": float(np.mean(censored_flags[u])) if censored_flags[u] else np.nan})
        if changed:
            # Containers already due at or before T keep their decision made under the old timeout.
            for c2 in tcid[(grp[tcid] == g) & ~done[tcid] & (pend[tcid] > t_end)]:
                version[c2] += 1
                t2, h2 = next_event(c2, t_end)
                heapq.heappush(heap, (t2, int(c2), int(version[c2]), h2))
                pend[c2] = t2
    taus = pd.DataFrame(log)
    taus.to_csv(OUT / "naive_tau_updates.csv", index=False)

    # Evaluate the naive outcome with the same accounting as evaluate().
    hit_idx = np.full(len(cont), -1, np.int64)
    sched = {}
    for (g, ph), d in taus.groupby(["gpu_bin", "phase"]):
        d = d.sort_values("time")
        times = d.time.astype("datetime64[ns]").astype("int64").to_numpy()
        times = times + 1  # apply after T, as in the event replay
        times[0] = np.iinfo(np.int64).min
        sched[(g, ph)] = (times, d.tau.to_numpy())
    res = ar.evaluate(cont, pd.read_parquet(ex.ALL_S), sched, ~test)
    # Cross-check: evaluate() with the final step-function must reproduce the event replay.
    mism = 0
    for c in tcid:
        ph, t_end = outcome[c]
        if (ph is None) != (not res.terminated.iat[c]) or (ph is not None and res.t_hit.iat[c] != t_end):
            mism += 1
    rows = ex.metrics(cont, res, test)
    summary = pd.DataFrame([{**r, "policy": "Naive window"} for r in rows if r["period"] == "Test"])
    final = {f"{g}/{ph}": round(float(d.sort_values('time').tau.iloc[-1]), 1)
             for (g, ph), d in taus.groupby(["gpu_bin", "phase"])}
    info = {"status": "PASS" if mism == 0 else "CHECK", "event_replay_vs_evaluate_mismatches": int(mism),
            "final_tau": final, "updates": int(len(taus) - 4)}
    summary.to_csv(OUT / "naive_summary.csv", index=False)
    (OUT / "validation.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(summary[["gpu_bin", "containers", "terminated", "later_active", "reclaim_pct"]].round(2).to_string(index=False))
    return taus, summary




def plot():
    """Timeout trajectories in the later period: naive window vs adaptive IR2 (W_main, 20 seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "font.family": "DejaVu Sans"})
    naive = pd.read_csv(OUT / "naive_tau_updates.csv", parse_dates=["time"])
    ad = pd.read_csv(ar.OUT / "experiments" / "tau_updates.csv", parse_dates=["time"])
    ad = ad.loc[ad.config.eq("W_main")]
    values = ar.calibration_values(ar.load_usage())
    days = pd.date_range(ar.TEST_START, "2026-09-01", freq="D")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    rows = []
    for i, g in enumerate(ar.GROUPS):
        for j, ph in enumerate(ar.PHASES):
            a = axes[i, j]
            start = float((ar.PRE if ph == "Pre" else ar.POST)[g])
            paths = []
            for seed, d in ad.loc[ad.gpu_bin.eq(g) & ad.phase.eq(ph)].groupby("seed"):
                d = d.sort_values("time")
                idx = np.searchsorted(d.time.to_numpy(), days.to_numpy(), side="right") - 1
                paths.append(np.where(idx >= 0, d.tau.to_numpy()[np.maximum(idx, 0)], start))
            paths = np.array(paths)
            med, lo, hi = np.median(paths, 0), paths.min(0), paths.max(0)
            n = naive.loc[naive.gpu_bin.eq(g) & naive.phase.eq(ph)].sort_values("time")
            idx = np.searchsorted(n.time.to_numpy(), days.to_numpy(), side="right") - 1
            npath = n.tau.to_numpy()[np.maximum(idx, 0)]
            v = values.loc[values.gpu_bin.eq(g) & values.phase.eq(ph) & values.population.eq("Test"), "value"]
            true_q = float(np.percentile(v, PCT[ph]))
            a.fill_between(days, lo, hi, color=blue, alpha=0.15, linewidth=0)
            a.plot(days, med, color=blue, lw=2, label="Adaptive IR2 (5% observation samples), median and range of 20 draws")
            a.plot(days, npath, color=orange, lw=2, label="Rolling window without observation samples")
            a.axhline(true_q, color=muted, lw=1, ls="--", label="Later-period percentile (P90 pre, P70 post)")
            a.set_title(f"{'GPU-enabled' if g == 'GPU-Enabled' else 'Non-GPU'}, {ph.lower()}-execution (P{PCT[ph]})", color=ink)
            a.set_ylim(0, max(hi.max(), true_q, npath.max()) * 1.12)
            a.grid(axis="y", color=grid, lw=0.6)
            a.spines[["top", "right"]].set_visible(False)
            if j == 0:
                a.set_ylabel("Timeout (minutes)")
            rows.append({"gpu_bin": g, "phase": ph, "start": start, "later_period_percentile": round(true_q, 1),
                         "naive_end": round(float(npath[-1]), 1), "adaptive_end_median": round(float(med[-1]), 1),
                         "adaptive_end_min": round(float(lo[-1]), 1), "adaptive_end_max": round(float(hi[-1]), 1)})
    for a in axes[1]:
        a.tick_params(axis="x", rotation=30)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"naive_vs_adaptive_tau.{ext}", dpi=200, bbox_inches="tight")
    # Copy as 0411 (the paper's Figure 13; see README).
    fig.savefig(ar.OUT.parents[1] / "section4/figures" / "0411_timeout_trajectories.png", dpi=200, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(OUT / "naive_vs_adaptive_tau.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
    plot()
