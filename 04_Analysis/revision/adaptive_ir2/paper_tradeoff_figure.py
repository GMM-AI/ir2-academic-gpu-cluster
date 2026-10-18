"""Paper figure: reclamation vs later-activity exposure as the IR2 percentile targets vary.

Row (a): Post percentile P50..P95 with Pre fixed at P90.
Row (b): Pre percentile P50..P95 with Post fixed at P70.
Percentile minute values are derived on Train (Section 4.2 definition) and applied
frozen to both periods with the official online rule. Reclamation = observed idle
resource-time after the online decision / allocated resource-time; exposure =
containers with observed activity after the decision / all containers.

Writes 05_Results/revision/adaptive_ir2/figures/paper_tradeoff.{pdf,png,csv}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_experiments as ex  # noqa: E402

ar = ex.ar
OUT = ar.OUT / "figures"
PCTS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1-2 (light surface)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def train_minutes(values):
    out = {}
    for g in ar.GROUPS:
        for ph in ar.PHASES:
            v = values.loc[values.population.eq("Train") & values.gpu_bin.eq(g) & values.phase.eq(ph), "value"]
            out[(g, ph)] = {p: float(np.floor(np.percentile(v, p) + 0.5)) for p in PCTS}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cont = pd.read_parquet(ex.ALL_C)
    s = pd.read_parquet(ex.ALL_S)
    values = ar.calibration_values(ar.load_usage())
    mins = train_minutes(values)
    none = np.zeros(len(cont), bool)
    rows = []
    for row_name, varied in (("post", "Post"), ("pre", "Pre")):
        for p in PCTS:
            pre = {g: (mins[(g, "Pre")][p] if varied == "Pre" else mins[(g, "Pre")][90]) for g in ar.GROUPS}
            post = {g: (mins[(g, "Post")][p] if varied == "Post" else mins[(g, "Post")][70]) for g in ar.GROUPS}
            res = ar.evaluate(cont, s, ar.constant_schedule(pre, post), none)
            for r in ex.metrics(cont, res, np.ones(len(cont), bool)):
                if r["gpu_bin"] == "All":
                    continue
                rows.append({"varied": varied, "percentile": p, "period": r["period"], "gpu_bin": r["gpu_bin"],
                             "pre_min": pre[r["gpu_bin"]], "post_min": post[r["gpu_bin"]],
                             "reclaim_pct": r["reclaim_pct"], "later_active_pct": 100 * r["later_active_rate"],
                             "later_active": r["later_active"], "containers": r["containers"]})
    df = pd.DataFrame(rows)
    # Sanity: (Pre P90, Post P70) equals the official fixed IR2 point.
    base = df.loc[df.varied.eq("Post") & df.percentile.eq(70)]
    fixed = pd.DataFrame(ex.metrics(cont, ar.evaluate(cont, s, ar.constant_schedule(ar.PRE, ar.POST), none),
                                    np.ones(len(cont), bool)))
    for _, r in base.iterrows():
        f = fixed.loc[fixed.period.eq(r.period) & fixed.gpu_bin.eq(r.gpu_bin)].iloc[0]
        assert np.isclose(f.reclaim_pct, r.reclaim_pct) and f.later_active == r.later_active
    df.to_csv(OUT / "paper_tradeoff.csv", index=False)
    plot(df)
    print(df.loc[df.percentile.isin([50, 70, 90, 95])].round(2).to_string())


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    groups = (("Non-GPU", "CPU-time"), ("GPU-Enabled", "GPU-time"))
    for i, (varied, chosen, fixed_txt) in enumerate((("Post", 70, "Pre fixed at P90"),
                                                     ("Pre", 90, "Post fixed at P70"))):
        for j, (g, res_label) in enumerate(groups):
            a = axes[i, j]
            for period, col, ls, mk in (("Train", BLUE, "-", "o"), ("Test", ORANGE, "--", "s")):
                d = df.loc[df.varied.eq(varied) & df.period.eq(period) & df.gpu_bin.eq(g)].sort_values("percentile")
                a.plot(d.later_active_pct, d.reclaim_pct, ls=ls, color=col, lw=1.6, zorder=2,
                       label="Derivation period" if period == "Train" else "Later period")  # manuscript terms
                a.scatter(d.later_active_pct, d.reclaim_pct, s=18, marker=mk, color=col,
                          edgecolor="white", linewidth=0.8, zorder=3)
                c = d.loc[d.percentile.eq(chosen)].iloc[0]
                a.scatter([c.later_active_pct], [c.reclaim_pct], s=70, marker=mk, facecolor="none",
                          edgecolor=INK, linewidth=1.2, zorder=4)
                if period == "Train":
                    offsets = {50: (-8, 7, "right"), chosen: (8, -4, "left"), 95: (8, -9, "left")}
                    if varied == "Pre":
                        offsets[95] = (6, -10, "left")  # keep label inside the axes
                        offsets[chosen] = (10, 2, "left")
                    elif g == "GPU-Enabled":
                        offsets[95] = (-12, 66, "left")  # empty upper-left area, with a leader line
                    for p in (50, chosen, 95):
                        q = d.loc[d.percentile.eq(p)].iloc[0]
                        minutes = q.post_min if varied == "Post" else q.pre_min
                        dx, dy, ha = offsets[p]
                        a.annotate(f"P{p} ({minutes:.0f} min)", (q.later_active_pct, q.reclaim_pct),
                                   xytext=(dx, dy), textcoords="offset points", ha=ha,
                                   arrowprops=({"arrowstyle": "-", "color": MUTED, "lw": 0.6}
                                               if abs(dy) > 20 else None),
                                   fontsize=7, color=INK if p == chosen else MUTED,
                                   fontweight="bold" if p == chosen else "normal")
            name = "Non-GPU" if g == "Non-GPU" else "GPU-enabled"
            a.set_title(f"({'a' if i == 0 else 'b'}{j + 1}) {name}: {varied} percentile varied ({fixed_txt})",
                        fontsize=7.5, color=INK, loc="left")
            a.set_xlabel("Containers active after the decision (%)")
            a.margins(x=0.08, y=0.10)
            a.set_ylabel(f"Observed idle {res_label} reclaimed (%)")
            a.grid(color=GRID, lw=0.6)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            if i == 0 and j == 0:
                a.legend(fontsize=7, frameon=False, loc="lower right")
    fig.text(0.01, 0.005, "Percentile minute values are derived on the derivation period and applied unchanged to both periods; "
             "circled points are the IR2 setting (Pre P90, Post P70).", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "paper_tradeoff.pdf")
    fig.savefig(OUT / "paper_tradeoff.png", dpi=200)
    # Copy as 0410 (the paper's Figure 12; see README).
    paper = ar.OUT.parents[1] / "section4/figures"
    fig.savefig(paper / "0410_percentile_tradeoff.png", dpi=200)


if __name__ == "__main__":
    main()
