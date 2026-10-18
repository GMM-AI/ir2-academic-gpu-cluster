"""Redraw the Section 3-4 paper figures with one common style.

Style follows the Section 3 figures (Arial, matplotlib tab10 blue/orange, full frame, no grid),
drawn at printed size (single column 3.45 in, double column 7.1 in) with 8 pt text, PNG at 300 dpi.
Data are not recomputed by new logic: each figure reads the data its original code plotted
(harvested notebook variables, captured plot inputs, or saved result tables), and every figure is
checked against the official captured outputs before it is written.

Writes 05_Results/figure_style/figures/<name>.png and validation.json.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

OUT = ROOT / "05_Results/figure_style"
FIG = OUT / "figures"
HARV = OUT / "harvest"
TAB4 = ROOT / "05_Results/section4/tables"
RUN1 = ROOT / "05_Results/section3/execution_runs/01_usage"
RUN2 = ROOT / "05_Results/section4/execution_runs/02_utilization"
RUN3 = ROOT / "05_Results/section4/execution_runs/03_idle"

SINGLE, DOUBLE = 3.45, 7.1
BLUE, ORANGE = "#1f77b4", "#ff7f0e"                    # Non-GPU / GPU-enabled, as in Section 3
TAB = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # matplotlib tab10 order
INK, GREY = "#222222", "#555555"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": True, "legend.framealpha": 0.9,
    "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.grid": False, "axes.unicode_minus": False,
    "lines.linewidth": 1.5, "lines.markersize": 4, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out", "savefig.dpi": 300, "figure.dpi": 100,
    "axes.spines.top": True, "axes.spines.right": True, "mathtext.default": "regular",
})
CHECKS: dict[str, dict] = {}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def harvest(key, cell):
    with open(HARV / key / f"cell{cell:02d}.pkl", "rb") as fh:
        return pickle.load(fh)


def log_axis(ax, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(mtick.LogLocator(base=10, numticks=20))   # every decade
    a.set_minor_locator(mtick.NullLocator())                      # no minor ticks
    a.set_major_formatter(mtick.LogFormatterMathtext(base=10))


def violin(ax, **kw):
    sns.violinplot(ax=ax, inner="box", saturation=1, linewidth=0.8, **kw)


# ---------------------------------------------------------------- Section 3
def fig_0301():
    d = harvest("section3_01", 19)
    labels = ["0 GPU", "0.x GPU", "1 GPU", "2 GPU", "3 GPU", "4 GPU", "1.x–3.x GPU"]
    t3 = pd.read_csv(ROOT / "05_Results/section3/tables/Table03_gpu_allocations.csv", index_col=0)
    t2 = pd.read_csv(ROOT / "05_Results/section3/tables/Table02_container_counts.csv")
    n = {lab: len(d[f"x{i}"]) for i, lab in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    for i, lab in enumerate(labels):
        x, y = d[f"x{i}"], d[f"y{i}"]
        assert len(x) == len(y)
        ax.plot(x, y, color=TAB[i], lw=1.3, label=lab)
    ax.set_xscale("log")
    log_axis(ax, "x")
    ax.set_xlabel("Run Time (minutes)")
    ax.set_ylabel("Cumulative Fraction of Containers")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.5, labelspacing=0.3)
    save(fig, "0301_gpu_count_usage_time_cdf")
    CHECKS["0301"] = {"containers_per_group": n, "table02_non_gpu": int(t2.loc[t2.iloc[:, 0].astype(str).str.contains("Non"), "count"].iloc[0]),
                      "table03": t3.to_dict()}


def device_columns(df):
    """Map the two device_type columns to (label, colour) by their totals, not by column order.

    The column order depends on the device_type labels, so each column is identified by its
    total, matched to Table 2 (Non-GPU 16,455 / GPU-enabled 17,118).
    """
    t2 = pd.read_csv(ROOT / "05_Results/section3/tables/Table02_container_counts.csv").set_index("gpu_class")["count"]
    totals = {col: int(df[col].sum()) for col in df.columns}
    by_total = {v: k for k, v in totals.items()}
    assert len(by_total) == 2 and set(by_total) == {int(t2["Non-GPU"]), int(t2["GPU-enabled"])}, totals
    return [(by_total[int(t2["Non-GPU"])], BLUE, "Non-GPU Containers"),
            (by_total[int(t2["GPU-enabled"])], ORANGE, "GPU-Enabled Containers")], totals


def fig_0303():
    d = harvest("section3_01", 16)["df_selected_group"]
    cap = pd.read_csv(RUN1 / "capture/cell16_df_selected_group.csv", header=[0, 1], index_col=0)
    assert np.array_equal(d.to_numpy(), cap.to_numpy()) and list(d.index) == list(cap.index)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    x = np.arange(len(d.index))
    mapping, totals = device_columns(d)
    for col, color, lab in mapping:
        ax.plot(x, d[col].to_numpy(), color=color, label=lab)
    ticks = list(range(0, len(x), 3))
    ax.set_xticks(ticks, [d.index[i] for i in ticks], rotation=45, ha="right")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Number of Containers")
    ax.set_ylim(0, 1200)
    ax.legend(loc="upper center", ncol=2, fontsize=6.5)
    save(fig, "0303_containers_vs_conference_schedule_year_month")
    CHECKS["0303"] = {"matches_captured_df_selected_group": True, "columns": [str(c) for c in d.columns], "months": len(x),
                      "legend_by_total": {lab: totals[col] for col, _, lab in mapping}}


def fig_0306():
    d = harvest("section3_01", 14)["df_clean"]
    counts = pd.DataFrame(d.value_counts(), columns=["count"]).unstack("device_type")
    art = pd.read_csv(RUN1 / "figures/cell14_plot01_artists.csv")
    for j, col in enumerate(counts.columns):
        series = art.loc[art.series.eq(art.series.unique()[j])].sort_values("point")
        assert np.array_equal(series.y.to_numpy(), counts[col].to_numpy())
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    mapping, totals = device_columns(counts)
    for col, color, lab in mapping:
        ax.plot(counts.index, counts[col].to_numpy(), color=color, label=lab)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Number of Containers")
    ax.set_ylim(0, 2000)
    ax.legend(loc="upper left", fontsize=6.5)
    save(fig, "0306_container_creation_by_time")
    CHECKS["0306"] = {"matches_captured_artists": True, "columns": [str(c) for c in counts.columns],
                      "legend_by_total": {lab: totals[col] for col, _, lab in mapping}}


def fig_0309():
    d = harvest("section3_01", 28)["df_viz"]
    order = ["0 GPU", "0.x GPU", "1 GPU", "2 GPU", "3 GPU", "4 GPU", "1.x–3.x GPU"]
    t5 = pd.read_csv(ROOT / "05_Results/section3/tables/Table05_termination_counts.csv")
    tab = d.groupby(["gpu_allocated(group)", "status_info(group)"]).size().unstack()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    violin(ax, data=d, x="gpu_allocated(group)", y="used_time_clip", hue="status_info(group)", order=order,
           hue_order=["Auto-stop", "Manual-stop"], palette={"Auto-stop": BLUE, "Manual-stop": ORANGE},
           log_scale=(False, True), split=False, cut=2, width=0.8)
    ax.set_xticks(range(len(order)), [o.replace(" GPU", "") for o in order])
    ax.set_xlabel("GPU Count Per Container")
    ax.set_ylabel("Run Time (minute)")
    log_axis(ax)
    ax.set_ylim(1e-2, 1e7)
    ax.legend(loc="upper right", ncol=2, fontsize=6.5, title=None)
    save(fig, "0309_execution_time_by_gpu_count_and_termination_reason")
    CHECKS["0309"] = {"rows_by_group_status": tab.fillna(0).astype(int).to_dict(), "table05_total_row": t5.tail(1).to_dict("records")}


def fig_0313():
    h = harvest("section3_01", 30)
    art = pd.read_csv(RUN1 / "figures/cell30_plot01_artists.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    colors = {"Auto-stop": BLUE, "Manual-stop": ORANGE}
    for idx, (ax1, sub, title) in enumerate(zip(axes, (h["df_1_gpu"], h["df_1_cpu"]),
                                                ("GPU-Enabled Containers", "Non-GPU Containers"))):
        hc = sub.groupby(["terminated_time(hour)", "status_info(group)"]).size().unstack(fill_value=0).reindex(range(24), fill_value=0)
        cols = [c for c in ("Auto-stop", "Manual-stop") if c in hc.columns]
        for c in ("Auto-stop", "Manual-stop"):
            a_art = art.loc[art.axis.eq(idx) & art.kind.eq("bar") & art.series.eq(c)].sort_values("point")
            assert np.array_equal(a_art.y.to_numpy(), hc[c].to_numpy().astype(float)), (idx, c)
        bottom = np.zeros(24)
        for c in cols:
            ax1.bar(range(24), hc[c].to_numpy(), bottom=bottom, width=0.8, color=colors[c], label=c)
            bottom += hc[c].to_numpy()
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xlim(-0.6, 23.6)
        ax1.set_ylim(0, 1600)
        ax1.set_xlabel("Time of Day")
        ax1.set_title(title)
        if idx == 0:
            ax1.set_ylabel("Number of Containers")
            ax1.legend(loc="upper left", fontsize=6.5)
        ax2 = ax1.twinx()
        total = hc[cols].sum(axis=1)
        ratio = np.where(total > 0, hc["Auto-stop"] / total, np.nan)
        ax2.plot(range(24), ratio, color="black", ls="--", lw=1.2, marker="o", ms=3.5,
                 mfc=BLUE, mec="black", mew=0.7, label="Auto-Stop Ratio")
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Auto-Stop Ratio")
        if idx == 0:
            ax2.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "0313_Hourly distribution of termination reasons for containers by GPU Usage")
    CHECKS["0313"] = {"stacked_bars_match_captured_artists": True}


# ---------------------------------------------------------------- Section 4
def fig_0401():
    pts = pd.read_csv(TAB4 / "figure7_cdf_points.csv")
    pts = pts.loc[pts.version.eq("normalized")]
    val = json.loads((TAB4 / "figure7_validation.json").read_text())
    assert val["status"] == "PASS"
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.45), sharey=True)
    styles = {"Non-GPU": (BLUE, "-", 1.5, 3.5, 1.0), "GPU-Enabled AVG.": (ORANGE, "-", 1.5, 3.5, 1.0),
              "1 GPU": (TAB[2], "--", 0.8, 2, 0.8), "2 GPU": (TAB[3], "--", 0.8, 2, 0.8),
              "3 GPU": (TAB[4], "--", 0.8, 2, 0.8), "4 GPU": (TAB[5], "--", 0.8, 2, 0.8)}
    specs = {"CPU": [(10, "Non-GPU, GPU-enabled: 10%", (0.30, 0.93), 1.04)],
             "Memory": [(15, "Non-GPU: 15%", (0.34, 0.955), 1.12), (20, "GPU-enabled: 20%", (0.34, 0.885), 1.05)],
             "GPU": [(0, "GPU-enabled: 0%", (0.22, 0.93), 1.04)]}
    for ax, res in zip(axes, ("CPU", "Memory", "GPU")):
        for grp, (c, ls, lw, ms, al) in styles.items():
            d = pts.loc[pts.resource.eq(res) & pts.group.eq(grp)].sort_values("utilization_pct")
            if d.empty:
                continue
            ax.plot(d.utilization_pct, d.container_fraction, color=c, ls=ls, lw=lw, marker="o", ms=ms, alpha=al, label=grp)
        for cutoff, label, pos, anchor in specs[res]:
            ax.axvline(cutoff, color=GREY, ls=":", lw=1.0, zorder=1)
            ax.annotate(label, xy=(cutoff, anchor), xytext=pos, textcoords="axes fraction", ha="left", va="center",
                        fontsize=6.5, color=INK, arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.6},
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 1})
        ax.set_title(res)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.2)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Utilization (%)" if res != "GPU" else "Utilization (%)\nrelative to allocated GPUs")
    axes[0].set_ylabel("Cumulative Fraction of Containers")
    axes[0].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    axes[0].legend(loc="lower right", fontsize=6.5, labelspacing=0.3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "0401_resource_utilization_container_count_cdf")
    CHECKS["0401"] = {"source": "figure7_cdf_points.csv (normalized)", "points": int(len(pts)), "figure7_validation": "PASS"}


def fig_0409():
    d = pd.read_csv(RUN2 / "capture/cell19_figure8_plot_inputs.csv")
    v = json.loads((TAB4 / "figure8_validation.json").read_text())
    rt = d["used_time(minute)"]
    assert len(d) == 2048 and round(float(rt.median()), 1) == 20.8
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2))
    violin(axes[0], y=rt, color=BLUE, cut=0, bw_adjust=0.6, density_norm="width", log_scale=(False, True))
    axes[0].set_ylabel("Run Time (minute)")
    axes[0].set_title("Run Time Distribution (GPU-Enabled, GPU util = 0)")
    log_axis(axes[0])
    violin(axes[1], y=d["CPU"], color=BLUE, cut=0, bw_adjust=0.6, density_norm="width")
    axes[1].set_ylabel("CPU Usage (%)")
    axes[1].set_title("CPU Usage Distribution (GPU-Enabled, GPU util = 0)")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_xticks([])
    fig.tight_layout(w_pad=2)
    save(fig, "0409_Resource_Usage_of_Containers_with_Unused_GPUs")
    CHECKS["0409"] = {"n": len(d), "median_runtime_min": round(float(rt.median()), 2), "over_120_pct": round(float((rt > 120).mean() * 100), 1),
                      "cpu_below_3_pct": round(float((d.CPU < 3).mean() * 100), 1), "validation_status": v.get("status")}


def fig_0403():
    d = harvest("section4_03", 21)["plot_df"]
    cap = pd.read_csv(RUN3 / "capture/cell21_high_count_tbl.csv", index_col=0)
    for b in ("Non-GPU", "GPU-Enabled"):
        assert int((d.gpu_bin == b).sum()) == int(cap.loc[b, "total"])
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    violin(ax, data=d, x="gpu_bin", y="idle_ratio_pct", order=["Non-GPU", "GPU-Enabled"], color=BLUE,
           cut=2, bw_adjust=0.6, density_norm="width")
    ax.set_xlabel("Device Type")
    ax.set_ylabel("Idle Ratio (%)")
    ax.set_ylim(-10, 110)
    save(fig, "0403_idle_ratio_of_normal_containers")
    CHECKS["0403"] = {"normal_containers": d.gpu_bin.value_counts().to_dict(), "matches_captured_totals": True,
                      "median_idle_ratio_pct": d.groupby("gpu_bin").idle_ratio_pct.median().round(2).to_dict()}


def _prepost():
    cases = pd.read_parquet(TAB4 / "Figure10_11_prepost_cases.parquet")
    assert len(cases) == 6724
    pre = cases.loc[cases.is_pre_case, ["gpu_bin", "pre_idle_min"]].rename(columns={"pre_idle_min": "value"})
    post = cases.loc[cases.is_post_case, ["gpu_bin", "post_eol_idle_min"]].rename(columns={"post_eol_idle_min": "value"})
    return pre, post


EXPECTED = {("Pre", "Non-GPU"): 85, ("Pre", "GPU-Enabled"): 119, ("Post", "Non-GPU"): 146, ("Post", "GPU-Enabled"): 170}


def fig_0405():
    pre, post = _prepost()
    q = pd.read_csv(TAB4 / "Figure11_threshold_recalculation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    for ax, frame, phase, pct, title in ((axes[0], pre, "Pre", 90, "Pre-Execution Idle Time"),
                                         (axes[1], post, "Post", 70, "Post-Execution Idle Time")):
        for g in ("Non-GPU", "GPU-Enabled"):
            vals = frame.loc[frame.gpu_bin.eq(g), "value"].to_numpy()
            saved = q.loc[q.phase.eq(phase) & q.group.eq(g)].iloc[0]
            assert len(vals) == saved.cases and abs(np.percentile(vals, pct) - saved.raw_minutes) < 1e-9
        violin(ax, data=frame, x="gpu_bin", y="value", order=["Non-GPU", "GPU-Enabled"], color=BLUE,
               log_scale=(False, True), cut=2, bw_adjust=0.6, density_norm="width")
        for x, g in enumerate(("Non-GPU", "GPU-Enabled")):
            v = EXPECTED[(phase, g)]
            ax.plot([x - .22, x + .22], [v, v], color=INK, ls="--", lw=1.0, zorder=6)
            ax.plot(x, v, marker="D", ms=3.5, color=INK, zorder=7)
            ax.annotate(f"P{pct}: {v} min", xy=(x, v), xytext=(7, 11), textcoords="offset points", fontsize=6.5,
                        ha="left", va="bottom", arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6},
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1}, zorder=8)
        ax.set_xticks([0, 1], ["Non-GPU", "GPU-enabled"])
        ax.set_title(title)
        ax.set_xlabel("Device Type")
        ax.set_ylabel("Time (minutes)")
        log_axis(ax)
        ax.set_ylim(1e-1, 1e6)
    fig.tight_layout(w_pad=2)
    save(fig, "0405_normal_container_idle_time_distribution")
    CHECKS["0405"] = {"percentile_cases_match_Figure11_threshold_recalculation": True, "markers": {f"{k[0]}/{k[1]}": v for k, v in EXPECTED.items()}}


def fig_0408():
    pre, post = _prepost()
    pcts = list(range(60, 100, 5)) + [99]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    for ax, frame, phase, pct, title in ((axes[0], pre, "Pre", 90, "Pre-Execution Idle Time"),
                                         (axes[1], post, "Post", 70, "Post-Execution Idle Time")):
        tbl = pd.DataFrame({"Percentile (%)": pcts})
        for g in ("Non-GPU", "GPU-Enabled"):
            tbl[g] = [float(np.percentile(frame.loc[frame.gpu_bin.eq(g), "value"], p)) for p in pcts]
        tbl = tbl.round(2)
        saved = pd.read_csv(TAB4 / f"Figure10_11_{phase.lower()}_pct_tbl.csv").drop(columns=["Unnamed: 0"], errors="ignore")
        assert np.allclose(tbl.to_numpy(), saved.to_numpy())
        ax.axvline(pct, color=GREY, ls=":", lw=1.0, zorder=1)
        offsets = {("Pre", "Non-GPU"): (0.60, 0.13), ("Pre", "GPU-Enabled"): (0.50, 0.62),
                   ("Post", "Non-GPU"): (0.34, 0.12), ("Post", "GPU-Enabled"): (0.34, 0.50)}
        for g, c, lab in (("Non-GPU", BLUE, "Non-GPU"), ("GPU-Enabled", ORANGE, "GPU-enabled")):
            y = tbl[g].clip(lower=1.0)
            ax.plot(tbl["Percentile (%)"], y, color=c, marker="o", ms=3.5, label=lab)
            point = float(tbl.loc[tbl["Percentile (%)"].eq(pct), g].iloc[0])
            ax.plot(pct, point, marker="o", mec=INK, mew=1.0, mfc="white", ms=6, zorder=5)
            ax.annotate(f"{lab}\nP{pct}: {EXPECTED[(phase, g)]} min", xy=(pct, point), xytext=offsets[(phase, g)],
                        textcoords="axes fraction", fontsize=6.5, ha="left", va="center",
                        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6},
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1}, zorder=7)
        ax.set_yscale("log")
        log_axis(ax)
        ax.set_xticks(pcts)
        ax.set_xlim(59, 99.5)
        ax.set_title(title)
        ax.set_xlabel("Percentile (%)")
        ax.set_ylabel("Time (minutes)")
        ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(w_pad=2)
    save(fig, "0408_normal_container_idle_time_cdf")
    CHECKS["0408"] = {"percentile_tables_match_saved": True}


def fig_0410():
    df = pd.read_csv(ROOT / "05_Results/revision/adaptive_ir2/figures/paper_tradeoff.csv")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.3))
    groups = (("Non-GPU", "CPU-time"), ("GPU-Enabled", "GPU-time"))
    for i, (varied, chosen, fixed_txt) in enumerate((("Post", 70, "Pre fixed at P90"), ("Pre", 90, "Post fixed at P70"))):
        for j, (g, res_label) in enumerate(groups):
            a = axes[i, j]
            for period, col, ls, mk in (("Train", BLUE, "-", "o"), ("Test", ORANGE, "--", "s")):
                d = df.loc[df.varied.eq(varied) & df.period.eq(period) & df.gpu_bin.eq(g)].sort_values("percentile")
                a.plot(d.later_active_pct, d.reclaim_pct, ls=ls, color=col, marker=mk, ms=3.5, zorder=2,
                       label="Derivation period" if period == "Train" else "Later period")
                c = d.loc[d.percentile.eq(chosen)].iloc[0]
                a.scatter([c.later_active_pct], [c.reclaim_pct], s=55, marker=mk, facecolor="none",
                          edgecolor=INK, linewidth=1.0, zorder=4)
                if period == "Train":
                    offsets = {50: (-8, 7, "right"), chosen: (8, -4, "left"), 95: (8, -9, "left")}
                    if varied == "Pre":
                        offsets[95] = (6, -10, "left")
                        offsets[chosen] = (4, 10 if g == "Non-GPU" else 18, "left")
                    elif g == "GPU-Enabled":
                        offsets[95] = (-12, 66, "left")
                    for p in (50, chosen, 95):
                        q = d.loc[d.percentile.eq(p)].iloc[0]
                        minutes = q.post_min if varied == "Post" else q.pre_min
                        dx, dy, ha = offsets[p]
                        a.annotate(f"P{p} ({minutes:.0f} min)", (q.later_active_pct, q.reclaim_pct), xytext=(dx, dy),
                                   textcoords="offset points", ha=ha, fontsize=6.5,
                                   arrowprops=({"arrowstyle": "-", "color": GREY, "lw": 0.6} if abs(dy) > 20 else None),
                                   color=INK if p == chosen else GREY, fontweight="bold" if p == chosen else "normal")
            name = "Non-GPU" if g == "Non-GPU" else "GPU-enabled"
            a.set_title(f"({'a' if i == 0 else 'b'}{j + 1}) {name}: {varied} percentile varied ({fixed_txt})", loc="left")
            a.set_xlabel("Containers active after the decision (%)")
            a.set_ylabel(f"Observed idle {res_label} reclaimed (%)")
            a.margins(x=0.08, y=0.15)
            if i == 0 and j == 0:
                a.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, "0410_percentile_tradeoff")
    base = df.loc[df.varied.eq("Post") & df.percentile.eq(70) & df.period.eq("Test")]
    CHECKS["0410"] = {"source": "paper_tradeoff.csv", "fixed_point_later_period": base[["gpu_bin", "reclaim_pct", "later_active"]].round(2).to_dict("records")}


def fig_0411():
    sys.path.insert(0, str(ROOT / "04_Analysis/revision/adaptive_ir2"))
    sys.path.insert(0, str(ROOT))
    import adaptive_experiments as ex
    import naive_window as nw
    ar = ex.ar
    naive = pd.read_csv(nw.OUT / "naive_tau_updates.csv", parse_dates=["time"])
    ad = pd.read_csv(ar.OUT / "experiments" / "tau_updates.csv", parse_dates=["time"])
    ad = ad.loc[ad.config.eq("W_main")]
    values = ar.calibration_values(ar.load_usage())
    days = pd.date_range(ar.TEST_START, "2026-09-01", freq="D")
    saved = pd.read_csv(nw.OUT / "naive_vs_adaptive_tau.csv")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.9), sharex=True)
    rows = []
    for i, g in enumerate(ar.GROUPS):
        for j, ph in enumerate(ar.PHASES):
            a = axes[i, j]
            start = float((ar.PRE if ph == "Pre" else ar.POST)[g])
            paths = []
            for _, d in ad.loc[ad.gpu_bin.eq(g) & ad.phase.eq(ph)].groupby("seed"):
                d = d.sort_values("time")
                idx = np.searchsorted(d.time.to_numpy(), days.to_numpy(), side="right") - 1
                paths.append(np.where(idx >= 0, d.tau.to_numpy()[np.maximum(idx, 0)], start))
            paths = np.array(paths)
            med, lo, hi = np.median(paths, 0), paths.min(0), paths.max(0)
            n = naive.loc[naive.gpu_bin.eq(g) & naive.phase.eq(ph)].sort_values("time")
            idx = np.searchsorted(n.time.to_numpy(), days.to_numpy(), side="right") - 1
            npath = n.tau.to_numpy()[np.maximum(idx, 0)]
            v = values.loc[values.gpu_bin.eq(g) & values.phase.eq(ph) & values.population.eq("Test"), "value"]
            true_q = float(np.percentile(v, nw.PCT[ph]))
            a.fill_between(days, lo, hi, color=BLUE, alpha=0.18, linewidth=0)
            a.plot(days, med, color=BLUE, label="Adaptive IR2 (5% observation samples), median and range over 20 random selections")
            a.plot(days, npath, color=ORANGE, label="Rolling window without observation samples")
            a.axhline(true_q, color=GREY, lw=1.0, ls="--", label="Later-period percentile (P90 pre, P70 post)")
            a.set_title(f"{'GPU-enabled' if g == 'GPU-Enabled' else 'Non-GPU'}, {ph.lower()}-execution (P{nw.PCT[ph]})")
            a.set_ylim(0, max(hi.max(), true_q, npath.max()) * 1.12)
            if j == 0:
                a.set_ylabel("Timeout (minutes)")
            rows.append({"gpu_bin": g, "phase": ph, "later_period_percentile": round(true_q, 1),
                         "naive_end": round(float(npath[-1]), 1), "adaptive_end_median": round(float(med[-1]), 1)})
    for a in axes[1]:
        a.tick_params(axis="x", rotation=30)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.01), fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1), h_pad=1.0, w_pad=1.5)
    save(fig, "0411_timeout_trajectories")
    m = pd.DataFrame(rows).merge(saved, on=["gpu_bin", "phase"], suffixes=("", "_saved"))
    for c in ("later_period_percentile", "naive_end", "adaptive_end_median"):
        assert np.allclose(m[c], m[c + "_saved"]), c
    CHECKS["0411"] = {"matches_naive_vs_adaptive_tau.csv": True, "rows": rows}


def main():
    for f in (fig_0301, fig_0303, fig_0306, fig_0309, fig_0313, fig_0401, fig_0409, fig_0403,
              fig_0405, fig_0408, fig_0410, fig_0411):
        f()
        print("ok", f.__name__)
    (OUT / "validation.json").write_text(json.dumps({"status": "PASS", "style": "Arial 8 pt, tab10 blue/orange, full frame, no grid, 300 dpi",
                                                     "widths_in": {"single": SINGLE, "double": DOUBLE}, "checks": CHECKS},
                                                    ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
