"""Harvest the exact data plotted by the manuscript notebooks.

Runs the original notebook code cells in order, in-process, with the notebook's own bootstrap,
but redirects the notebook output folder to 05_Results/figure_style/harvest/<key>/ so the
official execution_runs folders are never touched. When a target cell calls plt.show(), the data
that the original plot code used is pickled. No calculation is changed; only plt.show is wrapped.

Usage: python harvest_plot_data.py section3_01|section4_03
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "05_Results/figure_style/harvest"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

NOTEBOOKS = {
    "section3_01": ("04_Analysis/section3/Run01_Section3.ipynb", "05_Results/section3/execution_runs/01_usage"),
    "section4_03": ("04_Analysis/section4/Run03_IdleDistribution.ipynb", "05_Results/section4/execution_runs/03_idle"),
}


def grab(g, cell):
    """Return the plotted inputs for a target cell (exactly the variables the plot code used)."""
    if g.get("_KEY") == "section3_01":
        if cell == 19:   # Fig. 2 (0301): CDF arrays after the >0 filter
            return {f"x{i}": g[f"x{i}"] for i in range(7)} | {f"y{i}": g[f"y{i}"] for i in range(7)}
        if cell == 16:   # Fig. 3 (0303)
            return {"df_selected_group": g["df_selected_group"]}
        if cell == 14:   # Fig. 4 (0306)
            return {"df_clean": g["df_clean"][["created_time(hour)", "device_type"]].copy()}
        if cell == 28:   # Fig. 5 (0309)
            return {"df_viz": g["df_viz"][["gpu_allocated(group)", "used_time_clip", "status_info(group)"]].copy()}
        if cell == 30:   # Fig. 6 (0313)
            cols = ["terminated_time(hour)", "status_info(group)"]
            return {"df_1_gpu": g["df_1_gpu"][cols].copy(), "df_1_cpu": g["df_1_cpu"][cols].copy()}
    if g.get("_KEY") == "section4_03":
        if cell == 21:   # Fig. 9 (0403)
            return {"plot_df": g["plot_df"][["gpu_bin", "idle_ratio_pct"]].copy()}
    return None


TARGETS = {"section3_01": {14, 16, 19, 28, 30}, "section4_03": {21}}


def main(key):
    nb_rel, official = NOTEBOOKS[key]
    dest = OUT / key
    dest.mkdir(parents=True, exist_ok=True)
    nb = json.loads((ROOT / nb_rel).read_text())
    os.chdir(ROOT / Path(nb_rel).parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = {"__name__": "__main__", "_KEY": key, "get_ipython": lambda: None, "display": print}
    harvested = {}
    done = set()

    def show(*a, **k):
        cell = g.get("_MF_CELL")
        data = grab(g, cell) if cell in TARGETS[key] else None
        if data is not None and cell not in done:
            with open(dest / f"cell{cell:02d}.pkl", "wb") as fh:
                pickle.dump(data, fh)
            harvested[cell] = sorted(data)
            done.add(cell)
        plt.close("all")

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        if f"'{official}'" in src:   # bootstrap: redirect output folder, keep everything else
            src = src.replace(f"'{official}'", f"'{dest.relative_to(ROOT)}/run'")
            assert f"{dest.relative_to(ROOT)}/run" in src
        exec(compile(src, f"{nb_rel}:cell{i}", "exec"), g)
        plt.show = show
        g["plt"] = plt
        if done >= TARGETS[key]:
            break
    assert done >= TARGETS[key], (key, sorted(TARGETS[key] - done))
    (dest / "harvest.json").write_text(json.dumps({"notebook": nb_rel, "cells": {str(k): v for k, v in harvested.items()},
                                                    "official_outputs_untouched": official}, indent=2))
    print(json.dumps({"status": "PASS", "key": key, "cells": sorted(harvested)}))


if __name__ == "__main__":
    main(sys.argv[1])
