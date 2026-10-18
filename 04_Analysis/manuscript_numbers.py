"""Compute manuscript numbers that no pipeline step writes to a result file.

These values were aggregated by hand and inserted into the manuscript. This script recomputes each one from the existing results, the
processed data, or (for the empty-GPU-sample count) the raw time series, and
compares it with the value printed in the manuscript.

Read-only for all inputs. Writes only 05_Results/manuscript_numbers/.

Usage:
    python 04_Analysis/manuscript_numbers.py [--skip-raw]

The raw scan (about 20 s) reads 01_RawData/time_series; --skip-raw omits it.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_paths import PROCESSED_DATA, RAW_TIME_SERIES, RESULTS_REVISION, RESULTS_SECTION3, install_data_io_guard  # noqa: E402

install_data_io_guard()

EVENTS = RESULTS_REVISION / "_cache/online_ir2/container_events.csv"
ADAPTIVE = RESULTS_REVISION / "adaptive_ir2/experiments"
FILE_AUDIT = ROOT / "inputs/preprocessing_file_audit.csv"
TRAIN_USAGE = PROCESSED_DATA / "UsageLog20230401_20251130.csv"
OUT = ROOT / "05_Results/manuscript_numbers"
TEST_START = pd.Timestamp("2026-03-01")


def read_usage(path: Path, usecols: list[str]) -> pd.DataFrame:
    for encoding in ("utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, usecols=usecols, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def online_events() -> pd.DataFrame:
    e = pd.read_csv(EVENTS, parse_dates=["online_time", "first_active"], low_memory=False)
    return e.loc[e.population.eq("Test")]


def later_active_share(e: pd.DataFrame) -> dict:
    """Share of reclaimed resource-time from containers with later activity (4.4, Conclusion)."""
    gpu = e.loc[e.gpu_bin.eq("GPU-Enabled")]
    ng = e.loc[e.gpu_bin.eq("Non-GPU")]
    g_time = gpu.gpu_allocated * gpu.after_idle_min
    c_time = ng.cpu_allocated * ng.after_idle_min
    later_g = gpu.potential_interruption.astype(bool)
    later_c = ng.potential_interruption.astype(bool)
    return {"gpu_pct": g_time[later_g].sum() / g_time.sum() * 100,
            "ng_cpu_pct": c_time[later_c].sum() / c_time.sum() * 100}


def decision_counts(e: pd.DataFrame) -> dict:
    """Decision breakdown in the later period (4.2, 4.5)."""
    d = e.loc[e.online_time.notna()]
    pre, post = d.loc[d.online_kind.eq("Pre")], d.loc[d.online_kind.eq("Post")]
    later = d.potential_interruption.astype(bool)
    return {"decisions": len(d), "pre": len(pre), "post": len(post),
            "pre_never_active": int(pre.first_active.isna().sum()),
            "later_active": int(later.sum()),
            "post_later_active": int(post.potential_interruption.astype(bool).sum()),
            "post_later_active_pct": post.potential_interruption.astype(bool).mean() * 100}


def adaptive_all_cpu() -> float:
    """All-CPU reclaimed share of cold-start adaptive IR2, later period, incl. samples (4.4)."""
    r = pd.read_csv(ADAPTIVE / "results_by_seed.csv")
    r = r.loc[r.config.eq("C_main") & r.period.eq("Test") & r.scope.eq("all") & r.policy.eq("Adaptive IR2")]
    sums = r.groupby("seed")[["idle_cpu_min", "baseline_cpu_min"]].sum()
    per_seed = sums.idle_cpu_min / sums.baseline_cpu_min * 100
    assert len(per_seed) == 20, len(per_seed)
    return float(per_seed.mean())


def adaptive_update_counts() -> dict:
    """Median number of timeout updates per seed in the later period, runs of Fig. 13 (W_main)."""
    t = pd.read_csv(ADAPTIVE / "tau_updates.csv", parse_dates=["time"])
    t = t.loc[t.config.eq("W_main") & t.time.ge(TEST_START)]
    counts = t.groupby(["gpu_bin", "phase", "seed"]).size().groupby(["gpu_bin", "phase"]).median()
    return {f"{g} {p}": float(v) for (g, p), v in counts.items()}


def exit_code_share() -> dict:
    """Share of containers covered by the five most frequent exit codes (3.3)."""
    u = read_usage(TRAIN_USAGE, ["status_info"])
    v = u.status_info.value_counts()
    return {"codes": int(len(v)), "top5_pct": v.head(5).sum() / len(u) * 100, "containers": int(len(u))}


def noninteger_share() -> float:
    """Non-integer allocations (0.x and 1.x-3.x GPU) among GPU-enabled containers (3.2)."""
    t = pd.read_csv(RESULTS_SECTION3 / "tables/Table03_gpu_allocations.csv").set_index("gpu_allocated(group)")
    return float((t.loc["0.x_GPU", "container_count(EA)"] + t.loc["1.x+_GPU", "container_count(EA)"])
                 / t.loc["Total", "container_count(EA)"] * 100)


def short_run_range() -> dict:
    """Share of containers finishing within ten minutes, by allocation size (3.2, Fig. 2)."""
    u = read_usage(TRAIN_USAGE, ["gpu_allocated(EA)", "gpu_allocated(group)", "used_time(minute)"])
    ea = u["gpu_allocated(EA)"]
    group = u["gpu_allocated(group)"].where(~((ea > 1) & (ea % 1 != 0)), "1.x-3.x_GPU")
    share = u["used_time(minute)"].le(10).groupby(group).mean() * 100
    return {"by_group": share.round(2).to_dict(), "min": float(share.min()), "max": float(share.max())}


def empty_gpu_exceptions() -> dict:
    """Raw samples with an empty GPU field, and how many of them have non-zero CPU or memory (4.1)."""
    audit = pd.read_csv(FILE_AUDIT, usecols=["filename", "status", "gpu_allocated"], low_memory=False)
    ok = audit.loc[audit.status.eq("ok")].set_index("filename")
    files = {p.stem: p for p in RAW_TIME_SERIES.rglob("*.csv") if p.stem in ok.index}

    def one(item):
        stem, path = item
        d = pd.read_csv(path, usecols=["CPU util (%)", "Mem util (%)", "GPU util (%)"], low_memory=False)
        d = d.apply(pd.to_numeric, errors="coerce")
        empty = d["GPU util (%)"].isna()
        nonzero = empty & (d["CPU util (%)"].fillna(0).ne(0) | d["Mem util (%)"].fillna(0).ne(0))
        return ok.loc[stem, "gpu_allocated"] > 0, int(empty.sum()), int(nonzero.sum()), len(d)

    with ThreadPoolExecutor(8) as pool:
        rows = pd.DataFrame(list(pool.map(one, files.items())), columns=["gpu", "n_empty", "n_nonzero", "n_rows"])
    by = rows.groupby("gpu")[["n_empty", "n_nonzero"]].sum()
    return {"files": len(rows), "raw_rows": int(rows.n_rows.sum()), "empty_gpu_rows": int(rows.n_empty.sum()),
            "empty_with_nonzero_cpu_or_mem": int(rows.n_nonzero.sum()),
            "gpu_containers": int(by.loc[True, "n_nonzero"]), "non_gpu_containers": int(by.loc[False, "n_nonzero"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-raw", action="store_true", help="skip the raw time-series scan (empty GPU samples)")
    args = parser.parse_args()

    e = online_events()
    share = later_active_share(e)
    dec = decision_counts(e)
    codes = exit_code_share()
    short = short_run_range()
    updates = adaptive_update_counts()
    values = {"later_active_share": share, "decisions": dec, "adaptive_all_cpu_pct": adaptive_all_cpu(),
              "adaptive_updates_median": updates, "exit_codes": codes,
              "noninteger_pct": noninteger_share(), "short_runs": short}

    # (manuscript text, manuscript value, computed value, decimals used in the manuscript)
    checks = [
        ("4.4/5: reclaimed GPU-time from later-active containers (%)", 81.5, share["gpu_pct"], 1),
        ("4.4: reclaimed non-GPU CPU-time from later-active containers (%)", 40.9, share["ng_cpu_pct"], 1),
        ("4.5: online decisions", 1308, dec["decisions"], 0),
        ("4.5: decisions before the first active sample", 706, dec["pre"], 0),
        ("4.5: ... of which on never-active containers", 425, dec["pre_never_active"], 0),
        ("4.2/4.5: decisions after the first active sample", 602, dec["post"], 0),
        ("4.2/4.5: ... followed by activity", 347, dec["post_later_active"], 0),
        ("4.2: share of post decisions followed by activity (%)", 57.6, dec["post_later_active_pct"], 1),
        ("4.5: containers with later activity", 628, dec["later_active"], 0),
        ("4.4: adaptive IR2, all CPU-time (%)", 31.77, values["adaptive_all_cpu_pct"], 2),
        ("4.4: timeout updates, fewest median (Fig. 13 runs)", 4, min(updates.values()), 0),
        ("4.4: timeout updates, most median (Fig. 13 runs)", 6, max(updates.values()), 0),
        ("3.3: five main exit codes (%)", 99.5, codes["top5_pct"], 1),
        ("3.2: non-integer share of GPU-enabled containers (%)", 36.6, values["noninteger_pct"], 1),
        ("3.2: lowest share finishing within ten minutes (%)", 18, short["min"], 0),
        ("3.2: highest share finishing within ten minutes (%)", 30, short["max"], 0),
    ]
    if not args.skip_raw and RAW_TIME_SERIES.exists():
        raw = empty_gpu_exceptions()
        values["empty_gpu"] = raw
        checks += [("4.1: samples with an empty GPU field (%)", 0.8,
                    raw["empty_gpu_rows"] / 26_542_007 * 100, 1),
                   ("4.1: empty-GPU samples with non-zero CPU or memory", 46,
                    raw["empty_with_nonzero_cpu_or_mem"], 0)]

    rows = []
    for text, stated, computed, ndigits in checks:
        ok = round(float(computed), ndigits) == round(float(stated), ndigits)
        rows.append({"item": text, "manuscript": stated, "computed": round(float(computed), 4), "status": "PASS" if ok else "FAIL"})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "manuscript_numbers_check.csv", index=False)
    (OUT / "manuscript_numbers.json").write_text(json.dumps(values, indent=2, default=float, ensure_ascii=False))
    failed = table.status.eq("FAIL").sum()
    print(f"\n{len(table) - failed} PASS, {failed} FAIL -> {OUT.relative_to(ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
