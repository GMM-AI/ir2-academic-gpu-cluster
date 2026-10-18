# IR2: Reclaiming Idle Resources in an Academic GPU Cluster

Code and data for the paper

> M. Gil and E. Seo, "From Patterns to Practice: Reclaiming Idle Resources in Academic GPU Clusters," *Future Generation Computer Systems* (under revision).

The repository rebuilds every computed table, figure and number in the paper from the released data:
raw scheduler logs and minute-level container traces → processed data → analyses → figures.
The code is the code that produced the paper's results, adapted only to run inside this repository
(relative paths, comments, and checks against the authors' unreleased intermediate files removed).
No computation, threshold, random seed or selection rule was changed.

## Requirements

- macOS or Linux with `zsh` (the shell scripts use `#!/bin/zsh`; on Linux install the `zsh` package).
- Python 3.11 and the packages pinned in `requirements.txt`.
- About 10 GB of free disk space (raw traces 4.1 GB, processed data 2.8 GB when rebuilt, results 0.6 GB, archives 1.3 GB).
- Tested on a MacBook Pro (Apple M1 Pro, 32 GB RAM): preprocessing about 5 minutes, all analyses about 35 minutes.

## Quick start

```bash
# 1. Python 3.11 and the pinned packages
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Data (see "Data" below): download the archives and unpack them into the repository root
R=https://github.com/GMM-AI/ir2-academic-gpu-cluster/releases/download/data-v1
curl -LO $R/raw_scheduler_logs.zip && curl -LO $R/raw_minute_traces.zip && curl -LO $R/processed.zip
unzip -q raw_scheduler_logs.zip && unzip -q raw_minute_traces.zip   # -> 01_RawData/
unzip -q processed.zip                                              # -> 03_ProcessedData/ (skip if you run 3a)

# 3a. Optional: rebuild the processed data from the raw data (about 5 minutes).
#     Needs 01_RawData/; refuses to run if 03_ProcessedData/ already holds the parquets.
PYTHON=python ./run_preprocessing.sh

# 3b. Reproduce all results (about 35 minutes)
PYTHON=python ./run_all.sh
```

**What "PASS" means.** Every step asserts its own checks (cohort sizes, inputs, reproduced intermediate values) and stops with an error if one fails; `run_all.sh` then prints `FAILED: <step>` and the end of that step's log (`05_Results/run_all_logs/<step>.log`). `ALL STEPS PASSED` means that every step finished with all of its checks satisfied. The last step, `04_Analysis/manuscript_numbers.py`, recomputes the numbers in the paper that were aggregated by hand and prints PASS when the recomputed value equals the published value at the paper's precision (result: `05_Results/manuscript_numbers/manuscript_numbers_check.csv`). Two of its checks (empty GPU samples, Section 4.1) read the raw minute traces and are skipped when `01_RawData/time_series/` is absent.

## Where the paper's results are written

All paths are under `05_Results/`. The figures in their final style are in `figure_style/figures/`; the file names keep the authors' internal plot codes:

| Paper | File | Produced by |
|---|---|---|
| Figure 2 | `0301_gpu_count_usage_time_cdf.png` | `section3/Run01_Section3.ipynb` |
| Figure 3 | `0303_containers_vs_conference_schedule_year_month.png` | `section3/Run01_Section3.ipynb` |
| Figure 4 | `0306_container_creation_by_time.png` | `section3/Run01_Section3.ipynb` |
| Figure 5 | `0309_execution_time_by_gpu_count_and_termination_reason.png` | `section3/Run01_Section3.ipynb` |
| Figure 6 | `0313_Hourly distribution of termination reasons for containers by GPU Usage.png` | `section3/Run01_Section3.ipynb` |
| Figure 7 | `0401_resource_utilization_container_count_cdf.png` | `section4/Run00_Figure7.ipynb` |
| Figure 8 | `0409_Resource_Usage_of_Containers_with_Unused_GPUs.png` | `section4/Run02_Utilization.ipynb` |
| Figure 9 | `0403_idle_ratio_of_normal_containers.png` | `section4/Run03_IdleDistribution.ipynb` |
| Figure 10 | `0405_normal_container_idle_time_distribution.png` | `section4/Run03_IdleDistribution.ipynb` |
| Figure 11 | `0408_normal_container_idle_time_cdf.png` | `section4/Run03_IdleDistribution.ipynb` |
| Figure 12 | `0410_percentile_tradeoff.png` | `revision/adaptive_ir2/paper_tradeoff_figure.py` |
| Figure 13 | `0411_timeout_trajectories.png` | `revision/adaptive_ir2/naive_window.py`, `adaptive_experiments.py` |

`figure_style/harvest_plot_data.py` collects the data each notebook plots and `restyle_paper_figures.py` redraws the figures with one common style; the notebooks' own (unstyled) figures are in `section3/figures/` and `section4/figures/`.

| Paper | File |
|---|---|
| Tables 2, 3, 5 | `section3/tables/Table02_container_counts.csv`, `Table03_gpu_allocations.csv`, `Table05_termination_counts.csv` |
| Tables 7, 8 | `section4/tables/Table07_lowutil_counts.csv`, `Table08_idle_concentration.csv` |
| Table 9, fixed IR2 and single 150 min | `revision/06_policy_summary.csv`, rows `Test`: reclaimed = `idle_cpu_min / baseline_cpu_min` (non-GPU) or `idle_gpu_min / baseline_gpu_min` (GPU-enabled); later activity = `return_containers` |
| Table 9, adaptive IR2 | `revision/adaptive_ir2/experiments/summary.csv`, rows `config = C_main`, `scope = all`, `period = Test`: `reclaim_pct_mean`, `later_active_mean` (mean of 20 sample selections) |
| Table 9, rolling window | `revision/adaptive_ir2/experiments/naive_window/naive_summary.csv` |
| Other numbers in Sections 3–4 | the files listed in the tables below, and `manuscript_numbers/manuscript_numbers_check.csv` |

`comparison/manuscript_figure_table_map.csv` lists, for every Section 3–4 figure and table, the notebook cell that produced it. `revision/_cache/` holds the outputs of the online replay and of the threshold step (`run.sh online`, `run.sh sensitivity`), which later steps read; despite the name, these are result files, not temporary files.

Figure 1 and Tables 1, 4 and 6 are descriptive (cluster architecture, hardware, termination codes, thresholds) and are not computed. The termination-code grouping of Table 4 is applied in preprocessing step 1.

## Data

The three archives are attached to the GitHub release [`data-v1`](https://github.com/GMM-AI/ir2-academic-gpu-cluster/releases/tag/data-v1).

| Archive | Contents | Size (zipped) |
|---|---|---|
| `raw_scheduler_logs.zip` | `01_RawData/usage_logs/`: 14 scheduler log exports (mostly one per quarter) of the containers that terminated from April 2023 to November 2025 and from March to August 2026. 40,645 rows: 39,886 containers, of which 476 also appear in the next export, and 283 rows whose container ID is `None` (the preprocessing keeps the latest row per container and drops the `None` rows) | 19 MB (6.5 MB) |
| `raw_minute_traces.zip` | `01_RawData/time_series/`: 16,285 per-container minute-level traces (CPU, memory, GPU, I/O, network), one CSV per container, for containers created from 26 September 2024 with an integer (or no) GPU allocation; 825 of the files are empty | 4.1 GB (514 MB) |
| `processed.zip` | `03_ProcessedData/`: the four files the analyses read (below) | 1.3 GB (853 MB) |

| Processed file | Contents |
|---|---|
| `UsageLog20230401_20251130_20260301_20260831.csv` | One row per container, both periods (39,853 containers) |
| `UsageLog20230401_20251130.csv` | One row per container, derivation period only (33,573 containers; Section 3) |
| `TimeSeries_20240926_20251130_20260301_20260831.parquet` | Cleaned minute samples of the 15,460 monitored containers, one parquet row group per container |
| `TimeSeries20240926_20251130.parquet` | The same for the derivation period only |

`run_preprocessing.sh` also writes two per-container folders, `02. Data_TimeSeries_binary` (minute samples after unit conversion) and `03. Data_TimeSeries_binary_with_timefeat` (after cleaning, with elapsed time and the idle label); the merged parquets are built from the second one. The dates in the file names are the first and last days of the exported periods.

There is a gap in the monitoring data from 24 December 2025 to 25 February 2026 (metric-collection failure); the paper's two periods are the complete months before and after it.

**From raw logs to the analysed containers**

| Step | Containers |
|---|---|
| Raw scheduler logs (unique container IDs) | 39,886 |
| Created within the two periods (KST); 33 containers created before March 2023, in December 2025 – February 2026, or after August 2026 are dropped | 39,853 = 33,573 derivation + 6,280 later |
| Section 3 (Tables 2, 3, 5; Figures 2–6) | 33,573 (derivation period) |
| Minute traces: created from 26 September 2024, integer or no GPU allocation | 16,285 |
| With at least one sample after cleaning (Section 4, IR2) | 15,460 = 10,971 derivation + 4,489 later |
| Derivation period by class (Table 7) | 10,971 = 6,724 normal + 4,247 low-utilization |

**Anonymization.** E-mail addresses, session names, session IDs, project names, node host names and storage paths were replaced:
user `u0001`…, session name `n000001`… (one label per container), session ID `s000001`…, node `node001`… (number kept), project `project1`…, image registry host removed, mount paths emptied.
Container names are rebuilt from these fields, `<created yymmdd>-<terminated yymmdd>-<user>-<session name>-<session id>-<container id>`, so the preprocessing produces the same names from the raw data.
The labels were assigned so that names sort in the same order as the original names; the adaptive experiments and the bootstraps draw containers by position, so this keeps every result identical to the paper.
Container IDs (platform hashes) are kept because the files are joined on them. Inside the minute traces only the node name changes; every other value is unchanged.

**Notes on the data**
- One raw trace was not released: its file name did not match the scheduler log (session name `2026.03.18` versus `2026.3.18`), so the analysis never used it.
- The derivation-period UsageLog used for the paper had been opened and re-saved with a spreadsheet program. `run_preprocessing.sh` regenerates it with a different row order, readable `device_type` labels instead of garbled ones, and full precision in the I/O and network columns (the re-saved file had rounded I/O byte counts to six significant digits and a few network values slightly). The columns the analyses use are identical, and the results are identical.
- `device_type` is `GPU-enabled` or `Non-GPU` (the authors' original notebooks wrote the same labels in Korean; both sort in the same order).
- All times inside the files are Korea Standard Time (KST, UTC+9). The two dates at the start of each container name (`filename` column, trace file names) are UTC dates in the full-period UsageLog and the traces, and KST dates in the derivation-period UsageLog, as the original notebooks wrote them; the analyses join the files on container IDs, not on these names.
- Traces of fractional-GPU containers, of containers created before monitoring started (26 September 2024), and 22 traces without a scheduler record are not released; the analysis does not use them.

## Terms

| Term | Meaning |
|---|---|
| IR2 | Idle Resource Reclamation, the policy in the paper: a container is reclaimed after it has been idle for a timeout that depends on its class (non-GPU or GPU-enabled) and phase (pre or post). Timeouts 85/146 min (non-GPU, pre/post) and 119/170 min (GPU-enabled). |
| Derivation period (code: `Train`) | Containers created before 1 December 2025. Section 3 uses all of them (the logs start with containers that terminated in April 2023); Section 4 uses those with minute traces (created from 26 September 2024). The timeouts are derived on this period. |
| Later period (code: `Test`) | Containers created from 1 March to 31 August 2026; IR2 is evaluated on this period without recalibration. |
| Non-GPU, GPU-enabled (`gpu_bin`: `Non-GPU`, `GPU-Enabled`) | Containers without and with a GPU allocation. |
| Integer, fractional GPU | Whole GPUs (1–4) versus fractional allocations (`0.x`, or `1.x–3.x` GPU). Because utilization cannot be attributed reliably to fractional allocations in this cluster, they are excluded from Section 4 (Section 4.5 bounds their share). `gpu_allocated(group)` is the allocation group (`0.0_GPU`, `1.0_GPU`, …). |
| Idle sample (Eq. 1) | A minute sample with CPU ≤ 10%, memory ≤ 15% (non-GPU) or 20% (GPU-enabled) and, if a GPU is allocated, GPU = 0%. Stored per sample as `is_lowutil` = `low_util` (idle) or `normal`. |
| Low-utilization container (Table 7) | A container in which every sample is idle; all other containers are **normal**. The timeouts are derived from normal containers. |
| Pre, post (phase) | Idle time before the first non-idle sample (setup) and after the last one. ("Hoarding time" in `Run03` means the same.) |
| Online replay (Algorithm 1) | IR2 applied to the traces in time order, deciding only from what has been observed. **Hindsight accounting** (Eq. 2) knows the last active time in advance; it is reported for comparison. |
| Later activity, return | After a reclamation decision, the container was observed active again (the paper's disruption measure). |
| Observation samples | The fraction ρ (5%) of containers that adaptive IR2 never reclaims; their complete idle times update the timeouts. |
| Adaptive IR2 configurations | `C_*`: cold start (initial timeout `tau0` = 60 min); `W_*`: warm start from the derivation-period timeouts; `N_*`: one timeout per class without the pre/post split; `O`: best fixed timeouts in hindsight (frontier, for comparison only). `tau` is the timeout. |
| Single 150 min (`Static150`, "Single static") | One timeout of 150 min for both phases and classes, selected on the derivation period so that its share of containers with later activity does not exceed that of IR2. |
| 1%-for-1-hour rule | The cluster's existing static idle-utilization rule (utilization below 1% for one hour). IR2's savings are reported beyond what this rule already reclaims. |
| Baseline | Allocated resource-time (the denominator of the savings), e.g. `resources.baseline`, `baseline_gpu_min`. In `Run03` and `utilization_cutoff_uniform_pp.py`, `BASE`/`Baseline` is the paper's set of utilization cutoffs; in notebook setup cells `BASE` is the data folder. |
| Frozen | Fixed on the derivation period before the evaluation (timeouts, cohort sizes). |
| Official, original | An output of an earlier step of the same `run_all.sh` run that a later step must reproduce exactly; "original code" is the authors' notebook code that produced the submitted manuscript. |
| fill0, missing-as-zero | An empty CPU/memory/GPU value in an existing minute sample is counted as 0% utilization. |
| pp, P90, P70 | Percentage points; 90th and 70th percentiles. |
| `agent` | The compute node the container ran on (`node001`, …). |
| `status_info(group)` | The termination type grouped as in Table 4. |

## Repository layout

```
.
├── README.md, requirements.txt
├── project_paths.py          Folder constants and a guard that keeps reads/writes inside the repository
├── fgcs.py, run.sh           Command dispatcher: ./run.sh online|sensitivity|static|grace|manuscript|run-notebook <path>
├── run_preprocessing.sh      Raw data -> processed data (steps listed below)
├── run_all.sh                Processed data -> all results, in dependency order
├── inputs/                   Analysis inputs fixed before the evaluation, and preprocessing records
├── 02_Preprocessing/         Raw data -> processed data
├── 04_Analysis/              Analyses (section3, section4, revision), figures, number checks
└── 05_Results/               Created when you run the code (not in the repository)
```

`04_Analysis/section3` and `section4` hold the analyses of the original submission; `04_Analysis/revision` holds the code added during the revision, including the online replay of IR2. The number prefixes of the revision notebooks (04, 05, 06, 09, 10) and of output files (`02_`, `06_`, `08_`, …) are the authors' analysis IDs: output `02_*` comes from `online_ir2/simulation.py`, `06_*` from `policy_comparison.py` and notebook 06, `08_*` from `grace_return_analysis.py`. Missing numbers are analyses that are not part of the paper.

In the Section 3–4 notebooks, `_MF_CELL = n` and the cell metadata `original_cell_index` give the cell's position in the authors' original notebook; captured outputs (`cellNN_*`) and `manuscript_figure_table_map.csv` (`original_cell=n`) refer to that number, not to the position in the released notebook.

### `inputs/`

| File | Role |
|---|---|
| `cohort_contract.json` | Values fixed on the derivation period that the analyses assert against: containers per period (`population_counts`), class (`class_counts`) and group (`group_counts`); parquet row groups (= containers) and rows (= minute samples); allocated and saved resource-minutes per period (`resources`); online decisions and returns (`online_counts`); `calibration_counts` = [normal containers, valid pre cases, valid post cases]; applied timeouts (`frozen_pre`, `frozen_post`). |
| `calibration_thresholds_fill0.csv` | The derived percentile values (`raw_min`) and the applied timeouts (`frozen_applied_min`), used as a check by `sensitivity.py` and `adaptive_replay.py`. |
| `preprocessing_file_audit.csv` | Per-trace record of the preprocessing (status, rows read, rows before and after cleaning (`pre_rows`, `post_rows`; not the pre/post phases)); rewritten by `run_preprocessing.sh`. The columns `old_rows_removed`, `new_rows_added` and `common_*` belong to a comparison with earlier files that is not part of the release and are empty or trivial. |
| `preprocessing_validation.json` | Totals of the preprocessing (15,460 containers, 26,542,007 samples; 1,198 raw GPU values above 100%, of which 1,182 exceed the allocation and are removed; 218,547 empty GPU values set to 0; `usage_sha256` is the hash of the full-period UsageLog); rewritten by `run_preprocessing.sh`. |

### `02_Preprocessing/` (run by `run_preprocessing.sh`)

| Step | File | What it does |
|---|---|---|
| 1 | `1_UsageLog_FullPeriod.ipynb` | Scheduler logs → full-period UsageLog (deduplication, units, termination groups of Table 4, UTC→KST) |
| 2 | `2_UsageLog_DerivationPeriod.ipynb` | Scheduler logs → derivation-period UsageLog. Its container names carry KST dates because the time conversion runs before the names are built, as in the file used for the paper |
| 3 | `update_usage_logs.py` | Adds the GPU utilization normalized by the allocated GPU count |
| 4 | `used_time_from_run_start.py` | Execution time = termination − start of execution (Section 3.1) |
| 5 | `rebuild_time_series.py` | Minute traces → per-container parquet and the merged parquets; empty values → 0, GPU samples above the allocation removed, idle label of Eq. (1) |
| — | `run_notebook.py` | Executes a notebook (used by `run.sh run-notebook`) |

### `04_Analysis/` (run by `run_all.sh`)

**Core code**

| File | What it does | Paper |
|---|---|---|
| `revision/online_ir2/simulation.py` | Idle predicate (Eq. 1), online IR2 replay (Algorithm 1), reclaimed resource-time, later activity, idle gaps; hindsight accounting (Eq. 2) | 4.3–4.5, Table 9 |
| `revision/threshold_sensitivity/sensitivity.py` | Pre/post idle times of normal containers; percentile timeouts | 4.2, Fig. 12 inputs |
| `revision/adaptive_ir2/adaptive_replay.py` | Adaptive update (Eq. 3) with observation samples; shared replay functions | 4.3 |
| `revision/cohort_contract.py`, `revision/cache_paths.py`, `section4/common_monitored_cohort.py` | Shared checks and the monitored cohort | — |

**Section 3 and 4.1–4.2**

| File | Produces | Paper |
|---|---|---|
| `section3/Run01_Section3.ipynb` | Container counts, allocations, termination types, creation times | Tables 2, 3, 5; Figs. 2–6; 3.2–3.3 |
| `revision/09_FebruaryWorkloadMixThresholdStability.ipynb` | Monthly workload mix, February 2025 split | 3.2 (crossover, 59–72%, 79 vs 223 min) |
| `section4/Run00_Figure7.ipynb` + `figure7_gpu_normalization.py` + `figure7_original_cells.py` | Utilization CDFs | Fig. 7, 43% |
| `section4/Run02_Utilization.ipynb` + `validate_figure8.py` | Average utilization; containers with 0% GPU | Fig. 8 |
| `section4/Run03_IdleDistribution.ipynb` | Low-utilization containers, idle ratios, pre/post idle times, concentration, timeouts | Table 7, Table 8, Figs. 9–11, 85/119/146/170 min |
| `export_manuscript.py` | Copies the notebook tables into `05_Results/section3|4/tables`; recomputes Table 8 independently | Tables 2, 3, 5, 7, 8 |

**Section 4.4–4.5**

| File | Produces | Paper |
|---|---|---|
| `revision/policy_comparison.py` | Single 150-min timeout selected on the derivation period; fixed IR2 vs single timeout | 4.4, Table 9 |
| `revision/06_ConfidenceIntervalAnalysis.ipynb` | Bootstrap (5,000) of the hindsight accounting | 4.4 (19.97%, 14.26%, …) |
| `revision/additional_analyses/audit_analyses.py` | B1: online bootstrap confidence intervals; B2: 60-min duration predicate, best single vs best pair of timeouts; B3: share of fractional-GPU GPU-time | 4.4 (30.44% [26.48–34.67], …), 4.5 (16.0%, 25.5%) |
| `revision/adaptive_ir2/adaptive_experiments.py` | Adaptive IR2 (cold and warm start, 20 sample selections, parameter sensitivity), static grid | 4.4, Table 9, Fig. 13 |
| `revision/adaptive_ir2/subset_frontier.py` | Hindsight frontier on the same containers; concentration | 4.4 (0.23/0.47, 85.6%/94.3%) |
| `revision/adaptive_ir2/naive_window.py` | Rolling window without observation samples | 4.4 (146 → 8 min, 764), Table 9, Fig. 13 |
| `revision/adaptive_ir2/paper_tradeoff_figure.py` | Percentile trade-off | Fig. 12 |
| `revision/utilization_cutoff_uniform_pp.py` (needs `revision/10_UtilizationCutoffAndSamplingSensitivity.ipynb`) | CPU/memory cutoffs ±3 and ±5 pp; notebook 10 also checks how gaps between minute samples are treated | 4.4 |
| `revision/grace_return_analysis.py` | Return delays, idle gaps, grace periods | 4.5 (202 min, 295/125, 973/464) |
| `revision/04_FractionalGPUImpact.ipynb` | Scope of fractional-GPU containers | 3.2 (21.3%), 4.5 |
| `revision/05_ReclaimedCapacityAndQueueing.ipynb` + `additional_analyses/reuse_overlap_online.py` | Waiting requests that fit reclaimed allocations | 4.5 (258, 43, 4.0%) |

**Figures and numbers**

| File | Produces |
|---|---|
| `figure_style/harvest_plot_data.py`, `figure_style/restyle_paper_figures.py` | The paper's Figures 2–13 in their final style (`05_Results/figure_style/figures/`) |
| `manuscript_numbers.py` | Numbers aggregated by hand in the paper (81.5%/40.9%, 1,308/706/425/602/347, 31.77%, 99.5%, 36.6%, 18–30%, …), checked against the paper |
| `run_manuscript.py`, `manuscript_runs.json` | Runs the Section 3/4 notebooks, then `export_manuscript.py` |

## Notes

- Notebooks are executed in place by `run_notebook.py`, and a run also rewrites `04_Analysis/manuscript_runs.json` and `inputs/preprocessing_*`; `git checkout .` restores the committed versions.
- `rebuild_time_series.py` refuses to overwrite existing parquets; delete `03_ProcessedData/` before rerunning the preprocessing.
- The Section 3–4 notebooks are the authors' original analysis notebooks and are run in full, including exploratory cells whose output is not in the paper and commented-out code kept from the original. Their setup cells still contain a Colab/local path switch; in this repository both paths point to `03_ProcessedData/`.
- Comments in the code are partly in Korean.

## License

Code: to be decided. Data: to be decided.

## Contact

Myungmin Gil, Euiseong Seo (Sungkyunkwan University).
