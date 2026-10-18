"""Observed return timing and fixed-grace sensitivity.

A fixed grace that is cancelled by activity is modeled as the existing
continuous-idle timeout plus G. This is a hypothetical sensitivity analysis,
not a selected operational policy or a measurement of user harm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from project_paths import *
from cache_paths import require_frozen_cache

ONLINE_CACHE = require_frozen_cache("online_ir2", ['container_events.csv', 'intermediate_gaps.csv'])

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


WORK = Path(str(REVISION_CODE))
sys.path.insert(0, str(WORK / 'online_ir2'))
sys.path.insert(0, str(WORK / 'threshold_sensitivity'))
from simulation import (  # noqa: E402
    POST, PRE, TRACE, load_usage, normalize_filename, prepare_trace,
)
from sensitivity import first_crossing  # noqa: E402


SOURCE_EVENTS = ONLINE_CACHE / 'container_events.csv'
SOURCE_GAPS = ONLINE_CACHE / 'intermediate_gaps.csv'
REFERENCE_06 = RESULTS_REVISION / '06_policy_summary.csv'
OUTPUT = RESULTS_REVISION
GRACE_MIN = (0, 15, 30, 60, 120)


def checked_source_events() -> pd.DataFrame:
    events = pd.read_csv(SOURCE_EVENTS, low_memory=False)
    events['key'] = events.filename.map(normalize_filename)
    if events.key.isna().any() or events.key.duplicated().any():
        raise AssertionError('02 event filename key missing or duplicated')
    audit = pd.read_csv(ROOT/'inputs/preprocessing_file_audit.csv')
    expected_counts = audit.loc[audit.status.eq('ok')].groupby('period').size().to_dict()
    if events.population.value_counts().to_dict() != expected_counts:
        raise AssertionError('02 eligible population changed')
    events['online_time'] = pd.to_datetime(events.online_time)
    events['first_active_after_online'] = pd.to_datetime(events.first_active_after_online)
    return events.set_index('key', verify_integrity=True)


def return_timing(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = events.loc[events.potential_interruption.astype(bool)].copy()
    if subset.online_time.isna().any() or subset.first_active_after_online.isna().any():
        raise AssertionError('Potential interruption missing timing')
    subset['return_delay_min'] = (
        subset.first_active_after_online - subset.online_time
    ).dt.total_seconds() / 60
    if subset.return_delay_min.le(0).any():
        raise AssertionError('Nonpositive first observed return delay')
    rows = []
    for (pop, gpu_bin, phase), group in subset.groupby(
        ['population', 'gpu_bin', 'online_kind'], sort=True
    ):
        row = {'population': pop, 'gpu_bin': gpu_bin, 'first_event_phase': phase,
               'containers_with_return': len(group),
               'median_delay_min': group.return_delay_min.median(),
               'p90_delay_min': group.return_delay_min.quantile(.9),
               'max_delay_min': group.return_delay_min.max()}
        for grace in GRACE_MIN[1:]:
            row[f'return_within_{grace}_min'] = int(group.return_delay_min.le(grace).sum())
        rows.append(row)
    return subset[['filename', 'population', 'gpu_bin', 'online_kind',
                   'return_delay_min']].reset_index(drop=True), pd.DataFrame(rows)


def observed_gap_summary() -> pd.DataFrame:
    gaps = pd.read_csv(SOURCE_GAPS, low_memory=False)
    if gaps.empty or gaps.gap_min.isna().any() or gaps.gap_min.le(0).any():
        raise AssertionError('Invalid 02 gap data')
    gaps['key'] = gaps.filename.map(normalize_filename)
    if gaps[['key', 'population', 'gpu_bin']].isna().any().any():
        raise AssertionError('Gap provenance missing')
    rows = []
    for (pop, gpu_bin, confirmed), group in gaps.groupby(
        ['population', 'gpu_bin', 'confirmed_idle'], sort=True
    ):
        threshold = POST[gpu_bin]
        rows.append({'population': pop, 'gpu_bin': gpu_bin,
                     'fully_observed_idle': bool(confirmed),
                     'gaps': len(group), 'containers_with_gap': group.key.nunique(),
                     'median_min': group.gap_min.median(),
                     'p90_min': group.gap_min.quantile(.9),
                     'p99_min': group.gap_min.quantile(.99),
                     'max_min': group.gap_min.max(),
                     'at_or_above_frozen_post_timeout': int(group.gap_min.ge(threshold).sum()),
                     'containers_at_or_above_post_timeout':
                         group.loc[group.gap_min.ge(threshold), 'key'].nunique()})
    return pd.DataFrame(rows)


def replay_one(trace: pd.DataFrame, gpu_bin: str) -> list[dict]:
    pre_limits = {g: PRE[gpu_bin] + g for g in GRACE_MIN}
    post_limits = {g: POST[gpu_bin] + g for g in GRACE_MIN}
    times, pre_hits, post_hits, active_tail, idle_tail, active_idx = first_crossing(
        trace, pre_limits, post_limits)
    end = times[-1]
    rows = []
    for grace in GRACE_MIN:
        pre_hit, post_hit = pre_hits[grace], post_hits[grace]
        hit = min((x for x in (pre_hit, post_hit) if x >= 0), default=-1)
        if hit < 0:
            rows.append({'grace_min': grace, 'online_time': pd.NaT,
                         'online_kind': None, 'terminated': 0,
                         'later_active': 0, 'first_return_delay_min': np.nan,
                         'gross_min': 0.0, 'idle_min': 0.0,
                         'active_min': 0.0, 'unknown_min': 0.0})
            continue
        gross = float((end - times[hit]) / 60e9)
        idle = float(idle_tail[hit])
        active = float(active_tail[hit])
        unknown = max(gross - idle - active, 0.0)
        if idle + active > gross + 1e-6:
            raise AssertionError('Observed support exceeds post-event span')
        future = active_idx[active_idx > hit]
        delay = float((times[future[0]] - times[hit]) / 60e9) if len(future) else np.nan
        rows.append({'grace_min': grace, 'online_time': pd.Timestamp(times[hit]),
                     'online_kind': 'Pre' if pre_hit == hit else 'Post',
                     'terminated': 1, 'later_active': int(bool(len(future))),
                     'first_return_delay_min': delay,
                     'gross_min': gross, 'idle_min': idle,
                     'active_min': active, 'unknown_min': unknown})
    if any(rows[i]['gross_min'] < rows[i + 1]['gross_min'] - 1e-6
           for i in range(len(rows) - 1)):
        raise AssertionError('Longer grace increased first-event gross span')
    return rows


def replay_all(usage: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    parquet = pq.ParquetFile(TRACE)
    columns = ['filename', 'time', 'used_time(minute)', 'gpu_allocated(group)',
               'cpu_util(%)', 'mem_util(%)', 'gpu_util(%)']
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise AssertionError('Trace schema changed')
    records = []
    seen = set()
    for k in range(parquet.num_row_groups):
        frame = parquet.read_row_group(k, columns=columns).to_pandas()
        names = frame.filename.dropna().unique()
        if len(names) != 1:
            raise AssertionError(f'Row group {k} must contain one container')
        key = normalize_filename(names[0])
        if key in seen or key not in usage.index or key not in reference.index:
            raise AssertionError(f'Duplicate or unmatched container: {key}')
        seen.add(key)
        meta, original = usage.loc[key], reference.loc[key]
        gpu_bin = 'Non-GPU' if meta['gpu_allocated(group)'] == '0.0_GPU' else 'GPU-Enabled'
        if (meta.population, gpu_bin) != (original.population, original.gpu_bin):
            raise AssertionError('Population or GPU group differs from 02')
        trace, audit = prepare_trace(frame, gpu_bin, meta.terminated_time)
        if audit['duplicate_rows'] or audit['rows_after_termination'] or trace.state.eq('unknown').any():
            raise AssertionError('Trace preparation differs from validated input')
        if not np.isclose(float(trace.elapsed.max()), original.baseline_min):
            raise AssertionError('Baseline denominator differs from 02')
        outcomes = replay_one(trace, gpu_bin)
        first = outcomes[0]
        old_kind = original.online_kind if pd.notna(original.online_kind) else None
        if ((pd.isna(first['online_time']) != pd.isna(original.online_time)) or
            (pd.notna(first['online_time']) and first['online_time'] != original.online_time) or
            first['online_kind'] != old_kind or
            bool(first['later_active']) != bool(original.potential_interruption)):
            raise AssertionError(f'G=0 first event differs from 02: {key}')
        for outcome in outcomes:
            records.append({'filename': key, 'population': meta.population,
                            'gpu_bin': gpu_bin,
                            'cpu_allocated': float(meta['cpu_allocated(Core)']),
                            'gpu_allocated': float(meta['gpu_allocated(EA)']),
                            'baseline_min': float(trace.elapsed.max()),
                            **outcome})
        if (k + 1) % 3000 == 0:
            print(f'[RESULT] replayed {k + 1:,}/{parquet.num_row_groups:,} containers', flush=True)
    if len(seen) != len(reference) or len(seen) != parquet.num_row_groups:
        raise AssertionError('Eligible trace coverage mismatch')
    data = pd.DataFrame.from_records(records)
    if len(data) != len(reference) * len(GRACE_MIN) or data.duplicated(['filename', 'grace_min']).any():
        raise AssertionError('Missing or duplicated scenario rows')
    return data


def summarize(replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pop, grace), population in replay.groupby(['population', 'grace_min']):
        for group in ('GPU-Enabled', 'Non-GPU', 'All'):
            subset = population if group == 'All' else population.loc[population.gpu_bin.eq(group)]
            cpu = subset.cpu_allocated
            gpu = subset.gpu_allocated
            row = {'population': pop, 'grace_min': int(grace), 'gpu_bin': group,
                   'containers': len(subset), 'terminated': int(subset.terminated.sum()),
                   'later_active_containers': int(subset.later_active.sum()),
                   'baseline_cpu_min': float((cpu * subset.baseline_min).sum()),
                   'baseline_gpu_min': float((gpu * subset.baseline_min).sum())}
            for resource, allocation in (('cpu', cpu), ('gpu', gpu)):
                for state in ('gross', 'idle', 'active', 'unknown'):
                    row[f'{state}_{resource}_min'] = float((allocation * subset[f'{state}_min']).sum())
                denominator = row[f'baseline_{resource}_min']
                row[f'gross_{resource}_pct'] = (100 * row[f'gross_{resource}_min'] / denominator
                                               if denominator else np.nan)
            rows.append(row)
    summary = pd.DataFrame(rows).sort_values(['population', 'gpu_bin', 'grace_min'])
    for resource in ('cpu', 'gpu'):
        if not np.allclose(summary[f'idle_{resource}_min'] + summary[f'active_{resource}_min']
                           + summary[f'unknown_{resource}_min'],
                           summary[f'gross_{resource}_min'], atol=.05):
            raise AssertionError(f'{resource} post-event accounting does not close')
    return summary


def validate_reference(summary: pd.DataFrame) -> None:
    ref = pd.read_csv(REFERENCE_06)
    ref = ref.loc[ref.policy.str.startswith('IR2')]
    for row in ref.itertuples(index=False):
        here = summary.loc[summary.population.eq(row.population)
                           & summary.gpu_bin.eq(row.gpu_bin)
                           & summary.grace_min.eq(0)]
        if len(here) != 1:
            raise AssertionError('G=0 summary group missing')
        actual = here.iloc[0]
        for col in ('containers', 'terminated', 'baseline_gpu_min', 'gross_gpu_min',
                    'active_gpu_min', 'idle_gpu_min', 'unknown_gpu_min', 'gross_cpu_min'):
            other = getattr(row, col)
            if not np.isclose(actual[col], other, atol=.05, rtol=0):
                raise AssertionError(f'G=0 differs from 06: {row.population} {row.gpu_bin} {col}')
        if not np.isclose(actual.later_active_containers, row.return_containers):
            raise AssertionError('G=0 later-activity count differs from 06')


def run(output: Path = OUTPUT) -> dict:
    for source in (TRACE, SOURCE_EVENTS, SOURCE_GAPS, REFERENCE_06):
        if not source.is_file():
            raise FileNotFoundError(source)
    reference = checked_source_events()
    delays, timing = return_timing(reference)
    gaps = observed_gap_summary()
    usage = load_usage()
    replay = replay_all(usage, reference)
    summary = summarize(replay)
    validate_reference(summary)
    output.mkdir(parents=True, exist_ok=True)
    delays.to_csv(output / '08_first_observed_return_delays.csv', index=False)
    timing.to_csv(output / '08_return_timing_summary.csv', index=False)
    gaps.to_csv(output / '08_observed_gap_summary.csv', index=False)
    replay.to_parquet(output / '08_grace_container_results.parquet', index=False)
    summary.to_csv(output / '08_grace_summary.csv', index=False)
    checks = {'status': 'PASS', 'eligible': reference.population.value_counts().to_dict(),
              'scenario_grace_min': list(GRACE_MIN), 'scenario_rows': len(replay),
              '02_first_event_match': 'PASS: every container G=0 time/kind/later activity',
              '06_group_accounting_match': 'PASS: all IR2 group rows and resource totals',
              'partition_check': 'PASS: idle + active + unknown = gross',
              'source_events': str(SOURCE_EVENTS), 'source_gaps': str(SOURCE_GAPS),
              'raw_trace': str(TRACE),
              'meaning': 'hypothetical consecutive-idle grace, not realized savings or user harm'}
    (output / '08_validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print('[RESULT]', json.dumps(checks, ensure_ascii=False), flush=True)
    return checks


if __name__ == '__main__':
    run()
