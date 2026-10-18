"""Recalculate retained hindsight IR2 accounting with uniform +/-3/5pp cutoffs.

Input, cohort, inclusive predicates, elapsed-time accounting and frozen timeouts
match Notebook 10. This companion does not run online replay or sampling tests.
"""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '04_Analysis/revision'))
from project_paths import PROCESSED_DATA, RESULTS_REVISION, install_data_io_guard
from cohort_contract import current_contract
install_data_io_guard()

PRE = {'Non-GPU': 85, 'GPU-Enabled': 119}  # GPU Pre 119 (Train P90 119.0)
POST = {'Non-GPU': 146, 'GPU-Enabled': 170}
SCENARIOS = [('Baseline', 10, 15, 20, 0)]
for resource in ['CPU', 'Memory']:
    for delta in [-5, -3, 3, 5]:
        SCENARIOS.append((f'{resource} {delta:+d}pp',
                          10 + delta if resource == 'CPU' else 10,
                          15 + delta if resource == 'Memory' else 15,
                          20 + delta if resource == 'Memory' else 20, 0))
AUDIT = RESULTS_REVISION / '10_utilization_cutoff_uniform_pp_audit'
PREFIX = '10_utilization_cutoff_uniform_pp'
USAGE = PROCESSED_DATA / 'UsageLog20230401_20251130_20260301_20260831.csv'
TRACE = PROCESSED_DATA / 'TimeSeries_20240926_20251130_20260301_20260831.parquet'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def key(filename):
    name = str(filename).replace('\\', '/').split('/')[-1]
    return name[:-4] if name.lower().endswith('.csv') else name[:-8] if name.lower().endswith('.parquet') else name

def run():
    started = time.monotonic()
    AUDIT.mkdir(parents=True, exist_ok=True)
    contract = current_contract()
    assert contract['frozen_pre'] == PRE and contract['frozen_post'] == POST
    assert len(SCENARIOS) == 9 and all(s[-1] == 0 for s in SCENARIOS)
    old = pd.read_csv(RESULTS_REVISION / '10_utilization_cutoff_sensitivity.csv')
    protected = set()
    protected.update(str(p.relative_to(ROOT)) for p in [USAGE, TRACE,
        ROOT / '04_Analysis/revision/10_UtilizationCutoffAndSamplingSensitivity.ipynb',
        RESULTS_REVISION / '10_utilization_cutoff_sensitivity.csv'])
    before = {p: sha(ROOT/p) for p in sorted(protected)}
    (AUDIT / 'protected_before.json').write_text(json.dumps(before, ensure_ascii=False, indent=2))
    usage = pd.read_csv(USAGE, usecols=['filename','created_time','terminated_time',
                         'gpu_allocated(group)','gpu_allocated(EA)','cpu_allocated(Core)'], low_memory=False)
    usage['key'] = usage.filename.map(key)
    assert usage.key.notna().all() and not usage.key.duplicated().any()
    usage['created_time'] = pd.to_datetime(usage.created_time, errors='raise')
    usage['population'] = 'Excluded'
    usage.loc[usage.created_time.ge('2023-03-01') & usage.created_time.lt('2025-12-01'), 'population'] = 'Train'
    usage.loc[usage.created_time.ge('2026-03-01') & usage.created_time.lt('2026-09-01'), 'population'] = 'Test'
    for col in ['gpu_allocated(EA)','cpu_allocated(Core)']:
        usage[col] = pd.to_numeric(usage[col], errors='raise')
        assert usage[col].notna().all() and usage[col].ge(0).all()
    lookup = usage.set_index('key', verify_integrity=True)
    parquet = pq.ParquetFile(TRACE)
    assert parquet.metadata.num_row_groups == contract['row_groups']
    assert parquet.metadata.num_rows == contract['rows']
    rows, membership = [], []
    sample_rows = 0
    columns = ['filename','gpu_allocated(group)','used_time(minute)',
               'cpu_util(%)','mem_util(%)','gpu_util(%)']
    for group_id in range(parquet.metadata.num_row_groups):
        block = parquet.read_row_group(group_id, columns=columns).to_pandas()
        assert block.filename.nunique(dropna=False) == 1
        assert block['gpu_allocated(group)'].nunique(dropna=False) == 1
        name = str(block.filename.iloc[0]); k = key(name)
        meta = lookup.loc[k]
        group = str(block['gpu_allocated(group)'].iloc[0])
        assert group == meta['gpu_allocated(group)']
        assert group in {'0.0_GPU','1.0_GPU','2.0_GPU','3.0_GPU','4.0_GPU'}
        gpu_bin = 'Non-GPU' if group == '0.0_GPU' else 'GPU-Enabled'
        gpu_required = gpu_bin == 'GPU-Enabled'
        required = ['used_time(minute)','cpu_util(%)','mem_util(%)'] + (['gpu_util(%)'] if gpu_required else [])
        assert block[required].notna().all().all()
        assert meta.population in {'Train','Test'}
        order = np.argsort(block['used_time(minute)'].to_numpy(float), kind='stable')
        elapsed = block['used_time(minute)'].to_numpy(float)[order]
        cpu = block['cpu_util(%)'].to_numpy(float)[order]
        mem = block['mem_util(%)'].to_numpy(float)[order]
        gpu = block['gpu_util(%)'].to_numpy(float)[order]
        assert np.isfinite(elapsed).all() and np.all(elapsed >= 0)
        assert np.isfinite(cpu).all() and np.isfinite(mem).all()
        assert not gpu_required or (np.isfinite(gpu).all() and np.all(gpu >= 0))
        gpu_units, cpu_units = float(meta['gpu_allocated(EA)']), float(meta['cpu_allocated(Core)'])
        assert gpu_units == (0 if not gpu_required else int(group.split('_')[0].split('.')[0]))
        end, first_observed = float(elapsed[-1]), float(elapsed[0])
        sample_rows += len(block)
        membership.append((k, meta.population, gpu_bin, len(block), group_id))
        for scenario, cpu_limit, mem_non, mem_gpu, gpu_limit in SCENARIOS:
            idle = (cpu <= cpu_limit) & (mem <= (mem_gpu if gpu_required else mem_non))
            if gpu_required: idle &= gpu == 0
            active_indices = np.flatnonzero(~idle)
            normal = bool(len(active_indices))
            first_active = float(elapsed[active_indices[0]]) if normal else np.nan
            last_active = float(elapsed[active_indices[-1]]) if normal else np.nan
            if normal:
                pre_saved = max(first_active - 1 - PRE[gpu_bin], 0)
                post_saved = max(end - last_active - POST[gpu_bin], 0) if bool(idle[-1]) else 0
                pre_candidate = first_active - first_observed > PRE[gpu_bin]
            else:
                pre_saved, post_saved, pre_candidate = max(end-PRE[gpu_bin], 0), 0, False
            saved = pre_saved + post_saved
            assert 0 <= saved <= end + 1e-8
            rows.append((k, meta.population, gpu_bin, scenario, normal, bool(pre_candidate),
                         pre_saved, post_saved, gpu_units*end, cpu_units*end,
                         gpu_units*saved, cpu_units*saved, first_active, last_active,
                         end, first_observed, bool(idle[-1])))
        if (group_id+1) % 3000 == 0:
            print(f'Processed {group_id+1}/{parquet.num_row_groups} containers', flush=True)
    cohort = pd.DataFrame(membership, columns=['key','population','gpu_bin','samples','row_group'])
    assert not cohort.key.duplicated().any()
    assert cohort.population.value_counts().to_dict() == contract['population_counts']
    assert sample_rows == contract['rows']
    for pop in ['Train','Test']:
        assert cohort.loc[cohort.population.eq(pop), 'gpu_bin'].value_counts().to_dict() == contract['group_counts'][pop]
    details = pd.DataFrame(rows, columns=['key','population','gpu_bin','scenario','normal','pre_candidate',
                   'pre_saved_min','post_saved_min','gpu_baseline_min','cpu_baseline_min',
                   'gpu_saved_min','cpu_saved_min','first_active_min','last_active_min',
                   'end_min','first_observation_min','end_is_idle'])
    assert len(details) == contract['row_groups'] * 9
    aggregate = dict(containers=('key','size'),normal=('normal','sum'),pre_candidates=('pre_candidate','sum'),
                     gpu_den=('gpu_baseline_min','sum'),cpu_den=('cpu_baseline_min','sum'),
                     gpu_saved=('gpu_saved_min','sum'),cpu_saved=('cpu_saved_min','sum'))
    summary = details.groupby(['scenario','population']).agg(**aggregate)
    subgroups = details.groupby(['scenario','population','gpu_bin']).agg(**aggregate)
    for frame in [summary, subgroups]:
        frame['low_util'] = frame.containers-frame.normal
        frame['low_util_pct'] = frame.low_util/frame.containers*100
        frame['candidate_pct_normal'] = frame.pre_candidates/frame.normal*100
        frame['gpu_saved_pct'] = frame.gpu_saved/frame.gpu_den.replace(0,np.nan)*100
        frame['cpu_saved_pct'] = frame.cpu_saved/frame.cpu_den*100
    summary = summary.reindex(pd.MultiIndex.from_product([[s[0] for s in SCENARIOS],['Train','Test']],names=['scenario','population']))
    base = summary.loc['Baseline']
    for col in ['gpu_den','cpu_den']:
        assert summary[col].groupby('population').nunique().eq(1).all()
    for pop in ['Train','Test']:
        counts, resources = contract['class_counts'][pop], contract['resources'][pop]
        expected = dict(containers=counts['total'],normal=counts['normal'],low_util=counts['low_util'],
                        pre_candidates=contract['pre_candidates'][pop],gpu_den=resources['GPU']['baseline'],
                        cpu_den=resources['CPU']['baseline'],gpu_saved=resources['GPU']['total_saved'],
                        cpu_saved=resources['CPU']['total_saved'])
        for col,value in expected.items():
            assert np.isclose(base.loc[pop,col],value,rtol=0,atol=.11), (pop,col)
        for scenario, old_name in [('Baseline','원고 기준 / Baseline'),
                                   ('Memory -5pp','메모리 -5pp / Memory lower'),
                                   ('Memory +5pp','메모리 +5pp / Memory higher')]:
            previous = old.loc[old.population.eq(pop) & old.scenario.eq(old_name)].iloc[0]
            for col in ['containers','normal','low_util','pre_candidates','gpu_den','cpu_den','gpu_saved','cpu_saved']:
                assert np.isclose(summary.loc[(scenario,pop),col],previous[col],rtol=0,atol=.11),(scenario,pop,col)
    for col in ['low_util_pct','gpu_saved_pct','cpu_saved_pct']:
        summary[col+'_delta_pp'] = summary[col]-summary.index.get_level_values('population').map(base[col])
    check_columns = ['containers','normal','low_util','pre_candidates','gpu_den','cpu_den','gpu_saved','cpu_saved']
    reconciled = subgroups.groupby(['scenario','population'])[check_columns].sum().sort_index()
    assert np.allclose(reconciled.to_numpy(), summary[check_columns].sort_index().to_numpy(), rtol=0, atol=1e-6)
    for resource in ['CPU','Memory']:
        labels = [f'{resource} -5pp',f'{resource} -3pp','Baseline',f'{resource} +3pp',f'{resource} +5pp']
        for field in ['normal','gpu_saved_min','cpu_saved_min']:
            array = details.pivot(index='key',columns='scenario',values=field)[labels].to_numpy(float)
            differences = np.diff(array,axis=1)
            assert np.all(differences <= 0) if field == 'normal' else np.all(differences >= -1e-8),(resource,field)
    # Export only after the checks pass.
    RESULTS_REVISION.mkdir(parents=True,exist_ok=True)
    summary.to_csv(RESULTS_REVISION/(PREFIX+'.csv'))
    subgroups.to_csv(RESULTS_REVISION/(PREFIX+'_subgroups.csv'))
    details.to_parquet(RESULTS_REVISION/(PREFIX+'_details.parquet'),index=False)
    cohort.to_csv(AUDIT/'cohort_membership.csv',index=False)
    config = dict(scenarios=[dict(name=s[0],cpu=s[1],memory_non_gpu=s[2],memory_gpu_enabled=s[3],gpu=s[4]) for s in SCENARIOS],
                  pre=PRE,post=POST,population_counts=contract['population_counts'],rows=sample_rows,
                  periods={'Train':'created_time in [2023-03-01,2025-12-01)','Test':'created_time in [2026-03-01,2026-09-01)'},
                  accounting='Original Notebook 10 hindsight potential; normal Pre first_active - 1 - Pre; Post end - last_active - Post only if final sample idle; never-active Pre end - Pre',
                  units={'GPU':'GPU-minutes','CPU':'CPU-core-minutes'},gpu_missing_non_gpu='Not used in idle predicate')
    (AUDIT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    assert all(sha(ROOT/p) == h for p,h in before.items()), 'Protected input/reference artifacts changed.'
    validation=dict(status='PASS',containers=len(cohort),observations=sample_rows,scenario_detail_rows=len(details),
                    protected_files_unchanged=len(before),baseline_matches_contract=True,
                    retained_memory_5pp_results_match=True,subgroups_reconcile=True,
                    denominators_fixed=True,container_monotonicity=True,all_gpu_cutoffs_zero=True,
                    seconds=round(time.monotonic()-started,2),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [USAGE,TRACE]},
                    producer_sha256=sha(Path(__file__)))
    (AUDIT/'producer_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    print(json.dumps(validation,ensure_ascii=False),flush=True)
    print(summary[['containers','low_util','low_util_pct','gpu_saved_pct','cpu_saved_pct']].round(6).to_string(),flush=True)

if __name__ == '__main__': run()
