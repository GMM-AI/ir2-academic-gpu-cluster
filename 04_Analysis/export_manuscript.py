"""Export and verify canonical paper figures/tables after complete notebook execution."""
from pathlib import Path
import json, shutil, hashlib,re,argparse,subprocess,sys
parser=argparse.ArgumentParser()
args=parser.parse_args()
import pandas as pd,numpy as np

ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'project_paths.py').is_file():raise RuntimeError('Run from current project 04_Analysis')
jobs=json.loads((ROOT/'04_Analysis/manuscript_runs.json').read_text())
assert all(j['status']=='PASS' for j in jobs), [(j['key'],j['status']) for j in jobs]
outs={j['key']:ROOT/j['output'] for j in jobs}
inventory=[]
def record(kind,number,status,source,outputs,details=''):
    for p in outputs:assert p.is_file() and p.stat().st_size>0,p
    inventory.append({'kind':kind,'number':number,'status':status,'source':source,
      'outputs':[str(p) for p in outputs],'details':details,
      'output_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}})
def export_plot(key,cell,plot,number,name,section):
    folder=ROOT/f'05_Results/section{section}/figures';folder.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for ext in ['png','pdf']:
        source=outs[key]/f'figures/cell{cell:02d}_plot{plot:02d}.{ext}'
        target=folder/f'{name}.{ext}';shutil.copy2(source,target);outputs.append(target)
    record('Figure',number,'REEXECUTED',f"{next(j['notebook'] for j in jobs if j['key']==key)}:original_cell={cell}:plot={plot}",outputs)
def export_table(key,cell,var,number,name,section):
    source=outs[key]/f'capture/cell{cell:02d}_{var}.csv'
    data=pd.read_csv(source,index_col=0)
    folder=ROOT/f'05_Results/section{section}/tables';folder.mkdir(parents=True,exist_ok=True)
    target=folder/name;data.to_csv(target,index=False if number in [2,5] else True)
    record('Table',number,'REEXECUTED',f"{next(j['notebook'] for j in jobs if j['key']==key)}:original_cell={cell}:variable={var}",[target])
    return data

def export_figure8():
    # Validate the current executed notebook inputs before publishing its plot.
    subprocess.run([sys.executable, str(ROOT/'04_Analysis/section4/validate_figure8.py')], cwd=ROOT, check=True)
    export_plot('section4_02',19,1,8,'0409_Resource_Usage_of_Containers_with_Unused_GPUs',4)
    # Describe Figure 8 from its validation file.
    _f8=json.loads((ROOT/'05_Results/section4/tables/figure8_validation.json').read_text())
    inventory[-1]['details']=f"GPU average = 0 (missing averages set to 0 by the original code); integer GPU allocations; n={_f8['selected_containers']} ({_f8['missing_gpu_average_filled_zero_by_original_code']} missing averages)."


for number,cell,name in [(2,19,'0301_gpu_count_usage_time_cdf'),
 (3,16,'0303_containers_vs_conference_schedule_year_month'),(4,14,'0306_container_creation_by_time'),
 (5,28,'0309_execution_time_by_gpu_count_and_termination_reason'),
 (6,30,'0313_Hourly distribution of termination reasons for containers by GPU Usage')]:
    export_plot('section3_01',cell,1,number,name,3)
counts=export_table('section3_01',9,'gpu_class_table',2,'Table02_container_counts.csv',3)
assert counts['count'].sum()==33573
export_table('section3_01',12,'count_con_all',3,'Table03_gpu_allocations.csv',3)
export_table('section3_01',26,'df_table',5,'Table05_termination_counts.csv',3)


f7=[ROOT/f'05_Results/section4/figures/0401_resource_utilization_container_count_cdf.{ext}' for ext in ['png','pdf']]
assert json.loads((ROOT/'05_Results/section4/tables/figure7_validation.json').read_text())['status']=='PASS'
record('Figure',7,'REEXECUTED','04_Analysis/section4/Run00_Figure7.ipynb; original native plot code',f7)
export_figure8()
export_plot('section4_03',21,1,9,'0403_idle_ratio_of_normal_containers',4)
export_plot('section4_03',37,1,10,'0405_normal_container_idle_time_distribution',4)
export_plot('section4_03',37,2,11,'0408_normal_container_idle_time_cdf',4)
low=export_table('section4_03',14,'final_table',7,'Table07_lowutil_counts.csv',4)
# Expected values from the cohort contract
CONTRACT=json.loads((ROOT/'inputs/cohort_contract.json').read_text())
assert low.loc['Total','Total containers']==CONTRACT['class_counts']['Train']['total']
assert low.loc['Total','Low-util containers']==CONTRACT['class_counts']['Train']['low_util']
t8=ROOT/'05_Results/section4/tables/Table08_idle_concentration.csv'
record('Table',8,'REEXECUTED','04_Analysis/section4/Run03_IdleDistribution.ipynb:original_cell=37:BASE',[t8])

# Independent arithmetic of top-k concentration from the full exported BASE cases.
cases=pd.read_parquet(ROOT/'05_Results/section4/tables/Figure10_11_prepost_cases.parquet')
shares=pd.read_csv(t8);share_checks=[]
for _,row in shares.iterrows():
    d=cases if row.group=='Total' else cases.loc[cases.gpu_bin.eq(row.group)]
    flag,col=('is_pre_case','pre_idle_min') if row.phase=='Pre' else ('is_post_case','post_eol_idle_min')
    values=d.loc[d[flag],col].sort_values(ascending=False)
    k=int(np.ceil(len(values)*row.top_percent/100));pct=100*values.iloc[:k].sum()/values.sum()
    ok=k==int(row.n_containers) and np.isclose(round(pct,2),row.share_percent,atol=1e-8,rtol=0)
    share_checks.append({'phase':row.phase,'group':row.group,'top_percent':row.top_percent,'pass':bool(ok)})
assert all(c['pass'] for c in share_checks)
definition=ROOT/'05_Results/section4/tables/Figure10_11_definition.json'
meta=json.loads(definition.read_text());meta['input']='03_ProcessedData/TimeSeries20240926_20251130.parquet'
definition.write_text(json.dumps(meta,ensure_ascii=False,indent=2))

assert {i['number'] for i in inventory if i['kind']=='Figure'}==set(range(2,12))
assert {i['number'] for i in inventory if i['kind']=='Table'}=={2,3,5,7,8}
validation={'status':'PASS','computed_figures_reexecuted':10,'computed_tables_reexecuted':5,
 'all_original_code_cells_executed':all(not j.get('original_code_cells_unexecuted') for j in jobs),
 'execution_jobs':[{k:j[k] for k in ['key','status','seconds','original_code_cells','execution_errors']} for j in jobs],
 'table8_independent_checks':len(share_checks),
 'train_containers':CONTRACT['class_counts']['Train']['total'],'normal_containers':CONTRACT['class_counts']['Train']['normal'],'lowutil_containers':CONTRACT['class_counts']['Train']['low_util'],
 'policy_timeouts_frozen':[85,119,146,170],
 'coverage':sorted(inventory,key=lambda r:(r['kind'],r['number']))}
dest=ROOT/'05_Results/comparison/manuscript_full_execution_validation.json'
dest.write_text(json.dumps(validation,ensure_ascii=False,indent=2))
pd.DataFrame([{**{k:v for k,v in i.items() if k not in ['outputs','output_sha256']},
 'outputs':' | '.join(i['outputs'])} for i in inventory]).to_csv(ROOT/'05_Results/comparison/manuscript_figure_table_map.csv',index=False)
print(json.dumps({k:v for k,v in validation.items() if k not in ['coverage','execution_jobs']},indent=2))
