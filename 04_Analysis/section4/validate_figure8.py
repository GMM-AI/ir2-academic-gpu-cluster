"""Validate Figure 8 with original plot logic on the common monitored cohort."""
from pathlib import Path
import sys,json,hashlib,argparse
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from common_monitored_cohort import restrict_to_monitored, TRAIN_N
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-protected',action='store_true');args=parser.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
input_path=ROOT/'03_ProcessedData/UsageLog20230401_20251130.csv'
try:raw=pd.read_csv(input_path,low_memory=False)
except UnicodeDecodeError:raw=pd.read_csv(input_path,encoding='cp949',low_memory=False)
raw=restrict_to_monitored(raw)
a=pd.to_numeric(raw['gpu_allocated(EA)'],errors='coerce')
g=pd.to_numeric(raw['gpu_util_avg_normalized(%)'],errors='coerce').clip(0,100)
mask=a.isin([1,2,3,4])&g.fillna(0).eq(0)
expected=raw.loc[mask,['container_id','gpu_allocated(EA)','cpu_util_avg(%)','run_time','terminated_time']].copy()
expected['GPU']=g.loc[mask].fillna(0)
expected['CPU']=pd.to_numeric(expected['cpu_util_avg(%)'],errors='coerce').clip(0,100).fillna(0)
expected['used_time(minute)']=(pd.to_datetime(expected.terminated_time)-pd.to_datetime(expected.run_time)).dt.total_seconds()/60  # execution start to termination
expected['used_time(minute)']=expected['used_time(minute)'].replace([0,-1],np.nan).fillna(1).clip(lower=1)
expected['gpu_allocated(group)']=expected['gpu_allocated(EA)'].map(lambda v:f'{int(v)} GPU')
capture=ROOT/'05_Results/section4/execution_runs/02_utilization/capture'
actual=pd.read_csv(capture/'cell19_figure8_plot_inputs.csv').set_index('container_id').sort_index()
expected=expected.set_index('container_id').sort_index()
checks={'cohort_train_monitored':len(raw)==TRAIN_N,'plot_ids_match_independent_original_logic':actual.index.equals(expected.index),
 'plot_ids_unique':actual.index.is_unique,'selected_n_original_logic':len(actual)==len(expected),
 'gpu_values_match_original_missing_to_zero_logic':np.allclose(actual.GPU,expected.GPU),
 'cpu_values_match_original_logic':np.allclose(actual.CPU,expected.CPU,rtol=0,atol=1e-12),
 'runtime_matches_original_definition':np.allclose(actual['used_time(minute)'],expected['used_time(minute)'],rtol=0,atol=1e-8),
 'allocation_labels_match':actual['gpu_allocated(group)'].equals(expected['gpu_allocated(group)'])}
statistics={}
for label,col in [('runtime_plotted_min','used_time(minute)'),('cpu_plotted_pct','CPU')]:
 v=actual[col]
 statistics[label]={'valid_n':len(v),'missing_n':int(v.isna().sum()),'zero_n':int(v.eq(0).sum()),'mean':float(v.mean()),'p25':float(v.quantile(.25)),'median':float(v.median()),'p75':float(v.quantile(.75)),'p90':float(v.quantile(.9)),'p99':float(v.quantile(.99)),'max':float(v.max())}
# Containers with a missing GPU average are counted below.
job=next(j for j in json.loads((ROOT/'04_Analysis/manuscript_runs.json').read_text()) if j['key']=='section4_02')
nb=ROOT/job['notebook'];checks['notebook_execution_pass']=job['status']=='PASS' and job['notebook_sha256']==sha(nb) and job['execution_errors']==0 and not job['original_code_cells_unexecuted']
outputs=[ROOT/job['output']/f'figures/cell19_plot01.{ext}' for ext in ['png','pdf']]
checks['png_pdf_exist']=all(p.is_file() and p.stat().st_size>1000 for p in outputs)
result={'status':'PASS' if all(checks.values()) else 'FAIL','cohort_containers':TRAIN_N,'selected_containers':len(actual),
 'observed_zero_gpu_average_containers':int(g.loc[mask].notna().sum()),'missing_gpu_average_filled_zero_by_original_code':int(g.loc[mask].isna().sum()),
 'plot_logic':'Preserved original code including missing utilization to zero and one-minute log plotting floor; only input population changed',
 'duration_definition':'run_time to terminated_time, minimum 1 minute for logarithmic plotting',
 'statistics':statistics,'allocation_counts':{str(k):int(v) for k,v in actual['gpu_allocated(group)'].value_counts().sort_index().items()},'checks':{k:bool(v) for k,v in checks.items()}}
assert result['status']=='PASS',json.dumps(result,ensure_ascii=False,indent=2)
(ROOT/'05_Results/section4/tables/figure8_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
pd.DataFrame(statistics).T.to_csv(ROOT/'05_Results/section4/tables/figure8_summary.csv')
print(json.dumps(result,ensure_ascii=False,indent=2))
