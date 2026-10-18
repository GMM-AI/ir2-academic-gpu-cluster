# Locate the repository root and load the folder constants.
from pathlib import Path
import sys
_start = Path.cwd().resolve()
_root = next((p for p in [_start,*_start.parents] if (p/'project_paths.py').is_file()), None)
if _root is None:
    raise RuntimeError('Run from inside the repository.')
for _p in [_root,_root/'04_Analysis/revision',_root/'02_Preprocessing']:
    sys.path.insert(0,str(_p))
from project_paths import *
install_data_io_guard()
PROJECT_ROOT = REVISION_CODE
ensure_dirs()

"""Rebuild the per-container and merged minute-trace parquets from the raw CSV traces."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, hashlib, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from project_paths import *
install_data_io_guard()
OUT=PROCESSED_DATA
RAW=RAW_TIME_SERIES
USAGE=PROCESSED_DATA/'UsageLog20230401_20251130_20260301_20260831.csv'
usage=pd.read_csv(USAGE,low_memory=False)
usage['container_id']=usage.container_id.astype(str).str.strip()
assert not usage.container_id.duplicated().any()
valid=usage[usage['gpu_allocated(EA)'].notna() & (usage['gpu_allocated(EA)']%1==0)].set_index('container_id')
valid_sessions=set(usage.filename.astype(str).str.replace('.csv','',regex=False).str.strip())
allfiles=sorted(RAW.rglob('*.csv'))
files=[p for p in allfiles if p.name.split('-')[0].isdigit() and int(p.name.split('-')[0])>=20240926 and p.stem.split('-')[-1] in valid.index]
assert len(files)==16285
oldpost={}
paths=files+[USAGE]
source_stat={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
OUT.mkdir(parents=True,exist_ok=True)
for name in ['02. Data_TimeSeries_binary','03. Data_TimeSeries_binary_with_timefeat']:(OUT/name).mkdir(exist_ok=True)
KEEP=['time','cpu_util(%)','mem_util(%)','gpu_util(%)','time_progress_rate(%)','used_time(minute)','gpu_allocated(group)','is_lowutil','filename']
MERGED=OUT/'TimeSeries_20240926_20251130_20260301_20260831.parquet'
assert not MERGED.exists(),'Refusing to overwrite completed corrected data.'

def process(p):
    cid=p.stem.split('-')[-1];meta=valid.loc[cid];gpu=float(meta['gpu_allocated(EA)']);cpu=float(meta['cpu_allocated(Core)'])
    assert cpu>0 and np.isfinite(cpu)
    rel=p.relative_to(RAW).with_suffix('.parquet')
    r={'filename':p.stem,'container_id':cid,'gpu_allocated':gpu,'cpu_allocated':cpu,'period':'Train' if pd.Timestamp(meta.created_time)<pd.Timestamp('2025-12-01') else 'Test','raw_rows':0,'pre_rows':0,'post_rows':0,'raw_gpu_gt100':0,'gpu_above_allocation_max':0,'cpu_negative':0,'gpu_negative':0,'cpu_above_allocation_max':0,'status':'empty','raw_missing_time':0,'common_idle_differences':0,'common_used_time_differences':0,'old_rows_removed':0,'new_rows_added':0,'missing_filled_cpu':0,'missing_filled_mem':0,'missing_filled_gpu':0}
    if p.stat().st_size==0:return r
    try:d=pd.read_csv(p,low_memory=False)
    except pd.errors.EmptyDataError:return r
    r['raw_rows']=len(d)
    if d.empty:return r
    for col in ['CPU util (%)','Mem util (%)','GPU util (%)']:d[col]=pd.to_numeric(d[col],errors='coerce')
    # An existing sample with an empty metric is zero utilization.
    r.update(missing_filled_cpu=int(d['CPU util (%)'].isna().sum()),missing_filled_mem=int(d['Mem util (%)'].isna().sum()),missing_filled_gpu=int(d['GPU util (%)'].isna().sum()))
    d[['CPU util (%)','Mem util (%)','GPU util (%)']]=d[['CPU util (%)','Mem util (%)','GPU util (%)']].fillna(0.0)
    r.update(raw_gpu_gt100=int((d['GPU util (%)']>100).sum()),gpu_above_allocation_max=int((d['GPU util (%)']>gpu*100).sum()) if gpu>0 else 0,cpu_negative=int((d['CPU util (%)']<0).sum()),gpu_negative=int((d['GPU util (%)']<0).sum()),cpu_above_allocation_max=int((d['CPU util (%)']>cpu*100).sum()))
    # Provider timestamps are UTC. Store KST wall clock, matching UsageLog.
    d['time']=pd.to_datetime(d.time,format='mixed',errors='coerce',utc=True).dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    r['raw_missing_time']=int(d.time.isna().sum());d=d.dropna(subset=['time'])
    d['CPU util (%)']=d['CPU util (%)']/cpu
    d=d[np.isfinite(d['CPU util (%)']) & d['CPU util (%)'].between(0,100)].copy()
    if gpu>0:d['GPU util (%)']=d['GPU util (%)']/gpu
    # Provider defines GPU memory in MB; preserve the supplied unit.
    d=d.drop(columns=['agent'],errors='ignore');r['pre_rows']=len(d)
    if d.empty:return r
    pre=OUT/'02. Data_TimeSeries_binary'/rel;pre.parent.mkdir(parents=True,exist_ok=True);d.to_parquet(pre,index=False,compression='snappy')
    # Preserve the original stage-2 exact filename join in addition to stage-1 ID matching.
    if p.stem.strip() not in valid_sessions:
        r['status']='filename_unmatched'
        return r
    d=d.rename(columns={'CPU util (%)':'cpu_util(%)','Mem util (%)':'mem_util(%)','GPU util (%)':'gpu_util(%)'}).sort_values('time').reset_index(drop=True)
    sec=d.time.astype('datetime64[ns]').astype('int64')//10**9
    minimum=sec.min();span=sec.max()-minimum
    d['used_time(minute)']=(sec-minimum)/60
    d['time_progress_rate(%)']=(sec-minimum)/span*100 if span>0 else 0.
    # Missing metrics were set to zero above; GPU range is checked only for GPU-allocated containers.
    mask=d['cpu_util(%)'].between(0,100)&d['mem_util(%)'].between(0,100)
    mask &= np.isfinite(d[['cpu_util(%)','mem_util(%)']]).all(axis=1)
    if gpu>0:mask &= d['gpu_util(%)'].between(0,100)&np.isfinite(d['gpu_util(%)'])
    d=d.loc[mask].copy();r['post_rows']=len(d)
    if d.empty:return r
    group=meta['gpu_allocated(group)'];d['gpu_allocated(group)']=group
    idle=(d['cpu_util(%)']<=10)&(d['mem_util(%)']<=(15 if gpu==0 else 20))
    if gpu>0:idle &= d['gpu_util(%)']==0
    d['is_lowutil']=np.where(idle,'low_util','normal');d['filename']=p.stem
    r['normal']=bool((~idle).any());r['idle_rows']=int(idle.sum())
    d=d[KEEP]
    out=OUT/'03. Data_TimeSeries_binary_with_timefeat'/rel;out.parent.mkdir(parents=True,exist_ok=True);d.to_parquet(out,index=False,compression='snappy')
    r['status']='ok';r['post_path']=str(out.relative_to(OUT))
    if p.stem in oldpost:
        old=pq.read_table(oldpost[p.stem],columns=['time','used_time(minute)','is_lowutil'],use_threads=False).to_pandas()
        ot=old.time.dt.tz_localize(None).astype('datetime64[ns]').astype('int64').to_numpy();nt=d.time.astype('datetime64[ns]').astype('int64').to_numpy()
        assert len(np.unique(ot))==len(ot) and len(np.unique(nt))==len(nt)
        common,oi,ni=np.intersect1d(ot,nt,return_indices=True)
        r['old_rows_removed']=len(old)-len(common);r['new_rows_added']=len(d)-len(common)
        r['common_idle_differences']=int(np.count_nonzero(old.is_lowutil.to_numpy()[oi]!=d.is_lowutil.to_numpy()[ni]))
        r['common_used_time_differences']=int(np.count_nonzero(~np.isclose(old['used_time(minute)'].to_numpy()[oi],d['used_time(minute)'].to_numpy()[ni],rtol=0,atol=1e-8)))
    else:
        r['new_rows_added']=len(d)
    return r

start=time.monotonic();records=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for i,r in enumerate(pool.map(process,files),1):
        records.append(r)
        if i%1000==0:print('RAW_REBUILD',i,'/',len(files),'seconds',round(time.monotonic()-start,1),flush=True)
result=pd.DataFrame(records);result.to_csv(ROOT/'inputs/preprocessing_file_audit.csv',index=False)
ok=result[result.status=='ok']
schema=pa.schema([('time',pa.timestamp('us')),*[(c,pa.float64()) for c in KEEP[1:6]],*[(c,pa.string()) for c in KEEP[6:]]])
writer=pq.ParquetWriter(MERGED,schema,compression='snappy')
train=pq.ParquetWriter(OUT/'TimeSeries20240926_20251130.parquet',schema,compression='snappy')
try:
    for i,r in enumerate(ok.itertuples(index=False),1):
        t=pq.read_table(OUT/r.post_path,columns=KEEP,use_threads=False).cast(schema)
        writer.write_table(t)
        if r.period=='Train':train.write_table(t)
        if i%4000==0:print('MERGE',i,'/',len(ok),flush=True)
finally:writer.close();train.close()
assert pq.ParquetFile(MERGED).metadata.num_rows==int(result.post_rows.sum())
assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in source_stat.items())
report={'status':'PASS','files_selected':len(files),'files_post':len(ok),'rows_post':int(result.post_rows.sum()),'gpu_gt100_raw_rows':int(result.raw_gpu_gt100.sum()),'gpu_above_allocation_max_raw_rows':int(result.gpu_above_allocation_max.sum()),'cpu_negative_raw_rows':int(result.cpu_negative.sum()),'gpu_negative_raw_rows':int(result.gpu_negative.sum()),'cpu_above_allocation_max_raw_rows':int(result.cpu_above_allocation_max.sum()),'source_stat_unchanged':True,'usage_sha256':hashlib.sha256(USAGE.read_bytes()).hexdigest(),'normalization':'CPU / allocated cores; positive integer GPU / allocated GPU count; Non-GPU is not divided','range':'CPU/Memory finite 0..100; integer GPU finite 0..100; Non-GPU GPU metric not required','missing':'empty CPU/memory/GPU value in an existing sample = 0','missing_filled_cpu_rows':int(result.missing_filled_cpu.sum()),'missing_filled_mem_rows':int(result.missing_filled_mem.sum()),'missing_filled_gpu_rows':int(result.missing_filled_gpu.sum()),'fractional':'kept in unchanged UsageLog; excluded only from time-series cohort','timestamp_convention':'raw UTC converted to naive KST, matching UsageLog; elapsed seconds explicitly derived in nanoseconds','gpu_memory_unit':'MB'}
(ROOT/'inputs/preprocessing_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('FINAL',json.dumps(report,ensure_ascii=False),flush=True)
