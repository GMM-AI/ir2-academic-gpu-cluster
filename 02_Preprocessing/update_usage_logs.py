"""Preserve scheduler rows and normalize integer-GPU averages exactly once."""
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project_paths import install_data_io_guard, PROCESSED_DATA, RAW_USAGE, inside
install_data_io_guard()
import numpy as np
import pandas as pd

def run():
    reports=[]
    raw_frames=[pd.read_csv(p,usecols=['container_ids','terminated_at','gpu_allocated','gpu_util_avg'],low_memory=False) for p in sorted(RAW_USAGE.rglob('*.csv'))]
    raw_source=pd.concat(raw_frames,ignore_index=True)
    raw_source['terminated_at']=pd.to_datetime(raw_source.terminated_at,format='mixed',errors='coerce',utc=True)
    raw_source=raw_source.sort_values('terminated_at',ascending=False).drop_duplicates('container_ids',keep='first').dropna(subset=['container_ids']).set_index('container_ids')
    for source in sorted((ROOT/'05_Results/preprocessing/stage1').glob('UsageLog*.csv')):
        data=pd.read_csv(source,low_memory=False)
        before=data.copy(deep=True)
        gpu=pd.to_numeric(data['gpu_allocated(EA)'],errors='coerce')
        original_raw=pd.to_numeric(data['gpu_util_avg(%)'],errors='coerce')
        assert data.container_id.isin(raw_source.index).all(), 'Raw scheduler log lacks containers of the processed UsageLog'
        raw=pd.to_numeric(data.container_id.map(raw_source.gpu_util_avg),errors='coerce')
        raw_allocation=pd.to_numeric(data.container_id.map(raw_source.gpu_allocated),errors='coerce')
        assert np.isclose(raw,original_raw,rtol=0,atol=1e-10,equal_nan=True).all(), 'Raw GPU utilization differs from the processed UsageLog'
        assert np.isclose(gpu,raw_allocation,rtol=0,atol=1e-10,equal_nan=True).all(), 'Raw GPU allocation differs from the processed UsageLog'

        integer=gpu.gt(0)&gpu.mod(1).eq(0)&np.isfinite(gpu)
        valid=integer&np.isfinite(raw)&raw.ge(0)&raw.le(gpu*100)
        data['gpu_util_avg_raw(%)']=raw
        data['gpu_util_avg_normalized(%)']=np.where(valid,raw/gpu.where(integer),np.nan)
        data['gpu_util_normalization_status']=np.select(
            [gpu.eq(0),gpu.gt(0)&gpu.mod(1).ne(0),integer&raw.isna(),integer&~valid,valid],
            ['not_allocated','fractional_unreliable','missing','invalid_range','normalized_integer'],default='unknown_allocation')
        pd.testing.assert_frame_equal(data[before.columns],before)
        assert len(data)==len(before)
        assert data.loc[valid,'gpu_util_avg_normalized(%)'].between(0,100).all()
        assert data.loc[valid,'gpu_util_avg_raw(%)'].eq(0).equals(data.loc[valid,'gpu_util_avg_normalized(%)'].eq(0))
        target=inside(PROCESSED_DATA/source.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        data.to_csv(target,index=False)
        reports.append({'file':source.name,'rows':len(data),'fractional_rows_retained':int((gpu.gt(0)&gpu.mod(1).ne(0)).sum()),
                        'positive_integer_gpu_rows':int(integer.sum()),'invalid_gpu_averages':int((integer&raw.notna()&~valid).sum()),
                        'normalized_valid_rows':int(valid.sum()),'changed_positive_integer_values':int((valid&~np.isclose(raw,data['gpu_util_avg_normalized(%)'],equal_nan=True)).sum()),
                        'legacy_columns_unchanged':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                        'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    assert len(reports)==2
    report={'status':'PASS','strategy':'read GPU metrics from raw UsageLog; validate frozen metadata; retain all original columns and add normalized GPU average columns',
            'selection':'positive integer GPU only; Non-GPU and fractional remain in scheduler data without an invented normalized GPU value',
            'files':reports}
    output=ROOT/'05_Results/preprocessing/usage_normalization_validation.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':run()
