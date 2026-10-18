"""Use the existing IR2 monitored population for Figures 7 and 8, without changing plot logic."""
from pathlib import Path
from functools import lru_cache
import json
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
# Train monitored cohort size (10,971), read from the cohort contract.
TRAIN_N=json.loads((ROOT/'inputs/cohort_contract.json').read_text())['population_counts']['Train']
@lru_cache(maxsize=1)
def monitored_ids():
    path=ROOT/'03_ProcessedData/TimeSeries20240926_20251130.parquet'
    pf=pq.ParquetFile(path);column=pf.schema.names.index('filename');ids=set()
    for i in range(pf.num_row_groups):
        rg=pf.metadata.row_group(i);s=rg.column(column).statistics
        assert rg.num_rows>0 and s is not None and s.min==s.max
        name=s.min.decode() if isinstance(s.min,bytes) else str(s.min)
        for ext in ('.csv','.parquet'):
            if name.lower().endswith(ext):name=name[:-len(ext)]
        ids.add(name.rsplit('-',1)[-1])
    assert len(ids)==pf.num_row_groups==TRAIN_N
    return frozenset(ids)
def restrict_to_monitored(frame):
    assert frame.container_id.notna().all() and frame.container_id.is_unique
    result=frame.loc[frame.container_id.astype(str).isin(monitored_ids())].copy()
    assert len(result)==TRAIN_N
    return result
