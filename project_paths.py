"""Folder constants (relative to the repository root) and a guard that keeps reads and writes inside the repository."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT
RAW_USAGE = SOURCE_ROOT / '01_RawData/usage_logs'
RAW_TIME_SERIES = SOURCE_ROOT / '01_RawData/time_series'
PROCESSED_DATA = ROOT / '03_ProcessedData'
PREPROCESS_CODE = ROOT / '02_Preprocessing'
REVISION_CODE = ROOT / '04_Analysis/revision'
RESULTS_REVISION = ROOT / '05_Results/revision'
RESULTS_SECTION3 = ROOT / '05_Results/section3'
RESULTS_SECTION4 = ROOT / '05_Results/section4'
def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'External project data/output path rejected: {path}')
    return path
def ensure_dirs():
    for p in [PROCESSED_DATA,RESULTS_REVISION,RESULTS_SECTION3,RESULTS_SECTION4]:
        inside(p).mkdir(parents=True,exist_ok=True)

_io_guarded = False
def install_data_io_guard():
    """Reject external project data paths even if a notebook later changes a variable."""
    global _io_guarded
    if _io_guarded: return
    from functools import wraps
    import pandas as pd
    import pyarrow.parquet as pq
    def checked(value, is_write=False):
        if value is None or hasattr(value,'read') or hasattr(value,'write'): return
        if isinstance(value,(str,Path)):
            if '://' in str(value): raise ValueError('Remote data source is not an independent local input')
            resolved=Path(value).resolve()
            if is_write and any(resolved.is_relative_to(p.resolve()) for p in (RAW_USAGE, RAW_TIME_SERIES)):
                raise ValueError(f'Read-only raw data path rejected: {resolved}')
            if not is_write and any(resolved.is_relative_to(p.resolve()) for p in (RAW_USAGE, RAW_TIME_SERIES)): return
            inside(value)
    def wrap(function,index=0,keywords=(),is_write=False):
        @wraps(function)
        def call(*args,**kwargs):
            if len(args)>index: checked(args[index],is_write)
            for key in keywords:
                if key in kwargs: checked(kwargs[key],is_write)
            return function(*args,**kwargs)
        return call
    pd.read_csv=wrap(pd.read_csv,0,('filepath_or_buffer',))
    pd.read_parquet=wrap(pd.read_parquet,0,('path',))
    pd.DataFrame.to_csv=wrap(pd.DataFrame.to_csv,1,('path_or_buf',),is_write=True)
    pd.DataFrame.to_parquet=wrap(pd.DataFrame.to_parquet,1,('path',),is_write=True)
    pq.ParquetFile=wrap(pq.ParquetFile,0,('source',))
    pq.read_table=wrap(pq.read_table,0,('source',))
    pq.write_table=wrap(pq.write_table,1,('where',),is_write=True)
    _io_guarded=True
