"""Cache folder of the online replay and threshold outputs (05_Results/revision/_cache)."""
from project_paths import ROOT, REVISION_CODE, RESULTS_REVISION
PROJECT_ROOT = REVISION_CODE
CACHE_ROOT = RESULTS_REVISION / '_cache'
def require_frozen_cache(group, filenames):
    if group not in {'online_ir2','threshold_sensitivity'}:
        raise ValueError(group)
    folder=CACHE_ROOT/group
    missing=[name for name in filenames if not (folder/name).is_file()]
    if missing:
        raise FileNotFoundError('Corrected cache not generated; do not use original cache: '+str(missing))
    return folder
