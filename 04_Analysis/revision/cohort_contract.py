"""Cohort checks shared by the revision notebooks and scripts."""
from pathlib import Path
import json
from project_paths import ROOT
def current_contract():
    path=ROOT/'inputs/cohort_contract.json'
    c=json.loads(path.read_text())
    p=json.loads((ROOT/'inputs/preprocessing_validation.json').read_text())
    assert p['status']=='PASS' and c['row_groups']==p['files_post'] and c['rows']==p['rows_post']
    return c
