"""Project-local entry point; run with ./run.sh from any working directory."""
from pathlib import Path
import argparse, os, subprocess, sys
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ['PYTHONPATH']=str(ROOT)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'.cache/jupyter'))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('command',choices=['run-notebook','online','sensitivity','static','grace','manuscript'])
parser.add_argument('args',nargs=argparse.REMAINDER)
a=parser.parse_args()
scripts={'online':'04_Analysis/revision/online_ir2/simulation.py','sensitivity':'04_Analysis/revision/threshold_sensitivity/sensitivity.py','static':'04_Analysis/revision/policy_comparison.py','grace':'04_Analysis/revision/grace_return_analysis.py','manuscript':'04_Analysis/run_manuscript.py'}
if a.command=='run-notebook':
    if not a.args:parser.error('run-notebook requires a project-relative notebook path')
    path=(ROOT/a.args[0]).resolve()
    if not path.is_relative_to(ROOT):parser.error('Notebook must be inside this project')
    cmd=[sys.executable,str(ROOT/'02_Preprocessing/run_notebook.py'),str(path),'--python',sys.executable,*a.args[1:]]
else:
    cmd=[sys.executable,str(ROOT/scripts[a.command]),*a.args]
    if a.command=='manuscript':cmd.extend(['--python',sys.executable])
raise SystemExit(subprocess.run(cmd,cwd=ROOT,env=os.environ.copy()).returncode)
