"""Sequentially execute every original manuscript analysis; retain failures and cell outputs."""
from pathlib import Path
import argparse,json,subprocess,sys,time,hashlib
import nbformat

ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'project_paths.py').is_file():
    raise RuntimeError('run_manuscript.py must stay in 04_Analysis/ of the repository')
parser=argparse.ArgumentParser()
parser.add_argument('--python',required=True)
parser.add_argument('--keys',nargs='+')
args=parser.parse_args()
manifest=ROOT/'04_Analysis/manuscript_runs.json'
jobs=json.loads(manifest.read_text())
for job in jobs:
    if args.keys and job['key'] not in args.keys:continue
    path=ROOT/job['notebook'];out=ROOT/job['output'];out.mkdir(parents=True,exist_ok=True)
    job.update(status='RUNNING',started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    manifest.write_text(json.dumps(jobs,ensure_ascii=False,indent=2))
    print('RUN',job['key'],str(path),flush=True)
    start=time.monotonic()
    with (out/'runner.log').open('w') as stream:
        result=subprocess.run([sys.executable,str(ROOT/'02_Preprocessing/run_notebook.py'),str(path),
           '--python',args.python,'--timeout','1800'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    n=nbformat.read(path,as_version=4)
    original=[c for c in n.cells if c.cell_type=='code' and 'original_cell_index' in c.metadata]
    # Empty/comment-only original cells do not receive execution_count from nbclient.
    unexecuted=[c.metadata.original_cell_index for c in original if c.execution_count is None
                and any(l.strip() and not l.lstrip().startswith('#') for l in c.source.splitlines())]
    job.update(status='PASS' if result.returncode==0 and not unexecuted else 'FAIL',
       seconds=round(time.monotonic()-start,2),original_code_cells=len(original),
       original_code_cells_unexecuted=unexecuted,
       execution_errors=sum(o.get('output_type')=='error' for c in n.cells for o in c.get('outputs',[])),
       notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
       runner_log=str((out/'runner.log').relative_to(ROOT)))
    manifest.write_text(json.dumps(jobs,ensure_ascii=False,indent=2))
    print(job['status'],job['key'],job['seconds'],flush=True)
    if job['status']=='FAIL':
        print((out/'runner.log').read_text()[-4500:],flush=True)
        sys.exit(1)
print('ALL_REQUESTED_RUNS_PASS',flush=True)
if all(j['status']=='PASS' for j in jobs):
    subprocess.run([args.python,str(ROOT/'04_Analysis/export_manuscript.py')],cwd=ROOT,check=True)
