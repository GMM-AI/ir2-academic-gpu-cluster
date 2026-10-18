from pathlib import Path
import argparse,json,sys,time
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
ap=argparse.ArgumentParser();ap.add_argument('notebook');ap.add_argument('--python',required=True);ap.add_argument('--timeout',type=int,default=600)
args=ap.parse_args();path=Path(args.notebook).resolve()
root=next(p for p in path.parents if (p/'project_paths.py').is_file())
kernels=root/'02_Preprocessing/_kernels';spec=kernels/'fgcs-corrected';spec.mkdir(parents=True,exist_ok=True)
(spec/'kernel.json').write_text(json.dumps({'argv':[args.python,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'FGCS corrected analysis','language':'python'}))
manager=KernelSpecManager(kernel_dirs=[str(kernels)])
n=nbformat.read(path,as_version=4);nbformat.validate(n)
start=time.monotonic();status='PASS';error=None
try:
    from jupyter_client import KernelManager
    km=KernelManager(kernel_name='fgcs-corrected',kernel_spec_manager=manager)
    client=NotebookClient(n,km=km,kernel_name='fgcs-corrected',timeout=args.timeout,resources={'metadata':{'path':str(root)}})
    client.execute()
except Exception as e:
    status='FAIL';error=f'{type(e).__name__}: {str(e)[:2500]}'
finally:
    if 'km' in locals() and km.has_kernel:
        km.shutdown_kernel(now=True)
    nbformat.write(n,path)
    info={'notebook':str(path.relative_to(root)),'status':status,'seconds':round(time.monotonic()-start,2),'error':error,'code_cells':sum(c.cell_type=='code' for c in n.cells),'executed_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in n.cells)}
    output=root/'05_Results/comparison/notebook_execution';output.mkdir(parents=True,exist_ok=True)
    (output/(path.stem+'.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2));print(json.dumps(info,ensure_ascii=False,indent=2))
if error:sys.exit(1)
