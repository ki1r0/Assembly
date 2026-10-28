"""Run a labelled privileged calibration episode, not an actor evaluation."""
import argparse,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import settings
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--gpu',default='2');a=p.parse_args();r=settings.configure(a.run_dir);root=settings.ROOT;assets=Path(os.environ['ASSEMBLY_ASSET_ROOT']);name='assembly-evaluator-'+r.name.lower()
cmd=['docker','run','--name',name,'--gpus','device='+a.gpu,'--ipc','host','--network','host']
for key,value in {'ASSEMBLY_RUN_DIR':str(r),'ASSEMBLY_ASSET_ROOT':str(assets),'ASSEMBLY_CASE':'nominal-aligned','ASSEMBLY_FAST_COMPONENTS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','ACCEPT_EULA':'Y','OMNI_KIT_ACCEPT_EULA':'YES'}.items():cmd+=['-e',key+'='+value]
for mount in [f'{root}:{root}:ro',f'{r}:{r}',f'{assets}:{assets}:ro',f'/dev/null:{root/".env"}:ro']:cmd+=['-v',mount]
import shlex
cmd+=['--entrypoint','/bin/bash','nvcr.io/nvidia/isaac-lab:2.3.0','-c','umask 000; exec /isaac-sim/python.sh '+shlex.quote(str(root/'backend.py'))+' --headless --enable_cameras --device cpu --evaluator-probe']
try:
 with (r/'private/launcher.log').open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 print(r,'exit:',result.returncode)
finally:
 subprocess.run(['docker','stop','-t','15',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);subprocess.run(['docker','rm',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
