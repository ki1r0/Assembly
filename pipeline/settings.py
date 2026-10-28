"""Existing dependencies and explicit runtime selection; no credentials stored here."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RPENT=ROOT/'third_party/rpent'
ISAAC=Path(__file__).resolve().parent/'isaac_runtime'
PYTHON=Path(sys.executable)
# These CAD coordinates are millimetres despite their inconsistent USD unit metadata.
CAD_METERS_PER_SOURCE_UNIT=0.001
sys.path[:0]=[str(ROOT),str(RPENT),str(ISAAC)]
os.environ['RPENT_REPO_ROOT']=str(RPENT)
os.environ['M1_AUTONOMOUS']='1'

def configure(run):
    run=Path(run).resolve()
    os.environ.update(ASSEMBLY_RUN_DIR=str(run),M1_CONTROL_PATH=str(run/'private/control.json'),M1_RECORD_DIR=str(run/'episode/rpent-activity'))
    for name in ['private','checks','ipc','episode','episode/rpent-activity','episode/rpent/actor_outputs']:(run/name).mkdir(parents=True,exist_ok=True)
    return run
