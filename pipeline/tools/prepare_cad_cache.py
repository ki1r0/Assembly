"""Build ignored local CAD perception cache using the installed Isaac USD libraries."""
import os,shlex,subprocess,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import settings


def prepare_shaft_compound(task, assets, run, density_kg_m3, material, allow_open_mesh=False,
                          _local_only=False):
 """Static USD/cache preparation, without Isaac app startup or GPU access."""
 try:
  import pxr
 except ImportError:
  # A venv may hide an already installed, ABI-compatible user-site USD runtime.
  import site
  user_site=Path(site.getusersitepackages())
  if (user_site/'pxr').is_dir():
   sys.path.append(str(user_site))
  try:
   import pxr
  except ImportError:
   if _local_only:raise RuntimeError('Static USD runtime unavailable')
   root=settings.ROOT
   code=('import sys;sys.path.insert(0,'+repr(str(root))+');'
         'from tools.prepare_cad_cache import prepare_shaft_compound;'
         f'prepare_shaft_compound({task!r},{str(assets)!r},{str(run)!r},'
         f'{density_kg_m3!r},{material!r},allow_open_mesh={allow_open_mesh!r},_local_only=True)')
   script='\n'.join(['usd_library=(/isaac-sim/extscache/omni.usd.libs-*)',
    'export PYTHONPATH="${usd_library[0]}"',
    'export LD_LIBRARY_PATH="${usd_library[0]}/bin:${LD_LIBRARY_PATH}"',
    'exec /isaac-sim/python.sh -c '+shlex.quote(code)])
   subprocess.run(['docker','run','--rm','--entrypoint','/bin/bash',
    '-v',f'{root}:{root}:ro','-v',f'{Path(run).resolve()}:{Path(run).resolve()}',
    '-v',f'{Path(assets).resolve()}:{Path(assets).resolve()}:ro',
    *(['-v',f'/dev/null:{root/".env"}:ro'] if (root/'.env').exists() else []),
    os.environ.get('ISAAC_DOCKER_IMAGE','nvcr.io/nvidia/isaac-lab:2.3.0'),'-c',script],check=True)
   return json.loads((Path(run)/'checks/initial-shaft-compound.json').read_text())
 from shaft_compound import prepare_initial_shaft_compound
 return prepare_initial_shaft_compound(task,assets,run,density_kg_m3,material,
                                      allow_open_mesh=allow_open_mesh,
                                      meters_per_source_unit=settings.CAD_METERS_PER_SOURCE_UNIT)

def prepare(force=False):
 root=settings.ROOT;output=root/'assets/public_cad_models'
 import numpy as np
 model_names=['bolt','cover','cover_input','cover_small','gear_output','gear_transfer','shaft_output','shaft_transfer','shaft_input','m10_bolt','m10_nut','casing_top','oil_indicator','breather_plug','bearing_32013','bearing_32205_transfer','bearing_30304','bearing_32205_input']
 if not force and all((output/(name+'.npz')).exists() for name in model_names+['hand']):
  complete=True
  for name in model_names:
   with np.load(output/(name+'.npz')) as cache:
    complete=complete and all(key in cache for key in ['normals','volume_m3','uniform_density_com_local_m','inertia_per_mass_local_m2','closed_mesh_validated','collision_geometry_selected','collision_geometry_source','geometry_scale_m_per_coordinate'])
    complete=complete and np.isclose(float(cache['geometry_scale_m_per_coordinate']),settings.CAD_METERS_PER_SOURCE_UNIT)
  if complete:return
 output.mkdir(parents=True,exist_ok=True);assets=Path(os.environ['ASSEMBLY_ASSET_ROOT']).resolve()
 script='\n'.join(['usd_library=(/isaac-sim/extscache/omni.usd.libs-*)','export PYTHONPATH="${usd_library[0]}"','export LD_LIBRARY_PATH="${usd_library[0]}/bin:${LD_LIBRARY_PATH}"','exec /isaac-sim/python.sh '+shlex.quote('/tmp/export_public_cad.py')+' --output '+shlex.quote(str(output))])
 subprocess.run(['docker','run','--rm','--entrypoint','/bin/bash','-e','ASSEMBLY_ASSET_ROOT='+str(assets),'-e','ASSEMBLY_CAD_SCALE_M_PER_UNIT='+str(settings.CAD_METERS_PER_SOURCE_UNIT),'-v',f'{assets}:{assets}:ro','-v',f'{root/"settings.py"}:/tmp/settings.py:ro','-v',f'{root/"tools/export_public_cad.py"}:/tmp/export_public_cad.py:ro','-v',f'{root/"tools/calibrate_part_mass.py"}:/tmp/calibrate_part_mass.py:ro','-v',f'{output}:{output}',os.environ.get('ISAAC_DOCKER_IMAGE','nvcr.io/nvidia/isaac-lab:2.3.0'),'-c',script],check=True)
if __name__=='__main__':prepare()
