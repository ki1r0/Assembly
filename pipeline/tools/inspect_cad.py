"""Offline public CAD bounds/volume; no simulator state or task target pose."""
import os,json,numpy as np
from pathlib import Path
from pxr import Usd,UsdGeom
from settings import CAD_METERS_PER_SOURCE_UNIT
root=Path(os.environ['ASSEMBLY_ASSET_ROOT'])
rows={}
for file in ['M6 Hub Bolt.usd','Hub Cover Output.usd','Casing Top.usd']:
 stage=Usd.Stage.Open(str(root/'parts'/file));cache=UsdGeom.XformCache(Usd.TimeCode.Default());points=[];signed=0;planes={}
 for prim in stage.Traverse():
  if not prim.IsA(UsdGeom.Mesh):continue
  mesh=UsdGeom.Mesh(prim);vertices=np.array(mesh.GetPointsAttr().Get(),float);matrix=np.array(cache.GetLocalToWorldTransform(prim)).T;vertices=(matrix[:3,:3]@vertices.T+matrix[:3,3:4]).T*CAD_METERS_PER_SOURCE_UNIT;points.extend(vertices.tolist());idx=np.array(mesh.GetFaceVertexIndicesAttr().Get());sizes=np.array(mesh.GetFaceVertexCountsAttr().Get());start=0
  for n in sizes:
   face=idx[start:start+n];start+=n
   for j in range(1,n-1):
    triangle=vertices[[face[0],face[j],face[j+1]]];cross=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);area=np.linalg.norm(cross)/2;signed+=float(triangle[0]@np.cross(triangle[1],triangle[2]))/6
    if abs(cross[1])>.999*max(np.linalg.norm(cross),1e-12):
     key=round(float(triangle[:,1].mean()),6);planes[key]=planes.get(key,0)+float(area)
 points=np.array(points);rows[file]={'bounds_local_m':[points.min(0).tolist(),points.max(0).tolist()],'extent_local_m':np.ptp(points,axis=0).tolist(),'normal_y_planar_surface_area_m2':sorted(planes.items(),key=lambda x:-x[1])[:12],'signed_mesh_volume_m3':signed,'volume_note':'Signed polygon integral, not certified watertight material volume or a mass assignment.'}
rows['_metadata']={'cad_meters_per_source_unit':CAD_METERS_PER_SOURCE_UNIT}
print(json.dumps(rows,indent=2))
