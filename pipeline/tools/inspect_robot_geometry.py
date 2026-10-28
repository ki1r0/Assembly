import os,json,numpy as np
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics
path=Path(os.environ['ASSEMBLY_ASSET_ROOT'])/'vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd'
stage=Usd.Stage.Open(str(path));cache=UsdGeom.XformCache(Usd.TimeCode.Default());hand=next(p for p in stage.Traverse() if p.GetName()=='panda_hand');inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(hand)).T);rows=[]
for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
 if 'finger' not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh):continue
 pts=np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float);world=np.array(cache.GetLocalToWorldTransform(prim)).T;pts=((inv@world)[:3,:3]@pts.T+(inv@world)[:3,3:4]).T
 inner=pts[(np.abs(pts[:,1])<.0005)&(pts[:,2]>.09)]
 rows.append({'inner_pad_z_m': [float(inner[:,2].min()),float(inner[:,2].max()),float(np.median(inner[:,2]))] if len(inner) else None,'prim':str(prim.GetPath()),'collision':prim.HasAPI(UsdPhysics.CollisionAPI),'hand_frame_bounds_m':[pts.min(0).tolist(),pts.max(0).tolist()]})
print(json.dumps(rows,indent=2))
