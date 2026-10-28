"""Sample external public CAD into a local perception cache; not simulator state."""
import os,argparse,numpy as np
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics
from calibrate_part_mass import closed_mesh_validated,uniform_solid_properties
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(18)
geometry_scale=float(os.environ['ASSEMBLY_CAD_SCALE_M_PER_UNIT'])
for name,file in [
 ('bolt','M6 Hub Bolt.usd'),
 ('cover','Hub Cover Output.usd'),
 ('cover_input','Hub Cover Input.usd'),
 ('cover_small','Hub Cover Small.usd'),
 ('gear_output','Output Gear.usd'),
 ('gear_transfer','Transfer Gear.usd'),
 ('shaft_output','Output Shaft.usd'),
 ('shaft_transfer','Transfer Shaft.usd'),
 ('shaft_input','Input Shaft.usd'),
 ('m10_bolt','M10 Casing Bolt.usd'),
 ('m10_nut','M10 Casing Nut.usd'),
 ('casing_top','Casing Top.usd'),
 ('oil_indicator','Oil Level Indicator.usd'),
 ('breather_plug','Breather Plug.usd'),
 ('bearing_32013','../CAD_edit/Bearing 32013 Inner Assy.usd'),
 ('bearing_32205_transfer','../CAD_edit/Bearing 32205 Inner Assy.usd'),
 ('bearing_30304','../CAD_edit/Bearing 30304 Inner Assy.usd'),
 ('bearing_32205_input','../CAD_edit/Bearing 32205 Inner Assy.usd'),
]:
 source=Path(os.environ['ASSEMBLY_ASSET_ROOT'])/'parts'/file
 stage=Usd.Stage.Open(str(source.resolve()));cache=UsdGeom.XformCache(Usd.TimeCode.Default());mesh_data=[];all_vertices=[]
 for prim in stage.Traverse():
  if not prim.IsA(UsdGeom.Mesh):continue
  mesh=UsdGeom.Mesh(prim);v=np.array(mesh.GetPointsAttr().Get(),float);T=np.array(cache.GetLocalToWorldTransform(prim)).T;v=(T[:3,:3]@v.T+T[:3,3:4]).T*geometry_scale;all_vertices.append(v);indices=np.array(mesh.GetFaceVertexIndicesAttr().Get());sizes=mesh.GetFaceVertexCountsAttr().Get();start=0;mesh_triangles=[]
  for n in sizes:
   f=indices[start:start+n];start+=n
   for j in range(1,n-1):
    triangle=v[[f[0],f[j],f[j+1]]];mesh_triangles.append(triangle)
  mesh_data.append((prim,np.asarray(mesh_triangles)))
 collision_meshes=[triangles for prim,triangles in mesh_data if prim.HasAPI(UsdPhysics.CollisionAPI)]
 selected_meshes=collision_meshes if collision_meshes else [triangles for _,triangles in mesh_data]
 triangles=np.concatenate(selected_meshes)
 if name.startswith(('gear_','shaft_','m10_','casing_','oil_','breather_','bearing_')):print(name,'collision source=',('authored' if collision_meshes else 'visual-fallback-used-by-backend'),'closed_mesh_validated=',closed_mesh_validated(triangles),'mesh_paths=',[str(prim.GetPath()) for prim,_ in mesh_data])
 triangles=np.asarray(triangles);areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2;choice=rng.choice(len(triangles),16000,p=areas/areas.sum());chosen=triangles[choice];normal=np.cross(chosen[:,1]-chosen[:,0],chosen[:,2]-chosen[:,0]);normal/=np.linalg.norm(normal,axis=1,keepdims=True);u=rng.random((16000,1));v=rng.random((16000,1));u=np.sqrt(u);points=(1-u)*chosen[:,0]+u*(1-v)*chosen[:,1]+u*v*chosen[:,2]
 properties=uniform_solid_properties(triangles)
 all_vertices=np.concatenate(all_vertices);bounds_extent=np.ptp(all_vertices,axis=0)
 np.savez_compressed(out/(name+'.npz'),points=points.astype(np.float32),normals=normal.astype(np.float32),bounds_extent_m=bounds_extent.astype(np.float32),geometry_scale_m_per_coordinate=np.asarray(geometry_scale),collision_geometry_selected=np.asarray(bool(selected_meshes)),collision_geometry_source=np.asarray('authored' if collision_meshes else 'visual_fallback_used_by_backend'),**properties);print(name,'public USD-local properties', {k:np.asarray(v).tolist() for k,v in properties.items()},'bounds_extent_m=',bounds_extent.tolist(),'geometry_scale=',geometry_scale)

# Public Panda collision meshes, expressed at zero jaw opening in hand frame.
from scipy.spatial import ConvexHull
from pxr import UsdPhysics
robot_file=Path(os.environ['ASSEMBLY_ASSET_ROOT'])/'vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd'
stage=Usd.Stage.Open(str(robot_file));cache=UsdGeom.XformCache(Usd.TimeCode.Default());hand=next(p for p in stage.Traverse() if p.GetName()=='panda_hand');inverse=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(hand)).T);data={}
for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
 if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
 side='left' if '/panda_leftfinger/' in str(prim.GetPath()) else 'right' if '/panda_rightfinger/' in str(prim.GetPath()) else None
 if side is None:continue
 transform=inverse@np.array(cache.GetLocalToWorldTransform(prim)).T;points=np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)@transform[:3,:3].T+transform[:3,3]
 hull=ConvexHull(points);data[side+'_equations']=hull.equations;data[side+'_bounds']=np.array([points.min(0),points.max(0)])
if not all(side+'_equations' in data for side in ['left','right']):raise ValueError('Public Panda finger collision meshes unavailable')
np.savez_compressed(out/'hand.npz',**data);print('hand collision hulls',[(k,len(v)) for k,v in data.items()])
