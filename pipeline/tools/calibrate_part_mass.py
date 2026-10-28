"""Public closed STL volume and declared material mass, not simulator state."""
import struct
from pathlib import Path
import numpy as np
from settings import CAD_METERS_PER_SOURCE_UNIT

DENSITY_SOURCE='https://inldigitallibrary.inl.gov/sites/sti/sti/5922043.pdf'

def closed_stl_volume(path,scale):
    raw=Path(path).read_bytes();count=struct.unpack_from('<I',raw,80)[0]
    if len(raw)!=84+50*count:raise ValueError('Expected binary STL with complete triangle records')
    records=np.frombuffer(raw,offset=84,count=count,dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')]))
    vertices,indices=np.unique(records['vertices'].reshape(-1,3),axis=0,return_inverse=True);triangles=indices.reshape(-1,3)
    triangle_keys=np.sort(triangles,axis=1)
    unique_keys,inverse,counts=np.unique(triangle_keys,axis=0,return_inverse=True,return_counts=True)
    permutation=np.argsort(triangles,axis=1)
    inversions=(permutation[:,0]>permutation[:,1]).astype(np.int8)+(permutation[:,0]>permutation[:,2]).astype(np.int8)+(permutation[:,1]>permutation[:,2]).astype(np.int8)
    orientation=np.where(inversions%2,-1,1)
    orientation_sum=np.bincount(inverse,weights=orientation,minlength=len(unique_keys))
    same_winding=np.abs(orientation_sum)==counts
    opposite_pair=(counts==2)&(orientation_sum==0)
    if np.any(~((counts==1)|same_winding|opposite_pair)):
        raise ValueError('STL has duplicate triangle groups that cannot be safely cancelled')
    first=np.unique(inverse,return_index=True)[1]
    retained_groups=(counts==1)|same_winding
    keep=np.zeros(len(triangles),dtype=bool);keep[first[retained_groups]]=True
    cancelled_opposite_count=int(opposite_pair.sum()*2)
    collapsed_same_winding_count=int(np.sum(counts[same_winding]-1))
    triangles=triangles[keep]
    directed=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]);_,counts=np.unique(np.sort(directed,axis=1),axis=0,return_counts=True)
    if np.any(counts!=2):raise ValueError('STL volume requires a closed two-manifold mesh')
    keys,occurrences=np.unique(directed,axis=0,return_counts=True)
    if np.any(occurrences!=1):raise ValueError('STL triangle orientation is not consistently opposed across shared edges')
    points=vertices.astype(float)*float(scale);tri_points=points[triangles];tetra=np.einsum('ij,ij->i',tri_points[:,0],np.cross(tri_points[:,1],tri_points[:,2]))/6;signed=float(tetra.sum())
    if abs(signed)<1e-12:raise ValueError('STL volume is zero')
    centroid=(tetra[:,None]*tri_points.sum(axis=1)/4).sum(axis=0)/signed
    return {'volume_m3':abs(signed),'uniform_density_com_stl_frame_m':centroid.tolist(),'triangle_count':count,'effective_triangle_count':len(triangles),'cancelled_opposite_triangle_count':cancelled_opposite_count,'collapsed_same_winding_duplicate_count':collapsed_same_winding_count,'closed_mesh':True,'consistently_oriented':True,'geometry_scale_m_per_coordinate':float(scale)}

def cover_mass_profile(asset_root,material,mesh_name='Hub Cover Output.stl',geometry_cache=None):
    mesh=Path(asset_root)/'parts'/mesh_name
    if geometry_cache is None:
        if mesh_name!='Hub Cover Output.stl':
            raise ValueError('Input and small covers require a hierarchy-transformed USD geometry cache')
        geometry=closed_stl_volume(mesh,CAD_METERS_PER_SOURCE_UNIT)
    else:
        with np.load(geometry_cache) as cache:
            if 'geometry_scale_m_per_coordinate' not in cache or not np.isclose(float(cache['geometry_scale_m_per_coordinate']),CAD_METERS_PER_SOURCE_UNIT):
                raise ValueError(f'Public CAD cache was not generated at the current CAD scale ({CAD_METERS_PER_SOURCE_UNIT} m/source unit): {geometry_cache}')
            volume=float(cache['volume_m3']);com=np.asarray(cache['uniform_density_com_local_m'],dtype=float)
            closed_mesh_validated=bool(cache['closed_mesh_validated']) if 'closed_mesh_validated' in cache else False
        if not np.isfinite(volume) or volume<=0 or com.shape!=(3,) or not np.all(np.isfinite(com)):
            raise ValueError(f'Invalid transformed public CAD geometry cache: {geometry_cache}')
        geometry={'volume_m3':volume,'uniform_density_com_local_m':com.tolist(),'volume_source':'USD hierarchy-transformed public CAD triangle cache','closed_mesh_validated':closed_mesh_validated}
    if material=='legacy-steel':
        density=7800.;mass=geometry['volume_m3']*density
        status='Steel-density assumption applied to this cover CAD; not a measured material'
    elif material=='aluminum-6061-assumption':
        density=2700.;mass=geometry['volume_m3']*density;status='Explicit aluminum-6061 simulation material assumption; original CAD alloy is unverified'
    elif material=='polypropylene-assumption':
        density=900.;mass=geometry['volume_m3']*density
        status='User-confirmed plastic cover; polypropylene density assumed because exact polymer grade is unspecified' if mesh_name=='Hub Cover Output.stl' else 'Polypropylene density assumption; exact polymer grade and cover-variant material are unverified'
    else:raise ValueError('Unknown declared cover material')
    return {'material':material,'mass_kg':mass,'density_kg_m3':density,'volume_times_density_kg':geometry['volume_m3']*density,'material_status':status,'material_source':'https://www.kpic.co.kr/hp/en/product/polymer/pol_grade.asp?grade=5014L+HPT&pm_cd=1411' if material=='polypropylene-assumption' else DENSITY_SOURCE if material!='legacy-steel' else 'Original hrc_m1/calibration_manifest.json steel-density estimate','cad_path':str(mesh),**geometry,'material_scope_assumption':'The selected cover material model is applied to this cover CAD; the input/small-cover material grade has not been independently confirmed','authored_usd_mass_warning':'All inspected USD parts carry the same 0.2kg placeholder; not used as a measured cover mass','geometry_friction_force_limits_changed':False}


def cad_mass_profile(geometry_cache,density_kg_m3,material,allow_open_mesh=False):
    """Compute a declared-density mass; optionally mark an open mesh estimate provisional."""
    if not np.isfinite(density_kg_m3) or density_kg_m3<=0:
        raise ValueError('Material density must be finite and positive')
    with np.load(geometry_cache) as cache:
        closed=bool(cache['closed_mesh_validated']) if 'closed_mesh_validated' in cache else False
        if not closed and not allow_open_mesh:
            raise ValueError(f'Public CAD geometry cache lacks a closed-mesh validation: {geometry_cache}')
        if 'collision_geometry_selected' not in cache or not bool(cache['collision_geometry_selected']):
            raise ValueError(f'Public CAD cache does not identify geometry used by the runtime collision backend: {geometry_cache}')
        if 'geometry_scale_m_per_coordinate' not in cache or not np.isclose(float(cache['geometry_scale_m_per_coordinate']),CAD_METERS_PER_SOURCE_UNIT):
            raise ValueError(f'Public CAD cache was not generated at the current CAD scale ({CAD_METERS_PER_SOURCE_UNIT} m/source unit): {geometry_cache}')
        volume=float(cache['volume_m3']);com=np.asarray(cache['uniform_density_com_local_m'],dtype=float)
        inertia=np.asarray(cache['inertia_per_mass_local_m2'],dtype=float)
    if not np.isfinite(volume) or volume<=0 or com.shape!=(3,) or inertia.shape!=(3,3) or not np.all(np.isfinite(com)) or not np.all(np.isfinite(inertia)):
        raise ValueError(f'Invalid transformed public CAD geometry cache: {geometry_cache}')
    return {'material':material,'mass_kg':volume*float(density_kg_m3),'density_kg_m3':float(density_kg_m3),'volume_m3':volume,'uniform_density_com_local_m':com.tolist(),'inertia_per_mass_local_m2':inertia.tolist(),'closed_mesh_validated':closed,'collision_geometry_selected':True,'mass_estimate_status':'density_assumption_from_closed_collision_mesh' if closed else 'provisional_density_assumption_from_nonwatertight_collision_mesh','material_status':'Declared density assumption; exact part material and mass are unverified'+('; mesh is not watertight, so volume-derived mass/COM/inertia are provisional' if not closed else ''),'material_source':DENSITY_SOURCE if material=='steel-density-assumption' else 'user-declared density assumption','cad_geometry_cache':str(geometry_cache),'mass_source':'transformed authored public collision-mesh signed volume times declared density; USD placeholder mass is ignored'}


def closed_mesh_validated(triangles):
    """Check a triangle soup for two-manifold closure and opposite edge winding."""
    p=np.asarray(triangles,float)
    if p.ndim!=3 or p.shape[1:]!=(3,3) or not np.isfinite(p).all():return False
    _,indices=np.unique(p.reshape(-1,3),axis=0,return_inverse=True);faces=indices.reshape(-1,3)
    directed=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);_,edge_counts=np.unique(np.sort(directed,axis=1),axis=0,return_counts=True)
    _,directed_counts=np.unique(directed,axis=0,return_counts=True)
    return bool(np.all(edge_counts==2) and np.all(directed_counts==1))


def uniform_solid_properties(triangles):
    """Volume integrals in the supplied public mesh frame, uniform density."""
    p=np.asarray(triangles,float)
    if p.ndim!=3 or p.shape[1:]!=(3,3) or not np.isfinite(p).all():raise ValueError('Expected finite triangle soup with shape [N,3,3]')
    mesh_closed=closed_mesh_validated(p)
    tetra=np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2]))/6
    signed=float(tetra.sum())
    if abs(signed)<1e-12:raise ValueError('Public solid has zero signed volume')
    sums=p.sum(axis=1);com=(tetra[:,None]*sums/4).sum(axis=0)/signed
    moments=np.einsum('ni,nj->nij',sums,sums)+np.einsum('nvi,nvj->nij',p,p)
    second=(tetra[:,None,None]*moments/20).sum(axis=0)/signed
    covariance=second-np.outer(com,com)
    inertia=np.trace(covariance)*np.eye(3)-covariance
    return {'volume_m3':abs(signed),'uniform_density_com_local_m':com,'inertia_per_mass_local_m2':inertia,'closed_mesh_validated':mesh_closed}
