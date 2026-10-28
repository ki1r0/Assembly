"""Partial public RGB-D to public CAD registration. No simulator transforms.
When support_z is supplied, the explicit hypothesis is a part resting on that
calibrated plane, not a floating or already-held part."""
import itertools
import numpy as np
from scipy.spatial import cKDTree

def fit_transform(local,world):
    a=local.mean(0);b=world.mean(0);u,_,v=np.linalg.svd((local-a).T@(world-b));rotation=v.T@u.T
    if np.linalg.det(rotation)<0:v[-1]*=-1;rotation=v.T@u.T
    return rotation,b-rotation@a

def register(cad,observed,support_z=None,normals=None,initial_pose=None):
    cad=np.asarray(cad,float);observed=np.asarray(observed,float);full_cad=cad.copy();full_normals=None if normals is None else np.asarray(normals,float).copy()
    if len(cad)<30 or len(observed)<20:raise ValueError('Insufficient CAD/observed surface points')
    # Bounded deterministic subsampling affects fitting cost, not persisted sensor precision.
    stride=max(1,len(cad)//4500);cad=cad[::stride]
    if normals is not None:normals=np.asarray(normals,float)[::stride]
    observed=observed[::max(1,len(observed)//1000)]
    ca=cad.mean(0);ob=observed.mean(0);_,_,caxes=np.linalg.svd(cad-ca,full_matrices=False);_,_,oaxes=np.linalg.svd(observed-ob,full_matrices=False)
    fits=[];seeds=[]
    for perm in itertools.permutations(range(3)):
      for signs in itertools.product([-1.,1.],repeat=3):
        rotation=oaxes.T[:,perm]@np.diag(signs)@caxes
        if np.linalg.det(rotation)<0:continue
        seeds.append(rotation)
    from scipy.spatial.transform import Rotation
    for principal in [0,2]:
      for sign in [-1.,1.]:
        aligned,_=Rotation.align_vectors((oaxes[principal]*sign)[None],caxes[principal][None])
        for yaw in np.linspace(0,2*np.pi,12,endpoint=False):seeds.append(Rotation.from_rotvec(oaxes[principal]*yaw).as_matrix()@aligned.as_matrix())
    if initial_pose is not None:
        initial_pose=np.asarray(initial_pose,float)
        seeds=[initial_pose[:3,:3]]
    for rotation in seeds:
        rotated=cad@rotation.T
        observed_span=np.ptp(observed,axis=0);model_span=np.ptp(rotated,axis=0)
        translation=ob-rotation@ca
        if np.all(observed_span[:2]/np.maximum(model_span[:2],1e-9)>.85) and np.all(observed_span[:2]/np.maximum(model_span[:2],1e-9)<1.15):
            translation[:2]=((observed.min(0)+observed.max(0))-(rotated.min(0)+rotated.max(0)))[:2]/2
        if initial_pose is not None:translation=initial_pose[:3,3].copy()
        for _ in range(0 if initial_pose is not None and normals is not None else 35):
            predicted=cad@rotation.T+translation;dist,index=cKDTree(predicted).query(observed)
            keep=dist<min(.03,float(np.quantile(dist,.85))+1e-6)
            if keep.sum()<20:break
            new_rotation,new_translation=fit_transform(cad[index[keep]],observed[keep])
            difference=np.linalg.norm(new_translation-translation)+np.linalg.norm(new_rotation-rotation)
            rotation,translation=new_rotation,new_translation
            if support_z is not None:
                translation[2]+=float(support_z-(cad@rotation.T+translation)[:,2].min())
            if difference<1e-6:break
        predicted=cad@rotation.T+translation;dist,_=cKDTree(predicted).query(observed);keep=dist<=np.quantile(dist,.9)
        rms=float(np.sqrt(np.mean(dist[keep]**2)));below=max(0,float(support_z-predicted[:,2].min())) if support_z is not None else 0
        score=rms+below*2
        fits.append({'rotation':rotation,'translation':translation,'rms_m':rms,'score':score,'support_penetration_m':below,'observed_surface_coverage':float(np.mean(dist<.002))})
    fits.sort(key=lambda f:f['score'])
    if not fits:raise ValueError('No CAD registration hypothesis')
    initial_count=len(fits)
    cad=full_cad;normals=full_normals
    for candidate in fits:
        predicted=cad@candidate['rotation'].T+candidate['translation'];dist,_=cKDTree(predicted).query(observed);keep=dist<=np.quantile(dist,.9);candidate['rms_m']=float(np.sqrt(np.mean(dist[keep]**2)));candidate['score']=candidate['rms_m']+candidate['support_penetration_m']*2;candidate['observed_surface_coverage']=float(np.mean(dist<.002))
    fits.sort(key=lambda f:f['score'])
    unrefined=[dict(f) for f in fits[:4]]
    if normals is not None:
      from scipy.spatial.transform import Rotation
      for candidate in fits[:4]:
        rotation=candidate['rotation'].copy();translation=candidate['translation'].copy()
        for _ in range(35):
            predicted=cad@rotation.T+translation;dist,index=cKDTree(predicted).query(observed);keep=dist<.008
            if keep.sum()<20:break
            model=predicted[index[keep]];normal=normals[index[keep]]@rotation.T;observed_part=observed[keep];pivot=predicted.mean(0)
            error=np.einsum('ij,ij->i',normal,observed_part-model);A=np.c_[np.cross(model-pivot,normal),normal];weight=np.sqrt(np.minimum(1,.001/np.maximum(np.abs(error),1e-9)));step=np.linalg.lstsq(A*weight[:,None],error*weight,rcond=1e-5)[0]
            step[:3]*=min(1,.05/max(np.linalg.norm(step[:3]),1e-12));step[3:]*=min(1,.002/max(np.linalg.norm(step[3:]),1e-12));delta=Rotation.from_rotvec(step[:3]).as_matrix();rotation=delta@rotation;translation=delta@(translation-pivot)+pivot+step[3:]
            if support_z is not None:translation[2]+=float(support_z-(cad@rotation.T+translation)[:,2].min())
            if np.linalg.norm(step)<1e-7:break
        predicted=cad@rotation.T+translation;dist,index=cKDTree(predicted).query(observed);keep=dist<=np.quantile(dist,.9);normal=normals[index]@rotation.T;error=np.einsum('ij,ij->i',normal,observed-predicted[index]);rms=float(np.sqrt(np.mean(error[keep]**2)));below=max(0,float(support_z-predicted[:,2].min())) if support_z is not None else 0
        euclidean_rms=float(np.sqrt(np.mean(dist[keep]**2)))
        candidate.update(rotation=rotation,translation=translation,rms_m=euclidean_rms,point_to_plane_rms_m=rms,score=euclidean_rms+below*2,support_penetration_m=below,observed_surface_coverage=float(np.mean(dist<.002)))
      fits=sorted(fits[:4]+unrefined,key=lambda f:f['score'])
    best=fits[0];matrix=np.eye(4);matrix[:3,:3]=best['rotation'];matrix[:3,3]=best['translation']
    return {'world_from_cad':matrix.tolist(),'rms_m':best['rms_m'],'support_penetration_m':best['support_penetration_m'],'observed_surface_coverage':best['observed_surface_coverage'],'hypotheses_tested':initial_count,'refined_hypotheses':len(fits),'runner_up_score_m':fits[1]['score'] if len(fits)>1 else None,'source':'observed partial RGB-D to public CAD surface multi-start ICP; estimate, not identity or retention ground truth'}
