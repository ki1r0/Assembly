"""Estimate the held shaft from public RGB-D, not simulator object state."""
import numpy as np
from scipy.optimize import least_squares

def measure(sensors):
    tcp=np.asarray(sensors['tcp_pose'],float);d=np.load(sensors['camera_depth_path'],allow_pickle=False);k=np.asarray(sensors['camera_intrinsics']);t=np.asarray(sensors['camera_extrinsics'])
    camera=np.linalg.solve(t,np.r_[tcp[:3]-[0,0,.04],1])[:3];uv=k@camera;uv=uv[:2]/uv[2];x,y=map(round,uv)
    yy,xx=np.mgrid[max(0,y-65):min(d.shape[0],y+66),max(0,x-65):min(d.shape[1],x+66)];z=d[yy,xx];valid=np.isfinite(z)&(z>0)
    pc=(t[:3,:3]@(np.linalg.solve(k,np.stack([xx[valid],yy[valid],np.ones(valid.sum())]))*z[valid])+t[:3,3:4]).T
    # Isolate the exposed lower shaft below the fingers; known robot TCP is the anchor.
    pc=pc[(pc[:,2]>tcp[2]-.065)&(pc[:,2]<tcp[2]-.023)&(np.linalg.norm(pc[:,:2]-tcp[:2],axis=1)<.022)]
    if len(pc)<30 or np.ptp(pc[:,2])<.012:raise ValueError('Insufficient exposed shaft depth for an in-hand axis measurement')
    cz=float(np.median(pc[:,2]));radius=.011767996864/2
    def residual(v):
        axis=np.r_[v[2:],1.];axis/=np.linalg.norm(axis);delta=pc-np.r_[v[:2],cz]
        return np.linalg.norm(delta-(delta@axis)[:,None]*axis,axis=1)-radius
    bounds=(np.r_[tcp[:2]-.008,-.3,-.3],np.r_[tcp[:2]+.008,.3,.3])
    fit=least_squares(residual,np.r_[tcp[:2],0,0],bounds=bounds,loss='soft_l1',f_scale=.00007)
    inside=np.abs(fit.fun)<.00015
    if inside.sum()<20:raise ValueError('Visible shaft does not support the nominal cylinder')
    refined=least_squares(lambda v:residual(v)[inside],fit.x,bounds=bounds,loss='soft_l1',f_scale=.00005)
    axis=np.r_[refined.x[2:],1.];axis/=np.linalg.norm(axis);anchor=np.r_[refined.x[:2],cz]
    tip=anchor+axis*float(np.quantile((pc[inside]-anchor)@axis,.01))
    rms=float(np.sqrt(np.mean(refined.fun**2)))
    if rms>.00025:raise ValueError('In-hand cylinder residual too large for this estimate')
    return {'source':'current public RGB-D below measured fingers, fixed public shaft radius','source_depth':sensors['camera_depth_path'],'source_tcp_pose':tcp.tolist(),'point_count':len(pc),'inlier_count':int(inside.sum()),'axis_world_up':axis.tolist(),'tilt_deg':float(np.degrees(np.arccos(axis[2]))),'radial_fit_rms_m':rms,'line_anchor_world_m':anchor.tolist(),'estimated_visible_tip_world_m':tip.tolist(),'tip_uncertainty_note':'Visible shaft endpoint estimate; no hidden part pose. End occlusion/pixel sampling can bias depth.'}
