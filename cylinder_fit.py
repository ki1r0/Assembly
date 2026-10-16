"""Robust vertical-cylinder center from observed points and a public nominal radius."""
import numpy as np
from scipy.optimize import least_squares

def fit_xy(points,radius,initial_center,max_center_shift=.004):
    xy=np.asarray(points,float)[:,:2];initial_center=np.asarray(initial_center,float)
    if len(xy)<8:return None
    rng=np.random.default_rng(0);best=None
    for _ in range(1000):
        a,b=xy[rng.choice(len(xy),2,replace=False)];delta=b-a;length=np.linalg.norm(delta)
        if not 1e-5<length<2*radius:continue
        mid=(a+b)/2;normal=np.array([-delta[1],delta[0]])/length;offset=np.sqrt(radius**2-length**2/4)
        for center in (mid+normal*offset,mid-normal*offset):
            if np.linalg.norm(center-initial_center)>max_center_shift:continue
            inside=np.abs(np.linalg.norm(xy-center,axis=1)-radius)<.00015
            if best is None or inside.sum()>best[0]:best=(inside.sum(),center,inside)
    if best is None or best[0]<max(8,.35*len(xy)):return None
    count,center,inside=best
    fit=least_squares(lambda c:np.linalg.norm(xy[inside]-c,axis=1)-radius,center,loss='soft_l1',f_scale=.00005)
    angles=np.sort(np.mod(np.arctan2(*(xy[inside]-fit.x).T[::-1]),2*np.pi));coverage=2*np.pi-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))
    if coverage<np.pi/3:return None
    return {'center_xy_m':fit.x.tolist(),'nominal_radius_m':float(radius),'point_count':len(xy),'inlier_count':int(count),'inlier_fraction':float(count/len(xy)),'radial_rms_m':float(np.sqrt(np.mean(fit.fun**2))),'angular_coverage_rad':float(coverage),'method':'RANSAC visible vertical-cylinder wall, fixed public nominal radius; no object transform'}

def fit_observed_xy(points,initial_center,observed_radius):
    """Fit both bore radius and center; a partial larger rim must not inherit shaft radius."""
    xy=np.asarray(points,float)[:,:2];initial=np.asarray(initial_center,float)
    if len(xy)<12 or observed_radius is None or not np.isfinite(observed_radius):return None
    rng=np.random.default_rng(0);best=None
    for _ in range(500):
        sample=xy[rng.choice(len(xy),3,replace=False)]
        a=2*(sample[1:]-sample[0]);b=np.sum(sample[1:]**2,axis=1)-np.sum(sample[0]**2)
        if abs(np.linalg.det(a))<1e-9:continue
        center=np.linalg.solve(a,b);radius=float(np.linalg.norm(sample[0]-center))
        if abs(radius-observed_radius)>.0008 or np.linalg.norm(center-initial)>.003:continue
        residual=np.abs(np.linalg.norm(xy-center,axis=1)-radius);inside=residual<.00008
        score=(int(inside.sum()),-float(np.mean(residual[inside])) if inside.any() else -1.)
        if best is None or score>best[0]:best=(score,center,radius,inside)
    if best is None or best[0][0]<max(12,.35*len(xy)):return None
    _,center,radius,inside=best
    fit=least_squares(lambda p:np.linalg.norm(xy[inside]-p[:2],axis=1)-p[2],np.r_[center,radius],loss='soft_l1',f_scale=.00003)
    angles=np.sort(np.mod(np.arctan2(*(xy[inside]-fit.x[:2]).T[::-1]),2*np.pi));coverage=2*np.pi-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))
    if coverage<np.pi/3 or abs(fit.x[2]-observed_radius)>.0008 or np.linalg.norm(fit.x[:2]-initial)>.003:return None
    return {'center_xy_m':fit.x[:2].tolist(),'measured_radius_m':float(fit.x[2]),'point_count':len(xy),'inlier_count':int(inside.sum()),'radial_rms_m':float(np.sqrt(np.mean(fit.fun**2))),'angular_coverage_rad':float(coverage),'method':'RANSAC visible wall, center and radius both estimated from current RGB-D; no nominal radius'}
