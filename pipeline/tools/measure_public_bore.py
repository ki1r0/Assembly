"""Fit an axial circular surface in a standalone public CAD sample cache."""
import json,sys
import numpy as np
from scipy.optimize import least_squares


def measure_bore(points,normals,axis=1,radius_bounds=(.04,.11)):
    coordinates=[i for i in range(3) if i!=axis]
    xy=np.asarray(points)[abs(np.asarray(normals)[:,axis])<.15][:,coordinates]
    rng=np.random.default_rng(12);best=None
    for _ in range(1500):
        sample=xy[rng.choice(len(xy),3,replace=False)]
        try:centre=np.linalg.solve(2*(sample[1:]-sample[0]),(sample[1:]**2).sum(1)-(sample[0]**2).sum())
        except np.linalg.LinAlgError:continue
        radius=np.linalg.norm(sample[0]-centre)
        if not radius_bounds[0]<radius<radius_bounds[1]:continue
        inliers=abs(np.linalg.norm(xy-centre,axis=1)-radius)<.0002
        if best is None or inliers.sum()>best[0]:best=(int(inliers.sum()),centre,radius,inliers)
    if best is None or best[0]<100:raise ValueError('No supported circular CAD surface in the requested radius range')
    _,centre,radius,inliers=best
    fit=least_squares(lambda v:np.linalg.norm(xy[inliers]-v[:2],axis=1)-v[2],[*centre,radius])
    return {'coordinate_indices':coordinates,'centre_local_m':fit.x[:2].tolist(),'radius_m':float(fit.x[2]),'rms_m':float(np.std(fit.fun)),'inlier_count':best[0],'source':'Standalone public CAD surface points/normals; no scene or evaluator inputs'}

if __name__=='__main__':
    cache=np.load(sys.argv[1]);print(json.dumps(measure_bore(cache['points'],cache['normals']),indent=2))
