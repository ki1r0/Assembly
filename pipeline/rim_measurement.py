"""Circular mouth measurement from RGB edges and the observed surface outside them."""
import cv2
import numpy as np


def seed_matches_ellipse(seed,centre,axes,rotation):
    """Expanding a measurement ROI must not silently select a different opening."""
    half=np.asarray(axes,float)/2+2.
    local=(np.asarray(seed,float)-np.asarray(centre,float))@rotation
    return bool(np.linalg.norm(local/half)<=1.)


def ellipse_residual_supported(residual,axes):
    # A half-pixel edge error is significant for a five-pixel opening.
    radius=min(axes)/2
    return bool(residual<=max(.08,.5/radius))


def circle_boundary_supported(offsets):
    """Low residual on a short doubled-back Canny arc cannot locate a mouth."""
    angles=np.sort(np.arctan2(offsets[:,1],offsets[:,0]))
    if len(angles)<16:return False
    return bool(np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))<=np.pi/2)


def fit_rim_circle(xy):
    coeff=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],(xy*xy).sum(1),rcond=None)[0]
    centre=coeff[:2];radii=np.linalg.norm(xy-centre,axis=1)
    all_rms=float(radii.std());inliers=np.ones(len(xy),bool)
    if all_rms>.001:
        from scipy.optimize import least_squares
        fit=least_squares(lambda p:np.linalg.norm(xy-p[:2],axis=1)-p[2],
            np.r_[centre,np.median(radii)],loss='soft_l1',f_scale=.0005)
        if not fit.success:return None
        centre=fit.x[:2];radii=np.linalg.norm(xy-centre,axis=1)
        inliers=abs(radii-fit.x[2])<.001
        if inliers.mean()<.8:return None
    rms=float(radii[inliers].std())
    if rms>.001 or not circle_boundary_supported(xy[inliers]-centre):return None
    return centre,float(np.median(radii[inliers])),rms,float(inliers.mean()),all_rms


def observed_rim(rgb,depth,K,T,pixel,radius):
    x,y=map(float,pixel);h,w=depth.shape;all_results=[]
    for r in sorted(set(min(180,max(6,int(radius)*scale)) for scale in [1,2,4,8])):
        x0,y0=max(0,round(x)-r),max(0,round(y)-r);x1,y1=min(w,round(x)+r+1),min(h,round(y)+r+1)
        gray=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
        contours,_=cv2.findContours(cv2.Canny(gray,30,90),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE);results=[]
        for contour in contours:
            if len(contour)<16:continue
            (cx,cy),(a,b),angle=cv2.fitEllipse(contour);centre=np.array([cx+x0,cy+y0])
            if np.linalg.norm(centre-[x,y])>max(5,r*.35) or min(a,b)<3 or max(a,b)>1.9*r or max(a,b)/min(a,b)>4:continue
            uv=contour.reshape(-1,2)+[x0,y0];angle=np.deg2rad(angle);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            if not seed_matches_ellipse([x,y],centre,[a,b],rotation):continue
            ellipse_residual=np.std(np.linalg.norm(((uv-centre)@rotation)/[a/2,b/2],axis=1))
            if not ellipse_residual_supported(ellipse_residual,[a,b]):continue
            # A shaft's outer rim has table outside it; a hole has material
            # outside and a cavity inside. Sample both sides of the SAME edge.
            outside=np.concatenate([np.round(centre+(uv-centre)*scale).astype(int) for scale in [.9,1.,1.1]])
            outside=outside[(outside[:,0]>=0)&(outside[:,0]<w)&(outside[:,1]>=0)&(outside[:,1]<h)]
            z=depth[outside[:,1],outside[:,0]];valid=np.isfinite(z)&(z>0);outside,z=outside[valid],z[valid]
            if len(z)<16:continue
            rays=np.linalg.solve(K,np.c_[outside,np.ones(len(outside))].T)*z
            pts=(T[:3,:3]@rays+T[:3,3:4]).T
            # Assembly mouths are approached from the calibrated table-up side.
            # Use a local surface around the mouth, never the depth at its centre.
            levels,counts=np.unique(np.round(pts[:,2]/.001).astype(int),return_counts=True)
            supported=levels[counts>=max(8,round(len(uv)*.08))]
            if not len(supported):continue
            level=float(supported.max())*.001
            # Fit the broad face at the rim's measured level, not only edge pixels
            # where chamfers and depth discontinuities bias the normal.
            stride=2 if r>40 else 1
            yy,xx=np.mgrid[y0:y1:stride,x0:x1:stride];sample_uv=np.c_[xx.ravel(),yy.ravel()];sample_z=depth[yy,xx].ravel()
            valid=np.isfinite(sample_z)&(sample_z>0);sample_uv,sample_z=sample_uv[valid],sample_z[valid]
            rays=np.linalg.solve(K,np.c_[sample_uv,np.ones(len(sample_uv))].T)*sample_z
            surface=(T[:3,:3]@rays+T[:3,3:4]).T;plane=surface[abs(surface[:,2]-level)<.0015]
            if len(plane)<12:continue
            rng=np.random.default_rng(0);best=np.ones(len(plane),bool);count=0
            for _ in range(24):
                sample=plane[rng.choice(len(plane),3,replace=False)];n=np.cross(sample[1]-sample[0],sample[2]-sample[0]);length=np.linalg.norm(n)
                if length<1e-10:continue
                mask=abs((plane-sample[0])@(n/length))<.0002
                if mask.sum()>count:best,count=mask,int(mask.sum())
            if count>=12:plane=plane[best]
            origin=np.median(plane,axis=0);_,_,vh=np.linalg.svd(plane-origin,full_matrices=False);normal=vh[-1]
            if normal[2]<0:normal=-normal
            rays=T[:3,:3]@np.linalg.solve(K,np.c_[uv,np.ones(len(uv))].T);denominator=normal@rays
            if np.min(abs(denominator))<1e-6:continue
            distances=(normal@(origin-T[:3,3]))/denominator
            if np.min(distances)<=0:continue
            boundary=(T[:3,3:4]+rays*distances).T;basis=vh[:2].T;xy=(boundary-origin)@basis
            fit=fit_rim_circle(xy)
            if fit is None:continue
            circle_centre,circle_radius,rms,support_fraction,all_rms=fit
            position=origin+basis@circle_centre
            results.append({'position_m':position.tolist(),'axis_world':normal.tolist(),'circle_rms_m':rms,'circle_support_fraction':support_fraction,'circle_untrimmed_rms_m':all_rms,'plane_rms_m':float(np.sqrt(np.mean(((plane-origin)@normal)**2))),'point_count':len(plane),'pixel_xy':[x,y],'radius_px_used':r,'estimated_radius_m':circle_radius,'observed_ellipse_center_pixel_xy':centre.tolist(),'observed_ellipse_axes_px':[float(a),float(b)],'image_seed_distance_px':float(np.linalg.norm(centre-[x,y])),'source':'current RGB closed rim projected onto its observed outside surface; adaptive ROI, no object poses'})
        all_results.extend(results)
    if all_results:
        highest=max(v['position_m'][2] for v in all_results)
        return max((v for v in all_results if v['position_m'][2]>highest-.003),key=lambda v:(-v['image_seed_distance_px'],-v['circle_rms_m']))
    return None
