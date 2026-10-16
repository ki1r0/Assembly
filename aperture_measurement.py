"""Refine a model-selected image aperture using public RGB-D only."""
import numpy as np

def measure_aperture(geometry,sensors,rgb_path):
    import cv2
    from PIL import Image
    pixel=np.asarray(geometry['pixel_centroid_xy'],float);x,y=map(round,pixel)
    depth=np.load(sensors['camera_depth_path'],allow_pickle=False);h,w=depth.shape
    y0,y1=max(0,y-20),min(h,y+21);x0,x1=max(0,x-20),min(w,x+21)
    yy,xx=np.mgrid[y0:y1,x0:x1];z=depth[yy,xx];valid=np.isfinite(z)&(z>0)
    k=np.asarray(sensors['camera_intrinsics']);t=np.asarray(sensors['camera_extrinsics'])
    pc=(t[:3,:3]@(np.linalg.solve(k,np.stack([xx[valid],yy[valid],np.ones(valid.sum())]))*z[valid])+t[:3,3:4]).T
    if len(pc)<30:raise ValueError('Insufficient current depth around selected aperture')
    # The assembly's publicly visible horizontal top has a dominant height cluster;
    # hole-wall pixels are excluded before estimating the rim plane.
    bins=np.floor(pc[:,2]/.00025).astype(np.int64);values,counts=np.unique(bins,return_counts=True);mode=values[np.argmax(counts)]
    plane_points=pc[np.abs(pc[:,2]-(mode+.5)*.00025)<.0004]
    if len(plane_points)<30:raise ValueError('No supported local rim plane')
    origin=np.median(plane_points,axis=0);_,_,vh=np.linalg.svd(plane_points-origin,full_matrices=False);normal=vh[-1]
    if abs(normal[2])<.98:raise ValueError('Selected feature does not support this vertical insertion skill')
    gray=np.asarray(Image.open(rgb_path).convert('L'));crop=gray[y0:y1,x0:x1]
    mask=(crop<float(np.median(crop))-20).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    candidates=[]
    for contour in contours:
        if len(contour)<5 or not 8<cv2.contourArea(contour)<250:continue
        (cx,cy),(a,b),angle=cv2.fitEllipse(contour);center=np.array([cx+x0,cy+y0])
        if np.linalg.norm(center-pixel)<6 and min(a,b)>2 and max(a,b)<25:candidates.append((np.linalg.norm(center-pixel),center,(a,b),contour))
    if candidates:_,pixel,axes,contour=min(candidates,key=lambda a:a[0]);method='local dark-rim ellipse plus dominant RGB-D top plane'
    else:axes=None;contour=None;method='selected image center plus dominant RGB-D top plane'
    ray=t[:3,:3]@np.linalg.solve(k,[*pixel,1]);scale=float(np.dot(normal,origin-t[:3,3])/np.dot(normal,ray));center=t[:3,3]+scale*ray
    if scale<=0:raise ValueError('Rim intersection is behind camera')
    residual=(plane_points-origin)@normal
    from cylinder_fit import fit_observed_xy
    wall=pc[(pc[:,2]<origin[2]-.0005)&(pc[:,2]>origin[2]-.025)&(np.linalg.norm(pc[:,:2]-center[:2],axis=1)<.0081)]
    wall_fit=None
    diameter=None;circle_center=None
    if contour is not None:
        uv=contour.reshape(-1,2).astype(float)+[x0,y0];rays=t[:3,:3]@np.linalg.solve(k,np.c_[uv,np.ones(len(uv))].T)
        scales=np.dot(normal,origin-t[:3,3])/(normal@rays);boundary=(t[:3,3:4]+rays*scales).T
        offset=boundary[:,:2].mean(0);xy=boundary[:,:2]-offset
        coeff=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],np.sum(xy*xy,axis=1),rcond=None)[0]
        diameter=float(2*np.sqrt(max(0,coeff[2]+coeff[:2]@coeff[:2])));circle_center=offset+coeff[:2]
        # A horizontal circular aperture is fitted in world XY, accounting for perspective.
        center[:2]=circle_center
        center[2]=origin[2]-(normal[:2]@(center[:2]-origin[:2]))/normal[2]
    if diameter is not None:
        wall_fit=fit_observed_xy(wall,center[:2],diameter/2)
    if wall_fit is not None:
        center[:2]=wall_fit['center_xy_m'];center[2]=origin[2]-(normal[:2]@(center[:2]-origin[:2]))/normal[2]
    return {'position_m':center.tolist(),'observed_wall_fit':wall_fit,'observed_inner_diameter_m':diameter,'diameter_note':'Visible dark boundary estimate, possibly a counterbore or chamfer; not a hidden shaft-bore dimension.','normal':normal.tolist(),'pixel_center_xy':pixel.tolist(),'ellipse_axes_px':axes,'plane_point_count':len(plane_points),'plane_rms_m':float(np.sqrt(np.mean(residual**2))),'method':method,'source_depth':sensors['camera_depth_path']}
