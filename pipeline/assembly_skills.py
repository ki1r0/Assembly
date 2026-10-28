"""Task-neutral public RGB-D geometry, grasp poses and relational assembly goals."""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import cv2

@dataclass
class AssemblySkill:
    moving_part:str
    mating_feature:str
    relation:str='coaxial_insertion'
    approach_axis:str='observed_target_normal'
    grasp_pose:str='public_rgbd_6d'
    contact_mode:str='bounded_contact_search'
    verification:str='independent_seat_release_stability'
    def public(self):return asdict(self)

def tool_orientation(approach_axis_world,closing_axis_world):
    z=np.asarray(approach_axis_world,float);y=np.asarray(closing_axis_world,float)
    if z.shape!=(3,) or y.shape!=(3,) or not np.isfinite(np.r_[z,y]).all() or np.linalg.norm(z)<1e-9 or np.linalg.norm(y)<1e-9:
        raise ValueError('Finite nonzero 3D approach and closing vectors required')
    z=z/np.linalg.norm(z);y=y-z*np.dot(y,z)
    if np.linalg.norm(y)<1e-6:raise ValueError('Approach and closing axes must not be parallel')
    y=y/np.linalg.norm(y);matrix=np.column_stack([np.cross(y,z),y,z])
    return {'quaternion_wxyz':Rotation.from_matrix(matrix).as_quat()[[3,0,1,2]].tolist(),'approach_axis_world':z.tolist(),'closing_axis_world':y.tolist(),'meaning':'Pure orientation conversion: local +Z follows your approach; local +Y follows your closing axis projected perpendicular to approach. No pose, grasp, reachability or collision certification.'}

def cloud(depth,K,T):
    yy,xx=np.indices(depth.shape);good=np.isfinite(depth)&(depth>0)&(depth<5)
    points=(T[:3,:3]@ (np.linalg.solve(K,np.stack([xx[good],yy[good],np.ones(good.sum())]))*depth[good])+T[:3,3:4]).T
    return points,np.c_[xx[good],yy[good]]

def selected_points(depth,K,T,pixel,box,table):
    x,y=map(float,pixel);h,w=depth.shape
    bounds=np.asarray(box,float)
    if bounds.shape!=(4,) or not np.isfinite(bounds).all() or not (0<=bounds[0]<=x<=bounds[2]<w and 0<=bounds[1]<=y<=bounds[3]<h):raise ValueError('Selected pixel must be inside finite raw-image xyxy box')
    points,uv=cloud(depth,K,T);keep=(uv[:,0]>=bounds[0])&(uv[:,0]<=bounds[2])&(uv[:,1]>=bounds[1])&(uv[:,1]<=bounds[3])&(points[:,2]>table+.0015)
    pts=points[keep]
    if len(pts)<12:raise ValueError('Insufficient visible above-table object points; select object/rim ROI')
    # Connected 3-D component around the supplied visible pixel, not a named object mask.
    tree=cKDTree(pts);z=float(depth[round(y),round(x)])
    if not np.isfinite(z) or z<=0:raise ValueError('Invalid selected depth')
    seed=(T@np.r_[np.linalg.solve(K,[x,y,1])*z,1])[:3];_,start=tree.query(seed)
    visited={int(start)};front=[int(start)]
    while front:
        new=set(j for i in front for j in tree.query_ball_point(pts[i],.006))-visited;visited.update(new);front=list(new)
    pts=pts[sorted(visited)]
    if len(pts)<12:raise ValueError('Selected component is too small')
    return pts

def observed_object_envelope(points):
    """Geometry of a selected visible component, not an unobserved complete solid."""
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<12 or not np.isfinite(points).all():
        raise ValueError('At least twelve finite measured 3D points required')
    covariance=np.cov(points,rowvar=False)+np.eye(3)*1e-6
    return {'position_m':points.mean(0).tolist(),
            'bounds_world_m':[points.min(0).tolist(),points.max(0).tolist()],
            'position_covariance_m2':covariance.tolist(),
            'point_count':len(points),'complete_volume_observed':False,
            'source':'Current selected RGB-D connected visible component; no simulator object identity or complete hidden shape'}


def grasp_candidates(points,table):
    origin=(np.quantile(points,.05,axis=0)+np.quantile(points,.95,axis=0))/2
    _,_,axes=np.linalg.svd(points-points.mean(0),full_matrices=False)
    # Support plane is calibrated workcell information. Both principal horizontal
    # closing axes are tested; no bolt radius, length, head or task switch is used.
    result=[]
    normals=[np.array([0.,0,1.]),axes[-1],axes[1],-axes[1]]
    for principal in axes:
        for sign in [-1,1]:
            tilted=np.array([0.,0,1.])+sign*principal
            if np.linalg.norm(tilted)>.25:normals.append(tilted/np.linalg.norm(tilted))
    for normal in normals:
      normal=normal/np.linalg.norm(normal)
      if normal[2]<-.2:normal=-normal
      z=-normal
      for direction in [axes[0],axes[1],np.array([1.,0,0]),np.array([0.,1,0])]:
        y=direction.copy()-normal*np.dot(direction,normal)
        if np.linalg.norm(y)<.1:continue
        y/=np.linalg.norm(y);x=np.cross(y,z);matrix=np.column_stack([x,y,z]);span=float(np.ptp(points@y));center=origin.copy();center[2]=max(table+.004,origin[2])
        if not .002<span<.072:continue
        for sign in [1,-1]:
            rotation=matrix.copy();rotation[:,:2]*=sign;q=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
            result.append({'position_m':center.tolist(),'quaternion_wxyz':q.tolist(),'approach_axis_world':z.tolist(),'pregrasp_distance_m':max(.035,float(np.ptp(points@normal))+.02),'jaw_width_m':min(.08,span+.012),'observed_width_m':span,'lift_distance_m':max(.04,float(np.ptp(points[:,2]))+.02),'source':'selected RGB-D component and calibrated support plane'})
    return result

def integer_pixel_intrinsics(raster_intrinsics):
    """USD raster rays sample x+0.5,y+0.5; OpenCV coordinates centre pixels at integers."""
    intrinsics=np.array(raster_intrinsics,float,copy=True);intrinsics[:2,2]-=.5
    return intrinsics

def circular_feature(depth,K,T,pixel,radius_px=24,rgb=None):
    h,w=depth.shape;x,y=map(float,pixel);r=int(radius_px)
    if not 6<=r<=180 or not 0<=x<w or not 0<=y<h:raise ValueError('Raw image pixel and radius 6..180 required')
    if rgb is not None:
        from rim_measurement import observed_rim
        rim=observed_rim(rgb,depth,K,T,pixel,r)
        if rim is not None:return rim
        raise ValueError('Selected circular feature has no supported RGB rim/depth association; obtain another observation or select a visible surface. A background plane is not a verified mouth.')
    points,uv=cloud(depth,K,T);rad=np.linalg.norm(uv-[x,y],axis=1);pts=points[(rad<r)&(rad>r*.35)]
    if len(pts)<30:raise ValueError('Insufficient visible rim depth')
    bins=np.floor(pts[:,2]/.001).astype(int);values,counts=np.unique(bins,return_counts=True);level=values[np.argmax(counts)]
    plane=pts[np.abs(pts[:,2]-(level+.5)*.001)<.0012];origin=np.median(plane,axis=0);_,_,vh=np.linalg.svd(plane-origin,full_matrices=False);normal=vh[-1]
    if normal[2]<0:normal=-normal
    ray=T[:3,:3]@np.linalg.solve(K,[x,y,1]);scale=float(normal@(origin-T[:3,3])/(normal@ray));center=T[:3,3]+scale*ray
    if scale<=0:raise ValueError('Feature behind camera')
    circle_residual=None
    if rgb is not None:
        x0,y0=max(0,round(x)-r),max(0,round(y)-r);x1,y1=min(w,round(x)+r+1),min(h,round(y)+r+1)
        gray=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
        mask=(gray<np.median(gray)-15).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE);candidates=[]
        for contour in contours:
            if len(contour)<8 or cv2.contourArea(contour)<8:continue
            (cx,cy),(a,b),angle=cv2.fitEllipse(contour);distance=np.linalg.norm(np.array([cx+x0,cy+y0])-[x,y])
            if distance<max(5,r*.3) and min(a,b)>2 and max(a,b)<2*r*.95:candidates.append((distance,contour))
        if candidates:
            contour=min(candidates,key=lambda a:a[0])[1].reshape(-1,2)+[x0,y0]
            rays=T[:3,:3]@np.linalg.solve(K,np.c_[contour,np.ones(len(contour))].T)
            distances=(normal@(origin-T[:3,3]))/(normal@rays);boundary=(T[:3,3:4]+rays*distances).T
            basis=vh[:2].T;xy=(boundary-origin)@basis;coeff=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],(xy*xy).sum(1),rcond=None)[0]
            fitted=origin+basis@coeff[:2];radii=np.linalg.norm(xy-coeff[:2],axis=1);circle_residual=float(radii.std())
            if circle_residual<.001:center=fitted
    return {'position_m':center.tolist(),'axis_world':normal.tolist(),'circle_rms_m':circle_residual,'plane_rms_m':float(np.sqrt(np.mean(((plane-origin)@normal)**2))),'point_count':len(plane),'pixel_xy':[x,y],'source':'model-selected pixel ray intersected with locally observed support plane; no semantic identity proof'}

def aligned_tcp(current_tcp,moving_feature,target_feature,clearance,axis_alignment='nearest'):
    """Transport a feature; line-axis signs alone must not flip the seating face."""
    current=np.asarray(current_tcp);p=np.asarray(moving_feature['position_m']);target=np.asarray(target_feature['position_m']);a=np.asarray(moving_feature['axis_world']);b=np.asarray(target_feature['axis_world']);a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    if axis_alignment not in ('nearest','parallel','antiparallel'):
        raise ValueError('axis_alignment must be nearest, parallel or antiparallel')
    sign=-1. if axis_alignment=='antiparallel' or (axis_alignment=='nearest' and a@b<0) else 1.
    rotation,_=Rotation.align_vectors((b*sign)[None],a[None]);q=Rotation.from_quat(current[[4,5,6,3]])
    return np.r_[target+b*clearance+rotation.apply(current[:3]-p),(rotation*q).as_quat()[[3,0,1,2]]]

def assembly_release_clearance(current_tcp,moving_feature,target_feature,width,axis_alignment='nearest'):
    """Conservative Panda whole-hand plane bound, at the actual aligned pose."""
    seat=aligned_tcp(current_tcp,moving_feature,target_feature,0.,axis_alignment)
    normal=np.asarray(target_feature['axis_world'],float);normal/=np.linalg.norm(normal)
    clearance=hand_plane_clearance(seat[:3],Rotation.from_quat(seat[[4,5,6,3]]),width,normal,target_feature['position_m'])
    return max(0.,.003-clearance)


def hand_plane_clearance(position,rotation,width,normal,origin):
    """Conservative hand bounds from the authored public Franka collision mesh."""
    p=np.asarray(position);n=np.asarray(normal);o=np.asarray(origin);z=rotation.apply([0,0,1]);levels=[float((p-z*.065-o)@n)-.035]
    basis=rotation.as_matrix();half=np.array([.0105016,.0132147,.0268833])
    for sign in [-1,1]:
        center=p+rotation.apply([0,sign*(width/2+.0131309),-.0179834]);levels.append(float((center-o)@n)-float(np.abs(n@basis)@half))
    return min(levels)

def clears_support_plane(candidate,table):
    q=np.array(candidate['quaternion_wxyz']);rotation=Rotation.from_quat(q[[1,2,3,0]])
    return hand_plane_clearance(candidate['position_m'],rotation,candidate['jaw_width_m'],[0,0,1],[0,0,table])>=.001


def part_component(depth,K,T,pixel,table,radius=.18):
    points,uv=cloud(depth[::2,::2],np.array([[K[0,0]/2,0,K[0,2]/2],[0,K[1,1]/2,K[1,2]/2],[0,0,1]]),T)
    x,y=map(round,pixel);z=float(depth[y,x]);seed=(T@np.r_[np.linalg.solve(K,[x,y,1])*z,1])[:3]
    points=points[(points[:,2]>table+.0015)&(np.linalg.norm(points-seed,axis=1)<radius)]
    if len(points)<12:raise ValueError('Insufficient component depth')
    tree=cKDTree(points);_,start=tree.query(seed);visited={int(start)};front=[int(start)]
    while front:
        new=set(j for i in front for j in tree.query_ball_point(points[i],.012))-visited;visited.update(new);front=list(new)
    points=points[sorted(visited)]
    center=points.mean(0);_,singular,vh=np.linalg.svd(points-center,full_matrices=False);axis=vh[0]
    if axis[2]<0:axis=-axis
    projection=(points-center)@axis;ends=[center+axis*np.quantile(projection,q) for q in [.02,.98]]
    return {'center_m':center.tolist(),'axis_world':axis.tolist(),'endpoints_m':[p.tolist() for p in ends],'axis_observability_ratio':float(singular[0]/max(singular[1],1e-12)),'point_count':len(points),'source':'connected visible RGB-D component PCA, axis sign not semantic identity','axis_sign_ambiguous':True,'_surface_points':points}


def measure_pattern(rgb,center_pixel,bbox):
    h,w=rgb.shape[:2];x0,y0,x1,y1=map(round,bbox)
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Pattern ROI outside image')
    gray=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY);mask=(gray<np.median(gray)-22).astype(np.uint8);contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE);result=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if len(contour)<8 or not 8<area<350:continue
        (x,y),(a,b),angle=cv2.fitEllipse(contour)
        if min(a,b)<2 or max(a,b)/min(a,b)>3:continue
        pixel=np.array([x+x0,y+y0]);delta=pixel-np.array(center_pixel)
        if np.linalg.norm(delta)<8:continue
        hour=float((np.arctan2(delta[0],-delta[1])%(2*np.pi))*6/np.pi)
        result.append({'pixel_xy':pixel.tolist(),'clock_hour':hour,'ellipse_axes_px':[a,b],'detector_identity':'dark contour candidate, may be lettering or another opening'})
    return sorted(result,key=lambda c:c['clock_hour'])

def mating_plane_grasp(candidate,cad_world,cad_normals_world,feature_world,feature_axis,table,require_seat_clearance=True):
    """Task-oriented grasp using public CAD contacts and a canonical mating plane.

    The same rule applies to any part with a declared seating datum. No task
    names, world target coordinates or simulator object state enter this function.
    """
    c=dict(candidate);q=np.array(c['quaternion_wxyz']);R=Rotation.from_quat(q[[1,2,3,0]]);axis=np.asarray(feature_axis,float);axis/=np.linalg.norm(axis);feature=np.asarray(feature_world,float);p=np.array(c['position_m'])
    def clearance(point):return hand_plane_clearance(point,R,c['jaw_width_m'],axis,feature)
    if require_seat_clearance:p+=axis*max(0.,.003-clearance(p))
    p[2]+=max(0.,.0015-hand_plane_clearance(p,R,c['jaw_width_m'],[0,0,1],[0,0,table]))
    local=(cad_world-p)@R.as_matrix();normals=cad_normals_world@R.as_matrix()
    band=(np.abs(local[:,0])<.0105)&(local[:,2]>-.00875)&(local[:,2]<.0089)
    left=local[band&(normals[:,1]>.7)];right=local[band&(normals[:,1]<-.7)]
    if len(left)<4 or len(right)<4:return None
    if 'contact_pair_world_m' in c:
        contacts=(np.array(c['contact_pair_world_m'])-p)@R.as_matrix()
        if np.any(np.abs(contacts[:,0])>.0105) or np.any(contacts[:,2]<-.00875) or np.any(contacts[:,2]>.0089):return None
        left_y=float(contacts[:,1].max());right_y=float(contacts[:,1].min())
    else:
        # Closing fingers encounter the outer envelope before interior CAD surfaces.
        left_y=float(left[:,1].max());right_y=float(right[:,1].min())
    width=left_y-right_y
    if not .002<width<.072:return None
    p+=R.apply([0,(left_y+right_y)/2,0])
    if require_seat_clearance and clearance(p)<.001:return None
    local=(cad_world-p)@R.as_matrix()
    if np.any(np.sum((local-[0,0,-.065])**2,axis=1)<.030**2):return None
    c.update(position_m=p.tolist(),jaw_width_m=min(.08,width+.012),observed_width_m=width,canonical_mating_plane_clearance_m=float(clearance(p)),required_release_clearance_m=max(0.,.003-float(clearance(p))),contact_candidates={'left_points':len(left),'right_points':len(right),'method':'opposed public CAD surface normals in measured robot finger-pad band; not retention proof'},grasp_position_source='observed ROI plus public CAD mating datum and hand clearance')
    return c

def geometric_patterns(rgb,depth,K,T):
    """Unlabelled circular image features and relative clock geometry, RGB-D only."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,10);contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE);circles=[];h,w=depth.shape
    for contour in contours:
        area=cv2.contourArea(contour)
        if len(contour)<8 or not 8<area<70000:continue
        (x,y),(a,b),angle=cv2.fitEllipse(contour)
        if min(a,b)<2 or max(a,b)>360 or max(a,b)/min(a,b)>3:continue
        uv=contour.reshape(-1,2)-[x,y];rotation=Rotation.from_euler('z',-angle,degrees=True).as_matrix()[:2,:2];local=uv@rotation.T;residual=float(np.std(np.linalg.norm(local/np.array([a/2,b/2]),axis=1)))
        if residual>.22:continue
        if any(np.linalg.norm(np.array(c['pixel_xy'])-[x,y])<3 and abs(max(c['axes_px'])-max(a,b))<8 for c in circles):continue
        # Measure material just outside the dark ellipse, not the depth of its cavity.
        r=int(max(a,b)*.7)+3;x0,x1=max(0,round(x)-r),min(w,round(x)+r+1);y0,y1=max(0,round(y)-r),min(h,round(y)+r+1);yy,xx=np.mgrid[y0:y1,x0:x1];uv=np.c_[xx.ravel()-x,yy.ravel()-y]@rotation.T;rho=np.linalg.norm(uv/np.array([a/2,b/2]),axis=1);z=depth[yy.ravel(),xx.ravel()];valid=(rho>1.03)&(rho<1.4)&np.isfinite(z)&(z>0)
        height=None
        if valid.sum()>=10:
            pixels=np.stack([xx.ravel()[valid],yy.ravel()[valid],np.ones(valid.sum())]);points=(T[:3,:3]@(np.linalg.solve(K,pixels)*z[valid])+T[:3,3:4]).T;bins=np.floor(points[:,2]/.001).astype(int);values,counts=np.unique(bins,return_counts=True);mode=values[np.argmax(counts)];surface=points[np.abs(points[:,2]-(mode+.5)*.001)<.0012]
            if len(surface)>=10:height=float(np.median(surface[:,2]))
        circles.append({'pixel_xy':[float(x),float(y)],'axes_px':[float(a),float(b)],'rim_height_world_z_m':height,'ellipse_residual':residual})
    circles.sort(key=lambda c:-np.prod(c['axes_px']))
    for i,circle in enumerate(circles):circle['candidate_id']=f'C{i}'
    patterns=[]
    for large in circles:
        if min(large['axes_px'])<25 or max(large['axes_px'])<45:continue
        children=[]
        for small in circles:
            if max(small['axes_px'])>30:continue
            delta=np.array(small['pixel_xy'])-large['pixel_xy'];distance=np.linalg.norm(delta)
            if not max(12,min(large['axes_px'])*.45)<distance<max(large['axes_px'])*1.3:continue
            if large['rim_height_world_z_m'] is not None and small['rim_height_world_z_m'] is not None and abs(large['rim_height_world_z_m']-small['rim_height_world_z_m'])>.003:continue
            hour=float((np.arctan2(delta[0],-delta[1])%(2*np.pi))*6/np.pi);children.append({'candidate_id':small['candidate_id'],'pixel_xy':small['pixel_xy'],'clock_hour':hour})
        if len(children)>=3:
            patterns.append({'centre':large,'surrounding_candidates':sorted(children,key=lambda c:c['clock_hour'])[:16],'identity':'Unlabelled observed dark ellipses, not task-selected holes','clock_convention':'image 12 up, 3 right'})
        if len(patterns)>=3:break
    return patterns

def antipodal_grasp_candidates(points,normals,table,max_candidates=192,pad_offset_m=.0022670751,clearance_test=None,approach_tilts=(0.,-.45,.45,-.8,.8,-1.05,1.05)):
    """Local parallel-jaw contact pairs on observed-registered public CAD.

    Unlike whole-component PCA width, this can pinch a local ring wall. Normals
    and finite pad footprint reject adjacent, non-opposed surface samples.
    No task identity or scene object transforms are accepted.
    """
    points=np.asarray(points);normals=np.asarray(normals);tree=cKDTree(points);candidates=[];seen=set()
    for i in np.linspace(0,len(points)-1,min(700,len(points)),dtype=int):
        p=points[i];normal=normals[i];near=np.array(tree.query_ball_point(p,.071),int)
        delta=p-points[near];axial=delta@normal;lateral=np.linalg.norm(delta-axial[:,None]*normal,axis=1)
        valid=(axial>.004)&(lateral<.002)&(normals[near]@normal<-.85)
        ids=near[valid]
        if not len(ids):continue
        j=ids[np.argmin(lateral[valid])];direction=p-points[j];width=float(np.linalg.norm(direction));y=direction/width;centre=(p+points[j])/2
        key=tuple(np.round(np.r_[centre/.008,np.abs(y)/.3]).astype(int))
        if key in seen:continue
        seen.add(key)
        up=np.array([0.,0,1.]);up-=y*(up@y)
        if np.linalg.norm(up)<.3:continue
        up/=np.linalg.norm(up);side=np.cross(y,up)
        for tilt in approach_tilts:
            z=-(up*np.cos(tilt)+side*np.sin(tilt));x=np.cross(y,z);R=np.column_stack([x,y,z])
            for sign in [1,-1]:
                rotation=R.copy();rotation[:,:2]*=sign;q=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]];tcp=centre-rotation[:,2]*pad_offset_m
                candidate={'position_m':tcp.tolist(),'quaternion_wxyz':q.tolist(),'approach_axis_world':z.tolist(),'pregrasp_distance_m':.06,'jaw_width_m':min(.08,width+.008),'observed_width_m':width,'lift_distance_m':.06,'source':'local opposed normals on public CAD registered to current selected RGB-D component','antipodal_pair_distance_m':width,'contact_pair_world_m':[p.tolist(),points[j].tolist()]}
                if (clearance_test or clears_support_plane)(candidate,table):candidates.append(candidate)
        if len(candidates)>=max_candidates:break
    return candidates[:max_candidates]


def closed_fingers_clear_part(candidate,cad_world,hand_hulls,tolerance=.0005):
    """Reject part surfaces inside the complete public finger collision hulls.

    Intended pad contact is allowed at the boundary; an enlarged feature inside
    the proximal finger body is not an admissible smaller-width grasp.
    """
    rotation=Rotation.from_quat(np.array(candidate['quaternion_wxyz'])[[1,2,3,0]])
    local=(np.asarray(cad_world)-np.array(candidate['position_m']))@rotation.as_matrix();local[:,2]+=.1034
    for side,sign in [('left',1),('right',-1)]:
        points=local.copy();points[:,1]-=sign*candidate['observed_width_m']/2
        bounds=hand_hulls[side+'_bounds'];near=np.all(points>bounds[0]+tolerance,axis=1)&np.all(points<bounds[1]-tolerance,axis=1)
        if not near.any():continue
        equation=hand_hulls[side+'_equations'];inside=np.all(points[near]@equation[:,:3].T+equation[:,3]<-tolerance,axis=1)
        if inside.any():return False
    return True


def screen_grasp_geometry(options,cad_world,cad_normals,datum,datum_axis,table,hand_hulls,registration_supported):
    counts={'initial_generated':len(options),'contact_band_rejected':0,'finger_body_collision_rejected':0,'geometry_passed':0}
    passed=[]
    for candidate in options:
        if registration_supported:
            candidate=mating_plane_grasp(candidate,cad_world,cad_normals,datum,datum_axis,table,require_seat_clearance=False)
            if candidate is None:
                counts['contact_band_rejected']+=1
                continue
            if not closed_fingers_clear_part(candidate,cad_world,hand_hulls):
                counts['finger_body_collision_rejected']+=1
                continue
        passed.append(candidate)
    counts['geometry_passed']=len(passed)
    return passed,counts


def project_world_points(points,K,world_from_camera):
    """Project observed/estimated world geometry into the same raw RGB frame."""
    points=np.asarray(points,float).reshape(-1,3);camera=(points-np.asarray(world_from_camera)[:3,3])@np.asarray(world_from_camera)[:3,:3]
    pixels=camera@np.asarray(K).T
    result=np.full((len(points),2),np.nan);good=camera[:,2]>0
    result[good]=pixels[good,:2]/pixels[good,2,None]
    return result


def assembly_grasp_metrics(candidate,mating_axis,axial_protrusion_m,release_clearances=None):
    """Task/controller-independent hand clearance cost for a relational grasp."""
    rotation=Rotation.from_quat(np.array(candidate['quaternion_wxyz'])[[1,2,3,0]])
    axis=np.asarray(mating_axis,float);axis/=np.linalg.norm(axis)
    dot=abs(float(rotation.apply([0,1,0])@axis));gap=float(candidate.get('required_release_clearance_m',0.))
    result = {'absolute_dot_jaw_vs_mating_axis':dot,'closing_axis_alignment_to_insertion_axis':dot,'alignment_convention':'Absolute cosine: zero means perpendicular, one means parallel; not an angle','guided_entry_before_release_m':max(0.,float(axial_protrusion_m)-gap),'cost':4.*dot+gap/max(.005,float(axial_protrusion_m)),'source':'Public CAD mating axis and robot hand clearance; independent of task/controller names and fixture GT'}

    if release_clearances is not None:
        # A chosen axis sign is needed; do not report the zero-gap default as guided entry.
        result['guided_entry_before_release_m'] = None
        result['guided_entry_before_release_by_axis_alignment_m'] = {
            mode: max(0., float(axial_protrusion_m) - float(clearance))
            for mode, clearance in release_clearances.items()
        }
    return result
