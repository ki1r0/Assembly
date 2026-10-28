"""G1 opening from official URDF linkage and collision meshes, in metres."""
from pathlib import Path
from functools import lru_cache,cached_property
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


def antipodal_contact_quality(points,normals,friction=.5):
    points=np.asarray(points,float);normals=np.asarray(normals,float)
    direction=points[0]-points[1];length=np.linalg.norm(direction)
    if length<1e-9 or not np.isfinite(points).all() or not np.isfinite(normals).all() or (np.linalg.norm(normals,axis=1)<1e-9).any():raise ValueError('Distinct finite contacts and nonzero normals required')
    direction/=length;normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    cosines=np.clip([normals[0]@direction,-normals[1]@direction],-1,1)
    required=float(np.max(np.sqrt(np.maximum(0,1-cosines*cosines))/cosines)) if min(cosines)>0 else None
    return {'supported':bool(min(cosines)>=1/np.sqrt(1+friction*friction)),
            'assumed_friction_coefficient':friction,'required_friction_coefficient':required,
            'cone_cosines':cosines.tolist(),
            'friction_margin_cost':float(-np.log(max(1e-6,1-(required/friction)**2))) if required is not None and friction>0 and required<=friction else None,
            'margin_cost_semantics':'Dimensionless log barrier near the assumed friction-cone boundary; candidate-ranking heuristic, not a retention probability',
            'model':'Two-point antipodal friction cones; not a multi-contact or physical-retention certificate'}


def binary_stl_points(path):
    data=Path(path).read_bytes()
    if len(data)<84:raise ValueError('Truncated collision STL')
    count=struct.unpack_from('<I',data,80)[0]
    if len(data)!=84+count*50:raise ValueError('Expected binary collision STL')
    faces=np.frombuffer(data,dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')]),offset=84,count=count)
    points=faces['vertices'].reshape(-1,3).astype(float)
    if not len(points) or not np.isfinite(points).all():raise ValueError('Invalid collision mesh')
    return points


def origin(element):
    result=np.eye(4)
    if element is not None:
        result[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        result[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return result


class G1Geometry:
    def __init__(self,urdf):
        robot=ET.parse(urdf).getroot()
        self.joints={j.find('child').get('link'):j for j in robot.findall('joint')}
        self.points={}
        self.hand_points={}
        for link in robot.findall('link'):
            if link.get('name') not in {'left_finger','right_finger','left_inner_knuckle','right_inner_knuckle','left_outer_knuckle','right_outer_knuckle','xarm_gripper_base_link'}:continue
            collision=link.find('collision')
            mesh=collision.find('geometry/mesh')
            points=binary_stl_points(mesh.get('filename'))
            points*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            frame=origin(collision.find('origin'))
            transformed=points@frame[:3,:3].T+frame[:3,3]
            self.hand_points[link.get('name')]=transformed
            if link.get('name') in {'left_finger','right_finger'}:self.points[link.get('name')]=transformed
        if len(self.points)!=2:raise ValueError('Both G1 finger collision meshes required')
        from scipy.spatial import ConvexHull
        hulls={name:ConvexHull(points) for name,points in self.hand_points.items()}
        self.hand_equations={name:hull.equations for name,hull in hulls.items()}
        self.hand_support_points={name:self.hand_points[name][hull.vertices] for name,hull in hulls.items()}
        bounds=self.pad_bounds(.4)
        self.contact_z_range=(bounds[0,2],bounds[1,2])
        self.contact_offset_m=float(bounds[:,2].mean())
        self.contact_offset_samples=(self.contact_offset_m,bounds[0,2]+.0005,bounds[1,2]-.0005)

    def link_transform(self,link,drive):
        chain=[]
        while link!='xarm_gripper_base_link':
            joint=self.joints[link]
            rotation=np.eye(4)
            if joint.get('type')=='revolute':
                mimic=joint.find('mimic')
                if mimic is not None and mimic.get('joint')!='drive_joint':raise ValueError('Unsupported mimic reference')
                angle=drive if mimic is None else drive*float(mimic.get('multiplier','1'))+float(mimic.get('offset','0'))
                axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                rotation[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix()
            chain.append(origin(joint.find('origin'))@rotation)
            link=joint.find('parent').get('link')
        result=np.eye(4)
        for frame in reversed(chain):result=result@frame
        return result

    def finger_points(self,drive):
        if not np.isfinite(drive) or not 0<=drive<=.85:raise ValueError('G1 drive must be in [0,.85] radians')
        result={}
        for name,points in self.points.items():
            frame=self.link_transform(name,drive)
            result[name]=points@frame[:3,:3].T+frame[:3,3]
        return result

    def linkage_errors(self,base_pose,drive,measured_links):
        base=np.eye(4);base[:3,3]=base_pose[:3]
        base[:3,:3]=Rotation.from_quat(np.asarray(base_pose)[[4,5,6,3]]).as_matrix()
        result={}
        for name,pose in measured_links.items():
            expected=base@self.link_transform(name,drive)
            actual_rotation=Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix()
            result[name]={'position_error_m':float(np.linalg.norm(expected[:3,3]-pose[:3])),
                          'angular_error_rad':float(Rotation.from_matrix(expected[:3,:3]@actual_rotation.T).magnitude()),
                          'measured_pose_wxyz':np.asarray(pose,dtype=float).tolist(),
                          'expected_pose_wxyz':np.r_[expected[:3,3],Rotation.from_matrix(expected[:3,:3]).as_quat()[[3,0,1,2]]].tolist()}
        return result

    @lru_cache(maxsize=128)
    def pad_bounds(self,drive):
        faces=[]
        for name,points in self.finger_points(drive).items():
            distal=points[points[:,2]>points[:,2].max()-.015]
            inner=distal[:,1].min() if name=='left_finger' else distal[:,1].max()
            face=distal[abs(distal[:,1]-inner)<.00001]-[0,0,.172]
            faces.append(np.array([face.min(0),face.max(0)]))
        bounds=np.array([np.maximum(faces[0][0],faces[1][0]),np.minimum(faces[0][1],faces[1][1])])
        bounds[0,1]=min(face[0,1] for face in faces)
        bounds[1,1]=max(face[1,1] for face in faces)
        return bounds

    @lru_cache(maxsize=128)
    def opening(self,drive):
        points=self.finger_points(drive)
        # Restrict to the distal overlapping finger band, not proximal knuckles.
        distal=max(points['left_finger'][:,2].max(),points['right_finger'][:,2].max())
        left=points['left_finger'][points['left_finger'][:,2]>distal-.015]
        right=points['right_finger'][points['right_finger'][:,2]>distal-.015]
        return float(left[:,1].min()-right[:,1].max())

    @lru_cache(maxsize=128)
    def drive_for_opening(self,width):
        low,high=self.opening(.85),self.opening(0.)
        if not np.isfinite(width) or not low<=width<=high:raise ValueError(f'G1 opening outside measured range [{low},{high}]')
        return float(brentq(lambda q:self.opening(q)-width,0.,.85))

    def world_points(self,tcp_position,tcp_rotation,drive,*,support_only=False):
        points=[]
        for name,local in (self.hand_support_points if support_only else self.hand_points).items():
            frame=self.link_transform(name,drive)
            points.append(local@frame[:3,:3].T+frame[:3,3]-[0,0,.172])
        return np.concatenate(points)@np.asarray(tcp_rotation).T+np.asarray(tcp_position)

    def support_clearance(self,tcp_position,tcp_rotation,drive,plane_axis,plane_position):
        axis=np.asarray(plane_axis,dtype=float)
        if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9:
            raise ValueError('Finite nonzero support normal required')
        return float(np.min((self.world_points(tcp_position,tcp_rotation,drive,support_only=True)-plane_position)@(axis/np.linalg.norm(axis))))

    @cached_property
    def motion_coefficients(self):
        # The parallel 1:1 G1 linkage gives each vertex A*cos(q)+B*sin(q)+C.
        angles=np.array([0.,.425,.85])
        # Plane support extrema are unchanged by restricting each rigid link to hull vertices.
        samples=np.array([self.world_points([0,0,0],np.eye(3),q,support_only=True) for q in angles])
        basis=np.c_[np.cos(angles),np.sin(angles),np.ones(3)]
        return np.linalg.solve(basis,samples.reshape(3,-1)).reshape(samples.shape)

    def swept_support_clearance(self,tcp_position,tcp_rotation,drive_start,drive_end,plane_axis,plane_position):
        """Minimum over the entire physical jaw stroke, including interior extrema."""
        axis=np.asarray(plane_axis,float);rotation=np.asarray(tcp_rotation,float)
        position=np.asarray(tcp_position,float);plane=np.asarray(plane_position,float)
        drives=np.array([drive_start,drive_end],float)
        if axis.shape!=(3,) or position.shape!=(3,) or plane.shape!=(3,) or rotation.shape!=(3,3) or not all(np.isfinite(v).all() for v in (axis,rotation,position,plane,drives)) or np.linalg.norm(axis)<1e-9 or (drives<0).any() or (drives>.85).any():raise ValueError('Finite G1 stroke in [0,.85] and support plane required')
        axis=axis/np.linalg.norm(axis)
        a,b,c=self.motion_coefficients@(rotation.T@axis)
        c=c+(position-plane)@axis
        low,high=np.sort(drives)
        minimum=np.minimum(a*np.cos(low)+b*np.sin(low)+c,a*np.cos(high)+b*np.sin(high)+c)
        stationary=(np.arctan2(b,a)+np.pi)%(2*np.pi)
        inside=(stationary>=low)&(stationary<=high)
        minimum[inside]=np.minimum(minimum[inside],c[inside]-np.hypot(a[inside],b[inside]))
        return float(minimum.min())

    def clears_table(self,candidate,table):
        q=np.asarray(candidate['quaternion_wxyz'])
        rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        drive=self.drive_for_opening(float(np.clip(candidate['jaw_width_m'],self.opening(.85),self.opening(0))))
        return self.support_clearance(candidate['position_m'],rotation,drive,[0,0,1],[0,0,table])>=.001

    def mating_release_clearances(self,candidate,datum,axis):
        """Relative public-CAD plane support, not fixture identity or retention."""
        q=np.asarray(candidate['quaternion_wxyz'])
        rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        closed=self.drive_for_opening(candidate['observed_width_m'])
        opened=self.drive_for_opening(.08)
        return {mode:max(0.,.003-self.swept_support_clearance(candidate['position_m'],rotation,
            closed,opened,np.asarray(axis)*sign,datum)) for mode,sign in [('parallel',1),('antiparallel',-1)]}

    def contains_hand_points(self,points_tcp_local,drive,margin=.001):
        """Conservative visible self-filter using official meshes, not identity masks."""
        points=np.asarray(points_tcp_local,dtype=float)+[0,0,.172]
        remove=np.zeros(len(points),dtype=bool)
        for name,mesh in self.hand_points.items():
            frame=self.link_transform(name,drive)
            local=(points-frame[:3,3])@frame[:3,:3]
            near=(local>=mesh.min(0)-margin).all(1)&(local<=mesh.max(0)+margin).all(1)
            if not near.any():continue
            equations=self.hand_equations[name]
            remove[near]|=(local[near]@equations[:,:3].T+equations[:,3]<=margin).all(1)
        return remove

    def screen_candidates(self,options,cad_world,cad_normals,table,registration_supported):
        counts={'initial_generated':len(options),'contact_band_rejected':0,'friction_cone_rejected':0,'finger_body_collision_rejected':0,'duplicate_geometry_rejected':0,'geometry_passed':0}
        passed=[]
        for original in options:
            c=dict(original);p=np.array(c['position_m'],dtype=float)
            q=np.asarray(c['quaternion_wxyz']);rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            drive=self.drive_for_opening(float(np.clip(c['jaw_width_m'],self.opening(.85),self.opening(0))))
            p[2]+=max(0.,.0015-self.support_clearance(p,rotation,drive,[0,0,1],[0,0,table]))
            c['position_m']=p.tolist()
            if registration_supported:
                closed=self.drive_for_opening(float(np.clip(c['observed_width_m'],self.opening(.85),self.opening(0.))))
                pad=self.pad_bounds(closed)
                if 'contact_pair_world_m' in c:
                    seed=(np.asarray(c['contact_pair_world_m'])-p)@rotation
                    p+=rotation[:,2]*(seed[:,2].mean()-pad[:,2].mean())
                    c['position_m']=p.tolist()
                local=(np.asarray(cad_world)-p)@rotation
                normals=np.asarray(cad_normals)@rotation
                band=(local[:,0]>=pad[0,0])&(local[:,0]<=pad[1,0])&(local[:,2]>=pad[0,2])&(local[:,2]<=pad[1,2])
                left=local[band&(normals[:,1]>.7)];right=local[band&(normals[:,1]<-.7)]
                if len(left)<4 or len(right)<4:
                    counts['contact_band_rejected']+=1;continue
                if 'contact_pair_world_m' in c:
                    contacts=(np.asarray(c['contact_pair_world_m'])-p)@rotation
                    if (contacts[:,0]<pad[0,0]).any() or (contacts[:,0]>pad[1,0]).any() or (contacts[:,2]<pad[0,2]).any() or (contacts[:,2]>pad[1,2]).any():
                        counts['contact_band_rejected']+=1;continue
                    # A curved wall can protrude beyond its sampled central pair into a finite pad.
                    patch=band&(local[:,1]>=contacts[:,1].min()-.004)&(local[:,1]<=contacts[:,1].max()+.004)
                    left=local[patch&(normals[:,1]>.7)];right=local[patch&(normals[:,1]<-.7)]
                    if len(left)<4 or len(right)<4:
                        counts['contact_band_rejected']+=1;continue
                    pair=np.array([left[np.argmax(left[:,1])],right[np.argmin(right[:,1])]])
                    ly,ry=float(pair[0,1]),float(pair[1,1])
                    c['contact_pair_world_m']=(pair@rotation.T+p).tolist()
                    c['contact_width_source']='Finite-pad CAD surface envelope within 4mm of the seed contact pair; full hand collision screening still required'
                else:
                    pair=np.array([left[np.argmax(left[:,1])],right[np.argmin(right[:,1])]])
                    ly,ry=float(pair[0,1]),float(pair[1,1])
                    c['contact_pair_world_m']=(pair@rotation.T+p).tolist()
                ids=np.argmin(np.linalg.norm(local[:,None,:]-pair[None,:,:],axis=2),axis=0)
                quality=antipodal_contact_quality(pair,normals[ids])
                if not quality['supported']:
                    counts['friction_cone_rejected']+=1;continue
                c['contact_quality']=quality
                width=ly-ry
                if not max(.002,self.opening(.85))<width<.072:
                    counts['contact_band_rejected']+=1;continue
                p+=rotation@np.array([0,(ly+ry)/2,0]);c.update(position_m=p.tolist(),observed_width_m=width,jaw_width_m=min(.08,width+.008))
                closed=self.drive_for_opening(width)
                actual_pad=self.pad_bounds(closed)
                p+=rotation[:,2]*(pad[:,2].mean()-actual_pad[:,2].mean())
                c.update(position_m=p.tolist(),pad_contact_offset_m=float(actual_pad[:,2].mean()))
                if any(np.linalg.norm(p-np.asarray(old['position_m']))<1e-6 and abs(width-old['observed_width_m'])<1e-6 and min(np.linalg.norm(q-np.asarray(old['quaternion_wxyz'])),np.linalg.norm(q+np.asarray(old['quaternion_wxyz'])))<1e-6 for old in passed):
                    counts['duplicate_geometry_rejected']+=1;continue
                opened=self.drive_for_opening(.08)
                if self.swept_support_clearance(p,rotation,opened,closed,[0,0,1],[0,0,table])<.001:
                    counts['finger_body_collision_rejected']+=1;continue
                local=(np.asarray(cad_world)-p)@rotation+[0,0,.172]
                collides=False
                for name,mesh in self.hand_points.items():
                    frame=self.link_transform(name,closed)
                    query=(local-frame[:3,3])@frame[:3,:3]
                    near=query[(query>mesh.min(0)+.0005).all(1)&(query<mesh.max(0)-.0005).all(1)]
                    if not len(near):continue
                    equations=self.hand_equations[name]
                    if (near@equations[:,:3].T+equations[:,3]<-.0005).all(1).any():collides=True;break
                if collides:
                    counts['finger_body_collision_rejected']+=1;continue
                c['contact_candidates']={'left_points':len(left),'right_points':len(right),'pad_z_range_tcp_m':actual_pad[:,2].tolist(),'method':'opposed registered CAD normals within opening-dependent official G1 innermost planar finger faces; not retention proof'}
            c['gripper_geometry']='Official G1 complete collision meshes and revolute linkage'
            passed.append(c)
        counts['geometry_passed']=len(passed)
        return passed,counts
