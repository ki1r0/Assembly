"""Public URDF arm meshes against a calibrated support plane; no object poses."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from xarm_gripper import binary_stl_points, origin


class ArmSupportGeometry:
    def __init__(self, urdf, joint_names, base_link='link_base'):
        model=ET.parse(urdf).getroot()
        self.base_link=base_link
        self.joints={j.find('child').get('link'):j for j in model.findall('joint')}
        self.joint_names=list(joint_names)
        moving={j.find('child').get('link') for j in model.findall('joint') if j.get('name') in joint_names}
        self.points={}
        for link in model.findall('link'):
            if link.get('name') not in moving:continue
            parts=[]
            for collision in link.findall('collision'):
                mesh=collision.find('geometry/mesh')
                if mesh is None:raise ValueError('Arm support check requires explicit public collision meshes')
                points=binary_stl_points(mesh.get('filename'))*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                frame=origin(collision.find('origin'))
                parts.append(points@frame[:3,:3].T+frame[:3,3])
            if not parts:raise ValueError('Missing arm collision geometry')
            points=np.concatenate(parts)
            self.points[link.get('name')]=points[ConvexHull(points).vertices]
        if len(self.points)!=len(joint_names):raise ValueError('Incomplete moving-arm collision geometry')

    def link_poses(self, positions, world_from_base):
        q=np.asarray(positions,float);base=np.asarray(world_from_base,float)
        if q.shape!=(len(self.joint_names),) or base.shape!=(4,4) or not np.isfinite(q).all() or not np.isfinite(base).all():raise ValueError('Finite joint vector and base transform required')
        values=dict(zip(self.joint_names,q));poses={self.base_link:base}
        def pose(link):
            if link not in poses:
                joint=self.joints[link];motion=np.eye(4)
                if joint.get('type')!='fixed':
                    axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                    motion[:3,:3]=Rotation.from_rotvec(axis*values[joint.get('name')]).as_matrix()
                poses[link]=pose(joint.find('parent').get('link'))@origin(joint.find('origin'))@motion
            return poses[link]
        return {link:pose(link) for link in self.points}

    def plane_clearances(self, positions, world_from_base, plane_z_m):
        if not np.isfinite(plane_z_m):raise ValueError('Finite calibrated plane required')
        frames=self.link_poses(positions,world_from_base)
        return {link:float(np.min(points@frames[link][2,:3]+frames[link][2,3])-plane_z_m)
                for link,points in self.points.items()}
