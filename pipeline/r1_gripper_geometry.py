"""Public R1 CAD geometry only; no scene object state or calibrated pad model.

Matrices use column vectors, metres, and wrist orientation at the TCP. A hull
of each complete collision mesh is conservative: concavities are filled. These
are not PhysX's cooked convex decomposition and cannot certify grasp retention.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull


def _points(transform, points):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


@dataclass
class CollisionPiece:
    mesh_path: str
    collision_path: str
    approximation: str
    vertices_wrist_zero_m: np.ndarray


@dataclass
class FingerGeometry:
    joint_path: str
    body_path: str
    opening_axis_wrist: np.ndarray
    limits_m: tuple
    mimic_reference: str | None
    mimic_gearing: float | None
    pieces: list[CollisionPiece]


@dataclass
class R1GripperGeometry:
    source: str
    arm: str
    wrist_path: str
    tcp_offset_wrist_m: np.ndarray
    fingers: list[FingerGeometry]
    palm: list[CollisionPiece]
    pad_calibration: str = "unknown: no pad patches, contact band, or gap calibration"

    def closing_pressures(self, forces_world, world_from_tcp_rotation):
        """Axis-projected all-object contact, not target identity or retention."""
        forces=np.asarray(forces_world,float)
        rotation=np.asarray(world_from_tcp_rotation,float)
        if (forces.shape!=(2,3) or rotation.shape!=(3,3) or
                not np.isfinite(forces).all() or not np.isfinite(rotation).all() or
                not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7) or
                not np.isclose(np.linalg.det(rotation),1)):
            raise ValueError('Two finite forces and a rigid world-from-TCP rotation required')
        return [float(abs(force@(rotation@finger.opening_axis_wrist)))
                for force,finger in zip(forces,self.fingers)]

    def meshes_tcp(self, joint_positions_m):
        """Return (source prim, vertices) at two independently supplied openings.

        Inputs are actual joint coordinates, not jaw gap or a mimic prediction.
        """
        q = np.asarray(joint_positions_m, dtype=float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("Two finite finger joint positions required")
        meshes = [(p.mesh_path, p.vertices_wrist_zero_m - self.tcp_offset_wrist_m)
                  for p in self.palm]
        for finger, opening in zip(self.fingers, q):
            if not finger.limits_m[0] <= opening <= finger.limits_m[1]:
                raise ValueError("Finger joint position outside public joint limits")
            shift = finger.opening_axis_wrist * opening - self.tcp_offset_wrist_m
            meshes.extend((p.mesh_path, p.vertices_wrist_zero_m + shift)
                          for p in finger.pieces)
        return meshes

    def mesh_bounds_world(self, tcp_position_m, world_from_tcp_rotation, joint_positions_m):
        """Whole mesh AABBs at a proposed robot pose, not contact patches."""
        rotation = np.asarray(world_from_tcp_rotation, dtype=float)
        position = np.asarray(tcp_position_m, dtype=float)
        if (rotation.shape != (3, 3) or position.shape != (3,) or
                not np.isfinite(rotation).all() or not np.isfinite(position).all() or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or
                not np.isclose(np.linalg.det(rotation), 1)):
            raise ValueError("Finite rigid pose required")
        result = []
        palm_paths = {p.mesh_path for p in self.palm}
        for path, vertices in self.meshes_tcp(joint_positions_m):
            world = vertices @ rotation.T + position
            result.append({"mesh_path": path, "body_kind": "palm" if path in palm_paths else "finger",
                           "bounds_world_m": [world.min(0).tolist(), world.max(0).tolist()]})
        return result

    def support_clearance(self, tcp_position_m, world_from_tcp_rotation,
                          joint_positions_m, plane_normal_world, plane_origin_world):
        """Minimum signed hand/plane clearance, including palm, in metres."""
        rotation = np.asarray(world_from_tcp_rotation, dtype=float)
        position = np.asarray(tcp_position_m, dtype=float)
        normal = np.asarray(plane_normal_world, dtype=float)
        origin = np.asarray(plane_origin_world, dtype=float)
        if (rotation.shape != (3, 3) or position.shape != (3,) or
                normal.shape != (3,) or origin.shape != (3,) or
                not all(np.isfinite(x).all() for x in (rotation, position, normal, origin)) or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or
                not np.isclose(np.linalg.det(rotation), 1) or np.linalg.norm(normal) == 0):
            raise ValueError("Finite rigid pose and nonzero plane normal required")
        normal = normal / np.linalg.norm(normal)
        return min(float(np.min((v @ rotation.T + position - origin) @ normal))
                   for _, v in self.meshes_tcp(joint_positions_m))

    def collision_mask(self, points_tcp_m, joint_positions_m, margin_m=0.0,
                       include_palm=True):
        """Conservative point-in-hull mask; positive margin enlarges each hull.

        False does not prove mesh/mesh clearance: only supplied points are tested.
        No intended-contact exemption is applied without pad calibration.
        """
        points = np.asarray(points_tcp_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Finite Nx3 TCP points required")
        if not np.isfinite(margin_m) or margin_m < 0:
            raise ValueError("Collision margin must be finite and nonnegative")
        mask = np.zeros(len(points), dtype=bool)
        palm_paths = {p.mesh_path for p in self.palm}
        for path, vertices in self.meshes_tcp(joint_positions_m):
            if not include_palm and path in palm_paths:
                continue
            equations = ConvexHull(vertices).equations
            mask |= np.all(points @ equations[:, :3].T + equations[:, 3] <= margin_m, axis=1)
        return mask

    def to_dict(self):
        """JSON-ready public geometry, preserving per-arm/finger provenance."""
        from dataclasses import asdict

        def plain(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value

        return plain(asdict(self))

    @classmethod
    def from_dict(cls, data):
        """Restore one arm from to_dict()/the existing JSON export; no pxr needed."""
        def pieces(values):
            return [CollisionPiece(p["mesh_path"], p["collision_path"], p["approximation"],
                                   np.array(p["vertices_wrist_zero_m"], dtype=float, copy=True))
                    for p in values]

        fingers = [FingerGeometry(
            f["joint_path"], f["body_path"],
            np.array(f["opening_axis_wrist"], dtype=float, copy=True),
            tuple(f["limits_m"]), f["mimic_reference"], f["mimic_gearing"], pieces(f["pieces"]))
            for f in data["fingers"]]
        return cls(data["source"], data["arm"], data["wrist_path"],
                   np.array(data["tcp_offset_wrist_m"], dtype=float, copy=True),
                   fingers, pieces(data["palm"]), data["pad_calibration"])

    def public_summary(self):
        """Compact robot-only bounds at q1=q2=0, not pad or measured gap bounds.

        TCP axes retain wrist orientation. Bounds enclose complete collision
        geometry and must not be interpreted as calibrated contact patches.
        """
        def bounds(pieces):
            vertices = np.concatenate([p.vertices_wrist_zero_m for p in pieces])
            vertices = vertices - self.tcp_offset_wrist_m
            return [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]

        return {
            "source": self.source,
            "arm": self.arm,
            "wrist_path": self.wrist_path,
            "tcp_offset_wrist_m": self.tcp_offset_wrist_m.tolist(),
            "geometry_frame": "TCP metres, wrist orientation, both finger joints at zero",
            "pad_calibration": self.pad_calibration,
            "collision_semantics": "Complete collision-mesh bounds; convex hull checks fill concavities; not cooked PhysX geometry or contact calibration",
            "fingers": [{
                "joint_name": f.joint_path.rsplit("/", 1)[-1],
                "joint_path": f.joint_path,
                "body_name": f.body_path.rsplit("/", 1)[-1],
                "body_path": f.body_path,
                "opening_axis_wrist": f.opening_axis_wrist.tolist(),
                "limits_m": list(f.limits_m),
                "zero_joint_tcp_bounds_m": bounds(f.pieces),
            } for f in self.fingers],
            "palm_tcp_bounds_m": bounds(self.palm),
        }


def extract_r1_gripper_geometry(usd_path, arm, tcp_offset_wrist_m=(.0079, 0, .09089)):
    """Read a public R1 USD, reconstructing finger poses at joint q=0.

    Requires pxr only when extracting. Authored body poses are used exclusively
    to recover mesh-to-body transforms, never as zero-opening joint poses.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    offset = np.asarray(tcp_offset_wrist_m, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Finite complete wrist-to-TCP offset required")
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError("Cannot open public robot USD")
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    wrist_path = f"/R1/{arm}_arm_link6"
    wrist = stage.GetPrimAtPath(wrist_path)
    if not wrist:
        raise ValueError(f"Missing wrist {wrist_path}")

    def matrix(prim):
        value = np.array(cache.GetLocalToWorldTransform(prim), dtype=float).T
        if not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6):
            raise ValueError(f"Scaled/sheared robot transforms unsupported: {prim.GetPath()}")
        return value

    def joint_frame(position, quaternion):
        from pxr import Gf
        value = np.array(Gf.Matrix4d(1).SetRotate(Gf.Quatd(quaternion).GetNormalized()), dtype=float).T
        value[:3, 3] = position
        return value

    def pieces(body, wrist_from_body):
        result = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            ancestor = prim
            collision = None
            while ancestor and ancestor != body.GetParent():
                if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                    collision = ancestor
                    break
                ancestor = ancestor.GetParent()
            if collision is None or not UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get():
                continue
            mesh_points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if mesh_points is None or len(mesh_points) < 4:
                raise ValueError(f"Missing collision vertices: {prim.GetPath()}")
            local = np.linalg.inv(matrix(body)) @ matrix(prim)
            vertices = _points(wrist_from_body @ local, np.asarray(mesh_points, dtype=float)) * scale
            ConvexHull(vertices)  # Reject degenerate geometry rather than silently omitting it.
            approximation = collision.GetAttribute("physics:approximation").Get()
            result.append(CollisionPiece(str(prim.GetPath()), str(collision.GetPath()),
                                         str(approximation), vertices))
        if not result:
            raise ValueError(f"No enabled collision geometry for {body.GetPath()}")
        return result

    fingers = []
    for index in (1, 2):
        path = f"{wrist_path}/{arm}_gripper_axis{index}"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdPhysics.PrismaticJoint):
            raise ValueError(f"Missing prismatic joint {path}")
        joint = UsdPhysics.PrismaticJoint(prim)
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if parents != [wrist.GetPath()] or len(children) != 1:
            raise ValueError(f"Unexpected joint bodies for {path}")
        frame0 = joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        frame1 = joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        axis = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
        references = prim.GetRelationship("physxMimicJoint:rotX:referenceJoint").GetTargets()
        gearing = prim.GetAttribute("physxMimicJoint:rotX:gearing").Get()
        fingers.append(FingerGeometry(path, str(children[0]), frame0[:3, :3] @ axis,
            (float(joint.GetLowerLimitAttr().Get()) * scale,
             float(joint.GetUpperLimitAttr().Get()) * scale),
            str(references[0]) if references else None, gearing,
            pieces(stage.GetPrimAtPath(children[0]), frame0 @ np.linalg.inv(frame1))))
    return R1GripperGeometry(str(usd_path), arm, wrist_path, offset.copy(), fingers,
                             pieces(wrist, np.eye(4)))
