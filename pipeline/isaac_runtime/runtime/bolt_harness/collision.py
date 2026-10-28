"""Source-derived two-hull collision proxy for the M6 hub bolt task."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CAP_SHAFT_SPLIT_Y_SOURCE_UNITS = 8.925001144
PHYSX_GPU_HULL_VERTEX_LIMIT = 64
_POINT_WELD_TOLERANCE_SOURCE_UNITS = 1e-9


@dataclass(frozen=True)
class _HullMesh:
    name: str
    points: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    clipped_point_count: int
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    max_radius_xz: float


def _clip_polygon_y(points, split_y: float, keep_above: bool):
    """Clip one triangle against a root-local Y half-space."""
    output = []
    for a, b in zip(points, points[1:] + points[:1]):
        inside_a = a[1] >= split_y if keep_above else a[1] <= split_y
        inside_b = b[1] >= split_y if keep_above else b[1] <= split_y
        if inside_a != inside_b:
            t = (split_y - a[1]) / (b[1] - a[1])
            output.append(tuple(a[axis] + t * (b[axis] - a[axis]) for axis in range(3)))
        if inside_b:
            output.append(tuple(b))
    return output


def _nondegenerate_polygon(points):
    tolerance = _POINT_WELD_TOLERANCE_SOURCE_UNITS
    unique = []
    for point in points:
        if not any(max(abs(point[axis] - prior[axis]) for axis in range(3)) <= tolerance
                   for prior in unique):
            unique.append(point)
    if len(unique) > 1 and max(abs(unique[0][axis] - unique[-1][axis]) for axis in range(3)) <= tolerance:
        unique.pop()
    if len(unique) < 3:
        return []

    import numpy as np

    origin = np.asarray(unique[0], dtype=np.float64)
    area_vector = sum(
        (np.cross(np.asarray(unique[index]) - origin,
                  np.asarray(unique[index + 1]) - origin)
         for index in range(1, len(unique) - 1)),
        np.zeros(3, dtype=np.float64),
    )
    return unique if float(np.linalg.norm(area_vector)) > tolerance * tolerance else []


def _triangulate_mesh(mesh) -> list[tuple[int, int, int]]:
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    triangles = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        for index in range(1, count - 1):
            triangles.append((int(face[0]), int(face[index]), int(face[index + 1])))
    if cursor != len(indices):
        raise ValueError("source mesh face counts do not consume all indices")
    return triangles


def _make_convex_hull(name: str, clipped_points) -> _HullMesh:
    import numpy as np
    from scipy.spatial import ConvexHull

    raw = np.asarray(clipped_points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 4:
        raise ValueError(f"{name} split produced fewer than four 3D points")

    quantized = np.rint(raw / _POINT_WELD_TOLERANCE_SOURCE_UNITS).astype(np.int64)
    _, first_indices = np.unique(quantized, axis=0, return_index=True)
    unique_points = raw[np.sort(first_indices)]
    if len(unique_points) < 4:
        raise ValueError(f"{name} split produced a degenerate point cloud")

    hull = ConvexHull(unique_points)
    hull_indices = sorted({int(index) for face in hull.simplices for index in face})
    hull_index_map = {old: new for new, old in enumerate(hull_indices)}
    hull_points = unique_points[hull_indices]
    faces = []
    for face_index, simplex in enumerate(hull.simplices):
        a, b, c = (int(index) for index in simplex)
        pa, pb, pc = unique_points[[a, b, c]]
        normal = np.cross(pb - pa, pc - pa)
        if float(np.dot(normal, hull.equations[face_index, :3])) < 0.0:
            b, c = c, b
        faces.append((hull_index_map[a], hull_index_map[b], hull_index_map[c]))

    minimum = hull_points.min(axis=0)
    maximum = hull_points.max(axis=0)
    return _HullMesh(
        name=name,
        points=tuple(tuple(float(value) for value in point) for point in hull_points),
        faces=tuple(faces),
        clipped_point_count=len(unique_points),
        bounds=tuple((float(minimum[i]), float(maximum[i])) for i in range(3)),
        max_radius_xz=max(math.hypot(float(point[0]), float(point[2])) for point in hull_points),
    )


def _split_source_mesh(stage, root_prim, source_mesh_prim, split_y: float) -> tuple[_HullMesh, _HullMesh]:
    from pxr import Gf, UsdGeom

    root_world = UsdGeom.XformCache().GetLocalToWorldTransform(root_prim)
    cache = UsdGeom.XformCache()
    source_world = cache.GetLocalToWorldTransform(source_mesh_prim)
    source_to_root = source_world * root_world.GetInverse()
    source_mesh = UsdGeom.Mesh(source_mesh_prim)
    root_points = [
        tuple(float(value) for value in source_to_root.Transform(Gf.Vec3d(*point)))
        for point in (source_mesh.GetPointsAttr().Get() or [])
    ]
    if not root_points:
        raise ValueError(f"source mesh is empty: {source_mesh_prim.GetPath()}")

    shaft_points = []
    cap_points = []
    for triangle in _triangulate_mesh(source_mesh):
        polygon = [root_points[index] for index in triangle]
        if all(abs(point[1] - split_y) <= _POINT_WELD_TOLERANCE_SOURCE_UNITS for point in polygon):
            # A CAD face coplanar with the shoulder is the cap underside, not
            # extra shaft volume; assigning it only to the cap avoids bridging.
            cap_points.extend(polygon)
            continue
        for destination, keep_above in ((shaft_points, False), (cap_points, True)):
            clipped = _nondegenerate_polygon(_clip_polygon_y(polygon, split_y, keep_above))
            if clipped:
                destination.extend(clipped)

    return (
        _make_convex_hull("shaft", shaft_points),
        _make_convex_hull("cap", cap_points),
    )


def _copy_material_binding_relationships(source_prim, target_prim) -> None:
    for relationship in source_prim.GetRelationships():
        name = relationship.GetName()
        if not (name.startswith("material:binding") or name.startswith("physics:material:binding")):
            continue
        copied = target_prim.CreateRelationship(name, custom=relationship.IsCustom())
        copied.SetTargets(relationship.GetTargets())
        for key, value in relationship.GetAllMetadata().items():
            copied.SetMetadata(key, value)


def _author_hull_mesh(stage, root_prim, source_prim, source_physx, hull: _HullMesh) -> dict[str, Any]:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    prim_path = root_prim.GetPath().AppendChild(f"collision_{hull.name}")
    imageable = UsdGeom.Mesh.Define(stage, prim_path)
    imageable.CreatePointsAttr([Gf.Vec3f(*point) for point in hull.points])
    imageable.CreateFaceVertexCountsAttr([3] * len(hull.faces))
    imageable.CreateFaceVertexIndicesAttr([index for face in hull.faces for index in face])
    imageable.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    imageable.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    imageable.CreatePurposeAttr(UsdGeom.Tokens.guide)
    imageable.MakeInvisible()

    proxy_prim = imageable.GetPrim()
    collision = UsdPhysics.CollisionAPI.Apply(proxy_prim)
    collision.GetCollisionEnabledAttr().Set(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(proxy_prim)
    mesh_collision.GetApproximationAttr().Set("convexHull")
    hull_collision = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(proxy_prim)
    hull_collision.GetHullVertexLimitAttr().Set(PHYSX_GPU_HULL_VERTEX_LIMIT)

    proxy_physx = PhysxSchema.PhysxCollisionAPI.Apply(proxy_prim)
    if source_physx is not None:
        for getter, setter in (
            (source_physx.GetContactOffsetAttr, proxy_physx.GetContactOffsetAttr),
            (source_physx.GetRestOffsetAttr, proxy_physx.GetRestOffsetAttr),
        ):
            value = getter().Get()
            if value is not None and math.isfinite(float(value)):
                setter().Set(value)

    _copy_material_binding_relationships(source_prim, proxy_prim)
    return {
        "name": hull.name,
        "prim_path": str(prim_path),
        "collision_enabled": bool(collision.GetCollisionEnabledAttr().Get()),
        "render_visibility": str(UsdGeom.Imageable(proxy_prim).GetVisibilityAttr().Get()),
        "purpose": str(UsdGeom.Imageable(proxy_prim).GetPurposeAttr().Get()),
        "approximation": str(mesh_collision.GetApproximationAttr().Get()),
        "physx_hull_vertex_limit": int(hull_collision.GetHullVertexLimitAttr().Get()),
        "clipped_unique_point_count": hull.clipped_point_count,
        "hull_vertex_count": len(hull.points),
        "triangle_count": len(hull.faces),
        "bounds_root_local_source_units_xyz": [list(bound) for bound in hull.bounds],
        "max_radius_xz_source_units": hull.max_radius_xz,
    }


def author_split_bolt_convex_proxies(
    stage,
    *,
    bolt_root_path: str,
    source_mesh_path: str,
    split_y_source_units: float = CAP_SHAFT_SPLIT_Y_SOURCE_UNITS,
) -> dict[str, Any]:
    """Replace only the visual CAD mesh collider with two hidden, CAD-derived hulls.

    The bolt root must already own its RigidBodyAPI. Source mesh vertices are
    transformed into bolt-root coordinates before clipping, so the proxy points
    remain independent of the instance's world pose and scale.
    """
    if not math.isfinite(float(split_y_source_units)):
        raise ValueError("split_y_source_units must be finite")

    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(bolt_root_path)
    source_prim = stage.GetPrimAtPath(source_mesh_path)
    if not root or not root.IsValid():
        raise ValueError(f"bolt root is missing: {bolt_root_path}")
    if not source_prim or not source_prim.IsValid() or not source_prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"source mesh is missing or not a UsdGeom.Mesh: {source_mesh_path}")
    if not source_prim.GetPath().HasPrefix(root.GetPath()):
        raise ValueError("source mesh must be beneath the bolt rigid-body root")
    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("bolt root must already have RigidBodyAPI; this helper does not author body physics")

    shaft_hull, cap_hull = _split_source_mesh(stage, root, source_prim, float(split_y_source_units))
    shaft_min_y, shaft_max_y = shaft_hull.bounds[1]
    cap_min_y, cap_max_y = cap_hull.bounds[1]
    tolerance = 2e-6
    if abs(shaft_max_y - split_y_source_units) > tolerance or abs(cap_min_y - split_y_source_units) > tolerance:
        raise ValueError(
            "split plane does not intersect source on both sides: "
            f"shaft max Y={shaft_max_y}, cap min Y={cap_min_y}, split={split_y_source_units}"
        )

    source_collision = UsdPhysics.CollisionAPI.Apply(source_prim)
    source_collision.GetCollisionEnabledAttr().Set(False)
    source_physx = (
        PhysxSchema.PhysxCollisionAPI(source_prim)
        if source_prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
        else None
    )
    pieces = [
        _author_hull_mesh(stage, root, source_prim, source_physx, shaft_hull),
        _author_hull_mesh(stage, root, source_prim, source_physx, cap_hull),
    ]
    contact_offset = source_physx.GetContactOffsetAttr().Get() if source_physx is not None else None
    rest_offset = source_physx.GetRestOffsetAttr().Get() if source_physx is not None else None
    return {
        "bolt_root_path": bolt_root_path,
        "source_mesh_path": source_mesh_path,
        "source_collision_enabled": bool(source_collision.GetCollisionEnabledAttr().Get()),
        "rigid_body_preserved": bool(root.HasAPI(UsdPhysics.RigidBodyAPI)),
        "physx_contact_offset_m": (
            float(contact_offset) if contact_offset is not None and math.isfinite(float(contact_offset)) else None
        ),
        "physx_rest_offset_m": (
            float(rest_offset) if rest_offset is not None and math.isfinite(float(rest_offset)) else None
        ),
        "split_y_source_units": float(split_y_source_units),
        "pieces": pieces,
    }


__all__ = [
    "CAP_SHAFT_SPLIT_Y_SOURCE_UNITS",
    "PHYSX_GPU_HULL_VERTEX_LIMIT",
    "author_split_bolt_convex_proxies",
]
