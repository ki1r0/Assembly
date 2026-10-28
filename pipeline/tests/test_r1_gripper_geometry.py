import itertools
import json

import numpy as np
import pytest

from r1_gripper_geometry import (
    CollisionPiece, FingerGeometry, R1GripperGeometry, extract_r1_gripper_geometry,
)


def cube():
    return np.array(list(itertools.product((-.01, .01), repeat=3)))


def model():
    fingers = [FingerGeometry(str(i), str(i), np.array([0, sign, 0]), (0, .05),
                              None, None, [CollisionPiece(str(i), str(i), "convexHull", cube())])
               for i, sign in enumerate((-1, 1))]
    return R1GripperGeometry("synthetic", "left", "/wrist", np.array([.0079, 0, .09089]),
                             fingers, [CollisionPiece("palm", "palm", "convexHull", cube() - [0, 0, .04])])


def test_complete_tcp_offset_and_independent_opening_axes():
    geometry = model()
    meshes = dict(geometry.meshes_tcp([.02, .03]))
    np.testing.assert_allclose(meshes["0"], cube() + [-.0079, -.02, -.09089])
    np.testing.assert_allclose(meshes["1"], cube() + [-.0079, .03, -.09089])
    assert geometry.to_dict()["pad_calibration"].startswith("unknown")
    with pytest.raises(ValueError):
        geometry.meshes_tcp([.06, 0])


def test_pressure_projection_excludes_tangential_load_and_uses_each_axis():
    geometry=model()
    assert geometry.closing_pressures([[0,-3,20],[0,4,30]],np.eye(3))==[3,4]
    assert geometry.closing_pressures([[0,0,20],[0,0,30]],np.eye(3))==[0,0]
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    assert geometry.closing_pressures([[3,0,20],[-4,0,30]],rotation)==[3,4]
    with pytest.raises(ValueError):
        geometry.closing_pressures([[0,0,1]],np.eye(3))


def test_json_roundtrip_preserves_geometry_and_helpers():
    geometry = model()
    geometry.fingers[1].mimic_reference = geometry.fingers[0].joint_path
    geometry.fingers[1].mimic_gearing = -1.
    serialized = json.loads(json.dumps(geometry.to_dict()))
    restored = R1GripperGeometry.from_dict(serialized)
    assert restored.to_dict() == geometry.to_dict()
    assert restored.public_summary() == geometry.public_summary()
    for (_, expected), (_, actual) in zip(geometry.meshes_tcp([.02, .03]),
                                          restored.meshes_tcp([.02, .03])):
        np.testing.assert_array_equal(actual, expected)
    points = np.array([[0, -.02, 0], [.1, 0, 0]]) - geometry.tcp_offset_wrist_m
    np.testing.assert_array_equal(restored.collision_mask(points, [.02, .03]),
                                  geometry.collision_mask(points, [.02, .03]))
    args = ([0, 0, 1], np.eye(3), [.02, .03], [0, 0, 1], [0, 0, 0])
    assert restored.support_clearance(*args) == geometry.support_clearance(*args)
    restored.fingers[0].pieces[0].vertices_wrist_zero_m[0, 0] = 9
    assert serialized["fingers"][0]["pieces"][0]["vertices_wrist_zero_m"][0][0] != 9


def test_public_summary_is_compact_robot_only_zero_joint_bounds():
    geometry = model()
    geometry.fingers[0].joint_path = "/R1/left_arm_link6/left_gripper_axis1"
    geometry.fingers[0].body_path = "/R1/left_gripper_link1"
    # Multiple collision pieces must all contribute to the bounds.
    geometry.fingers[0].pieces.append(CollisionPiece("extra", "extra", "convexHull", cube() + [.02, 0, 0]))
    summary = geometry.public_summary()
    assert summary["source"] == "synthetic"
    assert summary["pad_calibration"].startswith("unknown")
    assert summary["tcp_offset_wrist_m"] == [.0079, 0, .09089]
    finger = summary["fingers"][0]
    assert finger["joint_name"] == "left_gripper_axis1"
    assert finger["body_name"] == "left_gripper_link1"
    assert finger["opening_axis_wrist"] == [0, -1, 0]
    assert finger["limits_m"] == [0, .05]
    np.testing.assert_allclose(finger["zero_joint_tcp_bounds_m"],
                               np.array([[-.01, -.01, -.01], [.03, .01, .01]]) - geometry.tcp_offset_wrist_m)
    np.testing.assert_allclose(summary["palm_tcp_bounds_m"],
                               np.array([[-.01, -.01, -.05], [.01, .01, -.03]]) - geometry.tcp_offset_wrist_m)
    encoded = json.dumps(summary)
    assert all(key not in encoded for key in ("vertices", "pieces", "world_from", "object_pose", "points"))


def test_support_clearance_transforms_every_body_and_normalizes_plane():
    geometry = model()
    rotation = np.diag([1., -1., -1.])
    assert geometry.support_clearance([0, 0, 1], rotation, [0, 0], [0, 0, 2], [0, 0, 0]) == pytest.approx(1.08089)
    assert geometry.support_clearance([0, 0, 0], np.eye(3), [0, 0], [0, 0, 1], [0, 0, 0]) == pytest.approx(-.14089)
    with pytest.raises(ValueError):
        geometry.support_clearance([0, 0, 0], np.eye(3), [0, 0], [0, 0, 0], [0, 0, 0])


def test_world_mesh_bounds_rotate_offsets_without_assuming_pad_locations():
    geometry = model()
    rotation = np.diag([1., -1., -1.])
    bounds = geometry.mesh_bounds_world([.3, .2, 1.], rotation, [.02, .03])
    assert len(bounds) == 3
    assert [b["body_kind"] for b in bounds] == ["palm", "finger", "finger"]
    for actual, (_, vertices) in zip(bounds, geometry.meshes_tcp([.02, .03])):
        world = vertices @ rotation.T + [.3, .2, 1.]
        np.testing.assert_allclose(actual["bounds_world_m"], [world.min(0), world.max(0)])
    # Local negative-Z finger geometry is above a downward-oriented TCP.
    assert bounds[1]["bounds_world_m"][0][2] > 1.
    assert "pad" not in bounds[1]
    with pytest.raises(ValueError):
        geometry.mesh_bounds_world([0, 0, 1], np.zeros((3, 3)), [0, 0])


def test_collision_is_conservative_includes_boundary_and_palm():
    geometry = model()
    points = np.array([[0, -.02, 0], [.01, -.02, 0], [0, 0, -.04], [.011, -.02, 0]]) - geometry.tcp_offset_wrist_m
    assert geometry.collision_mask(points, [.02, .03]).tolist() == [True, True, True, False]
    assert geometry.collision_mask(points, [.02, .03], include_palm=False).tolist() == [True, True, False, False]
    assert geometry.collision_mask(points, [.02, .03], margin_m=.002).all()
    with pytest.raises(ValueError):
        geometry.collision_mask(points, [0, 0], margin_m=-.001)


def test_public_usd_joint_frames_ancestor_collision_and_source_units(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, .001)
    UsdGeom.Xform.Define(stage, "/R1")
    wrist = UsdGeom.Xform.Define(stage, "/R1/left_arm_link6")
    wrist.AddTranslateOp().Set((100, 200, 300))

    def mesh(body):
        collision = UsdGeom.Xform.Define(stage, body + "/collisions")
        collision.AddTranslateOp().Set((2, 3, 4))
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexDecomposition")
        shape = UsdGeom.Mesh.Define(stage, body + "/collisions/shape")
        shape.CreatePointsAttr([Gf.Vec3f(*v) for v in cube() * 1000])
        visual = UsdGeom.Mesh.Define(stage, body + "/visual")
        visual.CreatePointsAttr([Gf.Vec3f(*v) for v in cube() * 100000])

    mesh(str(wrist.GetPath()))
    for i in (1, 2):
        body_path = f"/R1/left_gripper_link{i}"
        body = UsdGeom.Xform.Define(stage, body_path)
        # Deliberately unrelated saved pose must not become the zero joint pose.
        body.AddTranslateOp().Set((900, -500, 700))
        body.AddRotateZOp().Set(35)
        mesh(body_path)
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{wrist.GetPath()}/left_gripper_axis{i}")
        joint.CreateBody0Rel().SetTargets([wrist.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(10, 20, 30))
        joint.CreateLocalPos1Attr(Gf.Vec3f(1, 0, 0))
        rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90).GetQuat()
        joint.CreateLocalRot0Attr(Gf.Quatf(rotation))
        joint.CreateLocalRot1Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90).GetQuat()))
        joint.CreateLowerLimitAttr(0)
        joint.CreateUpperLimitAttr(50)
    stage.GetRootLayer().Save()
    geometry = extract_r1_gripper_geometry(path, "left")
    assert len(geometry.palm) == 1
    finger = geometry.fingers[0]
    assert len(finger.pieces) == 1
    assert finger.limits_m == pytest.approx((0, .05))
    np.testing.assert_allclose(finger.opening_axis_wrist, [0, -1, 0], atol=1e-7)
    child_inverse = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    parent_rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    expected = (cube() * 1000 + [1, 3, 4]) @ child_inverse @ parent_rotation + [10, 20, 30]
    np.testing.assert_allclose(finger.pieces[0].vertices_wrist_zero_m, expected * .001, atol=1e-8)
    assert finger.pieces[0].collision_path.endswith("/collisions")
    assert finger.pieces[0].approximation == "convexDecomposition"
