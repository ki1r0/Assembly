from pathlib import Path

import numpy as np
import pytest

from shaft_compound import build_shaft_compound

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics


@pytest.fixture
def synthetic_source(tmp_path):
    source, assets = tmp_path / "source", tmp_path / "assets"
    (source / "assembly").mkdir(parents=True)
    (source / "runtime").mkdir()
    (assets / "parts").mkdir(parents=True)
    (assets / "CAD_edit").mkdir()
    (source / "assembly/shaft_subassembly_steps.yaml").write_text(
        "steps:\n"
        "  - {id: gear, parent: Transfer_Shaft, child: Transfer_Gear, plug: plug_main, socket: socket_gear}\n"
        "  - {id: bearing, parent: Transfer_Shaft, child: Bearing_Transfer, plug: plug_main, socket: socket_gear}\n")
    (source / "runtime/magic_assembly.py").write_text(
        "HARDCODED_FIT_OFFSETS: dict = {"
        "('Transfer_Gear','Transfer_Shaft','plug_main','socket_gear'): "
        "{'translate': (22.1,0,0), 'rotate_xyz': (0,90,0)},"
        "('Bearing_Transfer','Transfer_Shaft','plug_main','socket_gear'): "
        "{'translate': (-50.563,0,0), 'rotate_xyz': (0,0,0)}}\n")
    for file in ("parts/Transfer Shaft.usd", "parts/Transfer Gear.usd",
                 "CAD_edit/Bearing 32205 Inner Assy.usd"):
        stage = Usd.Stage.CreateNew(str(assets / file))
        root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(root)
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.2)
        collision = UsdGeom.Xform.Define(stage, "/World/collisions")
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/collisions/mesh")
        mesh.CreatePointsAttr([(-1,-1,-1), (1,-1,-1), (0,1,-1), (0,0,1)])
        mesh.CreateFaceVertexCountsAttr([3] * 4)
        mesh.CreateFaceVertexIndicesAttr([0,2,1,0,1,3,1,2,3,2,0,3])
        visual = UsdGeom.Mesh.Define(stage, "/World/visual")
        visual.CreatePointsAttr([(100,100,100)])
        if "Shaft" in file:
            UsdGeom.Xform.Define(stage, "/World/socket_gear")
        elif "Gear" in file:
            UsdGeom.Xform.Define(stage, "/World/plug_main").AddTranslateOp().Set((0, 0, 5))
        stage.GetRootLayer().Save()
    return assets, source


def test_compound_preserves_geometry_but_only_one_rigid_body(synthetic_source, tmp_path):
    assets, source = synthetic_source
    stage, info = build_shaft_compound("transfer-shaft-into-base", assets, source)
    prims = list(stage.Traverse())
    assert [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)] == ["/Assembly"]
    assert not any(p.HasAPI(UsdPhysics.MassAPI) or p.IsA(UsdPhysics.Joint) for p in prims)
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 3
    assert all(p.IsA(UsdGeom.Mesh) and p.GetName() == "mesh" for p in colliders)
    assert all(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "sdf" for p in colliders)
    assert UsdGeom.GetStageMetersPerUnit(stage) == .001
    assert len(info["members"]) == 3
    assert info["members"][2]["placement_status"].startswith("provisional")
    assert "mass/COM/inertia" in info["mass_status"]
    assert not any("Casing" in str(p.GetPath()) for p in prims)
    # Nonidentity plug must be inverted before the source fit, not ignored.
    gear = UsdGeom.Xformable(stage.GetPrimAtPath("/Assembly/Transfer_Gear")).GetLocalTransformation()
    expected = (Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(0,0,-5)) *
                Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0),90)))
    expected.SetTranslateOnly(expected.ExtractTranslation() + Gf.Vec3d(22.1,0,0))
    np.testing.assert_allclose(np.asarray(gear), np.asarray(expected), atol=1e-10)
    output = tmp_path / "compound.usda"
    stage.GetRootLayer().Export(str(output))
    reopened = Usd.Stage.Open(str(output))
    assert reopened.GetDefaultPrim().GetPath() == stage.GetDefaultPrim().GetPath()
    assert len([p for p in reopened.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]) == 3


def test_disabled_collision_is_not_silently_reenabled(synthetic_source):
    assets, source = synthetic_source
    stage = Usd.Stage.Open(str(assets / "parts/Transfer Shaft.usd"))
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath("/World/collisions")).CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="no enabled collision"):
        build_shaft_compound("transfer-shaft-into-base", assets, source)


@pytest.mark.parametrize("kind,count", [("output",4), ("transfer",3), ("input",3)])
def test_actual_public_cad_static_build(kind, count):
    source = Path("/home/sunsiliang/Human-AI-Collab")
    if not (source / "assets/parts/Output Shaft.usd").exists():
        pytest.skip("Local public CAD/source checkout unavailable")
    stage, info = build_shaft_compound(kind + "-shaft-into-base", source / "assets", source)
    assert len(info["members"]) == count
    assert sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()) == 1
    assert sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()) == count
    assert info["input_shaft_open_mesh_provisional"] == (kind == "input")
    for member in info["members"]:
        if member["name"].startswith("Bearing_"):
            assert "missing_bearing_plug" in member["placement_status"]
    if kind == "output":
        transforms = {m["name"]: np.array(m["shaft_from_component_row_source_units"])
                      for m in info["members"]}
        assert transforms["Bearing_Output_Top"][3,1] == pytest.approx(51.156502)
        assert transforms["Bearing_Output_Bottom"][3,1] == pytest.approx(-51.156502)
        np.testing.assert_allclose(transforms["Bearing_Output_Top"][:3,:3].T @ [1,0,0], [0,-1,0], atol=1e-12)


def test_unrelated_tasks_and_invalid_scale_are_rejected():
    with pytest.raises(ValueError, match="Only the three"):
        build_shaft_compound("casing-top-to-base", ".", ".")
    with pytest.raises(ValueError, match="scale"):
        build_shaft_compound("input-shaft-into-base", ".", ".", float("nan"))
