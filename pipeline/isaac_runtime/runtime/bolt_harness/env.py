"""Real Panda bolt scene using Factory control and FORGE wrench processing."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from threading import Lock
from types import SimpleNamespace

import torch
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import axis_angle_from_quat
from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.forge import forge_utils

from .contacts import BoltFingerContact, decode_bolt_finger_contacts
from .env_cfg import (
    BoltInsertionEnvCfg,
    _condition_blocker_geometry,
    make_casing_cfg,
    make_cover_cfg,
)
from .measurement import BOLT_TOP_LOCAL_Y_SOURCE
from runtime.assembly_harness.perception import (
    PublicRGBDFrame,
    public_rgbd_frame_from_camera,
    rgbd_pointcloud_from_frame,
    robot_tcp_feature,
)


class BoltInsertionEnv(FactoryEnv):
    """FactoryEnv derivative with a free CAD bolt and the repository Panda."""

    cfg: BoltInsertionEnvCfg

    def __init__(self, cfg: BoltInsertionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._contact_processing_runtime_state = self._read_contact_processing_runtime_state()
        self._bolt_contact_report_runtime_state = self._read_bolt_contact_report_runtime_state()
        self._subscribe_private_contact_reports()

    @staticmethod
    def _env_prim_path_to_expr(prim_path: str) -> str:
        env_zero = "/World/envs/env_0"
        if not prim_path.startswith(env_zero + "/"):
            raise RuntimeError(f"expected an env_0 descendant prim path, got {prim_path}")
        return "/World/envs/env_.*" + prim_path[len(env_zero) :]

    @staticmethod
    def _flatten_filter_paths(paths: object) -> list[str]:
        if callable(getattr(paths, "tolist", None)):
            paths = paths.tolist()
        if isinstance(paths, str):
            return [paths]
        if isinstance(paths, (tuple, list)):
            return [path for group in paths for path in BoltInsertionEnv._flatten_filter_paths(group)]
        try:
            return [path for group in paths for path in BoltInsertionEnv._flatten_filter_paths(group)]
        except TypeError:
            return [str(paths)]

    def _setup_scene(self) -> None:
        cfg = self.cfg
        self._private_pending_contact_reports: list[dict[str, object]] = []
        self._private_contact_report_lock = Lock()
        self._private_contact_callback_error: str | None = None
        self._private_contact_report_subscription = None
        self._private_contact_callback_invocations = 0
        self._private_contact_header_count = 0
        self._private_contact_point_count = 0
        self._private_contact_event_type_counts: dict[str, int] = {}
        self._private_physx_simulation_interface = None
        self._private_synchronous_contact_report_polls: list[dict[str, object]] = []
        self._private_synchronous_report_last_step_index: int | None = None
        self._private_exact_cap_contact_ticks: dict[
            str, list[tuple[int, tuple[BoltFingerContact, ...]]]
        ] = {
            "left": [],
            "right": [],
        }
        spawn_ground_plane("/World/Ground", GroundPlaneCfg(), translation=(0.0, 0.0, 0.0))

        table_cfg = sim_utils.CuboidCfg(
            size=cfg.table_size_m,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.34, 0.36)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00005, rest_offset=0.0),
        )
        table_cfg.func("/World/envs/env_.*/Table", table_cfg, translation=cfg.table_center_m)

        casing_cfg = make_casing_cfg(cfg)
        casing_cfg.func(
            "/World/envs/env_.*/Casing",
            casing_cfg,
            translation=cfg.assembly_frame_pos_m,
            orientation=cfg.assembly_frame_quat_wxyz,
        )
        cover_cfg = make_cover_cfg(cfg)
        cover_cfg.func(
            "/World/envs/env_.*/Cover",
            cover_cfg,
            translation=cfg.cover_pos_m,
            orientation=cfg.cover_quat_wxyz,
        )
        self._condition_blocker_authoring_info = self._author_condition_blocker()

        self._robot = Articulation(cfg.robot)
        self._robot_gravity_body_flags = self._apply_factory_robot_gravity()
        self._robot_collision_offset_info = self._author_robot_collision_offsets()
        self._robot_joint_physics_info = self._read_robot_joint_physics()
        self._bolt = RigidObject(cfg.bolt)

        self._scene_collision_info = {
            "bolt": self._author_mesh_collision(
                "/World/envs/env_0/Bolt",
                dynamic=True,
                approximation="convexHull",
            ),
            "casing": self._author_mesh_collision(
                "/World/envs/env_0/Casing", dynamic=False, approximation="none"
            ),
            "cover": self._author_mesh_collision(
                "/World/envs/env_0/Cover", dynamic=False, approximation="none"
            ),
        }
        self._author_bolt_split_colliders()
        self._bolt_contact_report_threshold_info = self._author_bolt_contact_report_threshold()
        self._runtime_grip_material_bindings = self._read_runtime_grip_material_bindings()

        self._finger_bolt_sensors = {}
        for side, finger in (("left", "panda_leftfinger"), ("right", "panda_rightfinger")):
            self._finger_bolt_sensors[side] = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{finger}",
                    filter_prim_paths_expr=["/World/envs/env_.*/Bolt"],
                    history_length=1,
                    update_period=0.0,
                    track_contact_points=False,
                )
            )

        # Actor tactile sensors sum every contact, with no object identity filter.
        self._finger_touch_sensors = {}
        for side, finger in (("left", "panda_leftfinger"), ("right", "panda_rightfinger")):
            self._finger_touch_sensors[side] = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{finger}",
                history_length=1, update_period=0.0, track_contact_points=False,
            ))

        self._bolt_contact_filter_specs = {}
        for category, prim_paths in (
            ("cover", self._scene_collision_info["cover"]["mesh_prim_paths"]),
            ("casing", self._scene_collision_info["casing"]["mesh_prim_paths"]),
            ("robot", [item["prim_path"] for item in self._robot_gravity_body_flags]),
        ):
            for prim_path in prim_paths:
                filter_expr = self._env_prim_path_to_expr(prim_path)
                if filter_expr in self._bolt_contact_filter_specs:
                    raise RuntimeError(f"duplicate private bolt contact filter path: {filter_expr}")
                self._bolt_contact_filter_specs[filter_expr] = {
                    "category": category,
                    "source_prim_path": prim_path,
                }
        if not self._bolt_contact_filter_specs:
            raise RuntimeError("private bolt contact view has no fixture or robot collider filters")
        self._bolt_scene_contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Bolt",
                filter_prim_paths_expr=list(self._bolt_contact_filter_specs),
                history_length=1,
                update_period=0.0,
                track_contact_points=True,
                max_contact_data_count_per_prim=cfg.bolt_contact_max_data_count_per_prim,
            )
        )

        self._robot_fixture_contact_filter_specs = {}
        for category, prim_paths in (
            ("table", ["/World/envs/env_0/Table"]),
            ("casing", self._scene_collision_info["casing"]["mesh_prim_paths"]),
            ("cover", self._scene_collision_info["cover"]["mesh_prim_paths"]),
        ):
            for prim_path in prim_paths:
                filter_expr = self._env_prim_path_to_expr(prim_path)
                if filter_expr in self._robot_fixture_contact_filter_specs:
                    raise RuntimeError(f"duplicate robot fixture contact filter path: {filter_expr}")
                self._robot_fixture_contact_filter_specs[filter_expr] = {
                    "category": category,
                    "source_prim_path": prim_path,
                }
        self._robot_fixture_contact_sensors = {}
        for body in self._robot_gravity_body_flags:
            body_path = str(body["prim_path"])
            body_name = body_path.rsplit("/", 1)[-1]
            self._robot_fixture_contact_sensors[body_name] = ContactSensor(
                ContactSensorCfg(
                    prim_path=self._env_prim_path_to_expr(body_path),
                    filter_prim_paths_expr=list(self._robot_fixture_contact_filter_specs),
                    history_length=1,
                    update_period=0.0,
                    track_contact_points=False,
                )
            )

        self._task_rgb_camera = Camera(cfg.task_rgb_camera)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["bolt"] = self._bolt
        for side, sensor in self._finger_bolt_sensors.items():
            self.scene.sensors[f"{side}_finger_bolt_contact"] = sensor
        for side, sensor in self._finger_touch_sensors.items():
            self.scene.sensors[f"{side}_finger_unfiltered_touch"] = sensor
        self.scene.sensors["bolt_private_fixture_robot_contact"] = self._bolt_scene_contact_sensor
        self.scene.sensors["task_rgb_camera"] = self._task_rgb_camera
        for body_name, sensor in self._robot_fixture_contact_sensors.items():
            self.scene.sensors[f"robot_{body_name}_private_fixture_contact"] = sensor
        self._private_condition_collider_evidence = self._read_private_condition_collider_evidence()
        self._private_fixture_mesh_poses = self._read_private_fixture_mesh_poses()

        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._contact_processing_pre_physics_state = self._enable_contact_processing_before_physics_init()

    def _author_condition_blocker(self) -> dict[str, object] | None:
        geometry = _condition_blocker_geometry(self.cfg)
        if geometry is None:
            return None

        import omni.usd
        from pxr import UsdPhysics

        local_position, local_size = geometry
        prim_path = "/World/envs/env_0/Casing/TaskLayerS2Blocker"
        blocker_cfg = sim_utils.CuboidCfg(
            size=local_size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.00005,
                rest_offset=0.0,
            ),
        )
        blocker_cfg.func(
            prim_path,
            blocker_cfg,
            translation=local_position,
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"S2 task-layer blocker was not authored: {prim_path}")
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.GetCollisionEnabledAttr().Set(True)
        return {
            "prim_path": prim_path,
            "local_position_source_units": list(local_position),
            "local_size_source_units": list(local_size),
        }

    def _read_private_condition_collider_evidence(self) -> dict[str, object]:
        """Read back the private condition collider from the composed task stage."""
        evidence: dict[str, object] = {
            "condition": self.cfg.condition,
            "blocker_authored": self._condition_blocker_authoring_info is not None,
            "blocker": None,
        }
        if self._condition_blocker_authoring_info is None:
            return evidence

        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        path = str(self._condition_blocker_authoring_info["prim_path"])
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"S2 blocker collider is missing from the composed stage: {path}")
        collision_enabled = bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        if not collision_enabled:
            raise RuntimeError(f"S2 blocker collision is disabled: {path}")

        dynamic_ancestor = None
        ancestor = prim
        while ancestor and ancestor.IsValid():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                dynamic_ancestor = str(ancestor.GetPath())
                break
            ancestor = ancestor.GetParent()
        if dynamic_ancestor is not None:
            raise RuntimeError(f"S2 blocker must remain fixed; rigid body found at {dynamic_ancestor}")

        matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
        ).ComputeWorldBound(prim).ComputeAlignedRange()
        return {
            **evidence,
            "blocker": {
                **self._condition_blocker_authoring_info,
                "collision_enabled": collision_enabled,
                "fixed_static": True,
                "world_matrix_gf_row_major": [
                    [float(matrix[row][column]) for column in range(4)]
                    for row in range(4)
                ],
                "world_bounds_min_m": [float(value) for value in bounds.GetMin()],
                "world_bounds_max_m": [float(value) for value in bounds.GetMax()],
                "world_bounds_size_m": [float(value) for value in bounds.GetSize()],
            },
        }

    def _enable_contact_processing_before_physics_init(self) -> dict[str, object]:
        """Override SimulationContext's process default for this task before sim.reset()."""
        import carb

        settings = carb.settings.get_settings()
        setting_path = "/physics/disableContactProcessing"
        value_before = settings.get(setting_path)
        settings.set_bool(setting_path, False)
        value_after = settings.get(setting_path)
        if value_after is not False:
            raise RuntimeError(
                f"task contact processing override did not take effect before physics init: {value_after!r}"
            )
        return {
            "setting": setting_path,
            "value_before_task_override": value_before,
            "value_after_task_override": value_after,
            "contact_processing_enabled_before_physics_init": True,
            "effective_before_direct_rl_env_sim_reset": True,
            "physx_cfg_has_disable_contact_processing": hasattr(
                self.cfg.sim.physx, "disable_contact_processing"
            ),
        }

    def _read_contact_processing_runtime_state(self) -> dict[str, object]:
        """Record the setting after Factory's simulator reset and tensor initialization."""
        import carb

        setting_path = "/physics/disableContactProcessing"
        value = carb.settings.get_settings().get(setting_path)
        if value is not False:
            raise RuntimeError(f"runtime contact processing is disabled after sim reset: {value!r}")
        return {
            **self._contact_processing_pre_physics_state,
            "value_after_factory_sim_reset": value,
            "contact_processing_enabled_after_factory_sim_reset": True,
        }

    def _read_private_fixture_mesh_poses(self) -> list[dict[str, object]]:
        """Capture composed fixed-mesh poses for private physics provenance."""
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        poses = []
        for category in ("casing", "cover"):
            for path in self._scene_collision_info[category]["mesh_prim_paths"]:
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    raise RuntimeError(f"fixed fixture mesh disappeared from composed stage: {path}")
                matrix = cache.GetLocalToWorldTransform(prim)
                poses.append(
                    {
                        "category": category,
                        "prim_path": path,
                        "world_matrix_gf_row_major": [
                            [float(matrix[row][column]) for column in range(4)]
                            for row in range(4)
                        ],
                    }
                )
        if not poses:
            raise RuntimeError("private fixture pose trace has no composed casing or cover meshes")
        return poses

    def _author_bolt_contact_report_threshold(self) -> dict[str, object]:
        """Request all bolt contact events without changing collision or effort settings."""
        import omni.usd
        from pxr import PhysxSchema

        stage = omni.usd.get_context().get_stage()
        root_path = "/World/envs/env_0/Bolt"
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"dynamic bolt actor is missing for contact reporting: {root_path}")
        had_api = bool(root.HasAPI(PhysxSchema.PhysxContactReportAPI))
        previous_threshold = None
        if had_api:
            old_attr = PhysxSchema.PhysxContactReportAPI(root).GetThresholdAttr()
            if old_attr.IsValid() and old_attr.Get() is not None:
                previous_threshold = float(old_attr.Get())

        report_api = PhysxSchema.PhysxContactReportAPI.Apply(root)
        threshold_attr = report_api.GetThresholdAttr()
        if not threshold_attr.IsValid():
            threshold_attr = report_api.CreateThresholdAttr()
        threshold_attr.Set(0.0)
        threshold = threshold_attr.Get()
        if threshold is None or float(threshold) != 0.0:
            raise RuntimeError(f"bolt PhysX contact report threshold did not compose as zero: {threshold!r}")
        return {
            "prim_path": root_path,
            "api_previously_present": had_api,
            "previous_threshold": previous_threshold,
            "task_layer_threshold": float(threshold),
        }

    def _subscribe_private_contact_reports(self) -> None:
        """Subscribe to native PhysX reports; callback only copies IDs and contact scalars."""
        import omni.physx

        simulation_interface = omni.physx.get_physx_simulation_interface()
        self._private_physx_simulation_interface = simulation_interface
        subscribe = getattr(simulation_interface, "subscribe_contact_report_events", None)
        if not callable(subscribe):
            raise RuntimeError("installed PhysX simulation interface has no contact-report subscription API")
        self._private_contact_report_subscription = subscribe(self._copy_private_contact_report_event)
        if self._private_contact_report_subscription is None:
            raise RuntimeError("PhysX contact-report subscription returned no subscription handle")

    def _read_bolt_contact_report_runtime_state(self) -> dict[str, object]:
        """Verify report API and threshold on the composed prim after simulator reset."""
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        root_path = "/World/envs/env_0/Bolt"
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(root_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"dynamic bolt actor is missing after simulator reset: {root_path}")
        report_api_applied = bool(prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
        if not report_api_applied:
            raise RuntimeError("PhysxContactReportAPI is not applied to the runtime Bolt prim")
        threshold_attr = PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr()
        if not threshold_attr.IsValid():
            raise RuntimeError("runtime Bolt contact-report threshold attribute is invalid")
        threshold = threshold_attr.Get()
        if threshold is None or float(threshold) != 0.0:
            raise RuntimeError(f"runtime Bolt contact-report threshold is not zero: {threshold!r}")

        rigid_body_api_applied = bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        rigid_body_enabled = None
        if rigid_body_api_applied:
            rigid_body_enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
        if not rigid_body_api_applied or rigid_body_enabled is False:
            raise RuntimeError("runtime Bolt rigid body is not enabled for contact reporting")
        return {
            "prim_path": root_path,
            "checked_after_factory_env_init_and_sim_reset": True,
            "physx_contact_report_api_applied": report_api_applied,
            "threshold_attr_valid": True,
            "threshold": float(threshold),
            "rigid_body_api_applied": rigid_body_api_applied,
            "rigid_body_enabled": rigid_body_enabled,
        }

    @staticmethod
    def _copy_contact_vector(value: object, field_name: str) -> tuple[float, float, float]:
        try:
            values = tuple(float(component) for component in value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"PhysX contact {field_name} is not a three-vector") from exc
        if len(values) != 3:
            raise ValueError(f"PhysX contact {field_name} is not a three-vector")
        return values  # type: ignore[return-value]

    def _copy_contact_report_buffers(self, headers: object, contact_data: object) -> dict[str, object]:
        copied_headers = []
        event_type_counts: dict[str, int] = {}
        for header in headers:  # type: ignore[union-attr]
            event_type = getattr(header.type, "name", None)
            if not isinstance(event_type, str):
                event_type = str(header.type)
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            copied_headers.append(
                {
                    "type": event_type,
                    "actor0": int(header.actor0),
                    "actor1": int(header.actor1),
                    "collider0": int(header.collider0),
                    "collider1": int(header.collider1),
                    "stage_id": int(header.stage_id),
                    "contact_data_offset": int(header.contact_data_offset),
                    "num_contact_data": int(header.num_contact_data),
                }
            )
        copied_data = [
            {
                "position": self._copy_contact_vector(point.position, "position"),
                "normal": self._copy_contact_vector(point.normal, "normal"),
                "impulse": self._copy_contact_vector(point.impulse, "impulse"),
                "separation": float(point.separation),
            }
            for point in contact_data  # type: ignore[union-attr]
        ]
        return {
            "headers": copied_headers,
            "contact_data": copied_data,
            "event_type_counts": event_type_counts,
        }

    def _poll_synchronous_contact_report(self, completed_step_index: int) -> None:
        """Compare the installed synchronous report API with callbacks for one completed tick."""
        from pxr import PhysicsSchemaTools

        simulation_interface = self._private_physx_simulation_interface
        getter = getattr(simulation_interface, "get_contact_report", None)
        if not callable(getter):
            self._private_synchronous_contact_report_polls.append(
                {
                    "completed_physics_step_index": completed_step_index,
                    "available": False,
                    "error": "get_contact_report is unavailable on the installed simulation interface",
                }
            )
            return
        try:
            headers, contact_data = getter()
            copied = self._copy_contact_report_buffers(headers, contact_data)
            synchronous_headers = copied["headers"]
            if not isinstance(synchronous_headers, list):
                raise TypeError("synchronous contact-report headers were not copied as a list")
            decoded_synchronous = [
                {
                    **header,
                    "actor0_path": str(PhysicsSchemaTools.intToSdfPath(header["actor0"])),
                    "actor1_path": str(PhysicsSchemaTools.intToSdfPath(header["actor1"])),
                    "collider0_path": str(PhysicsSchemaTools.intToSdfPath(header["collider0"])),
                    "collider1_path": str(PhysicsSchemaTools.intToSdfPath(header["collider1"])),
                }
                for header in synchronous_headers
            ]
            with self._private_contact_report_lock:
                callback_reports = [dict(report) for report in self._private_pending_contact_reports]
            callback_headers = [
                header
                for report in callback_reports
                for header in report.get("headers", [])
            ]
            callback_signatures = {
                (
                    header.get("type"),
                    header.get("actor0"),
                    header.get("actor1"),
                    header.get("collider0"),
                    header.get("collider1"),
                )
                for header in callback_headers
            }
            synchronous_signatures = [
                (
                    header["type"],
                    header["actor0"],
                    header["actor1"],
                    header["collider0"],
                    header["collider1"],
                )
                for header in synchronous_headers
            ]
            copied["headers"] = decoded_synchronous
            copied["completed_physics_step_index"] = completed_step_index
            copied["available"] = True
            table_root_path = "/World/envs/env_0/Table"
            copied["bolt_table_pair_count"] = sum(
                (
                    header["actor0_path"] == "/World/envs/env_0/Bolt"
                    and (
                        header["actor1_path"] == table_root_path
                        or header["actor1_path"].startswith(table_root_path + "/")
                    )
                )
                or (
                    header["actor1_path"] == "/World/envs/env_0/Bolt"
                    and (
                        header["actor0_path"] == table_root_path
                        or header["actor0_path"].startswith(table_root_path + "/")
                    )
                )
                for header in decoded_synchronous
            )
            copied["subscription_header_count_same_boundary"] = len(callback_headers)
            copied["matching_header_signatures"] = sum(
                signature in callback_signatures for signature in synchronous_signatures
            )
            self._private_synchronous_contact_report_polls.append(copied)
        except Exception as exc:
            self._private_synchronous_contact_report_polls.append(
                {
                    "completed_physics_step_index": completed_step_index,
                    "available": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def _copy_private_contact_report_event(self, headers: object, contact_data: object) -> None:
        """Copy SDK callback buffers; no torch/GPU tensor access occurs in this callback."""
        with self._private_contact_report_lock:
            self._private_contact_callback_invocations += 1
        try:
            copied = self._copy_contact_report_buffers(headers, contact_data)
            copied_headers = copied["headers"]
            copied_data = copied["contact_data"]
            event_type_counts = copied["event_type_counts"]
            with self._private_contact_report_lock:
                self._private_contact_header_count += len(copied_headers)
                self._private_contact_point_count += len(copied_data)
                for event_type, count in event_type_counts.items():
                    self._private_contact_event_type_counts[event_type] = (
                        self._private_contact_event_type_counts.get(event_type, 0) + count
                    )
            if not copied_headers:
                return
            self.append_private_contact_report(copied)
        except Exception as exc:
            with self._private_contact_report_lock:
                if self._private_contact_callback_error is None:
                    self._private_contact_callback_error = f"{type(exc).__name__}: {exc}"

    def _read_robot_joint_physics(self) -> list[dict[str, object]]:
        """Read authored Panda joint friction and drive attributes without editing the USD."""
        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not root or not root.IsValid():
            raise RuntimeError("composed Panda root is missing while reading joint physics")
        rows = []
        for prim in Usd.PrimRange(root):
            attrs = {}
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if any(token in name.lower() for token in ("friction", "damping", "stiffness")):
                    value = attr.Get()
                    if value is not None:
                        attrs[name] = str(value)
            if attrs:
                rows.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "type_name": str(prim.GetTypeName()),
                        "applied_schemas": list(prim.GetAppliedSchemas()),
                        "friction_damping_stiffness_attributes": attrs,
                    }
                )
        return rows

    def _read_runtime_grip_material_bindings(self) -> list[dict[str, object]]:
        """Record composed physics-material bindings on the real grip/contact shapes."""
        import omni.usd
        from pxr import UsdShade

        stage = omni.usd.get_context().get_stage()
        paths = [
            item["prim_path"]
            for item in self._robot_collision_offset_info
            if item["prim_path"].endswith(
                ("/panda_leftfinger/geometry/panda_leftfinger", "/panda_rightfinger/geometry/panda_rightfinger")
            )
        ]
        paths.extend(self._scene_collision_info["bolt"].get("active_collision_prim_paths", []))
        rows = []
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                rows.append({"prim_path": path, "error": "composed collider is missing"})
                continue
            row: dict[str, object] = {
                "prim_path": path,
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "collision_enabled": bool(prim.GetAttribute("physics:collisionEnabled").Get()),
            }
            material_relationships = []
            ancestor = prim
            while ancestor and ancestor.IsValid():
                for relationship in ancestor.GetRelationships():
                    if "material:binding" not in relationship.GetName().lower():
                        continue
                    targets = [str(target) for target in relationship.GetTargets()]
                    target_materials = []
                    for target in relationship.GetTargets():
                        material_prim = stage.GetPrimAtPath(target)
                        if material_prim and material_prim.IsValid():
                            target_materials.append(
                                {
                                    "prim_path": str(target),
                                    "applied_schemas": list(material_prim.GetAppliedSchemas()),
                                    "physics_attributes": {
                                        attr.GetName(): str(attr.Get())
                                        for attr in material_prim.GetAttributes()
                                        if any(
                                            token in attr.GetName().lower()
                                            for token in ("friction", "restitution")
                                        )
                                        and attr.Get() is not None
                                    },
                                }
                            )
                    material_relationships.append(
                        {
                            "authored_on_prim": str(ancestor.GetPath()),
                            "relationship": relationship.GetName(),
                            "targets": targets,
                            "target_materials": target_materials,
                        }
                    )
                ancestor = ancestor.GetParent()
            row["authored_material_relationships"] = material_relationships
            try:
                material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                row["binding_relationship"] = str(relationship.GetPath()) if relationship else None
                row["binding_targets"] = (
                    [str(target) for target in relationship.GetTargets()] if relationship else []
                )
                if material and material.GetPrim().IsValid():
                    material_prim = material.GetPrim()
                    row["material_path"] = str(material_prim.GetPath())
                    row["material_physics_attributes"] = {
                        attr.GetName(): str(attr.Get())
                        for attr in material_prim.GetAttributes()
                        if any(token in attr.GetName().lower() for token in ("friction", "restitution"))
                        and attr.Get() is not None
                    }
                else:
                    row["material_path"] = None
                    row["material_physics_attributes"] = {}
            except Exception as exc:
                row["binding_read_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        return rows

    def _apply_factory_robot_gravity(self) -> list[dict[str, object]]:
        """Apply Factory's robot-only gravity setting on the composed task stage."""
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not root or not root.IsValid():
            raise RuntimeError("composed Panda root is missing while applying Factory gravity settings")
        bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not bodies:
            raise RuntimeError("composed Panda has no rigid bodies for Factory gravity settings")
        flags = []
        for prim in bodies:
            rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_body.GetDisableGravityAttr().Set(True)
            flags.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "disable_gravity": bool(rigid_body.GetDisableGravityAttr().Get()),
                }
            )
        if not all(item["disable_gravity"] for item in flags):
            raise RuntimeError(f"Factory robot gravity application failed: {flags}")
        return flags

    def _author_robot_collision_offsets(self) -> list[dict[str, object]]:
        """Keep all Panda collisions enabled while matching CAD contact margins."""
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not root or not root.IsValid():
            raise RuntimeError("composed Panda root is missing while configuring collision margins")
        # Nested robot geometry can be instanceable; make task-layer overrides editable.
        for _ in range(8):
            instances = [prim for prim in Usd.PrimRange(root) if prim.IsInstance()]
            if not instances:
                break
            for prim in instances:
                prim.SetInstanceable(False)
        colliders = [
            prim for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not colliders:
            visible = [
                f"{prim.GetPath()}<{prim.GetTypeName()}> schemas={list(prim.GetAppliedSchemas())}"
                for prim in Usd.PrimRange(root)
                if prim.IsA(UsdGeom.Gprim)
            ]
            raise RuntimeError(f"composed Panda has no collision shapes to configure; geometry={visible[:40]}")
        info = []
        for prim in colliders:
            collision = UsdPhysics.CollisionAPI(prim)
            enabled = collision.GetCollisionEnabledAttr().Get()
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.GetContactOffsetAttr().Set(0.00005)
            physx_collision.GetRestOffsetAttr().Set(0.0)
            info.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "collision_enabled": bool(enabled),
                    "contact_offset_m": float(physx_collision.GetContactOffsetAttr().Get()),
                    "rest_offset_m": float(physx_collision.GetRestOffsetAttr().Get()),
                }
            )
        if not all(item["collision_enabled"] for item in info):
            raise RuntimeError("Panda USD contains a disabled collision shape; refusing to mask it")
        return info

    def _author_mesh_collision(
        self,
        root_path: str,
        *,
        dynamic: bool,
        approximation: str,
    ) -> dict:
        """Author task-layer colliders; source CAD layers remain untouched."""
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"spawned CAD prim is missing: {root_path}")
        meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError(f"spawned CAD prim has no meshes: {root_path}")

        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and (not dynamic or prim != root):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if dynamic and not root.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(root)
        if dynamic:
            UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Set(False)

        for mesh in meshes:
            collision = UsdPhysics.CollisionAPI.Apply(mesh)
            collision.GetCollisionEnabledAttr().Set(True)
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
            physx_collision.GetContactOffsetAttr().Set(0.00005)
            physx_collision.GetRestOffsetAttr().Set(0.0)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh)
            mesh_collision.GetApproximationAttr().Set(approximation)

        world_bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
        ).ComputeWorldBound(root).ComputeAlignedRange()
        minimum = tuple(float(value) for value in world_bounds.GetMin())
        maximum = tuple(float(value) for value in world_bounds.GetMax())
        size = tuple(float(value) for value in world_bounds.GetSize())
        info = {
            "prim_path": root_path,
            "mesh_count": len(meshes),
            "mesh_prim_paths": [str(mesh.GetPath()) for mesh in meshes],
            "collision_approximation": approximation,
            "fixed_static_mesh": not dynamic,
            "contact_offset_m": 0.00005,
            "world_bounds_min_m": minimum,
            "world_bounds_max_m": maximum,
            "world_bounds_size_m": size,
        }
        expected_bolt_length_m = 2.0 * BOLT_TOP_LOCAL_Y_SOURCE * self.cfg.cad_stage_scale
        if dynamic and abs(max(size) - expected_bolt_length_m) > max(0.0005, 0.05 * expected_bolt_length_m):
            raise RuntimeError(
                f"scaled CAD bolt world bounds are {size} m; expected maximum dimension "
                f"near {expected_bolt_length_m:.6f} m at scale {self.cfg.cad_stage_scale:g}"
            )
        return info

    def _author_bolt_split_colliders(self) -> None:
        """Replace the dynamic bolt's visual-mesh collider with CAD-derived hulls."""
        import omni.usd
        from pxr import Usd, UsdGeom

        from .collision import author_split_bolt_convex_proxies

        root_path = "/World/envs/env_0/Bolt"
        source_mesh_path = f"{root_path}/node_/mesh_"
        stage = omni.usd.get_context().get_stage()
        proxy_info = author_split_bolt_convex_proxies(
            stage,
            bolt_root_path=root_path,
            source_mesh_path=source_mesh_path,
        )
        bounds_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
            useExtentsHint=False,
        )
        composed_bounds = []
        for piece in proxy_info["pieces"]:
            prim = stage.GetPrimAtPath(piece["prim_path"])
            if not prim or not prim.IsValid():
                raise RuntimeError(f"authored bolt collision proxy is missing: {piece['prim_path']}")
            bounds = bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum = tuple(float(value) for value in bounds.GetMin())
            maximum = tuple(float(value) for value in bounds.GetMax())
            size = tuple(float(value) for value in bounds.GetSize())
            piece["composed_world_bounds_min_m"] = minimum
            piece["composed_world_bounds_max_m"] = maximum
            piece["composed_world_bounds_size_m"] = size
            composed_bounds.append(
                {
                    "name": piece["name"],
                    "prim_path": piece["prim_path"],
                    "bounds_min_m": minimum,
                    "bounds_max_m": maximum,
                    "bounds_size_m": size,
                }
            )
        self._scene_collision_info["bolt"].update(
            {
                "collision_approximation": "splitConvexHull",
                "source_visual_collision_enabled": proxy_info["source_collision_enabled"],
                "active_collision_prim_paths": [piece["prim_path"] for piece in proxy_info["pieces"]],
                "split_proxy_info": proxy_info,
                "composed_proxy_world_bounds_m": composed_bounds,
                "proxy_bounds_source": "composed task-layer USD bounds; PhysX cooked bounds are not queried",
            }
        )

    def _init_tensors(self) -> None:
        task_camera = self.scene.sensors["task_rgb_camera"]
        if not task_camera.is_initialized:
            raise RuntimeError("task RGB camera must be initialized before clock precision promotion")
        task_camera._timestamp = task_camera._timestamp.to(dtype=torch.float64)
        task_camera._timestamp_last_update = task_camera._timestamp_last_update.to(dtype=torch.float64)

        names = self._robot.body_names
        required = (self.cfg.wrench_body_name,)
        missing = [name for name in required if name not in names]
        if missing:
            raise RuntimeError(f"local Panda USD is missing required body names: {missing}; found {names}")
        self.wrench_body_idx = names.index(self.cfg.wrench_body_name)
        self._tcp_uses_hand_offset = self.cfg.tcp_body_name not in names
        self.tcp_body_idx = names.index(self.cfg.tcp_body_name) if not self._tcp_uses_hand_offset else self.wrench_body_idx
        self.wrench_api_doc = getattr(
            self._robot.root_physx_view.get_link_incoming_joint_force, "__doc__", ""
        ) or ""
        api_doc = self.wrench_api_doc.lower()
        if "child" not in api_doc or "frame" not in api_doc:
            raise RuntimeError(
                "installed get_link_incoming_joint_force documentation did not confirm its child-frame basis"
            )
        (
            self._wrench_joint_local_pos,
            self._wrench_joint_local_quat,
            self.wrench_joint_prim_path,
        ) = self._read_wrench_child_joint_frame()
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()
        if self._robot.num_joints != 9:
            raise RuntimeError(f"Factory torque controller requires Panda's 7 arm + 2 finger joints; got {self._robot.num_joints}")
        self._finger_joint_ids = [
            index for index, name in enumerate(self._robot.joint_names) if "finger_joint" in name
        ]
        if len(self._finger_joint_ids) != 2:
            raise RuntimeError(f"expected two Panda finger joints, found {self._finger_joint_ids}")
        self._native_finger_effort_limits = self._robot.data.joint_effort_limits[:, self._finger_joint_ids].clone()

        self.ctrl_target_joint_pos = torch.zeros_like(self.joint_pos)
        self.commanded_tcp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.commanded_tcp_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.commanded_tcp_quat[:, 0] = 1.0
        self.actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(self.num_envs, 1)
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(self.num_envs, 1)
        self.task_prop_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(self.num_envs, 1)
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            self.task_prop_gains, rot_deriv_scale=self.cfg.task_rot_deriv_scale
        )
        self.dead_zone_thresholds = None
        self.gripper_target_joint_m = 0.04
        self.fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.fingertip_midpoint_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_jacobian = torch.zeros((self.num_envs, 6, 7), device=self.device)
        self.arm_mass_matrix = torch.zeros((self.num_envs, 7, 7), device=self.device)
        self._wrench_bias_child: torch.Tensor | None = None
        self.wrench_source_frame = "panda_hand incoming joint child frame"
        self.wrench_child_raw = torch.zeros((self.num_envs, 6), device=self.device)
        self.wrench_assembly = torch.zeros_like(self.wrench_child_raw)
        self.wrench_assembly_smooth = torch.zeros_like(self.wrench_assembly)
        self._wrench_raw_history = deque(maxlen=240)
        self._tcp_position_history = deque(maxlen=240)
        self._private_physics_samples: list[dict[str, object]] = []
        self._private_physics_epoch_timestamp_s: float | None = None
        self._private_physics_completed_steps = 0
        self._private_physics_step_in_flight = False
        self._private_physics_last_sample_index: int | None = None
        self._private_pose_provider_last_step_index = 0
        self._last_wrench_sample_timestamp = None
        self.wrench_baseline_stats = None
        self._reset_count = 0
        self._episode_ended = False
        self.motion_safety_armed = False
        self.last_update_timestamp = -1.0
        self.last_velocity_sample_dt_s = 0.0
        self._previous_tcp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_tcp_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.ee_linvel_fd = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_angvel_fd = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_factory_control_snapshot = None

    def _read_wrench_child_joint_frame(self) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Read the real Panda hand joint's local child frame from the composed USD."""
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not robot_root or not robot_root.IsValid():
            raise RuntimeError("composed Panda root is missing while resolving the hand wrench frame")
        matches = []
        for prim in Usd.PrimRange(robot_root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            body1_targets = joint.GetBody1Rel().GetTargets()
            if any(str(path).rstrip("/").endswith("/panda_hand") for path in body1_targets):
                matches.append(joint)
        if len(matches) != 1:
            raise RuntimeError(f"expected one Panda child joint for panda_hand, found {len(matches)}")
        joint = matches[0]
        local_pos = joint.GetLocalPos1Attr().Get()
        local_quat = joint.GetLocalRot1Attr().Get()
        if local_pos is None or local_quat is None:
            raise RuntimeError(f"Panda hand joint has no authored child-frame transform: {joint.GetPath()}")
        local_pos_tensor = torch.tensor([float(value) for value in local_pos], device=self.device)
        local_quat_tensor = torch.tensor(
            [float(local_quat.GetReal()), *(float(value) for value in local_quat.GetImaginary())],
            device=self.device,
        )
        local_quat_tensor /= torch.linalg.vector_norm(local_quat_tensor).clamp_min(1e-8)
        return local_pos_tensor, local_quat_tensor, str(joint.GetPath())

    def _set_default_dynamics_parameters(self) -> None:
        # No stock Factory randomization: retain task-space control gains only.
        self.default_gains = self.task_prop_gains.clone()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        # DirectRLEnv reset only; never call FactoryEnv's held-asset/pre-grasp reset.
        if self._reset_count:
            self._episode_ended = True
            raise RuntimeError(
                "bolt harness is single-episode: refusing an automatic or explicit second reset"
            )
        self._reset_count += 1
        DirectRLEnv._reset_idx(self, env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=env_ids)
        self.ctrl_target_joint_pos[env_ids] = joint_pos

        bolt_state = self._bolt.data.default_root_state[env_ids].clone()
        bolt_state[:, :3] += self.scene.env_origins[env_ids]
        bolt_state[:, 7:] = 0.0
        self._bolt.write_root_pose_to_sim(bolt_state[:, :7], env_ids=env_ids)
        self._bolt.write_root_velocity_to_sim(bolt_state[:, 7:], env_ids=env_ids)
        self._bolt.reset(env_ids)

        self.actions[env_ids] = 0.0
        self._wrench_bias_child = None
        self.wrench_baseline_stats = None
        self.motion_safety_armed = False
        self._wrench_raw_history.clear()
        self._tcp_position_history.clear()
        self._last_wrench_sample_timestamp = None
        self.wrench_assembly[env_ids] = 0.0
        self.wrench_assembly_smooth[env_ids] = 0.0
        self.scene.write_data_to_sim()
        self.sim.forward()
        # Refresh articulation buffers without advancing sensor/FK clocks as if a physics/render tick occurred.
        self.scene.update(dt=0.0)
        hand_idx = self.wrench_body_idx
        hand_pos = self._robot.data.body_pos_w[:, hand_idx]
        hand_quat = self._robot.data.body_quat_w[:, hand_idx]
        if self._tcp_uses_hand_offset:
            local_offset = torch.tensor(self.cfg.tcp_offset_from_hand_local_m, device=self.device).repeat(self.num_envs, 1)
            self._previous_tcp_pos = hand_pos + torch_utils.quat_apply(hand_quat, local_offset) - self.scene.env_origins
            self._previous_tcp_quat = hand_quat.clone()
        else:
            self._previous_tcp_pos = self._robot.data.body_pos_w[:, self.tcp_body_idx] - self.scene.env_origins
            self._previous_tcp_quat = self._robot.data.body_quat_w[:, self.tcp_body_idx].clone()
        self.last_update_timestamp = float(self._robot._data._sim_timestamp)
        self.last_velocity_sample_dt_s = 0.0
        self.last_factory_control_snapshot = None
        self._compute_intermediate_values(self.physics_dt)
        self.commanded_tcp_pos[env_ids] = self.fingertip_midpoint_pos[env_ids]
        self.commanded_tcp_quat[env_ids] = self.fingertip_midpoint_quat[env_ids]
        self._private_physics_samples.clear()
        with self._private_contact_report_lock:
            self._private_pending_contact_reports.clear()
        self._private_exact_cap_contact_ticks = {"left": [], "right": []}
        self._private_physics_epoch_timestamp_s = float(self._robot._data._sim_timestamp)
        self._private_physics_completed_steps = 0
        self._private_physics_step_in_flight = False
        self._private_physics_last_sample_index = None
        self._private_pose_provider_last_step_index = 0
        self._private_synchronous_contact_report_polls.clear()
        self._private_synchronous_report_last_step_index = None

    def _compute_intermediate_values(self, dt: float | None = None) -> None:
        robot_data = self._robot.data
        self.joint_pos = robot_data.joint_pos.clone()
        self.joint_vel = robot_data.joint_vel.clone()
        hand_pos = robot_data.body_pos_w[:, self.wrench_body_idx]
        hand_quat = robot_data.body_quat_w[:, self.wrench_body_idx]
        hand_linvel = robot_data.body_link_lin_vel_w[:, self.wrench_body_idx]
        hand_angvel = robot_data.body_ang_vel_w[:, self.wrench_body_idx]
        if self._tcp_uses_hand_offset:
            offset_local = torch.tensor(self.cfg.tcp_offset_from_hand_local_m, device=self.device).repeat(self.num_envs, 1)
            offset_world = torch_utils.quat_apply(hand_quat, offset_local)
            self.fingertip_midpoint_pos = hand_pos + offset_world - self.scene.env_origins
            self.fingertip_midpoint_quat = hand_quat
            self.fingertip_midpoint_linvel = hand_linvel + torch.cross(hand_angvel, offset_world, dim=-1)
            self.fingertip_midpoint_angvel = hand_angvel
        else:
            self.fingertip_midpoint_pos = robot_data.body_pos_w[:, self.tcp_body_idx] - self.scene.env_origins
            self.fingertip_midpoint_quat = robot_data.body_quat_w[:, self.tcp_body_idx]
            self.fingertip_midpoint_linvel = robot_data.body_lin_vel_w[:, self.tcp_body_idx]
            self.fingertip_midpoint_angvel = robot_data.body_ang_vel_w[:, self.tcp_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()
        jacobian_body_idx = self.wrench_body_idx if self._tcp_uses_hand_offset else self.tcp_body_idx
        tcp_jacobian = jacobians[:, jacobian_body_idx - 1, :6, :7]
        if self._tcp_uses_hand_offset:
            hand_jacobian = tcp_jacobian
            angular_axes = hand_jacobian[:, 3:6, :].transpose(1, 2)
            tcp_offset_world = (self.fingertip_midpoint_pos + self.scene.env_origins) - hand_pos
            linear_shift = torch.cross(angular_axes, tcp_offset_world[:, None, :], dim=-1).transpose(1, 2)
            tcp_jacobian = torch.cat((hand_jacobian[:, :3, :] + linear_shift, hand_jacobian[:, 3:6, :]), dim=1)
        self.fingertip_midpoint_jacobian = tcp_jacobian
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, :7, :7]

        timestamp = float(self._robot._data._sim_timestamp)
        elapsed_dt = timestamp - float(self.last_update_timestamp)
        if elapsed_dt > 1e-9:
            self.ee_linvel_fd = (self.fingertip_midpoint_pos - self._previous_tcp_pos) / elapsed_dt
            relative_quat = torch_utils.quat_mul(
                self.fingertip_midpoint_quat,
                torch_utils.quat_conjugate(self._previous_tcp_quat),
            )
            relative_quat *= torch.sign(relative_quat[:, :1])
            self.ee_angvel_fd = axis_angle_from_quat(relative_quat) / elapsed_dt
            self._previous_tcp_pos = self.fingertip_midpoint_pos.clone()
            self._previous_tcp_quat = self.fingertip_midpoint_quat.clone()
            self.last_update_timestamp = timestamp
            self.last_velocity_sample_dt_s = elapsed_dt

        incoming = self._robot.root_physx_view.get_link_incoming_joint_force()
        incoming = incoming.reshape(self.num_envs, -1, 6)
        self.wrench_child_raw = incoming[:, self.wrench_body_idx]
        corrected = (
            self.wrench_child_raw
            if self._wrench_bias_child is None
            else self.wrench_child_raw - self._wrench_bias_child
        )

        joint_local_pos = self._wrench_joint_local_pos.repeat(self.num_envs, 1)
        joint_local_quat = self._wrench_joint_local_quat.repeat(self.num_envs, 1)
        source_pos = hand_pos + torch_utils.quat_apply(hand_quat, joint_local_pos)
        source_quat = torch_utils.quat_mul(hand_quat, joint_local_quat)
        frame_quat = torch.tensor(self.cfg.assembly_frame_quat_wxyz, device=self.device).repeat(self.num_envs, 1)
        frame_pos = torch.tensor(self.cfg.assembly_frame_pos_m, device=self.device).repeat(self.num_envs, 1)
        frame_pos = frame_pos + self.scene.env_origins
        force, torque = forge_utils.change_FT_frame(
            corrected[:, :3],
            corrected[:, 3:6],
            (source_quat, source_pos),
            (frame_quat, frame_pos),
        )
        self.wrench_assembly = torch.cat((force, torque), dim=-1)
        alpha = self.cfg.wrench_smoothing_factor
        self.wrench_assembly_smooth.mul_(1.0 - alpha).add_(self.wrench_assembly, alpha=alpha)

        sample_timestamp = self._robot._data._sim_timestamp
        if sample_timestamp != self._last_wrench_sample_timestamp:
            self._wrench_raw_history.append(self.wrench_child_raw.detach().clone())
            self._tcp_position_history.append(self.fingertip_midpoint_pos.detach().clone())
            self._last_wrench_sample_timestamp = sample_timestamp

    def _pre_physics_step(self, action: torch.Tensor) -> None:
        if self._episode_ended:
            raise RuntimeError("bolt harness episode ended; refusing further physics steps")
        if action.shape != (self.num_envs, 6):
            raise ValueError(f"expected actions shaped {(self.num_envs, 6)}, received {tuple(action.shape)}")
        self.actions = action.to(self.device).clamp(-1.0, 1.0)
        relative_quat = torch_utils.quat_mul(
            self.commanded_tcp_quat,
            torch_utils.quat_conjugate(self.fingertip_midpoint_quat),
        )
        relative_quat *= torch.sign(relative_quat[:, :1])
        tracking_error_m = torch.linalg.vector_norm(
            self.commanded_tcp_pos - self.fingertip_midpoint_pos, dim=-1
        )
        orientation_error_rad = torch.linalg.vector_norm(axis_angle_from_quat(relative_quat), dim=-1)
        linear_speed_mps = torch.linalg.vector_norm(self.fingertip_midpoint_linvel, dim=-1)
        angular_speed_radps = torch.linalg.vector_norm(self.fingertip_midpoint_angvel, dim=-1)
        unsafe = (
            (tracking_error_m > self.cfg.tcp_tracking_stop_error_m)
            | (orientation_error_rad > self.cfg.tcp_tracking_stop_orientation_rad)
        )
        if self.motion_safety_armed:
            unsafe |= (linear_speed_mps > self.cfg.tcp_linear_speed_stop_mps) | (
                angular_speed_radps > self.cfg.tcp_angular_speed_stop_radps
            )
        elif bool(self.actions.abs().max() > 0.0):
            self._episode_ended = True
            raise RuntimeError("Panda is settling; non-zero TCP commands are disabled until motion safety arms")
        if bool(unsafe.any()):
            self._episode_ended = True
            raise RuntimeError(
                "TCP motion safety stop: "
                f"tracking={tracking_error_m.max().item():.4f} m, "
                f"orientation_error={orientation_error_rad.max().item():.4f} rad, "
                f"linear_speed={linear_speed_mps.max().item():.4f} m/s, "
                f"angular_speed={angular_speed_radps.max().item():.4f} rad/s"
            )

        position_delta = self.actions[:, :3] * torch.tensor(
            self.cfg.ctrl.pos_action_threshold, device=self.device
        )
        rotation_delta = self.actions[:, 3:] * torch.tensor(
            self.cfg.ctrl.rot_action_threshold, device=self.device
        )
        angle = torch.linalg.vector_norm(rotation_delta, dim=-1)
        axis = rotation_delta / angle.clamp_min(1e-8).unsqueeze(-1)
        delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        delta_quat = torch.where(
            (angle > 1e-8).unsqueeze(-1),
            delta_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.commanded_tcp_pos += position_delta
        self.commanded_tcp_quat = torch_utils.quat_mul(delta_quat, self.commanded_tcp_quat)

    def _apply_action(self) -> None:
        if self._private_physics_step_in_flight:
            self._private_physics_completed_steps += 1
            self._private_physics_step_in_flight = False
        self._capture_private_native_physics_state("pre_physics_step")
        self._compute_intermediate_values(self.physics_dt)
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=self.commanded_tcp_pos,
            ctrl_target_fingertip_midpoint_quat=self.commanded_tcp_quat,
            ctrl_target_gripper_dof_pos=self.gripper_target_joint_m,
        )
        self.last_factory_control_snapshot = {
            "sim_timestamp_s": float(self._robot._data._sim_timestamp),
            "velocity_sample_dt_s": self.last_velocity_sample_dt_s,
            "joint7_position_rad": self.joint_pos[:, 6].detach().clone(),
            "joint7_velocity_rad_s": self.joint_vel[:, 6].detach().clone(),
            "tcp_angular_velocity_world_rad_s": self.fingertip_midpoint_angvel.detach().clone(),
            "tcp_angular_velocity_fd_rad_s": self.ee_angvel_fd.detach().clone(),
            "task_prop_gains": self.task_prop_gains.detach().clone(),
            "task_deriv_gains": self.task_deriv_gains.detach().clone(),
            "task_wrench_N_Nm": self.applied_wrench.detach().clone(),
            "joint_torque_Nm": self.joint_torque.detach().clone(),
        }
        self._private_physics_step_in_flight = True

    def get_factory_control_snapshot(self) -> dict[str, object] | None:
        """Return the most recent pre-physics Factory torque input/output sample."""
        if self.last_factory_control_snapshot is None:
            return None
        return {
            key: value[0].detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
            for key, value in self.last_factory_control_snapshot.items()
        }

    def _capture_private_native_physics_state(self, phase: str) -> None:
        """Sample the native bolt state at one completed-physics-tick boundary."""
        if self._private_physics_epoch_timestamp_s is None:
            raise RuntimeError("private native physics trace has no reset epoch")
        tick_index = self._private_physics_completed_steps
        if tick_index > 0:
            completed_step_index = tick_index - 1
            if completed_step_index != self._private_synchronous_report_last_step_index:
                self._poll_synchronous_contact_report(completed_step_index)
                self._private_synchronous_report_last_step_index = completed_step_index
        with self._private_contact_report_lock:
            pending_reports = self._private_pending_contact_reports
            self._private_pending_contact_reports = []
            callback_error = self._private_contact_callback_error
        if callback_error is not None:
            raise RuntimeError(f"PhysX contact-report callback failed: {callback_error}")
        exact_contacts = self._decode_private_bolt_finger_contact_reports(pending_reports)
        if self._private_physics_last_sample_index == tick_index:
            if pending_reports:
                self._private_physics_samples[-1]["contact_reports"].extend(
                    pending_reports
                )
            self._append_decoded_contacts_to_sample(
                self._private_physics_samples[-1], exact_contacts, tick_index, new_tick=False
            )
            return

        view = self._bolt.root_physx_view
        transforms = view.get_transforms()
        velocities = view.get_velocities()
        if transforms.ndim != 2 or transforms.shape[0] != 1 or transforms.shape[1] != 7:
            raise RuntimeError(f"native bolt transforms have unexpected shape {tuple(transforms.shape)}")
        if velocities.ndim != 2 or velocities.shape[0] != 1 or velocities.shape[1] != 6:
            raise RuntimeError(f"native bolt velocities have unexpected shape {tuple(velocities.shape)}")
        if not bool(torch.isfinite(transforms).all()) or not bool(torch.isfinite(velocities).all()):
            raise RuntimeError("native bolt physics view returned non-finite pose or velocity")

        self._private_physics_samples.append(
            {
                "physics_step_index": tick_index,
                "sim_timestamp_s": self._private_physics_epoch_timestamp_s
                + tick_index * float(self.physics_dt),
                "capture_phase": phase,
                "native_root_transform_raw": transforms[0].detach().cpu().tolist(),
                "native_root_velocity_raw": velocities[0].detach().cpu().tolist(),
                "contact_reports": pending_reports,
                "decoded_exact_cap_finger_contacts": {"left": [], "right": []},
            }
        )
        self._append_decoded_contacts_to_sample(
            self._private_physics_samples[-1], exact_contacts, tick_index, new_tick=True
        )
        self._private_physics_last_sample_index = tick_index

    def _decode_private_bolt_finger_contact_reports(
        self, reports: list[dict[str, object]]
    ) -> dict[str, tuple[BoltFingerContact, ...]]:
        """Decode copied SDK data outside the callback, on the simulation thread."""
        from pxr import PhysicsSchemaTools

        decoded: dict[str, list[BoltFingerContact]] = {"left": [], "right": []}
        for report in reports:
            raw_headers = report["headers"]
            raw_points = report["contact_data"]
            if not isinstance(raw_headers, list) or not isinstance(raw_points, list):
                raise TypeError("copied PhysX callback batch has invalid header/contact arrays")
            report["decoded_header_paths"] = [
                {
                    key: str(PhysicsSchemaTools.intToSdfPath(header[key]))
                    for key in ("actor0", "actor1", "collider0", "collider1")
                }
                for header in raw_headers
            ]
            headers = [SimpleNamespace(**header) for header in raw_headers]
            points = [SimpleNamespace(**point) for point in raw_points]
            batch = decode_bolt_finger_contacts(
                headers,
                points,
                dt_s=float(self.physics_dt),
                path_decoder=PhysicsSchemaTools.intToSdfPath,
            )
            for side in decoded:
                decoded[side].extend(batch[side])
        return {side: tuple(records) for side, records in decoded.items()}

    def _append_decoded_contacts_to_sample(
        self,
        sample: dict[str, object],
        contacts: dict[str, tuple[BoltFingerContact, ...]],
        tick_index: int,
        *,
        new_tick: bool,
    ) -> None:
        stored = sample["decoded_exact_cap_finger_contacts"]
        if not isinstance(stored, dict):
            raise TypeError("native physics sample has malformed exact-contact storage")
        for side, records in contacts.items():
            side_ticks = self._private_exact_cap_contact_ticks[side]
            if new_tick:
                if tick_index > 0:
                    side_ticks.append((tick_index, records))
            elif side_ticks and side_ticks[-1][0] == tick_index:
                previous_tick, previous_records = side_ticks[-1]
                side_ticks[-1] = (previous_tick, previous_records + records)
            elif tick_index > 0 and records:
                side_ticks.append((tick_index, records))
            stored[side].extend(asdict(contact) for contact in records)

    def get_private_bolt_cap_finger_contacts(
        self,
    ) -> dict[str, tuple[BoltFingerContact, ...]]:
        """Consume per-physics-tick exact cap/finger contact batches."""
        result = {
            side: tuple(records for _, records in ticks)
            for side, ticks in self._private_exact_cap_contact_ticks.items()
        }
        self._private_exact_cap_contact_ticks = {"left": [], "right": []}
        return result

    def finalize_private_physics_trace(self) -> dict[str, object]:
        """Capture the final completed native state and return private-only evidence."""
        interrupted_tick_in_flight = self._private_physics_step_in_flight
        unassigned_contact_reports = []
        if not interrupted_tick_in_flight:
            self._capture_private_native_physics_state("final_completed_physics_tick")
        else:
            with self._private_contact_report_lock:
                unassigned_contact_reports = self._private_pending_contact_reports
                self._private_pending_contact_reports = []
        transform_doc = getattr(self._bolt.root_physx_view.get_transforms, "__doc__", "") or ""
        velocity_doc = getattr(self._bolt.root_physx_view.get_velocities, "__doc__", "") or ""
        return {
            "schema": "bolt-native-physics-trace-v1",
            "privacy": "private simulator diagnostics; never include in agent/model observations",
            "native_state_source": {
                "transform": "RigidObject.root_physx_view.get_transforms() raw 7-vector",
                "velocity": "RigidObject.root_physx_view.get_velocities() raw 6-vector",
                "interpretation": "values are preserved as returned by the installed PhysX view",
                "transform_api_doc": transform_doc,
                "velocity_api_doc": velocity_doc,
            },
            "time_basis": {
                "reset_epoch_sim_timestamp_s": self._private_physics_epoch_timestamp_s,
                "physics_dt_s": float(self.physics_dt),
                "sim_timestamp_s_formula": "reset_epoch + completed_physics_step_index * physics_dt",
                "sample_semantics": "native state immediately before the next physics step, after the preceding completed tick",
            },
            "fixed_fixture_mesh_poses_world": self._private_fixture_mesh_poses,
            "condition_collider_evidence": self._private_condition_collider_evidence,
            "contact_reporting": {
                "source": "installed PhysX callback and synchronous get_contact_report query",
                "contact_processing_setting": self._contact_processing_runtime_state,
                "threshold": self._bolt_contact_report_threshold_info,
                "runtime_prim_readback": self._bolt_contact_report_runtime_state,
                "subscription_retained": self._private_contact_report_subscription is not None,
                "callback_invocations": self._private_contact_callback_invocations,
                "callback_header_count": self._private_contact_header_count,
                "callback_contact_point_count": self._private_contact_point_count,
                "callback_event_type_counts": dict(self._private_contact_event_type_counts),
                "synchronous_contact_report_poll_count": len(
                    self._private_synchronous_contact_report_polls
                ),
                "synchronous_nonempty_contact_report_poll_count": sum(
                    int(poll.get("available") is True and bool(poll.get("headers")))
                    for poll in self._private_synchronous_contact_report_polls
                ),
                "synchronous_contact_report_polls": list(
                    self._private_synchronous_contact_report_polls
                ),
                "raw_callback_batches": sum(
                    len(sample["contact_reports"]) for sample in self._private_physics_samples
                ),
                "decoded_cap_finger_contact_points": sum(
                    len(sample["decoded_exact_cap_finger_contacts"][side])
                    for sample in self._private_physics_samples
                    for side in ("left", "right")
                ),
            },
            "samples": list(self._private_physics_samples),
            "sample_count": len(self._private_physics_samples),
            "completed_physics_steps": self._private_physics_completed_steps,
            "final_completed_tick_sample_captured": not interrupted_tick_in_flight,
            "in_flight_tick_not_counted_as_completed": interrupted_tick_in_flight,
            "unassigned_contact_reports_at_interrupted_boundary": unassigned_contact_reports,
        }

    def append_private_contact_report(self, report: dict[str, object]) -> None:
        """Receive an already-copied, CPU-only raw PhysX contact event from the callback."""
        if not isinstance(report, dict):
            raise TypeError("private contact report callback must provide a copied dictionary")
        with self._private_contact_report_lock:
            self._private_pending_contact_reports.append(dict(report))

    def _get_observations(self) -> dict[str, torch.Tensor]:
        policy = torch.cat(
            (
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                self.ee_linvel_fd,
                self.ee_angvel_fd,
                self.wrench_assembly_smooth[:, :3],
            ),
            dim=-1,
        )
        return {"policy": policy}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values(dt=self.physics_dt)
        if self._private_physics_step_in_flight:
            self._private_physics_completed_steps += 1
            self._private_physics_step_in_flight = False
        self._capture_private_native_physics_state("post_env_step")
        if not self.motion_safety_armed:
            linear_speed_mps = torch.linalg.vector_norm(self.fingertip_midpoint_linvel, dim=-1)
            angular_speed_radps = torch.linalg.vector_norm(self.fingertip_midpoint_angvel, dim=-1)
            self.motion_safety_armed = bool(
                (linear_speed_mps <= self.cfg.tcp_linear_speed_stop_mps).all()
                and (angular_speed_radps <= self.cfg.tcp_angular_speed_stop_radps).all()
            )
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(timeout), timeout

    def public_wrench(self) -> torch.Tensor:
        """FORGE-transformed real hand wrench; zero reference is explicit and measured."""
        return self.wrench_assembly_smooth.clone()

    def get_unfiltered_finger_contact(self) -> dict[str, torch.Tensor]:
        """Actor touch: total force on each finger, never conditioned on Bolt identity."""
        forces = {side: torch.linalg.vector_norm(sensor.data.net_forces_w[:, 0, :], dim=-1)
                  for side, sensor in self._finger_touch_sensors.items()}
        threshold = self.cfg.grasp_contact_min_force_n
        return {"left_force_n": forces["left"].clone(), "right_force_n": forces["right"].clone(),
                "left_contact": forces["left"] >= threshold,
                "right_contact": forces["right"] >= threshold}

    def get_bilateral_grasp_contact(self) -> dict[str, torch.Tensor]:
        """Return filtered finger-to-bolt forces; this is contact evidence, not a hold verdict."""
        force_by_side = {}
        for side, sensor in self._finger_bolt_sensors.items():
            matrix = sensor.data.force_matrix_w
            if matrix is None or matrix.ndim != 4 or matrix.shape[2] < 1:
                raise RuntimeError(f"{side} finger sensor has no filtered bolt force matrix")
            force_by_side[side] = torch.linalg.vector_norm(matrix[:, 0, 0, :], dim=-1)
        threshold = self.cfg.grasp_contact_min_force_n
        left = force_by_side["left"]
        right = force_by_side["right"]
        return {
            "left_force_n": left.clone(),
            "right_force_n": right.clone(),
            "left_contact": left >= threshold,
            "right_contact": right >= threshold,
            "bilateral_contact": (left >= threshold) & (right >= threshold),
        }

    def get_private_bolt_contact_source(self) -> dict[str, object]:
        """Expose raw filtered PhysX contacts to the evaluator, never to policy observations."""
        sensor = self._bolt_scene_contact_sensor
        if not sensor.is_initialized:
            raise RuntimeError("private bolt contact sensor is not initialized")
        view = sensor.contact_physx_view
        filter_paths = tuple(self._flatten_filter_paths(view.filter_paths))
        if len(filter_paths) != int(view.filter_count):
            raise RuntimeError(
                f"PhysX contact filter path/count mismatch: {len(filter_paths)} paths, "
                f"{int(view.filter_count)} filters"
            )
        filter_map = []
        for filter_index, path in enumerate(filter_paths):
            expression = (
                self._env_prim_path_to_expr(path)
                if path.startswith("/World/envs/env_0/")
                else path
            )
            spec = self._bolt_contact_filter_specs.get(expression)
            if spec is None:
                raise RuntimeError(f"unmapped private bolt contact filter path: {path}")
            filter_map.append(
                {
                    "filter_index": filter_index,
                    "filter_prim_path": path,
                    "category": spec["category"],
                    "source_prim_path": spec["source_prim_path"],
                }
            )
        capacity = (
            int(sensor.cfg.max_contact_data_count_per_prim)
            * int(sensor.num_bodies)
            * int(sensor.num_instances)
        )
        source: dict[str, object] = {
            "contact_physx_view": view,
            "filter_map": filter_map,
            "dt_s": float(self.physics_dt),
            "sensor_body_names": list(sensor.body_names),
            "max_contact_data_count_per_prim": int(sensor.cfg.max_contact_data_count_per_prim),
            "contact_data_capacity": capacity,
            "exact_cap_finger_contact_provider": self.get_private_bolt_cap_finger_contacts,
        }
        pose_tolerances = self.cfg.pose_window_tolerances
        if pose_tolerances is not None:
            source.update(
                {
                    "native_pose_sample_provider": self.get_private_native_pose_samples,
                    "native_pose_dt_s": float(self.physics_dt),
                    "pose_window_tolerances": dict(pose_tolerances),
                }
            )
        return source

    def get_private_native_pose_samples(self) -> tuple[dict[str, object], ...]:
        """Drain contiguous completed native bolt rows through the current env timestamp."""
        if self._private_physics_step_in_flight:
            raise RuntimeError("cannot drain native pose samples while a physics tick is in flight")
        if self._private_physics_epoch_timestamp_s is None:
            raise RuntimeError("private native physics trace has no reset epoch")

        cursor = self._private_pose_provider_last_step_index
        completed = self._private_physics_completed_steps
        if completed < cursor:
            raise RuntimeError("native pose sample cursor is ahead of completed physics state")

        rows_by_index: dict[int, dict[str, object]] = {}
        for row in self._private_physics_samples:
            index = row.get("physics_step_index")
            if type(index) is not int:
                raise RuntimeError("native physics trace contains a malformed physics-step index")
            if index in rows_by_index:
                raise RuntimeError(f"native physics trace duplicated completed tick {index}")
            rows_by_index[index] = row
        if cursor not in rows_by_index:
            raise RuntimeError(f"native physics trace is missing provider cursor tick {cursor}")

        expected_indices = tuple(range(cursor + 1, completed + 1))
        if not expected_indices:
            return ()
        missing = [index for index in expected_indices if index not in rows_by_index]
        if missing:
            raise RuntimeError(f"native pose trace has missing completed ticks: {missing[:8]}")
        rows = tuple(rows_by_index[index] for index in expected_indices)
        dt_s = float(self.physics_dt)
        timestamp_tolerance_s = max(1e-6, dt_s * 1e-3)
        previous_time = float(rows_by_index[cursor]["sim_timestamp_s"])
        for index, row in zip(expected_indices, rows):
            sample_time = float(row["sim_timestamp_s"])
            if abs((sample_time - previous_time) - dt_s) > timestamp_tolerance_s:
                raise RuntimeError(
                    f"native pose tick {index} timestamp is not one physics dt after its predecessor"
                )
            transform = row.get("native_root_transform_raw")
            velocity = row.get("native_root_velocity_raw")
            if not isinstance(transform, (list, tuple)) or len(transform) != 7:
                raise RuntimeError(f"native pose tick {index} has a malformed root transform")
            if not isinstance(velocity, (list, tuple)) or len(velocity) != 6:
                raise RuntimeError(f"native pose tick {index} has a malformed root velocity")
            previous_time = sample_time

        env_time = float(self._robot._data._sim_timestamp)
        if abs(previous_time - env_time) > timestamp_tolerance_s:
            raise RuntimeError(
                "completed native pose rows do not end at the current environment sample time "
                f"(native={previous_time:.9f}, env={env_time:.9f})"
            )
        self._private_pose_provider_last_step_index = completed
        return rows

    def get_private_robot_fixture_contact_source(self) -> dict[str, object]:
        """Expose real filtered arm-to-fixture contact forces for diagnostics only."""
        sensors = self._robot_fixture_contact_sensors
        if not sensors or any(not sensor.is_initialized for sensor in sensors.values()):
            raise RuntimeError("private robot fixture contact sensors are not initialized")
        filter_map = None
        for sensor in sensors.values():
            view = sensor.contact_physx_view
            paths = tuple(self._flatten_filter_paths(view.filter_paths))
            if len(paths) != int(view.filter_count):
                raise RuntimeError(
                    f"robot fixture filter path/count mismatch: {len(paths)} paths, "
                    f"{int(view.filter_count)} filters"
                )
            mapped = []
            for filter_index, path in enumerate(paths):
                expression = self._env_prim_path_to_expr(path) if path.startswith("/World/envs/env_0/") else path
                spec = self._robot_fixture_contact_filter_specs.get(expression)
                if spec is None:
                    raise RuntimeError(f"unmapped robot fixture contact filter path: {path}")
                mapped.append(
                    {
                        "filter_index": filter_index,
                        "filter_prim_path": path,
                        "category": spec["category"],
                        "source_prim_path": spec["source_prim_path"],
                    }
                )
            if filter_map is None:
                filter_map = mapped
            elif mapped != filter_map:
                raise RuntimeError("robot fixture contact filter order differs between Panda bodies")
        return {"sensors": sensors, "filter_map": filter_map}

    def set_stationary_wrench_baseline(
        self, *, stationary: bool, bilateral_contacts_clear: bool
    ) -> dict[str, object]:
        """Set a zero reference only after a caller-verified quiet no-contact window."""
        if not stationary or not bilateral_contacts_clear:
            raise RuntimeError("wrench zero reference requires stable TCP/joints and clear bilateral contacts")
        sample_count = self.cfg.wrench_stationary_sample_count
        if len(self._wrench_raw_history) < sample_count:
            raise RuntimeError(f"need {sample_count} child-frame samples; have {len(self._wrench_raw_history)}")
        samples = torch.stack(list(self._wrench_raw_history)[-sample_count:], dim=0)
        if not bool(torch.isfinite(samples).all()):
            raise RuntimeError("stationary child-frame wrench samples contain non-finite values")
        self._wrench_bias_child = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=False)
        self.wrench_assembly_smooth.zero_()
        self.wrench_baseline_stats = {
            "method": "mean stationary no-contact incoming wrench in the child joint frame",
            "sample_count": sample_count,
            "mean_child_frame_N_Nm": self._wrench_bias_child[0].detach().cpu().tolist(),
            "std_child_frame_N_Nm": std[0].detach().cpu().tolist(),
            "stationary_zero_reference_only": True,
            "force_sign_and_load_calibration_performed": False,
        }
        return dict(self.wrench_baseline_stats)

    def get_public_executor_state(self) -> dict[str, object]:
        """Measured state contract consumed by BoltSkillExecutor."""
        contacts = self.get_unfiltered_finger_contact()
        width = self.joint_pos[:, self._finger_joint_ids].clamp_min(0.0).sum(dim=-1)
        tcp_pose = torch.cat((self.fingertip_midpoint_pos, self.fingertip_midpoint_quat), dim=-1)
        commanded_tcp_pose = torch.cat((self.commanded_tcp_pos, self.commanded_tcp_quat), dim=-1)
        tcp_velocity = torch.cat((self.fingertip_midpoint_linvel, self.fingertip_midpoint_angvel), dim=-1)
        relative_quat = torch_utils.quat_mul(
            self.commanded_tcp_quat,
            torch_utils.quat_conjugate(self.fingertip_midpoint_quat),
        )
        relative_quat *= torch.sign(relative_quat[:, :1])
        return {
            "tcp_pose": tcp_pose.clone(),
            "commanded_tcp_pose": commanded_tcp_pose.clone(),
            "tcp_tracking_error_m": torch.linalg.vector_norm(
                self.commanded_tcp_pos - self.fingertip_midpoint_pos, dim=-1
            ).clone(),
            "tcp_orientation_tracking_error_rad": torch.linalg.vector_norm(
                axis_angle_from_quat(relative_quat), dim=-1
            ).clone(),
            "motion_safety_armed": self.motion_safety_armed,
            "tcp_velocity": tcp_velocity.clone(),
            "joint_positions": self.joint_pos.clone(),
            "joint_velocities": self.joint_vel.clone(),
            "gripper_width_m": float(width[0].item()),
            # Legacy key retained for executor compatibility; values are unfiltered touch.
            "finger_bolt_contacts": (
                bool(contacts["left_contact"][0].item()),
                bool(contacts["right_contact"][0].item()),
            ),
        }

    def get_public_rgbd_frame(self) -> PublicRGBDFrame:
        """Return only rendered camera pixels and camera-owned calibration."""
        camera = self.scene.sensors.get("task_rgb_camera")
        if camera is None:
            raise RuntimeError("task_rgb_camera is missing from env.scene.sensors")
        return public_rgbd_frame_from_camera(camera)

    def get_public_rgbd_pointcloud(
        self,
        frame: PublicRGBDFrame,
        *,
        observation_id: str,
        stride: int = 6,
        max_points: int = 30_000,
    ) -> dict[str, object]:
        """Return raw camera points for adapter-owned robot-link self-filtering."""
        return rgbd_pointcloud_from_frame(
            frame,
            observation_id=observation_id,
            stride=stride,
            max_points=max_points,
        )

    def get_public_assembly_state(self) -> dict[str, object]:
        """Return public robot sensors and FK only, with camera/state clocks separated."""
        measured = self.get_public_executor_state()
        body_names = self._robot.body_names
        left_name, right_name = "panda_leftfinger", "panda_rightfinger"
        if left_name not in body_names or right_name not in body_names:
            raise RuntimeError("Panda finger links required for public gripper FK are unavailable")
        left_position = self._robot.data.body_link_pos_w[0, body_names.index(left_name)]
        right_position = self._robot.data.body_link_pos_w[0, body_names.index(right_name)]
        opening_axis = right_position - left_position
        if torch.linalg.vector_norm(opening_axis) < 1e-8:
            raise RuntimeError("Panda finger FK produced a degenerate opening axis")
        tcp_pose = measured["tcp_pose"][0].clone()
        tcp_pose[:3] += self.scene.env_origins[0]
        width = float(measured["gripper_width_m"])
        wrench = self.wrench_assembly_smooth[0]
        return {
            "joint_positions": measured["joint_positions"][0].detach().cpu().tolist(),
            "joint_velocities": measured["joint_velocities"][0].detach().cpu().tolist(),
            "tcp_pose": tcp_pose.detach().cpu().tolist(),
            "tcp_velocity": measured["tcp_velocity"][0].detach().cpu().tolist(),
            "gripper_width": width,
            "gripper_width_m": width,
            "gripper_axis_world": opening_axis.detach().cpu().tolist(),
            "wrench": wrench.detach().cpu().tolist(),
            "contacts": list(measured["finger_bolt_contacts"]),
            "robot_timestamp_s": float(self._robot.data._sim_timestamp),
        }

    def get_public_tcp_feature(self, *, observation_id: str, camera_frame_id: int) -> dict[str, object]:
        """Expose the live robot-model TCP/gripper feature for the current observation."""
        state = self.get_public_assembly_state()
        return robot_tcp_feature(
            observation_id=observation_id,
            camera_frame_id=camera_frame_id,
            tcp_pose_wxyz=state["tcp_pose"],
            finger_opening_axis_world=state["gripper_axis_world"],
            gripper_width_m=float(state["gripper_width_m"]),
            robot_timestamp_s=float(state["robot_timestamp_s"]),
        )

    def set_gripper_target_width(self, width_m: float, max_force_n: float) -> None:
        """Command measured Panda finger joints with a per-request physical effort cap."""
        if not torch.isfinite(torch.tensor([width_m, max_force_n])).all():
            raise ValueError("gripper width and force limit must be finite")
        max_width = 2.0 * float(self._robot.data.joint_pos_limits[0, self._finger_joint_ids, 1].min().item())
        width_roundoff_m = 1e-8
        if width_m < 0.0 or width_m > max_width + width_roundoff_m or max_force_n <= 0.0:
            raise ValueError(f"gripper target must satisfy 0<=width<={max_width:.4f} m and force>0 N")
        width_m = min(width_m, max_width)
        per_finger_force = float(max_force_n) / 2.0
        native_limits = self._native_finger_effort_limits
        if per_finger_force > float(native_limits.min().item()):
            raise ValueError("requested gripper force exceeds the installed Panda finger effort limits")
        limits = torch.full(
            (self.num_envs, len(self._finger_joint_ids)), per_finger_force, device=self.device
        )
        self._robot.write_joint_effort_limit_to_sim(limits, joint_ids=self._finger_joint_ids)
        self.gripper_target_joint_m = width_m / 2.0
        target = torch.full((self.num_envs, len(self._finger_joint_ids)), self.gripper_target_joint_m, device=self.device)
        self._robot.set_joint_position_target(target, joint_ids=self._finger_joint_ids)
