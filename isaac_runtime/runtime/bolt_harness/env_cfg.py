"""Single-environment, known-layout bolt scene derived from Factory."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryEnvCfg
from isaaclab_tasks.direct.factory.factory_tasks_cfg import FactoryTask

from .measurement import ASSEMBLY_FRAME_POS_M, CASING_ENTRY_PLANE_Z_M


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("ASSEMBLY_ASSET_ROOT", str(REPO_ROOT / "assets"))).expanduser()
PANDA_USD = (
    ASSET_ROOT
    / "vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
)
BOLT_USD = ASSET_ROOT / "parts/M6 Hub Bolt.usd"
CASING_USD = ASSET_ROOT / "parts/Casing Top.usd"
COVER_USD = ASSET_ROOT / "parts/Hub Cover Output.usd"
S2_BLOCKER_DEPTH_M = 0.008
S2_BLOCKER_SIZE_M = (0.012, 0.0015, 0.0015)


@configclass
class BoltFactoryTask(FactoryTask):
    """Task metadata only; stock Factory held-object reset is not used."""

    name = "bolt_insert_smoke"


@configclass
class BoltInsertionEnvCfg(FactoryEnvCfg):
    """Factory controller configuration with the repository's real Panda/CAD."""

    task_name: str = "bolt_insert_smoke"
    condition: str = "S0"
    task: FactoryTask = BoltFactoryTask()
    episode_length_s: float = 120.0
    decimation: int = 2
    action_space: int = 6
    obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    state_order: list = ["fingertip_pos", "fingertip_quat", "ee_linvel", "ee_angvel", "joint_pos"]
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, clone_in_fabric=False)
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
        render_interval=10,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.65),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=2,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**20,
            gpu_max_num_partitions=1,
        ),
    )

    bolt_contact_max_data_count_per_prim: int = 1024
    task_translation_prop_gain: float = 300.0
    pose_window_tolerances: dict[str, float] | None = None
    task_rgb_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/TaskRgbCamera",
        update_period=10.0 / 120.0,
        height=int(os.environ.get("ASSEMBLY_CAMERA_HEIGHT","720")),
        width=int(os.environ.get("ASSEMBLY_CAMERA_WIDTH","960")),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # A high oblique view keeps the held bolt and annular socket visible together.
            pos=(0.95, 0.15, 1.45),
            rot=(0.18858389, -0.52040357, -0.78300949, 0.28374704),
            convention="ros",
        ),
    )

    # USD references retain authored vertex coordinates; apply 0.002 world m/source unit.
    cad_stage_scale: float = 0.002
    table_top_z_m: float = 0.68
    table_size_m: tuple[float, float, float] = (1.4, 0.9, 0.06)
    table_center_m: tuple[float, float, float] = (0.45, 0.0, 0.65)
    bolt_reset_pos_m: tuple[float, float, float] = (0.30, 0.25, 0.706149998)
    # The free bolt's tip is on the table; local +Y (cap side) maps to world +Z.
    bolt_reset_quat_wxyz: tuple[float, float, float, float] = (0.70710678, 0.70710678, 0.0, 0.0)
    bolt_mass_kg: float = 0.045
    assembly_frame_pos_m: tuple[float, float, float] = (0.50, -0.15, 0.735896652)
    assembly_frame_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    cover_pos_m: tuple[float, float, float] = (0.50, -0.06316282, 0.797896652)
    cover_quat_wxyz: tuple[float, float, float, float] = (0.0, 0.0, 0.70710678, 0.70710678)
    cover_usd_path: str = str(COVER_USD)
    bolt_seat_root_pos_m: tuple[float, float, float] = (0.58, -0.15, 0.794105702)
    bolt_seat_root_quat_wxyz: tuple[float, float, float, float] = (0.70710678, 0.70710678, 0.0, 0.0)
    tcp_body_name: str = "tool_center"
    tcp_offset_from_hand_local_m: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    wrench_body_name: str = "panda_hand"
    wrench_smoothing_factor: float = 0.25
    task_rot_deriv_scale: float = 10.0
    grasp_contact_min_force_n: float = 0.5
    wrench_warmup_steps: int = 60
    wrench_stationary_sample_count: int = 30
    wrench_stationary_joint_speed_limit_rad_s: float = 0.05
    wrench_stationary_tcp_drift_limit_m: float = 0.002
    tcp_tracking_stop_error_m: float = 0.02
    tcp_tracking_stop_orientation_rad: float = 0.2
    tcp_linear_speed_stop_mps: float = 0.08
    tcp_angular_speed_stop_radps: float = 0.8

    robot: ArticulationCfg = copy.deepcopy(FRANKA_PANDA_CFG)
    bolt: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bolt",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BOLT_USD),
            scale=(0.002, 0.002, 0.002),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                max_depenetration_velocity=2.0,
                linear_damping=0.02,
                angular_damping=0.02,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.045),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.18, 0.706149998), rot=(0.70710678, 0.70710678, 0.0, 0.0)
        ),
    )


def _condition_blocker_geometry(
    cfg: BoltInsertionEnvCfg,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if cfg.condition == "S0":
        return None
    if cfg.condition != "S2":
        raise ValueError("condition must be 'S0' or 'S2'")

    scale = float(cfg.cad_stage_scale)
    if scale <= 0.0:
        raise ValueError("cad_stage_scale must be positive for the S2 blocker")
    local_position = (
        (cfg.bolt_seat_root_pos_m[0] - cfg.assembly_frame_pos_m[0]) / scale,
        (cfg.bolt_seat_root_pos_m[1] - cfg.assembly_frame_pos_m[1]) / scale,
        (CASING_ENTRY_PLANE_Z_M - ASSEMBLY_FRAME_POS_M[2] - S2_BLOCKER_DEPTH_M) / scale,
    )
    local_size = tuple(dimension / scale for dimension in S2_BLOCKER_SIZE_M)
    return local_position, local_size


def make_env_cfg(*, device: str = "cuda:0", condition: str = "S0") -> BoltInsertionEnvCfg:
    """Build the bounded smoke config and point it at repository-local assets."""
    if condition not in ("S0", "S2"):
        raise ValueError("condition must be 'S0' or 'S2'")
    cfg = BoltInsertionEnvCfg()
    cfg.sim.device = device
    cfg.condition = condition
    # Keep the task's rotation gain; translation gain is a bounded YAML calibration value.
    cfg.ctrl.default_task_prop_gains = [
        *([float(cfg.task_translation_prop_gain)] * 3),
        *cfg.ctrl.default_task_prop_gains[3:],
    ]

    cfg.robot = copy.deepcopy(FRANKA_PANDA_CFG)
    cfg.robot.prim_path = "/World/envs/env_.*/Robot"
    cfg.robot.spawn.usd_path = str(PANDA_USD)
    cfg.robot.spawn.activate_contact_sensors = True
    # Factory torque control compensates gravity by disabling it on the arm,
    # while task assets (including the free bolt) retain world gravity.
    cfg.robot.spawn.rigid_props.disable_gravity = True
    cfg.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.robot.spawn.articulation_props.solver_position_iteration_count = 192
    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
    cfg.robot.init_state.pos = (0.0, 0.0, cfg.table_top_z_m)
    cfg.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    factory_reset_joints = [float(value) for value in cfg.ctrl.reset_joints]
    cfg.ctrl.default_dof_pos_tensor = factory_reset_joints
    cfg.robot.init_state.joint_pos = {
        **{f"panda_joint{index + 1}": value for index, value in enumerate(factory_reset_joints)},
        "panda_finger_joint.*": 0.04,
    }
    # Factory torque mode has no implicit arm drive; the Jacobian controller
    # supplies arm torque. Leave static friction inherited from the Panda USD.
    for actuator_name in ("panda_shoulder", "panda_forearm"):
        cfg.robot.actuators[actuator_name].stiffness = 0.0
        cfg.robot.actuators[actuator_name].damping = 0.0

    cfg.bolt = copy.deepcopy(cfg.bolt)
    cfg.bolt.spawn.usd_path = str(BOLT_USD)
    cfg.bolt.spawn.scale = (cfg.cad_stage_scale,) * 3
    cfg.bolt.spawn.mass_props.mass = cfg.bolt_mass_kg
    cfg.bolt.init_state.pos = cfg.bolt_reset_pos_m
    cfg.bolt.init_state.rot = cfg.bolt_reset_quat_wxyz
    return cfg


def make_casing_cfg(cfg: BoltInsertionEnvCfg) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(usd_path=str(CASING_USD), scale=(cfg.cad_stage_scale,) * 3)


def make_cover_cfg(cfg: BoltInsertionEnvCfg) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(usd_path=cfg.cover_usd_path, scale=(cfg.cad_stage_scale,) * 3)
