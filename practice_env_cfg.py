from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

_PKG_ROOT = Path(__file__).resolve().parents[3]  # .../source/practice/practice
DOOSAN_USD = str(_PKG_ROOT / "assets/e0509_gripper/e0509_model.usd")


@configclass
class PracticeEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 8.0 #length of episode in seconds

    action_space = 8
    observation_space = 22
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)

    # -------------------------
    # Robot (Doosan)
    # -------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DOOSAN_USD,
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.45, 0.0, 0.53),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": -1.2,
                "joint_3": 1.5,
                "joint_4": -1.8,
                "joint_5": 0.0,
                "joint_6": 1.6,
                "rh_.*": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                stiffness=1500.0,
                damping=100.0,
            ),
            #gripper control 
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["rh_l1", "rh_r1_joint"],
                stiffness=2e4,
                damping=1e2,
            ),
        },
    )

    # -------------------------
    # Table: 1.20 x 0.60 x 0.02
    # -------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.60, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=2.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.51),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # -------------------------
    # Table legs (4x): 3cm x 3cm x 50cm
    # -------------------------
    leg_size = (0.03, 0.03, 0.50)
    _leg_x = 0.60 - 0.015  # 0.585
    _leg_y = 0.30 - 0.015  # 0.285
    _leg_z = 0.25

    leg_fl_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg_FL",
        spawn=sim_utils.CuboidCfg(
            size=leg_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(+_leg_x, +_leg_y, _leg_z)),
    )

    leg_fr_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg_FR",
        spawn=sim_utils.CuboidCfg(
            size=leg_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(+_leg_x, -_leg_y, _leg_z)),
    )

    leg_bl_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg_BL",
        spawn=sim_utils.CuboidCfg(
            size=leg_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-_leg_x, +_leg_y, _leg_z)),
    )

    leg_br_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leg_BR",
        spawn=sim_utils.CuboidCfg(
            size=leg_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-_leg_x, -_leg_y, _leg_z)),
    )

    # -------------------------
    # Metal post (vertical): 3cm x 3cm x 1.0m
    # -------------------------
    metal_post_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/MetalPost",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 1.00),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.6),
                metallic=1.0,
                roughness=0.25,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(+0.595, 0.0, 1.02),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # -------------------------
    # Camera on metal post (RGBD)
    # -------------------------
    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/MetalPost/RealsenseCamera",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.48),
            rot=(0.0, 0.7071068, 0.0, 0.7071068),
        ),
    )

    # -------------------------
    # Large cube (for stacking)
    # -------------------------
    cube_large_size = (0.05, 0.05, 0.05)  # 7cm cube (change later if you want)

    cube_large_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeLarge",
        spawn=sim_utils.CuboidCfg(
            size=cube_large_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,   # dynamic object
                disable_gravity=False,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),  # red-ish
                metallic=0.0,
                roughness=0.6,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Place on table top. Table top z ~ 0.51 + 0.01 = 0.52
            # Cube half height = 0.05 => center z ~ 0.52 + 0.05 = 0.57
            pos=(0.20, 0.0, 0.57),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )


    # -------------------------
    # Reach task params
    # -------------------------
    action_scale = 0.10
    target_radius = 0.05
    target_pos = (0.45, 0.0, 0.55)
    rew_scale_dist = 1.0
    rew_scale_action = 0.01

    # --- Pick & move (large cube) task params ---
    goal_pos = (-0.25, 0.0, 0.57)  # on table, closer to robot
    goal_radius = 0.05

    table_top_z = 0.52              # table top height (0.51 + 0.01)
    cube_half_z = 0.05             # for 0.10 cube
    place_z_tol = 0.03             # allowed vertical tolerance when "placed"
    lift_height = 0.08             # "picked" if cube_z > table_top_z + lift_height
    lift_hold_steps = 6          # must stay lifted for N control steps (~N*dt*decimation)

    arm_action_scale = 0.3         # rad per step
    gripper_action_scale = 0.03     # rad per step (tune after seeing motion)

    rew_scale_reach = 0.5           # EE -> cube approach reward
    rew_scale_lift = 2.0            # reward for lifting
    rew_scale_goal = 4.0            # reward for moving cube to goal
    rew_scale_action = 0.01         # penalty

