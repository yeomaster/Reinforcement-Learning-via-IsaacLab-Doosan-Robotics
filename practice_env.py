# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause





# THIS WAS PAINFUL 
# LESSON LEARNT:JUST CODE MYSELF FROM NOW ON




from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.assets import RigidObject
from isaaclab.sensors import Camera

from .practice_env_cfg import PracticeEnvCfg


class PracticeEnv(DirectRLEnv):
    cfg: PracticeEnvCfg

    def __init__(self, cfg: PracticeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- Arm joint mapping (known joint names) ----
        self._arm_joint_names = [f"joint_{i}" for i in range(1, 7)]
        self._arm_dof_ids, _ = self.robot.find_joints(self._arm_joint_names)

        # ---- Gripper joint mapping ----
        self._gripper_joint_names = ["rh_l1", "rh_r1_joint"]
        self._gripper_dof_ids, _ = self.robot.find_joints(self._gripper_joint_names)

        # ---- Goal position (env-local) ----
        self.goal_pos = torch.tensor(self.cfg.goal_pos, device=self.device).repeat(self.num_envs, 1)


        # ---- Pick an end-effector body ONCE ----
        candidates = ["tool0", "ee_link", "end_effector", "flange", "tcp"]
        ee_id = None
        for name in candidates:
            try:
                ids, _ = self.robot.find_bodies(name)
                if len(ids) > 0:
                    ee_id = int(ids[0])
                    break
            except Exception:
                pass

        # Fallback: last body
        self._ee_body_id = ee_id if ee_id is not None else -1

        # ---- Fixed target position in "env-local" frame ----
        # We convert it to world frame by adding scene.env_origins each step.
        self.target_pos = torch.tensor(self.cfg.target_pos, device=self.device).repeat(self.num_envs, 1)

        self.lifted_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.lift_hold_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self._prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._smoothed_actions = torch.zeros_like(self._prev_actions)




    def _setup_scene(self):
        # -------------------------
        # Spawn assets
        # -------------------------
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Table + legs + post
        self.table = RigidObject(self.cfg.table_cfg)
        self.leg_fl = RigidObject(self.cfg.leg_fl_cfg)
        self.leg_fr = RigidObject(self.cfg.leg_fr_cfg)
        self.leg_bl = RigidObject(self.cfg.leg_bl_cfg)
        self.leg_br = RigidObject(self.cfg.leg_br_cfg)
        self.metal_post = RigidObject(self.cfg.metal_post_cfg)
        self.cube_large = RigidObject(self.cfg.cube_large_cfg)

        # Camera
        #self.camera = Camera(self.cfg.camera_cfg)

        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # -------------------------
        # Clone environments
        # -------------------------
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # -------------------------
        # Register into scene
        # -------------------------
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["leg_fl"] = self.leg_fl
        self.scene.rigid_objects["leg_fr"] = self.leg_fr
        self.scene.rigid_objects["leg_bl"] = self.leg_bl
        self.scene.rigid_objects["leg_br"] = self.leg_br
        self.scene.rigid_objects["metal_post"] = self.metal_post
        # self.scene.sensors["camera"] = self.camera
        self.scene.rigid_objects["cube_large"] = self.cube_large

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # clamp to [-1, 1]
        actions = torch.clamp(actions, -1.0, 1.0)

        # low-pass filter to remove jitter
        alpha = 0.85  # higher = smoother (try 0.85; if still jittery 0.90)
        self._smoothed_actions = alpha * self._smoothed_actions + (1.0 - alpha) * actions

        # store for optional "action rate" penalty
        self._prev_actions = self.actions.clone() if hasattr(self, "actions") else actions.clone()

        self.actions = self._smoothed_actions



    # def _pre_physics_step(self, actions: torch.Tensor) -> None:
    #     # Clamp actions to [-1, 1]
    #     self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
    # converting numbers between -1 and 1 into actionable joint motions
    # PPO outputs numbers between -1 and 1
    # _pre_physics_step() stores these numbers
    # _apply_action() converts them into actual robot commands in the simulator

        # actions: (num_envs, 8) = [arm6, grip2]  (we will USE only actions[:,6] for gripper)
        a_arm = self.actions[:, :6]
        # self.actions is shaped (N, 8) where N = num_envs
        # essentially = [all robots, joints 1~6]
        # each row = one environment’s arm action (6 numbers in [-1, 1])

        g = self.actions[:, 6]  # (N,) single scalar for gripper
        # g is storing all actions of of the GRIPPER
        # [all robots, 7th index] 7th index = gripper joint

        # --- Arm: delta joint-position targets ---
        q_arm = self.robot.data.joint_pos[:, self._arm_dof_ids]
        # 

        q_arm_target = q_arm + self.cfg.arm_action_scale * a_arm
        # 

        self.robot.set_joint_position_target(q_arm_target, joint_ids=self._arm_dof_ids)


        # --- Gripper: symmetric open/close with ONE scalar ---
        q_grip = self.robot.data.joint_pos[:, self._gripper_dof_ids]
        q_grip_target = q_grip.clone()

        # Convention: +g closes. If it opens instead, flip sign: g = -g
        q_grip_target[:, 0] = q_grip[:, 0] + self.cfg.gripper_action_scale * g
        q_grip_target[:, 1] = q_grip[:, 1] - self.cfg.gripper_action_scale * g

        # Limit per-step finger motion (prevents chatter)
        max_delta_grip = 0.01
        q_grip_target = torch.clamp(q_grip_target, q_grip - max_delta_grip, q_grip + max_delta_grip)

        self.robot.set_joint_position_target(q_grip_target, joint_ids=self._gripper_dof_ids)

        

    def _get_observations(self) -> dict:
        q_arm = self.robot.data.joint_pos[:, self._arm_dof_ids]      # (N, 6)
        qd_arm = self.robot.data.joint_vel[:, self._arm_dof_ids]     # (N, 6)

        q_grip = self.robot.data.joint_pos[:, self._gripper_dof_ids] # (N, 2)
        qd_grip = self.robot.data.joint_vel[:, self._gripper_dof_ids]# (N, 2)

        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_id, :]
        cube_pos = self.cube_large.data.root_pos_w
        goal_pos_w = self.scene.env_origins + self.goal_pos

        ee_to_cube = cube_pos - ee_pos           # (N,3)
        cube_to_goal = goal_pos_w - cube_pos     # (N,3)

        obs = torch.cat([q_arm, qd_arm, q_grip, qd_grip, ee_to_cube, cube_to_goal], dim=-1)
        return {"policy": obs}


    # HATE HATE HATE HATE HATE HATE WHY NO WORK AAAAAAAA
    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_id, :]
        cube_pos = self.cube_large.data.root_pos_w

        # distance EE -> cube
        dist = torch.linalg.norm(ee_pos - cube_pos, dim=-1)
        rew_reach = torch.exp(-10.0 * dist)   # [0..1]
        # rew_reach = torch.exp(-20.0 * dist)   # was -10

        # encourage closing when near the cube
        near = (dist < 0.06).float()   # tune 0.05~0.08
        g = self.actions[:, 6]         # + means close (your convention)
        rew_close = 0.5 * near * torch.clamp(g, min=0.0)


        # lift proxy
        cube_z = cube_pos[:, 2]
        lift_thresh = self.cfg.table_top_z + self.cfg.lift_height
        lifted = cube_z > lift_thresh

        # update lift-hold counter
        self.lift_hold_count = torch.where(
            lifted,
            self.lift_hold_count + 1,
            torch.zeros_like(self.lift_hold_count),
        )

        # lift reward: only positive after threshold
        rew_lift = torch.clamp(cube_z - lift_thresh, min=0.0)

        # small bonus when lifted (helps PPO discover it)
        rew_lift_bonus = lifted.float() * 1.0

        # action penalty
        rew_act = -self.cfg.rew_scale_action * torch.sum(self.actions**2, dim=-1)

        rew_not_closing = -0.3 * near * torch.clamp(-g, min=0.0)  # penalize opening when near

        return 2.0 * rew_reach + 6.0 * rew_lift + rew_lift_bonus + rew_act+ rew_close + rew_not_closing



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cube_pos = self.cube_large.data.root_pos_w
        cube_z = cube_pos[:, 2]

        lift_thresh = self.cfg.table_top_z + self.cfg.lift_height

        # success when held above threshold long enough
        success = self.lift_hold_count >= self.cfg.lift_hold_steps

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return success, time_out


    # def _get_rewards(self) -> torch.Tensor:
    #     ee_pos = self.robot.data.body_pos_w[:, self._ee_body_id, :]
    #     cube_pos = self.cube_large.data.root_pos_w

    #     # 1) Reach (keep exactly as you had)
    #     dist = torch.linalg.norm(ee_pos - cube_pos, dim=-1)
    #     rew_reach = torch.exp(-10.0 * dist)   # [0..1]

    #     # 2) Lift proxy
    #     cube_z = cube_pos[:, 2]
    #     lift_thresh = self.cfg.table_top_z + self.cfg.lift_height
    #     lifted = cube_z > lift_thresh

    #     # update lift-hold counter
    #     self.lift_hold_count = torch.where(
    #         lifted,
    #         self.lift_hold_count + 1,
    #         torch.zeros_like(self.lift_hold_count),
    #     )

    #     # --- Grasp shaping: encourage closing when near the cube ---
    #     near = (dist < self.cfg.near_thresh).float()

    #     # g in [-1, 1], with + meaning close (same convention as apply_action)
    #     g = self.actions[:, 6]

    #     # reward closing (+g) when near, and discourage closing when far
    #     rew_grasp = self.cfg.rew_scale_grasp * (near * torch.clamp(g, min=0.0) - (1.0 - near) * torch.clamp(g, min=0.0))


    #     # ---- NEW: gate lift reward so it only counts when EE is close ----
    #     # If the EE is far, lifting is probably from bumping/knocking.
    #     near = (dist < self.cfg.near_thresh).float()

    #     # Lift reward only when near
    #     rew_lift = torch.clamp(cube_z - lift_thresh, min=0.0) * near

    #     # ---- NEW: reward holding lifted (stability) ----
    #     # Gives a smooth incentive to keep it up, instead of a single pop.
    #     rew_hold = (self.lift_hold_count.float() / float(self.cfg.lift_hold_steps)).clamp(0.0, 1.0) * near

    #     # keep your bonus (but also gated)
    #     rew_lift_bonus = lifted.float() * 1.0 * near

    #     # action penalty (keep)
    #     rew_act = -self.cfg.rew_scale_action * torch.sum(self.actions**2, dim=-1)

    #     return (
    #         2.0 * rew_reach
    #         + 10.0 * rew_lift
    #         + 2.0 * rew_hold
    #         + rew_lift_bonus
    #         + rew_act
    #         + rew_grasp
    #     )
    
    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     cube_z = self.cube_large.data.root_pos_w[:, 2]
    #     lift_thresh = self.cfg.table_top_z + self.cfg.lift_height

    #     lifted = cube_z > lift_thresh
    #     success = lifted & (self.lift_hold_count >= self.cfg.lift_hold_steps)

    #     time_out = self.episode_length_buf >= self.max_episode_length - 1
    #     return success, time_out



    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Default joint state from robot init/defaults
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # Place root at env origin
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write to sim
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ---- Reset large cube pose ----
        cube_root = self.cube_large.data.default_root_state[env_ids].clone()
        cube_root[:, :3] += self.scene.env_origins[env_ids]
        self.cube_large.write_root_pose_to_sim(cube_root[:, :7], env_ids)
        self.cube_large.write_root_velocity_to_sim(cube_root[:, 7:], env_ids)

        self.lifted_once[env_ids] = False

        self.lift_hold_count[env_ids] = 0

        # self.lift_hold_count[env_ids] = 0

