# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@configclass
class NaoEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 0.1
    action_space = 42  # Updated to match the number of joints
    observation_space = 1380 # What it does?
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    # Robot Configuration
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Nao",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/isaaclab_assets/data/Nao/nao.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # Corrected Joint Names
                "HeadYaw": 0.0,
                "HeadPitch": 0.0,
                "LHipYawPitch": 0.0,
                "LHipRoll": 0.0,
                "LHipPitch": -0.5,
                "LKneePitch": 0.5,
                "LAnklePitch": -0.5,
                "LAnkleRoll": 0.0,
                "RHipYawPitch": 0.0,
                "RHipRoll": 0.0,
                "RHipPitch": -0.5,
                "RKneePitch": 0.5,
                "RAnklePitch": -0.5,
                "RAnkleRoll": 0.0,
                "LShoulderPitch": 1.0,
                "LShoulderRoll": 0.5,
                "LElbowYaw": 0.5,
                "LElbowRoll": -1.0,
                "LWristYaw": 0.0,
                "RShoulderPitch": 1.0,
                "RShoulderRoll": -0.5,
                "RElbowYaw": 0.5,
                "RElbowRoll": 1.0,
                "RWristYaw": 0.0,
                "LHand": 0.0,
                "RHand": 0.0,
                "LFinger11": 0.0,
                "LFinger21": 0.0,
                "RFinger11": 0.0,
                "RFinger21": 0.0,
                "LThumb1": 0.0,
                "RThumb1": 0.0,
                "LFinger12": 0.0,
                "LFinger22": 0.0,
                "RFinger12": 0.0,
                "RFinger22": 0.0,
                "LFinger13": 0.0,
                "LFinger23": 0.0,
                "RFinger13": 0.0,
                "RFinger23": 0.0,
                "LThumb2": 0.0,
                "RThumb2": 0.0,
            },
            # pos=(1.0, 0.0, 0.00),  # Position of the robot in the environment
            pos=(1.0, 0.0, 0.4),  # Position of the robot in the environment
            # rot=(0.0, 0.0, 0.0, 1.0),  # Orientation of the robot (quaternion)
            rot=(0.0, -0.174, 0.0, 0.985),  # 后倾斜20度
        ),
        actuators={
            # Updated Actuator Configurations
            "Nao_arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
                    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"
                ],
                effort_limit_sim=5.0,
                velocity_limit_sim=0.5,
                stiffness=80.0,
                damping=10.0,
                armature = 0.01,
                friction = 0.04
            ),
            "Nao_legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch",
                    "LAnklePitch", "LAnkleRoll", "RHipYawPitch", "RHipRoll",
                    "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"
                ],
                effort_limit_sim=40.0,
                velocity_limit_sim=1.5,
                stiffness=70.0,
                damping=3.0,
                armature = 0.01,
                friction = 0.04
            ),
            "Nao_head": ImplicitActuatorCfg(
                joint_names_expr=["HeadYaw", "HeadPitch"],
                effort_limit_sim=10.0,
                velocity_limit_sim=1.0,
                stiffness=100.0,
                damping=5.0,
                armature = 0.01,
                friction = 0.04
            ),
            "Nao_hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    "LHand", "RHand", "LFinger11", "RFinger11",
                    "LFinger12", "RFinger12", "LFinger21", "RFinger21",
                    "LFinger22", "RFinger22", "LThumb1", "RThumb1",
                    "LThumb2", "RThumb2", "LFinger13", "LFinger23",
                    "RFinger13", "RFinger23", "LWristYaw", "RWristYaw"
                ],
                effort_limit_sim=0.5,
                velocity_limit_sim=0.1,
                stiffness=80.0,
                damping=10.0,
                armature = 0.03,
                friction = 0.2
            ),
        },

    )

    joint_gears: list = [
        10.0,  # HeadYaw
        10.0,  # HeadPitch
        50.0,  # LHipYawPitch
        50.0,  # LHipRoll
        50.0,  # LHipPitch
        40.0,  # LKneePitch
        40.0,  # LAnklePitch
        50.0,  # LAnkleRoll
        50.0,  # RHipYawPitch
        50.0,  # RHipRoll
        50.0,  # RHipPitch
        40.0,  # RKneePitch
        40.0,  # RAnklePitch
        50.0,  # RAnkleRoll
        80.0,  # LShoulderPitch
        80.0,  # LShoulderRoll
        50.0,  # LElbowYaw
        50.0,  # LElbowRoll
        30.0,  # LWristYaw
        80.0,  # RShoulderPitch
        80.0,  # RShoulderRoll
        50.0,  # RElbowYaw
        50.0,  # RElbowRoll
        30.0,  # RWristYaw
        20.0,  # LHand
        20.0,  # RHand
        10.0,  # LFinger11
        10.0,  # LFinger21
        10.0,  # RFinger11
        10.0,  # RFinger21
        40.0,  # LThumb1
        40.0,  # RThumb1
        10.0,  # LFinger12
        10.0,  # LFinger22
        10.0,  # RFinger12
        10.0,  # RFinger22
        10.0,  # LFinger13
        10.0,  # LFinger23
        10.0,  # RFinger13
        10.0,  # RFinger23
        30.0,  # LThumb2
        30.0,  # RThumb2
    ]

    # joint_gears: list = [
    #     1.0,  # HeadYaw
    #     1.0,  # HeadPitch
    #     5.0,  # LHipYawPitch
    #     5.0,  # LHipRoll
    #     5.0,  # LHipPitch
    #     4.0,  # LKneePitch
    #     4.0,  # LAnklePitch
    #     5.0,  # LAnkleRoll
    #     5.0,  # RHipYawPitch
    #     5.0,  # RHipRoll
    #     5.0,  # RHipPitch
    #     4.0,  # RKneePitch
    #     4.0,  # RAnklePitch
    #     5.0,  # RAnkleRoll
    #     8.0,  # LShoulderPitch
    #     8.0,  # LShoulderRoll
    #     5.0,  # LElbowYaw
    #     5.0,  # LElbowRoll
    #     0.0,  # LWristYaw
    #     8.0,  # RShoulderPitch
    #     8.0,  # RShoulderRoll
    #     5.0,  # RElbowYaw
    #     5.0,  # RElbowRoll
    #     0.0,  # RWristYaw
    #     0.0,  # LHand
    #     0.0,  # RHand
    #     0.0,  # LFinger11
    #     0.0,  # LFinger21
    #     0.0,  # RFinger11
    #     0.0,  # RFinger21
    #     0.0,  # LThumb1
    #     0.0,  # RThumb1
    #     0.0,  # LFinger12
    #     0.0,  # LFinger22
    #     0.0,  # RFinger12
    #     0.0,  # RFinger22
    #     0.0,  # LFinger13
    #     0.0,  # LFinger23
    #     0.0,  # RFinger13
    #     0.0,  # RFinger23
    #     0.0,  # LThumb2
    #     0.0,  # RThumb2

    #     # 1.0,  # LFinger11
    #     # 1.0,  # LFinger21
    #     # 1.0,  # RFinger11
    #     # 1.0,  # RFinger21
    #     # 4.0,  # LThumb1
    #     # 4.0,  # RThumb1
    #     # 1.0,  # LFinger12
    #     # 1.0,  # LFinger22
    #     # 1.0,  # RFinger12
    #     # 1.0,  # RFinger22
    #     # 1.0,  # LFinger13
    #     # 1.0,  # LFinger23
    #     # 1.0,  # RFinger13
    #     # 1.0,  # RFinger23
    #     # 3.0,  # LThumb2
    #     # 3.0,  # RThumb2
    # ]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.01
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.25

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01



class NaoEnv(LocomotionEnv):
    cfg: NaoEnvCfg

    def __init__(self, cfg: NaoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize reward components for logging
        self.reward_components = {}
        
        # Initialize accumulated reward components for iteration averaging
        self.accumulated_reward_components = {
            "progress_reward": torch.zeros(self.num_envs, device=self.sim.device),
            "alive_reward": torch.zeros(self.num_envs, device=self.sim.device),
            "up_reward": torch.zeros(self.num_envs, device=self.sim.device),
            "heading_reward": torch.zeros(self.num_envs, device=self.sim.device),
            "actions_penalty": torch.zeros(self.num_envs, device=self.sim.device),
            "energy_penalty": torch.zeros(self.num_envs, device=self.sim.device),
            "dof_at_limit_cost": torch.zeros(self.num_envs, device=self.sim.device),
            "death_penalty": torch.zeros(self.num_envs, device=self.sim.device),
        }
        self.step_count = torch.zeros(self.num_envs, device=self.sim.device)

    def _get_rewards(self) -> torch.Tensor:
        # Compute rewards with components for logging
        (total_reward, progress_reward, alive_reward, up_reward, 
         heading_reward, actions_penalty, energy_penalty, 
         dof_at_limit_cost, death_penalty) = compute_rewards_with_components(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.motor_effort_ratio,
        )
        
        # Store reward components for logging
        self.reward_components = {
            "progress_reward": progress_reward,
            "alive_reward": alive_reward,
            "up_reward": up_reward,
            "heading_reward": heading_reward,
            "actions_penalty": actions_penalty,
            "energy_penalty": energy_penalty,
            "dof_at_limit_cost": dof_at_limit_cost,
            "death_penalty": death_penalty,
        }
        
        # Accumulate reward components for iteration averaging
        self.accumulated_reward_components["progress_reward"] += progress_reward
        self.accumulated_reward_components["alive_reward"] += alive_reward
        self.accumulated_reward_components["up_reward"] += up_reward
        self.accumulated_reward_components["heading_reward"] += heading_reward
        self.accumulated_reward_components["actions_penalty"] += actions_penalty
        self.accumulated_reward_components["energy_penalty"] += energy_penalty
        self.accumulated_reward_components["dof_at_limit_cost"] += dof_at_limit_cost
        self.accumulated_reward_components["death_penalty"] += death_penalty
        
        # Increment step count for averaging
        self.step_count += 1
        
        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        
        # Call parent reset
        super()._reset_idx(env_ids)
        
        # Reset accumulated reward components for reset environments
        for key in self.accumulated_reward_components:
            self.accumulated_reward_components[key][env_ids] = 0.0
        self.step_count[env_ids] = 0

    def get_averaged_reward_components(self) -> dict[str, torch.Tensor]:
        """Calculate averaged reward components over the current iteration."""
        averaged_components = {}
        for key, accumulated_values in self.accumulated_reward_components.items():
            # Avoid division by zero
            step_count_safe = torch.where(self.step_count > 0, self.step_count.float(), torch.ones_like(self.step_count).float())
            averaged_components[key] = accumulated_values / step_count_safe
        return averaged_components


@torch.jit.script
def compute_rewards_with_components(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
):
    """Compute rewards with individual components for logging."""
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    # Compute individual components (matching original implementation)
    actions_penalty = actions_cost_scale * actions_cost
    energy_penalty = energy_cost_scale * electricity_cost

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_penalty
        - energy_penalty
        - dof_at_limit_cost
    )
    # adjust reward for fallen agents (matching original implementation)
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    # Calculate death_penalty for logging (difference between adjusted and unadjusted reward)
    unadjusted_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_penalty
        - energy_penalty
        - dof_at_limit_cost
    )
    death_penalty = total_reward - unadjusted_reward
    
    return total_reward, progress_reward, alive_reward, up_reward, heading_reward, actions_penalty, energy_penalty, dof_at_limit_cost, death_penalty


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
