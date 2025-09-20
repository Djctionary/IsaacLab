# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class NaoEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1
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
                effort_limit_sim=50.0,
                velocity_limit_sim=1.5,
                stiffness=80.0,
                damping=4.0,
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
            ),
            "Nao_head": ImplicitActuatorCfg(
                joint_names_expr=["HeadYaw", "HeadPitch"],
                effort_limit_sim=10.0,
                velocity_limit_sim=1.0,
                stiffness=100.0,
                damping=5.0,
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
                velocity_limit_sim=0.5,
                stiffness=50.0,
                damping=200.0,
            ),
        },

    )

    # joint_gears: list = [
    #     10.0,  # HeadYaw
    #     10.0,  # HeadPitch
    #     50.0,  # LHipYawPitch
    #     50.0,  # LHipRoll
    #     50.0,  # LHipPitch
    #     40.0,  # LKneePitch
    #     40.0,  # LAnklePitch
    #     50.0,  # LAnkleRoll
    #     50.0,  # RHipYawPitch
    #     50.0,  # RHipRoll
    #     50.0,  # RHipPitch
    #     40.0,  # RKneePitch
    #     40.0,  # RAnklePitch
    #     50.0,  # RAnkleRoll
    #     80.0,  # LShoulderPitch
    #     80.0,  # LShoulderRoll
    #     50.0,  # LElbowYaw
    #     50.0,  # LElbowRoll
    #     30.0,  # LWristYaw
    #     80.0,  # RShoulderPitch
    #     80.0,  # RShoulderRoll
    #     50.0,  # RElbowYaw
    #     50.0,  # RElbowRoll
    #     30.0,  # RWristYaw
    #     20.0,  # LHand
    #     20.0,  # RHand
    #     10.0,  # LFinger11
    #     10.0,  # LFinger21
    #     10.0,  # RFinger11
    #     10.0,  # RFinger21
    #     40.0,  # LThumb1
    #     40.0,  # RThumb1
    #     10.0,  # LFinger12
    #     10.0,  # LFinger22
    #     10.0,  # RFinger12
    #     10.0,  # RFinger22
    #     10.0,  # LFinger13
    #     10.0,  # LFinger23
    #     10.0,  # RFinger13
    #     10.0,  # RFinger23
    #     30.0,  # LThumb2
    #     30.0,  # RThumb2
    # ]

    joint_gears: list = [
        1.0,  # HeadYaw
        1.0,  # HeadPitch
        5.0,  # LHipYawPitch
        5.0,  # LHipRoll
        5.0,  # LHipPitch
        4.0,  # LKneePitch
        4.0,  # LAnklePitch
        5.0,  # LAnkleRoll
        5.0,  # RHipYawPitch
        5.0,  # RHipRoll
        5.0,  # RHipPitch
        4.0,  # RKneePitch
        4.0,  # RAnklePitch
        5.0,  # RAnkleRoll
        8.0,  # LShoulderPitch
        8.0,  # LShoulderRoll
        5.0,  # LElbowYaw
        5.0,  # LElbowRoll
        3.0,  # LWristYaw
        8.0,  # RShoulderPitch
        8.0,  # RShoulderRoll
        5.0,  # RElbowYaw
        5.0,  # RElbowRoll
        3.0,  # RWristYaw
        2.0,  # LHand
        2.0,  # RHand
        1.0,  # LFinger11
        1.0,  # LFinger21
        1.0,  # RFinger11
        1.0,  # RFinger21
        4.0,  # LThumb1
        4.0,  # RThumb1
        1.0,  # LFinger12
        1.0,  # LFinger22
        1.0,  # RFinger12
        1.0,  # RFinger22
        1.0,  # LFinger13
        1.0,  # LFinger23
        1.0,  # RFinger13
        1.0,  # RFinger23
        3.0,  # LThumb2
        3.0,  # RThumb2
    ]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.2

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01



class NaoEnv(LocomotionEnv):
    cfg: NaoEnvCfg

    def __init__(self, cfg: NaoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
    #     # 调试计数器
    #     self.debug_step = 0

    # def _get_rewards(self) -> torch.Tensor:
    #     """重写奖励函数，添加调试信息"""
    #     # 调用基类方法获取奖励
    #     total_reward = super()._get_rewards()
        
    #     # 每100步打印一次调试信息
    #     if self.debug_step % 100 == 0:
    #         self._debug_print_info()
    #         self._debug_print_reward_breakdown()
        
    #     self.debug_step += 1
    #     return total_reward
    
    # def _debug_print_info(self):
    #     """打印调试信息"""
    #     # 获取当前状态
    #     torso_position = self.robot.data.root_pos_w
    #     torso_rotation = self.robot.data.root_quat_w
    #     velocity = self.robot.data.root_lin_vel_w
    #     ang_velocity = self.robot.data.root_ang_vel_w
        
    #     # 计算姿态角度
    #     import torch
    #     from isaacsim.core.utils.torch.rotations import compute_heading_and_up
        
    #     # 计算up_proj和heading_proj
    #     to_target = self.targets - torso_position
    #     to_target[:, 2] = 0.0
        
    #     torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
    #         torso_rotation, self.inv_start_rot, to_target, self.basis_vec0, self.basis_vec1, 2
    #     )
        
    #     # 计算角度（弧度转度）
    #     up_angle_deg = torch.acos(torch.clamp(up_proj, 0, 1)) * 180 / torch.pi
    #     heading_angle_deg = torch.acos(torch.clamp(heading_proj, 0, 1)) * 180 / torch.pi
        
    #     # 获取终止状态
    #     died = torso_position[:, 2] < self.cfg.termination_height
    #     time_out = self.episode_length_buf >= self.max_episode_length - 1
        
    #     # 打印第一个环境的信息
    #     env_idx = 0
    #     print(f"\n=== 调试信息 (步骤 {self.debug_step}) ===")
    #     print(f"环境 {env_idx}:")
    #     print(f"  位置: {torso_position[env_idx].cpu().numpy()}")
    #     print(f"  高度: {torso_position[env_idx, 2].item():.3f}m")
    #     print(f"  速度: {velocity[env_idx].cpu().numpy()}")
    #     print(f"  角速度: {ang_velocity[env_idx].cpu().numpy()}")
    #     print(f"  四元数: {torso_rotation[env_idx].cpu().numpy()}")
    #     print(f"  直立角度: {up_angle_deg[env_idx].item():.1f}° (up_proj: {up_proj[env_idx].item():.3f})")
    #     print(f"  朝向角度: {heading_angle_deg[env_idx].item():.1f}° (heading_proj: {heading_proj[env_idx].item():.3f})")
    #     print(f"  是否摔倒: {died[env_idx].item()}")
    #     print(f"  是否超时: {time_out[env_idx].item()}")
    #     print(f"  重置终止: {self.reset_terminated[env_idx].item()}")
    #     print(f"  终止高度阈值: {self.cfg.termination_height}m")
    #     print(f"  直立奖励阈值: 21.6° (up_proj > 0.93)")
    #     print(f"  朝向奖励阈值: 36.9° (heading_proj > 0.8)")
    #     print("=" * 50)
    
    # def _debug_print_reward_breakdown(self):
    #     """打印奖励分解信息"""
    #     import torch
    #     from isaacsim.core.utils.torch.rotations import compute_heading_and_up
        
    #     # 获取当前状态
    #     torso_position = self.robot.data.root_pos_w
    #     torso_rotation = self.robot.data.root_quat_w
    #     dof_vel = self.robot.data.joint_vel
    #     dof_pos = self.robot.data.joint_pos
        
    #     # 计算中间值
    #     to_target = self.targets - torso_position
    #     to_target[:, 2] = 0.0
        
    #     torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
    #         torso_rotation, self.inv_start_rot, to_target, self.basis_vec0, self.basis_vec1, 2
    #     )
        
    #     # 计算奖励组件
    #     env_idx = 0
        
    #     # 1. 进度奖励
    #     progress_reward = self.potentials[env_idx] - self.prev_potentials[env_idx]
        
    #     # 2. 存活奖励
    #     alive_reward = self.cfg.alive_reward_scale  # 2.0
        
    #     # 3. 直立奖励
    #     up_reward = self.cfg.up_weight if up_proj[env_idx] > 0.93 else 0.0  # 0.1
        
    #     # 4. 朝向奖励
    #     heading_weight_tensor = self.cfg.heading_weight
    #     heading_reward = heading_weight_tensor if heading_proj[env_idx] > 0.8 else self.cfg.heading_weight * heading_proj[env_idx] / 0.8
        
    #     # 5. 动作惩罚
    #     actions_cost = torch.sum(self.actions[env_idx]**2)
    #     actions_penalty = self.cfg.actions_cost_scale * actions_cost
        
    #     # 6. 能耗惩罚
    #     dof_pos_scaled = torch_utils.maths.unscale(dof_pos[env_idx], 
    #                                               self.robot.data.soft_joint_pos_limits[0, :, 0], 
    #                                               self.robot.data.soft_joint_pos_limits[0, :, 1])
    #     electricity_cost = torch.sum(torch.abs(self.actions[env_idx] * dof_vel[env_idx] * self.cfg.dof_vel_scale) * self.motor_effort_ratio)
    #     energy_penalty = self.cfg.energy_cost_scale * electricity_cost
        
    #     # 7. 关节限制惩罚
    #     dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98)
        
    #     # 8. 死亡惩罚
    #     death_penalty = self.cfg.death_cost if self.reset_terminated[env_idx] else 0.0
        
    #     print(f"\n=== 奖励分解 (环境 {env_idx}) ===")
    #     print(f"进度奖励:     {progress_reward.item():.4f}")
    #     print(f"存活奖励:     {alive_reward:.4f}")
    #     print(f"直立奖励:     {up_reward:.4f} (up_proj: {up_proj[env_idx].item():.3f})")
    #     print(f"朝向奖励:     {heading_reward:.4f} (heading_proj: {heading_proj[env_idx].item():.3f})")
    #     print(f"动作惩罚:     -{actions_penalty.item():.4f} (actions_cost: {actions_cost.item():.4f})")
    #     print(f"能耗惩罚:     -{energy_penalty.item():.4f} (electricity_cost: {electricity_cost.item():.4f})")
    #     print(f"关节限制惩罚: -{dof_at_limit_cost.item():.4f}")
    #     print(f"死亡惩罚:     {death_penalty:.4f}")
    #     print(f"总奖励:       {self._compute_total_reward_debug(env_idx):.4f}")
    #     print("=" * 40)
    
    # def _compute_total_reward_debug(self, env_idx):
    #     """计算单个环境的总奖励用于调试"""
    #     import torch
    #     from isaacsim.core.utils.torch.rotations import compute_heading_and_up
        
    #     # 获取状态
    #     torso_position = self.robot.data.root_pos_w[env_idx:env_idx+1]
    #     torso_rotation = self.robot.data.root_quat_w[env_idx:env_idx+1]
    #     dof_vel = self.robot.data.joint_vel[env_idx:env_idx+1]
    #     dof_pos = self.robot.data.joint_pos[env_idx:env_idx+1]
        
    #     # 计算中间值
    #     to_target = self.targets[env_idx:env_idx+1] - torso_position
    #     to_target[:, 2] = 0.0
        
    #     torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
    #         torso_rotation, self.inv_start_rot[env_idx:env_idx+1], to_target, 
    #         self.basis_vec0[env_idx:env_idx+1], self.basis_vec1[env_idx:env_idx+1], 2
    #     )
        
    #     # 计算奖励组件
    #     progress_reward = self.potentials[env_idx] - self.prev_potentials[env_idx]
    #     alive_reward = self.cfg.alive_reward_scale
        
    #     up_reward = self.cfg.up_weight if up_proj[0] > 0.93 else 0.0
    #     heading_reward = self.cfg.heading_weight if heading_proj[0] > 0.8 else self.cfg.heading_weight * heading_proj[0] / 0.8
        
    #     actions_cost = torch.sum(self.actions[env_idx]**2)
    #     actions_penalty = self.cfg.actions_cost_scale * actions_cost
        
    #     dof_pos_scaled = torch_utils.maths.unscale(dof_pos[0], 
    #                                               self.robot.data.soft_joint_pos_limits[0, :, 0], 
    #                                               self.robot.data.soft_joint_pos_limits[0, :, 1])
    #     electricity_cost = torch.sum(torch.abs(self.actions[env_idx] * dof_vel[0] * self.cfg.dof_vel_scale) * self.motor_effort_ratio)
    #     energy_penalty = self.cfg.energy_cost_scale * electricity_cost
        
    #     dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98)
    #     death_penalty = self.cfg.death_cost if self.reset_terminated[env_idx] else 0.0
        
    #     total = (progress_reward + alive_reward + up_reward + heading_reward 
    #             - actions_penalty - energy_penalty - dof_at_limit_cost + death_penalty)
        
    #     return total.item()

