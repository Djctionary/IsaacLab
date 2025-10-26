# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint and collect performance metrics for an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Performance test for RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to run performance test.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
import csv
from datetime import datetime
from typing import Any

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent and collect performance metrics."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    num_envs = env.unwrapped.num_envs
    iterations = args_cli.iterations

    # Create output directory for results
    results_dir = os.path.join(log_dir, "performance_test")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv_path = os.path.join(results_dir, f"performance_metrics_{timestamp}.csv")
    
    # Get terrain information if available
    terrain = env.unwrapped.scene.terrain
    has_terrain_info = hasattr(terrain, 'terrain_levels') and hasattr(terrain, 'terrain_types')
    
    if has_terrain_info:
        terrain_levels = terrain.terrain_levels.cpu().numpy()
        terrain_types = terrain.terrain_types.cpu().numpy()
    
    print(f"\n{'='*80}")
    print(f"Performance Test Started")
    print(f"{'='*80}")
    print(f"Task: {args_cli.task}")
    print(f"Number of Environments: {num_envs}")
    print(f"Number of Iterations: {iterations}")
    print(f"Steps per Iteration: {int(env.unwrapped.max_episode_length)}")
    print(f"Results will be saved to: {results_csv_path}")
    if has_terrain_info:
        print(f"Terrain tracking: ENABLED")
        print(f"  - Terrain levels range: [{terrain_levels.min()}, {terrain_levels.max()}]")
        print(f"  - Terrain types range: [{terrain_types.min()}, {terrain_types.max()}]")
    else:
        print(f"Terrain tracking: DISABLED (flat terrain or no terrain info)")
    print(f"{'='*80}\n")

    # Prepare CSV file
    with open(results_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'iteration', 
            'env_id',
        ]
        
        # Add terrain fields if available
        if has_terrain_info:
            fieldnames.extend(['terrain_level', 'terrain_type'])
        
        # Add velocity and performance fields
        fieldnames.extend([
            'avg_actual_lin_vel_x', 
            'avg_actual_lin_vel_y',
            'avg_actual_ang_vel_z',
            'avg_actual_lin_vel_xy_norm',
            'avg_command_lin_vel_x', 
            'avg_command_lin_vel_y',
            'avg_command_ang_vel_z',
            'avg_command_lin_vel_xy_norm',
            'lin_vel_tracking_error',
            'ang_vel_tracking_error',
            'fell_down',
            'episode_length'
        ])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Run performance test for multiple iterations
        for iteration in range(iterations):
            print(f"\n{'='*80}")
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"{'='*80}")
            
            # reset environment
            obs, _ = env.get_observations()
            
            # Storage for metrics
            actual_lin_vel_x_sum = torch.zeros(num_envs, device=env.unwrapped.device)
            actual_lin_vel_y_sum = torch.zeros(num_envs, device=env.unwrapped.device)
            actual_ang_vel_z_sum = torch.zeros(num_envs, device=env.unwrapped.device)
            command_lin_vel_x_sum = torch.zeros(num_envs, device=env.unwrapped.device)
            command_lin_vel_y_sum = torch.zeros(num_envs, device=env.unwrapped.device)
            command_ang_vel_z_sum = torch.zeros(num_envs, device=env.unwrapped.device)
            fell_down = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
            step_count = torch.zeros(num_envs, device=env.unwrapped.device)
            
            timestep = 0
            max_steps = int(env.unwrapped.max_episode_length)
            
            # simulate environment for one episode
            while timestep < max_steps and simulation_app.is_running():
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # Get robot asset
                    robot = env.unwrapped.scene["robot"]
                    
                    # Get actual velocities (in body frame)
                    actual_lin_vel = robot.data.root_lin_vel_b[:, :2]  # [num_envs, 2] (x, y)
                    actual_ang_vel = robot.data.root_ang_vel_b[:, 2]   # [num_envs] (z)
                    
                    # Get command velocities
                    command_vel = env.unwrapped.command_manager.get_command("base_velocity")
                    command_lin_vel = command_vel[:, :2]  # [num_envs, 2] (x, y)
                    command_ang_vel = command_vel[:, 2]   # [num_envs] (z)
                    
                    # Check for falling using the environment's termination manager
                    base_contact = env.unwrapped.termination_manager.terminated
                    
                    # Accumulate metrics
                    actual_lin_vel_x_sum += actual_lin_vel[:, 0]
                    actual_lin_vel_y_sum += actual_lin_vel[:, 1]
                    actual_ang_vel_z_sum += actual_ang_vel
                    command_lin_vel_x_sum += command_lin_vel[:, 0]
                    command_lin_vel_y_sum += command_lin_vel[:, 1]
                    command_ang_vel_z_sum += command_ang_vel
                    fell_down = fell_down | base_contact
                    step_count += 1
                    
                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, _, _, _ = env.step(actions)
                
                if args_cli.video:
                    timestep += 1
                    # Exit the play loop after recording one video
                    if timestep == args_cli.video_length:
                        break
                else:
                    timestep += 1

                # time delay for real-time evaluation
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # Calculate averages and write to CSV
            print(f"\nSaving data for iteration {iteration + 1}...")
            for env_id in range(num_envs):
                avg_actual_lin_vel_x = (actual_lin_vel_x_sum[env_id] / step_count[env_id]).item()
                avg_actual_lin_vel_y = (actual_lin_vel_y_sum[env_id] / step_count[env_id]).item()
                avg_actual_ang_vel_z = (actual_ang_vel_z_sum[env_id] / step_count[env_id]).item()
                avg_command_lin_vel_x = (command_lin_vel_x_sum[env_id] / step_count[env_id]).item()
                avg_command_lin_vel_y = (command_lin_vel_y_sum[env_id] / step_count[env_id]).item()
                avg_command_ang_vel_z = (command_ang_vel_z_sum[env_id] / step_count[env_id]).item()
                
                # Calculate norms
                avg_actual_lin_vel_xy_norm = np.sqrt(avg_actual_lin_vel_x**2 + avg_actual_lin_vel_y**2)
                avg_command_lin_vel_xy_norm = np.sqrt(avg_command_lin_vel_x**2 + avg_command_lin_vel_y**2)
                
                # Calculate tracking errors
                lin_vel_error = np.sqrt(
                    (avg_actual_lin_vel_x - avg_command_lin_vel_x)**2 + 
                    (avg_actual_lin_vel_y - avg_command_lin_vel_y)**2
                )
                ang_vel_error = abs(avg_actual_ang_vel_z - avg_command_ang_vel_z)
                
                # Prepare row data
                row_data: dict[str, Any] = {
                    'iteration': iteration + 1,
                    'env_id': env_id,
                }
                
                # Add terrain info if available
                if has_terrain_info:
                    row_data['terrain_level'] = int(terrain_levels[env_id])
                    row_data['terrain_type'] = int(terrain_types[env_id])
                
                # Add velocity and performance metrics
                row_data['avg_actual_lin_vel_x'] = f"{avg_actual_lin_vel_x:.4f}"
                row_data['avg_actual_lin_vel_y'] = f"{avg_actual_lin_vel_y:.4f}"
                row_data['avg_actual_ang_vel_z'] = f"{avg_actual_ang_vel_z:.4f}"
                row_data['avg_actual_lin_vel_xy_norm'] = f"{avg_actual_lin_vel_xy_norm:.4f}"
                row_data['avg_command_lin_vel_x'] = f"{avg_command_lin_vel_x:.4f}"
                row_data['avg_command_lin_vel_y'] = f"{avg_command_lin_vel_y:.4f}"
                row_data['avg_command_ang_vel_z'] = f"{avg_command_ang_vel_z:.4f}"
                row_data['avg_command_lin_vel_xy_norm'] = f"{avg_command_lin_vel_xy_norm:.4f}"
                row_data['lin_vel_tracking_error'] = f"{lin_vel_error:.4f}"
                row_data['ang_vel_tracking_error'] = f"{ang_vel_error:.4f}"
                row_data['fell_down'] = int(fell_down[env_id].item())
                row_data['episode_length'] = int(step_count[env_id].item())
                
                writer.writerow(row_data)
            
            # Print summary statistics for this iteration
            print(f"\nIteration {iteration + 1} Summary Statistics:")
            print(f"  Fallen Environments: {fell_down.sum().item()}/{num_envs}")
            print(f"  Fall Rate: {fell_down.sum().item() / num_envs * 100:.2f}%")
            print(f"  Avg Actual Linear Velocity Norm: {(torch.sqrt(actual_lin_vel_x_sum**2 + actual_lin_vel_y_sum**2) / step_count).mean().item():.4f} m/s")
            print(f"  Avg Command Linear Velocity Norm: {(torch.sqrt(command_lin_vel_x_sum**2 + command_lin_vel_y_sum**2) / step_count).mean().item():.4f} m/s")
            print(f"  Avg Actual Angular Velocity: {(actual_ang_vel_z_sum / step_count).mean().item():.4f} rad/s")
            print(f"  Avg Command Angular Velocity: {(command_ang_vel_z_sum / step_count).mean().item():.4f} rad/s")

    print(f"\n{'='*80}")
    print(f"Performance Test Completed!")
    print(f"Results saved to: {results_csv_path}")
    print(f"{'='*80}\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore
    # close sim app
    simulation_app.close()
