# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
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

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import SimpleAMP.tasks  # noqa: F401
from pathlib import Path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play with Motion Data."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # check the motion data nums
    motion_data_dir = Path(env_cfg.motion_data.motion_data_dir)
    if not motion_data_dir.exists():
        raise ValueError(f"Motion data directory {motion_data_dir} does not exist.")
    motion_data_files = list(motion_data_dir.rglob("*.pkl"))
    all_motion_names = [p.stem for p in motion_data_files]
    motion_names = []
    if env_cfg.motion_data.motion_data_weights:
        for motion_name in env_cfg.motion_data.motion_data_weights.keys():
            if motion_name not in all_motion_names:
                raise ValueError(
                    f"Motion name {motion_name} specified in motion_data_weights does not exist in motion data directory."
                )
            else:
                motion_names.append(motion_name)
    else:
        motion_names = all_motion_names
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = len(motion_names)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # specify directory for logging experiments
    log_dir = os.path.join("logs", "replay_motion_data")
    log_dir = os.path.abspath(log_dir)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = env.unwrapped
    dt = env.step_dt

    # reset environment
    timestep = 0
    # simulate environment
    motion_data = env.motion_data
    motion_ids = torch.arange(len(motion_names), dtype=torch.int, device=env.device)
    motion_times = torch.zeros(len(motion_names), dtype=torch.float, device=env.device)
    robot = env.scene["robot"]
    while simulation_app.is_running():
        start_time = time.time()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        motion_state = motion_data.get_motion_data(
            motion_ids, motion_times, robot.joint_names, robot.body_names
        )
        ref_root_pos_w = motion_state["root_pos_w"] + env.scene.env_origins
        ref_root_quat_w = motion_state["root_quat_w"]
        ref_root_lin_vel_w = motion_state["root_lin_vel_w"]
        ref_root_ang_vel_w = motion_state["root_ang_vel_w"]

        joint_limits = robot.data.soft_joint_pos_limits
        joint_vel_limits = robot.data.soft_joint_vel_limits
        ref_joint_pos = torch.clamp(
            motion_state["joint_pos"], joint_limits[..., 0], joint_limits[..., 1]
        )
        ref_joint_vel = torch.clamp(
            motion_state["joint_vel"], -joint_vel_limits, joint_vel_limits
        )

        robot.write_root_pose_to_sim(
            torch.cat([ref_root_pos_w, ref_root_quat_w], dim=-1)
        )
        robot.write_root_velocity_to_sim(
            torch.cat([ref_root_lin_vel_w, ref_root_ang_vel_w], dim=-1)
        )
        robot.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel)
        robot.update(dt)
        # update articulation kinematics
        env.scene.write_data_to_sim()
        env.sim.forward()
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        motion_times += dt
        motion_times %= motion_data.motion_durations

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
