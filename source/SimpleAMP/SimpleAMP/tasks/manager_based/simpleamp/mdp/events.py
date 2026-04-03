# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from SimpleAMP.utils.motion_data import MotionData


def reset_from_ref(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: DeformableObject = env.scene[asset_cfg.name]
    dt = env.cfg.sim.dt * env.cfg.decimation
    motion_data: MotionData = env.motion_data
    motion_ids = motion_data.sample_motion_ids(len(env_ids))
    motion_times = motion_data.sample_motion_times(motion_ids, truncate_time_end=dt)
    motion_state = motion_data.get_motion_data(
        motion_ids, motion_times, asset.joint_names, asset.body_names
    )  # (num_envs, [n_bodies], dim)
    ref_root_pos_w = motion_state["root_pos_w"] + env.scene.env_origins[env_ids]
    ref_root_pos_w[..., 2] += 0.05  # lift up a bit to avoid penetration
    ref_root_quat_w = motion_state["root_quat_w"]
    ref_root_lin_vel_w = motion_state["root_lin_vel_w"]
    ref_root_ang_vel_w = motion_state["root_ang_vel_w"]

    joint_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    ref_joint_pos = torch.clamp(
        motion_state["joint_pos"], joint_limits[..., 0], joint_limits[..., 1]
    )
    ref_joint_vel = torch.clamp(
        motion_state["joint_vel"], -joint_vel_limits, joint_vel_limits
    )

    asset.write_root_pose_to_sim(
        torch.cat([ref_root_pos_w, ref_root_quat_w], dim=-1), env_ids=env_ids
    )
    asset.write_root_velocity_to_sim(
        torch.cat([ref_root_lin_vel_w, ref_root_ang_vel_w], dim=-1), env_ids=env_ids
    )
    asset.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel, env_ids=env_ids)
