from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from SimpleAMP.utils.motion_data import MotionData
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    
def amp_data_discriminator_obs(env: ManagerBasedEnv, n_steps: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                               ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    motion_data: MotionData = env.motion_data
    motion_ids = motion_data.sample_motion_ids(env.num_envs)
    motion_seq_times = motion_data.sample_motion_seq_times(motion_ids, n_steps, env.step_dt)
    motion_seq_data = motion_data.get_motion_seq_data(motion_ids, motion_seq_times, asset.joint_names, asset.body_names) # (num_envs, [n_bodies], n_steps, dim)
    amp_input = torch.cat([
        motion_seq_data['root_lin_vel_b'],
        motion_seq_data['root_ang_vel_b'],
        motion_seq_data['joint_pos'],
        motion_seq_data['joint_vel'],
        ], dim=-1).to(env.device) # (num_envs, n_steps, d)
    return amp_input
