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

def gait_phase(env: ManagerBasedEnv, period: float) -> torch.Tensor:                        
    if not hasattr(env, "episode_length_buf"):                                                   # 检查环境是否已存在回合步数缓冲区
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)  # 初始化回合步数计数器（每个子环境独立计数）
        
    global_phase = (env.episode_length_buf * env.step_dt) % period / period                      # 计算全局相位：[0, 1) 区间内循环（基于仿真时间对周期取模）

    phase = torch.zeros(env.num_envs, 2, device=env.device)                                      # 初始化相位张量：形状 [num_envs, 2]
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)                                       # 第0维：正弦分量（2π周期）
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)                                       # 第1维：余弦分量（2π周期）
    return phase      