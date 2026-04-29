from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from .commands import MotionCommand


def amp_data_discriminator_obs(
    env: ManagerBasedEnv,
    command_name: str,
    n_steps: int,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_data = command.sample_motion_seq_data(env, n_steps)
    yaw_quat = math_utils.yaw_quat(motion_data["root_quat_w"])
    root_quat_b = math_utils.quat_mul(
        math_utils.quat_conjugate(yaw_quat), motion_data["root_quat_w"]
    )  # shape: (num_envs, num_steps, 4)
    root_rotm_b = math_utils.matrix_from_quat(root_quat_b)  # (num_envs, num_steps, 3, 3)

    tan_vec = root_rotm_b[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = root_rotm_b[:, :, :, 2]  # (num_envs, num_steps, 3)
    root_tan_norm = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)
    amp_input = torch.cat(
        [
            root_tan_norm,
            motion_data["root_lin_vel_b"],
            motion_data["root_ang_vel_b"],
            motion_data["joint_pos"],
            motion_data["joint_vel"],
        ],
        dim=-1,
    )  # (num_envs, n_steps, d)
    return amp_input


def root_local_rot_tan_norm(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)

    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)

    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local)  # shape: (num_envs, num_steps, 3, 3)

    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)

    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs


def gait_phase(env: ManagerBasedEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):  # 检查环境是否已存在回合步数缓冲区
        env.episode_length_buf = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )  # 初始化回合步数计数器（每个子环境独立计数）

    global_phase = (
        (env.episode_length_buf * env.step_dt) % period / period
    )  # 计算全局相位：[0, 1) 区间内循环（基于仿真时间对周期取模）

    phase = torch.zeros(env.num_envs, 2, device=env.device)  # 初始化相位张量：形状 [num_envs, 2]
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)  # 第0维：正弦分量（2π周期）
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)  # 第1维：余弦分量（2π周期）
    return phase
    # 返回二维相位编码（用于步态控制与运动协调）
