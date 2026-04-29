# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _expand_command_range(
    current_range: tuple[float, float] | list[float],
    step_size: float,
    target_range: tuple[float, float],
) -> tuple[float, float]:
    """按对称步长扩展命令范围，并裁剪到目标上限。"""
    current = torch.tensor(current_range, dtype=torch.float)
    delta = torch.tensor([-step_size, step_size], dtype=torch.float)
    target_min = min(target_range)
    target_max = max(target_range)
    updated = torch.clamp(current + delta, min=target_min, max=target_max)
    return tuple(updated.tolist())


def _shrink_command_range(
    current_range: tuple[float, float] | list[float],
    step_size: float,
    minimum_range: tuple[float, float],
) -> tuple[float, float]:
    """按对称步长收缩命令范围，并限制不小于最小课程范围。"""
    current = torch.tensor(current_range, dtype=torch.float)
    minimum = torch.tensor(minimum_range, dtype=torch.float)
    updated = torch.empty_like(current)
    updated[0] = torch.minimum(current[0] + step_size, minimum[0])
    updated[1] = torch.maximum(current[1] - step_size, minimum[1])
    return tuple(updated.tolist())


def _should_update_curriculum(env: ManagerBasedRLEnv, update_interval_episodes: int) -> bool:
    """仅在整回合边界更新课程，避免训练过程中频繁抖动。"""
    if update_interval_episodes <= 0:
        raise ValueError("update_interval_episodes must be >= 1.")
    return env.common_step_counter % (env.max_episode_length * update_interval_episodes) == 0


def _mean_episode_reward(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str) -> torch.Tensor:
    """计算指定 reward term 的平均每秒回报。"""
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
    return reward / env.max_episode_length_s


def _update_logger_style_episode_length_stats(env: ManagerBasedRLEnv) -> None:
    """按 logger 的口径维护 completed episode length 的滑动窗口。"""
    if not hasattr(env, "episode_length_buf"):
        return

    if not hasattr(env, "_curriculum_lenbuffer"):
        env._curriculum_lenbuffer = deque(maxlen=100)
        env._curriculum_cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        env._curriculum_prev_episode_length_buf = env.episode_length_buf.clone()
        env._curriculum_episode_length_last_step = -1

    current_step = int(env.common_step_counter)
    if env._curriculum_episode_length_last_step == current_step:
        return

    env._curriculum_episode_length_last_step = current_step
    env._curriculum_cur_episode_length += 1

    current_episode_length_buf = env.episode_length_buf.clone()
    completed_mask = current_episode_length_buf < env._curriculum_prev_episode_length_buf

    if torch.any(completed_mask):
        completed_lengths = env._curriculum_cur_episode_length[completed_mask].detach().cpu().tolist()
        env._curriculum_lenbuffer.extend(completed_lengths)
        env._curriculum_cur_episode_length[completed_mask] = 0

    env._curriculum_prev_episode_length_buf.copy_(current_episode_length_buf)


def _mean_episode_length(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """读取与 logger 完全同口径的平均回合长度，单位为 step。"""
    _update_logger_style_episode_length_stats(env)
    if hasattr(env, "_curriculum_lenbuffer") and len(env._curriculum_lenbuffer) > 0:
        return torch.tensor(sum(env._curriculum_lenbuffer) / len(env._curriculum_lenbuffer), device=env.device)
    return torch.tensor(0.0, device=env.device)


def adaptive_lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    command_name: str = "base_velocity",
    min_lin_vel_x_range: tuple[float, float] | None = None,
    min_lin_vel_y_range: tuple[float, float] | None = None,
    max_lin_vel_x_range: tuple[float, float] = (-1.0, 3.0),
    max_lin_vel_y_range: tuple[float, float] = (-0.8, 0.8),
    lin_vel_x_step: float = 0.15,
    lin_vel_y_step: float = 0.1,
    increase_threshold: float = 0.5,
    decrease_threshold: float = 0.35,
    min_mean_episode_length_for_increase: float = 0.0,
    update_interval_episodes: int = 1,
) -> torch.Tensor:
    """根据线速度跟踪质量对 x/y 命令范围做可升可降课程。

    当回报高于升档阈值时，向更大速度范围扩展一档；
    当回报低于降档阈值时，回退一档；
    中间区间保持当前档位，避免课程在边界附近来回抖动。
    """
    if decrease_threshold >= increase_threshold:
        raise ValueError("decrease_threshold must be smaller than increase_threshold.")

    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges

    if not hasattr(env, "_adaptive_lin_vel_curriculum_initialized"):
        env._adaptive_lin_vel_curriculum_initialized = True
        env._adaptive_lin_vel_x_min = tuple(min_lin_vel_x_range or ranges.lin_vel_x)
        env._adaptive_lin_vel_y_min = tuple(min_lin_vel_y_range or ranges.lin_vel_y)
        env._adaptive_lin_vel_x_max = tuple(max_lin_vel_x_range)
        env._adaptive_lin_vel_y_max = tuple(max_lin_vel_y_range)

        ranges.lin_vel_x = list(env._adaptive_lin_vel_x_min)
        ranges.lin_vel_y = list(env._adaptive_lin_vel_y_min)

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = _mean_episode_reward(env, env_ids, reward_term_name)
    mean_episode_length = _mean_episode_length(env, env_ids)

    if _should_update_curriculum(env, update_interval_episodes):
        if (
            reward > reward_term.weight * increase_threshold
            and mean_episode_length >= min_mean_episode_length_for_increase
        ):
            ranges.lin_vel_x = _expand_command_range(ranges.lin_vel_x, lin_vel_x_step, env._adaptive_lin_vel_x_max)
            ranges.lin_vel_y = _expand_command_range(ranges.lin_vel_y, lin_vel_y_step, env._adaptive_lin_vel_y_max)
        elif reward < reward_term.weight * decrease_threshold:
            ranges.lin_vel_x = _shrink_command_range(ranges.lin_vel_x, lin_vel_x_step, env._adaptive_lin_vel_x_min)
            ranges.lin_vel_y = _shrink_command_range(ranges.lin_vel_y, lin_vel_y_step, env._adaptive_lin_vel_y_min)

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    command_name: str = "base_velocity",
    max_lin_vel_x_range: tuple[float, float] = (-1.0, 3.0),
    max_lin_vel_y_range: tuple[float, float] = (-0.8, 0.8),
    lin_vel_x_step: float = 0.15,
    lin_vel_y_step: float = 0.1,
    performance_threshold: float = 0.75,
    update_interval_episodes: int = 1,
) -> torch.Tensor:
    """根据线速度跟踪质量逐步放宽 x/y 方向的速度命令范围。"""
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = _mean_episode_reward(env, env_ids, reward_term_name)

    if _should_update_curriculum(env, update_interval_episodes):
        if reward > reward_term.weight * performance_threshold:
            ranges.lin_vel_x = _expand_command_range(ranges.lin_vel_x, lin_vel_x_step, max_lin_vel_x_range)
            ranges.lin_vel_y = _expand_command_range(ranges.lin_vel_y, lin_vel_y_step, max_lin_vel_y_range)

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z_exp",
    command_name: str = "base_velocity",
    max_ang_vel_z_range: tuple[float, float] = (-1.0, 1.0),
    ang_vel_z_step: float = 0.1,
    performance_threshold: float = 0.75,
    update_interval_episodes: int = 1,
) -> torch.Tensor:
    """根据角速度跟踪质量逐步放宽偏航角速度命令范围。"""
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = _mean_episode_reward(env, env_ids, reward_term_name)

    if _should_update_curriculum(env, update_interval_episodes):
        if reward > reward_term.weight * performance_threshold:
            ranges.ang_vel_z = _expand_command_range(ranges.ang_vel_z, ang_vel_z_step, max_ang_vel_z_range)

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def curriculum_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_force: float,
    threshold_height: float,
    decay_rate: float = 20,  # 力衰减速率
    min_force: float = 0.0,  # 最小力值
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
) -> None:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    current_heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    mean_height = torch.mean(current_heights).squeeze(-1)
    current_force = getattr(env, "_curriculum_force", max_force)

    if env.common_step_counter % env.max_episode_length == 0:
        if mean_height > threshold_height:
            force_reduction = decay_rate
            current_force = max(min_force, current_force - force_reduction)
            setattr(env, "_curriculum_force", current_force)
    return current_force


def curriculum_scale(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_scale: float,
    threshold_height: float,
    decay_scale: float = 0.05,  # 力衰减速率
    min_scale: float = 0.5,  # 最小力值
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
) -> None:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    current_heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    mean_height = torch.mean(current_heights).squeeze(-1)

    current_scale = getattr(env, "_curriculum_scale", max_scale)
    if env.common_step_counter % env.max_episode_length == 0:

        if mean_height > threshold_height:
            force_reduction = decay_scale
            current_scale = max(min_scale, current_scale - force_reduction)
            setattr(env, "_curriculum_scale", current_scale)
    env.cfg.actions.joint_pos.scale = current_scale
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    return current_scale


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
