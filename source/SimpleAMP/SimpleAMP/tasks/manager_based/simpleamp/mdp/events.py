from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal
import random

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from .commands import MotionCommand


def reset_from_ref(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: DeformableObject = env.scene[asset_cfg.name]
    motion_command: MotionCommand = env.command_manager.get_term(command_name)
    motion_data = motion_command.sample_motion_data(env, env_ids)
    ref_root_pos_w = motion_data["root_pos_w"] + env.scene.env_origins[env_ids]
    ref_root_pos_w[..., 2] += 0.05  # lift up a bit to avoid penetration
    ref_root_quat_w = motion_data["root_quat_w"]
    ref_root_lin_vel_w = motion_data["root_lin_vel_w"]
    ref_root_ang_vel_w = motion_data["root_ang_vel_w"]

    joint_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    ref_joint_pos = torch.clamp(motion_data["joint_pos"], joint_limits[..., 0], joint_limits[..., 1])
    ref_joint_vel = torch.clamp(motion_data["joint_vel"], -joint_vel_limits, joint_vel_limits)

    asset.write_root_pose_to_sim(torch.cat([ref_root_pos_w, ref_root_quat_w], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([ref_root_lin_vel_w, ref_root_ang_vel_w], dim=-1), env_ids=env_ids)
    asset.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel, env_ids=env_ids)
