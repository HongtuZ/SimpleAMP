from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from SimpleAMP.utils.motion_loader import MotionLoader


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.motion_loader = MotionLoader(
            motion_data_dir=cfg.motion_data_dir, motion_data_weights=cfg.motion_data_weights, device=self.device
        )

    def sample_motion_seq_data(self, env: ManagerBasedRLEnv, n_steps: int):
        motion_ids = self.motion_loader.sample_motion_ids(env.num_envs)
        motion_seq_times = self.motion_loader.sample_motion_seq_times(motion_ids, n_steps, env.step_dt)
        return self.motion_loader.get_motion_seq_data(
            motion_ids=motion_ids,
            motion_seq_times=motion_seq_times,
            joint_names=self.robot.joint_names,
            body_names=self.robot.body_names,
        )

    def sample_motion_data(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        motion_ids = self.motion_loader.sample_motion_ids(len(env_ids))
        motion_times = self.motion_loader.sample_motion_times(motion_ids, truncate_time_end=env.step_dt)
        return self.motion_loader.get_motion_data(
            motion_ids=motion_ids,
            motion_times=motion_times,
            joint_names=self.robot.joint_names,
            body_names=self.robot.body_names,
        )

    # No use
    @property
    def command(self) -> torch.Tensor:
        return torch.zeros((self.num_envs, 1))

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids: Sequence[int]):
        return

    def _update_command(self):
        return


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING
    motion_data_dir: str = MISSING
    motion_data_weights: dict | None = None
