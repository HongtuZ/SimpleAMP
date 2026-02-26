from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn
from .simpleamp_env_cfg import SimpleampEnvCfg
import torch
from SimpleAMP.utils.motion_data import MotionData

class SimpleampEnv(ManagerBasedRLEnv):
    cfg: SimpleampEnvCfg

    def __init__(self, cfg: SimpleampEnvCfg, render_mode: str | None = None, **kwargs):
        self.motion_data = MotionData(cfg.motion_data.motion_data_dir, cfg.motion_data.motion_data_weights, device=cfg.sim.device)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    