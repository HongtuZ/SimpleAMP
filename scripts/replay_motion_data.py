import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument(
    "--motion_dir",
    "-d",
    type=str,
    required=True,
    help="The path to the motion directory.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from SimpleAMP.assets.robots.ths_23dof import THS23DOF_CFG
from SimpleAMP.utils.motion_data import MotionData


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = THS23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator():
    # Load motion data
    motion_data = MotionData(
        motion_data_dir=args_cli.motion_dir,
        device=args_cli.device,
    )

    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=motion_data.motion_durations.shape[0], env_spacing=2.5
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion_ids = torch.arange(scene.num_envs, dtype=torch.int, device=sim.device)
    motion_times = torch.zeros(scene.num_envs, dtype=torch.float, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        motion_state = motion_data.get_motion_data(
            motion_ids, motion_times, robot.joint_names, robot.body_names
        )
        ref_root_pos_w = motion_state["root_pos_w"] + scene.env_origins
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

        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)
        # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)
        motion_times += sim_dt
        motion_times %= motion_data.motion_durations


def main():
    # Run the simulator
    run_simulator()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
