# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, MeshPlaneTerrainCfg, HfRandomUniformTerrainCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from SimpleAMP.assets.robots.ths_23dof import THS23DOF_CFG

AMP_INPUT_STEPS = 3

##
# Scene definition
##

COBBLESTONE_ROAD_CFG = TerrainGeneratorCfg(
    size=(5.0, 5.0),                                              # 单块地形尺寸：8m × 8m（原注释写2.0但通常用8.0，按需调整）
    border_width=20.0,                                            # 地形边界宽度（防止机器人走出场景）
    num_rows=40,                                                  # 地形行数（沿Y轴）
    num_cols=40,                                                  # 地形列数（沿X轴）
    horizontal_scale=0.1,                                         # 水平方向网格分辨率（米/顶点）
    vertical_scale=0.005,                                         # 垂直方向高度缩放系数
    slope_threshold=0.75,                                         # 坡度阈值（超过此值视为不可行走区域）
    difficulty_range=(0.0, 1.0),                                  # 难度范围（0=最简单，1=最困难）
    use_cache=False,                                              # 禁用地形缓存（每次重新生成）
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.5),          # 平坦地形占比50%
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.50, noise_range=(0.0, 0.02), noise_step=0.02, border_width=0.25
        ),
    },
)

@configclass
class SimpleampSceneCfg(InteractiveSceneCfg):
    """Configuration for a legged robot scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = THS23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.35, n_max=0.35))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.75, n_max=1.75))
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        def __post_init__(self) -> None:
            self.history_length = 1
            self.enable_corruption = True 
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True
    
    critic: CriticCfg = CriticCfg()

    @configclass
    class DiscriminatorCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = AMP_INPUT_STEPS
            self.flatten_history_dim = False
            
    disc: DiscriminatorCfg = DiscriminatorCfg()
            
    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        # This should be same with the above disc obs
        amp_data_disc_obs = ObsTerm(func=mdp.amp_data_discriminator_obs, params={'n_steps': AMP_INPUT_STEPS})
    
    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "base_link"]),
            "com_range": {"x": (-0.06, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}, # 0.02
        },
    )
    
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    
    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    
    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset_from_ref = EventTerm(
    #     func=mdp.reset_from_ref, 
    #     mode="reset",
    #     params=MISSING
    # )

    # # interval

    reset_base=EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints=EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 20.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-1.0, 1.0)}},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=10.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=10.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- reward
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    feet_air_time_positive_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), 
            "threshold": 0.4},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.5
        }
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=-1e-4)
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=-1e-3)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0)                                   # # 惩罚动作的变化量   

    low_speed_sway_penalty = RewTerm(
        func=mdp.low_speed_sway_penalty,
        weight=-1e-2,
        params={"command_name": "base_velocity", "command_threshold": 0.1}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight= -0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=-5e-5,
        params={
            "sensor_cfg": SceneEntityCfg( "contact_forces", body_names=".*_ankle_roll_link",),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*"),
            "threshold": 1.0
        },
    )
    # -----------------------纯 RL控 奖励--------------------------------

    
    # -- 行走模式步态运动相关奖励项 
    gait_walk = RewTerm(
        func=mdp.feet_gait,
        weight=1.,  
        params={
            "period": 0.8,                                                                # 步态周期0.8秒
            "offset": [0.0, 0.5],                                                         # 左右足相位偏移（0.0表示同相，0.5表示反相）
            "threshold": 0.7,                                                            # 接触判断阈值
            "command_name": "base_velocity",                                              # 关联的速度指令名称
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),  # 传感器配置：监测脚踝滚转关节的接触力
        },
    )
    



    # 手臂跟踪腿部协调奖励项
    shoulder_thigh_coordination = RewTerm(func=mdp.shoulder_thigh_coordination, weight=0.5) 
    
    # 基座高度 奖励
    base_height = RewTerm(
        func=mdp.base_height_exp,
        weight=0.5, 
        params={"target_height": 0.74, "std": 0.05}
    )
 


    # -- 重惩罚不希望运动的关节
    joint_deviation_big = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_roll_.*",         # 腿部横滚关节
                    ".*_shoulder_roll_.*",    # 肩膀横滚关节   
                    ".*_elbow_.*",            # 手肘关节
                    ".*_shoulder_yaw_.*",     # 肩膀偏航关节
                    ".*_wrist_.*",            # 手腕关节
                    "torso_.*",               # 腰部
                ],
            )
        },
    )  
   

    # -- 轻微惩罚不希望运动的关节 
    joint_deviation_small = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    ".*_hip_yaw_.*",    # 腿部偏航关节
                    ".*_ankle_.*",      # 脚踝关节
                ],
            )
        },
    )  


    # 鼓励摆动相足部保持适当离地高度（避免拖地），权重1.0
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,                                                        # 高斯分布标准差（控制奖励敏感度 越大越不敏感，越小越敏感）
            "tanh_mult": 2.0,                                                   # tanh函数缩放系数（平滑奖励过渡）
            "target_height": 0.06,                                               # 目标离地高度0.1米
            "command_name": "base_velocity",                                              # 关联的速度指令名称
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),  # 作用关节：脚踝滚转
        },
    )  

    
    # 惩罚机器人脚踝之间的横向Y距离过近或过远
    feet_distance_y = RewTerm(
        func=mdp.feet_distance_y,     #feet_too_near, 
        weight=-2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "min": 0.2,
            "max": 0.3
        },
    )

    hip_roll_side = RewTerm(func=mdp.hip_roll_side, weight=-10)              # 惩罚 双腿过渡靠近内侧导致绊倒
    #ankle_roll_boundary = RewTerm(func=mdp.ankle_roll_boundary, weight=-10)  # 惩罚 双腿脚踝横滚出界

    # 惩罚脚踝关节横滚扭矩过大
    ankle_torque_limit = RewTerm(
        func=mdp.ankle_roll_torque_penalty,             # 假设你把上面函数放到了 mdp 模块
        weight=-1.0,                                    # 【关键】权重为负，使 1.0 变成 -1.0 的惩罚
        params={
            "threshold": 2.0,                           # 扭矩阀值
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
 
    # 左右腿对称性奖励
    leg_symmetry = RewTerm(
        func=mdp.leg_symmetry_reward,  # 确保该函数已导入到 mdp 模块
        weight=1.0,                    # 权重可根据实验调整
        params={
            "asset_cfg": SceneEntityCfg("robot"),"std": 0.1,               # 容忍度，越大越容易得分
        },
    )
    




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={
            "limit_angle": math.radians(60.0),
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

@configclass
class MotionDataCfg:
    """Motion data terms for the AMP."""
    motion_data_dir='/home/robot/hongtu/SimpleAMP/robot_assets/ths_23dof/motion_data'
    motion_data_weights=None
    # motion_data_weights={
    #     '127_03_stageii': 1.0,
    #     '127_04_stageii': 1.0,
    #     '127_06_stageii': 1.0,
    # }
    

##
# Environment configuration
##


@configclass
class SimpleampEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: SimpleampSceneCfg = SimpleampSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # AMP motion data setting
    motion_data: MotionDataCfg = MotionDataCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation

@configclass
class SimpleampEnvCfgPlay(SimpleampEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.push_robot = None