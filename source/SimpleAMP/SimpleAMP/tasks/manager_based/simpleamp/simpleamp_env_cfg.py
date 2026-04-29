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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
    HfRandomUniformTerrainCfg,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.terrains as terrain_gen

from . import mdp

##
# Pre-defined configs
##

from SimpleAMP.assets.robots.ths_23dof import THS23DOF_CFG
from SimpleAMP import ROOT_DIR

AMP_INPUT_STEPS = 3


# 地形配置-一半崎岖
COBBLESTONE_ROAD_CFG = TerrainGeneratorCfg(
    size=(5.0, 5.0),  # 单块地形尺寸：8m × 8m（原注释写2.0但通常用8.0，按需调整）
    border_width=20.0,  # 地形边界宽度（防止机器人走出场景）
    num_rows=40,  # 地形行数（沿Y轴）
    num_cols=40,  # 地形列数（沿X轴）
    horizontal_scale=0.1,  # 水平方向网格分辨率（米/顶点）
    vertical_scale=0.005,  # 垂直方向高度缩放系数
    slope_threshold=0.75,  # 坡度阈值（超过此值视为不可行走区域）
    difficulty_range=(0.0, 1.0),  # 难度范围（0=最简单，1=最困难）
    use_cache=False,  # 禁用地形缓存（每次重新生成）
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.5),  # 平坦地形占比50%
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.50, noise_range=(0.0, 0.02), noise_step=0.02, border_width=0.25
        ),
    },
)


##
# 场景定义
##
@configclass
class SimpleampSceneCfg(InteractiveSceneCfg):

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

    robot: ArticulationCfg = THS23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,  # Same with decimation
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,  # 光照强度
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(10000.0, 10000.0),
        motion_data_dir=str(ROOT_DIR / "robot_assets/ths_23dof/motion_data"),
        motion_data_weights={
            "A1-_Stand_stageii": 1.0,
            "B10_-__Walk_turn_left_45_stageii": 1.0,
            "B14_-__Walk_turn_right_45_t2_stageii": 2.0,
            "B22_-__side_step_left_stageii": 1.0,
            "B23_-__side_step_right_stageii": 1.0,
            "C4_-_run_to_walk_a_stageii": 1.0,
            "C5_-_walk_to_run_stageii": 1.0,
            "daotuizou_000": 1.0,
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.35, n_max=0.35))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.75, n_max=1.75))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测组定义
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class DiscriminatorCfg(ObsGroup):
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
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
        amp_data_disc_obs = ObsTerm(
            func=mdp.amp_data_discriminator_obs, params={"command_name": "motion", "n_steps": AMP_INPUT_STEPS}
        )

    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()


# 随机事件配置类
@configclass
class EventCfg:

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
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 3.0),  # 基座质量加上该偏移不能小于0
            "operation": "add",
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "base_link"]),
            "com_range": {"x": (-0.06, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
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
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
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

    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    # 从参考轨迹重置（已禁用）
    # reset_from_ref = EventTerm(
    #     func=mdp.reset_from_ref,
    #     mode="reset",
    #     params=MISSING
    # )

    # # interval

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
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

    reset_robot_joints = EventTerm(
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

    # -- 主任务，线速度奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # -- 主任务，角速度跟踪奖励权重（z轴指数形式）
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # -- 低速前进起步奖励：专门修正 0~0.5m/s 段只前倾不走的问题
    low_speed_forward_track = RewTerm(
        func=mdp.low_speed_forward_track_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.12,
            "min_cmd": 0.05,
            "max_cmd": 0.6,
            "lateral_command_threshold": 0.15,
            "yaw_command_threshold": 0.2,
        },
    )

    # 定义角速度跟踪奖励 (L2/线性衰减版)
    # track_ang_vel_z_l2 = RewTerm(func=mdp.track_ang_vel_z_l2,weight=1.,params={"command_name": "base_velocity"})

    # -- 存活奖励
    alive = RewTerm(func=mdp.is_alive, weight=0.0)

    # -- 基座相关惩罚
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)  # 基座Z 轴 上下线速度
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)  # 基座XY轴运动惩罚 -0.1
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)  # 基座平面姿态惩罚

    # 基座高度 奖励
    base_height = RewTerm(func=mdp.base_height_exp, weight=0.2, params={"target_height": 0.73, "std": 0.05})

    # -- 关节类的惩罚
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)  # 关节速度惩罚
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # 关节加速度惩罚
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)  # 动作变化率惩罚
    # smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0)                                   # # 惩罚动作的变化量
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-50.0)  # 关节位置限位惩罚
    joint_energy = RewTerm(func=mdp.joint_energy, weight=-2e-5)  # 关节能耗惩罚
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=-1e-3)  # 关节正则化惩罚
    # joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2,weight=-1e-5 )                       # 关节扭矩惩罚

    # -- 惩罚髋关节偏航偏离默认值，抑制行走外八字
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw_joint",
                ],
            )
        },
    )

    # -- 惩罚腰部偏离默认值，抑制 torso 扭转摆动
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.75,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "torso_joint",
                ],
            )
        },
    )

    # -- 惩罚机器人在低速移动时的身体摇摆/晃动
    low_speed_sway_penalty = RewTerm(
        func=mdp.low_speed_sway_penalty,
        weight=-5e-2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    # -- 脚部滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # -- 脚部绊倒惩罚
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    # 惩罚机器人脚踝之间的横向Y距离过近或过远
    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,  # feet_too_near,
        weight=-10,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "threshold": (0.2, 0.60),
        },  # 设定期望的脚踝间距范围
    )

    # 手臂惩罚
    # 允许手臂微调平衡，但限制大幅度乱挥
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                ],
            )
        },
    )

    # -- 惩罚手臂roll关节偏离默认值，抑制手臂过度外展
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                ],
            )
        },
    )

    # 惩罚双腿过渡靠近内侧导致绊倒
    hip_roll_inner_side = RewTerm(func=mdp.hip_roll_inner_side, weight=-10)

    # 手臂跟踪腿部协调奖励项
    shoulder_thigh_coordination = RewTerm(func=mdp.shoulder_thigh_coordination, weight=0.5)

    # 惩罚脚掌 roll 过度外展导致不稳定
    ankle_roll_penalty = RewTerm(
        func=mdp.ankle_roll_boundary,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # # -- 脚腾空时间奖励
    # feet_air_time_positive_biped = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.4},
    # )

    # -- 落地冲击惩罚
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=-4e-5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
        },
    )

    # -- 惩罚机器人身体非脚部区域与环境的接触
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """MDP 的终止条件项配置。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["torso_link", ".*_hip_.*_link", ".*_elbow_link"],
            ),
            "threshold": 1.0,
        },
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": math.radians(60.0),
        },
    )


@configclass
class CurriculumCfg:
    lin_vel_command_levels = CurrTerm(
        func=mdp.adaptive_lin_vel_cmd_levels,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "command_name": "base_velocity",
            "min_lin_vel_x_range": (-0.5, 1.0),
            "min_lin_vel_y_range": (-0.3, 0.3),
            "max_lin_vel_x_range": (-1.0, 3.0),
            "max_lin_vel_y_range": (-0.8, 0.8),
            "lin_vel_x_step": 0.15,
            "lin_vel_y_step": 0.1,
            "increase_threshold": 0.75,
            "decrease_threshold": 0.5,
            "min_mean_episode_length_for_increase": 800.0,
            "update_interval_episodes": 1,
        },
    )

    # ang_vel_command_levels = CurrTerm(
    #     func=mdp.ang_vel_cmd_levels,
    #     params={
    #         "reward_term_name": "track_ang_vel_z_exp",
    #         "command_name": "base_velocity",
    #         "max_ang_vel_z_range": (-1.0, 1.0),
    #         "ang_vel_z_step": 0.1,
    #         "performance_threshold": 0.75,
    #         "update_interval_episodes": 1,
    #     },
    # )


@configclass
class SimpleampEnvCfg(ManagerBasedRLEnvCfg):
    scene: SimpleampSceneCfg = SimpleampSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4  # 降采样：每 4 个仿真步执行一次控制指令
        self.episode_length_s = 20  # 回合时长：每个训练回合持续 20 秒
        # simulation settings
        self.sim.dt = 1 / 400  # 仿真步长：设置物理仿真间隔为 1/200 = 0.005 秒 (即 200Hz 仿真频率)
        self.sim.render_interval = self.decimation


@configclass
class SimpleampEnvCfgPlay(SimpleampEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 为演示场景缩小规模
        self.scene.num_envs = 1  # 环境数量设为1（单机器人演示）
        self.scene.env_spacing = 2.5  # 环境间距设为2.5米（避免碰撞）
        self.episode_length_s = 40.0  # 单回合时长设为40秒

        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.5)  # X轴线速度范围
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # Y轴线速度范围
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)  # Z轴角速度范围

        self.observations.policy.enable_corruption = False  # 关闭观测噪声
        self.events.push_robot = None  # 禁用机器人推搡干扰
