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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.terrains as terrain_gen

from . import mdp

##
# Pre-defined configs
##

from SimpleAMP.assets.robots.ths_23dof import THS23DOF_CFG

AMP_INPUT_STEPS = 3           # AMP 判别器输入最近连续的 3 帧运动状态




# 地形配置-一半崎岖
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),          # 平坦地形占比50%
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.50, noise_range=(0.0, 0.02), noise_step=0.02, border_width=0.25
        ),
    },
)



##
# 场景定义
##
@configclass
class SimpleampSceneCfg(InteractiveSceneCfg):                                           # 腿式机器人仿真场景配置类
    """用于腿式机器人场景的配置。"""                                                       

    # 地面地形                                                                          
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",                                                       # 地形在 USD 场景中的路径
        terrain_type="generator",                                                        # 地形类型：使用生成器（可选"plane"平面或"generator"生成器）
        terrain_generator=COBBLESTONE_ROAD_CFG,                                          # 指定地形生成器配置（当前使用鹅卵石道路）
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,                        # 最大初始地形难度等级（使用最后一行）

        collision_group=-1,                                                             # 碰撞组 ID（-1 表示默认组）
        physics_material=sim_utils.RigidBodyMaterialCfg(                                # 物理材质（影响摩擦与反弹）
            friction_combine_mode="multiply",                                           # 摩擦系数组合方式：相乘
            restitution_combine_mode="multiply",                                        # 恢复系数组合方式：相乘
            static_friction=1.0,                                                        # 静摩擦系数
            dynamic_friction=1.0,                                                       # 动摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(                                           # 视觉材质（仅影响渲染外观）
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                                                                                        # 材质定义文件路径（大理石砖纹理）
            project_uvw=True,                                                           # 启用 UVW 投影以正确贴图
            texture_scale=(0.25, 0.25),                                                 # 纹理缩放比例（越小纹理越密集）
        ),
        debug_vis=False,                                                                # 是否显示地形调试可视化（如网格线）
    )
    # 机器人                                                                             # 实例化指定构型的机器人
    robot: ArticulationCfg = THS23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")     # 将机器人放置到每个环境的命名空间下

    # 传感器                                                                             # 配置接触力传感器
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*",              # 监测机器人所有刚体的接触
                                      history_length=3,                                 # 保存最近 3 帧的接触历史
                                      track_air_time=True)                              # 记录离地时间（用于步态分析）

    # 光照                                                                              # 添加天空光照（HDRI 环境光）
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",                                                    # 光源在场景中的路径
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,                                                            # 光照强度
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                                                                                        # HDRI 天空盒纹理（提供环境光照）
        ),
    )




# MDP（马尔可夫决策过程）配置                      
                        

@configclass                        
class CommandsCfg:                  
    """MDP 的命令规格说明.""" 

    base_velocity = mdp.UniformVelocityCommandCfg(      # 基础速度命令配置（均匀分布）
        asset_name="robot",                             # 资产名称：机器人
        resampling_time_range=(10.0, 10.0),             # 重采样时间范围（秒）
        rel_standing_envs=0.02,                         # 站立环境的相对比例
        rel_heading_envs=1.0,                           # 航向命令环境的相对比例
        heading_command=True,                           # 启用航向命令
        heading_control_stiffness=0.5,                  # 航向控制刚度
        debug_vis=True,                                 # 启用调试可视化

        # 原地站立命令范围
        #ranges=mdp.UniformVelocityCommandCfg.Ranges(    # 速度命令的范围定义
        #    lin_vel_x=(0.0, 0.0),                      # 线速度 X 轴范围（米/秒）：后退 0.8 到前进 2.5
        #    lin_vel_y=(0.0, 0.0),                      # 线速度 Y 轴范围（米/秒）：左移 0.8 到右移 0.8
        #    ang_vel_z=(0.0, 0.0),                      # 角速度 Z 轴范围（弧度/秒）：左转 1.0 到右转 1.0
        #    heading=(0.0, 0.0)                         # 目标航向角范围（弧度）：-π 到 π
        #),

        # 行走速度命令范围
        ranges=mdp.UniformVelocityCommandCfg.Ranges(    # 速度命令的范围定义 
            lin_vel_x=(-1.0, 1.0),                      # 线速度 X 轴范围（米/秒）：后退 0.8 到前进 2.5
            lin_vel_y=(-0.8, 0.8),                      # 线速度 Y 轴范围（米/秒）：左移 0.8 到右移 0.8
            ang_vel_z=(-1.0, 1.0),                      # 角速度 Z 轴范围（弧度/秒）：左转 1.0 到右转 1.0
            heading=(-math.pi, math.pi)                  # 目标航向角范围（弧度）：-π 到 π
        ),

        # 跑步速度命令范围
        #ranges=mdp.UniformVelocityCommandCfg.Ranges(    # 速度命令的范围定义 
        #    lin_vel_x=(-1.0, 2.0),                      # 线速度 X 轴范围（米/秒）：后退 0.8 到前进 2.5
        #    lin_vel_y=(-0.8, 0.8),                      # 线速度 Y 轴范围（米/秒）：左移 0.8 到右移 0.8
        #    ang_vel_z=(-1.0, 1.0),                      # 角速度 Z 轴范围（弧度/秒）：左转 1.0 到右转 1.0
        #    heading=(-math.pi, math.pi)                  # 目标航向角范围（弧度）：-π 到 π
        #),


    )

@configclass                                            # 定义配置类装饰器
class ActionsCfg:                                       # 动作配置类
    """Action specifications for the MDP."""            # MDP 的动作规格说明

    joint_pos = mdp.JointPositionActionCfg(             # 关节位置动作配置
        asset_name="robot",                             # 资产名称：机器人
        joint_names=[".*"],                             # 关节名称匹配模式：匹配所有关节
        scale=0.25,                                     # 动作缩放比例
        use_default_offset=True                         # 使用默认偏移量（即零位偏移）
    )



@configclass                        
class ObservationsCfg:              
    """ MDP 的观测规格说明."""  
    # 基座角速度      3位
    # 投影重力向量    3位
    # 速度命令       3位
    # 关节位置       23位
    # 关节速度       23位
    # 上一时刻动作    23位
    # 步态相位      2位
    # 共计 80 位观测值（不包含历史帧）X 10

    # 动作缩放：0.25


    @configclass                                                       
    class PolicyCfg(ObsGroup):                                                                              # 策略组的观测配置（继承自 ObsGroup）
        """策略组的观测项定义."""  # 

        # 观测项列表（顺序保持不变）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.35, n_max=0.35))                # 基座角速度：添加均匀噪声 [-0.35, 0.35]
        projected_gravity = ObsTerm(                                                                        # 投影重力向量 添加均匀噪声 [-0.05, 0.05]
            func=mdp.projected_gravity,                                                                     
            noise=Unoise(n_min=-0.05, n_max=0.05),                                                         
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 速度命令：获取名为 "base_velocity" 的生成命令
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.03, n_max=0.03))                  # 相对关节位置：添加均匀噪声 [-0.03, 0.03]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.75, n_max=1.75))                  # 相对关节速度：添加均匀噪声 [-1.75, 1.75]
        actions = ObsTerm(func=mdp.last_action)                                                             # 上一时刻的动作
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})                                   # 行走步态相位：周期设为 0.8 秒
        #gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.5})                                   # 行走步态相位：周期设为 0.8 秒

        def __post_init__(self) -> None:                                                    # 初始化后处理函数
            self.history_length = 1                                                         # 历史观测长度设为 10
            self.enable_corruption = True                                                   # 启用观测数据损坏（模拟传感器噪声）
            self.concatenate_terms = True                                                   # 将所有观测项拼接成一个向量

    # 观测组定义
    policy: PolicyCfg = PolicyCfg()                                                         # 实例化策略观测组

 
 

    @configclass                    
    class CriticCfg(ObsGroup):       
        """# 评论家组的观测项定义（包含特权观测信息）"""  

        # 观测项列表（顺序保持不变）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)                                                       # 基座线速度（特权信息：真实值）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)                                                       # 基座角速度（特权信息：真实值）
        projected_gravity = ObsTerm(func=mdp.projected_gravity)                                             # 投影重力向量（特权信息：真实值）
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 速度命令：获取名为 "base_velocity" 的生成命令
        joint_pos = ObsTerm(func=mdp.joint_pos)                                                             # 绝对关节位置（特权信息：真实值，无相对偏移）
        joint_vel = ObsTerm(func=mdp.joint_vel)                                                             # 绝对关节速度（特权信息：真实值）
        actions = ObsTerm(func=mdp.last_action)                                                             # 上一时刻的动作
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})                                   # 步态相位：周期设为 0.8 秒
        #gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.5}) 

        def __post_init__(self):                                                # 初始化后处理函数
            self.history_length = 3                                             # 历史观测长度设为 3（使用过去 3 帧的历史信息）
            self.enable_corruption = False                                      # 禁用观测数据损坏（特权观测通常假设是完美的）
            self.concatenate_terms = True                                       # 将所有观测项拼接成一个向量
    
    critic: CriticCfg = CriticCfg()                                             # 实例化评论家观测组


    @configclass
    class DiscriminatorCfg(ObsGroup):                                           # 判别器（Discriminator）网络的观测配置
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)                           # 基座线速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)                           # 基座角速度
        joint_pos = ObsTerm(func=mdp.joint_pos)                                 # 绝对关节位置
        joint_vel = ObsTerm(func=mdp.joint_vel)                                 # 绝对关节速度

        def __post_init__(self):                                                # 初始化后处理函数
            self.enable_corruption = False                                      # 禁用观测数据损坏
            self.concatenate_terms = True                                       # 将观测项拼接
            self.concatenate_dim = -1                                           # 拼接维度设为最后一维
            self.history_length = AMP_INPUT_STEPS                               # 历史长度设为 AMP 输入步数（由常量定义）
            self.flatten_history_dim = False                                    # 不展平历史维度（保持时间序列结构）
            
    disc: DiscriminatorCfg = DiscriminatorCfg()                                 # 实例化判别器观测组
            
    @configclass                     
    class DiscriminatorDemoCfg(ObsGroup):                                       # 判别器演示数据（Demo）的观测配置
        # 此项应与上述判别器观测保持一致
        amp_data_disc_obs = ObsTerm(func=mdp.amp_data_discriminator_obs, params={'n_steps': AMP_INPUT_STEPS})  # 从 AMP 数据集中提取判别器观测，步数为 AMP_INPUT_STEPS
    
    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()  # 实例化判别器演示观测组


 


# 随机事件配置类
@configclass
class EventCfg:
    """事件配置。"""

    # 启动阶段随机化事件（仅在环境初始化时执行一次）
    physics_material = EventTerm(                                           # 物理材质随机化
        func=mdp.randomize_rigid_body_material,                             # 随机化刚体物理材质
        mode="startup",                                                     # 执行模式：启动时
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),          # 作用于机器人所有连杆
            "static_friction_range": (0.3, 1.6),                            # 静摩擦系数范围
            "dynamic_friction_range": (0.3, 1.2),                           # 动摩擦系数范围
            "restitution_range": (0.0, 0.5),                                # 恢复系数范围（弹性）
            "num_buckets": 64,                                              # 摩擦桶数量（用于一致性分组）
            "make_consistent": True,                                        # 使同组连杆材质一致
        },
    )



    # 基座质量扰动
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 2.0),                       # 这个范围 负数要小于基座质量，避免 -3+2.7<0
            "operation": "add",
        },
    )

    # 刚体质心随机化
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "base_link"]),
            "com_range": {"x": (-0.06, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},  
        },
    )
    

    # 连杆质量缩放
    scale_link_mass = EventTerm(                                        
        func=mdp.randomize_rigid_body_mass,                                 # 随机化刚体质量
        mode="startup",                                                     # 执行模式：启动时
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),  # 作用于左右肢体连杆
            "mass_distribution_params": (0.8, 1.2),                         # 质量缩放因子范围（80%~120%）
            "operation": "scale",                                           # 操作类型：缩放质量
        },
    )


    # 执行器增益缩放
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),  # 作用于所有关节
            "stiffness_distribution_params": (0.8, 1.2),                     # 刚度缩放范围（80%~120%）
            "damping_distribution_params": (0.8, 1.2),                       # 阻尼缩放范围（80%~120%）
            "operation": "scale",                                            # 操作类型：缩放增益
        },
    )

    # 关节参数缩放
    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),  # 作用于所有关节
            "friction_distribution_params": (1.0, 1.0),                      # 摩擦缩放范围（固定为1.0，即不缩放）
            "armature_distribution_params": (0.8, 1.2),                      # 臂惯性缩放范围（80%~120%）
            "operation": "scale",                                            # 操作类型：缩放参数
        },
    )

    # 重置阶段事件（每次环境重置时触发）
    base_external_force_torque = EventTerm(                                # 基座外力/力矩施加
        func=mdp.apply_external_force_torque,                              # 施加外力和力矩
        mode="reset",                                                      # 执行模式：重置时
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), # 作用于躯干连杆
            "force_range": (0.0, 0.0),                                     # 力范围（当前禁用）
            "torque_range": (-0.0, 0.0),                                   # 力矩范围（当前禁用）
        },
    )

    # 从参考轨迹重置（已禁用）
    # reset_from_ref = EventTerm(
    #     func=mdp.reset_from_ref, 
    #     mode="reset",
    #     params=MISSING
    # )

    # # interval

    # 基座状态重置
    reset_base = EventTerm(                                             
        func=mdp.reset_root_state_uniform,                              # 均匀随机重置根关节状态
        mode="reset",                                                   # 执行模式：重置时
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},  # 位姿重置范围（xy位置±0.5m，偏航角±π）
            "velocity_range": {                                         # 速度重置范围
                "x": (-0.2, 0.2),                                       # X线速度 ±0.2 m/s
                "y": (-0.2, 0.2),                                       # Y线速度 ±0.2 m/s
                "z": (-0.2, 0.2),                                       # Z线速度 ±0.2 m/s
                "roll": (-0.2, 0.2),                                    # 横滚角速度 ±0.2 rad/s
                "pitch": (-0.2, 0.2),                                   # 俯仰角速度 ±0.2 rad/s
                "yaw": (-0.2, 0.2),                                     # 偏航角速度 ±0.2 rad/s
            },
        },
    )


    # 关节状态重置
    reset_robot_joints=EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),                               # 关节位置缩放范围（80%~120%）
            "velocity_range": (0.0, 0.0),                               # 关节速度重置为0
        },
    )
    
    # 随机推搡机器人
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s= (8.0, 12.0),           # 训练动态行走 与 跑步
        #interval_range_s= (2.0, 5.0),          # 训练静态站立
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-1.0, 1.0)}},
    )

# MDP 奖励项配置
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

   
    # -- 主任务，线速度奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,weight=2.,params={"command_name": "base_velocity", "std": 0.5},)                               # 运动 weight=2.  "std": 0.5     静态站立  1.   "std": 0.7
    # -- 主人任务，角速度跟踪奖励权重（z轴指数形式）
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=2., params={"command_name": "base_velocity", "std": 0.5})     # 运动 weight=2.   "std": 0.5    静态站立  1.   "std": 0.7
    
    # 定义角速度跟踪奖励 (L2/线性衰减版)
    track_ang_vel_z_l2 = RewTerm(func=mdp.track_ang_vel_z_l2,weight=1.,params={"command_name": "base_velocity"})


    # -- 存活奖励
    alive = RewTerm(func=mdp.is_alive, weight=0.15)                                                                                  # 运动 weight=0.15 静态站立  weight=0.5
    
    # -- 基座相关惩罚
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)                                # 基座Z 轴 上下线速度
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)                              # 基座XY轴运动惩罚 -0.1
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.)                  # 基座平面姿态惩罚

    # -- 关节类的惩罚
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight= -2e-4)                              # 关节速度惩罚
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight= -2.5e-7)                            # 关节加速度惩罚                        # 运动  weight= -2.5e-7   静态站立  -5.e-7
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight= -0.01)                          # 动作变化率惩罚                        # 运动  weight= -0.01     静态站立  -0.02
    smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0)                                   # # 惩罚动作的变化量   
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight= -1.)                        # 关节位置限位惩罚  
    joint_energy = RewTerm(func=mdp.joint_energy, weight= -1e-4)                              # 关节能耗惩罚 
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight= -1e-3 )               # 关节正则化惩罚
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2,weight=-1e-5 )                       # 关节扭矩惩罚
        

    # -- 惩罚机器人在低速移动时的身体摇摆/晃动
    low_speed_sway_penalty = RewTerm(
        func=mdp.low_speed_sway_penalty,
        weight=-1e-2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )
        
    # -- 脚部滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight= -0.1,
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
    # -- 脚腾空时间奖励
    feet_air_time_positive_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), 
            "threshold": 0.4},
    )
    # -- 落地冲击惩罚
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=-5e-5,
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
    
    
    '''
    # -- 跑步模式步态运动相关奖励项 
    gait_run = RewTerm(
        func=mdp.feet_gait,
        weight=1.,  
        params={
            "period": 0.5,                                                                
            "offset": [0.0, 0.5],                                                         
            "threshold": 0.35,                                                            
            "command_name": "base_velocity",                                             
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),  
        },
    )  
    '''



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
    feet_y_distance = RewTerm(
        func=mdp.feet_y_distance,     #feet_too_near, 
        weight=-2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": (0.20, 0.30) },   # 设定期望的脚踝间距范围 
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




    #  若要训练静态站立则需要将  
    #       1：CommandsCfg.ranges 设置为静态站立范围， 
    #       2：EventCfg.push_robot.interval_range_s 设置为静态站立范围
    #       3: PolicyCfg.gait_phase.params["period"] = 0.8
    #       4：CriticCfg.gait_phase.params["period"] = 0.8
    #       5: 步态奖励用 gait_walk

    #  若要训练动态行走则需要将 
    #       1：CommandsCfg.ranges 设置为动态行走范围， 
    #       2：EventCfg.push_robot.interval_range_s 设置为动态行走范围
    #       3: PolicyCfg.gait_phase.params["period"] = 0.8
    #       4：CriticCfg.gait_phase.params["period"] = 0.8
    #       5: 步态奖励用 gait_walk

    #  若要训练动态跑步则需要将 
    #       1：CommandsCfg.ranges 设置为动态跑步范围， 
    #       2：EventCfg.push_robot.interval_range_s 设置为动态跑步范围
    #       3: CriticCfg.gait_phase.params["period"] = 0.5
    #       4：PolicyCfg.gait_phase.params["period"] = 0.5
    #       5: 步态奖励用 gait_run
    




@configclass
class TerminationsCfg:
    """MDP 的终止条件项配置。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)                                                    # 超时终止：达到最大回合时长时结束当前回合 
    base_contact = DoneTerm(                                                                                 # 不希望的触终止 
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})               # 基座高度过低终止：根关节高度低于0.4米视为摔倒 
                                                                                                             # 身体倾斜角超过60度（约1.047弧度）即终止 
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={
            "limit_angle": math.radians(60.0),
        },
    )

@configclass
class CurriculumCfg:
    """MDP 的课程学习项配置。"""
    pass

@configclass
class MotionDataCfg:
    """Motion data terms for the AMP."""
    motion_data_dir='/home/robot/hyj/SimpleAMP/robot_assets/ths_23dof/motion_data'

    motion_data_weights=None            #  = None 表示全选
    '''
    motion_data_weights={               # 否则指定部分
        '127_03_stageii': 1.0,
        '127_04_stageii': 1.0,
        '127_06_stageii': 1.0,
    }
    '''

##
# 环境配置
##

@configclass
class SimpleampEnvCfg(ManagerBasedRLEnvCfg):
    # 场景设置
    scene: SimpleampSceneCfg = SimpleampSceneCfg(num_envs=4096, env_spacing=2.5)
    # 基本设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP设置
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
        self.decimation = 4                                # 降采样：每 4 个仿真步执行一次控制指令
        self.episode_length_s = 20                         # 回合时长：每个训练回合持续 20 秒
        # simulation settings
        self.sim.dt = 1 / 200                              # 仿真步长：设置物理仿真间隔为 1/200 = 0.005 秒 (即 200Hz 仿真频率)
        self.sim.render_interval = self.decimation

@configclass
class SimpleampEnvCfgPlay(SimpleampEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 为演示场景缩小规模
        self.scene.num_envs = 1                                                      # 环境数量设为1（单机器人演示）
        self.scene.env_spacing = 2.5                                                 # 环境间距设为2.5米（避免碰撞）
        self.episode_length_s = 40.0                                                 # 单回合时长设为40秒

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)                    # X轴线速度范围
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)                    # Y轴线速度范围
        self.commands.base_velocity.ranges.ang_vel_z = (0., 0.)                    # Z轴角速度范围

        self.observations.policy.enable_corruption = False                           # 关闭观测噪声
        # self.events.push_robot = None                                                # 禁用机器人推搡干扰


