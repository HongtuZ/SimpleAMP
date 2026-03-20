from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    



def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用指数核函数奖励线性速度命令(xy 轴）的跟踪效果。"""
    # 提取使用的量（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算误差                                                                            
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    #return torch.exp(-lin_vel_error / std**2)                                               # 仅基于误差的指数奖励
    reward = torch.exp(-lin_vel_error / std**2)                                              # 计算基础速度跟踪奖励（误差越小奖励越接近 1）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 乘以姿态系数：若机器人倾斜严重（重力 Z 投影异常），则降低奖励，直立时为 1.0
    return reward                                                                            # 返回经过姿态修正后的最终奖励值



def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用指数核函数奖励角速度命令（偏航轴 yaw）的跟踪效果。"""
    # 提取使用的量（以启用类型提示）                                                              # 从场景中获取机器人刚体对象
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算误差                                                                                # 计算命令角速度与本体实际角速度在 Z 轴（偏航）上的平方误差
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    #return torch.exp(-ang_vel_error / std**2)                                               # 仅基于误差的指数奖励
    reward = torch.exp(-ang_vel_error / std**2)                                              # 计算基础角速度跟踪奖励（误差越小奖励越接近 1）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7  # 乘以姿态系数：若机器人倾斜严重（重力 Z 投影异常），则降低奖励，直立时为 1.0
    return reward                                                                            # 返回经过姿态修正后的最终奖励值


def track_ang_vel_z_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ 使用线性（类L2）衰减对角速度指令（偏航）进行奖励跟踪。.""" 
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.abs(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    ang_vel_error = torch.clamp(ang_vel_error, min=0.0, max=1.0)           # 将误差控制在0-1
    reward = 1.0 - ang_vel_error

    return reward



def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """存活奖赏."""
    return (~env.termination_manager.terminated).float()


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用L2平方核函数对z轴基线速度进行惩罚."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 使用L2平方核函数对xy轴基角速度进行惩罚 """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 使用L2平方核函数对非平面基方向进行惩罚
        这是通过惩罚投影重力向量的xy分量来计算的.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:  
    """ 使用 L2 平方核惩罚非水平的基座朝向
        计算方式：通过惩罚投影重力向量的 X 和 Y 分量来实现
    """
    # 提取使用的变量（以启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]                                # 获取场景中的机器人资产对象（刚性物体）
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)  # 返回投影重力向量前两个分量（X, Y）的平方和（即 L2 范数的平方）



def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 使用L2平方核惩罚关节处的关节速度.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
          只有配置在:attr:`asset_cfg.joint_ids`中的关节，其关节速度才会对该项做出贡献
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 使用L2平方核惩罚关节处的联合加速度.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
          只有配置在:attr:`asset_cfg.joint_ids`中的关节，其关节加速度才会对该项做出贡献
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚偏离默认位置的关节角度。"""                                       
    asset: Articulation = env.scene[asset_cfg.name]                       
    # 计算当前关节角度与默认角度的差值 (偏差)
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)                             # 返回所有指定关节偏差绝对值的总和 (L1 范数)




def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚关节位置如果它们超出软限制。

    计算方法为：关节位置与软限制之间差值的绝对值之和。
    """
    # 提取使用的量（以启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算超出限制约束
    out_of_limits = -(                                                                                      # 计算低于下限的违规量
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)                                                                                         # 仅保留负值部分（即低于下限的部分）并取反为正
    out_of_limits += (                                                                                      # 加上高于上限的违规量
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)                                                                                         # 仅保留正值部分（即高于上限的部分）
    return torch.sum(out_of_limits, dim=1)                                                                  # 沿关节维度求和，返回每个环境的总违规值




def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ 使用L2平方核惩罚动作的变化率."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 使用L2平方核函数对施加在关节上的联合扭矩进行惩罚.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
          只有配置在:attr:`asset_cfg.joint_ids`中的关节，其关节扭矩才会对该项做出贡献
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

# 计算机器人双脚在身体坐标系下的横向（Y轴）距离，并奖励该距离保持在指定范围 [min, max] 内的情况
# 当双脚间距在理想范围内时返回接近 1.0 的高分；当间距过小或过大时，分数会呈指数级迅速衰减趋近于 0
def feet_distance_y(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    min: float = 0.2, 
    max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_air_time(                                                               
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float  
) -> torch.Tensor:                                                                
    """
        计算脚部腾空时间奖励
        使用 L2 核函数奖励脚部长步幅行走
        说明：该函数奖励智能体迈出超过阈值的步长，确保机器人抬脚迈步
        计算方式：累加脚部处于腾空状态的时间
        特殊情况：若命令速度很小（无需迈步），则奖励为零
    """
    # extract the used quantities (to enable type-hinting)                       # 提取使用的变量（以启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]           # 获取场景中的接触传感器对象
    # compute the reward                                                         # 计算奖励值
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # 计算当前步长内是否首次接触地面（布尔值）
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]    # 获取上次腾空的持续时间
    positive_air = torch.clamp(last_air_time - threshold, min=0.0)               # 计算超出阈值的腾空时间（小于0则截断为0）
    reward = torch.sum(positive_air * first_contact.float(), dim=1)              # 求和：仅当脚刚落地时，累加其之前的超额腾空时间
    # no reward for zero command                                                 # 零命令时无奖励
    reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()  # 掩码：仅当水平速度命令大于 0.1 时才保留奖励
    return reward                                                                 # 返回最终奖励值



def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励双足机器人迈步时脚部在空中的时间。
       该函数鼓励智能体单次迈步时间不超过指定阈值，并始终保持**仅一只脚离地**（即交替迈步）。
    
       若运动指令很小（即不应迈步），则不给予奖励。
    """
    asset: Articulation = env.scene["robot"]                                                # 获取机器人实体（此处未实际使用）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]                      # 获取接触传感器数据

    # 计算奖励                                                                               # 获取每只脚当前的离地时间（air time）
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]        # 获取每只脚当前的触地时间
    in_contact = contact_time > 0.0                                                        # 判断每只脚是否处于触地状态（布尔张量）
    in_mode_time = torch.where(in_contact, contact_time, air_time)                         # 触地时用触地时间，离地时用离地时间（实际仅离地时间用于奖励）
    single_stance = torch.sum(in_contact.int(), dim=1) == 1                                # 判断是否恰好有一只脚触地（即另一只脚在空中）
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]  # 仅当满足单支撑相时，取两只脚中有效的“模式时间”（实为离地时间）的最小值 → 即离地那只脚的时间
    reward = torch.clamp(reward, max=threshold)                                            # 将奖励限制在 [0, threshold] 范围内

    # 指令为零时不给奖励                                                                      # 若 xy 平面指令速度小于 0.1，则奖励置零（避免静止时误奖）
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


# 惩罚动作的变化量  
def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
                                                                                              # 计算当前动作与上一帧动作差值的平方
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    diff = diff * (env.action_manager.prev_action[:, :] != 0)                                 # 忽略第一步（prev_action为0时），防止初始随机噪声产生巨大惩罚
    return torch.sum(diff, dim=1)                                                             # 沿动作维度求和，返回每个环境的总平滑度惩罚值


def feet_orientation_l2(env: ManagerBasedRLEnv, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 当脚部与地面接触时，若其方向与地面不平行，则应受到处罚
        这是通过惩罚投影重力向量的xy分量来计算的
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, 
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )
    
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ 当指令非常小时，对偏离默认关节位置的偏移进行惩罚."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def joint_energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 对机器人关节所消耗的能量进行惩罚"""
    asset = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ 对滑步进行处罚.
        此函数会惩罚智能体在地面上滑动其脚部。奖励的计算方法是脚部线速度的范数乘以一个二进制接触传感器。这确保了只有当脚部与地面接触时，智能体才会受到惩罚。
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward

def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 使用L2平方核函数对z轴基线速度进行惩罚."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    每只脚单独计算：
    脚接触地面时，z 方向加速度大 → 惩罚
    """

    asset = env.scene["robot"]

    # 1️⃣ 取所有 body 的线加速度 (world)
    # shape: (Nenv, Nbody, 6)
    body_acc = asset.data.body_acc_w

    # 2️⃣ 取“脚”的 z 方向线加速度
    # shape: (Nenv, Nfeet)
    foot_acc_z = body_acc[:, sensor_cfg.body_ids, 2]

    # 3️⃣ 取脚的接触状态
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    in_contact = torch.abs(contact_force_z) > 1.0  # (Nenv, Nfeet)

    # 4️⃣ 每只脚：加速度平方 × 接触状态
    acc_penalty = (foot_acc_z ** 2) * in_contact.float()

    # 防止数值爆炸（非常重要）
    acc_penalty = torch.clamp(acc_penalty, max=50.0)

    # 5️⃣ 所有脚加起来
    penalty = acc_penalty.sum(dim=1)
    reward = penalty

    # 仅当速度命令较小（小于 1.5）时才启用该奖励
    cmd = env.command_manager.get_command(command_name)
    
    # 使用 xy 分量的速度范数作为速度大小判断
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    reward = reward * (cmd_speed < 1.5).float()

    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """ 当违规次数超过阈值时，对非预期接触进行处罚."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)



def low_speed_sway_penalty(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当命令速度低于阈值时，惩罚线速度和角速度。

    此函数在命令速度非常小时惩罚机器人的运动（包括线速度和角速度），
    鼓励机器人在低速命令期间保持静止。
    """
    # 提取使用的量（以启用类型提示）                                                              # 获取场景中的刚体资产对象
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取命令速度                                                                                   # 从命令管理器获取指定名称的命令张量
    command = env.command_manager.get_command(command_name)
    command_speed = torch.norm(command[:, :2], dim=1)                                               # 计算命令在水平面 (x, y) 上的速度范数
    # 惩罚 xy 平面内的线速度                                                                         # 计算机器人本体坐标系下水平线速度的平方和
    lin_vel_penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    # 惩罚角速度                                                                                     # 计算机器人本体坐标系下角速度的平方和
    ang_vel_penalty = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    # 总速度惩罚                                                                                     # 合并线速度和角速度的惩罚项
    vel_penalty = lin_vel_penalty + ang_vel_penalty
    # 仅当命令速度低于阈值时应用惩罚                                                                 # 生成掩码并转换为浮点数，低速时保留惩罚值，高速时归零
    return vel_penalty * (command_speed < command_threshold).float()





def staged_navigation_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    heading_threshold: float = 0.78,  # 45°，朝向误差阈值
    distance_threshold: float = 0.5,   #  距离目标的阈值
    near_goal_threshold: float = 2.0,  # 接近目标的距离阈值
    obstacle_threshold: float = 0.8,  # 前方障碍物的距离阈值
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ray_caster: RayCaster = env.scene[sensor_cfg.name]
    
    command = env.command_manager.get_command(command_name)
    des_pos = command[:, :2] # 目标位置
    des_heading = command[:, 2] # 目标朝向
    distance = torch.norm(des_pos, dim=1) # 机器人到目标位置的距离
    
    vx = asset.data.root_lin_vel_b[:, 0] # 机器人在base frame下的前向速度
    vy = asset.data.root_lin_vel_b[:, 1] # 机器人在base frame下的侧向速度
    speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1) # 机器人在水平面的速度大小 
    ang_speed = torch.abs(asset.data.root_ang_vel_b[:, 2]) # 机器人绕垂直轴的角速度

    # 当前移动方向与期望朝向误差（规范化到 [-pi, pi]）
    move_dir_angle = torch.atan2(vy, vx)
    raw_diff = move_dir_angle - des_heading
    diff_wrapped = torch.remainder(raw_diff + torch.pi, 2 * torch.pi) - torch.pi
    move_heading_error = diff_wrapped.abs()
    
    # 雷达前方最近障碍物距离（已在外部被 clamp）
    origin = ray_caster.data.pos_w.unsqueeze(1)  # [num_envs, 1, 3]
    hits = ray_caster.data.ray_hits_w  # [num_envs, num_rays, 3]
    distances = torch.norm(hits - origin, dim=-1).clamp(min=0.2, max=5.0)  # [num_envs, num_rays]
    front_min_dist = torch.min(distances, dim=1).values  # [num_envs]
    
    # 1) 朝向匹配奖励：误差越小奖励越高
    heading_reward = 1.0 / (1.0 + (move_heading_error / (heading_threshold + 1e-6))**2)

    # 2) 沿期望朝向的速度（越朝向目标前进越好），只奖励正向分量
    proj_vel = vx * torch.cos(des_heading) + vy * torch.sin(des_heading)
    progress_reward = torch.tanh(2.0 * proj_vel.clamp(min=0.0, max=1.0))  # 正向速度越大奖励越高，最大值接近1.0

    # 3) 障碍物清除奖励：鼓励与障碍物保持距离
    # 当 front_min_dist < obstacle_threshold 时，距离越大奖励越高（在 safe_min..obstacle_threshold 区间归一化到 0..1）
    safe_min = 0.5
    denom = max(obstacle_threshold - safe_min, 1e-6)
    obs_clearance = torch.clamp(front_min_dist - safe_min, min=0.0, max=obstacle_threshold - safe_min) / denom  # 0..1 when front_min_dist within [safe_min, obstacle_threshold]
    # 保持变量名兼容下游使用
    obs_approach_raw = obs_clearance
    
    # 对于 front_min_dist < safe_min 给出负向惩罚（碰撞/过近）
    collision_penalty = torch.clamp(safe_min - front_min_dist, min=0.0) / safe_min  # 0..1

    # 距离目标的奖励：距离越小越好，使用 near_goal_threshold 归一化尺度
    dist_reward = 1.0 / (1.0 + (distance / (near_goal_threshold + 1e-6))**2)

    # 分阶段加权：远离目标优先前进与保持清除，接近目标优先朝向精确并靠近目标
    is_far = distance > near_goal_threshold
    is_near = torch.logical_and(distance <= near_goal_threshold, distance > distance_threshold)
    is_at_goal = distance <= distance_threshold

    # 权重已调整并规范化（每阶段权重之和约为1）：
    # - far: 优先前进/清除障碍
    # - near: 优先朝向准确
    # - goal: 优先朝向与靠近目标（含站姿保持项）
    far_reward = 0.50 * progress_reward + 0.15 * heading_reward + 0.25 * obs_approach_raw + 0.10 * dist_reward
    near_reward = 0.20 * progress_reward + 0.45 * heading_reward + 0.15 * obs_approach_raw + 0.20 * dist_reward
    goal_reward = 0.05 * progress_reward + 0.60 * heading_reward + 0.15 * torch.exp(-torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)) + 0.20 * dist_reward

    reward = torch.zeros_like(distance)
    reward = torch.where(is_far, far_reward, reward)
    reward = torch.where(is_near, near_reward, reward)
    reward = torch.where(is_at_goal, goal_reward, reward)

    # 减去碰撞/过近惩罚及不良行为惩罚（横向速度、角速度）
    lateral_speed = torch.abs(vy)
    reward = reward - collision_penalty #- 0.08 * lateral_speed - 0.05 * ang_speed

    # 限幅防止数值爆炸
    reward = torch.clamp(reward, min=-1.0, max=2.0)

    return reward





    #----------------------------------自加---------------------------------------


# 基座高度惩罚
def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """使用 L2 平方核函数惩罚资产高度与其目标值的偏差。
    注意:
        对于平坦地形，目标高度基于世界坐标系。对于崎岖地形，
        传感器读数可调整目标高度以适配地形变化。
    """
    # 提取使用的量（以启用类型提示）                                                          # 从场景中获取机器人刚体对象
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:                                                            # 如果配置了传感器（用于崎岖地形适应）
        sensor = env.scene[sensor_cfg.name]                                               # 获取传感器对象
        # 使用传感器数据调整目标高度                                                       # 目标高度 + 射线击中点 Z 轴坐标的平均值（即地面高度偏移）
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:                                                                                 # 如果没有传感器（平坦地形模式）
        # 直接使用提供的目标高度                                                           # 目标高度保持不变，不随地形调整
        adjusted_target_height = target_height
    # 计算 L2 平方惩罚                                                                     # 计算机器人实际 Z 轴位置与调整后目标高度的差值平方，并通过指数函数转换为奖励（越接近目标奖励越高）
    return torch.exp(-torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)/(std**2))






# 自然行走步态相位奖励
def feet_gait(
    env: ManagerBasedRLEnv,                                                                 # 环境对象，包含机器人、传感器和命令管理器
    period: float,                                                                          # 步态周期（单位：秒），定义一个完整步态循环的时间长度
    offset: list[float],                                                                    # 各腿的相位偏移列表，如 [0.0, 0.5] 表示左右足反相
    sensor_cfg: SceneEntityCfg,                                                             # 接触传感器配置，指定要监测的身体部位（如脚踝）
    threshold: float = 0.5,                                                                 # 支撑相占比阈值（相位 < threshold 时应触地）
    command_name=None,                                                                      # 关联的运动指令名称，用于判断是否应激活奖励
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]                      # 获取指定名称的接触传感器对象
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0       # 判断各足当前是否接触地面（布尔张量）
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)  # 计算全局归一化相位（范围 [0, 1)）

    phases = []                                                              # 初始化各腿相位列表
    for offset_ in offset:                                                   # 遍历每条腿的相位偏移
        phase = (global_phase + offset_) % 1.0                               # 加上偏移并取模，确保相位在 [0, 1) 范围内
        phases.append(phase)                                                 # 将当前腿的相位加入列表
    leg_phase = torch.cat(phases, dim=-1)                                    # 拼接所有腿的相位，形成 [num_envs, num_legs] 张量

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)  # 初始化奖励张量（每个环境一个标量，初始为0）

    for i in range(len(sensor_cfg.body_ids)):                                # 遍历每条腿（或每个监测部位）
        is_stance = leg_phase[:, i] < threshold                              # 根据相位判断该腿当前是否应处于支撑相 相位比例
        reward += ~(is_stance ^ is_contact[:, i])                            # 若实际接触状态与期望一致，则异或为False，取反后加1分


    if command_name is not None:                                                     # 如果指定了运动指令名称
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)  # 计算速度指令的模长（线速度大小）
        reward *= cmd_norm > 0.1                                                     # 仅当指令非零（>0.1）时保留奖励，否则置0

    return reward                                                           

 
 
# 奖励摆动中的脚部在离地时达到指定高度
def foot_clearance_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_height: float, 
    std: float, 
    tanh_mult: float,
    command_name=None,                                                                      
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]                                                                        # 从环境中获取指定的机器人刚体对象
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)                   # 计算脚部当前高度与目标高度的平方误差（Z轴方向）
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))  # 计算脚部水平速度的双曲正切值，用于区分摆动相与支撑相
    reward = foot_z_target_error * foot_velocity_tanh                                                                     # 将高度误差与速度调制项相乘，仅在摆动时施加高度奖励
    reward = torch.exp(-torch.sum(reward, dim=1) / std)                                                                   # 对所有脚求和后通过指数衰减生成最终奖励（值越小误差越小，奖励越高）
    
    if command_name is not None:                                                     # 如果指定了运动指令名称
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)  # 计算速度指令的模长（线速度大小）
        reward *= cmd_norm > 0.1                                                     # 仅当指令非零（>0.1）时保留奖励，否则置0

    return reward


# 惩罚双脚横向Y距离 “太近 或 太远”
def feet_y_distance(
    env: ManagerBasedRLEnv, threshold = (0.2, 0.3), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]                                                    # 获取机器人实体
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]                                         # 获取指定足端的世界坐标位置 (N, num_feet, 3)
    distance = torch.abs(feet_pos[:, 0,1] - feet_pos[:, 1,1])                                          # 计算两只脚之间的距离 (N,)
    min_dist, max_dist = threshold                                                                     # 解包最小和最大允许距离

    # Compute how much the distance violates the bounds
    violation_low = torch.clamp(min_dist - distance, min=0.0)   # >0 if too close                      # 距离过小的违反量（脚太近）
    violation_high = torch.clamp(distance - max_dist, min=0.0)  # >0 if too far                        # 距离过大的违反量（脚太远）

    # Total violation magnitude
    total_violation = violation_low + violation_high  # >= 0                                           # 总违反量（越界程度之和）
    return total_violation                                                                             # 返回非负标量，作为惩罚值（通常在奖励中取负）





# 手臂跟踪腿部协调奖励项  --------------------
def shoulder_thigh_coordination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gain: float = 1.0,
    std: float = 0.1,
) -> torch.Tensor:

    # 获取机器人资产
    asset: Articulation = env.scene[asset_cfg.name]


    # 获取腿部俯仰关节索引
    left_hip_pitch_idx = asset.find_joints("left_hip_pitch_joint")[0][0]
    right_hip_pitch_idx = asset.find_joints("right_hip_pitch_joint")[0][0]
    # 获取手部俯仰关节索引
    left_shoulder_pitch_idx = asset.find_joints("left_shoulder_pitch_joint")[0][0]
    right_shoulder_pitch_idx = asset.find_joints("right_shoulder_pitch_joint")[0][0]


    # 获取当前关节角度 和 默认关节角度 [num_envs, num_joints]
    joint_pos = asset.data.joint_pos
    default_joint_pos = asset.data.default_joint_pos  # shape: [num_envs, num_joints]

    # 计算腿部相对于默认姿态的偏移量（这才是“摆动量”）
    left_hip_offset = joint_pos[:, left_hip_pitch_idx] - default_joint_pos[:, left_hip_pitch_idx]
    right_hip_offset = joint_pos[:, right_hip_pitch_idx] - default_joint_pos[:, right_hip_pitch_idx]
    # 计算手部相对于默认姿态的偏移量（这才是“摆动量”）
    left_shoulder_offset = joint_pos[:, left_shoulder_pitch_idx] - default_joint_pos[:, left_shoulder_pitch_idx]
    right_shoulder_offset = joint_pos[:, right_shoulder_pitch_idx] - default_joint_pos[:, right_shoulder_pitch_idx]
    # 计算手脚协同角度差值
    err_left = torch.abs(left_shoulder_offset + left_hip_offset)
    err_right = torch.abs(right_shoulder_offset + right_hip_offset)
    reward_s = (err_left + err_right)/2
    # 高斯奖励
    reward = torch.exp(-0.5 * (reward_s / std) ** 2)
    return reward




# 奖励：左右腿俯仰关节摆幅均衡（基于默认姿态的偏移）
def hip_pitch_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    # 获取机器人资产
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取左右腿pitch关节索引
    left_idx = asset.find_joints("left_hip_pitch_joint")[0][0]
    right_idx = asset.find_joints("right_hip_pitch_joint")[0][0]

    # 获取当前关节角度 和 默认关节角度 [num_envs, num_joints]
    joint_pos = asset.data.joint_pos
    default_joint_pos = asset.data.default_joint_pos  # shape: [num_envs, num_joints]

    # 计算相对于默认姿态的偏移量（这才是“摆动量”）
    left_hip_offset = joint_pos[:, left_idx] - default_joint_pos[:, left_idx]
    right_hip_offset = joint_pos[:, right_idx] - default_joint_pos[:, right_idx]
    # 取差的绝对值（衡量摆幅是否一致）
    diff = torch.abs(left_hip_offset - right_hip_offset)

    # 高斯奖励
    reward = torch.exp(-0.5 * (diff / std) ** 2)
    return reward




  
# 惩罚零指令下 双腿俯仰关节摆幅不均衡 
def zero_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 10,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    # 获取机器人资产
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取左右腿pitch关节索引
    left_hip_pitch_idx = asset.find_joints("left_hip_pitch_joint")[0][0]
    right_hip_pitch_idx = asset.find_joints("right_hip_pitch_joint")[0][0]
    # 获取左右腿roll关节索引
    left_hip_roll_idx = asset.find_joints("left_hip_roll_joint")[0][0]
    right_hip_roll_idx = asset.find_joints("right_hip_roll_joint")[0][0]
    # 获取左右腿knee关节索引
    left_knee_idx = asset.find_joints("left_knee_joint")[0][0]
    right_knee_idx = asset.find_joints("right_knee_joint")[0][0]


    # 获取当前关节角度
    joint_pos = asset.data.joint_pos
    left_hip_pitch_pos = joint_pos[:, left_hip_pitch_idx]
    right_hip_pitch_pos = joint_pos[:, right_hip_pitch_idx]
    left_hip_roll_pos = joint_pos[:, left_hip_roll_idx]
    right_hip_roll_pos = joint_pos[:, right_hip_roll_idx]
    left_knee_pos = joint_pos[:, left_knee_idx]
    right_knee_pos = joint_pos[:, right_knee_idx]

    # 取差的绝对值（衡量摆幅是否一致）
    diff_1 = torch.abs(torch.abs(left_hip_pitch_pos) - torch.abs(right_hip_pitch_pos))
    diff_2 = torch.abs(torch.abs(left_hip_roll_pos) - torch.abs(right_hip_roll_pos))
    diff_3 = torch.abs(torch.abs(left_knee_pos) - torch.abs(right_knee_pos))
    reward = (diff_1 + diff_2 + diff_3) / 3 * std

    if command_name is not None:                                                     # 如果指定了运动指令名称
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)  # 计算速度指令的模长（线速度大小）
        reward *= cmd_norm < 0.1                                                     # 仅当指令非零（>0.1）时保留奖励，否则置0

    return reward




 

# 惩罚双腿 roll 关节过度向一侧倾斜
def hip_roll_side(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]                # 获取机器人资产
    left_idx = asset.find_joints("left_hip_roll_joint")[0][0]      # 获取左右腿 roll 关节索引
    right_idx = asset.find_joints("right_hip_roll_joint")[0][0]
    joint_pos = asset.data.joint_pos                               # 获取当前关节角度 [num_envs, num_joints]
    left_roll = joint_pos[:, left_idx]                             # 提取左右腿角度 (形状: [num_envs])
    right_roll = joint_pos[:, right_idx]
    threshold = 0.08                                                # 定义阈值
    violation = (left_roll > threshold) | (right_roll < -threshold) # 创建布尔掩码：左腿 > 0.08 或 右腿 < -0.08 (假设右腿向右倒为负值)
    reward = torch.zeros_like(left_roll)                            # 初始化奖励为 0
    reward[violation] = 1.0                                        # 对违规的环境施加固定惩罚 
    return reward



# 惩罚双腿 脚踝横滚关节角度超过范围
def ankle_roll_boundary(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]                # 获取机器人资产
    left_idx = asset.find_joints("left_ankle_roll_joint")[0][0]      # 获取左右腿 roll 关节索引
    right_idx = asset.find_joints("right_ankle_roll_joint")[0][0]
    joint_pos = asset.data.joint_pos                               # 获取当前关节角度 [num_envs, num_joints]
    left_roll = joint_pos[:, left_idx]                             # 提取左右腿角度 (形状: [num_envs])
    right_roll = joint_pos[:, right_idx]
    threshold = 0.5                                                # 定义阈值
    violation = (torch.abs(left_roll) > threshold) | (torch.abs(right_roll) > threshold) # 创建布尔掩码：左右腿 roll 关节绝对值超过阈值
    reward = torch.zeros_like(left_roll)                            # 初始化奖励为 0
    reward[violation] = 1.0                                        # 对违规的环境施加固定惩罚 
    return reward 


# 惩罚双腿 脚踝横滚关节扭矩超过范围
def ankle_roll_torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,  # 默认阈值 1 Nm
) -> torch.Tensor:
    """惩罚脚踝横滚关节扭矩超过 1 Nm 的情况。"""
    asset: Articulation = env.scene[asset_cfg.name]                                                # 获取机器人资产对象       
    # 获取左右腿 roll 关节索引
    # find_joints 返回 List[List[int]]，取 [0][0] 获取标量索引
    left_idx = asset.find_joints("left_ankle_roll_joint")[0][0]      
    right_idx = asset.find_joints("right_ankle_roll_joint")[0][0]
    joint_torques = asset.data.applied_torque                                                      # 获取当前施加的扭矩 [num_envs, num_joints]   这里使用的是 applied_torque，即控制器输出的目标扭矩                      
    left_torque = joint_torques[:, left_idx]                                                       # 提取左右腿扭矩 (形状: [num_envs])                    
    right_torque = joint_torques[:, right_idx]
    # 创建布尔掩码：左右腿扭矩绝对值超过阈值 (|tau| > 1.0)
    violation = (torch.abs(left_torque) > threshold) | (torch.abs(right_torque) > threshold)
    penalty = torch.zeros_like(left_torque)                                                        # 初始化惩罚值为 0             
    penalty[violation] = 1.0                                                                       # 对违规的环境施加固定惩罚值 1.0
    return penalty



# 左右腿对称性奖励(适合行走和跑步等动态，不适合静态站立)
def leg_symmetry_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """
    奖励左右双腿所有对应关节的运动对称性。
    当左右腿做相反的对称运动时（如交替摆腿）获得更高的奖励。
    
    关键说明：
    - 轴向相反的关节对（如 hip_pitch）：使用 l_offset + r_offset 判断对称性
    - 轴向相同的关节对（如某些 roll/yaw）：使用 l_offset - r_offset 判断对称性
    - 通过 axis_signs 向量指定每对关节的轴向关系（+1 相反，-1 相同）
    
    优化说明：此函数针对大规模环境数量（4096+）进行了完全向量化优化。
    """
    # 1. 获取机器人资产          
    asset: Articulation = env.scene[asset_cfg.name]                                   
    # 2. 定义固定的左右关节名称对 (顺序必须一一对应)
    left_names = [                                                      
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"
    ]
    right_names = [
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
    ]
    # 3. 定义每对关节的轴向关系
    # +1.0: 轴向相反（如 hip_pitch，摆动时应相反符号）
    # -1.0: 轴向相同（如某些 roll/yaw，摆动时应相同符号）
    # NOTE: 请根据实际机器人结构校准这些值！
    axis_signs = torch.tensor(
        [1.0, -1.0, -1.0, 1.0, 1.0, -1.0],  # 示例值，需要根据实际机器人调整
        dtype=torch.float32,
        device=env.device
    )
    # 4. 一次性获取所有关节索引（高效：避免循环中重复查询）
    left_indices = [asset.find_joints(name)[0][0] for name in left_names]
    right_indices = [asset.find_joints(name)[0][0] for name in right_names]
    
    # 5. 获取当前姿态和默认姿态 [num_envs, num_joints]
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos
    
    # 6. 向量化计算所有对称关节的差值
    # 计算所有左腿偏移：[num_envs, num_pairs]
    left_offsets = joint_pos[:, left_indices] - default_pos[:, left_indices]
    # 计算所有右腿偏移：[num_envs, num_pairs]
    right_offsets = joint_pos[:, right_indices] - default_pos[:, right_indices]
    
    # 7. 根据轴向关系计算对称性差值
    # 对于每对关节，根据 axis_signs 应用正确的运算：
    # - axis_signs[i] = +1: 计算 l_offset + r_offset（轴向相反）
    # - axis_signs[i] = -1: 计算 l_offset - r_offset（轴向相同）
    # 公式：l_offset + r_offset * axis_signs
    symmetry_diff = torch.abs(left_offsets + right_offsets * axis_signs)  # [num_envs, num_pairs]
    avg_diff = torch.mean(symmetry_diff, dim=1)                             # [num_envs]
    
    # 8. 高斯奖励 (标准的高斯核函数: exp(-(x/sigma)^2 / 2))
    return torch.exp(-torch.square(avg_diff / std) / 2)