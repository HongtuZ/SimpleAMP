# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from . import symmetry


@configclass
class RslRlAmpCfg:
    """Configuration class for the AMP (Adversarial Motion Priors) in the training
    """
    
    disc_obs_buffer_size: int = 1000
    """Size of the replay buffer for storing discriminator observations"""
    
    grad_penalty_scale: float = 10.0
    """Scale for the gradient penalty in AMP training"""
    
    disc_trunk_weight_decay: float = 1.0e-4
    """Weight decay for the discriminator trunk network"""
    
    disc_linear_weight_decay: float = 1.0e-2
    """Weight decay for the discriminator linear network"""
    
    disc_learning_rate: float = 1.0e-5
    """Learning rate for the discriminator networks"""
    
    disc_max_grad_norm: float = 1.0
    """Maximum gradient norm for the discriminator networks"""

    @configclass
    class AMPDiscriminatorCfg:
        """Configuration for the AMP discriminator network."""

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the AMP discriminator network."""

        activation: str = "elu"
        """The activation function for the AMP discriminator network."""

        style_reward_scale: float = 1.0
        """Scale for the style reward in the training"""
        
        task_style_lerp: float = 0.0
        """Linear interpolation factor for the task style reward in the AMP training."""

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    """Configuration for the AMP discriminator network."""
    
    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"
    """Type of loss function used for the AMP discriminator (e.g., 'GAN', 'LSGAN', 'WGAN')"""


@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with convolutional layers."""

    class_name: str = "ActorCriticConv2d"
    """The policy class name. Default is ActorCriticConv2d."""

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    """List of convolutional layer parameters for the convolutional network."""

    conv_linear_output_size: int = 16
    """Output size of the linear layer after the convolutional features are flattened."""


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm."""
    
    class_name: str = "PPOAmp"
    """The algorithm class name. Default is PPOAmp."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""



@configclass
class RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"                      # 运行器类名：指定使用的 AMP 训练运行器类
    num_steps_per_env = 24                        # 每个环境每轮步数：每次更新前在每个并行环境中收集的步数
    max_iterations = 5000                         # 最大迭代次数：训练循环的最大轮数（epochs）
    save_interval = 100                           # 保存间隔：每训练多少轮保存一次模型检查点
    experiment_name = "simple_amp"                # 实验名称：用于标识本次训练任务的唯一名称 
    wandb_project = "simple_amp"                  # WandB 项目名称：用于将训练日志上传到 Weights & Biases 的项目名
    obs_groups = {                                # 观测组配置：定义不同网络组件使用的观测数据键名
        "policy": ["policy"],                     # 策略网络观测组：使用标记为 "policy" 的观测数据
        "critic": ["critic"],                     # 评论家网络观测组：使用标记为 "critic" 的观测数据
        "discriminator": ["disc"],                    # 判别器观测组：使用标记为 "disc" 的观测数据（用于真实机器人数据）
        "discriminator_demonstration": ["disc_demo"]  # 判别器演示观测组：使用标记为 "disc_demo" 的观测数据（用于专家演示数据/参考动作）
    }



    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     actor_obs_normalization=False,
    #     critic_obs_normalization=False,
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=128,
    #     rnn_num_layers=1
    # ) 
  

 
    # 初始化 RSL RL PPO 算法的 Actor-Critic 网络配置
    policy = RslRlPpoActorCriticCfg(                 
        init_noise_std=1.0,                          # 初始动作探索噪声的标准差设为 1.0
        # noise_std_type='log', # default is scalar  # 噪声标准差类型（默认是标量，此处注释掉表示不使用对数形式）
        actor_hidden_dims=[512, 256, 128],           # Actor 网络的隐藏层维度结构：512 -> 256 -> 128
        critic_hidden_dims=[512, 256, 128],          # Critic 网络的隐藏层维度结构：512 -> 256 -> 128
        actor_obs_normalization=True,                # 启用 Actor 网络的观测值归一化 (启用这项，观测值不再需要缩放系数，模型内部会自动进行归一化处理)
        critic_obs_normalization=True,               # 启用 Critic 网络的观测值归一化
        activation="elu",                            # 激活函数使用 ELU (Exponential Linear Unit)
    )
 


        
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",                      # 算法类名：指定使用集成了 AMP 功能的 PPO 算法
        value_loss_coef=1.0,                      # 价值损失系数：Critic 网络（价值函数）损失在总损失中的权重
        use_clipped_value_loss=True,              # 使用裁剪价值损失：启用值函数的裁剪机制，防止价值估计更新过大导致训练不稳定
        clip_param=0.2,                           # 裁剪参数：PPO 策略更新和价值函数更新的裁剪范围 (epsilon)，限制新旧策略/价值的差异
        entropy_coef=0.01,                        # 熵系数：鼓励探索的权重，防止策略过早收敛到局部最优
        num_learning_epochs=5,                    # 每次更新的训练轮数：每收集一批数据后，重复训练网络的次数
        num_mini_batches=4,                       # 小批量数量：将每批数据划分为多少个小批次进行梯度下降
        learning_rate=1.0e-4,                     # 学习率：优化器更新参数的步长大小
        schedule="adaptive",                      # 学习率调度策略："adaptive" 表示根据 KL 散度自适应调整学习率
        gamma=0.99,                               # 折扣因子：计算回报时对未来奖励的折现率 (越接近 1 越看重长期奖励)
        lam=0.95,                                 # GAE .lambda 参数：广义优势估计中的衰减因子，用于平衡偏差和方差
        desired_kl=0.01,                          # 目标 KL 散度：自适应学习率试图维持的策略分布差异阈值
        max_grad_norm=1.0,                        # 最大梯度范数：梯度裁剪阈值，防止梯度爆炸
        # symmetry_cfg=RslRlSymmetryCfg(          # （已注释）对称性配置：用于利用左右对称性增强训练
        #     use_data_augmentation=True,         # （已注释）启用数据增强：通过对称变换生成额外训练数据
        #     use_mirror_loss=True,               # （已注释）启用镜像损失：强制策略在对称状态下输出对称动作
        #     mirror_loss_coeff=0.1,              # （已注释）镜像损失系数：镜像损失在总损失中的权重 (建议值 0.1)
        #     data_augmentation_func=symmetry.compute_symmetric_states # （已注释）数据增强函数：计算对称状态的具体方法
        # ),




        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,           # 判别器观测缓冲区大小：存储历史帧用于时序判别
            grad_penalty_scale=10.0,            # 梯度惩罚系数：用于 WGAN-GP 等损失，稳定训练防止模式坍塌
            disc_trunk_weight_decay=1.0e-4,     # 判别器主干网络权重衰减：防止过拟合
            disc_linear_weight_decay=1.0e-2,    # 判别器线性层权重衰减：通常比主干层更大，限制输出层复杂度
            disc_learning_rate=1.0e-4,          # 判别器学习率：控制判别器参数更新速度
            disc_max_grad_norm=1.0,             # 判别器最大梯度范数：梯度裁剪阈值，防止梯度爆炸
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],        # 隐藏层维度：判别器神经网络各层的神经元数量
                activation="elu",               # 激活函数：使用 ELU (指数线性单元) 增加非线性
                style_reward_scale= 0., #2.0,         # 风格奖励缩放系数：放大 AMP 风格匹配奖励的权重
                task_style_lerp= 1.,  # 0.3           # 任务与风格插值系数：1.0 100#强化学习奖励  = 0.3 
            ),
            loss_type="LSGAN"                   # 损失函数类型：使用最小二乘 GAN (Least Squares GAN)，相比标准 GAN 训练更稳定
 

        ),
    ) 