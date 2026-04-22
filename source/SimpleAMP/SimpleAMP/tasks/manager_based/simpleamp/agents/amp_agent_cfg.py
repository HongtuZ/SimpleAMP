# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from . import symmetry


@configclass
class RslRlAmpCfg:
    """Configuration class for the AMP (Adversarial Motion Priors) in the training"""

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
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "simple_amp"
    wandb_project = "simple_amp"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
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
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",  # default is scalar
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation=True,
        #     use_mirror_loss=True,
        #     mirror_loss_coeff=0.1, # 0.1
        #     data_augmentation_func=symmetry.compute_symmetric_states
        # ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=2.0,
                task_style_lerp=0.3,
            ),
            loss_type="LSGAN",
        ),
    )
