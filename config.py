from typing import List
from dataclasses import dataclass, field, asdict

import torch


@dataclass
class EnvConfig:
    id: str = 'gym_examples/GridWorld-v0'
    grid_size: int = 5
    max_steps: int = 15
    obs_dist: int = 2
    wrappers: List[str] = field(default_factory=lambda: ['FlattenObs'])
    vectorized: bool = True
    tensor_state: bool = False
    render: bool = False
    reward_configuration: str = 'positive_stripe'  # default, checkers, positive_stripe


@dataclass
class PPOTrainConfig:
    """
    Config for training the expert PPO
    """
    do_train: bool = True
    env_steps: int = 1000000
    load_from: str = 'saved_models/ppo_expert.pt'
    # load_from: str = None
    save_to: str = 'saved_models/ppo_expert.pt'


@dataclass
class PPOConfig:
    """
    Global PPO config
    """
    batch_size: int = 1024
    n_workers: int = 64
    lr: float = 1e-3
    entropy_reg: float = 0.05
    gamma: float = 0.9
    epsilon: float = 0.2
    update_epochs: int = 5
    nonlinear: str = 'relu'  # tanh, relu
    dimensions: List[int] = field(default_factory=lambda: [256, 256])
    simple_architecture: bool = True


@dataclass
class DemosConfig:
    n_steps: int = 10000


@dataclass
class AIRLConfig:
    """
    Config for training with AIRL
    """
    expert_data_path: str = 'demonstrations/ppo_demos_size5.pk'
    env_steps: int = 500000

    # disc_load_from: str = None
    # ppo_load_from: str = None
    disc_load_from: str = 'saved_models/discriminator.pt'
    ppo_load_from: str = 'saved_models/airl_ppo.pt'
    load_from_checkpoint: bool = False  # if True, overwrites disc_load_from and ppo_load_from

    disc_save_to: str = 'saved_models/discriminator.pt'
    ppo_save_to: str = 'saved_models/airl_ppo.pt'


@dataclass
class DiscriminatorConfig:
    batch_size: int = 1024
    lr: float = 5e-4
    simple_architecture: bool = True
    dimensions: List[int] = field(default_factory=lambda: [32, 32])


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo_train: PPOTrainConfig = field(default_factory=PPOTrainConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    demos: DemosConfig = field(default_factory=DemosConfig)
    airl: AIRLConfig = field(default_factory=AIRLConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    device: str = 'cuda:0'
    continued_ppo_training: bool = False
    continued_airl_training: bool = False

    def as_dict(self):
        return asdict(self)


CONFIG: Config = Config()

PPO_CHECKPOINT = 'saved_models/checkpoints/ppo_latest.pt'
DISC_CHECKPOINT = 'saved_models/checkpoints/discriminator_latest.pt'
EXPERT_DATA_PREFIX = 'demonstrations/ppo_demos_size'
EXPERT_DATA_SUFFIX = '.pk'

if CONFIG.ppo_train.load_from is not None:
    CONFIG.continued_ppo_training = True

if CONFIG.airl.ppo_load_from is not None:
    CONFIG.continued_airl_training = True

if CONFIG.airl.load_from_checkpoint:
    CONFIG.airl.disc_load_from = DISC_CHECKPOINT
    CONFIG.airl.ppo_load_from = PPO_CHECKPOINT

CONFIG.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if CONFIG.device == 'cpu':
    print('WARNING: CUDA not available. Using CPU.')


def set_experiment_config(
        grid_size: int = None,
        wrappers: List[str] = None,
        max_steps: int = None,
        reward_configuration: str = None
) -> None:
    print('Setting experiment config')
    if grid_size is not None:
        CONFIG.env.grid_size = grid_size
        CONFIG.airl.expert_data_path = EXPERT_DATA_PREFIX + str(grid_size) + EXPERT_DATA_SUFFIX
    if wrappers is not None:
        CONFIG.env.wrappers = wrappers
    if max_steps is not None:
        CONFIG.env.max_steps = max_steps
    if reward_configuration is not None:
        CONFIG.env.reward_configuration = reward_configuration


def check_config(config: Config):
    if config.airl.ppo_load_from is not None and config.airl.disc_load_from is None or \
            config.airl.ppo_load_from is None and config.airl.disc_load_from is not None:
        raise ValueError('Must specify both or neither of ppo_load_from and disc_load_from')


check_config(CONFIG)


if __name__ == '__main__':
    print(CONFIG.as_dict())
    print(CONFIG.ppo.lr)
    CONFIG.ppo.lr = 1e-4
    print(CONFIG.ppo.lr)
