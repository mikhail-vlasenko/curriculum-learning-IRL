from typing import List
from dataclasses import dataclass, field, asdict

import torch


@dataclass
class EnvConfig:
    id: str = 'gym_examples/GridWorld-v0'
    grid_size: int = 5
    wrappers: List[str] = field(default_factory=lambda: ['RelativePosition'])
    vectorized: bool = True
    tensor_state: bool = False


@dataclass
class PPOTrainConfig:
    """
    Config for training the expert PPO
    """
    do_train: bool = True
    env_steps: int = 30000
    load_from: str = 'saved_models/size5_ppo_expert.pt'
    ppo_save_path: str = 'saved_models/ppo_expert.pt'


@dataclass
class PPOConfig:
    """
    Global PPO config
    """
    batch_size: int = 256
    n_workers: int = 64
    lr: float = 1e-3
    entropy_reg: float = 0.05
    gamma: float = 0.8
    epsilon: float = 0.2
    update_epochs: int = 5


@dataclass
class DemosConfig:
    n_steps: int = 10000


@dataclass
class AIRLConfig:
    """
    Config for training with AIRL
    """
    expert_data_path: str = 'demonstrations/ppo_demos_size5.pk'
    env_steps: int = 600000

    disc_load_from: str = None
    ppo_load_from: str = None
    disc_save_path: str = 'saved_models/discriminator.pt'
    ppo_save_path: str = 'saved_models/airl_ppo.pt'


@dataclass
class DiscriminatorConfig:
    batch_size: int = 1024
    lr: float = 5e-4


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo_train: PPOTrainConfig = field(default_factory=PPOTrainConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    demos: DemosConfig = field(default_factory=DemosConfig)
    airl: AIRLConfig = field(default_factory=AIRLConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    device: str = 'cuda:0'

    def as_dict(self):
        return asdict(self)


CONFIG = Config()

CONFIG.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if CONFIG.device == 'cpu':
    print('WARNING: CUDA not available. Using CPU.')

EXPERT_DATA_PREFIX = 'demonstrations/ppo_demos_size'
EXPERT_DATA_SUFFIX = '.pk'


def set_experiment_config(grid_size: int = None, wrappers: List[str] = None):
    if grid_size is not None:
        CONFIG.env.grid_size = grid_size
        CONFIG.airl.expert_data_path = EXPERT_DATA_PREFIX + str(grid_size) + EXPERT_DATA_SUFFIX
    if wrappers is not None:
        CONFIG.env.wrappers = wrappers


if __name__ == '__main__':
    print(CONFIG.as_dict())
    print(CONFIG.ppo.lr)
    CONFIG.ppo.lr = 1e-4
    print(CONFIG.ppo.lr)
