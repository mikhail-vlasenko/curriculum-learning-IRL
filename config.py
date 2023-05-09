from typing import List
from dataclasses import dataclass, field, asdict


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
    expert_data_path: str = 'demonstrations/ppo_demos_size5.pk'
    env_steps: int = 1000000


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

if __name__ == '__main__':
    print(CONFIG.as_dict())
    print(CONFIG.ppo.lr)
    CONFIG.ppo.lr = 1e-4
    print(CONFIG.ppo.lr)
