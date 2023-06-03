from typing import List
from dataclasses import dataclass, field, asdict

import torch


@dataclass
class EnvConfig:
    id: str = 'gym_examples/GridWorld-v0'
    grid_size: int = 10
    max_steps: int = 30
    obs_dist: int = 2
    # 'OnlyEndReward', 'RelativePosition', 'FlattenObs'
    wrappers: List[str] = field(default_factory=lambda: ['FlattenObs'])
    vectorized: bool = True
    tensor_state: bool = False
    render: bool = False
    reward_configuration: str = 'default'  # default, checkers, positive_stripe, walk_around
    spawn_distance: int = -1  # -1 means fully random spawns


@dataclass
class PPOTrainConfig:
    """
    Config for training the expert PPO
    """
    env_steps: int = 500000
    load_from: str = 'saved_models/rew_per_tile_ppo_expert10.pt'
    # load_from: str = None
    save_to: str = f'saved_models/rew_per_tile_ppo_expert{EnvConfig.grid_size}.pt'


@dataclass
class PPOConfig:
    """
    Global PPO config
    """
    batch_size: int = 1024
    n_workers: int = 64
    lr: float = 1e-3 / 2
    entropy_reg: float = 0.05
    gamma: float = 0.99
    epsilon: float = 0.1
    update_epochs: int = 5
    nonlinear: str = 'relu'  # tanh, relu
    dimensions: List[int] = field(default_factory=lambda: [128, 128])
    simple_architecture: bool = True
    test_episodes: int = 30


@dataclass
class DemosConfig:
    n_steps: int = 50000
    load_from: str = f'saved_models/rew_per_tile_ppo_expert{EnvConfig.grid_size}.pt'


@dataclass
class AIRLConfig:
    env_steps: int = 2000000  # total steps from training, even with curriculum
    expert_data_path: str = None
    optimizer_disc: str = 'adam'  # adam, sgd (with no momentum)
    freeze_ppo_weights: bool = False

    disc_load_from: str = None
    ppo_load_from: str = None
    # disc_load_from: str = 'saved_models/discriminator.pt'
    # ppo_load_from: str = 'saved_models/airl_ppo.pt'
    load_from_checkpoint: bool = False  # if True, overwrites disc_load_from and ppo_load_from

    disc_save_to: str = 'saved_models/discriminator.pt'
    ppo_save_to: str = 'saved_models/airl_ppo.pt'


@dataclass
class DiscriminatorConfig:
    batch_size: int = 1024
    lr: float = 5e-4 / 2
    simple_architecture: bool = True
    dimensions: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo_train: PPOTrainConfig = field(default_factory=PPOTrainConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    demos: DemosConfig = field(default_factory=DemosConfig)
    airl: AIRLConfig = field(default_factory=AIRLConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    continued_ppo_training: bool = False
    continued_airl_training: bool = False

    def as_dict(self):
        return asdict(self)


CONFIG: Config = Config()

PPO_CHECKPOINT = 'saved_models/checkpoints/ppo_latest.pt'
DISC_CHECKPOINT = 'saved_models/checkpoints/discriminator_latest.pt'
EXPERT_DATA_PREFIX = 'demonstrations/ppo_demos_size'
EXPERT_DATA_SUFFIX = '.pk'


def augment_config():
    if CONFIG.ppo_train.load_from is not None:
        CONFIG.continued_ppo_training = True
    
    if CONFIG.airl.ppo_load_from is not None:
        CONFIG.continued_airl_training = True
    
    if CONFIG.airl.load_from_checkpoint:
        CONFIG.airl.disc_load_from = DISC_CHECKPOINT
        CONFIG.airl.ppo_load_from = PPO_CHECKPOINT


def get_demo_name():
    name = f'demonstrations/ppo_demos_' + \
           f'size-{CONFIG.env.grid_size}_' + \
           (f'end-reward_' if "OnlyEndReward" in CONFIG.env.wrappers else 'tile-reward_') + \
           f'reward-conf-{CONFIG.env.reward_configuration}.pk'
    return name


def set_experiment_config(
        grid_size: int = None,
        wrappers: List[str] = None,
        max_steps: int = None,
        reward_configuration: str = None,
        spawn_distance: int = None,
        ppo_lr: float = None,
) -> None:
    print('Setting experiment config')
    if grid_size is not None:
        CONFIG.env.grid_size = grid_size
    if wrappers is not None:
        CONFIG.env.wrappers = wrappers
    if max_steps is not None:
        CONFIG.env.max_steps = max_steps
    if reward_configuration is not None:
        CONFIG.env.reward_configuration = reward_configuration
    if spawn_distance is not None:
        CONFIG.env.spawn_distance = spawn_distance
    if ppo_lr is not None:
        CONFIG.ppo.lr = ppo_lr
    CONFIG.airl.expert_data_path = get_demo_name()
    check_config()


def check_config():
    if CONFIG.airl.ppo_load_from is not None and CONFIG.airl.disc_load_from is None or \
            CONFIG.airl.ppo_load_from is None and CONFIG.airl.disc_load_from is not None:
        raise ValueError('Must specify both or neither of ppo_load_from and disc_load_from')
    if CONFIG.env.max_steps != 3 * CONFIG.env.grid_size and CONFIG.env.max_steps != -1:
        print('WARNING: max_steps is not 3 * grid_size. This may cause issues with the reward function or demos.')
    if 'FlattenObs' in CONFIG.env.wrappers and CONFIG.ppo.gamma < 0.9:
        print('WARNING: gamma is less than 0.9 for a complex environment.')


augment_config()
check_config()


if __name__ == '__main__':
    print(CONFIG.as_dict())
    print(CONFIG.ppo.lr)
    CONFIG.ppo.lr = 1e-4
    print(CONFIG.ppo.lr)
