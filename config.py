from typing import Type


class EnvConfig:
    id = 'gym_examples/GridWorld-v0'
    grid_size = 15
    wrappers = ['RelativePosition']
    vectorized = True


class PPOTrainConfig:
    do_train = True
    env_steps = 50000
    load_from = 'saved_models/size5_ppo_expert.pt'


class PPOConfig:
    batch_size = 32
    n_workers = 64
    lr = 1e-3
    entropy_reg = 0.05
    gamma = 0.8
    epsilon = 0.2
    update_epochs = 5


class DemosConfig:
    n_steps = 10000


class Config:
    env: Type[EnvConfig] = EnvConfig
    ppo_train: Type[PPOTrainConfig] = PPOTrainConfig
    ppo: Type[PPOConfig] = PPOConfig
    demos: Type[DemosConfig] = DemosConfig

    def __str__(self):
        return str(to_dict(self))

    def __repr__(self):
        return self.__str__()


def to_dict(obj):
    if not hasattr(obj, '__dict__'):
        return obj
    result = {}
    for key, value in obj.__dict__.items():
        if not key.startswith('__'):
            if isinstance(value, (list, tuple)):
                result[key] = [to_dict(item) for item in value]
            else:
                result[key] = to_dict(value)
    return result


if __name__ == '__main__':
    print(to_dict(Config))
    print(Config.ppo.lr)
