import wandb

from airl_gridworld_train import main as airl_main
from train_ppo import main as ppo_main
from config import CONFIG, set_experiment_config, set_curriculum_loading_paths, set_curriculum_steps
from envs.env_factory import make_env


main = airl_main if CONFIG.curriculum_for_airl else ppo_main
WANDB_PROJECT = 'AIRL' if CONFIG.curriculum_for_airl else 'PPO'


def increasing_grid_size_curriculum(test_env):
    # todo: dynamically trigger next env when trained on current env
    share_of_env_steps = [0.3, 0.7]
    grid_sizes = [5, 10]
    # max_steps = [15, 45]
    max_steps = [15, 30]
    lrs = [CONFIG.ppo.lr * 5, CONFIG.ppo.lr]

    last_trained_step = 0

    wandb.config['curriculum'] = 'increasing_grid_size'
    wandb.config['grid_sizes'] = grid_sizes
    wandb.config['max_steps'] = max_steps

    for i in range(len(grid_sizes)):
        set_experiment_config(grid_size=grid_sizes[i], max_steps=max_steps[i], ppo_lr=lrs[i])
        set_curriculum_steps(share_of_env_steps[i], wandb.config['total_steps'])
        set_curriculum_loading_paths(i)
        if i == len(share_of_env_steps) - 1:
            test_env = None
        last_trained_step = main(logging_start_step=last_trained_step, test_env=test_env)

    wandb.finish()


def positive_stripe_reward_curriculum(test_env):
    share_of_env_steps = [0.3, 0.7]
    reward_configuration = ['positive_stripe', 'default']

    last_trained_step = 0

    wandb.config['curriculum'] = 'positive_stripe_reward'
    wandb.config['reward_configuration'] = reward_configuration

    for i in range(len(reward_configuration)):
        set_experiment_config(reward_configuration=reward_configuration[i])
        set_curriculum_steps(share_of_env_steps[i], wandb.config['total_steps'])
        set_curriculum_loading_paths(i)
        if i == len(share_of_env_steps) - 1:
            test_env = None
        last_trained_step = main(logging_start_step=last_trained_step, test_env=test_env)

    wandb.finish()


def close_starts_curriculum(test_env):
    share_of_env_steps = [0.3, 0.7]
    spawn_distance = [2, -1]

    last_trained_step = 0

    wandb.config['curriculum'] = 'close_starts'
    wandb.config['shares_of_steps'] = share_of_env_steps

    for i in range(len(share_of_env_steps)):
        set_experiment_config(spawn_distance=spawn_distance[i])
        set_curriculum_steps(share_of_env_steps[i], wandb.config['total_steps'])
        set_curriculum_loading_paths(i)
        if i == len(share_of_env_steps) - 1:
            test_env = None
        last_trained_step = main(logging_start_step=last_trained_step, test_env=test_env)

    wandb.finish()


def sequential_curriculum(test_env):
    share_of_env_steps = [0.06] * 9
    share_of_env_steps.append(0.46)
    max_steps = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

    last_trained_step = 0

    wandb.config['curriculum'] = 'sequential'
    wandb.config['shares_of_steps'] = share_of_env_steps

    for i in range(len(share_of_env_steps)):
        set_experiment_config(max_steps=max_steps[i])
        set_curriculum_steps(share_of_env_steps[i], wandb.config['total_steps'])
        set_curriculum_loading_paths(i)
        if i == len(share_of_env_steps) - 1:
            test_env = None
        last_trained_step = main(logging_start_step=last_trained_step, test_env=test_env)

    wandb.finish()


if __name__ == '__main__':
    target_env = make_env()
    wandb.init(project=WANDB_PROJECT, dir='wandb', config=CONFIG.as_dict(),  tags=["curriculum"])
    wandb.config['total_steps'] = CONFIG.airl.env_steps if CONFIG.curriculum_for_airl else CONFIG.ppo_train.env_steps
    # increasing_grid_size_curriculum(target_env)
    # positive_stripe_reward_curriculum(target_env)
    # close_starts_curriculum(target_env)
    sequential_curriculum(target_env)
