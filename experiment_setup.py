import wandb

from airl_gridworld_train import main
from config import CONFIG, set_experiment_config
from envs.env_factory import make_env


def increasing_grid_size_curriculum():
    CONFIG.env.vectorized = False
    test_env = make_env()
    CONFIG.env.vectorized = True

    # todo: dynamically trigger next env when trained on current env
    share_of_env_steps = [0.3, 0.7]
    grid_sizes = [5, 10]
    # max_steps = [15, 45]
    max_steps = [15, 30]
    lrs = [CONFIG.ppo.lr * 5, CONFIG.ppo.lr]

    last_trained_step = 0

    wandb.init(project='AIRL', dir='wandb', config=CONFIG.as_dict(),  tags=["curriculum", "increasing_grid_size"])
    wandb.config['curriculum'] = 'increasing_grid_size'
    wandb.config['grid_sizes'] = grid_sizes
    wandb.config['max_steps'] = max_steps
    wandb.config['total_steps'] = CONFIG.airl.env_steps

    for i in range(len(grid_sizes)):
        set_experiment_config(grid_size=grid_sizes[i], max_steps=max_steps[i], ppo_lr=lrs[i])
        CONFIG.airl.env_steps = int(share_of_env_steps[i] * wandb.config['total_steps'])
        if i > 0:
            CONFIG.airl.ppo_load_from = CONFIG.airl.ppo_save_to
            CONFIG.airl.disc_load_from = CONFIG.airl.disc_save_to
        last_trained_step = main(logging_start_step=last_trained_step, test_env=test_env)

    wandb.finish()


def positive_stripe_reward_curriculum():
    reward_configuration = ['positive_stripe', 'default']

    # convert from total steps to steps per curriculum item
    CONFIG.airl.env_steps = CONFIG.airl.env_steps // len(reward_configuration)

    wandb.init(project='AIRL', dir='wandb', config=CONFIG.as_dict(), tags=["curriculum", "positive_stripe_reward"])
    wandb.config['curriculum'] = 'positive_stripe_reward'
    wandb.config['reward_configuration'] = reward_configuration
    wandb.config['total_steps'] = CONFIG.airl.env_steps * (len(reward_configuration))

    for i in range(len(reward_configuration)):
        set_experiment_config(reward_configuration=reward_configuration[i])
        if i > 0:
            CONFIG.airl.ppo_load_from = CONFIG.airl.ppo_save_to
            CONFIG.airl.disc_load_from = CONFIG.airl.disc_save_to
        main(logging_start_step=i*CONFIG.airl.env_steps)

    wandb.finish()


if __name__ == '__main__':
    increasing_grid_size_curriculum()
    # positive_stripe_reward_curriculum()
