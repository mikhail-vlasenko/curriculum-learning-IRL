import wandb

from airl_gridworld_train import main
from config import CONFIG, set_experiment_config


def increasing_grid_size_curriculum(start_from=0):
    grid_sizes = [5, 15]
    max_steps = [15, 45]
    # learning rate schedule might be nice

    wandb.init(project='AIRL', dir='wandb', config=CONFIG.as_dict(),  tags=["curriculum", "increasing_grid_size"])
    wandb.config['curriculum'] = 'increasing_grid_size'
    wandb.config['grid_sizes'] = grid_sizes
    wandb.config['max_steps'] = max_steps
    wandb.config['start_from'] = start_from
    wandb.config['total_steps'] = CONFIG.airl.env_steps * (len(grid_sizes) - start_from)

    for i in range(start_from, len(grid_sizes)):
        set_experiment_config(grid_size=grid_sizes[i], max_steps=max_steps[i])
        if i > 0:
            CONFIG.airl.ppo_load_from = CONFIG.airl.ppo_save_to
            CONFIG.airl.disc_load_from = CONFIG.airl.disc_save_to
        main()

    wandb.finish()


def checkers_negative_reward_curriculum():
    checkers_negative_rewards = [True, False]
    wandb.init(project='AIRL', dir='wandb', config=CONFIG.as_dict(), tags=["curriculum", "checkers_negative_reward"])
    wandb.config['curriculum'] = 'checkers_negative_reward'
    wandb.config['checkers_negative_rewards'] = checkers_negative_rewards

    for i in range(len(checkers_negative_rewards)):
        set_experiment_config(checkers_negative_reward=checkers_negative_rewards[i])
        if i > 0:
            CONFIG.airl.ppo_load_from = CONFIG.airl.ppo_save_to
            CONFIG.airl.disc_load_from = CONFIG.airl.disc_save_to
        main()

    wandb.finish()


if __name__ == '__main__':
    increasing_grid_size_curriculum()
