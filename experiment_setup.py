import wandb

from airl_gridworld_train import main
from config import CONFIG, set_experiment_config


def increasing_grid_size_curriculum(start_from=0):
    grid_sizes = [5, 15]
    # learning rate schedule might be nice

    wandb.init(project='AIRL', dir='wandb', config=CONFIG.as_dict(),  tags=["curriculum", "increasing_grid_size"])
    wandb.config['curriculum'] = 'increasing_grid_size'
    wandb.config['grid_sizes'] = grid_sizes
    wandb.config['start_from'] = start_from
    wandb.config['total_steps'] = CONFIG.airl.env_steps * (len(grid_sizes) - start_from)

    for i in range(start_from, len(grid_sizes)):
        set_experiment_config(grid_size=grid_sizes[i])
        if i > 0:
            CONFIG.airl.ppo_load_from = CONFIG.airl.save_to
            CONFIG.airl.disc_load_from = CONFIG.airl.disc_save_to
        main()

    wandb.finish()


if __name__ == '__main__':
    increasing_grid_size_curriculum()
