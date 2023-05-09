import wandb

from airl_gridworld_train import main
from config import CONFIG


EXPERT_DATA_PREFIX = 'demonstrations/ppo_demos_size'
EXPERT_DATA_SUFFIX = '.pk'


def increasing_grid_size_curriculum(start_from=0):
    grid_sizes = [5, 15]
    # learning rate schedule might be nice

    wandb.init(project='AIRL', dir='wandb', config=CONFIG.as_dict(),  tags=["curriculum", "increasing_grid_size"])
    wandb.config['curriculum'] = 'increasing_grid_size'
    wandb.config['grid_sizes'] = grid_sizes
    wandb.config['start_from'] = start_from
    wandb.config['total_steps'] = CONFIG.airl.env_steps * (len(grid_sizes) - start_from)

    for i in range(start_from, len(grid_sizes)):
        CONFIG.env.grid_size = grid_sizes[i]
        CONFIG.airl.expert_data_path = EXPERT_DATA_PREFIX + str(grid_sizes[i]) + EXPERT_DATA_SUFFIX
        if i > 0:
            CONFIG.airl.ppo_load_from = CONFIG.airl.ppo_save_path
            CONFIG.airl.disc_load_from = CONFIG.airl.disc_save_path
        main()

    wandb.finish()


if __name__ == '__main__':
    increasing_grid_size_curriculum()
