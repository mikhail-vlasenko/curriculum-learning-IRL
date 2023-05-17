import wandb
from tqdm import tqdm

from config import CONFIG
from rl_algos.ppo_from_airl import PPO, TrajectoryDataset, update_policy, device
import torch
from envs.env_factory import make_env


def main():
    env = make_env()

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions).to(device)

    if CONFIG.ppo_train.load_from is not None:
        print('Loading PPO from: ', CONFIG.ppo_train.load_from)
        ppo.load_state_dict(torch.load(CONFIG.ppo_train.load_from))

    optimizer = torch.optim.Adam(ppo.parameters(), lr=CONFIG.ppo.lr)
    dataset = TrajectoryDataset(batch_size=CONFIG.ppo.batch_size, n_workers=CONFIG.ppo.n_workers)

    states, info = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    for t in tqdm(range(int(CONFIG.ppo_train.env_steps / CONFIG.ppo.n_workers))):
        action, log_probs = ppo.act(states_tensor)
        next_states, rewards, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        train_ready = dataset.write_tuple(states, action, rewards, done, log_probs, logs=rewards)

        if train_ready:
            update_policy(ppo, dataset, optimizer, CONFIG.ppo.gamma, CONFIG.ppo.epsilon, CONFIG.ppo.update_epochs,
                          entropy_reg=CONFIG.ppo.entropy_reg)
            wandb.log({'Reward': dataset.log_objectives().mean(),
                       'Returns': dataset.log_returns().mean(),
                       'Lengths': dataset.log_lengths().mean()}, step=t * CONFIG.ppo.n_workers)

            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    torch.save(ppo.state_dict(), CONFIG.ppo_train.save_to)
    model_art = wandb.Artifact('expert_model', type='model')
    model_art.add_file(CONFIG.ppo_train.save_to)
    wandb.log_artifact(model_art)

    model_code_art = wandb.Artifact('model_code', type='code')
    model_code_art.add_file('rl_algos/ppo_from_airl.py')
    wandb.log_artifact(model_code_art)

    env.close()


if __name__ == '__main__':
    tags = []
    if CONFIG.ppo_train.load_from is not None:
        tags.append('continued_training')
    if 'OnlyEndReward' in CONFIG.env.wrappers:
        tags.append('only_end_reward')
    if 'FlattenObs' in CONFIG.env.wrappers:
        tags.append('flatten_obs')
    wandb.init(project='PPO', config=CONFIG.as_dict(), tags=tags)
    main()
