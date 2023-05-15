import random

from tqdm import tqdm

from config import CONFIG
from rl_algos.ppo_from_airl import PPO, device
import torch
import pickle
from envs.env_factory import make_env


def make_demos():
    reward_sum = 0
    CONFIG.env.vectorized = False
    env = make_env()

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # Load Pretrained PPO
    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions).to(device)
    ppo.load_state_dict(torch.load(CONFIG.ppo_train.load_from))

    states, info = env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    dataset = []
    episode = {'states': [], 'actions': []}

    for _ in tqdm(range(CONFIG.demos.n_steps)):
        action, log_probs = ppo.act(states_tensor)
        action = action.item()
        next_states, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward_sum += reward
        episode['states'].append(states)
        episode['actions'].append(action)

        if done:
            next_states, info = env.reset()
            dataset.append(episode)
            episode = {'states': [], 'actions': []}

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
    env.close()
    return dataset, reward_sum


if __name__ == '__main__':
    dataset, reward_sum = make_demos()
    print(f'Number of episodes: {len(dataset)}')
    print('Sample:')
    for _ in range(2):
        print(dataset[random.randint(0, len(dataset) - 1)])
    print(f'average reward: {reward_sum / len(dataset)}')
    pickle.dump(dataset, open(f'demonstrations/ppo_demos_size{CONFIG.env.grid_size}.pk', 'wb'))
