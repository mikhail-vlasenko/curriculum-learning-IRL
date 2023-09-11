import random

from tqdm import tqdm

from config import CONFIG, get_demo_name
from rl_algos.ppo_from_airl import PPO, device
import torch
import pickle
from envs.env_factory import make_env


LOAD_FROM = CONFIG.demos.load_from
# LOAD_FROM = 'saved_models/airl_ppo.pt'


def make_demos(load_from: str = None):
    """
    Generates demonstrations using a loaded PPO policy and writes them to file.
    """
    if load_from is None:
        load_from = CONFIG.demos.load_from

    reward_sum = 0
    CONFIG.env.vectorized = False
    env = make_env()

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # Load Pretrained PPO
    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions).to(device)
    print('Loading PPO from: ', load_from)
    ppo.load_state_dict(torch.load(load_from))

    states, info = env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    dataset = []
    episode = {'states': [], 'actions': [], 'rewards': []}

    for _ in tqdm(range(CONFIG.demos.n_steps)):
        action, log_probs = ppo.act(states_tensor)
        action = action.item()
        next_states, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward_sum += reward
        episode['states'].append(states)
        episode['actions'].append(action)
        episode['rewards'].append(reward)

        if done:
            next_states, info = env.reset()
            dataset.append(episode)
            episode = {'states': [], 'actions': [], 'rewards': []}

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
    env.close()
    return dataset, reward_sum


if __name__ == '__main__':
    dataset, reward_sum = make_demos(LOAD_FROM)
    print(f'Number of episodes: {len(dataset)}')
    print('Sample:')
    for _ in range(2):
        print(dataset[random.randint(0, len(dataset) - 1)])
    print(f'average reward: {reward_sum / len(dataset)}')
    if LOAD_FROM == 'saved_models/airl_ppo.pt':
        print('Saving to demonstrations/from_airl_policy.pk')
        pickle.dump(dataset, open('demonstrations/from_airl_policy.pk', 'wb'))
    else:
        if CONFIG.demos.save_path is None:
            print(f'Saving to {get_demo_name()}')
            pickle.dump(dataset, open(get_demo_name(), 'wb'))
        else:
            print(f'Saving to {CONFIG.demos.save_path}')
            pickle.dump(dataset, open(CONFIG.demos.save_path, 'wb'))
