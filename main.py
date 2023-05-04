import pickle

import numpy as np
import gymnasium as gym
from tqdm import tqdm

from envs import small_gridworld as gridworld
from rl_algos.q_learning import q_learning, QLearningAgent
import seaborn as sns
import matplotlib.pyplot as plt
import random


def main():
    H = 5
    W = 5
    ACT_RAND = 0.0
    rmap_gt = np.zeros([H, W])
    rmap_gt[H - 2, W - 2] = 5
    rmap_gt[1, 1] = 1

    env = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)
    env.show_grid()
    q_table, rewards, lengths = q_learning(env, 1000, discount_factor=0.9, alpha=0.1, epsilon=0.3)

    env.print_q_table(q_table)
    sns.lineplot(x=np.arange(len(rewards)), y=rewards, label='rewards')
    sns.lineplot(x=np.arange(len(lengths)), y=lengths, label='lengths')
    plt.legend()
    plt.show()

    epsilon = 0.2
    agent = QLearningAgent(q_table, epsilon, env.action_space.n)
    states, info = env.reset()
    dataset = []
    episode = {'states': [], 'actions': []}
    episode_cnt = 0

    # Fetch Shapes
    # n_actions = env.action_space.n
    # obs_shape = env.observation_space.shape
    # state_shape = obs_shape[:-1]
    # in_channels = obs_shape[-1]

    # Load Pretrained PPO
    max_steps = 25
    n_demos = 100
    for i in tqdm(range((max_steps - 1) * n_demos)):
        action = agent.act(states)
        next_states, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode['states'].append(states)
        # Note: Actions currently append as arrays and not integers!
        episode['actions'].append(action)

        if done:
            next_states, info = env.reset()
            dataset.append(episode)
            episode = {'states': [], 'actions': []}
            episode_cnt += 1

        # Prepare state input for next time step
        states = next_states
        # states_tensor = torch.tensor(states).float().to(device)

    print('Sample:')
    for _ in range(2):
        print(dataset[random.randint(0, len(dataset) - 1)])
    pickle.dump(dataset, open('./demonstrations/try1' + '.pk', 'wb'))
    print(f'Dumped {episode_cnt} episodes to ./demonstrations/try1.pk')


if __name__ == '__main__':
    main()
