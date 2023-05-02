import numpy as np
import gymnasium as gym
from envs import small_gridworld as gridworld
from algos.q_learning import q_learning
import seaborn as sns
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    main()
