import numpy as np
import gymnasium as gym
from algos.q_learning import q_learning
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    env = gym.make('CliffWalking-v0')
    q_table, rewards, lengths = q_learning(env, 100)
    print()
    for i in range(37):
        assert i in q_table
        print(np.argmax(q_table[i]), end=' ')
        if (i + 1) % 12 == 0:
            print()
    sns.lineplot(x=np.arange(len(rewards)), y=rewards, label='rewards')
    sns.lineplot(x=np.arange(len(lengths)), y=lengths, label='lengths')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
