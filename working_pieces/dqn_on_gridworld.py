import numpy as np
from envs import small_gridworld as gridworld
from algos.dqn import DQN


def dqn_on_gridworld():
    gamma = 0.9
    agent = DQN(2, 5, gamma)
    # agent = QLearning(5, gamma=gamma)

    H = 5
    W = 5
    R_MAX = 2
    ACT_RAND = 0.0
    rmap_gt = np.zeros([H, W])
    rmap_gt[H - 2, W - 2] = R_MAX
    rmap_gt[1, 1] = R_MAX / 2

    gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)
    for i in range(1000):
        obs = gw.reset()
        rw = 0
        total_rw = 0
        while True:
            action = agent.update(rw, obs)
            obs, rw, done, info = gw.step(action)
            total_rw += rw
            if done:
                print(f'Episode {i} total reward: {total_rw}')
                print(f'Final state: {obs}')
                break


if __name__ == '__main__':
    dqn_on_gridworld()
