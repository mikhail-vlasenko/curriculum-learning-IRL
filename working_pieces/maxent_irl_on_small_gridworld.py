#!/usr/bin/env python

from irl_maxent import gridworld as W
from irl_maxent import maxent as M
from irl_maxent import plot as P
from irl_maxent import trajectory as T
from irl_maxent import solver as S
from irl_maxent import optimizer as O

import numpy as np
import matplotlib.pyplot as plt
from irl_maxent.trajectory import Trajectory

from utils.demonstrations import generate_demonstrations
from envs import small_gridworld as gridworld
from rl_algos.q_learning import q_learning, QLearningAgent


def setup_mdp():
    """
    Set-up our MDP/GridWorld
    """
    # create our world
    world = W.GridWorld(size=5)

    # set up the reward function
    reward = np.zeros(world.n_states)
    reward[18] = 1.0
    # reward[8] = 0.65

    # set up terminal states
    terminal = [18]

    return world, reward, terminal


def generate_trajectories(world, reward, terminal):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    n_trajectories = 100
    discount = 0.7
    weighting = lambda x: x**5

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[0] = 1.0

    # generate trajectories
    value = S.value_iteration(world.p_transition, reward, discount)
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal))

    return tjs, policy


def maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = W.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward


def main():
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    # set-up mdp
    world, reward, terminal = setup_mdp()

    H = 5
    W = 5
    ACT_RAND = 0.0
    rmap_gt = np.zeros([H, W])
    rmap_gt[H - 2, W - 2] = 5
    # rmap_gt[1, 1] = 1
    terminals = {(H - 2, W - 2)}

    env = gridworld.GridWorld(rmap_gt, terminals, 1 - ACT_RAND)
    env.show_grid()

    q_table, rewards, lengths = q_learning(env, 5000, discount_factor=0.7, alpha=0.1, epsilon=0.3)
    print('Using the following Q-table:')
    env.print_q_table(q_table)

    epsilon = 0.2
    agent = QLearningAgent(q_table, epsilon, env.action_space.n)

    # show our original reward
    ax = plt.figure(num='Original Reward').add_subplot(111)
    P.plot_state_values(ax, world, reward, **style)
    plt.draw()

    # # generate "expert" trajectories
    # trajectories, expert_policy = generate_trajectories(world, reward, terminal)

    trajectories = generate_demonstrations(env, agent, False, num_demos=500)
    for i in range(len(trajectories)):
        trajectories[i] = Trajectory(trajectories[i])
    print(trajectories[0])

    # show our expert policies
    ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    # P.plot_stochastic_policy(ax, world, expert_policy, **style)

    for t in trajectories:
        P.plot_trajectory(ax, world, t, lw=5, color='black', alpha=0.025)

    plt.draw()

    # maximum causal entropy reinforcement learning
    reward_maxcausal = maxent_causal(world, terminal, trajectories)
    print(reward_maxcausal)

    # show the computed reward
    ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
    P.plot_state_values(ax, world, reward_maxcausal, **style)
    plt.draw()

    plt.show()


if __name__ == '__main__':
    main()
