import torch
import torch.nn as nn
import torch.optim as optim
from utils import value_iteration as value_iteration

from utils.utils import *


class DeepIRLFC(nn.Module):
    def __init__(self, n_input, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_fc'):
        super(DeepIRLFC, self).__init__()
        self.n_input = n_input
        self.lr = lr
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.name = name

        self.fc1 = nn.Linear(n_input, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.reward = nn.Linear(n_h2, 1)

        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=l2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.reward(x)
        return x


def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
    """compute the expected states visition frequency p(s| theta, T)
    using dynamic programming

    inputs:
      P_a     NxNxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of list of Steps - collected from expert
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


    returns:
      p       Nx1 vector - state visitation frequencies
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    T = len(trajs[0])
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    for traj in trajs:
        mu[traj[0].cur_state, 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajs)

    for s in range(N_STATES):
        for t in range(T - 1):
            if deterministic:
                mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
            else:
                mu[s, t + 1] = sum(
                    [sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in
                     range(N_STATES)])
    p = np.sum(mu, 1)
    return p


def demo_svf(trajs, n_states):
    """
    compute state visitation frequences from demonstrations

    input:
      trajs   list of list of Steps - collected from expert
    returns:
      p       Nx1 vector - state visitation frequences
    """

    p = np.zeros(n_states)
    for traj in trajs:
        for step in traj:
            p[step.cur_state] += 1
    p = p / len(trajs)
    return p


def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    feat_map = torch.tensor(feat_map, dtype=torch.float32)

    nn_r = DeepIRLFC(feat_map.shape[1], lr, 3, 3)
    nn_r.train()

    mu_D = demo_svf(trajs, N_STATES)

    for iteration in range(n_iters):
        if iteration % (n_iters/10) == 0:
            print('iteration: {}'.format(iteration))

        rewards = nn_r(feat_map)

        _, policy = value_iteration.value_iteration(P_a, rewards.detach().numpy(), gamma, error=0.01, deterministic=True)

        mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)

        grad_r = torch.tensor(mu_D - mu_exp, dtype=torch.float32).unsqueeze(-1)

        nn_r.zero_grad()
        rewards.backward(gradient=grad_r, retain_graph=True)  # Changed this line
        nn_r.optimizer.step()

    nn_r.eval()
    rewards = nn_r(feat_map)
    return normalize(rewards.detach().numpy())


# def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
#     N_STATES, _, N_ACTIONS = np.shape(P_a)
#     feat_map = torch.tensor(feat_map, dtype=torch.float32)
#
#     nn_r = DeepIRLFC(feat_map.shape[1], lr, 3, 3)
#     nn_r.train()
#
#     mu_D = demo_svf(trajs, N_STATES)
#
#     for iteration in range(n_iters):
#         if iteration % (n_iters/10) == 0:
#             print('iteration: {}'.format(iteration))
#
#         rewards = nn_r(feat_map)
#
#         _, policy = value_iteration.value_iteration(P_a, rewards.detach().numpy(), gamma, error=0.01, deterministic=True)
#
#         mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
#
#         grad_r = torch.tensor(mu_D - mu_exp, dtype=torch.float32).unsqueeze(-1)
#
#         nn_r.zero_grad()
#         rewards.backward(grad_r)
#         nn_r.optimizer.step()
#
#     nn_r.eval()
#     rewards = nn_r(feat_map)
#     return normalize(rewards.detach().numpy())

