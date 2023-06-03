from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import CONFIG
from rl_algos.ppo_from_airl import PPO, device


class DiscriminatorMLP(nn.Module):
    def __init__(self, state_shape, in_channels=6):
        super(DiscriminatorMLP, self).__init__()

        self.state_shape = state_shape
        self.in_channels = in_channels
        self.simple_architecture = CONFIG.discriminator.simple_architecture

        # Layers
        # self.action_embedding = nn.Linear(n_actions, state_shape[0]*state_shape[1])
        if self.simple_architecture:
            self.reward_l1 = nn.Linear(state_shape, CONFIG.discriminator.dimensions[0])
            self.reward_l2 = nn.Linear(CONFIG.discriminator.dimensions[0], CONFIG.discriminator.dimensions[1])
            self.reward_out = nn.Linear(CONFIG.discriminator.dimensions[1], 1)
        else:
            self.reward_l1 = nn.Linear(self.in_channels*self.state_shape[0]*self.state_shape[1], 256)
            self.reward_l2 = nn.Linear(256, 512)
            self.reward_out = nn.Linear(256, 1)

        if self.simple_architecture:
            self.value_l1 = nn.Linear(state_shape, CONFIG.discriminator.dimensions[0])
            self.value_l2 = nn.Linear(CONFIG.discriminator.dimensions[0], CONFIG.discriminator.dimensions[1])
            self.value_out = nn.Linear(CONFIG.discriminator.dimensions[1], 1)
        else:
            self.value_l1 = nn.Linear(self.in_channels*self.state_shape[0]*self.state_shape[1], 256)
            self.value_l2 = nn.Linear(256, 512)
            self.value_out = nn.Linear(256, 1)

        self.value_l3 = nn.Linear(512, 256)
        self.reward_l3 = nn.Linear(512, 256)

        # Activation
        # self.relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()

    def g(self, state):
        state = state.view(state.shape[0], -1)

        if self.simple_architecture:
            x = self.relu(self.reward_l1(state))
            x = self.relu(self.reward_l2(x))
            x = x.view(x.shape[0], -1)
            x = self.reward_out(x)
            return x

        x = self.relu(self.reward_l1(state))
        x = self.relu(self.reward_l2(x))
        x = self.relu(self.reward_l3(x))
        x = x.view(x.shape[0], -1)
        x = self.reward_out(x)

        return x

    def h(self, state):
        state = state.view(state.shape[0], -1)

        if self.simple_architecture:
            x = self.relu(self.value_l1(state))
            x = self.relu(self.value_l2(x))
            x = x.view(x.shape[0], -1)
            x = self.value_out(x)
            return x

        x = self.relu(self.value_l1(state))
        x = self.relu(self.value_l2(x))
        x = self.relu(self.value_l3(x))
        x = x.view(x.shape[0], -1)
        x = self.value_out(x)

        return x

    def forward(self, state, next_state, gamma):
        reward = self.g(state)
        value_state = self.h(state)
        value_next_state = self.h(next_state)

        advantage = reward + gamma*value_next_state - value_state

        return advantage

    def sub_probs(self, state, next_state, gamma, log_prob):
        return self.forward(state, next_state, gamma) - log_prob

    def discriminate(self, state, next_state, gamma, action_probability):
        advantage = self.forward(state, next_state, gamma)
        advantage = advantage.squeeze(1)
        exp_advantage = torch.exp(advantage)
        # print((exp_advantage/(exp_advantage + action_probability + 1e-5)).shape)

        print(exp_advantage/(exp_advantage + action_probability))
        return exp_advantage/(exp_advantage + action_probability)

    def predict_reward(self, state, next_state, gamma, action_probability):
        advantage = self.forward(state, next_state, gamma)
        advantage = advantage.squeeze(1)

        return advantage - torch.log(action_probability)


class Discriminator(nn.Module):
    def __init__(self, state_shape, in_channels=6, latent_dim=None):
        super(Discriminator, self).__init__()

        self.state_shape = state_shape
        self.in_channels = in_channels
        self.eval = False
        self.utopia_point = None

        # Latent conditioning
        if latent_dim is not None:
            self.latent_dim = latent_dim
            self.latent_embedding_value = nn.Linear(latent_dim, state_shape[0] * state_shape[1])
            self.latent_embedding_reward = nn.Linear(latent_dim, state_shape[0] * state_shape[1])
            self.in_channels = in_channels+1

        # Layers
        # self.action_embedding = nn.Linear(n_actions, state_shape[0]*state_shape[1])
        self.reward_conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=2)
        self.reward_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.reward_conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2)
        self.reward_out = nn.Linear(16*(state_shape[0]-3)*(state_shape[1]-3), 1)

        self.value_conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=2)
        self.value_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.value_conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2)
        self.value_out = nn.Linear(16*(state_shape[0]-3)*(state_shape[1]-3), 1)

        # Activation
        self.relu = nn.LeakyReLU(0.01)

    def set_eval(self):
        self.eval = True

    def g(self, state, latent=None):
        state = state.view(-1, self.in_channels, self.state_shape[0], self.state_shape[1])

        if latent is not None:
            latent = F.one_hot(latent.long(), self.latent_dim).float().to(device)
            latent = self.latent_embedding_reward(latent)
            latent = latent.view(-1, 1, self.state_shape[0], self.state_shape[1])
            if latent.shape[0] == 1:
                latent = latent.repeat_interleave(repeats=state.shape[0], dim=0)

            state = torch.cat([state, latent], dim=1)
        # Conv + Linear
        x = self.relu(self.reward_conv1(state))
        x = self.relu(self.reward_conv2(x))
        x = self.relu(self.reward_conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.reward_out(x)

        return x

    def h(self, state, latent=None):
        state = state.view(-1, self.in_channels, self.state_shape[0], self.state_shape[1])

        if latent is not None:
            latent = F.one_hot(latent.long(), self.latent_dim).float().to(device)
            latent = self.latent_embedding_value(latent)
            latent = latent.view(-1, 1, self.state_shape[0], self.state_shape[1])
            if latent.shape[0] == 1:
                latent = latent.repeat_interleave(repeats=state.shape[0], dim=0)

            state = torch.cat([state, latent], dim=1)
        # Conv + Linear
        x = self.relu(self.value_conv1(state))
        x = self.relu(self.value_conv2(x))
        x = self.relu(self.value_conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.value_out(x)

        return x

    def forward(self, state, next_state, gamma, latent=None):
        reward = self.g(state, latent)
        value_state = self.h(state, latent)
        value_next_state = self.h(next_state, latent)

        advantage = reward + gamma*value_next_state - value_state

        if self.eval:
            return advantage/np.abs(self.utopia_point)
        else:
            return advantage

    def discriminate(self, state, next_state, gamma, action_probability, latent=None):
        if latent is not None:
            advantage = self.forward(state, next_state, gamma, latent)
        else:
            advantage = self.forward(state, next_state, gamma)
        advantage = advantage.squeeze(1)
        exp_advantage = torch.exp(advantage)
        #print((exp_advantage/(exp_advantage + action_probability + 1e-5)).shape)

        print(exp_advantage/(exp_advantage + action_probability))
        return exp_advantage/(exp_advantage + action_probability)

    def predict_reward(self, state, next_state, gamma, action_probability, latent=None):
        if latent is not None:
            advantage = self.forward(state, next_state, gamma, latent)
        else:
            advantage = self.forward(state, next_state, gamma)

        advantage = advantage.squeeze(1)

        return advantage - torch.log(action_probability)

    # def estimate_utopia(self, imitation_policy, config, steps=10000):
    #     env = GymWrapper(config.env_id)
    #     states = env.reset()
    #     states_tensor = torch.tensor(states).float().to(device)
    #
    #     # Fetch Shapes
    #     n_actions = env.action_space.n
    #     obs_shape = env.observation_space.shape
    #     state_shape = obs_shape[:-1]
    #     in_channels = obs_shape[-1]
    #
    #     # Init returns
    #     estimated_returns = []
    #     running_returns = 0
    #
    #     for t in range(steps):
    #         actions, log_probs = imitation_policy.act(states_tensor)
    #         next_states, rewards, done, info = env.step(actions)
    #
    #         airl_state = torch.tensor(states).to(device).float()
    #         airl_next_state = torch.tensor(next_states).to(device).float()
    #         airl_rewards = self.forward(airl_state, airl_next_state, config.gamma).item()
    #         if done:
    #             airl_rewards = 0
    #             next_states = env.reset()
    #         running_returns += airl_rewards
    #
    #         if done:
    #             estimated_returns.append(running_returns)
    #             running_returns = 0
    #
    #         states = next_states.copy()
    #         states_tensor = torch.tensor(states).float().to(device)
    #
    #     self.utopia_point = sum(estimated_returns)/len(estimated_returns)
    #
    #     return self.utopia_point


def training_sampler(
        expert_trajectories: List[Dict], policy_trajectories: List[Dict], ppo: PPO, batch_size,
        only_expert=False, only_policy=False
):
    assert not (only_expert and only_policy), "Cannot sample only expert and only policy"
    states = []
    selected_actions = []
    next_states = []
    labels = []
    latents = []
    for i in range(batch_size):
        # 1 if (s,a,s') comes from expert, 0 otherwise
        if only_expert:
            expert_boolean = 1
        elif only_policy:
            expert_boolean = 0
        else:
            expert_boolean = np.random.randint(2)
        # expert_boolean = 1 if i < batch_size/2 else 0
        if expert_boolean == 1:
            selected_trajectories = expert_trajectories
        else:
            selected_trajectories = policy_trajectories

        random_tau = []
        random_tau_idx = 0
        while len(random_tau) < 2:
            random_tau_idx = np.random.randint(len(selected_trajectories))
            random_tau = selected_trajectories[random_tau_idx]['states']

        random_state_idx = np.random.randint(len(random_tau)-1)
        state = random_tau[random_state_idx]
        next_state = random_tau[random_state_idx+1]

        # Get the action that was actually selected in the trajectory
        selected_action = selected_trajectories[random_tau_idx]['actions'][random_state_idx]

        states.append(state)
        next_states.append(next_state)
        selected_actions.append(selected_action)
        labels.append(expert_boolean)

    states = np.stack(states, axis=0)
    batched_action_probability, _ = ppo.forward(torch.tensor(states).float().to(device))
    batched_action_probability = batched_action_probability.squeeze(0)
    action_probabilities = [batched_action_probability[i][selected_actions[i]].item() for i in range(batch_size)]

    return torch.tensor(states).float().to(device), \
        torch.tensor(np.stack(next_states, axis=0)).float().to(device), \
        torch.tensor(action_probabilities).float().to(device),\
        torch.tensor(labels).long().to(device), torch.tensor(latents).float().to(device)


def update_discriminator(
        discriminator: DiscriminatorMLP, optimizer: torch.optim.Optimizer, gamma,
        expert_trajectories, policy_trajectories,
        ppo, batch_size, latent_posterior=None
):
    criterion = nn.CrossEntropyLoss()
    states, next_states, action_probabilities, labels, latents = training_sampler(
        expert_trajectories, policy_trajectories, ppo, batch_size, latent_posterior
    )
    if CONFIG.airl.freeze_ppo_weights:
        action_probabilities = action_probabilities.detach()
    if len(latents) > 0:
        raise NotImplementedError
    else:
        advantages = discriminator.forward(states, next_states, gamma)
    # Cat advantages and log_probs to (batch_size, 2)
    # todo: why do we compare log action probas with advantage?
    class_predictions = torch.cat([torch.log(action_probabilities).unsqueeze(1), advantages], dim=1)
    # Compute Loss function
    loss = criterion(class_predictions, labels)
    # Compute Accuracies
    label_predictions = torch.argmax(class_predictions, dim=1)
    predicted_fake = (label_predictions[labels == 0] == 0).float()
    predicted_expert = (label_predictions[labels == 1] == 1).float()

    # print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), torch.mean(predicted_fake).item(), torch.mean(predicted_expert).item()


def update_discriminator_mine(discriminator, optimizer, gamma, expert_trajectories, policy_trajectories,
                              ppo, batch_size, latent_posterior=None):
    ex_states, ex_next_states, ex_action_probabilities, labels, latents\
        = training_sampler(expert_trajectories, policy_trajectories, ppo, batch_size, latent_posterior, only_expert=True)
    po_states, po_next_states, po_action_probabilities, labels, latents\
        = training_sampler(expert_trajectories, policy_trajectories, ppo, batch_size, latent_posterior, only_policy=True)
    if len(latents) > 0:
        raise NotImplementedError
    else:
        ex_advantages = discriminator.sub_probs(ex_states, ex_next_states, gamma, ex_action_probabilities)
        po_advantages = discriminator.sub_probs(ex_states, ex_next_states, gamma, po_action_probabilities)

    # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
    loss_exp = -F.logsigmoid(ex_advantages).mean()
    loss_po = -F.logsigmoid(-po_advantages).mean()
    loss = loss_exp + loss_po

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # # Compute Accuracies
    # label_predictions = torch.argmax(class_predictions, dim=1)
    # predicted_fake = (label_predictions[labels == 0] == 0).float()
    # predicted_expert = (label_predictions[labels == 1] == 1).float()
    predicted_fake = torch.zeros(batch_size)
    predicted_expert = torch.ones(batch_size)

    return loss.item(), torch.mean(predicted_fake).item(), torch.mean(predicted_expert).item()

