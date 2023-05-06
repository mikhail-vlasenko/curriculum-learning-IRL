import wandb

from tqdm import tqdm
from irl_algos.airl import *
from rl_algos.ppo_from_airl import *
import torch
import numpy as np
import pickle
import gym_examples
from gym_examples.wrappers import RelativePosition
import gymnasium as gym

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Load demonstrations
    expert_trajectories = pickle.load(open('../demonstrations/ppo_demos.pk', 'rb'))

    # Init WandB & Parameters
    wandb.init(project='AIRL', dir='../wandb', config={
        'env_id': 'randomized_v3',
        'env_steps': 50000,
        'batchsize_discriminator': 256,
        'batchsize_ppo': 32,
        'n_workers': 1,
        'entropy_reg': 0,
        'gamma': 0.8,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config

    # Create Environment
    env = gym.make('gym_examples/GridWorld-v0', render_mode=None)
    env = RelativePosition(env)
    states, info = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # Initialize Models
    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions, simple_architecture=True).to(device)
    discriminator = DiscriminatorMLP(state_shape=obs_shape[0], simple_architecture=True).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=5e-4)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=5e-5)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    # Logging
    objective_logs = []

    for t in tqdm(range((int(config.env_steps/config.n_workers)))):

        # Act
        action, log_probs = ppo.act(states_tensor)
        next_states, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Log Objectives
        objective_logs.append(rewards)

        # Calculate (vectorized) AIRL reward
        airl_state = torch.tensor(states).to(device).float()
        airl_next_state = torch.tensor(next_states).to(device).float()
        airl_action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
        airl_rewards = discriminator.predict_reward(airl_state.unsqueeze(0), airl_next_state.unsqueeze(0), config.gamma, airl_action_prob)
        airl_rewards = list(airl_rewards.detach().cpu().numpy() * (0 if done else 1))
        # airl_rewards = list(airl_rewards.detach().cpu().numpy())

        # Save Trajectory
        train_ready = dataset.write_tuple([states], [action], airl_rewards, [done], [log_probs])

        if train_ready:
            # Log Objectives
            # objective_logs = dataset.log_objectives()
            # for ret in objective_logs:
            #     wandb.log({'Reward': ret})
            objective_logs = []

            # Update Models
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            d_loss, fake_acc, real_acc = update_discriminator(discriminator=discriminator,
                                                              optimizer=optimizer_discriminator,
                                                              gamma=config.gamma,
                                                              expert_trajectories=expert_trajectories,
                                                              policy_trajectories=dataset.trajectories.copy(), ppo=ppo,
                                                              batch_size=config.batchsize_discriminator)

            # Log Loss Statsitics
            wandb.log({'Discriminator Loss': d_loss,
                       'Fake Accuracy': fake_acc,
                       'Real Accuracy': real_acc})
            wandb.log({'Reward': dataset.log_objectives().mean()})
            wandb.log({'Returns': dataset.log_returns().mean()})
            wandb.log({'Lengths': dataset.log_lengths().mean()})

            dataset.reset_trajectories()

        if done:
            next_states, info = env.reset()
        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    torch.save(discriminator.state_dict(), '../saved_models/discriminator.pt')
    env.close()
