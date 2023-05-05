import wandb

from envs.small_gridworld import GridWorld
from tqdm import tqdm
from irl_algos.airl import *
from rl_algos.ppo_from_airl import *
import torch
import numpy as np
import pickle

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Load demonstrations
    expert_trajectories = pickle.load(open('../demonstrations/ppo_demos_v3_[0,1,0,1].pk', 'rb'))

    # Init WandB & Parameters
    wandb.init(project='AIRL', config={
        'env_id': 'randomized_v3',
        'env_steps': 6e6,
        'batchsize_discriminator': 512,
        'batchsize_ppo': 12,
        'n_workers': 12,
        'entropy_reg': 0,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config

    # Create Environment
    H = 5
    W = 5
    ACT_RAND = 0.0
    rmap_gt = np.zeros([H, W])
    rmap_gt[H - 2, W - 2] = 5
    rmap_gt[1, 1] = 1

    env = GridWorld(rmap_gt, {}, 1 - ACT_RAND)
    env.show_grid()
    states = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    ppo = PPO(state_shape=state_shape, n_actions=n_actions, in_channels=in_channels).to(device)
    discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=5e-4)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=5e-5)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    # Logging
    objective_logs = []

    for t in tqdm(range((int(config.env_steps/config.n_workers)))):

        # Act
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Log Objectives
        objective_logs.append(rewards)

        # Calculate (vectorized) AIRL reward
        airl_state = torch.tensor(states).to(device).float()
        airl_next_state = torch.tensor(next_states).to(device).float()
        airl_action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
        airl_rewards = discriminator.predict_reward(airl_state, airl_next_state, config.gamma, airl_action_prob)
        airl_rewards = list(airl_rewards.detach().cpu().numpy() * [0 if i else 1 for i in done])

        # Save Trajectory
        train_ready = dataset.write_tuple(states, actions, airl_rewards, done, log_probs)

        if train_ready:
            # Log Objectives
            objective_logs = np.array(objective_logs).sum(axis=0)
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
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
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    # env.close()
    torch.save(discriminator.state_dict(), 'saved_models/discriminator_v3_[0,1,0,1].pt')
