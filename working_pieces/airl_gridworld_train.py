import wandb

from tqdm import tqdm

from gym_examples.wrappers.vec_env import VecEnv
from irl_algos.airl import *
from rl_algos.ppo_from_airl import *
import torch
import numpy as np
import pickle
import gym_examples
from gym_examples.wrappers import RelativePosition
import gymnasium as gym

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('WARNING: CUDA not available. Using CPU.')


def main():
    config = wandb.config

    expert_trajectories = pickle.load(open(config.expert_data_path, 'rb'))

    # Create Environment
    sub_envs = []
    for i in range(config.n_workers):
        env = gym.make(config.env_id)
        sub_envs.append(RelativePosition(env))

    env = VecEnv(sub_envs)

    states, info = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # Initialize Models
    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions, simple_architecture=True).to(device)
    discriminator = DiscriminatorMLP(state_shape=obs_shape[0], simple_architecture=True).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=config.lr_discriminator)
    dataset = TrajectoryDataset(batch_size=config.ppo_update_episodes, n_workers=config.n_workers)

    for t in tqdm(range((int(config.env_steps / config.n_workers)))):
        action, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, _, info = env.step(action)

        # Calculate (vectorized) AIRL reward
        airl_state = torch.tensor(states).to(device).float()
        airl_next_state = torch.tensor(next_states).to(device).float()
        airl_action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
        airl_rewards = discriminator.predict_reward(airl_state, airl_next_state, 0.8,
                                                    airl_action_prob)
        airl_rewards = airl_rewards.detach().cpu().numpy()
        airl_rewards[done] = 0

        # Save Trajectory
        train_ready = dataset.write_tuple(states, action, airl_rewards, done, log_probs, logs=rewards)

        if train_ready:
            wandb.log({'Reward': dataset.log_objectives().mean()})
            wandb.log({'Returns': dataset.log_returns().mean()})
            wandb.log({'Lengths': dataset.log_lengths().mean()})

            # Update Models
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            d_loss, fake_acc, real_acc = update_discriminator(discriminator=discriminator,
                                                              optimizer=optimizer_discriminator,
                                                              gamma=config.gamma,
                                                              expert_trajectories=expert_trajectories,
                                                              policy_trajectories=dataset.trajectories.copy(),
                                                              ppo=ppo,
                                                              batch_size=config.batchsize_discriminator)

            wandb.log({'Discriminator Loss': d_loss,
                       'Fake Accuracy': fake_acc,
                       'Real Accuracy': real_acc})

            torch.save(discriminator.state_dict(), '../saved_models/checkpoints/discriminator_latest.pt')
            torch.save(ppo.state_dict(), '../saved_models/checkpoints/ppo_latest.pt')

            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    torch.save(discriminator.state_dict(), '../saved_models/discriminator.pt')
    torch.save(ppo.state_dict(), '../saved_models/ppo.pt')

    print('Saving models and data to wandb...')
    # save model artifacts to wandb
    model_art = wandb.Artifact('airl_models', type='model')
    model_art.add_file('../saved_models/discriminator.pt')
    model_art.add_file('../saved_models/ppo.pt')
    wandb.log_artifact(model_art)

    data_art = wandb.Artifact('airl_data', type='dataset')
    data_art.add_file(wandb.config.expert_data_path)
    wandb.log_artifact(data_art)

    wandb.finish()

    env.close()


if __name__ == '__main__':
    wandb.init(project='AIRL', dir='../wandb', config={
        'env_id': 'gym_examples/GridWorld-v0',
        'env_steps': 1000000,
        'batchsize_discriminator': 1024,
        'lr_discriminator': 5e-4,
        'ppo_update_episodes': 64,
        'lr_ppo': 1e-3,
        'n_workers': 64,
        'entropy_reg': 0,
        'gamma': 0.8,
        'epsilon': 0.1,
        'ppo_epochs': 5,
        'expert_data_path': '../demonstrations/ppo_demos.pk',
    })
    main()
