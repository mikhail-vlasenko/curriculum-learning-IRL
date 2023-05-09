import wandb

from tqdm import tqdm

from config import CONFIG
from env_factory import make_env
from irl_algos.airl import *
from rl_algos.ppo_from_airl import *
import torch
import numpy as np
import pickle


def main():
    CONFIG.ppo.entropy_reg = 0.0

    expert_trajectories = pickle.load(open(CONFIG.airl.expert_data_path, 'rb'))

    # Create Environment
    env = make_env()

    state, info = env.reset()
    state_tensor = torch.tensor(state).float().to(device)

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # Initialize Models
    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions, simple_architecture=True).to(device)
    discriminator = DiscriminatorMLP(state_shape=obs_shape[0], simple_architecture=True).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=CONFIG.ppo.lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=CONFIG.discriminator.lr)
    dataset = TrajectoryDataset(batch_size=CONFIG.ppo.batch_size, n_workers=CONFIG.ppo.n_workers)

    for _ in tqdm(range((int(CONFIG.airl.env_steps / CONFIG.ppo.n_workers)))):
        action, log_probs = ppo.act(state_tensor)
        next_state, reward, done, _, info = env.step(action)
        next_state_tensor = torch.tensor(next_state).to(device).float()

        # Calculate (vectorized) AIRL reward
        airl_action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
        airl_rewards = discriminator.predict_reward(
            state_tensor, next_state_tensor, 0.8, airl_action_prob
        )
        airl_rewards = airl_rewards.detach().cpu().numpy()
        airl_rewards[done] = 0

        # Save Trajectory
        train_ready = dataset.write_tuple(state, action, airl_rewards, done, log_probs, logs=reward)

        if train_ready:
            wandb.log({'Reward': dataset.log_objectives().mean()})
            wandb.log({'Returns': dataset.log_returns().mean()})
            wandb.log({'Lengths': dataset.log_lengths().mean()})

            # Update Models
            update_policy(ppo, dataset, optimizer, CONFIG.ppo.gamma, CONFIG.ppo.epsilon, CONFIG.ppo.update_epochs,
                          entropy_reg=CONFIG.ppo.entropy_reg)
            d_loss, fake_acc, real_acc = update_discriminator(discriminator=discriminator,
                                                              optimizer=optimizer_discriminator,
                                                              gamma=CONFIG.ppo.gamma,
                                                              expert_trajectories=expert_trajectories,
                                                              policy_trajectories=dataset.trajectories,
                                                              ppo=ppo,
                                                              batch_size=CONFIG.discriminator.batch_size)

            wandb.log({'Discriminator Loss': d_loss,
                       'Fake Accuracy': fake_acc,
                       'Real Accuracy': real_acc})

            torch.save(discriminator.state_dict(), 'saved_models/checkpoints/discriminator_latest.pt')
            torch.save(ppo.state_dict(), 'saved_models/checkpoints/ppo_latest.pt')

            dataset.reset_trajectories()

        # Prepare state input for next time step
        state = next_state
        state_tensor = next_state_tensor

    torch.save(discriminator.state_dict(), 'saved_models/discriminator.pt')
    torch.save(ppo.state_dict(), 'saved_models/ppo.pt')

    print('Saving models and data to wandb...')
    # save model artifacts to wandb
    model_art = wandb.Artifact('airl_models', type='model')
    model_art.add_file('saved_models/discriminator.pt')
    model_art.add_file('saved_models/ppo.pt')
    wandb.log_artifact(model_art)

    data_art = wandb.Artifact('airl_data', type='dataset')
    data_art.add_file(CONFIG.airl.expert_data_path)
    wandb.log_artifact(data_art)

    wandb.finish()

    env.close()


if __name__ == '__main__':
    CONFIG.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('WARNING: CUDA not available. Using CPU.')

    wandb.init(project='AIRL', dir='wandb', config=CONFIG.as_dict())
    main()
