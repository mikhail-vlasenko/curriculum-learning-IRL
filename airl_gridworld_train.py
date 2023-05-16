import wandb

from tqdm import tqdm

from config import *
from envs.env_factory import make_env
from irl_algos.airl import *
from rl_algos.ppo_from_airl import *
import torch
import pickle


def main(logging_start_step=0):
    CONFIG.ppo.entropy_reg = 0.0

    print(f'Using data from {CONFIG.airl.expert_data_path}')
    expert_trajectories = pickle.load(open(CONFIG.airl.expert_data_path, 'rb'))

    # Create Environment
    env = make_env()

    state, info = env.reset()
    state_tensor = torch.tensor(state).float().to(device)

    # Fetch Shapes
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    # Initialize Models
    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions).to(device)
    discriminator = DiscriminatorMLP(state_shape=obs_shape[0]).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=CONFIG.ppo.lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=CONFIG.discriminator.lr)
    dataset = TrajectoryDataset(batch_size=CONFIG.ppo.batch_size, n_workers=CONFIG.ppo.n_workers)

    if CONFIG.airl.disc_load_from is not None:
        discriminator.load_state_dict(torch.load(CONFIG.airl.disc_load_from))

    if CONFIG.airl.ppo_load_from is not None:
        ppo.load_state_dict(torch.load(CONFIG.airl.ppo_load_from))

    for t in tqdm(range((int(CONFIG.airl.env_steps / CONFIG.ppo.n_workers)))):
        action, log_probs = ppo.act(state_tensor)
        next_state, reward, done, _, info = env.step(action)
        next_state_tensor = torch.tensor(next_state).to(device).float()

        # Calculate (vectorized) AIRL reward
        airl_action_prob = torch.exp(torch.tensor(log_probs)).to(device).float()
        airl_rewards = discriminator.predict_reward(
            state_tensor, next_state_tensor, CONFIG.ppo.gamma, airl_action_prob
        )
        airl_rewards = airl_rewards.detach().cpu().numpy()
        airl_rewards[done] = 0

        # Save Trajectory
        train_ready = dataset.write_tuple(state, action, airl_rewards, done, log_probs, logs=reward)

        if train_ready:
            wandb.log({'Reward': dataset.log_objectives().mean(),
                       'Returns': dataset.log_returns().mean(),
                       'Lengths': dataset.log_lengths().mean()}, step=t * CONFIG.ppo.n_workers + logging_start_step)

            # Update Models
            update_policy(
                ppo, dataset, optimizer,
                CONFIG.ppo.gamma, CONFIG.ppo.epsilon, CONFIG.ppo.update_epochs,
                entropy_reg=CONFIG.ppo.entropy_reg
            )
            d_loss, fake_acc, real_acc = update_discriminator(
                discriminator=discriminator,
                optimizer=optimizer_discriminator,
                gamma=CONFIG.ppo.gamma,
                expert_trajectories=expert_trajectories,
                policy_trajectories=dataset.trajectories,
                ppo=ppo,
                batch_size=CONFIG.discriminator.batch_size
            )

            wandb.log({'Discriminator Loss': d_loss,
                       'Fake Accuracy': fake_acc,
                       'Real Accuracy': real_acc}, step=t * CONFIG.ppo.n_workers + logging_start_step)

            torch.save(discriminator.state_dict(), DISC_CHECKPOINT)
            torch.save(ppo.state_dict(), PPO_CHECKPOINT)

            dataset.reset_trajectories()

        # Prepare state input for next time step
        state = next_state
        state_tensor = next_state_tensor

    torch.save(discriminator.state_dict(), CONFIG.airl.disc_save_to)
    torch.save(ppo.state_dict(), CONFIG.airl.ppo_save_to)

    # save model artifacts to wandb
    model_art = wandb.Artifact('airl_models', type='model')
    model_art.add_file(CONFIG.airl.disc_save_to)
    model_art.add_file(CONFIG.airl.ppo_save_to)
    wandb.log_artifact(model_art)

    data_art = wandb.Artifact('airl_data', type='dataset')
    data_art.add_file(CONFIG.airl.expert_data_path)
    wandb.log_artifact(data_art)

    env.close()


if __name__ == '__main__':
    set_experiment_config(grid_size=15, max_steps=-1)
    if CONFIG.airl.expert_data_path is None:
        CONFIG.airl.expert_data_path = get_demo_name()
    tags = ['single_dataset']
    if CONFIG.airl.disc_load_from is not None or CONFIG.airl.ppo_load_from is not None:
        tags.append('continued_training')
    wandb.init(project='AIRL', dir='wandb', config=CONFIG.as_dict(), tags=tags)
    main()
