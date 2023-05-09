import random

import wandb
from tqdm import tqdm

from config import Config, to_dict
from rl_algos.ppo_from_airl import *
import torch
import pickle
from env_factory import make_env


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('WARNING: CUDA not available. Using CPU.')

# Initialize Environment
env = make_env(Config)

# Fetch Shapes
n_actions = env.action_space.n
obs_shape = env.observation_space.shape

# Load Pretrained PPO
ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions, simple_architecture=True).to(device)

if Config.ppo_train.do_train:
    wandb.init(project='PPO', config=to_dict(Config))

    if Config.ppo_train.load_from is not None:
        ppo.load_state_dict(torch.load(Config.ppo_train.load_from))

    optimizer = torch.optim.Adam(ppo.parameters(), lr=Config.ppo.lr)
    dataset = TrajectoryDataset(batch_size=Config.ppo.batch_size, n_workers=Config.ppo.n_workers)

    states, info = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    for t in tqdm(range(int(Config.ppo_train.env_steps / Config.ppo.n_workers))):
        action, log_probs = ppo.act(states_tensor)
        next_states, rewards, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        # scalarized_rewards = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]

        train_ready = dataset.write_tuple(states, action, rewards, done, log_probs, logs=rewards)

        if train_ready:
            update_policy(ppo, dataset, optimizer, Config.ppo.gamma, Config.ppo.epsilon, Config.ppo.update_epochs,
                          entropy_reg=Config.ppo.entropy_reg)
            wandb.log({'Reward': dataset.log_objectives().mean()})
            wandb.log({'Returns': dataset.log_returns().mean()})
            wandb.log({'Lengths': dataset.log_lengths().mean()})

            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    torch.save(ppo.state_dict(), 'saved_models/ppo_expert.pt')
    env.close()
else:
    ppo.load_state_dict(torch.load(Config.ppo_train.load_from))


Config.env.vectorized = False
env = make_env(Config)

states, info = env.reset()
states_tensor = torch.tensor(states).float().to(device)
dataset = []
episode = {'states': [], 'actions': []}
episode_cnt = 0


for t in tqdm(range(Config.demos.n_steps)):
    action, log_probs = ppo.act(states_tensor)
    action = action.item()
    next_states, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode['states'].append(states)
    # Note: Actions currently append as arrays and not integers!
    episode['actions'].append(action)

    if done:
        next_states, info = env.reset()
        dataset.append(episode)
        episode = {'states': [], 'actions': []}

    # Prepare state input for next time step
    states = next_states.copy()
    states_tensor = torch.tensor(states).float().to(device)

print('Sample:')
for _ in range(2):
    print(dataset[random.randint(0, len(dataset) - 1)])
pickle.dump(dataset, open(f'demonstrations/ppo_demos_size{Config.env.grid_size}.pk', 'wb'))
env.close()
