import random

import wandb
from tqdm import tqdm

from config import CONFIG
from rl_algos.ppo_from_airl import *
import torch
import pickle
from env_factory import make_env


# Use GPU if available
CONFIG.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('WARNING: CUDA not available. Using CPU.')

# Initialize Environment
env = make_env()

# Fetch Shapes
n_actions = env.action_space.n
obs_shape = env.observation_space.shape

# Load Pretrained PPO
ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions, simple_architecture=True).to(device)

if CONFIG.ppo_train.do_train:
    wandb.init(project='PPO', config=CONFIG.as_dict())

    if CONFIG.ppo_train.load_from is not None:
        ppo.load_state_dict(torch.load(CONFIG.ppo_train.load_from))

    optimizer = torch.optim.Adam(ppo.parameters(), lr=CONFIG.ppo.lr)
    dataset = TrajectoryDataset(batch_size=CONFIG.ppo.batch_size, n_workers=CONFIG.ppo.n_workers)

    states, info = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    for t in tqdm(range(int(CONFIG.ppo_train.env_steps / CONFIG.ppo.n_workers))):
        action, log_probs = ppo.act(states_tensor)
        next_states, rewards, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        # scalarized_rewards = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]

        train_ready = dataset.write_tuple(states, action, rewards, done, log_probs, logs=rewards)

        if train_ready:
            update_policy(ppo, dataset, optimizer, CONFIG.ppo.gamma, CONFIG.ppo.epsilon, CONFIG.ppo.update_epochs,
                          entropy_reg=CONFIG.ppo.entropy_reg)
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
    ppo.load_state_dict(torch.load(CONFIG.ppo_train.load_from))


CONFIG.env.vectorized = False
env = make_env()

states, info = env.reset()
states_tensor = torch.tensor(states).float().to(device)
dataset = []
episode = {'states': [], 'actions': []}
episode_cnt = 0


for t in tqdm(range(CONFIG.demos.n_steps)):
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
pickle.dump(dataset, open(f'demonstrations/ppo_demos_size{CONFIG.env.grid_size}.pk', 'wb'))
env.close()
