import random

import wandb
from tqdm import tqdm
from rl_algos.ppo_from_airl import *
import torch
import pickle
import gym_examples
from gym_examples.wrappers import RelativePosition
import gymnasium as gym


TRAIN = True

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('WARNING: CUDA not available. Using CPU.')


n_demo_steps = 10000

# Initialize Environment
env = gym.make('gym_examples/GridWorld-v0', render_mode=None)
env = RelativePosition(env)

# Fetch Shapes
n_actions = env.action_space.n
obs_shape = env.observation_space.shape

# Load Pretrained PPO
ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions, simple_architecture=True).to(device)

if TRAIN:
    wandb.init(project='PPO', config={
        'env_id': 'randomized_v3',
        'env_steps': 50000,
        'batchsize_ppo': 12,
        'n_workers': 1,
        'lr_ppo': 3e-4,
        'entropy_reg': 0.05,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
    states, info = env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        actions, log_probs = ppo.act(states_tensor)
        action = actions.item()
        log_probs = log_probs.item()
        next_states, rewards, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        # scalarized_rewards = [sum([config.lambd[i] * r[i] for i in range(len(r))]) for r in rewards]

        train_ready = dataset.write_tuple([states], [action], [rewards], [done], [log_probs], logs=[rewards])

        if train_ready:
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            objective_logs = dataset.log_objectives()
            for ret in objective_logs:
                wandb.log({'Reward': ret})
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    torch.save(ppo.state_dict(), 'saved_models/ppo.pt')
    # env.close()
else:
    ppo.load_state_dict(torch.load('saved_models/ppo.pt'))


states, info = env.reset()
states_tensor = torch.tensor(states).float().to(device)
dataset = []
episode = {'states': [], 'actions': []}
episode_cnt = 0


for t in tqdm(range(n_demo_steps)):
    actions, log_probs = ppo.act(states_tensor)
    action = actions.item()
    log_probs = log_probs.item()
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
pickle.dump(dataset, open('demonstrations/ppo_demos' + '.pk', 'wb'))
env.close()
