import random

import wandb
from tqdm import tqdm

from config import CONFIG
from rl_algos.ppo_from_airl import *
import torch
import pickle
from envs.env_factory import make_env


# Initialize Environment
CONFIG.env.vectorized = False
CONFIG.env.render = True
env = make_env()

# Fetch Shapes
n_actions = env.action_space.n
obs_shape = env.observation_space.shape

# Load Pretrained PPO
ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions).to(device)

ppo.load_state_dict(torch.load('../' + CONFIG.ppo_train.load_from))

states, info = env.reset()
states_tensor = torch.tensor(states).float().to(device)

for t in range(1000):
    action, log_probs = ppo.act(states_tensor)
    action = action.item()
    next_states, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    if done:
        next_states, info = env.reset()

    # Prepare state input for next time step
    states = next_states.copy()
    states_tensor = torch.tensor(states).float().to(device)

