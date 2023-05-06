import random

from irl_algos.airl import *
from rl_algos.ppo_from_airl import *
import torch
import numpy as np
import pickle
import gym_examples
from gym_examples.wrappers import RelativePosition
import gymnasium as gym


expert_trajectories = pickle.load(open('demonstrations/ppo_demos.pk', 'rb'))

for _ in range(10):
    i = random.randint(0, len(expert_trajectories)-1)
    print(len(expert_trajectories[i]['actions']), expert_trajectories[i])
