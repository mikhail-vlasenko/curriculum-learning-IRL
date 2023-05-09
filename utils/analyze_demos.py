import random

from irl_algos.airl import *
from rl_algos.ppo_from_airl import *
import torch
import numpy as np
import pickle
import gym_examples
from gym_examples.wrappers import RelativePosition
import gymnasium as gym


expert_trajectories = pickle.load(open('../demonstrations/ppo_demos_size15.pk', 'rb'))
print(f'Total number of trajectories: {len(expert_trajectories)}')
for _ in range(10):
    i = random.randint(0, len(expert_trajectories)-1)
    print(len(expert_trajectories[i]['actions']), expert_trajectories[i])

print(f'Averaged length of trajectories: {np.mean([len(traj["actions"]) for traj in expert_trajectories])}')

n = 15
s = 0
cnt = 0
for i in range(n):
    for j in range(n):
        p1 = (i, j)
        for k in range(n):
            for l in range(n):
                p2 = (k, l)
                if p1 != p2:
                    s += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                    cnt += 1

print(f'For {n=}, the ideal average length is {s / cnt}')
