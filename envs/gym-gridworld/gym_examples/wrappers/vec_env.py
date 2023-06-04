from typing import List, Dict

import numpy as np
import torch


class VecEnv:
    def __init__(self, env_list, tensor_state: bool, device=None):
        self.env_list = env_list
        self.tensor_state = tensor_state
        self.device = device
        self.n_envs = len(env_list)
        self.action_space = self.env_list[0].action_space
        self.observation_space = self.env_list[0].observation_space
        self.dones = np.full(self.n_envs, False)

    def reset(self) -> (np.ndarray[np.ndarray], List[Dict]):
        obs_list = []
        info_list = []
        for i in range(self.n_envs):
            obs, info = self.env_list[i].reset()
            obs_list.append(obs)
            info_list.append(info)

        return np.stack(obs_list, axis=0), info_list

    def step(self, actions) -> \
            (np.ndarray[np.ndarray], np.ndarray[float], np.ndarray[bool], np.ndarray[bool], List[Dict]):
        """
        Does not automatically reset the environment when done.
        :param actions:
        :return:
        """
        obs_list = []
        rew_list = []
        done_list = []
        info_list = []
        for i in range(self.n_envs):
            assert not self.dones[i], f'Env {i} is done, but step() was called.'
            obs_i, rew_i, terminated, truncated, info_i = self.env_list[i].step(actions[i])
            done_i = terminated | truncated
            self.dones[i] = done_i

            obs_list.append(obs_i)
            rew_list.append(rew_i)
            done_list.append(done_i)
            info_list.append(info_i)

        states = np.stack(obs_list, axis=0)
        if self.tensor_state:
            states = torch.tensor(states).float().to(self.device)
        return states, np.array(rew_list), \
            np.array(done_list), np.full(self.n_envs, False), info_list

    def substitute_states(self, states):
        """
        Modifies the state array in-place with observations from resetting the envs that are done.
        :param states:
        :return:
        """
        for i in range(self.n_envs):
            if self.dones[i]:
                states[i], _ = self.env_list[i].reset()
                self.dones[i] = False

    def close(self):
        for env in self.env_list:
            env.close()
