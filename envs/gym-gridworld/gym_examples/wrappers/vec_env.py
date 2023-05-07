import numpy as np


class VecEnv:
    def __init__(self, env_list):
        self.env_list = env_list
        self.n_envs = len(env_list)
        self.action_space = self.env_list[0].action_space
        self.observation_space = self.env_list[0].observation_space

    def reset(self):
        obs_list = []
        info_list = []
        for i in range(self.n_envs):
            obs, info = self.env_list[i].reset()
            obs_list.append(obs)
            info_list.append(info)

        return np.stack(obs_list, axis=0), info_list

    def step(self, actions):
        """
        Automatically resets the environment if done
        :param actions:
        :return:
        """
        obs_list = []
        rew_list = []
        done_list = []
        info_list = []
        for i in range(self.n_envs):
            obs_i, rew_i, terminated, truncated, info_i = self.env_list[i].step(actions[i])
            done_i = terminated | truncated

            if done_i:
                obs_i, info_i = self.env_list[i].reset()

            obs_list.append(obs_i)
            rew_list.append(rew_i)
            done_list.append(done_i)
            info_list.append(info_i)

        return np.stack(obs_list, axis=0), np.array(rew_list), np.array(done_list), np.full(self.n_envs, False), info_list

    def close(self):
        for env in self.env_list:
            env.close()
