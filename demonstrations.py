import numpy as np


def generate_demonstrations(env, agent, include_rew_done=True, num_demos=100):
    demos = []
    for i in range(num_demos):
        obs, info = env.reset()
        demo = []
        while True:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if include_rew_done:
                demo.append((obs, action, next_obs, reward, done))
            else:
                demo.append((obs[0] * 5 + obs[1], action, next_obs[0] * 5 + next_obs[1]))
            obs = next_obs
            if done:
                break
        demos.append(demo)
    return demos
