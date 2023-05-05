import gym_examples
from gym_examples.wrappers import RelativePosition
import gymnasium as gym


env = gym.make('gym_examples/GridWorld-v0', render_mode=None)
env = RelativePosition(env)
obs, info = env.reset()
print(obs)
obs, reward, terminated, truncated, info = env.step(0)
print(obs)
env.step(1)
env.step(2)
env.step(3)
env.close()
