import gym_examples
import gymnasium as gym


env = gym.make('gym_examples/GridWorld-v0', render_mode='human')
env.reset()
env.step(0)
env.step(1)
env.step(2)
env.step(3)
env.step(4)
env.close()
