import io

from tqdm import tqdm

from airl_gridworld_train import init_models
from config import *
from envs.env_factory import make_env
from gym_examples.wrappers.vec_env import VecEnv
from irl_algos.airl import *
from rl_algos.ppo_from_airl import *
import torch
import pickle
import torch.onnx


def main():
    env: VecEnv = make_env()
    state, info = env.reset()
    state_tensor = torch.tensor(state).float().to(device)
    ppo, discriminator, optimizer, optimizer_discriminator, dataset = init_models(env)

    action, log_probs = ppo.act(state_tensor)
    next_state, reward, done, _, info = env.step(action)
    next_state_tensor = torch.tensor(next_state).to(device).float()

    torch.onnx.export(discriminator,  # model being run
                      # state_tensor,  # model input (or a tuple for multiple inputs)
                      (state_tensor, next_state_tensor, CONFIG.ppo.gamma),
                      '../saved_models/model.onnx',  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['state', 'next_state', 'gamma'],  # the model's input names
                      output_names=['advantage'],  # the model's output names
                      dynamic_axes={'state': {0: 'batch_size'},  # variable length axes
                                    'next_state': {0: 'batch_size'},
                                    'advantage': {0: 'batch_size'}})


if __name__ == '__main__':
    main()
