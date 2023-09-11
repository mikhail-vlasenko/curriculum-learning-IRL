import wandb
from tqdm import tqdm

from config import CONFIG
from rl_algos.ppo_from_airl import PPO, TrajectoryDataset, update_policy, device
import torch
from envs.env_factory import make_env


def test_policy(ppo, env):
    dataset = TrajectoryDataset(batch_size=env.env_list[0].max_steps * 100, n_workers=CONFIG.ppo.n_workers)

    state, info = env.reset()
    state_tensor = torch.tensor(state).float().to(device)
    ready = False
    dummy_state = torch.empty(CONFIG.ppo.n_workers)
    while not ready:
        action, log_probs = ppo.act(state_tensor)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        env.substitute_states(state)
        state_tensor = torch.tensor(state).float().to(device)

        ready = dataset.write_tuple(dummy_state, action, reward, done, log_probs, logs=reward)

    return dataset.log_objectives().mean()


def test_policy_wandb_helper(ppo, test_env, step, dataset):
    if test_env is not None:
        test_reward = test_policy(ppo, test_env)
        wandb.log({'Reward': test_reward,
                   'Current env reward': dataset.log_objectives().mean()}, step=step)
    else:
        wandb.log({'Reward': dataset.log_objectives().mean(),
                   'Current env reward': dataset.log_objectives().mean()}, step=step)


def main(logging_start_step=0, test_env=None):
    """
    The main training loop for PPO.
    Initializes the environment and the PPO.
    Trains the policy on the environment specified in the config.
    Saves the model weights.
    :param logging_start_step: The step at which to start logging (for continued training)
    :param test_env: Environment to test on.
    """
    env = make_env()

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    ppo = PPO(state_shape=obs_shape[0], n_actions=n_actions).to(device)

    if CONFIG.ppo_train.load_from is not None:
        print('Loading PPO from: ', CONFIG.ppo_train.load_from)
        ppo.load_state_dict(torch.load(CONFIG.ppo_train.load_from))

    optimizer = torch.optim.Adam(ppo.parameters(), lr=CONFIG.ppo.lr)
    dataset = TrajectoryDataset(batch_size=CONFIG.ppo.batch_size, n_workers=CONFIG.ppo.n_workers)

    state, info = env.reset()
    state_tensor = torch.tensor(state).float().to(device)

    step = 0
    for t in tqdm(range(int(CONFIG.ppo_train.env_steps / CONFIG.ppo.n_workers))):
        action, log_probs = ppo.act(state_tensor)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        train_ready = dataset.write_tuple(state, action, reward, done, log_probs, logs=reward)

        env.substitute_states(next_state)

        if train_ready:
            step = t * CONFIG.ppo.n_workers + logging_start_step
            test_policy_wandb_helper(ppo, test_env, step, dataset)
            wandb.log({'Returns': dataset.log_returns().mean(),
                       'Lengths': dataset.log_lengths().mean()}, step=step)

            update_policy(ppo, dataset, optimizer, CONFIG.ppo.gamma, CONFIG.ppo.epsilon, CONFIG.ppo.update_epochs,
                          entropy_reg=CONFIG.ppo.entropy_reg)

            dataset.reset_trajectories()

        # Prepare state input for next time step
        state = next_state.copy()
        state_tensor = torch.tensor(state).float().to(device)

    torch.save(ppo.state_dict(), CONFIG.ppo_train.save_to)
    print(f'Saved PPO to: {CONFIG.ppo_train.save_to}')
    model_art = wandb.Artifact('expert_model', type='model')
    model_art.add_file(CONFIG.ppo_train.save_to)
    wandb.log_artifact(model_art)

    env.close()
    return step


if __name__ == '__main__':
    tags = []
    if CONFIG.ppo_train.load_from is not None:
        tags.append('continued_training')
    if 'OnlyEndReward' in CONFIG.env.wrappers:
        tags.append('only_end_reward')
    if 'FlattenObs' in CONFIG.env.wrappers:
        tags.append('flatten_obs')
    wandb.init(project='PPO-minigrid', config=CONFIG.as_dict(), tags=tags)
    main()
