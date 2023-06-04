import wandb
from tqdm import tqdm

from config import CONFIG
from rl_algos.ppo_from_airl import PPO, TrajectoryDataset, update_policy, device
import torch
from envs.env_factory import make_env


def test_policy(ppo, env, n_episodes):
    states, info = env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    cnt = 0
    reward_sum = 0
    while cnt < n_episodes:
        action, log_probs = ppo.act(states_tensor)
        action = action.item()
        next_states, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward_sum += reward

        if done:
            next_states, info = env.reset()
            cnt += 1

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)

    return reward_sum / n_episodes


def test_policy_wandb_helper(ppo, test_env, step, dataset):
    if test_env is not None:
        test_reward = test_policy(ppo, test_env, n_episodes=CONFIG.ppo.test_episodes)
        wandb.log({'Reward': test_reward,
                   'Current env reward': dataset.log_objectives().mean()}, step=step)
    else:
        wandb.log({'Reward': dataset.log_objectives().mean(),
                   'Current env reward': dataset.log_objectives().mean()}, step=step)


def main(logging_start_step=0, test_env=None):
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
    states_tensor = torch.tensor(state).float().to(device)

    step = 0
    for t in tqdm(range(int(CONFIG.ppo_train.env_steps / CONFIG.ppo.n_workers))):
        action, log_probs = ppo.act(states_tensor)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        train_ready = dataset.write_tuple(state, action, reward, done, log_probs, logs=reward)

        env.substitute_states(next_state)

        if train_ready:
            step = t * CONFIG.ppo.n_workers + logging_start_step
            test_policy_wandb_helper(ppo, test_env, step, dataset)

            update_policy(ppo, dataset, optimizer, CONFIG.ppo.gamma, CONFIG.ppo.epsilon, CONFIG.ppo.update_epochs,
                          entropy_reg=CONFIG.ppo.entropy_reg)
            wandb.log({'Reward': dataset.log_objectives().mean(),
                       'Returns': dataset.log_returns().mean(),
                       'Lengths': dataset.log_lengths().mean()}, step=step)

            dataset.reset_trajectories()

        # Prepare state input for next time step
        state = next_state.copy()
        states_tensor = torch.tensor(state).float().to(device)

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
    wandb.init(project='PPO', config=CONFIG.as_dict(), tags=tags)
    main()
