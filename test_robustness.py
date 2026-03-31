import gym
import d4rl
import torch
import numpy as np
import argparse
import pandas as pd
from src.value_functions import StateEncoder, TwinQ, ValueFunction
from src.policy import GaussianPolicy, DeterministicPolicy
from src.util import torchify


class NormalizedEnv(gym.Wrapper):
    def __init__(self, env, mean, std):
        super().__init__(env)
        self.mean = mean
        self.std = std

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return (obs - self.mean) / self.std

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        norm_next_obs = (next_obs - self.mean) / self.std
        return norm_next_obs, reward, done, info


# 与训练时 MaskedPolicyWrapper 完全一致：单帧 obs → encoder → policy
class AgentWrapper(torch.nn.Module):
    def __init__(self, state_dim, action_dim, embedding_dim, hidden_dim=256):
        super().__init__()
        self.encoder = StateEncoder(state_dim, embedding_dim, hidden_dim)
        self.policy  = GaussianPolicy(embedding_dim, action_dim, hidden_dim)

    def act(self, obs, mask, deterministic=True):
        with torch.no_grad():
            z = self.encoder(obs, mask)
            return self.policy.act(z, deterministic=deterministic)


def evaluate_with_attack(env, agent, mask_rate, n_episodes=10, missing_mode='dynamic'):
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_ret = 0

        if missing_mode == 'factor_reduction' and mask_rate > 0:
            episode_mask = np.random.binomial(1, 1 - mask_rate, size=obs.shape)

        while not done:
            if mask_rate > 0:
                if missing_mode == 'dynamic':
                    mask = np.random.binomial(1, 1 - mask_rate, size=obs.shape)
                elif missing_mode == 'factor_reduction':
                    mask = episode_mask
                else:
                    raise ValueError(f"Unknown missing_mode: {missing_mode}")
            else:
                mask = np.ones_like(obs)

            action = agent.act(
                torchify(obs),
                torchify(mask.astype(np.float32)),
                deterministic=True
            ).cpu().numpy()

            obs, reward, done, _ = env.step(action)
            total_ret += reward
        returns.append(total_ret)
    return np.mean(returns)


def load_agent_and_env(model_path, env_name, embedding_dim=256):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = AgentWrapper(obs_dim, act_dim, embedding_dim).to('cuda')

    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        print(">> Found normalization stats! Wrapping environment.")
        state_dict = checkpoint['model_state']
        obs_mean = checkpoint['obs_mean']
        obs_std  = checkpoint['obs_std']
        env = NormalizedEnv(env, obs_mean, obs_std)
    else:
        print(">> WARNING: No normalization stats found (Old Model?). Testing with RAW environment.")
        state_dict = checkpoint

    agent.encoder.load_state_dict(
        {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    )
    agent.policy.load_state_dict(
        {k.replace('policy.', ''): v for k, v in state_dict.items() if k.startswith('policy.')}
    )

    return env, agent


def run_full_evaluation(env, agent, args):
    all_results = []
    attack_rates = [0.0, 0.1, 0.3, 0.5, 0.7]

    for mode in ['dynamic', 'factor_reduction']:
        print(f"\n{'='*50}")
        print(f"Testing: {mode} scenario")
        print(f"{'='*50}")
        for rate in attack_rates:
            score = evaluate_with_attack(env, agent, rate,
                                         n_episodes=args.n_episodes,
                                         missing_mode=mode)
            norm_score = d4rl.get_normalized_score(args.env, score) * 100
            print(f"  [{mode}] [Missing {rate*100:.0f}%] Score: {norm_score:.2f}")
            all_results.append({
                'Attack Rate': rate,
                'Score': norm_score,
                'Model': args.label,
                'Missing Mode': mode
            })

    df = pd.DataFrame(all_results)
    output_filename = f'results_{args.label}_full.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nSaved all results to {output_filename}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--missing_mode', type=str, default='dynamic',
                        choices=['dynamic', 'factor_reduction'])
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--full_eval', action='store_true')
    args = parser.parse_args()

    env, agent = load_agent_and_env(args.model_path, args.env, args.embedding_dim)

    if args.full_eval:
        run_full_evaluation(env, agent, args)
        return

    attack_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
    results = []

    print(f"Testing model: {args.label}  [mode: {args.missing_mode}]")
    for rate in attack_rates:
        score = evaluate_with_attack(env, agent, rate,
                                     n_episodes=args.n_episodes,
                                     missing_mode=args.missing_mode)
        norm_score = d4rl.get_normalized_score(args.env, score) * 100
        print(f"  [{args.missing_mode}] [Missing {rate*100:.0f}%] Score: {norm_score:.2f}")
        results.append({
            'Attack Rate': rate,
            'Score': norm_score,
            'Model': args.label,
            'Missing Mode': args.missing_mode
        })

    df = pd.DataFrame(results)
    output_filename = f'results_{args.label}_{args.missing_mode}.csv'
    df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}")


if __name__ == '__main__':
    main()
