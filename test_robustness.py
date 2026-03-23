import gym
import d4rl
import torch
import numpy as np
import argparse
import pandas as pd
from collections import deque
from src.value_functions import StateEncoder, WindowAggregator, StateDecoder, TwinQ, ValueFunction
from src.policy import GaussianPolicy, DeterministicPolicy
from src.util import torchify

# === 必须重新定义 Wrapper，或者从 main.py import ===
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

# Agent Wrapper：与训练时的 MaskedPolicyWrapper 保持一致
class AgentWrapper(torch.nn.Module):
    def __init__(self, state_dim, action_dim, embedding_dim, window_size, hidden_dim=256, agg_hidden_dim=256):
        super().__init__()
        self.encoder     = StateEncoder(state_dim, embedding_dim, hidden_dim)
        self.window_agg  = WindowAggregator(window_size, embedding_dim, agg_hidden_dim)
        self.policy      = GaussianPolicy(embedding_dim, action_dim, hidden_dim)
        self.window_size = window_size
        self._buffer     = []  # list of (obs_tensor, mask_tensor)

    def reset(self):
        """每局开始时清空历史 buffer"""
        self._buffer = []

    def act(self, obs, mask, deterministic=True):
        """
        Args:
            obs:  (state_dim,) – 归一化后的单步观测 tensor（已经过 mask_token 处理前的原始值）
            mask: (state_dim,) – 1=可观测, 0=缺失
        """
        with torch.no_grad():
            self._buffer.append((obs, mask))
            if len(self._buffer) > self.window_size:
                self._buffer = self._buffer[-self.window_size:]

            state_dim = obs.shape[0]
            device    = obs.device

            # 构造窗口：不足 K 步时用零填充（零掩码 = 全部缺失）
            z_list = []
            for i in range(self.window_size):
                buf_idx = i - (self.window_size - len(self._buffer))
                if buf_idx < 0:
                    obs_k  = torch.zeros(state_dim, device=device)
                    mask_k = torch.zeros(state_dim, device=device)
                else:
                    obs_k, mask_k = self._buffer[buf_idx]
                z_list.append(self.encoder(obs_k, mask_k))

            # 聚合：stack 成 (K, emb_dim)，再过 window_agg 得到 (emb_dim,)
            z_window = torch.stack(z_list, dim=0)   # (K, emb_dim)
            z_final  = self.window_agg(z_window)     # (emb_dim,)

            return self.policy.act(z_final, deterministic=deterministic)


def evaluate_with_attack(env, agent, mask_rate, n_episodes=10, missing_mode='dynamic'):
    """
    Args:
        env: 环境（已归一化）
        agent: AgentWrapper 实例
        mask_rate: 缺失率 ρ (0.0 到 1.0)
        n_episodes: 评估 episode 数量
        missing_mode: 'dynamic'（每步独立掩码）或 'factor_reduction'（每集固定掩码）
    """
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()  # 归一化后的观测
        agent.reset()
        done = False
        total_ret = 0

        # Factor Reduction: episode 开头生成一次掩码，整集复用
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

            # 将 obs 和 mask 转为 tensor，由 encoder 内部用 mask_token 填充缺失维度
            action = agent.act(
                torchify(obs),
                torchify(mask.astype(np.float32)),
                deterministic=True
            ).cpu().numpy()

            obs, reward, done, _ = env.step(action)
            total_ret += reward
        returns.append(total_ret)
    return np.mean(returns)


def load_agent_and_env(model_path, env_name, embedding_dim=256, window_size=5, agg_hidden_dim=256):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = AgentWrapper(obs_dim, act_dim, embedding_dim, window_size, agg_hidden_dim=agg_hidden_dim).to('cuda')

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

    # 加载权重（state_dict 来自 iql.state_dict()，含 encoder.*, window_agg.*, policy.* 等前缀）
    agent.encoder.load_state_dict(
        {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    )
    agent.window_agg.load_state_dict(
        {k.replace('window_agg.', ''): v for k, v in state_dict.items() if k.startswith('window_agg.')}
    )
    agent.policy.load_state_dict(
        {k.replace('policy.', ''): v for k, v in state_dict.items() if k.startswith('policy.')}
    )

    return env, agent


def run_full_evaluation(env, agent, args):
    """一次性跑完两种场景的所有缺失率"""
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
    parser.add_argument('--window-size', type=int, default=5,
                        help='Sliding window size K (must match training)')
    parser.add_argument('--agg-hidden-dim', type=int, default=256,
                        help='Hidden dim of window aggregator MLP')
    parser.add_argument('--missing_mode', type=str, default='dynamic',
                        choices=['dynamic', 'factor_reduction'],
                        help='Missing scenario: dynamic or factor_reduction')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of evaluation episodes per attack rate')
    parser.add_argument('--full_eval', action='store_true',
                        help='Run both dynamic and factor_reduction across all attack rates')
    args = parser.parse_args()

    # 加载环境和模型
    env, agent = load_agent_and_env(
        args.model_path, args.env, args.embedding_dim,
        args.window_size, args.agg_hidden_dim
    )

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

    # 保存结果
    df = pd.DataFrame(results)
    output_filename = f'results_{args.label}_{args.missing_mode}.csv'
    df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}")


if __name__ == '__main__':
    main()
