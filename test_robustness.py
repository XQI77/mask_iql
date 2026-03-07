import gym
import d4rl
import torch
import numpy as np
import argparse
import pandas as pd
from src.value_functions import StateEncoder, StateDecoder, TwinQ, ValueFunction
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

# Agent Wrapper
class AgentWrapper(torch.nn.Module):
    def __init__(self, state_dim, action_dim, embedding_dim, hidden_dim=256):
        super().__init__()
        self.encoder = StateEncoder(state_dim, embedding_dim, hidden_dim)
        self.policy = GaussianPolicy(embedding_dim, action_dim, hidden_dim) 

    def act(self, obs, deterministic=True):
        with torch.no_grad():
            z = self.encoder(obs)
            return self.policy.act(z, deterministic=deterministic)

def evaluate_with_attack(env, agent, mask_rate, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs = env.reset() # 这里的 obs 已经是归一化之后的了
        done = False
        total_ret = 0
        while not done:
            # === 攻击逻辑 ===
            # 我们在归一化特征的基础上进行 Mask
            # 这符合训练时的逻辑：Train: Normalize -> Mask -> Encoder
            if mask_rate > 0:
                mask = np.random.binomial(1, 1 - mask_rate, size=obs.shape)
                obs_input = obs * mask
            else:
                obs_input = obs
            
            # Agent 接收的是归一化且残缺的状态
            action = agent.act(torchify(obs_input), deterministic=True).cpu().numpy()
            
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
    
    # === [核心修改] 处理新的保存格式 ===
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        print(">> Found normalization stats! Wrapping environment.")
        state_dict = checkpoint['model_state']
        obs_mean = checkpoint['obs_mean']
        obs_std = checkpoint['obs_std']
        
        # 包装环境
        env = NormalizedEnv(env, obs_mean, obs_std)
    else:
        print(">> WARNING: No normalization stats found (Old Model?). Testing with RAW environment.")
        state_dict = checkpoint

    # 加载权重
    agent.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')})
    agent.policy.load_state_dict({k.replace('policy.', ''): v for k, v in state_dict.items() if k.startswith('policy.')})
    
    return env, agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--embedding-dim', type=int, default=256)
    args = parser.parse_args()

    # 加载环境和模型
    env, agent = load_agent_and_env(args.model_path, args.env, args.embedding_dim)
    
    attack_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    print(f"Testing model: {args.label}")
    for rate in attack_rates:
        score = evaluate_with_attack(env, agent, rate)
        norm_score = d4rl.get_normalized_score(args.env, score) * 100
        print(f"  [Attack {rate*100:.0f}%] Score: {norm_score:.2f}")
        results.append({'Attack Rate': rate, 'Score': norm_score, 'Model': args.label})
    
    # 保存结果
    df = pd.DataFrame(results)
    output_filename = f'results_{args.label}.csv'
    df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}")

if __name__ == '__main__':
    main()