import gym
import d4rl
import torch
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

# 引入你的模型定义
from src.value_functions import StateEncoder, StateDecoder, TwinQ, ValueFunction
from src.policy import GaussianPolicy, DeterministicPolicy
from src.util import torchify

# 简单的 Wrapper 来复现训练时的结构
class AgentWrapper(torch.nn.Module):
    def __init__(self, state_dim, action_dim, embedding_dim, hidden_dim=256):
        super().__init__()
        self.encoder = StateEncoder(state_dim, embedding_dim, hidden_dim)
        # 这里的 Policy 初始化要和你训练时 main.py 的逻辑一致
        self.policy = GaussianPolicy(embedding_dim, action_dim, hidden_dim) 

    def act(self, obs, deterministic=True):
        with torch.no_grad():
            z = self.encoder(obs)
            return self.policy.act(z, deterministic=deterministic)

def evaluate_with_attack(env, agent, mask_rate, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_ret = 0
        while not done:
            # === 核心攻击逻辑 ===
            # 在测试时，随机把 mask_rate 比例的传感器置 0
            # 模拟真实世界的传感器故障
            if mask_rate > 0:
                mask = np.random.binomial(1, 1 - mask_rate, size=obs.shape)
                obs_input = obs * mask
            else:
                obs_input = obs
            # ===================

            # 模型基于残缺输入决策
            action = agent.act(torchify(obs_input), deterministic=True).cpu().numpy()
            
            # 环境基于真实物理响应
            obs, reward, done, _ = env.step(action)
            total_ret += reward
        returns.append(total_ret)
    
    return np.mean(returns), np.std(returns)

def load_agent(model_path, env_name, embedding_dim=256):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # 初始化网络结构
    agent = AgentWrapper(obs_dim, act_dim, embedding_dim).to('cuda')
    
    # 加载权重
    # 注意：因为 iql 保存的是整个 iql 对象的 state_dict
    # 我们需要手动把 encoder 和 policy 的权重提取出来
    checkpoint = torch.load(model_path)
    
    # 提取 Encoder 权重
    enc_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if k.startswith('encoder.')}
    agent.encoder.load_state_dict(enc_dict)
    
    # 提取 Policy 权重
    pol_dict = {k.replace('policy.', ''): v for k, v in checkpoint.items() if k.startswith('policy.')}
    agent.policy.load_state_dict(pol_dict)
    
    return env, agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--model_path', type=str, required=True, help='Path to final.pt')
    parser.add_argument('--label', type=str, required=True, help='Label for the plot (e.g., Baseline or Ours)')
    args = parser.parse_args()

    env, agent = load_agent(args.model_path, args.env)
    
    # 测试不同程度的破坏
    attack_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    print(f"Testing model: {args.label}")
    for rate in attack_rates:
        mean_ret, std_ret = evaluate_with_attack(env, agent, rate)
        norm_score = d4rl.get_normalized_score(args.env, mean_ret) * 100
        print(f"Attack Rate: {rate} | Score: {norm_score:.2f}")
        results.append({'Attack Rate': rate, 'Score': norm_score, 'Model': args.label})
    
    # 保存结果到 CSV
    df = pd.DataFrame(results)
    df.to_csv(f'results_{args.label}.csv', index=False)

if __name__ == '__main__':
    main()