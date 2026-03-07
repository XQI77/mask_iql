from pathlib import Path

import gym
import d4rl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy, NormalizedEnv


def get_env_and_dataset(log, env_name, max_episode_steps):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    # === [关键修改] 计算均值和标准差 ===
    obs = dataset['observations']
    # keepdims=True 保持形状为 (1, obs_dim)，方便广播
    obs_mean = obs.mean(axis=0, keepdims=True)
    obs_std = obs.std(axis=0, keepdims=True) + 1e-3  # 加上 1e-3 防止除以 0
    
    # === [关键修改] 对数据集进行归一化 ===
    dataset['observations'] = (dataset['observations'] - obs_mean) / obs_std
    dataset['next_observations'] = (dataset['next_observations'] - obs_mean) / obs_std
    
    # 打印统计信息，确认归一化生效
    log(f'State Normalization Applied.')
    log(f'Mean (first 3 dims): {obs_mean[0][:3]}')
    log(f'Std  (first 3 dims): {obs_std[0][:3]}')
    
    # 处理 Reward (保持原有逻辑)
    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    # 转为 Tensor
    for k, v in dataset.items():
        dataset[k] = torchify(v)

    # 返回 obs_mean 和 obs_std 供后续使用
    # 注意：squeeze() 去掉多余的维度，变回 (obs_dim,)
    return env, dataset, obs_mean.squeeze(), obs_std.squeeze()

# [MODIFIED] 新增一个评估用的 Wrapper，把 Encoder 和 Policy 包装在一起
class MaskedPolicyWrapper(torch.nn.Module):
    def __init__(self, encoder, policy):
        super().__init__()
        self.encoder = encoder
        self.policy = policy
    
    def act(self, obs, deterministic=False, enable_grad=False):
        # 1. 编码
        z = self.encoder(obs)
        # 2. 决策
        return self.policy.act(z, deterministic, enable_grad)

def main(args):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir)/args.env_name, vars(args))
    log(f'Log dir: {log.dir}')

    # [修改 1] 接收 mean 和 std
    env, dataset, obs_mean, obs_std = get_env_and_dataset(log, args.env_name, args.max_episode_steps)
    
    # [修改 2] 包装环境，确保 evaluate_policy 看到的是归一化后的状态
    # 注意：env.seed() 应该在 wrap 之前调用，或者 wrap 之后确保传递
    env = NormalizedEnv(env, mean=obs_mean, std=obs_std)

    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    set_seed(args.seed, env=env)

    # [MODIFIED] 1. 定义 Latent Dimension
    embedding_dim = args.embedding_dim 

    if args.deterministic_policy:
        policy = DeterministicPolicy(embedding_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    else:
        policy = GaussianPolicy(embedding_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
        
     # [MODIFIED] 3. 初始化 IQL agent
    iql = ImplicitQLearning(
        qf=TwinQ(embedding_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(embedding_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount,
        # 传入新参数
        state_dim=obs_dim,
        embedding_dim=embedding_dim,
        mask_prob=args.mask_prob,
        recon_weight=args.recon_weight
    )

    # 我们需要构建一个包含 Encoder 的 Policy Wrapper 给 evaluate_policy 使用
    eval_agent = MaskedPolicyWrapper(iql.encoder, policy)
    
    def eval_policy():
        # 这里进行标准评估 (Mask Rate = 0)
        # 如果你想做 Attack 实验，可以修改 evaluate_policy 函数支持传入 mask
        eval_returns = np.array([evaluate_policy(env, eval_agent, args.max_episode_steps) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        
        # 获取 iql update 返回的 loss 字典（需要稍微改一下 train loop 获取 loss）
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })

    # === [NEW] Pre-training Phase ===
    print("Starting Encoder Pre-training...")
    # 预训练步数，通常 50k 足够
    n_pretrain = 100000 
    
    # 确保 IQL 处于训练模式 (这样 generate_mask 才会生效)
    iql.train()
    
    for _ in trange(n_pretrain, desc="Pre-training"):
        # 1. 采样数据
        batch = sample_batch(dataset, args.batch_size)
        obs = batch['observations']
        
        # 2. 生成掩码并编码
        # 注意：generate_mask 需要在 iql.py 中被定义为 public 方法
        masked_obs, mask = iql.generate_mask(obs)
        
        # 3. 前向传播
        z = iql.encoder(masked_obs)
        recon = iql.decoder(z)
        
        # 4. 计算 Loss (使用 F.mse_loss)
        loss = F.mse_loss(recon, obs)
        
        # 5. 反向传播更新
        # 确保使用了 iql 对象中定义的优化器
        iql.enc_opt.zero_grad()
        iql.dec_opt.zero_grad()
        loss.backward()
        iql.enc_opt.step()
        iql.dec_opt.step()
        
    print("Pre-training Finished. Starting RL Training...")
    # ================================

    for step in trange(args.n_steps):
        # [MODIFIED] 获取 loss 并 log（可选）
        loss_dict = iql.update(**sample_batch(dataset, args.batch_size))
        
        if (step+1) % args.eval_period == 0:
            # 可以顺便打印一下 recon loss 看看有没有下降
            print(f"Step {step}: Recon Loss = {loss_dict['loss/recon']:.6f}")
            eval_policy()

    save_dict = {
        'model_state': iql.state_dict(),
        'obs_mean': obs_mean,
        'obs_std': obs_std
    }
    torch.save(save_dict, log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)

    # [MODIFIED] 新增参数
    parser.add_argument('--embedding-dim', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--mask-prob', type=float, default=0.3, help='Probability of masking a sensor')
    parser.add_argument('--recon-weight', type=float, default=1.0, help='Weight for reconstruction loss')
    main(parser.parse_args())