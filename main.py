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
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class NormalizedEnv(gym.Wrapper):
    def __init__(self, env, mean, std):
        super().__init__(env)
        self.mean = mean.cpu().numpy() if torch.is_tensor(mean) else mean
        self.std  = std.cpu().numpy()  if torch.is_tensor(std)  else std
        self.mean = self.mean.reshape(-1)
        self.std  = self.std.reshape(-1)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return (obs - self.mean) / self.std

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return (next_obs - self.mean) / self.std, reward, done, info


def get_env_and_dataset(log, env_name, max_episode_steps):
    env     = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    obs         = dataset['observations']
    obs_mean    = obs.mean(axis=0, keepdims=True)
    obs_std     = obs.std(axis=0,  keepdims=True) + 1e-3

    dataset['observations']      = (dataset['observations']      - obs_mean) / obs_std
    dataset['next_observations'] = (dataset['next_observations'] - obs_mean) / obs_std

    log(f'State Normalization Applied.')
    log(f'Mean (first 3 dims): {obs_mean[0][:3]}')
    log(f'Std  (first 3 dims): {obs_std[0][:3]}')

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset, obs_mean.squeeze(), obs_std.squeeze()


# 评估用 Wrapper：单帧 obs → encoder → policy（与训练完全一致）
class MaskedPolicyWrapper(torch.nn.Module):
    def __init__(self, encoder, policy):
        super().__init__()
        self.encoder = encoder
        self.policy  = policy

    def act(self, obs, deterministic=False, enable_grad=False, mask=None):
        if mask is None:
            mask = torch.ones_like(obs)
        z = self.encoder(obs, mask)
        return self.policy.act(z, deterministic, enable_grad)


def main(args):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir)/args.env_name, vars(args))
    log(f'Log dir: {log.dir}')

    use_wandb = args.use_wandb and HAS_WANDB
    if args.use_wandb and not HAS_WANDB:
        log('WARNING: --use-wandb specified but wandb not installed. pip install wandb')
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f'{args.env_name}_seed{args.seed}',
            config=vars(args),
            dir=str(log.dir),
        )
        log(f'W&B run: {wandb.run.url}')

    env, dataset, obs_mean, obs_std = get_env_and_dataset(log, args.env_name, args.max_episode_steps)
    env = NormalizedEnv(env, mean=obs_mean, std=obs_std)

    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    set_seed(args.seed, env=env)

    embedding_dim = args.embedding_dim

    if args.deterministic_policy:
        policy = DeterministicPolicy(embedding_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    else:
        policy = GaussianPolicy(embedding_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)

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
        state_dim=obs_dim,
        embedding_dim=embedding_dim,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        recon_weight=args.recon_weight,
        alpha_consistency=args.alpha_consistency,
    )

    eval_agent = MaskedPolicyWrapper(iql.encoder, policy)
    iql.train()

    for step in trange(args.n_steps):
        loss_dict = iql.update(**sample_batch(dataset, args.batch_size))

        if use_wandb:
            wandb.log(loss_dict, step=step)

        if (step + 1) % args.eval_period == 0:
            print(
                f"Step {step}: "
                f"Recon={loss_dict['loss/recon']:.4f} | "
                f"Consist={loss_dict['loss/consistency']:.4f} | "
                f"V={loss_dict['loss/v']:.4f} | "
                f"Q={loss_dict['loss/q']:.4f} | "
                f"Policy={loss_dict['loss/policy']:.4f} | "
                f"adv_mean={loss_dict['adv/mean']:.4f} | "
                f"adv_frac+={loss_dict['adv/frac_positive']:.2f} | "
                f"enc_grad={loss_dict['grad/enc_norm']:.4f}"
            )

            eval_returns = np.array([
                evaluate_policy(env, eval_agent, args.max_episode_steps)
                for _ in range(args.n_eval_episodes)
            ])
            normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
            eval_dict = {
                'step': step,
                'return mean':            eval_returns.mean(),
                'return std':             eval_returns.std(),
                'normalized return mean': normalized_returns.mean(),
                'normalized return std':  normalized_returns.std(),
            }
            log.row(eval_dict)

            if use_wandb:
                wandb.log({
                    'eval/return_mean':            eval_returns.mean(),
                    'eval/return_std':             eval_returns.std(),
                    'eval/normalized_return_mean': normalized_returns.mean(),
                    'eval/normalized_return_std':  normalized_returns.std(),
                }, step=step)

    torch.save({
        'model_state': iql.state_dict(),
        'obs_mean':    obs_mean,
        'obs_std':     obs_std,
    }, log.dir/'final.pt')
    log.close()
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name',           required=True)
    parser.add_argument('--log-dir',            required=True)
    parser.add_argument('--seed',               type=int,   default=0)
    parser.add_argument('--discount',           type=float, default=0.99)
    parser.add_argument('--hidden-dim',         type=int,   default=256)
    parser.add_argument('--n-hidden',           type=int,   default=2)
    parser.add_argument('--n-steps',            type=int,   default=10**6)
    parser.add_argument('--batch-size',         type=int,   default=256)
    parser.add_argument('--learning-rate',      type=float, default=3e-4)
    parser.add_argument('--alpha',              type=float, default=0.005)
    parser.add_argument('--tau',                type=float, default=0.7)
    parser.add_argument('--beta',               type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period',        type=int,   default=5000)
    parser.add_argument('--n-eval-episodes',    type=int,   default=10)
    parser.add_argument('--max-episode-steps',  type=int,   default=1000)
    parser.add_argument('--embedding-dim',      type=int,   default=256)
    parser.add_argument('--mask-ratio-min',     type=float, default=0.0)
    parser.add_argument('--mask-ratio-max',     type=float, default=0.5)
    parser.add_argument('--recon-weight',       type=float, default=1.0)
    parser.add_argument('--alpha-consistency',  type=float, default=0.5)
    parser.add_argument('--use-wandb',          action='store_true')
    parser.add_argument('--wandb-project',      type=str,   default='masked-iql')
    main(parser.parse_args())
