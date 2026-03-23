import csv
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x



def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}


def build_window_dataset(dataset, window_size):
    """
    Pre-compute sliding window observations from the dataset.
    Respects trajectory boundaries (terminals / timeouts); out-of-bound steps
    are zero-padded with a corresponding zero validity mask.

    Adds two keys to the dataset in-place and returns it:
      'window_observations': (N, K, state_dim) – per-window raw observations
      'window_valid':        (N, K)            – 1.0 = real step, 0.0 = padded
    """
    observations = dataset['observations']          # (N, state_dim)
    N, state_dim = observations.shape
    device = observations.device

    # Trajectory boundaries: terminal=1 OR timeout=1 ends a trajectory
    is_boundary = dataset['terminals'].bool()
    if 'timeouts' in dataset:
        is_boundary = is_boundary | dataset['timeouts'].bool()

    # Indices where new trajectories begin (index 0 always starts one)
    boundary_indices = torch.where(is_boundary)[0]
    traj_starts = torch.cat([
        torch.tensor([0], dtype=torch.long, device=device),
        (boundary_indices + 1).long()
    ])
    traj_starts = traj_starts[traj_starts < N]      # guard against last-step terminal

    # For each step i: which trajectory does it belong to?
    # traj_id[i] = largest j s.t. traj_starts[j] <= i  (via searchsorted)
    all_idx_f = torch.arange(N, dtype=torch.float32, device=device)
    traj_id = torch.searchsorted(
        traj_starts.float().contiguous(), all_idx_f.contiguous(), right=True
    ) - 1
    traj_id = traj_id.long().clamp(min=0)

    # Position of step i within its trajectory
    traj_pos = torch.arange(N, device=device) - traj_starts[traj_id]

    # Build window tensors
    window_obs   = torch.zeros(N, window_size, state_dim,
                               dtype=observations.dtype, device=device)
    window_valid = torch.zeros(N, window_size,
                               dtype=observations.dtype, device=device)

    for k in range(window_size):
        # window[:, k, :] = observation at step  t - (K-1-k)
        # k=0 → oldest (t-K+1),  k=K-1 → current (t)
        offset = window_size - 1 - k
        valid = (traj_pos >= offset)                          # (N,) bool
        src_idx = (torch.arange(N, device=device) - offset).clamp(min=0)
        window_obs[:, k, :] = observations[src_idx]
        window_obs[~valid, k, :] = 0.0                       # zero-pad invalid steps
        window_valid[:, k] = valid.float()

    dataset['window_observations'] = window_obs    # (N, K, state_dim)
    dataset['window_valid']        = window_valid  # (N, K)
    return dataset


# 修改 evaluate_policy 以支持测试时攻击
def evaluate_policy(env, policy, max_episode_steps, deterministic=True, attack_prob=0.0):
    # 重置历史 buffer（支持带窗口的 policy wrapper）
    if hasattr(policy, 'reset'):
        policy.reset()
    obs = env.reset()
    total_reward = 0.
    for _ in range(max_episode_steps):
        if attack_prob > 0:
            mask = np.random.binomial(1, 1 - attack_prob, size=obs.shape)
        else:
            mask = np.ones_like(obs)

        # 关键修复：传入原始 obs（不在外部置零）和 mask，由 encoder 内部用 mask_token 填充
        with torch.no_grad():
            action = policy.act(
                torchify(obs),
                deterministic=deterministic,
                mask=torchify(mask.astype(np.float32))
            ).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward


def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()
