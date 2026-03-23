import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# [MODIFIED] 引入新定义的 Encoder/Decoder/WindowAggregator
from .value_functions import StateEncoder, StateDecoder, WindowAggregator
from .util import DEFAULT_DEVICE, update_exponential_moving_average

EXP_ADV_MAX = 100.

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005,
                 # [MODIFIED] 新增参数
                 state_dim=None, embedding_dim=256,
                 mask_ratio_min=0.0, mask_ratio_max=0.5,
                 recon_weight=1.0, alpha_consistency=0.5,
                 window_size=5, agg_hidden_dim=256):
        super().__init__()

        # [MODIFIED] 1. 初始化 Encoder、Decoder 和 WindowAggregator
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.recon_weight = recon_weight
        self.alpha_consistency = alpha_consistency
        self.window_size = window_size

        # 注意：这里假设外部传入的 qf, vf, policy 已经将其 input_dim 设置为 embedding_dim 了
        self.encoder     = StateEncoder(state_dim, embedding_dim).to(DEFAULT_DEVICE)
        self.decoder     = StateDecoder(embedding_dim, state_dim).to(DEFAULT_DEVICE)
        self.window_agg  = WindowAggregator(window_size, embedding_dim, agg_hidden_dim).to(DEFAULT_DEVICE)

        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)

        # [MODIFIED] 2. 优化器：enc_opt 同时优化 Encoder 和 WindowAggregator
        self.v_optimizer      = optimizer_factory(self.vf.parameters())
        self.q_optimizer      = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())

        self.enc_opt = optimizer_factory(
            list(self.encoder.parameters()) + list(self.window_agg.parameters())
        )
        self.dec_opt = optimizer_factory(self.decoder.parameters())
        
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    # [MODIFIED] 掩码生成函数：只返回 mask，掩码填充逻辑在 Encoder 内部处理
    def generate_mask(self, state):
        if self.training and self.mask_ratio_max > 0:
            # per-batch 随机采样掩码比例 p ~ U(mask_ratio_min, mask_ratio_max)
            # 当 min == max 时退化为固定比例
            if self.mask_ratio_min == self.mask_ratio_max:
                p = self.mask_ratio_min
            else:
                p = torch.empty(1).uniform_(self.mask_ratio_min, self.mask_ratio_max).item()
            mask = torch.bernoulli(torch.ones_like(state) * (1 - p))
            return mask
        return torch.ones_like(state)

    def update(self, observations, actions, next_observations, rewards, terminals,
               window_observations=None, window_valid=None):
        """
        Masked-IQL Update Logic (with sliding-window history)
        Phase 1: Representation Learning (Reconstruction + Consistency)
        Phase 2: Reinforcement Learning (IQL with Fixed Features)

        window_observations : (batch, K, state_dim)
        window_valid        : (batch, K)  – 1.0 = real step, 0.0 = padded
        """
        K = self.window_size

        # =======================================================
        # Phase 1: Representation Learning (Encoder & Decoder)
        # =======================================================

        # --- Per-step encoding over the window (masked path) ---
        z_list = []
        for k in range(K):
            obs_k   = window_observations[:, k, :]            # (batch, state_dim)
            valid_k = window_valid[:, k].unsqueeze(1)         # (batch, 1)
            # Random mask; padded steps forced to zero mask (all dims missing)
            mask_k  = self.generate_mask(obs_k) * valid_k
            z_list.append(self.encoder(obs_k, mask_k))

        z_t      = z_list[-1]                                 # current-step z (for recon)
        z_window = torch.stack(z_list, dim=1)                 # (batch, K, emb_dim)
        z_final  = self.window_agg(z_window)                  # (batch, emb_dim)

        # --- Consistency target: full-state window (no gradient) ---
        with torch.no_grad():
            z_full_list = []
            for k in range(K):
                obs_k    = window_observations[:, k, :]
                valid_k  = window_valid[:, k].unsqueeze(1)
                full_mask_k = torch.ones_like(obs_k) * valid_k   # 1 for real, 0 for padded
                z_full_list.append(self.encoder(obs_k, full_mask_k))
            z_full_window = torch.stack(z_full_list, dim=1)
            z_full_agg    = self.window_agg(z_full_window)        # (batch, emb_dim)

        consistency_loss = F.mse_loss(z_final, z_full_agg)

        # --- Reconstruction: current-step z_t → decoder → current obs ---
        recon_obs  = self.decoder(z_t)
        recon_loss = F.mse_loss(recon_obs, observations)

        repr_loss = recon_loss * self.recon_weight + self.alpha_consistency * consistency_loss

        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        repr_loss.backward()
        self.enc_opt.step()
        self.dec_opt.step()

        # =======================================================
        # Phase 2: Reinforcement Learning (IQL)
        # =======================================================

        # RL uses z_final (window-aggregated); detach to prevent feature collapse
        z_detached = z_final.detach()

        with torch.no_grad():
            # Next state: 单步编码后构造伪窗口，保证特征分布与 z_final 一致
            z_next_single = self.encoder(next_observations, torch.ones_like(next_observations))
            # 关键修复：将单步特征复制 K 次作为窗口，让 z_next 也过 window_agg
            z_next_window = z_next_single.unsqueeze(1).expand(-1, self.window_size, -1)  # (B, K, emb)
            z_next_agg    = self.window_agg(z_next_window)  # (B, emb)
            target_q = self.q_target(z_detached, actions)
            next_v   = self.vf(z_next_agg)

        # --- Update Value Function (V) ---
        v   = self.vf(z_detached)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # --- Update Q Function (Critic) ---
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs      = self.qf.both(z_detached, actions)
        q_loss  = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # --- Update Target Q Network ---
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # --- Update Policy (Actor) ---
        exp_adv    = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(z_detached)

        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError(f"Policy output shape {policy_out.shape} != actions shape {actions.shape}")
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
        else:
            raise NotImplementedError

        policy_loss = torch.mean(exp_adv * bc_losses)

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        return {
            'loss/recon':       recon_loss.item(),
            'loss/consistency': consistency_loss.item(),
            'loss/v':           v_loss.item(),
            'loss/q':           q_loss.item(),
            'loss/policy':      policy_loss.item(),
            'value/mean':       v.mean().item(),
            'value/adv_mean':   adv.mean().item()
        }