import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# [MODIFIED] 引入新定义的 Encoder/Decoder/WindowAggregator
from .value_functions import StateEncoder, StateDecoder, WindowAggregator
from .util import DEFAULT_DEVICE, update_exponential_moving_average

EXP_ADV_MAX = 100.

#计算模型中所有参数梯度的 L2 范数
def _grad_norm(model):
    total = 0.
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5

# 非对称 L2 损失
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

        # EMA target networks for encoder and window_agg
        self.encoder_ema    = copy.deepcopy(self.encoder).requires_grad_(False).to(DEFAULT_DEVICE)
        self.window_agg_ema = copy.deepcopy(self.window_agg).requires_grad_(False).to(DEFAULT_DEVICE)

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

    def _freeze_representation(self):
        """Phase 1 结束后冻结 encoder、decoder、window_agg 的参数"""
        for module in [self.encoder, self.decoder, self.window_agg]:
            for param in module.parameters():
                param.requires_grad_(False)
        # 同步 EMA 到冻结后的 encoder/window_agg 最终状态
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.window_agg_ema.load_state_dict(self.window_agg.state_dict())

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
               window_observations=None, window_valid=None,
               next_window_observations=None, next_window_valid=None,
               warmup=False):
        """
        Masked-IQL Update Logic (with sliding-window history)
        Phase 1: Representation Learning (Reconstruction + Consistency)
        Phase 2: Reinforcement Learning (IQL with Fixed Features)

        window_observations : (batch, K, state_dim)
        window_valid        : (batch, K)  – 1.0 = real step, 0.0 = padded
        """
        K = self.window_size

        # =======================================================
        # Phase 1: Representation Learning (仅在 warmup 阶段更新)
        # =======================================================

        if warmup:
            # --- Per-step encoding over the window (masked path) ---
            z_list = []
            for k in range(K):
                obs_k   = window_observations[:, k, :]            # (batch, state_dim)
                valid_k = window_valid[:, k].unsqueeze(1)         # (batch, 1)
                mask_k  = self.generate_mask(obs_k) * valid_k
                z_list.append(self.encoder(obs_k, mask_k))

            z_t      = z_list[-1]                                 # current-step z (for recon)
            z_window = torch.stack(z_list, dim=1)                 # (batch, K, emb_dim)
            z_final  = self.window_agg(z_window)                  # (batch, emb_dim)

            # --- 方案C: 一致性目标使用在线 encoder + stop-gradient（而非 EMA）---
            with torch.no_grad():
                z_full_list = []
                for k in range(K):
                    obs_k    = window_observations[:, k, :]
                    valid_k  = window_valid[:, k].unsqueeze(1)
                    full_mask_k = torch.ones_like(obs_k) * valid_k
                    z_full_list.append(self.encoder(obs_k, full_mask_k))
                z_full_window = torch.stack(z_full_list, dim=1)
                z_full_agg    = self.window_agg(z_full_window)

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

            # EMA update for encoder and window_agg
            update_exponential_moving_average(self.encoder_ema, self.encoder, self.alpha)
            update_exponential_moving_average(self.window_agg_ema, self.window_agg, self.alpha)

            # --- Encoder 评估指标（不参与梯度计算）---
            with torch.no_grad():
                # 当前步的 mask（最后一步窗口的 mask）
                last_mask = self.generate_mask(observations) * window_valid[:, -1].unsqueeze(1)
                per_dim_se = (recon_obs.detach() - observations) ** 2  # (batch, state_dim)

                masked_dims = (1 - last_mask)   # 1=被遮盖的维度
                observed_dims = last_mask        # 1=可观测的维度

                # 仅在被遮盖维度上的 MSE
                masked_count = masked_dims.sum()
                masked_mse = (per_dim_se * masked_dims).sum() / masked_count.clamp(min=1)

                # 仅在可观测维度上的 MSE
                observed_count = observed_dims.sum()
                observed_mse = (per_dim_se * observed_dims).sum() / observed_count.clamp(min=1)

                # masked embedding 与 full embedding 的余弦相似度
                cosine_sim = F.cosine_similarity(z_final, z_full_agg, dim=-1).mean()

                # 实际平均 mask 比例（被遮盖维度占比）
                mean_mask_ratio = masked_dims.mean()

            return {
                'loss/recon': recon_loss.item(),
                'loss/consistency': consistency_loss.item(),
                'encoder/masked_mse': masked_mse.item(),
                'encoder/observed_mse': observed_mse.item(),
                'encoder/cosine_sim': cosine_sim.item(),
                'encoder/mask_ratio': mean_mask_ratio.item(),
            }

        # =======================================================
        # Phase 2: Reinforcement Learning (IQL, encoder 已冻结)
        # =======================================================

        # 使用冻结的在线 encoder（参数不再更新，等价于固定特征提取器）
        with torch.no_grad():
            # current state: 单帧，full mask
            full_mask   = torch.ones_like(observations)
            z_for_rl    = self.encoder(observations, full_mask)

            # next state: 单帧，full mask
            full_mask_next = torch.ones_like(next_observations)
            z_next_agg     = self.encoder(next_observations, full_mask_next)

            target_q = self.q_target(z_for_rl, actions)
            next_v   = self.vf(z_next_agg)

        # --- Update Value Function (V) ---
        v   = self.vf(z_for_rl)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vf.parameters(), max_norm=10.0)
        v_grad_norm = _grad_norm(self.vf)
        self.v_optimizer.step()

        # --- Update Q Function (Critic) ---
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs      = self.qf.both(z_for_rl, actions)
        q_loss  = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qf.parameters(), max_norm=10.0)
        q_grad_norm = _grad_norm(self.qf)
        self.q_optimizer.step()

        # --- Update Target Q Network ---
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # --- Update Policy (Actor) ---
        exp_adv    = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(z_for_rl)

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
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        policy_grad_norm = _grad_norm(self.policy)
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        # --- 诊断指标（不参与梯度计算）---
        with torch.no_grad():
            q1, q2 = qs[0].detach(), qs[1].detach()
            td_error = (q1 - targets.detach()).abs()

            # Policy 分布诊断
            policy_diag = {}
            if isinstance(policy_out, torch.distributions.Distribution):
                policy_diag['policy/log_std_mean'] = self.policy.log_std.data.mean().item()
                policy_diag['policy/log_std_min'] = self.policy.log_std.data.min().item()
                policy_diag['policy/log_std_max'] = self.policy.log_std.data.max().item()
                policy_diag['policy/entropy'] = policy_out.entropy().mean().item()
                policy_diag['policy/action_mean_abs'] = policy_out.mean.abs().mean().item()
            elif torch.is_tensor(policy_out):
                policy_diag['policy/action_mean_abs'] = policy_out.detach().abs().mean().item()
                policy_diag['policy/action_std'] = policy_out.detach().std(dim=0).mean().item()

        diagnostics = {
            # 损失
            'loss/v':               v_loss.item(),
            'loss/q':               q_loss.item(),
            'loss/policy':          policy_loss.item(),
            # Q 值分布
            'q/q1_mean':            q1.mean().item(),
            'q/q1_std':             q1.std().item(),
            'q/q2_mean':            q2.mean().item(),
            'q/target_q_mean':      target_q.mean().item(),
            'q/target_q_std':       target_q.std().item(),
            # V 值分布
            'v/mean':               v.detach().mean().item(),
            'v/std':                v.detach().std().item(),
            'v/next_v_mean':        next_v.mean().item(),
            # 优势函数（关键诊断）
            'adv/mean':             adv.detach().mean().item(),
            'adv/std':              adv.detach().std().item(),
            'adv/min':              adv.detach().min().item(),
            'adv/max':              adv.detach().max().item(),
            'adv/frac_positive':    (adv.detach() > 0).float().mean().item(),
            # exp_adv 截断率（关键诊断）
            'exp_adv/mean':         exp_adv.mean().item(),
            'exp_adv/max':          exp_adv.max().item(),
            'exp_adv/frac_clipped': (exp_adv >= EXP_ADV_MAX).float().mean().item(),
            # TD-error
            'td/error_mean':        td_error.mean().item(),
            'td/error_max':         td_error.max().item(),
            # 梯度范数
            'grad/v_norm':          v_grad_norm,
            'grad/q_norm':          q_grad_norm,
            'grad/policy_norm':     policy_grad_norm,
            # Embedding 统计
            'embed/z_mean':         z_for_rl.mean().item(),
            'embed/z_std':          z_for_rl.std().item(),
            'embed/z_norm':         z_for_rl.norm(dim=-1).mean().item(),
            # Reward 统计（当前 batch）
            'data/reward_mean':     rewards.mean().item(),
            'data/reward_std':      rewards.std().item(),
        }
        diagnostics.update(policy_diag)
        return diagnostics