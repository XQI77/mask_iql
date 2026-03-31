import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .value_functions import StateEncoder, StateDecoder
from .util import DEFAULT_DEVICE, update_exponential_moving_average

EXP_ADV_MAX = 100.

def _grad_norm(model):
    total = 0.
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005,
                 state_dim=None, embedding_dim=256,
                 mask_ratio_min=0.0, mask_ratio_max=0.5,
                 recon_weight=1.0, alpha_consistency=0.5):
        super().__init__()

        self.state_dim        = state_dim
        self.embedding_dim    = embedding_dim
        self.mask_ratio_min   = mask_ratio_min
        self.mask_ratio_max   = mask_ratio_max
        self.recon_weight     = recon_weight
        self.alpha_consistency = alpha_consistency

        self.encoder = StateEncoder(state_dim, embedding_dim).to(DEFAULT_DEVICE)
        self.decoder = StateDecoder(embedding_dim, state_dim).to(DEFAULT_DEVICE)

        self.qf       = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf       = vf.to(DEFAULT_DEVICE)
        self.policy   = policy.to(DEFAULT_DEVICE)

        self.v_optimizer      = optimizer_factory(self.vf.parameters())
        self.q_optimizer      = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.enc_opt          = optimizer_factory(self.encoder.parameters())
        self.dec_opt          = optimizer_factory(self.decoder.parameters())

        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau      = tau
        self.beta     = beta
        self.discount = discount
        self.alpha    = alpha

    def generate_mask(self, state):
        if self.training and self.mask_ratio_max > 0:
            if self.mask_ratio_min == self.mask_ratio_max:
                p = self.mask_ratio_min
            else:
                p = torch.empty(1).uniform_(self.mask_ratio_min, self.mask_ratio_max).item()
            return torch.bernoulli(torch.ones_like(state) * (1 - p))
        return torch.ones_like(state)

    def update(self, observations, actions, next_observations, rewards, terminals, **kwargs):
        """
        联合训练：每步同时优化自监督表征损失（MAE）和 IQL 强化学习损失。

        梯度流向设计：
          - Encoder 接收来自：自监督 loss（recon + consistency）+ Q loss（Bellman 信号）
          - V / Policy 更新时对 z 做 detach，梯度不回传 encoder
          - next_obs 对应的 z_next 始终 detach，避免 bootstrap 不稳定
        """
        full_mask = torch.ones_like(observations)

        # ── 0. 提前清零 encoder / decoder 梯度 ──────────────────────────────
        #    必须在 Q backward 之前执行，确保 Q→encoder 梯度得以保留
        self.enc_opt.zero_grad(set_to_none=True)
        self.dec_opt.zero_grad(set_to_none=True)

        # ── 1. 自监督前向（masked path，梯度图保留，暂不 backward）──────────
        mask      = self.generate_mask(observations)
        z_masked  = self.encoder(observations, mask)       # in graph
        recon_obs = self.decoder(z_masked)
        recon_loss = F.mse_loss(recon_obs, observations)

        with torch.no_grad():                              # stop-gradient 一致性目标
            z_full = self.encoder(observations, full_mask)
        consistency_loss = F.mse_loss(z_masked, z_full)
        self_sup_loss = self.recon_weight * recon_loss + self.alpha_consistency * consistency_loss

        # ── 2. RL 前向 ───────────────────────────────────────────────────────
        # z：不 detach，Q loss 的梯度将通过 z 回传给 encoder
        z = self.encoder(observations, full_mask)

        with torch.no_grad():
            z_next   = self.encoder(next_observations, full_mask)  # next detach
            next_v   = self.vf(z_next)
            target_q = self.q_target(z.detach(), actions)          # frozen q_target

        # ── 3. 更新 V（z detach，梯度不回传 encoder）────────────────────────
        v   = self.vf(z.detach())
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vf.parameters(), max_norm=10.0)
        v_grad_norm = _grad_norm(self.vf)
        self.v_optimizer.step()

        # ── 4. 更新 Q（z 不 detach → Bellman 梯度回传 encoder）─────────────
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs      = self.qf.both(z, actions)
        q_loss  = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()        # encoder 积累来自 Q 的梯度
        torch.nn.utils.clip_grad_norm_(self.qf.parameters(), max_norm=10.0)
        q_grad_norm = _grad_norm(self.qf)
        self.q_optimizer.step()

        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # ── 5. 更新 Policy（z detach，梯度不回传 encoder）───────────────────
        exp_adv    = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(z.detach())

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

        # ── 6. 更新 Encoder + Decoder ────────────────────────────────────────
        # encoder 当前已积累 step4 的 Q 梯度，再叠加自监督梯度
        self_sup_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=5.0)
        enc_grad_norm = _grad_norm(self.encoder)
        self.enc_opt.step()
        self.dec_opt.step()

        # ── 7. 诊断指标 ──────────────────────────────────────────────────────
        with torch.no_grad():
            q1, q2   = qs[0].detach(), qs[1].detach()
            td_error = (q1 - targets.detach()).abs()

            masked_dims   = (1 - mask)
            observed_dims = mask
            per_dim_se    = (recon_obs.detach() - observations) ** 2
            masked_mse    = (per_dim_se * masked_dims).sum()   / masked_dims.sum().clamp(min=1)
            observed_mse  = (per_dim_se * observed_dims).sum() / observed_dims.sum().clamp(min=1)
            cosine_sim    = F.cosine_similarity(z_masked.detach(), z_full, dim=-1).mean()
            mean_mask_ratio = masked_dims.mean()

            policy_diag = {}
            if isinstance(policy_out, torch.distributions.Distribution):
                policy_diag['policy/log_std_mean']   = self.policy.log_std.data.mean().item()
                policy_diag['policy/log_std_min']    = self.policy.log_std.data.min().item()
                policy_diag['policy/log_std_max']    = self.policy.log_std.data.max().item()
                policy_diag['policy/entropy']        = policy_out.entropy().mean().item()
                policy_diag['policy/action_mean_abs']= policy_out.mean.abs().mean().item()
            elif torch.is_tensor(policy_out):
                policy_diag['policy/action_mean_abs']= policy_out.detach().abs().mean().item()
                policy_diag['policy/action_std']     = policy_out.detach().std(dim=0).mean().item()

        diagnostics = {
            # 自监督损失
            'loss/recon':              recon_loss.item(),
            'loss/consistency':        consistency_loss.item(),
            'encoder/masked_mse':      masked_mse.item(),
            'encoder/observed_mse':    observed_mse.item(),
            'encoder/cosine_sim':      cosine_sim.item(),
            'encoder/mask_ratio':      mean_mask_ratio.item(),
            # RL 损失
            'loss/v':                  v_loss.item(),
            'loss/q':                  q_loss.item(),
            'loss/policy':             policy_loss.item(),
            # Q 值分布
            'q/q1_mean':               q1.mean().item(),
            'q/q1_std':                q1.std().item(),
            'q/q2_mean':               q2.mean().item(),
            'q/target_q_mean':         target_q.mean().item(),
            'q/target_q_std':          target_q.std().item(),
            # V 值分布
            'v/mean':                  v.detach().mean().item(),
            'v/std':                   v.detach().std().item(),
            'v/next_v_mean':           next_v.mean().item(),
            # 优势函数
            'adv/mean':                adv.detach().mean().item(),
            'adv/std':                 adv.detach().std().item(),
            'adv/min':                 adv.detach().min().item(),
            'adv/max':                 adv.detach().max().item(),
            'adv/frac_positive':       (adv.detach() > 0).float().mean().item(),
            # exp_adv 截断率
            'exp_adv/mean':            exp_adv.mean().item(),
            'exp_adv/max':             exp_adv.max().item(),
            'exp_adv/frac_clipped':    (exp_adv >= EXP_ADV_MAX).float().mean().item(),
            # TD-error
            'td/error_mean':           td_error.mean().item(),
            'td/error_max':            td_error.max().item(),
            # 梯度范数
            'grad/enc_norm':           enc_grad_norm,
            'grad/v_norm':             v_grad_norm,
            'grad/q_norm':             q_grad_norm,
            'grad/policy_norm':        policy_grad_norm,
            # Embedding 统计
            'embed/z_mean':            z.detach().mean().item(),
            'embed/z_std':             z.detach().std().item(),
            'embed/z_norm':            z.detach().norm(dim=-1).mean().item(),
            # Reward 统计
            'data/reward_mean':        rewards.mean().item(),
            'data/reward_std':         rewards.std().item(),
        }
        diagnostics.update(policy_diag)
        return diagnostics
