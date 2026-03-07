import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# [MODIFIED] 引入新定义的 Encoder/Decoder (假设都在 value_functions 里)
from .value_functions import StateEncoder, StateDecoder 
from .util import DEFAULT_DEVICE, update_exponential_moving_average

EXP_ADV_MAX = 100.

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005,
                 # [MODIFIED] 新增参数
                 state_dim=None, embedding_dim=256, 
                 mask_prob=0.3, recon_weight=1.0):
        super().__init__()
        
        # [MODIFIED] 1. 初始化 Encoder 和 Decoder
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.mask_prob = mask_prob
        self.recon_weight = recon_weight
        
        # 注意：这里假设外部传入的 qf, vf, policy 已经将其 input_dim 设置为 embedding_dim 了
        self.encoder = StateEncoder(state_dim, embedding_dim).to(DEFAULT_DEVICE)
        self.decoder = StateDecoder(embedding_dim, state_dim).to(DEFAULT_DEVICE)
        
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)

        # [MODIFIED] 2. 优化器包含 Encoder/Decoder
        # Encoder 随 RL 一起训练，Decoder 单独有重建 Loss
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        
        # Encoder 和 Decoder 的优化器
        self.enc_opt = optimizer_factory(self.encoder.parameters())
        self.dec_opt = optimizer_factory(self.decoder.parameters())
        
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    # [MODIFIED] 新增：掩码生成函数
    def generate_mask(self, state):
        if self.training and self.mask_prob > 0:
            probs = torch.full_like(state, 1 - self.mask_prob)
            mask = torch.bernoulli(probs)
            return state * mask, mask
        return state, torch.ones_like(state)

    def update(self, observations, actions, next_observations, rewards, terminals):
        """
        Masked-IQL Update Logic
        Phase 1: Representation Learning (Reconstruction)
        Phase 2: Reinforcement Learning (IQL with Fixed Features)
        """
        
        # =======================================================
        # Phase 1: Representation Learning (Encoder & Decoder)
        # =======================================================
        
        # 1. 生成掩码并构造残缺状态
        # Generate mask and masked state
        masked_obs, mask = self.generate_mask(observations)
        
        # 2. 编码 (Encode)
        # z: [batch, embedding_dim]
        z = self.encoder(masked_obs)
        
        # 3. 解码与重建 (Decode & Reconstruct)
        # 试图从 z 还原回完整的原始 observations
        recon_obs = self.decoder(z)
        
        # 4. 计算重建 Loss
        # 这里只计算 MSE，迫使 Encoder 学会利用传感器间的相关性补全信息
        recon_loss = F.mse_loss(recon_obs, observations)

        # 5. 更新 Encoder 和 Decoder
        # 注意：我们在这里直接更新 Encoder，而不是等到后面。
        # 这样 Encoder 的梯度完全由重建任务主导，不受 RL 噪音干扰。
        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        (recon_loss * self.recon_weight).backward()
        self.enc_opt.step()
        self.dec_opt.step()

        # =======================================================
        # Phase 2: Reinforcement Learning (IQL)
        # =======================================================

        # [关键修改] Detach z!
        # 我们希望 RL 策略去适应 Encoder 提取的特征，而不是去改变特征。
        # 这样可以极大稳定训练过程，防止 "Feature Collapse"。
        z_detached = z.detach()

        # 预计算 Target Value 所需的 next_z
        with torch.no_grad():
            # 为了提供稳定的 TD Target，Next State 通常不进行 Mask
            z_next = self.encoder(next_observations) 
            target_q = self.q_target(z_detached, actions)
            next_v = self.vf(z_next)

        # --- Update Value Function (V) ---
        v = self.vf(z_detached)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # --- Update Q Function (Critic) ---
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(z_detached, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # --- Update Target Q Network ---
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # --- Update Policy (Actor) ---
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
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
            'loss/recon': recon_loss.item(),
            'loss/v': v_loss.item(),
            'loss/q': q_loss.item(),
            'loss/policy': policy_loss.item(),
            'value/mean': v.mean().item(),
            'value/adv_mean': adv.mean().item()
        }