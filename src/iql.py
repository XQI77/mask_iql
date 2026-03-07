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
        # [MODIFIED] === Step 1: 掩码与状态编码 (Representation Learning) ===
        
        # 1.1 对当前状态应用 Mask
        masked_obs, mask = self.generate_mask(observations)
        
        # 1.2 编码得到潜变量 z (用于当前步决策)
        z = self.encoder(masked_obs)
        
        # 1.3 重建原始状态 (Self-Supervised Task)
        recon_obs = self.decoder(z)
        recon_loss = F.mse_loss(recon_obs, observations) # 目标是还原未 Mask 的原始 obs

        # 1.4 对 Next State 编码 (用于计算 Target Value)
        # 策略选择：为了 Target 的稳定性，通常不对 Next State 做 mask，或者只做推理
        with torch.no_grad():
            # 这里的 Encoder 共享参数，但不传导梯度
            z_next = self.encoder(next_observations)

        # [MODIFIED] === Step 2: 更新 Encoder & Decoder ===
        # 我们先更新这一部分，或者加到总 Loss 里。这里选择单独 step 清晰一些。
        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        (self.recon_weight * recon_loss).backward(retain_graph=True) # retain_graph 因为 z 还要给 RL 用
        self.dec_opt.step()
        # 注意：Encoder 的梯度稍后会叠加 RL 的梯度一起更新，或者这里先 step。
        # 为了让 Encoder 同时适配重建和 RL，我们通常把梯度累积。
        # 这里简化处理：Encoder 已经在上面计算了重建梯度，待会 RL 梯度反向传播时会累加到 Encoder 上。
        
        # [MODIFIED] === Step 3: RL Update (使用 z 和 z_next 替代 obs 和 next_obs) ===
        
        with torch.no_grad():
            target_q = self.q_target(z, actions) # 使用 z
            next_v = self.vf(z_next)             # 使用 z_next

        # Update value function
        v = self.vf(z) # 使用 z
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward(retain_graph=True) 
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(z, actions) # 使用 z
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward(retain_graph=True)
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(z) # 使用 z
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward() # 这里的梯度会回传给 Encoder
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()
        
        # 最后统一更新 Encoder (包含了 Recon 梯度 + V梯度 + Q梯度 + Policy梯度)
        self.enc_opt.step() 

        # 返回 Loss 方便 Log
        return {
            'loss/recon': recon_loss.item(),
            'loss/v': v_loss.item(),
            'loss/q': q_loss.item(),
            'loss/policy': policy_loss.item()
        }