import torch
import torch.nn as nn
from .util import mlp


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)
    
#Masked-IQL Components
class StateEncoder(nn.Module):
    def __init__(self, state_dim, embedding_dim=256, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.state_dim = state_dim
        # Learnable mask token缺失维度用 mask_token 填充而非置零
        self.mask_token = nn.Parameter(torch.zeros(state_dim))
        # 输入为 [masked_state, M] 拼接，维度为 state_dim * 2
        self.net = mlp([state_dim * 2, *([hidden_dim] * n_hidden), embedding_dim])
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, state, mask):
        # mask: 1=可观测, 0=缺失
        # 缺失维度用 mask_token 填充，而非置零
        masked_state = state * mask + self.mask_token * (1 - mask)
        # 拼接掩码矩阵作为额外输入，让网络感知哪些维度缺失
        encoder_input = torch.cat([masked_state, mask], dim=-1)
        z = self.net(encoder_input)
        return self.ln(z)

class StateDecoder(nn.Module):
    def __init__(self, embedding_dim, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        # 输出层不需要激活函数，直接输出预测的状态值
        self.net = mlp([embedding_dim, *([hidden_dim] * n_hidden), state_dim])

    def forward(self, z):
        return self.net(z)


