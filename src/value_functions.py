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
    
# --- [NEW CODE START] Masked-IQL Components ---
class StateEncoder(nn.Module):
    def __init__(self, state_dim, embedding_dim=256, hidden_dim=256, n_hidden=2):
        super().__init__()
        # 使用 util.py 中的 mlp 构建，最后加一个 LayerNorm 增加训练稳定性
        self.net = mlp([state_dim, *([hidden_dim] * n_hidden), embedding_dim])
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, state):
        z = self.net(state)
        return self.ln(z)

class StateDecoder(nn.Module):
    def __init__(self, embedding_dim, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        # 输出层不需要激活函数，直接输出预测的状态值
        self.net = mlp([embedding_dim, *([hidden_dim] * n_hidden), state_dim])

    def forward(self, z):
        return self.net(z)
# --- [NEW CODE END] ---