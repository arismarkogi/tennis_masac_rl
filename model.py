import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX, EPS = -20, 2, 1e-6

def _mlp(dims, act=nn.ReLU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Per-agent stochastic policy. Runs locally at execution time."""
    def __init__(self, state_size, action_size, hidden=(256, 256)):
        super().__init__()
        self.trunk         = _mlp([state_size] + list(hidden))
        self.relu          = nn.ReLU()
        self.mean_layer    = nn.Linear(hidden[-1], action_size)
        self.log_std_layer = nn.Linear(hidden[-1], action_size)

    def forward(self, s):
        x = self.relu(self.trunk(s))
        return self.mean_layer(x), self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, s):
        """Returns action, log_prob, deterministic_action."""
        mean, log_std = self(s)
        std  = log_std.exp()
        x_t  = Normal(mean, std).rsample()
        y_t  = torch.tanh(x_t)
        lp   = Normal(mean, std).log_prob(x_t) - torch.log(1 - y_t.pow(2) + EPS)
        return y_t, lp.sum(-1, keepdim=True), torch.tanh(mean)


class CentralizedCritic(nn.Module):
    """Twin Q-networks conditioned on the full joint (obs, action) tuple."""
    def __init__(self, joint_state_size, joint_action_size, num_agents=2, hidden=(256, 256)):
        super().__init__()
        in_dim  = joint_state_size + joint_action_size
        dims    = [in_dim] + list(hidden) + [num_agents]
        self.q1 = _mlp(dims)
        self.q2 = _mlp(dims)

    def forward(self, js, ja):
        x = torch.cat([js, ja], dim=-1)
        return self.q1(x), self.q2(x)