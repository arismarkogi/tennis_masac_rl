import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, num_agents, state_size, action_size, device, seed=0):
        self.capacity = capacity
        self.device   = device
        self.rng      = np.random.default_rng(seed)
        self.states      = np.zeros((capacity, num_agents, state_size),  dtype=np.float32)
        self.actions     = np.zeros((capacity, num_agents, action_size), dtype=np.float32)
        self.rewards     = np.zeros((capacity, num_agents),              dtype=np.float32)
        self.next_states = np.zeros((capacity, num_agents, state_size),  dtype=np.float32)
        self.dones       = np.zeros((capacity, num_agents),              dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, states, actions, rewards, next_states, dones):
        self.states[self.ptr]      = states
        self.actions[self.ptr]     = actions
        self.rewards[self.ptr]     = rewards
        self.next_states[self.ptr] = next_states
        self.dones[self.ptr]       = dones
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        def t(a): return torch.from_numpy(a[idx]).to(self.device)
        return (t(self.states), t(self.actions),
                t(self.rewards).unsqueeze(-1),
                t(self.next_states),
                t(self.dones).unsqueeze(-1))

    def ready(self, batch_size): return self.size >= batch_size
    def __len__(self):           return self.size