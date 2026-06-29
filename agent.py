import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import GaussianActor, CentralizedCritic
from buffer import ReplayBuffer

class MASACAgent:
    def __init__(self, num_agents, state_size, action_size, device, seed=0,
                 hidden=(256, 256), actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=5e-3, buffer_size=int(1e6), batch_size=256,
                 warmup_steps=2000, updates_per_step=2):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.na = num_agents
        self.device, self.gamma, self.tau = device, gamma, tau
        self.batch_size     = batch_size
        self.warmup_steps   = warmup_steps
        self.updates_per_step = updates_per_step
        
        js = num_agents * state_size
        ja = num_agents * action_size

        # ── Per-agent actors (decentralized execution) ──
        self.actors = [GaussianActor(state_size, action_size, hidden).to(device) for _ in range(num_agents)]
        self.actor_opts = [optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]

        # ── Shared centralized twin critics ──
        self.critic = CentralizedCritic(js, ja, num_agents=num_agents, hidden=hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── Per-agent auto temperature ──
        self.target_entropy = -action_size  
        self.log_alphas = [torch.zeros(1, requires_grad=True, device=device) for _ in range(num_agents)]
        self.alpha_opts = [optim.Adam([la], lr=alpha_lr) for la in self.log_alphas]

        # ── Replay buffer ──
        self.buffer = ReplayBuffer(buffer_size, num_agents, state_size, action_size, device, seed)

    @property
    def alphas(self):
        return [la.exp().item() for la in self.log_alphas]

    def act(self, states, deterministic=False):
        acts = []
        for i, actor in enumerate(self.actors):
            actor.eval()
            with torch.no_grad():
                # Unleashed clamp to allow pure exploitation
                self.log_alphas[i].clamp_(min=-10.0) 
                s = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                a, _, mu = actor.sample(s)
            actor.train()
            acts.append((mu if deterministic else a).cpu().numpy().squeeze(0))
        return np.array(acts)

    def step(self, s, a, r, ns, d):
        self.buffer.add(np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d))

    def learn(self):
        if not self.buffer.ready(self.batch_size):
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        B, N = self.batch_size, self.na

        # ── 1. Critic Update ──
        with torch.no_grad():
            na_list, lp_list = [], []
            for i, actor in enumerate(self.actors):
                na, lp, _ = actor.sample(next_states[:, i])
                na_list.append(na)
                lp_list.append(lp)
            
            j_ns = next_states.view(B, -1)
            j_na = torch.cat(na_list, dim=-1)
            
            q1n, q2n = self.critic_target(j_ns, j_na) 
            ent_corr = torch.cat([self.log_alphas[i].exp() * lp_list[i] for i in range(N)], dim=-1)
            
            r = rewards.squeeze(-1)
            d = dones.squeeze(-1)
            q_tgt = r + self.gamma * (1 - d) * (torch.min(q1n, q2n) - ent_corr)

        j_s = states.view(B, -1)
        j_a = actions.view(B, -1)
        q1, q2 = self.critic(j_s, j_a)

        c_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── 2. Actor + Temperature Update (per agent) ──
        for i, actor in enumerate(self.actors):
            a_i, lp_i, _ = actor.sample(states[:, i])
            other = [actions[:, j].clone() for j in range(N)] 
            other[i] = a_i                                    

            q1p, q2p = self.critic(j_s.detach(), torch.cat(other, dim=-1))
            q_i = torch.min(q1p[:, i], q2p[:, i]).unsqueeze(-1) 
            a_loss = (self.log_alphas[i].exp().detach() * lp_i - q_i).mean()

            self.actor_opts[i].zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            self.actor_opts[i].step()

            if self.log_alphas[i].requires_grad:
                al_loss = -(self.log_alphas[i] * (lp_i + self.target_entropy).detach()).mean()
                self.alpha_opts[i].zero_grad()
                al_loss.backward()
                self.alpha_opts[i].step()

        # ── 3. Soft-update target critic ──
        for po, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * po.data + (1 - self.tau) * pt.data)

    def save(self, path):
        d = {"critic": self.critic.state_dict(),
             "critic_target": self.critic_target.state_dict(),
             "log_alphas": [la.data.clone() for la in self.log_alphas]}
        for i, a in enumerate(self.actors):
            d[f"actor_{i}"] = a.state_dict()
        torch.save(d, path)
        print(f"Saved -> {path}")

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        for i, a in enumerate(self.actors):
            a.load_state_dict(d[f"actor_{i}"])
        for i, la in enumerate(self.log_alphas):
            la.data.copy_(d["log_alphas"][i])
        print(f"Loaded <- {path}")