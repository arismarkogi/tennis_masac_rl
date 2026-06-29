import time
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from agent import MASACAgent

# ── Configuration ──
ENV_PATH = "/data/Tennis_Linux_NoVis/Tennis"
N_EPISODES = 3000
MAX_T = 1000
SOLVE_SCORE = 0.5

def train():
    # Initialize Environment
    env = UnityEnvironment(file_name=ENV_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Initialize Agent
    agent = MASACAgent(num_agents, state_size, action_size, device)

    scores_all = []
    score_window = deque(maxlen=100)
    best_avg = -float("inf")
    total_steps = 0
    t0 = time.time()

    print("Starting Training...")
    for ep in range(1, N_EPISODES + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        ep_rewards = np.zeros(num_agents)

        for t in range(MAX_T):
            if total_steps < agent.warmup_steps:
                actions = np.random.uniform(-1, 1, (num_agents, action_size))
            else:
                actions = agent.act(states)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = np.array(env_info.rewards)
            dones = np.array(env_info.local_done, dtype=np.float32)

            # Pass raw, individual rewards
            agent.step(states, actions, rewards, next_states, dones)
            ep_rewards += rewards
            states = next_states
            total_steps += 1

            if total_steps >= agent.warmup_steps:
                for _ in range(agent.updates_per_step):
                    agent.learn()

            if np.any(dones):
                break

        ep_score = float(np.max(ep_rewards))
        scores_all.append(ep_score)
        score_window.append(ep_score)
        avg = float(np.mean(score_window))

        if avg > best_avg:
            best_avg = avg
            agent.save("best_masac.pth")

        if ep % 100 == 0:
            print(f"Ep {ep:4d} | score {ep_score:.3f} | avg100 {avg:.3f} | "
                  f"best {best_avg:.3f} | alpha {[round(a,4) for a in agent.alphas]} "
                  f"| steps {total_steps:,} | {time.time()-t0:.0f}s")

        if len(score_window) == 100 and avg >= SOLVE_SCORE:
            print(f"\nSolved in {ep} episodes! avg100 = {avg:.4f}")
            agent.save(f"solved_ep{ep}_masac.pth")
            break

    env.close()

    # Plot results
    plot_scores(scores_all, window=100)

def plot_scores(scores, window=100):
    rolling = [np.mean(scores[max(0, i - window): i + 1]) for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(scores, alpha=0.25, color="steelblue", label="Episode score")
    ax.plot(rolling, lw=2, color="steelblue", label=f"{window}-ep avg")
    ax.axhline(0.5, color="green", ls="--", lw=1.5, label="Solve threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (max over agents)")
    ax.legend()
    plt.savefig("training_curve.png", dpi=150)
    print("Saved training_curve.png")

if __name__ == "__main__":
    train()