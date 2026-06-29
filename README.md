# Multi-Agent Soft Actor-Critic (MASAC) - Unity Tennis

This repository contains a modular PyTorch implementation of the MASAC algorithm, designed to solve the collaborative Unity ML-Agents Tennis environment.

## 1. Environment Details
In this environment, two agents control tennis rackets on opposite sides of a net. The goal is to collaborate and maintain a rally for as long as possible.
* **State Space:** Each agent receives a local observation vector of **24 continuous variables**, tracking the position and velocity of the ball and racket over three stacked frames.
* **Action Space:** Each agent has a continuous action space of **2 variables** for movement (toward/away from the net) and jumping, bounded between `-1` and `1`.
* **Reward:** `+0.1` for hitting the ball over the net, `-0.01` for dropping it or hitting out of bounds.
* **Solve Criteria:** The environment is considered solved when the average maximum score of the two agents over **100 consecutive episodes reaches at least +0.5**.

## 2. Architecture Details
The algorithm utilizes **Centralized Training with Decentralized Execution (CTDE)**.
* **Centralized Critic:** Twin Q-networks evaluate joint states and joint actions during training to stabilize learning.
* **Decentralized Actors:** Agents make independent decisions based entirely on local observations during inference.
* **Auto-Tuning Temperature:** The entropy parameter ($\alpha$) dynamically adjusts to balance exploration and pure exploitation.

## 3. Repository Structure
* `model.py`: Defines the PyTorch neural network architectures (`GaussianActor` and `CentralizedCritic`).
* `buffer.py`: Implements the `ReplayBuffer` for off-policy experience sampling.
* `agent.py`: Contains the `MASACAgent` orchestrator, managing updates, target networks, and action sampling.
* `train.py`: The main executable script containing the environment interaction and training loop.
* `best_masac.pth`: Saved model weights of the highest-performing network configuration.

## 4. Installation & Getting Started
Ensure you have Python 3, PyTorch, and the Unity ML-Agents toolkit installed:
```bash
pip install protobuf==3.20.3
pip install torch==2.1.0
pip install .
```
## 5. Instructions
To train the agents from scratch using the modular codebase, simply execute the main training script from your terminal:
```bash
python train.py
```
