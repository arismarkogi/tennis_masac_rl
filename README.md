# Multi-Agent Soft Actor-Critic (MASAC) - Unity Tennis

This repository contains a modular PyTorch implementation of the MASAC algorithm, designed to solve the collaborative Unity ML-Agents Tennis environment.

## 1. Architecture Details
The algorithm utilizes **Centralized Training with Decentralized Execution (CTDE)**.
* **Centralized Critic:** Twin Q-networks evaluate joint states and joint actions during training to stabilize learning.
* **Decentralized Actors:** Agents make independent decisions based entirely on local observations during inference.
* **Auto-Tuning Temperature:** The entropy parameter ($\alpha$) dynamically adjusts to balance exploration and pure exploitation.

## 2. Repository Structure
* `model.py`: Defines the PyTorch neural network architectures (`GaussianActor` and `CentralizedCritic`).
* `buffer.py`: Implements the `ReplayBuffer` for off-policy experience sampling.
* `agent.py`: Contains the `MASACAgent` orchestrator, managing updates, target networks, and action sampling.
* `train.py`: The main executable script containing the environment interaction and training loop.
* `best_masac.pth`: Saved model weights of the highest-performing network configuration.

## 3. Installation
Ensure you have Python 3, PyTorch, and the Unity ML-Agents toolkit installed:
```bash
pip install protobuf==3.20.3
pip install torch==2.1.0
pip install .
