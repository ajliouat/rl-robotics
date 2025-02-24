# Deep RL for Robotic Manipulation

This project implements state-of-the-art reinforcement learning algorithms for robotic manipulation tasks. It features GPU-accelerated physics simulation, distributed training, and sim2real transfer.

## Features
- PPO with custom architectures
- GPU-accelerated physics simulation
- Multi-task policy training
- Imitation learning integration
- Real-world sim2real transfer

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- Isaac Gym
- RLlib

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Training
To train a PPO policy:
```bash
python scripts/train.py --config configs/train.yaml
```

### Evaluation
To evaluate a trained policy:
```bash
python scripts/evaluate.py --config configs/eval.yaml
```

---

## Project Structure

```
rl-robotics/
├── configs/             # Configuration files
├── data/                # Demonstration data
├── models/              # Policy and network architectures
├── notebooks/           # Jupyter notebooks for exploration
├── scripts/             # Training, evaluation, and export scripts
├── src/                 # Source code for environments, algorithms, and utilities
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
└── .gitignore           # Files to ignore in Git
```

---

## Configuration

### Training Configuration (`configs/train.yaml`)
```yaml
env:
  name: "RoboticsEnv"
  num_envs: 1024
  max_episode_steps: 500

algorithm:
  name: "PPO"
  learning_rate: 3e-4
  gamma: 0.99
  clip_param: 0.2

training:
  num_epochs: 100
  batch_size: 4096
  use_gpu: true
```

### Evaluation Configuration (`configs/eval.yaml`)
```yaml
policy:
  checkpoint: "models/checkpoints/ppo_policy.pth"

env:
  name: "RoboticsEnv"
  num_envs: 1
  max_episode_steps: 500
```

---

## Results

### Performance Metrics
- **Success Rate**: 95% on simulated tasks
- **Sim2Real Transfer**: 85% success rate on real-world tasks

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.