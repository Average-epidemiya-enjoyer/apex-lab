# apex-lab

Research framework for benchmarking path-planning algorithms on racing circuits — from A* and RRT to genetic algorithms, MPC, and deep reinforcement learning.

## Goal

Provide a clean, reproducible environment for comparing how different planning and learning algorithms find the optimal racing line (apex trajectory) on a given track. All algorithms share a unified interface so results are directly comparable under the same physics model and metrics.

## Algorithms

### Classical Planners
| Algorithm | Class | Description |
|-----------|-------|-------------|
| A* | `AStarPlanner` | Grid-based heuristic search |
| RRT | `RRTPlanner` | Rapidly-exploring Random Trees |
| RRT* | `RRTStarPlanner` | Asymptotically optimal RRT |
| Genetic Algorithm | `GAPlanner` | Evolutionary trajectory optimisation |
| MPC | `MPCPlanner` | Model Predictive Control |

### Learning Agents
| Algorithm | Class | Description |
|-----------|-------|-------------|
| Q-Learning | `QLearningAgent` | Tabular Q-Learning |
| DQN | `DQNAgent` | Deep Q-Network (discrete actions) |
| PPO | `PPOAgent` | Proximal Policy Optimisation (continuous) |

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Run a default experiment
python -m apex_lab.cli run --config experiments/default.yaml

# 3. List available algorithms
python -m apex_lab.cli list-algorithms

# 4. Compare two planners on the oval track
python -m apex_lab.cli compare --algorithms a_star,rrt_star --track oval
```

## Project Structure

```
apex_lab/
├── environment/     # Track geometry, car physics, simulator step
├── algorithms/
│   ├── base.py      # BasePlanner / BaseAgent interfaces
│   ├── classical/   # A*, RRT, RRT*, GA, MPC
│   └── learning/    # Q-Learning, DQN, PPO
├── metrics/         # Lap time, path length, smoothness, reward curves
├── visualization/   # Pygame renderer, matplotlib plots
├── configs/         # Shared config schemas
└── cli.py           # Entry point
experiments/         # Ready-to-run YAML scenarios
tests/
notebooks/
```

## Configuration

Experiments are driven by YAML files in `experiments/`. See `experiments/default.yaml` for a minimal example.

## License

MIT
