"""
Autoscaling for OpenFaaS

Provides:
- Gym environment for real OpenFaaS deployment
- RL agents (SARSA, Q-Learning, Monte Carlo, PPO)
- Reactive baseline policies (Average-based, Maximum-based)
"""

from .openfaas_autoscaling_env import OpenFaaSAutoscalingEnv

__version__ = "1.0.0"
__all__ = ["OpenFaaSAutoscalingEnv"]
