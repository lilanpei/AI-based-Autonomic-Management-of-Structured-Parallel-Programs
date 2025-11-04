"""Tabular SARSA agent for the OpenFaaS autoscaling environment."""

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Iterable, Tuple

import numpy as np


@dataclass
class DiscretizationConfig:
    """Configuration describing how to discretize each observation dimension."""

    bins_per_dimension: Tuple[int, ...]
    observation_low: Iterable[float]
    observation_high: Iterable[float]

    def __post_init__(self) -> None:
        low = np.array(self.observation_low, dtype=float)
        high = np.array(self.observation_high, dtype=float)

        if low.shape != high.shape:
            raise ValueError("observation bounds must have matching shapes")
        if len(self.bins_per_dimension) != low.size:
            raise ValueError("bins_per_dimension must match observation dimensionality")

        self.low = low
        self.high = high


@dataclass
class SARSAHyperparams:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995


@dataclass
class SARSAAgent:
    """Simple tabular SARSA agent with configurable discretization."""

    action_size: int
    discretization: DiscretizationConfig
    hyperparams: SARSAHyperparams = field(default_factory=SARSAHyperparams)

    def __post_init__(self) -> None:
        self._build_bins()
        self._q_table: DefaultDict[Tuple[int, ...], np.ndarray]
        self._q_table = defaultdict(lambda: np.zeros(self.action_size, dtype=np.float32))
        self._epsilon = self.hyperparams.epsilon

    def _build_bins(self) -> None:
        """Pre-compute bin edges for each observation dimension."""
        eps = 1e-6
        self._bins = []
        for i, n_bins in enumerate(self.discretization.bins_per_dimension):
            low = self.discretization.low[i]
            high = self.discretization.high[i]
            if math.isinf(low) or math.isinf(high):
                # Fallback to a reasonable numeric range
                low = -1.0
                high = 1.0
            if high <= low:
                high = low + 1.0
            # np.linspace returns n_bins+1 edges; we keep internal boundaries only
            edges = np.linspace(low, high, n_bins + 1, dtype=np.float32)[1:-1]
            if edges.size:
                edges[0] -= eps
                edges[-1] += eps
            self._bins.append(edges)

    def discretize(self, observation: np.ndarray) -> Tuple[int, ...]:
        """Convert a continuous observation into a tuple of bin indices."""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.size != len(self._bins):
            raise ValueError("Observation dimensionality mismatch")
        indices = []
        for value, edges in zip(observation, self._bins):
            if edges.size == 0:
                indices.append(0)
            else:
                indices.append(int(np.digitize(value, edges)))
        return tuple(indices)

    def select_action(self, state: Tuple[int, ...]) -> int:
        """Return an action index using epsilon-greedy policy."""
        if np.random.random() < self._epsilon:
            return np.random.randint(self.action_size)
        return self.greedy_action(state)

    def greedy_action(self, state: Tuple[int, ...]) -> int:
        """Return the greedy action with randomized tie-breaking."""
        q_values = self._q_table[state]
        max_value = q_values.max()
        best_actions = np.flatnonzero(q_values == max_value)
        if best_actions.size == 0:
            return 0
        return int(np.random.choice(best_actions))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        next_action: int,
    ) -> None:
        """Apply the SARSA update rule."""
        q_sa = self._q_table[state][action]
        q_next = self._q_table[next_state][next_action]
        td_target = reward + self.hyperparams.gamma * q_next
        self._q_table[state][action] += self.hyperparams.alpha * (td_target - q_sa)

    def decay_epsilon(self) -> None:
        self._epsilon = max(self.hyperparams.epsilon_min, self._epsilon * self.hyperparams.epsilon_decay)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        return self._q_table[state].copy()

    def save(self, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "action_size": self.action_size,
            "hyperparams": self.hyperparams,
            "discretization": self.discretization,
            "epsilon": self._epsilon,
            "q_table": dict(self._q_table),
        }
        with output_path.open("wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path) -> "SARSAAgent":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        agent = cls(
            action_size=payload["action_size"],
            discretization=payload["discretization"],
            hyperparams=payload["hyperparams"],
        )
        agent._q_table = defaultdict(lambda: np.zeros(agent.action_size, dtype=np.float32))
        for state, values in payload["q_table"].items():
            agent._q_table[state] = np.array(values, dtype=np.float32)
        agent._epsilon = payload.get("epsilon", agent.hyperparams.epsilon)
        return agent
