from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple replay buffer that stores transitions as numpy arrays."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self._capacity = int(capacity)
        self._memory: list[Transition] = []
        self._position: int = 0

    def __len__(self) -> int:
        return len(self._memory)

    def push(self, transition: Transition) -> None:
        if len(self._memory) < self._capacity:
            self._memory.append(transition)
        else:
            self._memory[self._position] = transition
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        return random.sample(self._memory, batch_size)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_units: Tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for width in hidden_units:
            layers.append(nn.Linear(last_dim, width))
            layers.append(nn.ReLU())
            last_dim = width
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return Q-values for each action."""
        return self.model(x)


@dataclass
class DQNHyperparams:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 0.4
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 64
    replay_capacity: int = 50_000
    target_update_interval: int = 500
    warmup_steps: int = 1_000
    max_grad_norm: float | None = 5.0
    hidden_layers: Tuple[int, ...] = (128, 64)


@dataclass
class DQNAgent:
    observation_size: int
    action_size: int
    hyperparams: DQNHyperparams = field(default_factory=DQNHyperparams)
    device: Optional[torch.device] = None
    observation_low: Optional[np.ndarray] = None
    observation_high: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.observation_size <= 0:
            raise ValueError("Observation size must be positive")
        if self.action_size <= 0:
            raise ValueError("Action size must be positive")

        self.device = self.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = QNetwork(
            self.observation_size,
            self.action_size,
            self.hyperparams.hidden_layers,
        ).to(self.device)
        self.target_net = QNetwork(
            self.observation_size,
            self.action_size,
            self.hyperparams.hidden_layers,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.hyperparams.learning_rate
        )
        self.replay_buffer = ReplayBuffer(self.hyperparams.replay_capacity)

        self._epsilon = float(self.hyperparams.epsilon_start)
        self._steps_done = 0
        self._updates_done = 0

        self._setup_normalization()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def select_action(self, observation: np.ndarray) -> int:
        """Return an action using epsilon-greedy exploration."""
        self._steps_done += 1
        if random.random() < self._epsilon:
            return random.randrange(self.action_size)
        return self.greedy_action(observation)

    def greedy_action(self, observation: np.ndarray) -> int:
        normalized = self._normalize(observation)
        obs_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        # Tie-breaker prefers higher index (mirroring SARSA behavior)
        max_q = q_values.max(dim=1, keepdim=True)[0]
        best_actions = (q_values == max_q).nonzero(as_tuple=False)[:, 1]
        return int(best_actions[-1].item())

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_norm = self._normalize(state)
        next_state_norm = self._normalize(next_state)
        self.replay_buffer.push(Transition(state_norm, action, reward, next_state_norm, done))

    def decay_epsilon(self) -> None:
        if self._epsilon > self.hyperparams.epsilon_end:
            self._epsilon = max(
                self.hyperparams.epsilon_end,
                self._epsilon * self.hyperparams.epsilon_decay,
            )

    def update(self) -> Optional[float]:
        """Perform a single gradient update. Returns loss if training occurred."""
        if len(self.replay_buffer) < self.hyperparams.batch_size:
            return None
        if self._steps_done < self.hyperparams.warmup_steps:
            return None

        transitions = self.replay_buffer.sample(self.hyperparams.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.as_tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_action_indices = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_action_indices)
            targets = rewards + self.hyperparams.gamma * (1.0 - dones) * next_q_values

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.hyperparams.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.hyperparams.max_grad_norm)
        self.optimizer.step()

        self._updates_done += 1
        if self._updates_done % self.hyperparams.target_update_interval == 0:
            self._sync_target_network()

        return float(loss.item())

    def _sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "hyperparams": self.hyperparams,
            "epsilon": self._epsilon,
            "steps_done": self._steps_done,
            "updates_done": self._updates_done,
            "observation_low": self._orig_low.tolist(),
            "observation_high": self._orig_high.tolist(),
        }
        torch.save(payload, output_path)

    @classmethod
    def load(cls, path: Path, device: Optional[torch.device] = None) -> "DQNAgent":
        checkpoint = torch.load(Path(path), map_location=device or "cpu")
        hyperparams = checkpoint.get("hyperparams")
        agent = cls(
            observation_size=checkpoint["observation_size"],
            action_size=checkpoint["action_size"],
            hyperparams=hyperparams,
            device=device,
            observation_low=np.array(checkpoint.get("observation_low"), dtype=np.float32)
            if checkpoint.get("observation_low") is not None
            else None,
            observation_high=np.array(checkpoint.get("observation_high"), dtype=np.float32)
            if checkpoint.get("observation_high") is not None
            else None,
        )
        agent.policy_net.load_state_dict(checkpoint["state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent._epsilon = float(checkpoint.get("epsilon", agent.hyperparams.epsilon_end))
        agent._steps_done = int(checkpoint.get("steps_done", 0))
        agent._updates_done = int(checkpoint.get("updates_done", 0))
        agent.policy_net.to(agent.device)
        agent.target_net.to(agent.device)
        return agent

    def set_eval_mode(self) -> None:
        """Set networks to evaluation mode (e.g., before checkpoint evaluation)."""
        self.policy_net.eval()
        self.target_net.eval()

    def set_train_mode(self) -> None:
        self.policy_net.train()
        self.target_net.eval()

    def _setup_normalization(self) -> None:
        if self.observation_low is None or self.observation_high is None:
            self._orig_low = np.zeros(self.observation_size, dtype=np.float32)
            self._orig_high = np.ones(self.observation_size, dtype=np.float32)
        else:
            self._orig_low = np.asarray(self.observation_low, dtype=np.float32)
            self._orig_high = np.asarray(self.observation_high, dtype=np.float32)
            if self._orig_low.shape[0] != self.observation_size or self._orig_high.shape[0] != self.observation_size:
                raise ValueError("Observation bounds must match observation_size")

        span = self._orig_high - self._orig_low
        span = np.where(np.isfinite(span), span, 1.0)
        span[span == 0.0] = 1.0
        self._scale = span.astype(np.float32)

    def _normalize(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape[0] != self.observation_size:
            raise ValueError("Observation size mismatch")
        clipped = np.clip(obs, self._orig_low, self._orig_high)
        return (clipped - self._orig_low) / self._scale
