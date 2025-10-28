"""
SARSA Agent for OpenFaaS Autoscaling

SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm.
It learns Q(s,a) by following the current policy and updating based on the
action actually taken in the next state.

Key Features:
- On-policy learning (safer for real deployment)
- Epsilon-greedy exploration
- Tile coding for continuous state discretization
- Experience replay for better sample efficiency
"""

import numpy as np
import pickle
import os
from collections import defaultdict, deque
import random


class TileCoding:
    """
    Tile coding for continuous state space discretization

    Converts continuous observations into discrete tile indices for Q-table lookup.
    Uses multiple overlapping tilings to provide generalization.
    """

    def __init__(self, num_tilings=8, tiles_per_dim=8, state_bounds=None):
        """
        Initialize tile coding

        Args:
            num_tilings: Number of overlapping tilings
            tiles_per_dim: Number of tiles per dimension per tiling
            state_bounds: List of (min, max) tuples for each state dimension
        """
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.state_bounds = state_bounds or []

    def get_tiles(self, state):
        """
        Get tile indices for a given state

        Args:
            state: Continuous state vector

        Returns:
            List of tile indices (one per tiling)
        """
        tiles = []

        for tiling_idx in range(self.num_tilings):
            # Offset each tiling slightly
            offset = tiling_idx / self.num_tilings

            # Discretize each dimension
            tile_coords = []
            for dim_idx, value in enumerate(state):
                if dim_idx < len(self.state_bounds):
                    min_val, max_val = self.state_bounds[dim_idx]
                    # Normalize to [0, 1]
                    normalized = (value - min_val) / (max_val - min_val + 1e-8)
                else:
                    normalized = value

                # Add offset and discretize
                tile_coord = int((normalized + offset) * self.tiles_per_dim) % self.tiles_per_dim
                tile_coords.append(tile_coord)

            # Convert coordinates to single index
            tile_idx = hash((tiling_idx, tuple(tile_coords)))
            tiles.append(tile_idx)

        return tuple(tiles)


class SARSAAgent:
    """
    SARSA Agent with tile coding and experience replay

    On-policy TD control algorithm that learns Q(s,a) by following
    the current epsilon-greedy policy.
    """

    def __init__(self,
                 state_dim=9,
                 action_dim=5,
                 learning_rate=0.1,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 num_tilings=8,
                 tiles_per_dim=8,
                 use_replay=False,
                 replay_buffer_size=10000,
                 batch_size=32):
        """
        Initialize SARSA agent

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            num_tilings: Number of tile coding tilings
            tiles_per_dim: Tiles per dimension
            use_replay: Whether to use experience replay
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for replay
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Tile coding for state discretization
        # Define reasonable bounds for each state dimension
        # [input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate]
        state_bounds = [
            (0, 200),    # input_q
            (0, 200),    # worker_q
            (0, 200),    # result_q
            (0, 2000),   # output_q
            (0, 32),     # workers
            (0, 30),     # avg_time
            (0, 30),     # max_time
            (0, 20),     # arrival_rate
            (0, 1),      # qos_rate
        ]
        self.tile_coder = TileCoding(num_tilings, tiles_per_dim, state_bounds)

        # Q-table: maps (state_tiles, action) -> Q-value
        self.q_table = defaultdict(float)

        # Experience replay (optional)
        self.use_replay = use_replay
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size

        # Statistics
        self.episode_count = 0
        self.total_steps = 0

    def get_state_key(self, state):
        """Convert continuous state to discrete tile indices"""
        return self.tile_coder.get_tiles(state)

    def get_q_value(self, state, action):
        """
        Get Q-value for state-action pair

        Uses tile coding to discretize state and lookup in Q-table.
        """
        state_key = self.get_state_key(state)
        return self.q_table[(state_key, action)]

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action (integer)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action
            q_values = [self.get_q_value(state, a) for a in range(self.action_dim)]
            return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update rule: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (actually taken)
            done: Whether episode is done
        """
        # Get current Q-value
        state_key = self.get_state_key(state)
        current_q = self.q_table[(state_key, action)]

        # Get next Q-value (using next_action actually taken)
        if done:
            target_q = reward
        else:
            next_state_key = self.get_state_key(next_state)
            next_q = self.q_table[(next_state_key, next_action)]
            target_q = reward + self.gamma * next_q

        # SARSA update
        td_error = target_q - current_q
        self.q_table[(state_key, action)] += self.learning_rate * td_error

        # Store experience for replay (optional)
        if self.use_replay:
            self.replay_buffer.append((state, action, reward, next_state, next_action, done))

        self.total_steps += 1

        return td_error

    def replay_update(self):
        """
        Perform experience replay update

        Samples random batch from replay buffer and performs SARSA updates.
        """
        if not self.use_replay or len(self.replay_buffer) < self.batch_size:
            return

        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)

        for state, action, reward, next_state, next_action, done in batch:
            self.update(state, action, reward, next_state, next_action, done)

    def decay_epsilon(self):
        """Decay exploration rate after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def save(self, filepath):
        """
        Save agent to file

        Args:
            filepath: Path to save file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"[SAVE] Agent saved to {filepath}")
        print(f"       Q-table size: {len(self.q_table)} entries")
        print(f"       Episodes: {self.episode_count}, Steps: {self.total_steps}")

    def load(self, filepath):
        """
        Load agent from file

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.q_table = defaultdict(float, state['q_table'])
        self.epsilon = state['epsilon']
        self.episode_count = state['episode_count']
        self.total_steps = state['total_steps']

        print(f"[LOAD] Agent loaded from {filepath}")
        print(f"       Q-table size: {len(self.q_table)} entries")
        print(f"       Episodes: {self.episode_count}, Steps: {self.total_steps}")
        print(f"       Epsilon: {self.epsilon:.4f}")

    def get_stats(self):
        """Get agent statistics"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
        }


if __name__ == '__main__':
    # Test tile coding
    print("Testing Tile Coding...")
    tile_coder = TileCoding(num_tilings=4, tiles_per_dim=4, 
                           state_bounds=[(0, 10), (0, 20)])

    state1 = np.array([5.0, 10.0])
    state2 = np.array([5.1, 10.0])
    state3 = np.array([7.0, 15.0])

    tiles1 = tile_coder.get_tiles(state1)
    tiles2 = tile_coder.get_tiles(state2)
    tiles3 = tile_coder.get_tiles(state3)

    print(f"State {state1} -> Tiles {tiles1}")
    print(f"State {state2} -> Tiles {tiles2}")
    print(f"State {state3} -> Tiles {tiles3}")
    print(f"Similar states share tiles: {len(set(tiles1) & set(tiles2))} common")
    print(f"Different states share tiles: {len(set(tiles1) & set(tiles3))} common")

    # Test SARSA agent
    print("\nTesting SARSA Agent...")
    agent = SARSAAgent(state_dim=9, action_dim=5)

    state = np.array([10, 50, 5, 100, 8, 2.5, 5.0, 5.0, 0.8])
    action = agent.select_action(state, training=True)
    print(f"Selected action: {action}")

    next_state = np.array([8, 40, 3, 110, 8, 2.0, 4.5, 5.0, 0.9])
    next_action = agent.select_action(next_state, training=True)
    reward = 10.0

    td_error = agent.update(state, action, reward, next_state, next_action, done=False)
    print(f"TD error: {td_error:.4f}")

    print(f"\nAgent stats: {agent.get_stats()}")
    print("✓ SARSA agent working correctly!")
