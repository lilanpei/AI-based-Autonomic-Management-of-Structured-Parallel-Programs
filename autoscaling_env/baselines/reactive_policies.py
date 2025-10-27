"""
Reactive Baseline Policies for Gym Environment

Two performance-based baseline approaches:
1. ReactiveAverage: Uses average processing time (conservative)
2. ReactiveMaximum: Uses maximum processing time (aggressive)

Both calculate optimal parallelism based on:
- Task arrival rate
- Task processing time
- Target utilization
- Safety margin
"""

import numpy as np


class ReactiveAverage:
    """
    Reactive Average-Based Policy (Baseline 1)

    Queue-based model using AVERAGE processing time:
    - Calculates required workers based on: worker_queue_length, arrival_rate, avg_processing_time
    - Formula: optimal_workers = (worker_queue_length / avg_processing_time) + (arrival_rate * avg_processing_time)
    - First term: workers needed to clear current queue
    - Second term: workers needed to handle incoming tasks

    This is a conservative approach (uses average time)

    Compatible with OpenFaaS Autoscaling Gym Environment
    """

    def __init__(self):
        """
        Initialize reactive average-based policy
        No hyperparameters - purely queue-based calculation
        """
        # For compatibility with RL agents
        self.name = "Reactive-Average"

    def select_action(self, observation, training=True):
        """
        Select action based on queue-driven calculation (average time)

        Args:
            observation: [input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate]
            training: Ignored (for compatibility)

        Returns:
            action: 0=-2, 1=-1, 2=0, 3=+1, 4=+2

        Formula:
            optimal_workers = (worker_queue / avg_time) + (arrival_rate * avg_time)
            - First term: workers to clear current queue
            - Second term: workers to handle incoming rate
        """
        input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate = observation

        # Calculate optimal workers based on queue and arrival rate
        if avg_time > 0:
            # Workers needed to clear current queue
            workers_for_queue = worker_q / avg_time if worker_q > 0 else 0

            # Workers needed to handle incoming tasks
            workers_for_arrival = arrival_rate * avg_time if arrival_rate > 0 else 0

            # Total optimal workers
            optimal_workers = workers_for_queue + workers_for_arrival

            # Ensure at least 1 worker
            optimal_workers = max(1, optimal_workers)
        else:
            # No processing time data: scale down to minimum
            optimal_workers = 1

        # Calculate difference
        delta = optimal_workers - workers

        # Map delta to action
        if delta >= 2.5:
            return 4  # Scale up by 2
        elif delta >= 1.0:
            return 3  # Scale up by 1
        elif delta <= -2.5:
            return 0  # Scale down by 2
        elif delta <= -1.0:
            return 1  # Scale down by 1
        else:
            return 2  # No change

    def get_stats(self):
        """Get policy statistics"""
        return {
            'policy': 'reactive_average',
            'approach': 'queue-based with average processing time'
        }


class ReactiveMaximum:
    """
    Reactive Maximum-Based Policy (Baseline 2)

    Queue-based model using MAXIMUM processing time:
    - Calculates required workers based on: worker_queue_length, arrival_rate, max_processing_time
    - Formula: optimal_workers = (worker_queue / max_time) + (arrival_rate * max_time)
    - First term: workers needed to clear current queue
    - Second term: workers needed to handle incoming tasks

    This is an aggressive approach (uses maximum time for safety)

    Compatible with OpenFaaS Autoscaling Gym Environment
    """

    def __init__(self):
        """
        Initialize reactive maximum-based policy
        No hyperparameters - purely queue-based calculation
        """
        # For compatibility with RL agents
        self.name = "Reactive-Maximum"

    def select_action(self, observation, training=True):
        """
        Select action based on queue-driven calculation (maximum time)

        Args:
            observation: [input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate]
            training: Ignored (for compatibility)

        Returns:
            action: 0=-2, 1=-1, 2=0, 3=+1, 4=+2

        Formula:
            optimal_workers = (worker_queue / max_time) + (arrival_rate * max_time)
            - First term: workers to clear current queue
            - Second term: workers to handle incoming rate
        """
        input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate = observation

        # Calculate optimal workers based on queue and arrival rate using MAX time
        if max_time > 0:
            # Workers needed to clear current queue (using max time for safety)
            workers_for_queue = worker_q / max_time if worker_q > 0 else 0

            # Workers needed to handle incoming tasks (using max time for safety)
            workers_for_arrival = arrival_rate * max_time if arrival_rate > 0 else 0

            # Total optimal workers
            optimal_workers = workers_for_queue + workers_for_arrival

            # Ensure at least 1 worker
            optimal_workers = max(1, optimal_workers)
        else:
            # No processing time data: scale down to minimum
            optimal_workers = 1

        # Calculate difference
        delta = optimal_workers - workers

        # Map delta to action
        if delta >= 2.5:
            return 4  # Scale up by 2
        elif delta >= 1.0:
            return 3  # Scale up by 1
        elif delta <= -2.5:
            return 0  # Scale down by 2
        elif delta <= -1.0:
            return 1  # Scale down by 1
        else:
            return 2  # No change

    def get_stats(self):
        """Get policy statistics"""
        return {
            'policy': 'reactive_maximum',
            'approach': 'queue-based with maximum processing time'
        }
