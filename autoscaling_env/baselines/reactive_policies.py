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


def _estimate_active_tasks(
    total_enqueued: float,
    worker_queue: float,
    result_queue: float,
    output_queue: float,
) -> float:
    """Estimate tasks currently being serviced by workers."""

    # total_enqueued tracks everything pushed into the worker queue. Anything still
    # sitting in the worker, result, or output queues has not yet left the system,
    # so we subtract those counts to approximate tasks actively being processed.
    downstream = worker_queue + result_queue + output_queue
    active = max(0.0, total_enqueued - downstream)
    return active


class ReactiveAverage:
    """
    Reactive Average-Based Policy (Baseline 1)

    Queue-based model using AVERAGE processing time:
    - Calculates required workers based on: worker_queue_length, arrival_rate, avg_processing_time
    - Estimates how many workers are needed to clear the current queue AND process
      incoming load during the next control window.

    This is a conservative approach (uses average time)

    Compatible with OpenFaaS Autoscaling Gym Environment
    """

    def __init__(self, horizon_seconds: float = 10.0, safety_factor: float = 1.0):
        """Initialize policy with an optional planning horizon and safety margin."""
        # For compatibility with RL agents
        self.name = "Reactive-Average"
        self.horizon_seconds = max(horizon_seconds, 0.1)
        self.safety_factor = max(safety_factor, 1.0)
        self._enqueued_total = 0.0

    def set_horizon(self, horizon_seconds: float) -> None:
        """Allow callers to update the planning horizon dynamically."""
        if horizon_seconds > 0:
            self.horizon_seconds = horizon_seconds

    def select_action(self, observation, training=True):
        """
        Select action based on queue-driven calculation (average time)

        Args:
            observation: [input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate]
            training: Ignored (for compatibility)

        Returns:
            action: 0=-1, 1=0, 2=+1

        Formula:
            1. Estimate service time with avg_time (lower bound of 100ms).
            2. Determine how many tasks a worker can finish during the next
               action horizon (step_duration).
            3. Combine backlog clearance and expected arrivals, apply optional
               safety factor, and convert to an optimal worker target.
        """
        input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate = observation
        total_enqueued = self._enqueued_total

        service_time = max(avg_time, 0.1)  # seconds per task (avg case)
        horizon = max(self.horizon_seconds, service_time)

        # Tasks a single worker can handle within the horizon
        tasks_per_worker = horizon / service_time if service_time > 0 else float("inf")

        # Workers needed to clear current queue within the horizon
        workers_for_queue = (worker_q / tasks_per_worker) if tasks_per_worker > 0 else 0

        # Workers needed to handle expected incoming tasks during the horizon
        workers_for_arrival = arrival_rate * service_time if arrival_rate > 0 else 0

        # Estimate how many workers are currently busy.
        active_tasks = _estimate_active_tasks(total_enqueued, worker_q, result_q, output_q)
        collector_adjustment = 1.0 if active_tasks > 0 else 0.0
        busy_workers = min(workers, max(0.0, active_tasks - collector_adjustment))

        optimal_workers = (workers_for_queue + workers_for_arrival) * self.safety_factor
        optimal_workers = max(1.0, optimal_workers)

        # Calculate difference relative to currently busy workers rather than the
        # raw replica count so idle capacity doesn't mask backlog pressure.
        delta = optimal_workers - busy_workers

        # Map delta to action
        if delta >= 0.5:
            return 2  # Scale up by 1
        elif delta <= -0.5:
            return 0  # Scale down by 1
        else:
            return 1  # No change

    def get_stats(self):
        """Get policy statistics"""
        return {
            'policy': 'reactive_average',
            'approach': 'queue-based with average processing time'
        }

    def reset(self) -> None:
        self._enqueued_total = 0.0

    def update_from_info(self, info: dict | None) -> None:
        if not info:
            return
        value = info.get("enqueued_total")
        if value is None:
            return
        try:
            self._enqueued_total = max(0.0, float(value))
        except (TypeError, ValueError):
            pass


class ReactiveMaximum:
    """
    Reactive Maximum-Based Policy (Baseline 2)

    Queue-based model using MAXIMUM processing time:
    - Calculates required workers based on: worker_queue_length, arrival_rate, max_processing_time
    - Uses a more conservative service time to protect against QoS violations.

    This is an aggressive approach (uses maximum time for safety)

    Compatible with OpenFaaS Autoscaling Gym Environment
    """

    def __init__(self, horizon_seconds: float = 10.0, safety_factor: float = 1.0):
        """Initialize policy with planning horizon and a safety margin."""
        # For compatibility with RL agents
        self.name = "Reactive-Maximum"
        self.horizon_seconds = max(horizon_seconds, 0.1)
        self.safety_factor = max(safety_factor, 1.0)
        self._enqueued_total = 0.0

    def set_horizon(self, horizon_seconds: float) -> None:
        """Allow callers to update the planning horizon dynamically."""
        if horizon_seconds > 0:
            self.horizon_seconds = horizon_seconds

    def select_action(self, observation, training=True):
        """
        Select action based on queue-driven calculation (maximum time)

        Args:
            observation: [input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate]
            training: Ignored (for compatibility)

        Returns:
            action: 0=-1, 1=0, 2=+1

        Formula mirrors ReactiveAverage, but uses max_time to represent worst-case
        service time, plus a safety factor.
        """
        input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate = observation
        total_enqueued = self._enqueued_total

        service_time = max(max_time, avg_time, 0.1)
        horizon = max(self.horizon_seconds, service_time)

        tasks_per_worker = horizon / service_time if service_time > 0 else float("inf")

        workers_for_queue = (worker_q / tasks_per_worker) if tasks_per_worker > 0 else 0
        workers_for_arrival = arrival_rate * service_time if arrival_rate > 0 else 0

        active_tasks = _estimate_active_tasks(total_enqueued, worker_q, result_q, output_q)
        collector_adjustment = 1.0 if active_tasks > 0 else 0.0
        busy_workers = min(workers, max(0.0, active_tasks - collector_adjustment))

        optimal_workers = (workers_for_queue + workers_for_arrival) * self.safety_factor
        optimal_workers = max(1.0, optimal_workers)

        # Calculate difference relative to currently busy workers.
        delta = optimal_workers - busy_workers

        # Map delta to action
        if delta >= 0.5:
            return 2  # Scale up by 1
        elif delta <= -0.5:
            return 0  # Scale down by 1
        else:
            return 1  # No change

    def get_stats(self):
        """Get policy statistics"""
        return {
            'policy': 'reactive_maximum',
            'approach': 'queue-based with maximum processing time'
        }

    def reset(self) -> None:
        self._enqueued_total = 0.0

    def update_from_info(self, info: dict | None) -> None:
        if not info:
            return
        value = info.get("enqueued_total")
        if value is None:
            return
        try:
            self._enqueued_total = max(0.0, float(value))
        except (TypeError, ValueError):
            pass
