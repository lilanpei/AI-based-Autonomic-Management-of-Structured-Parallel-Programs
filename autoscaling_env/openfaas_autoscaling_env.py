"""
OpenFaaS Autoscaling Gym Environment
Real deployment environment for RL-based autoscaling
"""

import os
import sys
import gymnasium as gym
import json
import time
import subprocess
import numpy as np
from gymnasium import spaces
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from kubernetes import client, config

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utilities.utilities import (
    get_config,
    init_farm,
    initialize_environment,
    clear_queues,
    get_redis_client_with_retry,
    get_deployment_replicas,
    get_utc_now,
    scale_function_deployment,
    deploy_function,
    send_control_messages,
    safe_invoke_function_async,
    run_faas_cli
)

from utilities.logger import get_logger, bind_logger_to_print

logger = get_logger(__name__)
print = bind_logger_to_print(logger)


class OpenFaaSAutoscalingEnv(gym.Env):
    """
    Gym environment for OpenFaaS autoscaling using real deployment
    
    Compatible with:
    - RL agents
    - Reactive baselines (threshold-based policies)
    
    Observation Space:
        - input_queue_length: Number of tasks in input_queue
        - worker_queue_length: Number of tasks in worker_queue
        - result_queue_length: Number of tasks in result_queue
        - output_queue_length: Number of tasks in output_queue
        - current_workers: Number of active worker replicas
        - avg_processing_time: Moving average of task processing times
        - max_processing_time: Maximum processing time in observation window
        - task_arrival_rate: Tasks arriving per second (smoothed)
        - qos_success_rate: Recent QoS success rate

    Action Space:
        - 0: Scale down by 1 (-1)
        - 1: No change (0)
        - 2: Scale up by 1 (+1)

    Reward:
        - Positive: High QoS success rate, low queue length
        - Negative: QoS violations, excessive workers (cost), queue buildup
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 max_workers=32,
                 min_workers=1,
                 observation_window=10,  # seconds
                 step_duration=10,  # seconds between actions
                 max_steps=50,
                 initial_workers=1,
                 initialize_workflow=True,
                 phase_shuffle=False,
                 phase_shuffle_seed=None,
                 task_seed=None):
        """
        Initialize the OpenFaaS autoscaling environment

        Args:
            max_workers: Maximum number of worker replicas
            min_workers: Minimum number of worker replicas
            observation_window: Time window for computing metrics (seconds)
            step_duration: Total time per step including scaling and observation (seconds)
            max_steps: Maximum steps per episode
            initial_workers: Number of workers to deploy initially
            initialize_workflow: If True, deploy emitter/collector/workers (like workflow_controller)
        """
        super(OpenFaaSAutoscalingEnv, self).__init__()

        # Environment parameters
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.observation_window = observation_window
        self.step_duration = step_duration
        self.max_steps = max_steps
        self.initial_workers = initial_workers
        self.initialize_workflow = initialize_workflow
        self.phase_shuffle = phase_shuffle
        self.phase_shuffle_seed = phase_shuffle_seed
        self.default_task_seed = task_seed
        self._pending_task_seed: int | None = None
        self._current_task_seed: int | None = task_seed
        self.redis_client = None

        # Define action/observation spaces immediately so agents can query bounds
        self.action_space = spaces.Discrete(3)
        self.action_map = {0: -1, 1: 0, 2: +1}
        # Bounds derived from recent training history (runs/sarsa_run_20251104_012016)
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,  # input_queue_length
                0.0,  # worker_queue_length
                0.0,  # result_queue_length
                0.0,  # output_queue_length
                float(self.min_workers),
                0.0,  # avg_processing_time
                0.0,  # max_processing_time
                0.0,  # arrival_rate
                0.0,  # qos_success_rate
            ], dtype=np.float32),
            high=np.array([
                3.0,    # input_queue_length (observed max ~3)
                800.0,  # worker_queue_length (observed max ~556)
                10.0,   # result_queue_length (observed max ~3)
                1140.0, # output_queue_length (observed max ~1140)
                float(self.max_workers),
                150.0,  # avg_processing_time (observed max ~143)
                160.0,  # max_processing_time (observed max ~146)
                12.0,   # arrival_rate (observed max ~10.3)
                1.0,    # qos_success_rate
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize workflow if requested (ensures emitter/collector have new program_start_time)
        if self.initialize_workflow:
            print("\n" + "="*70)
            print("INITIALIZING OPENFAAS WORKFLOW")
            print("="*70)
            # Use the same episode initialization logic as reset()
            self._initialize_episode()
            print("="*70 + "\n")

    def _reset_episode_state(self):
        """
        Reset episode-specific state variables.
        Called by both __init__() and reset() to ensure consistency.
        """
        # Reset State tracking
        self.workflow_initialized = False

        # Initialize episode state (will be reset in reset())
        self.redis_client = None

        # Reset step counter
        self.current_step = 0
        self.episode_start_time = time.time()

        # Reset program start time for new episode
        self.program_start_time = get_utc_now()
        print(f"[INFO] Set program_start_time to: {self.program_start_time}")

        # Determine seed to use for this episode
        if self._pending_task_seed is not None:
            self._current_task_seed = int(self._pending_task_seed)
        else:
            self._current_task_seed = int(self.default_task_seed) if self.default_task_seed is not None else None
        self._pending_task_seed = None

        # Reinitialize config with new program_start_time
        self.config = initialize_environment(self.program_start_time)

        # Reset Metrics tracking
        self.task_history = []  # (timestamp, processing_time, qos_success)
        self.queue_history = []  # (timestamp, queue_length)
        self.worker_history = []  # (timestamp, num_workers)
        self.enqueue_history = []  # (timestamp, total_enqueued)
        self.arrival_history = []  # (timestamp, arrival_rate)
        self.worker_enqueue_total = 0
        self._last_arrival_sample: tuple[float, float] | None = None
        self.processed_task_ids = set()  # Track which tasks we've already processed

        # Reset Episode statistics
        self.total_reward = 0
        self.total_qos_violations = 0
        self.total_scaling_actions = 0
        self._episode_final_qos = None

        # Reset task generation timing markers
        self.task_generation_start_time = 0.0
        self.task_generation_end_time = 0.0

        # Kubernetes clients
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        self.core_v1_api = client.CoreV1Api()
        self.apps_v1_api = client.AppsV1Api()


    def _initialize_episode(self):
        """
        Initialize/reset episode with consistent timing logic.
        Used by both __init__() and reset() to ensure identical behavior.
        """

        # Reset all episode-specific state
        self._reset_episode_state()

        # Reinitialize farm if requested (ensures emitter/collector have new program_start_time)

        print("[INFO] Initializing farm with program_start_time...")
        print("[INFO] This will take ~30-60 seconds...")
        self._initialize_farm()
        print("[INFO] ✓ Farm initialized")

        # Ensure Redis client is connected
        if self.redis_client is None:
            print("[INFO] Initializing Redis client...")
            self.redis_client = get_redis_client_with_retry()

        # Clear queues for fresh episode
        print("[INFO] Clearing queues for new episode...")
        clear_queues(self.redis_client, None)

        # Reset workers to initial state (clean slate for new episode)
        current_workers = get_deployment_replicas(
            self.apps_v1_api,
            namespace="openfaas-fn",
            name_or_prefix="worker-",
            exact_match=False
        )

        if current_workers != self.initial_workers:
            print(f"[INFO] Resetting workers from {current_workers} to {self.initial_workers}...")
            delta = self.initial_workers - current_workers
            if delta > 0:
                self._scale_up(current_workers, delta)
            else:
                self._scale_down(current_workers, abs(delta))

            # Verify
            current_workers = get_deployment_replicas(
                self.apps_v1_api,
                namespace="openfaas-fn",
                name_or_prefix="worker-",
                exact_match=False
            )
            print(f"[INFO] Workers reset to: {current_workers}")

        # Generate tasks for this episode (Step 3)
        self.task_generation_start_time = (get_utc_now()-self.program_start_time).total_seconds()
        print(f"[INFO] Generating tasks for episode at {self.task_generation_start_time:.2f} seconds after program start")
        self._generate_tasks(seed=self._current_task_seed)
        self.task_generation_end_time = (get_utc_now()-self.program_start_time).total_seconds()

        print(f"[INFO] Initial workers: {current_workers}")
        print(f"[INFO] Program start time: {self.program_start_time}")

        # Get initial observation
        observation = self._get_observation()

        print(f"[INFO] Initial observation: {observation}")

        return observation

    def reset(self, seed: int | None = None):
        """
        Reset environment to initial state for new episode.
        Uses the same logic as __init__() to ensure consistent timing.
        """
        if seed is not None:
            self._pending_task_seed = seed

        print("\n" + "="*70)
        print("RESETTING ENVIRONMENT FOR NEW EPISODE")
        print("="*70)

        # Use shared episode initialization logic
        observation = self._initialize_episode()

        print("="*70 + "\n")

        return observation

    def step(self, action):
        """
        Execute one scaling action and return new state
        
        Args:
            action: Integer 0-2 representing scaling action
            
        Returns:
            observation: New state
            reward: Reward for this action
            done: Whether episode is complete
            info: Additional information
        """
        self.current_step += 1
        step_start_time = time.time()

        # Map action to requested scaling delta
        requested_delta = int(self.action_map[action])
        applied_delta = 0

        print(f"\n{'='*70}")
        print(f"STEP {self.current_step}/{self.max_steps}")
        print(f"{'='*70}")
        print(f"[ACTION] Requested scaling action: {requested_delta:+d}")

        # Get current state before action
        queue_before = self.redis_client.llen(self.config.get("worker_queue_name"))
        workers_before = get_deployment_replicas(
            self.apps_v1_api,
            namespace="openfaas-fn",
            name_or_prefix="worker-",
            exact_match=False
        )

        print(f"[STATE] Queue: {queue_before}, Workers: {workers_before}")

        # Execute scaling action and measure time
        scaling_start = time.time()
        if requested_delta != 0:
            applied_delta = self._execute_scaling_action(requested_delta, workers_before)
            if applied_delta != 0:
                self.total_scaling_actions += 1
        else:
            print("[INFO] No scaling action (delta=0)")
            applied_delta = 0

        if applied_delta != requested_delta:
            print(f"[ACTION] Effective delta after bounds: {applied_delta:+d}")

        delta = applied_delta
        scaling_time = time.time() - scaling_start

        # Wait for remaining step duration to observe effects
        # step_duration is the TOTAL time for the step (scaling + observation)
        remaining_time = max(0, self.step_duration - scaling_time - 0.2) # 0.2s for observation
        if remaining_time > 0:
            print(f"[INFO] Scaling took {scaling_time:.2f}s, waiting {remaining_time:.2f}s more (total step: {self.step_duration}s)...")
            time.sleep(remaining_time)
        else:
            print(f"[INFO] Scaling took {scaling_time:.2f}s (exceeds step_duration of {self.step_duration}s)")


        # Collect metrics during observation window
        self._collect_metrics()

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(observation, delta)

        # Check if episode is done
        # Episode ends when: (1) max steps reached OR (2) all tasks completed
        done = self._check_episode_done(observation)

        reward_cfg = self.config.get("reward", {})
        unfinished_penalty_scale = float(reward_cfg.get("unfinished_penalty_scale"))
        completion_bonus = float(reward_cfg.get("completion_bonus"))
        queue_target = float(reward_cfg.get("queue_target"))

        expected_tasks = self._get_expected_task_count()

        # If episode is done, optionally adjust reward and wait for tasks to complete
        unfinished_tasks = 0
        if done:
            pending_tasks = max(0, int(expected_tasks - observation[3]))
            unfinished_tasks = pending_tasks
            if self.current_step >= self.max_steps:
                if unfinished_penalty_scale > 0 and pending_tasks > 0:
                    pending_capped = min(pending_tasks, queue_target)
                    penalty = unfinished_penalty_scale * pending_capped
                    reward -= penalty
                    print(
                        f"[REWARD] Applied unfinished-task penalty: -{penalty:.2f} "
                        f"({pending_capped} capped tasks × {unfinished_penalty_scale:.2f})"
                    )
            elif completion_bonus > 0:
                reward += completion_bonus
                print(f"[REWARD] Applied early completion bonus: +{completion_bonus:.2f}")

            if self.current_step >= self.max_steps:
                print(f"[INFO] Episode finished: Max steps ({self.max_steps}) reached")
            else:
                print(f"[INFO] Episode finished: All tasks completed at step {self.current_step}")
            print(f"[INFO] Waiting for all tasks to complete...")
            self._wait_for_task_completion()

            self._episode_final_qos = self._calculate_episode_qos()
            print(f"[METRICS] Episode final QoS over processed tasks: {self._episode_final_qos:.4f}")
        else:
            self._episode_final_qos = None

        self.total_reward += reward

        step_duration = time.time() - step_start_time

        # Additional info
        episode_final_qos = self._episode_final_qos if done else None
        info = {
            'step': self.current_step,
            'queue_length': observation[1],  # worker_queue
            'workers': observation[4],
            'qos_rate': observation[8],
            'requested_delta': requested_delta,
            'applied_delta': delta,
            'total_reward': self.total_reward,
            'scaling_actions': self.total_scaling_actions,
            'qos_violations': self.total_qos_violations,
            'scaling_time': scaling_time,
            'step_duration': step_duration,
            'program_start_time': self.program_start_time,
            'task_generation_start_time': self.task_generation_start_time,
            'task_generation_end_time': self.task_generation_end_time,
            'episode_final_qos': episode_final_qos,
            'processed_tasks': len(self.task_history),
            'unfinished_tasks': unfinished_tasks,
            'enqueued_total': self.worker_enqueue_total,
        }

        print(f"[REWARD] {reward:.2f} | Total: {self.total_reward:.2f}")
        print(f"[INFO] Step completed in {step_duration:.2f}s")
        print(f"{'='*70}\n")

        return observation, reward, done, info

    def _check_episode_done(self, observation):
        """
        Check if episode should terminate

        Episode ends when:
        1. Max steps reached, OR
        2. All tasks completed (all queues empty + all tasks in output_queue)

        Args:
            observation: Current observation vector

        Returns:
            bool: True if episode should end
        """
        # Check max steps
        if self.current_step >= self.max_steps:
            return True

        # Check if all tasks completed
        # observation = [input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate]
        input_queue = observation[0]
        worker_queue = observation[1]
        result_queue = observation[2]
        output_queue = observation[3]

        # Expected total tasks (from config)
        expected_tasks = self._get_expected_task_count()

        # All tasks completed if:
        # 1. All processing queues are empty (input, worker, result)
        # 2. Output queue has all expected tasks
        all_queues_empty = (input_queue == 0 and worker_queue == 0 and result_queue == 0)
        all_tasks_in_output = (output_queue >= expected_tasks)

        if all_queues_empty and all_tasks_in_output:
            print(f"[INFO] All tasks completed: {int(output_queue)}/{expected_tasks} tasks in output_queue")
            return True

        return False

    def _get_expected_task_count(self):
        """
        Calculate expected number of tasks based on config

        Returns:
            int: Expected total tasks
        """
        base_rate = self.config.get("base_rate")  # tasks/min
        configured_phases = self.config.get("phase_definitions")

        if not configured_phases:
            raise ValueError("phase_definitions missing from configuration.yml")

        target_tasks = 0
        for _ , phase_cfg in enumerate(configured_phases):
            multiplier = phase_cfg.get("phase_multiplier")
            duration = int(phase_cfg.get("phase_duration"))
            target_tasks += int(base_rate * multiplier * (duration / 60))

        return target_tasks

    def _execute_scaling_action(self, delta, current_workers):
        """Execute scaling action (scale up or down) and return applied delta."""
        # Convert to native Python int (avoid numpy types for K8s API)
        current_workers = int(current_workers)
        delta = int(delta)

        new_workers = int(np.clip(current_workers + delta, self.min_workers, self.max_workers))
        actual_delta = int(new_workers - current_workers)

        if actual_delta == 0:
            print(f"[INFO] Scaling capped at boundaries (min={self.min_workers}, max={self.max_workers})")
            return 0

        print(f"[SCALING] {current_workers} → {new_workers} (delta: {actual_delta:+d})")

        try:
            if actual_delta > 0:
                self._scale_up(current_workers, actual_delta)
            else:
                self._scale_down(current_workers, abs(actual_delta))
            return actual_delta
        except Exception as e:
            print(f"[ERROR] Scaling failed: {e}")
            import traceback
            traceback.print_exc()
            return 0
    def _scale_up(self, current, delta):
        """
        Scale up workers (matches worker_scaler.py logic)

        Steps:
        1. Deploy new worker function instances
        2. Scale queue-worker deployment
        3. Invoke new workers with payload
        """
        new_replicas = current + delta
        print(f"[SCALE_UP] Scaling up from {current} to {new_replicas} replicas...")
        print(f"[TIMER] Scale up started at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds")

        # Step 1: Deploy new worker instances
        deployed_workers = deploy_function(
            function_name_prefix="worker",
            replicas=delta,
            max_retries=3,
            delay=1
        )
        print(f"[TIMER] Deployed {delta} new worker instances at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds")

        # Step 2: Scale queue-worker deployment to match new replicas
        current_queue_worker_replicas = get_deployment_replicas(
            self.apps_v1_api,
            namespace="openfaas",
            name_or_prefix="queue-worker",
            exact_match=True
        )
        print(f"[INFO] Current Queue Worker Replicas: {current_queue_worker_replicas}")

        scale_function_deployment(
            current_queue_worker_replicas + delta,
            self.apps_v1_api,
            deployment_name="queue-worker",
            namespace="openfaas"
        )
        print(f"[TIMER] Scaled queue worker deployment to {current_queue_worker_replicas + delta} replicas at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds")

        # Step 3: Invoke new workers with payload
        payload = self._get_worker_payload()
        safe_invoke_function_async(
            deployed_workers,
            payload,
            self.redis_client,
            self.config.get("worker_start_queue_name"),
            delta,
            timeout=self.config.get("function_invoke_timeout"),
            retries=self.config.get("function_invoke_retries")
        )
        print(f"[TIMER] Finished invoking {delta} worker functions at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds")
        print(f"[SCALE_UP] ✓ Successfully scaled up to {new_replicas} workers")

    def _scale_down(self, current, delta):
        """
        Scale down workers using control messages and ACKs (matches worker_scaler.py logic)

        Steps:
        1. Send SCALE_DOWN control messages
        2. Wait for ACKs from workers
        3. Delete functions for ACKed pods
        4. Wait for pod termination
        """

        control_syn_q = self.config.get("worker_control_syn_queue_name")
        control_ack_q = self.config.get("worker_control_ack_queue_name")
        new_replicas = max(current - delta, 1)
        count = current - new_replicas

        print(f"[SCALE_DOWN] Scaling down from {current} to {new_replicas} replicas...")
        print(f"[TIMER] Scale down started at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds")
        print(f"[TIMER] Sending {count} control requests at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds...")

        # Step 1: Send SCALE_DOWN control messages
        message = {
            "type": "SCALE_DOWN",
            "action": "SYN",
            "message": "Scale down request from RL agent",
            "SYN_timestamp": (get_utc_now() - self.program_start_time).total_seconds(),
        }
        send_control_messages(message, self.redis_client, control_syn_q, count)
        print(f"[TIMER] Sent {count} control messages at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds")

        # Step 2: Wait for ACKs
        acked_pods = []
        timeout = self.config.get("scale_down_timeout", 1)
        retries = self.config.get("scale_down_retries", 30)
        attempts = 0

        while len(acked_pods) < count:
            print(f"[INFO] Waiting for ACKs... {len(acked_pods)}/{count}")
            msg_raw = self.redis_client.rpop(control_ack_q)
            if not msg_raw:
                attempts += 1
                print(f"[INFO] No ACK received for {attempts} tries, waiting {timeout} seconds...")
                time.sleep(timeout)
                if attempts >= retries:
                    print("[WARNING] No ACKs received for a long time, exiting scale down.")
                    print(f"[INFO] Current worker replicas: {get_deployment_replicas(self.apps_v1_api, namespace='openfaas-fn', name_or_prefix='worker-', exact_match=False)}")
                    return
                continue

            try:
                msg = json.loads(msg_raw)
                # Check if the message is an ACK for SCALE_DOWN
                if msg.get("type") == "SCALE_DOWN" and msg.get("action") == "ACK":
                    pod_name = msg.get("pod_name")
                    acked_pods.append(pod_name)
                    print(f"[INFO] ACK received from pod: {pod_name}")
            except Exception as e:
                print(f"[WARNING] Malformed ACK message: {e}")

        print(f"[TIMER] Received ACKs for {len(acked_pods)} pods at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds")

        # Step 3: Delete functions based on ACKed pod names
        print("[INFO] Deleting functions for ACKed pods...")
        for pod in acked_pods:
            try:
                result = subprocess.run([
                    "kubectl", "get", "pod", pod, "-n", "openfaas-fn",
                    "-o", "jsonpath={.metadata.labels.faas_function}"
                ], capture_output=True, text=True, check=True)
                function_name = result.stdout.strip()
                if not function_name:
                    print(f"[WARNING] No faas_function label found for pod {pod}, skipping deletion.")
                    continue

                print(f"[INFO] Removing function: {function_name}")
                run_faas_cli(["remove", function_name])
                print(f"[TIMER] Finished removing function: {function_name} at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds")

                # Step 4: Wait for pod termination
                for attempt in range(retries):
                    pod_list = self.core_v1_api.list_namespaced_pod(
                        namespace="openfaas-fn",
                        label_selector=f"faas_function={function_name}"
                    ).items
                    if not pod_list:
                        print(f"[TIMER] Function pod for '{function_name}' has terminated at {(get_utc_now() - self.program_start_time).total_seconds():.4f} seconds")
                        break
                    print(f"[INFO] Waiting for pod of '{function_name}' to terminate (Attempt {attempt+1})...")
                    time.sleep(timeout)
                else:
                    print(f"[WARNING] Function pod for '{function_name}' still exists after timeout.")

            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to fetch function name for pod {pod}: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error while deleting function for pod {pod}: {e}")

        print(f"[SCALE_DOWN] ✓ Successfully scaled down to {new_replicas} workers")

    def _collect_metrics(self):
        """Collect metrics from Redis queues and completed tasks"""
        current_time = time.time()

        # Get queue length
        queue_length = self.redis_client.llen(self.config.get("worker_queue_name"))
        self.queue_history.append((current_time, queue_length))

        # Get worker count
        workers = get_deployment_replicas(
            self.apps_v1_api,
            namespace="openfaas-fn",
            name_or_prefix="worker-",
            exact_match=False
        )
        self.worker_history.append((current_time, workers))

        # Mirror worker enqueue counter
        worker_queue = self.config.get("worker_queue_name")
        counter_key = f"{worker_queue}:enqueued_total"
        current_total = self.worker_enqueue_total or 0.0
        try:
            counter_value = self.redis_client.get(counter_key)
            if counter_value is not None:
                current_total = max(0.0, float(counter_value))
        except Exception as exc:  # noqa: BLE001 - best-effort metric pull
            print(f"[WARNING] Failed to read enqueue counter '{counter_key}': {exc}")

        if current_total < self.worker_enqueue_total:
            # Counter was reset externally; treat as new baseline.
            self.worker_enqueue_total = current_total
        else:
            self.worker_enqueue_total = current_total

        self.enqueue_history.append((current_time, self.worker_enqueue_total))

        # Get completed tasks from output queue (sample recent tasks)
        output_queue = self.config.get("output_queue_name")
        queue_len = self.redis_client.llen(output_queue)

        if queue_len > 0:
            # Process all tasks in queue using efficient batch operation
            # LRANGE is much faster than multiple LINDEX calls
            new_tasks_found = 0

            # Fetch all tasks in one batch operation (much faster than LINDEX loop)
            # LRANGE returns entire list in one network call
            all_tasks = self.redis_client.lrange(output_queue, 0, -1)

            print(f"[METRICS] Fetched {len(all_tasks)} tasks from output_queue")

            # Process all tasks
            for task_raw in all_tasks:
                if task_raw:
                    try:
                        task = json.loads(task_raw)
                        task_id = task.get("task_id")

                        # Only process if we haven't seen this task before
                        if task_id and task_id not in self.processed_task_ids:
                            processing_time = task.get("task_completion_time", 0)
                            qos_success = task.get("task_QoS", False)

                            # Use actual task completion timestamp, not collection time
                            # This ensures metrics reflect when tasks actually completed
                            task_timestamp = task.get("task_collect_timestamp")
                            if task_timestamp is not None:
                                # Convert relative timestamp to absolute time
                                task_completion_time = self.program_start_time.timestamp() + task_timestamp
                            else:
                                # Fallback to current time if timestamp not available
                                task_completion_time = current_time
                                if new_tasks_found == 0:  # Only print once per step
                                    print(f"[WARNING] Task {task_id[:8]} missing task_collect_timestamp, using current_time")

                            self.task_history.append((task_completion_time, processing_time, qos_success))
                            self.processed_task_ids.add(task_id)
                            new_tasks_found += 1

                            if not qos_success:
                                self.total_qos_violations += 1
                    except:
                        pass

            if new_tasks_found > 0:
                # Show timestamp range of collected tasks
                recent_timestamps = [t[0] for t in self.task_history[-new_tasks_found:]]
                if recent_timestamps:
                    min_ts = min(recent_timestamps)
                    max_ts = max(recent_timestamps)
                    print(f"[METRICS] Collected {new_tasks_found} new tasks (timestamps: {min_ts:.1f} - {max_ts:.1f})")

    def _calculate_episode_qos(self) -> float:
        total_tasks = len(self.task_history)
        if total_tasks == 0:
            return 1.0
        successful_tasks = total_tasks - self.total_qos_violations
        return float(successful_tasks) / float(total_tasks)

    def get_episode_final_qos(self) -> float | None:
        """Return the QoS success rate for all tasks seen this episode."""
        if self._episode_final_qos is not None:
            return self._episode_final_qos
        if self.task_history:
            return self._calculate_episode_qos()
        return None

    def _get_observation(self):
        """Get current observation vector"""
        current_time = time.time()
        window_start = current_time - self.observation_window

        # All queue lengths (current)
        input_queue_length = self.redis_client.llen(self.config.get("input_queue_name"))
        worker_queue_length = self.redis_client.llen(self.config.get("worker_queue_name"))
        result_queue_length = self.redis_client.llen(self.config.get("result_queue_name"))
        output_queue_length = self.redis_client.llen(self.config.get("output_queue_name"))

        # Current workers
        workers = get_deployment_replicas(
            self.apps_v1_api,
            namespace="openfaas-fn",
            name_or_prefix="worker-",
            exact_match=False
        )

        # What we can track accurately:
        # - worker_queue_length: Tasks WAITING for workers (not being processed)
        # - len(task_history): Tasks that have been COMPLETED and collected

        # Estimate idle workers (rough approximation)
        # If worker_queue is empty, workers are likely idle
        # If worker_queue has tasks, some workers are likely processing
        completed_tasks = len(self.task_history)

        print(f"[QUEUES] Input: {input_queue_length}, Worker: {worker_queue_length}, "
              f"Result: {result_queue_length}, Output: {output_queue_length}")
        print(f"[PROGRESS] Workers: {workers}, Completed: {completed_tasks}, "
              f"Waiting: {worker_queue_length}")

        # Processing time metrics (from recent tasks)
        recent_tasks = [t for t in self.task_history if t[0] >= window_start]

        if recent_tasks:
            avg_processing_time = np.mean([t[1] for t in recent_tasks])
            max_processing_time = np.max([t[1] for t in recent_tasks])

            # Debug: show window details
            task_timestamps = [t[0] for t in recent_tasks]
            print(f"[OBSERVATION] Window: [{window_start:.1f}, {current_time:.1f}]")
            print(f"[OBSERVATION] Using {len(recent_tasks)} tasks (total: {len(self.task_history)})")
            print(f"[OBSERVATION] Task timestamp range: [{min(task_timestamps):.1f}, {max(task_timestamps):.1f}]")
            print(f"[OBSERVATION] Avg time: {avg_processing_time:.3f}s, Max time: {max_processing_time:.3f}s")
        else:
            # Default values when no task data available
            avg_processing_time = 1.5
            max_processing_time = 1.5
            print(f"[WARNING] No recent tasks in observation window (total history: {len(self.task_history)})")

        arrival_rate = 0.0
        worker_queue = self.config.get("worker_queue_name")
        total_enqueued = self.worker_enqueue_total
        self.arrival_history.append((current_time, total_enqueued))
        if self._last_arrival_sample is not None:
            last_time, last_total = self._last_arrival_sample
            elapsed = current_time - last_time
            delta_tasks = max(0.0, total_enqueued - last_total)
            if elapsed > 0:
                arrival_rate = delta_tasks / elapsed
        self._last_arrival_sample = (current_time, total_enqueued)

        # QoS success rate (from recent tasks)
        if recent_tasks:
            qos_rate = np.mean([1.0 if t[2] else 0.0 for t in recent_tasks])
        else:
            qos_rate = 1.0  # Assume good if no data

        observation = np.array([
            input_queue_length,
            worker_queue_length,
            result_queue_length,
            output_queue_length,
            workers,
            avg_processing_time,
            max_processing_time,
            arrival_rate,
            qos_rate
        ], dtype=np.float32)

        return observation

    def _calculate_reward(self, observation, delta):
        """Calculate reward for the current state/action pair.

        The reward balances four signals using configuration keys under ``reward``:

        - ``target_qos``: desired QoS success rate (default 0.95)
        - ``queue_target``: worker-queue length considered healthy
        - ``balanced_workers``: replica count treated as cost-neutral when queues are light
        - ``idle_queue_threshold``: queue length regarded as idle for efficiency checks
        - ``weights``: dict with keys ``qos``, ``backlog``, ``scaling``, ``efficiency``, and
          ``scale`` (with sub-keys ``up`` and ``down``)
        """

        # Unpack observation vector
        input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate = observation

        # Load explicit tuning knobs from configuration. Defaulting to zero keeps
        # experimentation easy—unused signals simply drop out of the sum.
        reward_cfg = self.config.get("reward", {})
        weights = reward_cfg.get("weights", {})

        # Primary weights for the four base signals.
        qos_weight = float(weights.get("qos"))
        backlog_weight = float(weights.get("backlog"))
        scaling_weight = float(weights.get("scaling"))
        efficiency_weight = float(weights.get("efficiency"))
        # Directional scale incentives are grouped in a nested dict.
        scale_weights = weights.get("scale")
        scale_up_weight = float(scale_weights.get("up"))
        scale_down_weight = float(scale_weights.get("down"))

        # Shared thresholds/targets for interpreting the observation.
        target_qos = float(reward_cfg.get("target_qos"))
        queue_target = max(1.0, float(reward_cfg.get("queue_target")))
        balanced_workers = float(reward_cfg.get("balanced_workers", self.initial_workers))
        idle_queue_threshold = float(reward_cfg.get("idle_queue_threshold", queue_target * 0.25))

        # QoS signal: encourage hitting the target rate, penalise shortfalls.
        qos_term = qos_weight * (qos_rate - target_qos)

        backlog_ratio = worker_q / queue_target
        backlog_pressure = max(0.0, backlog_ratio - 1.0)
        # Quadratic backlog penalty so extreme queues dominate the negative signal.
        backlog_penalty = backlog_weight * (backlog_pressure ** 2)

        efficiency_penalty = 0.0
        if worker_q <= idle_queue_threshold and qos_rate >= target_qos:
            # Only penalise surplus replicas once the system is already calm.
            worker_surplus = max(0.0, workers - balanced_workers)
            efficiency_penalty = efficiency_weight * worker_surplus

        # Any scaling movement carries a small generic cost to discourage thrashing.
        scaling_penalty = scaling_weight * abs(delta)

        overloaded = backlog_ratio > 1.0 or qos_rate < target_qos
        scale_bonus = 0.0
        if delta > 0 and overloaded:
            # Reward scale-ups only when queues/QoS signal stress.
            scale_bonus += scale_up_weight * abs(delta)
        elif delta < 0 and not overloaded:
            # Reward trimming replicas when the system is comfortably within targets.
            scale_bonus += scale_down_weight * abs(delta)

        stability_bonus = 0.0
        if qos_rate >= target_qos and backlog_ratio <= 0.5:
            # Reinvest a fraction of the QoS weight to reinforce steady healthy states.
            stability_bonus = qos_weight * 0.1

        reward = qos_term - backlog_penalty - efficiency_penalty - scaling_penalty + scale_bonus + stability_bonus

        return reward

    def _initialize_farm(self):
        """
        Initialize OpenFaaS farm (Step 2 from workflow_controller.py)
        Deploys emitter, collector, and initial workers
        """

        # Prepare payload
        payload = self._get_worker_payload()

        # Initialize farm (deploys emitter, collector, workers)
        print(f"[INFO] Deploying emitter, collector, and {self.initial_workers} workers...")
        self.redis_client = init_farm(
            self.program_start_time,
            self.config,
            self.initial_workers,
            payload,
            self.apps_v1_api,
            self.core_v1_api
        )

        self.workflow_initialized = True
        print(f"[INFO] ✓ Workflow initialized successfully")

    def _generate_tasks(self, seed: int | None = None):
        """
        Generate tasks for episode (Step 3 from workflow_controller.py)
        Runs task_generator.py to push phase configs to input_queue
        """

        # Get path to task_generator.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        task_generator_path = os.path.join(project_root, "image_processing", "task_generator.py")

        if not os.path.exists(task_generator_path):
            print(f"[WARNING] task_generator.py not found at {task_generator_path}")
            print(f"[WARNING] Skipping task generation")
            return

        # Get task generation parameters from config
        base_rate = self.config.get("base_rate")
        phase_definitions = self.config.get("phase_definitions")
        total_generation_time = sum(int(phase.get("phase_duration", 0)) for phase in phase_definitions)

        print(f"[INFO] Generating tasks (base_rate={base_rate}, total_generation_time={total_generation_time})...")

        # Set environment variable for task_generator.py
        env = os.environ.copy()
        env["START_TIMESTAMP"] = str(self.program_start_time)
        if getattr(self, "phase_shuffle", False):
            env["PHASE_SHUFFLE"] = "true"
            if getattr(self, "phase_shuffle_seed", None) is not None:
                env["PHASE_SHUFFLE_SEED"] = str(self.phase_shuffle_seed)
        else:
            env["PHASE_SHUFFLE"] = "false"

        if seed is not None:
            env["TASK_GENERATOR_SEED"] = str(seed)

        # Calculate timeout: total_generation_time + buffer for processing
        # More robust for any phase_duration
        buffer = max(60, int(total_generation_time * 0.25))  # 25% buffer, min 60s
        timeout = total_generation_time + buffer
        print(f"[INFO] Task generation timeout: {timeout}s (generation: {total_generation_time}s + buffer: {buffer}s)")

        try:
            result = subprocess.run(
                ["python3", task_generator_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env  # Pass environment with START_TIMESTAMP
            )
            if result.returncode == 0:
                print(f"[INFO] ✓ Tasks generated successfully")
                if result.stdout:
                    print(f"[DEBUG] {result.stdout.strip()}")
            else:
                print(f"[WARNING] Task generation failed: {result.stderr.strip()}")
        except Exception as e:
            print(f"[WARNING] Task generation error: {e}")

    def _get_worker_payload(self):
        """Get payload for worker invocation"""
        return {
            "input_queue_name": self.config.get("input_queue_name"),
            "worker_queue_name": self.config.get("worker_queue_name"),
            "result_queue_name": self.config.get("result_queue_name"),
            "output_queue_name": self.config.get("output_queue_name"),
            "emitter_control_syn_queue_name": self.config.get("emitter_control_syn_queue_name"),
            "worker_control_syn_queue_name": self.config.get("worker_control_syn_queue_name"),
            "worker_control_ack_queue_name": self.config.get("worker_control_ack_queue_name"),
            "collector_control_syn_queue_name": self.config.get("collector_control_syn_queue_name"),
            "emitter_start_queue_name": self.config.get("emitter_start_queue_name"),
            "worker_start_queue_name": self.config.get("worker_start_queue_name"),
            "collector_start_queue_name": self.config.get("collector_start_queue_name"),
            "processing_delay": self.config.get("processing_delay"),
            "wait_time": self.config.get("wait_time"),
            "deadline_coeff": self.config.get("deadline_coeff"),
            "deadline_cap": self.config.get("deadline_cap"),
            "deadline_floor": self.config.get("deadline_floor"),
            "program_start_time": str(self.program_start_time),
            "collector_feedback_flag": False,
            "calibrated_model_a": self.config.get("calibrated_model")["a"],
            "calibrated_model_b": self.config.get("calibrated_model")["b"],
            "calibrated_model_seed": self.config.get("calibrated_model")["seed"],
            "calibrated_model_r_squared": self.config.get("calibrated_model")["r_squared"],
            "base_rate": self.config.get("base_rate"),
            "task_seed": self._current_task_seed,
        }

    def _wait_for_task_completion(self, timeout=8, check_interval=5):
        """
        Wait for all tasks to complete processing

        Args:
            timeout: Maximum time to wait (seconds)
            check_interval: Time between checks (seconds)
        """
        start_time = time.time()
        input_queue = self.config.get("input_queue_name")
        worker_queue = self.config.get("worker_queue_name")
        result_queue = self.config.get("result_queue_name")

        print(f"[INFO] Waiting for queues to drain...")

        while time.time() - start_time < timeout:
            # Keep collecting metrics so completed tasks end up in task_history
            self._collect_metrics()

            # Check all processing queues
            input_len = self.redis_client.llen(input_queue)
            worker_len = self.redis_client.llen(worker_queue)
            result_len = self.redis_client.llen(result_queue)

            total_pending = input_len + worker_len + result_len

            if total_pending == 0:
                # Final sweep to capture any remaining completed tasks
                self._collect_metrics()
                elapsed = time.time() - start_time
                print(f"[INFO] ✓ All tasks completed after {elapsed:.1f}s")
                return

            print(f"[INFO] Pending tasks: input={input_len}, worker={worker_len}, result={result_len} (total={total_pending})")
            time.sleep(check_interval)

        # Timeout reached
        elapsed = time.time() - start_time
        # One last collection attempt before reporting timeout
        self._collect_metrics()
        input_len = self.redis_client.llen(input_queue)
        worker_len = self.redis_client.llen(worker_queue)
        result_len = self.redis_client.llen(result_queue)
        print(f"[WARNING] Timeout after {elapsed:.1f}s with {input_len + worker_len + result_len} tasks still pending")

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            obs = self._get_observation()
            print(f"\n{'='*50}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Input Queue: {obs[0]:.0f}")
            print(f"Worker Queue: {obs[1]:.0f}")
            print(f"Result Queue: {obs[2]:.0f}")
            print(f"Output Queue: {obs[3]:.0f}")
            print(f"Workers: {obs[4]:.0f}")
            print(f"Avg Processing Time: {obs[5]:.2f}s")
            print(f"Max Processing Time: {obs[6]:.2f}s")
            print(f"Arrival Rate: {obs[7]:.2f} tasks/s")
            print(f"QoS Success Rate: {obs[8]:.2%}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"{'='*50}\n")

    def close(self):
        """Clean up resources and send termination signals"""
        print("[INFO] Closing environment and sending termination signals...")

        # Send terminate control messages to all components
        redis_client = getattr(self, "redis_client", None)
        if not redis_client:
            print("[INFO] No Redis client available; skipping termination messages.")
        else:
            try:
                end_time = (get_utc_now() - self.program_start_time).total_seconds()

                message = {
                    "type": "TERMINATE",
                    "action": "SYN",
                    "message": "Terminate function from RL environment",
                    "SYN_timestamp": end_time
                }

                # Get current worker count for sending messages
                current_workers = get_deployment_replicas(
                    self.apps_v1_api,
                    namespace="openfaas-fn",
                    name_or_prefix="worker-",
                    exact_match=False
                )

                # Send termination messages to all components
                send_control_messages(message, redis_client, self.config.get("emitter_control_syn_queue_name"), 1)
                send_control_messages(message, redis_client, self.config.get("worker_control_syn_queue_name"), current_workers)
                send_control_messages(message, redis_client, self.config.get("collector_control_syn_queue_name"), 1)

                print("[INFO] Termination control messages sent to all components.")

                # Close Redis connection
                redis_client.close()
                print("[INFO] Redis connection closed.")

            except Exception as e:
                print(f"[WARNING] Error during cleanup: {e}")
                if redis_client:
                    redis_client.close()

        print("[INFO] Environment closed")
