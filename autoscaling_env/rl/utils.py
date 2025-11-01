"""Helper utilities for RL experiments with the OpenFaaS autoscaling environment."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .sarsa_agent import DiscretizationConfig


def build_discretization_config(
    observation_low: Iterable[float],
    observation_high: Iterable[float],
    bins_per_dimension: Sequence[int],
) -> DiscretizationConfig:
    """Convenience wrapper to create a :class:`DiscretizationConfig` instance."""

    return DiscretizationConfig(
        bins_per_dimension=tuple(bins_per_dimension),
        observation_low=np.asarray(observation_low, dtype=float),
        observation_high=np.asarray(observation_high, dtype=float),
    )


def prepare_output_directory(base_dir: Path | str) -> Path:
    """Create an experiment directory for logs, models, and plots."""

    base_path = Path(base_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_path / f"sarsa_run_{timestamp}"
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "models").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    return experiment_dir


def configure_logging(log_dir: Path) -> logging.Logger:
    """Set up a rotating logger that writes to stdout and file."""

    logger = logging.getLogger("sarsa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
