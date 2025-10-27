"""
Reactive Baseline Policies for Autoscaling
Compatible with OpenFaaS Autoscaling Gym Environment
"""

from .reactive_policies import (
    ReactiveAverage,
    ReactiveMaximum
)

__all__ = ["ReactiveAverage", "ReactiveMaximum"]
