"""vehicle-dynamics-python: single-track vehicle models for control and learning.

Quick start::

    import numpy as np
    from vehicle_dynamics import LinearBicycle, midsize_sedan, simulate, step_steer

    model = LinearBicycle(midsize_sedan(), vx=20.0)
    t = np.arange(0.0, 5.0, 0.01)
    result = simulate(model, x0=[0.0, 0.0], t=t, u=step_steer(0.02))
    print(result["r"][-1])  # steady-state yaw rate [rad/s]
"""

from .linearize import discretize, linearize
from .maneuvers import (
    double_lane_change,
    lane_change,
    ramp_steer,
    sine_steer,
    sine_with_dwell,
    step_steer,
)
from .models import LaneKeepingErrorModel, LinearBicycle, NonlinearSingleTrack
from .params import VehicleParams, midsize_sedan
from .simulate import SimResult, simulate
from .tires import LinearTire, PacejkaTire, force_curve

__version__ = "0.1.0"

__all__ = [
    "VehicleParams",
    "midsize_sedan",
    "LinearBicycle",
    "LaneKeepingErrorModel",
    "NonlinearSingleTrack",
    "LinearTire",
    "PacejkaTire",
    "force_curve",
    "linearize",
    "discretize",
    "simulate",
    "SimResult",
    "step_steer",
    "ramp_steer",
    "sine_steer",
    "lane_change",
    "double_lane_change",
    "sine_with_dwell",
]
