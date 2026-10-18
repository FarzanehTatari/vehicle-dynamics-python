"""Open-loop steering inputs.

Each function returns a callable ``delta(t) -> float`` giving the front-wheel
steer angle [rad] at time ``t`` [s]. These are *road-wheel* angles; divide a
handwheel angle by the steering ratio before using it here.
"""

from __future__ import annotations

import math
from typing import Callable

SteerProfile = Callable[[float], float]


def step_steer(amplitude: float, t_start: float = 0.5) -> SteerProfile:
    """Steer jumps from 0 to ``amplitude`` at ``t_start``."""
    return lambda t: amplitude if t >= t_start else 0.0


def ramp_steer(rate: float, t_start: float = 0.5, max_angle: float | None = None) -> SteerProfile:
    """Steer increases linearly at ``rate`` [rad/s] from ``t_start``.

    A slow ramp approximates the ISO 4138 steady-state circular test when
    used to read off understeer behaviour versus lateral acceleration.
    """

    def delta(t: float) -> float:
        value = max(0.0, (t - t_start) * rate)
        return value if max_angle is None else min(value, max_angle)

    return delta


def sine_steer(amplitude: float, frequency: float, t_start: float = 0.5) -> SteerProfile:
    """Continuous sine steer of given frequency [Hz], starting at ``t_start``."""
    w = 2 * math.pi * frequency
    return lambda t: amplitude * math.sin(w * (t - t_start)) if t >= t_start else 0.0


def lane_change(amplitude: float, period: float = 2.0, t_start: float = 0.5) -> SteerProfile:
    """One full sine period of steer: left, then right, then straight.

    An open-loop approximation of a single lane change. It does not track the
    ISO 3888 cone layout; that test is closed-loop and needs a path follower.
    """
    w = 2 * math.pi / period

    def delta(t: float) -> float:
        tau = t - t_start
        return amplitude * math.sin(w * tau) if 0.0 <= tau <= period else 0.0

    return delta


def double_lane_change(
    amplitude: float, period: float = 2.0, hold: float = 1.0, t_start: float = 0.5
) -> SteerProfile:
    """Two opposite lane changes separated by ``hold`` seconds of straight driving.

    Open-loop approximation; see :func:`lane_change`.
    """
    first = lane_change(amplitude, period, t_start)
    second = lane_change(-amplitude, period, t_start + period + hold)
    return lambda t: first(t) + second(t)


def sine_with_dwell(
    amplitude: float, frequency: float = 0.7, dwell: float = 0.5, t_start: float = 0.5
) -> SteerProfile:
    """Sine-with-dwell steer, as in the US FMVSS 126 / UN R140 ESC test.

    A 0.7 Hz sine: a quarter-period up to ``+amplitude``, a half-period down
    to ``-amplitude``, a ``dwell`` at ``-amplitude``, then a quarter-period
    back to zero. Designed to excite oversteer and yaw instability.
    """
    w = 2 * math.pi * frequency
    quarter = 1.0 / (4 * frequency)
    t_peak2 = 3 * quarter  # time of the negative peak, relative to start
    t_end_dwell = t_peak2 + dwell
    t_end = t_end_dwell + quarter

    def delta(t: float) -> float:
        tau = t - t_start
        if tau < 0 or tau > t_end:
            return 0.0
        if tau <= t_peak2:
            return amplitude * math.sin(w * tau)
        if tau <= t_end_dwell:
            return -amplitude
        return amplitude * math.sin(w * (tau - dwell))

    return delta
