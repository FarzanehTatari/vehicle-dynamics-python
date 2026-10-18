"""Single-track (bicycle) vehicle models.

Three models, sharing one interface (``dynamics(x, u) -> x_dot``):

* :class:`LinearBicycle` -- linear lateral dynamics in the body frame,
  state ``[vy, r]``. The workhorse for analysis and controller design.
* :class:`LaneKeepingErrorModel` -- the same physics written in path-error
  coordinates, state ``[e_y, e_y_dot, e_psi, e_psi_dot]``. The standard plant
  for lane-keeping MPC and LQR.
* :class:`NonlinearSingleTrack` -- nonlinear planar model with global pose,
  exact slip-angle kinematics and a pluggable tyre model.

Conventions (ISO 8855, SAE-compatible in the plane): x forward, y left,
yaw positive counter-clockwise, positive steer turns left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .params import VehicleParams
from .tires import LinearTire, TireModel


def _check_speed(vx: float) -> None:
    if not math.isfinite(vx) or vx <= 0:
        raise ValueError(f"longitudinal speed vx must be positive, got {vx!r}")


# ---------------------------------------------------------------------------
# Linear models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearBicycle:
    """Linear lateral bicycle model at constant longitudinal speed.

    State ``x = [vy, r]``: lateral velocity [m/s] and yaw rate [rad/s].
    Input ``u = [delta]``: front-wheel steer angle [rad].

    ``x_dot = A x + B u``, with linear tyres and small-angle kinematics.
    """

    params: VehicleParams
    vx: float

    state_names = ("vy", "r")
    input_names = ("delta",)

    def __post_init__(self) -> None:
        _check_speed(self.vx)

    @property
    def A(self) -> np.ndarray:
        p, vx = self.params, self.vx
        cf, cr, lf, lr = p.cf, p.cr, p.lf, p.lr
        return np.array(
            [
                [-(cf + cr) / (p.m * vx), -(lf * cf - lr * cr) / (p.m * vx) - vx],
                [-(lf * cf - lr * cr) / (p.iz * vx), -(lf**2 * cf + lr**2 * cr) / (p.iz * vx)],
            ]
        )

    @property
    def B(self) -> np.ndarray:
        p = self.params
        return np.array([[p.cf / p.m], [p.lf * p.cf / p.iz]])

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ np.asarray(x, dtype=float) + self.B @ np.atleast_1d(u).astype(float)

    def lateral_acceleration(self, x: np.ndarray, u: np.ndarray) -> float:
        """Lateral acceleration at the centre of gravity, ``vy_dot + vx r`` [m/s^2]."""
        x = np.asarray(x, dtype=float)
        return float(self.dynamics(x, u)[0] + self.vx * x[1])

    def steady_state(self, delta: float) -> np.ndarray:
        """Equilibrium state for a constant steer angle."""
        return -np.linalg.solve(self.A, self.B[:, 0] * delta)

    def slip_angles(self, x: np.ndarray, u: np.ndarray) -> tuple[float, float]:
        """Small-angle front and rear slip angles [rad]."""
        vy, r = np.asarray(x, dtype=float)
        delta = float(np.atleast_1d(u)[0])
        p = self.params
        return delta - (vy + p.lf * r) / self.vx, -(vy - p.lr * r) / self.vx


@dataclass(frozen=True)
class LaneKeepingErrorModel:
    """Linear lateral dynamics in path-error coordinates.

    State ``x = [e_y, e_y_dot, e_psi, e_psi_dot]``: lateral offset from the
    path [m], its rate [m/s], heading error [rad], and its rate [rad/s].
    Input ``u = [delta, psi_dot_des]``: steer angle [rad] and the yaw rate the
    path demands, ``vx * curvature`` [rad/s]. The second input is a known
    disturbance, not something the controller chooses.

    ``x_dot = A x + B delta + E psi_dot_des``. Follows Rajamani (2012),
    Section 2.5, with per-axle stiffnesses.
    """

    params: VehicleParams
    vx: float

    state_names = ("e_y", "e_y_dot", "e_psi", "e_psi_dot")
    input_names = ("delta", "psi_dot_des")

    def __post_init__(self) -> None:
        _check_speed(self.vx)

    @property
    def A(self) -> np.ndarray:
        p, vx = self.params, self.vx
        cf, cr, lf, lr, m, iz = p.cf, p.cr, p.lf, p.lr, p.m, p.iz
        return np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -(cf + cr) / (m * vx), (cf + cr) / m, (-cf * lf + cr * lr) / (m * vx)],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, -(cf * lf - cr * lr) / (iz * vx), (cf * lf - cr * lr) / iz,
                 -(cf * lf**2 + cr * lr**2) / (iz * vx)],
            ]
        )

    @property
    def B(self) -> np.ndarray:
        """Steer input matrix (4 x 1)."""
        p = self.params
        return np.array([[0.0], [p.cf / p.m], [0.0], [p.cf * p.lf / p.iz]])

    @property
    def E(self) -> np.ndarray:
        """Road-curvature disturbance matrix (4 x 1), multiplies ``psi_dot_des``."""
        p, vx = self.params, self.vx
        return np.array(
            [
                [0.0],
                [-(p.cf * p.lf - p.cr * p.lr) / (p.m * vx) - vx],
                [0.0],
                [-(p.cf * p.lf**2 + p.cr * p.lr**2) / (p.iz * vx)],
            ]
        )

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(u).astype(float)
        if u.shape != (2,):
            raise ValueError("LaneKeepingErrorModel expects u = [delta, psi_dot_des]")
        return self.A @ np.asarray(x, dtype=float) + self.B[:, 0] * u[0] + self.E[:, 0] * u[1]


# ---------------------------------------------------------------------------
# Nonlinear model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonlinearSingleTrack:
    """Nonlinear planar single-track model.

    State ``x = [X, Y, psi, vx, vy, r]``: global position [m], heading [rad],
    body-frame longitudinal and lateral velocity [m/s], yaw rate [rad/s].

    Inputs:
        * ``hold_speed=True`` (default): ``u = [delta]``. Longitudinal speed
          is held constant, which isolates lateral dynamics and makes direct
          comparison with :class:`LinearBicycle` meaningful.
        * ``hold_speed=False``: ``u = [delta, fx]``, where ``fx`` is a
          longitudinal force at the rear axle [N] (rear-wheel drive / braking).

    Tyre models default to linear tyres built from ``params.cf`` and
    ``params.cr``. Pass :class:`~vehicle_dynamics.tires.PacejkaTire` instances
    to model saturation.
    """

    params: VehicleParams
    front_tire: TireModel | None = None
    rear_tire: TireModel | None = None
    hold_speed: bool = True
    min_speed: float = field(default=0.5, repr=False)

    state_names = ("X", "Y", "psi", "vx", "vy", "r")

    def __post_init__(self) -> None:
        if self.front_tire is None:
            object.__setattr__(self, "front_tire", LinearTire(self.params.cf))
        if self.rear_tire is None:
            object.__setattr__(self, "rear_tire", LinearTire(self.params.cr))

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("delta",) if self.hold_speed else ("delta", "fx")

    def slip_angles(self, x: np.ndarray, delta: float) -> tuple[float, float]:
        """Exact front and rear slip angles [rad]."""
        _, _, _, vx, vy, r = x
        p = self.params
        vx = max(vx, self.min_speed)
        alpha_f = delta - math.atan2(vy + p.lf * r, vx)
        alpha_r = -math.atan2(vy - p.lr * r, vx)
        return alpha_f, alpha_r

    def tire_forces(self, x: np.ndarray, delta: float) -> tuple[float, float]:
        """Front and rear lateral axle forces [N], in each axle's own frame."""
        alpha_f, alpha_r = self.slip_angles(x, delta)
        return self.front_tire.force(alpha_f), self.rear_tire.force(alpha_r)

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.atleast_1d(u).astype(float)
        expected = 1 if self.hold_speed else 2
        if u.shape != (expected,):
            raise ValueError(f"expected u = {list(self.input_names)}, got shape {u.shape}")
        delta = u[0]
        fx = 0.0 if self.hold_speed else u[1]

        _, _, psi, vx, vy, r = x
        p = self.params
        fyf, fyr = self.tire_forces(x, delta)

        x_dot = vx * math.cos(psi) - vy * math.sin(psi)
        y_dot = vx * math.sin(psi) + vy * math.cos(psi)
        vx_dot = 0.0 if self.hold_speed else (fx - fyf * math.sin(delta)) / p.m + vy * r
        vy_dot = (fyf * math.cos(delta) + fyr) / p.m - vx * r
        r_dot = (p.lf * fyf * math.cos(delta) - p.lr * fyr) / p.iz
        return np.array([x_dot, y_dot, r, vx_dot, vy_dot, r_dot])

    def lateral_acceleration(self, x: np.ndarray, u: np.ndarray) -> float:
        """Body-frame lateral acceleration at the centre of gravity [m/s^2]."""
        x = np.asarray(x, dtype=float)
        return float(self.dynamics(x, u)[4] + x[3] * x[5])
