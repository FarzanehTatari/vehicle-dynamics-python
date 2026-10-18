"""Vehicle parameters for single-track (bicycle) models."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

G = 9.81  # gravitational acceleration [m/s^2]


@dataclass(frozen=True)
class VehicleParams:
    """Physical parameters of a single-track vehicle.

    Cornering stiffnesses are **per axle** (both tyres of the axle combined),
    in N/rad. Many textbooks, including Rajamani, write ``2 * C_alpha`` with a
    per-tyre stiffness; ``cf = 2 * C_alpha_f`` in that notation.

    Attributes:
        m: total vehicle mass [kg].
        iz: yaw moment of inertia about the centre of gravity [kg m^2].
        lf: distance from the centre of gravity to the front axle [m].
        lr: distance from the centre of gravity to the rear axle [m].
        cf: front-axle cornering stiffness [N/rad].
        cr: rear-axle cornering stiffness [N/rad].
        mu: tyre-road friction coefficient [-], used by saturating tyre models.
        g: gravitational acceleration [m/s^2].
    """

    m: float
    iz: float
    lf: float
    lr: float
    cf: float
    cr: float
    mu: float = 1.0
    g: float = G

    def __post_init__(self) -> None:
        for name in ("m", "iz", "lf", "lr", "cf", "cr", "mu", "g"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"VehicleParams.{name} must be positive and finite, got {value!r}")

    # ------------------------------------------------------------------ geometry
    @property
    def wheelbase(self) -> float:
        """Wheelbase L = lf + lr [m]."""
        return self.lf + self.lr

    @property
    def fz_front(self) -> float:
        """Static normal load on the front axle [N]."""
        return self.m * self.g * self.lr / self.wheelbase

    @property
    def fz_rear(self) -> float:
        """Static normal load on the rear axle [N]."""
        return self.m * self.g * self.lf / self.wheelbase

    # ------------------------------------------------------- handling metrics
    @property
    def understeer_gradient(self) -> float:
        """Understeer gradient K_us [rad/(m/s^2)].

        Positive: understeer. Zero: neutral steer. Negative: oversteer.
        Defined by the steady-state relation ``delta = L/R + K_us * a_y``.
        """
        return (self.m / self.wheelbase) * (self.lr / self.cf - self.lf / self.cr)

    @property
    def characteristic_speed(self) -> float:
        """Speed at which steady-state yaw-rate gain peaks [m/s].

        Only defined for understeering vehicles (``understeer_gradient > 0``).
        """
        k = self.understeer_gradient
        if k <= 0:
            raise ValueError("characteristic_speed is only defined for understeering vehicles")
        return math.sqrt(self.wheelbase / k)

    @property
    def critical_speed(self) -> float:
        """Speed above which the linear model becomes unstable [m/s].

        Only defined for oversteering vehicles (``understeer_gradient < 0``).
        """
        k = self.understeer_gradient
        if k >= 0:
            raise ValueError("critical_speed is only defined for oversteering vehicles")
        return math.sqrt(-self.wheelbase / k)

    def steady_state_yaw_gain(self, vx: float) -> float:
        """Steady-state yaw-rate gain r/delta at longitudinal speed ``vx`` [1/s]."""
        return vx / (self.wheelbase + self.understeer_gradient * vx**2)

    def with_(self, **changes: float) -> "VehicleParams":
        """Return a copy with some fields replaced, e.g. ``p.with_(mu=0.3)``."""
        return replace(self, **changes)


def midsize_sedan() -> VehicleParams:
    """A mid-size passenger sedan.

    Values from Rajamani, *Vehicle Dynamics and Control*, 2nd ed., Springer
    2012 (lateral control examples): m = 1573 kg, Iz = 2873 kg m^2,
    lf = 1.10 m, lr = 1.58 m, and 80 000 N/rad per tyre (160 000 N/rad
    per axle). Mildly understeering.
    """
    return VehicleParams(m=1573.0, iz=2873.0, lf=1.10, lr=1.58, cf=160_000.0, cr=160_000.0)
