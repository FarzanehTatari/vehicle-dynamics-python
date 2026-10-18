"""Lateral tyre force models.

Each model maps a slip angle ``alpha`` [rad] to a lateral axle force [N].
Sign convention: positive slip angle produces positive lateral force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TireModel(Protocol):
    """Anything that maps slip angle [rad] to lateral force [N]."""

    def force(self, alpha: float) -> float: ...

    @property
    def cornering_stiffness(self) -> float:
        """Slope dF/d(alpha) at alpha = 0 [N/rad]."""
        ...


@dataclass(frozen=True)
class LinearTire:
    """Linear tyre: ``F = C * alpha``. Valid for small slip angles only."""

    c: float

    def force(self, alpha: float) -> float:
        return self.c * alpha

    @property
    def cornering_stiffness(self) -> float:
        return self.c


@dataclass(frozen=True)
class PacejkaTire:
    """Pacejka "Magic Formula" for lateral force, without shifts.

    ``F = D sin(C atan(B a - E (B a - atan(B a))))``

    Attributes:
        b: stiffness factor [1/rad].
        c: shape factor [-]. Around 1.3 for lateral force.
        d: peak force [N]. Usually ``mu * Fz`` for the axle.
        e: curvature factor [-]. Must be <= 1.
    """

    b: float
    c: float
    d: float
    e: float = 0.0

    def __post_init__(self) -> None:
        if self.e > 1:
            raise ValueError("Pacejka curvature factor E must be <= 1")
        if self.b <= 0 or self.c <= 0 or self.d <= 0:
            raise ValueError("Pacejka B, C and D must be positive")

    @classmethod
    def from_stiffness(
        cls, cornering_stiffness: float, fz: float, mu: float = 1.0, c: float = 1.3, e: float = 0.0
    ) -> "PacejkaTire":
        """Build a Magic Formula tyre with a given small-slip stiffness.

        Peak force is ``mu * fz``. ``B`` is chosen so that ``B * C * D``
        equals ``cornering_stiffness``, so the curve agrees with
        ``LinearTire(cornering_stiffness)`` for small slip angles and
        saturates at large ones.
        """
        d = mu * fz
        b = cornering_stiffness / (c * d)
        return cls(b=b, c=c, d=d, e=e)

    def force(self, alpha: float) -> float:
        ba = self.b * alpha
        return self.d * math.sin(self.c * math.atan(ba - self.e * (ba - math.atan(ba))))

    @property
    def cornering_stiffness(self) -> float:
        return self.b * self.c * self.d

    @property
    def peak_force(self) -> float:
        return self.d

    @property
    def peak_slip_angle(self) -> float:
        """Slip angle at which the force peaks [rad] (exact for E = 0)."""
        if self.c <= 1:
            return math.inf
        x = math.tan(math.pi / (2 * self.c))  # value of B*a - E(...) at the peak
        if self.e == 0:
            return x / self.b
        # Solve B a - E (B a - atan(B a)) = x for a by Newton's method.
        a = x / self.b
        for _ in range(50):
            ba = self.b * a
            g = ba - self.e * (ba - math.atan(ba)) - x
            dg = self.b * (1 - self.e + self.e / (1 + ba**2))
            step = g / dg
            a -= step
            if abs(step) < 1e-12:
                break
        return a


def force_curve(tire: TireModel, alphas: np.ndarray) -> np.ndarray:
    """Evaluate a tyre model over an array of slip angles."""
    return np.array([tire.force(float(a)) for a in np.asarray(alphas, dtype=float)])
