"""Fixed-step simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

InputSignal = Callable[[float], "float | np.ndarray"] | np.ndarray


@dataclass(frozen=True)
class SimResult:
    """Simulation output.

    Attributes:
        t: time vector, shape ``(N,)``.
        x: states, shape ``(N, n)``. Row ``k`` is the state at ``t[k]``.
        u: inputs, shape ``(N, m)``. Row ``k`` is the input applied from
            ``t[k]`` to ``t[k+1]``.
        state_names: names of the state columns, if the model provides them.
    """

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    state_names: tuple[str, ...] = ()

    def __getitem__(self, name: str) -> np.ndarray:
        """Column of ``x`` by state name, e.g. ``result["r"]``."""
        try:
            return self.x[:, self.state_names.index(name)]
        except ValueError:
            raise KeyError(f"no state named {name!r}; available: {self.state_names}") from None


def simulate(model, x0, t, u: InputSignal) -> SimResult:
    """Simulate a model with classical fourth-order Runge-Kutta.

    The input is held constant over each step (zero-order hold), matching
    how a digital controller drives a real vehicle.

    Args:
        model: any object with ``dynamics(x, u) -> x_dot``.
        x0: initial state.
        t: increasing time vector. The step may vary.
        u: either a callable ``u(t)`` returning a scalar or array, or an
            array of shape ``(N,)`` or ``(N, m)`` with one row per time.

    Returns:
        A :class:`SimResult`.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a 1-D array with at least two samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    if callable(u):
        u_arr = np.array([np.atleast_1d(u(tk)).astype(float) for tk in t])
    else:
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1:
            u_arr = u_arr[:, None]
        if u_arr.shape[0] != t.size:
            raise ValueError(f"input array has {u_arr.shape[0]} rows but t has {t.size} samples")

    f = model.dynamics
    x = np.zeros((t.size, np.size(x0)))
    x[0] = np.asarray(x0, dtype=float)
    for k in range(t.size - 1):
        h = t[k + 1] - t[k]
        uk = u_arr[k]
        xk = x[k]
        k1 = f(xk, uk)
        k2 = f(xk + 0.5 * h * k1, uk)
        k3 = f(xk + 0.5 * h * k2, uk)
        k4 = f(xk + h * k3, uk)
        x[k + 1] = xk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return SimResult(t=t, x=x, u=u_arr, state_names=tuple(getattr(model, "state_names", ())))
