"""Numerical linearization and discretization."""

from __future__ import annotations

from typing import Callable

import numpy as np

Dynamics = Callable[[np.ndarray, np.ndarray], np.ndarray]


def expm(m: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring with a Taylor series.

    Accurate to machine precision for the small, well-conditioned matrices
    that appear in vehicle models, and keeps the package numpy-only. Use
    ``scipy.linalg.expm`` instead for large or badly scaled matrices.
    """
    m = np.asarray(m, dtype=float)
    norm = np.linalg.norm(m, ord=1)
    squarings = max(0, int(np.ceil(np.log2(norm / 0.5)))) if norm > 0.5 else 0
    scaled = m / (2.0**squarings)
    result = np.eye(m.shape[0])
    term = np.eye(m.shape[0])
    for k in range(1, 30):
        term = term @ scaled / k
        result = result + term
        if np.linalg.norm(term, ord=1) < 1e-18 * np.linalg.norm(result, ord=1):
            break
    for _ in range(squarings):
        result = result @ result
    return result


def linearize(
    f: Dynamics | object, x0: np.ndarray, u0: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobians of ``x_dot = f(x, u)`` at ``(x0, u0)`` by central differences.

    Args:
        f: a dynamics function, or any model with a ``dynamics`` method.
        x0: operating-point state.
        u0: operating-point input.
        eps: relative perturbation size.

    Returns:
        ``(A, B)`` with ``A = df/dx`` and ``B = df/du`` evaluated at the
        operating point. ``(x0, u0)`` need not be an equilibrium; if it is not,
        the affine term ``f(x0, u0)`` is what you neglect.
    """
    fn = getattr(f, "dynamics", f)
    x0 = np.asarray(x0, dtype=float)
    u0 = np.atleast_1d(u0).astype(float)
    n, m = x0.size, u0.size

    def step(v: float) -> float:
        return eps * max(1.0, abs(v))

    a = np.zeros((n, n))
    for i in range(n):
        h = step(x0[i])
        dx = np.zeros(n)
        dx[i] = h
        a[:, i] = (fn(x0 + dx, u0) - fn(x0 - dx, u0)) / (2 * h)

    b = np.zeros((n, m))
    for j in range(m):
        h = step(u0[j])
        du = np.zeros(m)
        du[j] = h
        b[:, j] = (fn(x0, u0 + du) - fn(x0, u0 - du)) / (2 * h)

    return a, b


def discretize(
    a: np.ndarray, b: np.ndarray, dt: float, method: str = "zoh"
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize ``x_dot = A x + B u`` with sample time ``dt``.

    Args:
        method: ``"zoh"`` (zero-order hold, exact for piecewise-constant
            inputs; the right choice for MPC) or ``"euler"`` (forward Euler,
            ``Ad = I + A dt``; cheap but only first-order accurate).

    Returns:
        ``(Ad, Bd)`` such that ``x[k+1] = Ad x[k] + Bd u[k]``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b[:, None]
    n, m = b.shape
    if dt <= 0:
        raise ValueError("dt must be positive")

    if method == "zoh":
        block = np.zeros((n + m, n + m))
        block[:n, :n] = a
        block[:n, n:] = b
        phi = expm(block * dt)
        return phi[:n, :n], phi[:n, n:]
    if method == "euler":
        return np.eye(n) + a * dt, b * dt
    raise ValueError(f"unknown discretization method {method!r}; use 'zoh' or 'euler'")
