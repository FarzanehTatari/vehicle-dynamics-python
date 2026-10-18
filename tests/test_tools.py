"""Tyres, linearization, discretization, maneuvers, simulation plumbing."""

import math

import numpy as np
import pytest

from vehicle_dynamics import (
    LinearBicycle,
    LinearTire,
    PacejkaTire,
    discretize,
    double_lane_change,
    lane_change,
    linearize,
    midsize_sedan,
    ramp_steer,
    simulate,
    sine_with_dwell,
    step_steer,
)

P = midsize_sedan()


# --------------------------------------------------------------- tyres
def test_pacejka_small_slip_slope_equals_requested_stiffness():
    tire = PacejkaTire.from_stiffness(160_000.0, fz=8000.0, mu=1.0)
    a = 1e-6
    slope = (tire.force(a) - tire.force(-a)) / (2 * a)
    assert slope == pytest.approx(160_000.0, rel=1e-6)
    assert tire.cornering_stiffness == pytest.approx(160_000.0)


def test_pacejka_peak_is_mu_fz_at_reported_slip():
    tire = PacejkaTire.from_stiffness(160_000.0, fz=8000.0, mu=0.9)
    alphas = np.linspace(0, 0.5, 20001)
    forces = np.array([tire.force(a) for a in alphas])
    assert forces.max() == pytest.approx(0.9 * 8000.0, rel=1e-6)
    assert alphas[forces.argmax()] == pytest.approx(tire.peak_slip_angle, abs=1e-4)


def test_pacejka_peak_slip_with_curvature_factor():
    tire = PacejkaTire(b=10.0, c=1.3, d=5000.0, e=-0.5)
    alphas = np.linspace(0, 0.6, 60001)
    forces = np.array([tire.force(a) for a in alphas])
    assert alphas[forces.argmax()] == pytest.approx(tire.peak_slip_angle, abs=1e-4)


def test_pacejka_is_odd():
    tire = PacejkaTire(b=10.0, c=1.3, d=5000.0, e=0.2)
    for a in [0.01, 0.05, 0.2]:
        assert tire.force(-a) == pytest.approx(-tire.force(a))


def test_pacejka_rejects_bad_curvature():
    with pytest.raises(ValueError):
        PacejkaTire(b=10.0, c=1.3, d=5000.0, e=1.5)


def test_linear_tire():
    assert LinearTire(1000.0).force(0.1) == pytest.approx(100.0)


# --------------------------------------------------------------- linearize
def test_linearize_recovers_exact_linear_system():
    model = LinearBicycle(P, 20.0)
    a, b = linearize(model, [0.3, -0.1], [0.01])
    np.testing.assert_allclose(a, model.A, rtol=1e-7)
    np.testing.assert_allclose(b, model.B, rtol=1e-7)


def test_linearize_accepts_plain_function():
    a, b = linearize(lambda x, u: np.array([x[0] ** 2 + u[0]]), [3.0], [0.0])
    assert a[0, 0] == pytest.approx(6.0)
    assert b[0, 0] == pytest.approx(1.0)


# --------------------------------------------------------------- discretize
def test_zoh_matches_simulation_over_one_step():
    model = LinearBicycle(P, 20.0)
    dt = 0.05
    ad, bd = discretize(model.A, model.B, dt)
    x0, u0 = np.array([0.2, 0.05]), 0.03
    t = np.linspace(0.0, dt, 2001)
    x_sim = simulate(model, x0, t, lambda _t: u0).x[-1]
    np.testing.assert_allclose(ad @ x0 + bd[:, 0] * u0, x_sim, rtol=1e-9, atol=1e-12)


def test_euler_is_first_order():
    model = LinearBicycle(P, 20.0)
    ad, bd = discretize(model.A, model.B, 0.01, method="euler")
    np.testing.assert_allclose(ad, np.eye(2) + model.A * 0.01)
    np.testing.assert_allclose(bd, model.B * 0.01)


def test_zoh_converges_to_euler_for_tiny_steps():
    model = LinearBicycle(P, 20.0)
    dt = 1e-6
    ad_z, _ = discretize(model.A, model.B, dt)
    ad_e, _ = discretize(model.A, model.B, dt, method="euler")
    np.testing.assert_allclose(ad_z, ad_e, atol=1e-9)


def test_discretize_rejects_unknown_method():
    with pytest.raises(ValueError):
        discretize(np.eye(2), np.ones((2, 1)), 0.1, method="tustin-ish")


# --------------------------------------------------------------- maneuvers
def test_step_steer():
    s = step_steer(0.05, t_start=1.0)
    assert s(0.99) == 0.0 and s(1.0) == 0.05


def test_ramp_steer_saturates():
    s = ramp_steer(0.1, t_start=0.0, max_angle=0.2)
    assert s(1.0) == pytest.approx(0.1)
    assert s(10.0) == pytest.approx(0.2)


def test_lane_change_returns_to_zero():
    s = lane_change(0.03, period=2.0, t_start=0.0)
    assert s(0.5) == pytest.approx(0.03)
    assert s(1.5) == pytest.approx(-0.03)
    assert s(2.5) == 0.0


def test_double_lane_change_is_antisymmetric():
    s = double_lane_change(0.03, period=2.0, hold=1.0, t_start=0.0)
    assert s(0.5) == pytest.approx(0.03)
    assert s(3.5) == pytest.approx(-0.03)  # second change starts at t=3


def test_sine_with_dwell_shape():
    amp, f, dwell = 0.1, 0.7, 0.5
    s = sine_with_dwell(amp, f, dwell, t_start=0.0)
    q = 1 / (4 * f)
    assert s(q) == pytest.approx(amp)  # first peak
    assert s(3 * q) == pytest.approx(-amp)  # second peak
    assert s(3 * q + dwell / 2) == pytest.approx(-amp)  # dwelling
    assert s(4 * q + dwell) == pytest.approx(0.0, abs=1e-12)  # back to zero
    assert s(4 * q + dwell + 0.5) == 0.0  # and stays there
    # continuous at the end of the dwell
    eps = 1e-9
    assert s(3 * q + dwell + eps) == pytest.approx(-amp, abs=1e-6)


# --------------------------------------------------------------- simulate
def test_simulate_accepts_input_array():
    model = LinearBicycle(P, 20.0)
    t = np.linspace(0, 1, 101)
    res = simulate(model, [0, 0], t, np.full(101, 0.01))
    assert res.x.shape == (101, 2) and res.u.shape == (101, 1)


def test_simulate_rejects_mismatched_input():
    with pytest.raises(ValueError):
        simulate(LinearBicycle(P, 20.0), [0, 0], np.linspace(0, 1, 10), np.zeros(5))


def test_simulate_rejects_non_increasing_time():
    with pytest.raises(ValueError):
        simulate(LinearBicycle(P, 20.0), [0, 0], [0.0, 0.0, 1.0], lambda t: 0.0)


def test_result_lookup_by_name():
    res = simulate(LinearBicycle(P, 20.0), [0, 0], np.linspace(0, 1, 11), lambda t: 0.0)
    assert res["r"].shape == (11,)
    with pytest.raises(KeyError):
        res["nope"]


def test_rk4_accuracy_on_exponential():
    class Decay:
        state_names = ("x",)

        def dynamics(self, x, u):
            return -x

    t = np.linspace(0, 1, 101)
    res = simulate(Decay(), [1.0], t, lambda _t: 0.0)
    assert res.x[-1, 0] == pytest.approx(math.exp(-1), rel=1e-9)
