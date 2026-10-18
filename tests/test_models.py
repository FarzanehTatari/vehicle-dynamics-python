"""Physics checks: each test asserts something a vehicle dynamicist would expect."""

import numpy as np
import pytest

from vehicle_dynamics import (
    LaneKeepingErrorModel,
    LinearBicycle,
    NonlinearSingleTrack,
    PacejkaTire,
    VehicleParams,
    linearize,
    midsize_sedan,
    simulate,
    step_steer,
)

P = midsize_sedan()


# --------------------------------------------------------------- parameters
def test_axle_loads_sum_to_weight():
    assert P.fz_front + P.fz_rear == pytest.approx(P.m * P.g)


def test_preset_understeers():
    assert P.understeer_gradient > 0
    assert 0 < P.characteristic_speed < 100


def test_oversteer_has_critical_speed_not_characteristic_speed():
    over = P.with_(cf=200_000.0, cr=100_000.0)
    assert over.understeer_gradient < 0
    assert over.critical_speed > 0
    with pytest.raises(ValueError):
        _ = over.characteristic_speed


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        VehicleParams(m=-1, iz=1, lf=1, lr=1, cf=1, cr=1)


# --------------------------------------------------------------- linear bicycle
@pytest.mark.parametrize("vx", [5.0, 15.0, 30.0])
def test_steady_state_yaw_gain_matches_closed_form(vx):
    model = LinearBicycle(P, vx)
    r_ss = model.steady_state(delta=1.0)[1]
    assert r_ss == pytest.approx(P.steady_state_yaw_gain(vx), rel=1e-12)


def test_steady_state_cornering_obeys_understeer_relation():
    """delta = L/R + K_us * a_y at steady state."""
    vx, delta = 20.0, 0.02
    model = LinearBicycle(P, vx)
    x_ss = model.steady_state(delta)
    r = x_ss[1]
    radius = vx / r
    a_y = vx * r
    assert delta == pytest.approx(P.wheelbase / radius + P.understeer_gradient * a_y, rel=1e-10)


def test_understeering_vehicle_is_stable_at_all_speeds():
    for vx in [2.0, 20.0, 60.0]:
        eig = np.linalg.eigvals(LinearBicycle(P, vx).A)
        assert np.all(eig.real < 0)


def test_oversteering_vehicle_goes_unstable_above_critical_speed():
    over = P.with_(cf=200_000.0, cr=100_000.0)
    v_crit = over.critical_speed
    below = np.linalg.eigvals(LinearBicycle(over, 0.9 * v_crit).A)
    above = np.linalg.eigvals(LinearBicycle(over, 1.1 * v_crit).A)
    assert np.all(below.real < 0)
    assert np.any(above.real > 0)


def test_simulation_converges_to_steady_state():
    model = LinearBicycle(P, 20.0)
    t = np.arange(0.0, 8.0, 0.005)
    res = simulate(model, [0.0, 0.0], t, step_steer(0.02, t_start=0.0))
    np.testing.assert_allclose(res.x[-1], model.steady_state(0.02), rtol=1e-6)


def test_zero_speed_rejected():
    with pytest.raises(ValueError):
        LinearBicycle(P, 0.0)


# --------------------------------------------------------------- linear vs nonlinear
def test_nonlinear_model_linearizes_to_linear_bicycle():
    """Linearizing the nonlinear model about straight driving gives A, B exactly."""
    vx = 20.0
    nl = NonlinearSingleTrack(P)  # linear tyres, held speed
    a_full, b_full = linearize(nl, x0=[0, 0, 0, vx, 0, 0], u0=[0.0])
    lat = [4, 5]  # vy, r
    lin = LinearBicycle(P, vx)
    np.testing.assert_allclose(a_full[np.ix_(lat, lat)], lin.A, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(b_full[lat, :], lin.B, rtol=1e-6)


def test_nonlinear_with_pacejka_also_linearizes_to_linear_bicycle():
    """Pacejka tyres built from the same stiffness agree for small slip."""
    vx = 20.0
    nl = NonlinearSingleTrack(
        P,
        front_tire=PacejkaTire.from_stiffness(P.cf, P.fz_front, P.mu),
        rear_tire=PacejkaTire.from_stiffness(P.cr, P.fz_rear, P.mu),
    )
    a_full, b_full = linearize(nl, x0=[0, 0, 0, vx, 0, 0], u0=[0.0])
    lat = [4, 5]
    lin = LinearBicycle(P, vx)
    np.testing.assert_allclose(a_full[np.ix_(lat, lat)], lin.A, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(b_full[lat, :], lin.B, rtol=1e-5)


def test_linear_and_nonlinear_trajectories_agree_for_small_steer():
    vx = 20.0
    t = np.arange(0.0, 4.0, 0.005)
    steer = step_steer(0.002, t_start=0.0)  # ~0.1 degree
    lin = simulate(LinearBicycle(P, vx), [0, 0], t, steer)
    nl = simulate(NonlinearSingleTrack(P), [0, 0, 0, vx, 0, 0], t, steer)
    np.testing.assert_allclose(nl["r"], lin["r"], rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(nl["vy"], lin["vy"], rtol=1e-3, atol=1e-6)


def test_pacejka_saturates_where_linear_does_not():
    """A large step steer: linear tyres overpredict lateral acceleration."""
    vx = 20.0
    t = np.arange(0.0, 4.0, 0.005)
    steer = step_steer(0.12, t_start=0.0)  # ~7 degrees of road-wheel angle
    pac = NonlinearSingleTrack(
        P,
        front_tire=PacejkaTire.from_stiffness(P.cf, P.fz_front, P.mu),
        rear_tire=PacejkaTire.from_stiffness(P.cr, P.fz_rear, P.mu),
    )
    lin = NonlinearSingleTrack(P)
    res_p = simulate(pac, [0, 0, 0, vx, 0, 0], t, steer)
    res_l = simulate(lin, [0, 0, 0, vx, 0, 0], t, steer)
    ay_p = pac.lateral_acceleration(res_p.x[-1], [0.12])
    ay_l = lin.lateral_acceleration(res_l.x[-1], [0.12])
    assert abs(ay_p) <= P.mu * P.g * 1.01  # friction limit
    assert abs(ay_l) > abs(ay_p)


def test_hold_speed_false_needs_two_inputs():
    nl = NonlinearSingleTrack(P, hold_speed=False)
    assert nl.input_names == ("delta", "fx")
    with pytest.raises(ValueError):
        nl.dynamics([0, 0, 0, 20, 0, 0], [0.0])
    x_dot = nl.dynamics([0, 0, 0, 20, 0, 0], [0.0, P.m * 1.0])
    assert x_dot[3] == pytest.approx(1.0)  # 1 m/s^2 from fx = m * 1


def test_straight_driving_moves_along_x():
    vx = 10.0
    nl = NonlinearSingleTrack(P)
    res = simulate(nl, [0, 0, 0, vx, 0, 0], np.linspace(0, 2, 201), lambda t: 0.0)
    assert res["X"][-1] == pytest.approx(vx * 2.0)
    assert res["Y"][-1] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------- error model
def test_error_model_shares_eigenvalues_with_bicycle_plus_two_integrators():
    vx = 20.0
    err = np.sort_complex(np.linalg.eigvals(LaneKeepingErrorModel(P, vx).A))
    bike = np.linalg.eigvals(LinearBicycle(P, vx).A)
    expected = np.sort_complex(np.concatenate([bike, [0.0, 0.0]]))
    np.testing.assert_allclose(err, expected, atol=1e-9)


def test_error_model_steer_input_matches_bicycle():
    vx = 20.0
    err = LaneKeepingErrorModel(P, vx)
    bike = LinearBicycle(P, vx)
    np.testing.assert_allclose(err.B[[1, 3], 0], bike.B[:, 0])


def test_error_model_steady_cornering_heading_error_equals_minus_sideslip():
    """On a constant-radius curve with feedforward steer, the car holds the path
    with a constant heading error equal to minus its sideslip angle.

    Two equations (the e_y and e_psi acceleration rows) must vanish with one
    unknown (e_psi). That only works if the feedforward steer is exactly
    L/R + K_us * a_y, so this also checks the understeer gradient.
    """
    vx, radius = 20.0, 200.0
    err = LaneKeepingErrorModel(P, vx)
    psi_dot_des = vx / radius
    delta_ff = P.wheelbase / radius + P.understeer_gradient * vx**2 / radius
    a, b, e = err.A, err.B[:, 0], err.E[:, 0]

    # Row 1 (e_y acceleration) fixes e_psi:
    e_psi = -(b[1] * delta_ff + e[1] * psi_dot_des) / a[1, 2]
    x_ss = np.array([0.0, 0.0, e_psi, 0.0])
    x_dot = a @ x_ss + b * delta_ff + e * psi_dot_des
    assert np.allclose(x_dot, 0.0, atol=1e-9)  # row 3 agrees, so this is a true equilibrium

    vy_ss = LinearBicycle(P, vx).steady_state(delta_ff)[0]
    assert e_psi == pytest.approx(-vy_ss / vx, rel=1e-9)


def test_error_model_requires_both_inputs():
    with pytest.raises(ValueError):
        LaneKeepingErrorModel(P, 20.0).dynamics(np.zeros(4), [0.0])
