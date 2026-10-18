"""Where does the linear bicycle model stop being trustworthy?

Left: steady-state lateral acceleration versus steer angle. Linear tyres keep
climbing; Pacejka tyres saturate at the friction limit.
Right: the same aggressive double lane change, simulated with both tyre
models. They agree on gentle inputs and diverge once the tyres saturate.

Run:  python examples/linear_vs_pacejka.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vehicle_dynamics import (
    NonlinearSingleTrack,
    PacejkaTire,
    double_lane_change,
    midsize_sedan,
    simulate,
    step_steer,
)

params = midsize_sedan()
vx = 20.0  # m/s, about 72 km/h

linear = NonlinearSingleTrack(params)
pacejka = NonlinearSingleTrack(
    params,
    front_tire=PacejkaTire.from_stiffness(params.cf, params.fz_front, params.mu),
    rear_tire=PacejkaTire.from_stiffness(params.cr, params.fz_rear, params.mu),
)
x0 = [0.0, 0.0, 0.0, vx, 0.0, 0.0]

# --- steady-state lateral acceleration sweep -------------------------------
t = np.arange(0.0, 6.0, 0.005)
steers = np.linspace(0.0, 0.10, 21)
ay_lin, ay_pac = [], []
for d in steers:
    for model, store in ((linear, ay_lin), (pacejka, ay_pac)):
        res = simulate(model, x0, t, step_steer(d, t_start=0.0))
        store.append(model.lateral_acceleration(res.x[-1], [d]))

# --- double lane change ----------------------------------------------------
t2 = np.arange(0.0, 8.0, 0.005)
dlc_amp = 0.09  # rad, about 5.2 degrees of road-wheel angle
maneuver = double_lane_change(amplitude=dlc_amp, period=2.0, hold=1.0)
path_lin = simulate(linear, x0, t2, maneuver)
path_pac = simulate(pacejka, x0, t2, maneuver)

# --- plot ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
lin_style = dict(color="#4C72B0", lw=2.2, label="linear tyres")
pac_style = dict(color="#DD8452", lw=2.2, label="Pacejka tyres")

ax1.plot(np.degrees(steers), np.array(ay_lin) / params.g, **lin_style)
ax1.plot(np.degrees(steers), np.array(ay_pac) / params.g, **pac_style)
ax1.axhline(params.mu, color="0.5", ls="--", lw=1)
ax1.text(0.2, params.mu + 0.03, "friction limit  μg", color="0.4", fontsize=9)
ax1.set(xlabel="road-wheel steer angle [deg]", ylabel="steady-state lateral accel. [g]",
        title=f"Steady-state cornering at {vx * 3.6:.0f} km/h")
ax1.legend(frameon=False)
ax1.grid(alpha=0.3)

ax2.plot(path_lin["X"], path_lin["Y"], **lin_style)
ax2.plot(path_pac["X"], path_pac["Y"], **pac_style)
ax2.set(xlabel="X [m]", ylabel="Y [m]", title=f"Open-loop double lane change, {np.degrees(dlc_amp):.1f}° peak steer")
ax2.legend(frameon=False)
ax2.grid(alpha=0.3)

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "assets" / "linear_vs_pacejka.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
