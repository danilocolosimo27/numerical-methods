"""Finite differences for the wave equation u_tt = c^2 u_xx on an interval.

Homogeneous Dirichlet conditions (u = 0 at both ends, like a string with fixed endpoints).
The grid x_0, ..., x_M includes the two boundary nodes.
"""

import numpy as np


def _second_difference(u):
    """u_{i+1} - 2 u_i + u_{i-1} at the interior nodes."""
    return u[2:] - 2 * u[1:-1] + u[:-2]


def wave_leapfrog(u0, v0, dx, dt, n_steps, c=1.0, check_cfl=True):
    """Central differences in time and space (leapfrog):

        u_i^{n+1} = 2 u_i^n - u_i^{n-1} + C^2 (u_{i+1}^n - 2 u_i^n + u_{i-1}^n),   C = c dt / dx.

    u0, v0 are the initial displacement u(x, 0) and velocity u_t(x, 0) on the grid.
    Stable if the Courant number C <= 1 (CFL condition). Returns u at time n_steps * dt.
    """
    C = c * dt / dx
    if check_cfl and C > 1:
        raise ValueError(f"Leapfrog is unstable for Courant number C = c*dt/dx = {C:.3f} > 1.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")

    u_prev = np.array(u0, dtype=float)
    u_prev[0] = u_prev[-1] = 0.0
    v0 = np.asarray(v0, dtype=float)

    # First step from the Taylor expansion u(dt) = u + dt u_t + dt^2/2 c^2 u_xx + O(dt^3).
    u = np.zeros_like(u_prev)
    u[1:-1] = u_prev[1:-1] + dt * v0[1:-1] + 0.5 * C**2 * _second_difference(u_prev)

    for _ in range(n_steps - 1):
        u_next = np.zeros_like(u)
        u_next[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + C**2 * _second_difference(u)
        u_prev, u = u, u_next
    return u
