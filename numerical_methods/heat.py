"""Finite differences for the heat equation u_t = alpha u_xx on an interval.

Both solvers use homogeneous Dirichlet conditions (u = 0 at both ends).
The grid x_0, ..., x_M includes the two boundary nodes; dx = x_{i+1} - x_i.
"""

import numpy as np


def mesh_ratio(alpha, dx, dt):
    """The ratio r = alpha dt / dx^2 that controls the stability of explicit schemes."""
    return alpha * dt / dx**2


def solve_tridiagonal(lower, diag, upper, rhs):
    """Solve a tridiagonal linear system with the Thomas algorithm in O(n) operations.

    lower[i], diag[i], upper[i] are the entries in row i left of, on and right of the diagonal
    (lower[0] and upper[-1] are not used). No pivoting: fine for diagonally dominant matrices.
    """
    n = len(rhs)
    c = np.empty(n)  # modified upper diagonal
    d = np.empty(n)  # modified right-hand side
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):  # forward elimination
        m = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m

    x = np.empty(n)  # back substitution
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def heat_ftcs(u0, dx, dt, n_steps, alpha=1.0, check_stability=True):
    """Explicit scheme: forward Euler in time, central differences in space (FTCS).

        u_i^{n+1} = u_i^n + r (u_{i+1}^n - 2 u_i^n + u_{i-1}^n),   r = alpha dt / dx^2.

    Stable only if r <= 1/2. Set check_stability=False to see what happens otherwise.
    """
    r = mesh_ratio(alpha, dx, dt)
    if check_stability and r > 0.5:
        raise ValueError(f"FTCS is unstable for r = alpha*dt/dx^2 = {r:.3f} > 1/2.")

    u = np.array(u0, dtype=float)
    u[0] = u[-1] = 0.0
    for _ in range(n_steps):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    return u


def heat_crank_nicolson(u0, dx, dt, n_steps, alpha=1.0):
    """Crank-Nicolson scheme: the average of the explicit and implicit Euler steps.

        (I - r/2 D) u^{n+1} = (I + r/2 D) u^n,   D = tridiag(1, -2, 1).

    Second order in time and space, and stable for every r.
    """
    r = mesh_ratio(alpha, dx, dt)
    u = np.array(u0, dtype=float)
    u[0] = u[-1] = 0.0

    m = u.size - 2  # number of interior unknowns
    lower = np.full(m, -r / 2)
    diag = np.full(m, 1 + r)
    upper = np.full(m, -r / 2)

    for _ in range(n_steps):
        rhs = u[1:-1] + (r / 2) * (u[2:] - 2 * u[1:-1] + u[:-2])
        u[1:-1] = solve_tridiagonal(lower, diag, upper, rhs)
    return u
