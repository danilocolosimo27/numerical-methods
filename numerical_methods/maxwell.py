"""Yee scheme (FDTD) for Maxwell's equations in one space dimension.

In normalized units (c = epsilon = mu = 1), a wave travelling along x with fields
E = E_z(x, t) and H = H_y(x, t) satisfies

    dE/dt = dH/dx,      dH/dt = dE/dx.

The Yee scheme staggers the unknowns in space AND time:
    E at integer nodes x_i = i dx       and integer times t_n = n dt,
    H at half nodes x_{i+1/2}           and half times t_{n+1/2}.
Every derivative is then a centred difference, so the scheme is second order.
The walls at x_0 and x_N are perfect conductors: E = 0 there.
"""

import numpy as np


def maxwell_yee_1d(E0, H_half, dx, dt, n_steps, check_cfl=True, history=False):
    """Advance the 1D Yee scheme by n_steps time steps.

    Parameters
    ----------
    E0 : array, shape (N+1,)
        E at the nodes x_0, ..., x_N at time 0.
    H_half : array, shape (N,)
        H at the half nodes x_{1/2}, ..., x_{N-1/2} at time dt/2.
    dx, dt : float
        Mesh size and time step. Stable if the Courant number S = dt/dx <= 1.
    n_steps : int
    history : bool
        If True, also return E at every time step (useful for space-time plots).

    Returns
    -------
    E : E at time n_steps * dt
    H : H at time (n_steps + 1/2) * dt
    E_history : array, shape (n_steps + 1, N + 1)   (only if history=True)
    """
    S = dt / dx
    if check_cfl and S > 1:
        raise ValueError(f"Yee scheme is unstable for Courant number S = dt/dx = {S:.3f} > 1.")

    E = np.array(E0, dtype=float)
    H = np.array(H_half, dtype=float)
    if H.size != E.size - 1:
        raise ValueError("H must have one value fewer than E (it lives between the E nodes).")
    E[0] = E[-1] = 0.0

    E_hist = [E.copy()] if history else None
    for _ in range(n_steps):
        E[1:-1] += S * (H[1:] - H[:-1])  # E^{n+1}   from H^{n+1/2}
        H += S * (E[1:] - E[:-1])        # H^{n+3/2} from E^{n+1}
        if history:
            E_hist.append(E.copy())

    if history:
        return E, H, np.array(E_hist)
    return E, H
