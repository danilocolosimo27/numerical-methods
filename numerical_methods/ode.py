"""Explicit Runge-Kutta methods for initial value problems y' = f(t, y), y(t0) = y0."""

import numpy as np

# Butcher tableaus (A, b, c) of some classical explicit methods.
TABLEAUS = {
    "euler": (
        [[0.0]],
        [1.0],
        [0.0],
    ),
    "heun": (
        [[0.0, 0.0],
         [1.0, 0.0]],
        [0.5, 0.5],
        [0.0, 1.0],
    ),
    "midpoint": (
        [[0.0, 0.0],
         [0.5, 0.0]],
        [0.0, 1.0],
        [0.0, 0.5],
    ),
    "kutta3": (
        [[0.0, 0.0, 0.0],
         [0.5, 0.0, 0.0],
         [-1.0, 2.0, 0.0]],
        [1 / 6, 2 / 3, 1 / 6],
        [0.0, 0.5, 1.0],
    ),
    "rk4": (
        [[0.0, 0.0, 0.0, 0.0],
         [0.5, 0.0, 0.0, 0.0],
         [0.0, 0.5, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]],
        [1 / 6, 1 / 3, 1 / 3, 1 / 6],
        [0.0, 0.5, 0.5, 1.0],
    ),
}

# Theoretical order of each method.
ORDERS = {"euler": 1, "heun": 2, "midpoint": 2, "kutta3": 3, "rk4": 4}


def explicit_rk(f, t_span, y0, n_steps, A, b, c):
    """Solve y' = f(t, y) with a generic explicit Runge-Kutta method.

    Parameters
    ----------
    f : callable
        Right-hand side, f(t, y) -> array with the same shape as y.
    t_span : tuple (t0, t1)
        Integration interval.
    y0 : float or array_like
        Initial value.
    n_steps : int
        Number of uniform steps, so h = (t1 - t0) / n_steps.
    A, b, c : array_like
        Butcher tableau. A must be strictly lower triangular.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
    y : ndarray, shape (n_steps + 1, dim)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    if np.any(np.triu(A) != 0):
        raise ValueError("A must be strictly lower triangular for an explicit method.")

    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))

    t = t0 + h * np.arange(n_steps + 1)
    y = np.empty((n_steps + 1, y0.size))
    y[0] = y0
    k = np.empty((len(b), y0.size))  # stage derivatives

    for n in range(n_steps):
        for i in range(len(b)):
            y_stage = y[n] + h * (A[i, :i] @ k[:i])
            k[i] = f(t[n] + c[i] * h, y_stage)
        y[n + 1] = y[n] + h * (b @ k)

    return t, y


def solve(f, t_span, y0, n_steps, method="rk4"):
    """Solve y' = f(t, y) with one of the methods in TABLEAUS."""
    if method not in TABLEAUS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(TABLEAUS)}.")
    return explicit_rk(f, t_span, y0, n_steps, *TABLEAUS[method])


def estimate_order(step_sizes, errors):
    """Estimate the convergence order p from error ~ C h^p (least-squares fit in log-log scale)."""
    slope, _ = np.polyfit(np.log(step_sizes), np.log(errors), 1)
    return slope
