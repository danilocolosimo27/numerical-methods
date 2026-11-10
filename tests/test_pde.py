"""Tests for the PDE solvers, using problems with known exact solutions on [0, 1]."""

import numpy as np
import pytest

from numerical_methods import (
    estimate_order,
    heat_crank_nicolson,
    heat_ftcs,
    maxwell_yee_1d,
    solve_tridiagonal,
    wave_leapfrog,
)

# ---------------------------------------------------------------- heat equation
# u_t = u_xx, u(x, 0) = sin(pi x)   ->   u(x, t) = exp(-pi^2 t) sin(pi x)
T_HEAT = 0.1


def heat_error(solver, M, dt_rule):
    x = np.linspace(0, 1, M + 1)
    dx = 1 / M
    n_steps = int(np.ceil(T_HEAT / dt_rule(dx)))
    u = solver(np.sin(np.pi * x), dx, T_HEAT / n_steps, n_steps)
    return np.max(np.abs(u - np.exp(-np.pi**2 * T_HEAT) * np.sin(np.pi * x)))


def test_thomas_matches_dense_solver():
    rng = np.random.default_rng(0)
    n = 8
    lower, upper = rng.normal(size=n), rng.normal(size=n)
    diag = 4 + rng.random(n)  # diagonally dominant
    rhs = rng.normal(size=n)
    A = np.diag(diag) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    assert np.allclose(solve_tridiagonal(lower, diag, upper, rhs), np.linalg.solve(A, rhs))


@pytest.mark.parametrize(
    "solver, dt_rule",
    [
        (heat_ftcs, lambda dx: 0.4 * dx**2),  # explicit: dt must shrink like dx^2
        (heat_crank_nicolson, lambda dx: dx),  # implicit: dt ~ dx is allowed
    ],
)
def test_heat_second_order(solver, dt_rule):
    Ms = np.array([10, 20, 40, 80])
    errors = [heat_error(solver, M, dt_rule) for M in Ms]
    assert estimate_order(1 / Ms, errors) == pytest.approx(2, abs=0.15)


def test_ftcs_rejects_unstable_ratio():
    x = np.linspace(0, 1, 11)
    with pytest.raises(ValueError):
        heat_ftcs(np.sin(np.pi * x), 0.1, 0.006, 10)  # r = 0.6 > 1/2


# ---------------------------------------------------------------- wave equation
# u_tt = u_xx, u(x, 0) = sin(pi x), u_t(x, 0) = 0   ->   u(x, t) = cos(pi t) sin(pi x)
T_WAVE = 0.75


def wave_error(M, courant):
    x = np.linspace(0, 1, M + 1)
    dx = 1 / M
    n_steps = int(round(T_WAVE / (courant * dx)))
    u = wave_leapfrog(np.sin(np.pi * x), np.zeros_like(x), dx, T_WAVE / n_steps, n_steps)
    return np.max(np.abs(u - np.cos(np.pi * T_WAVE) * np.sin(np.pi * x)))


def test_wave_second_order():
    Ms = np.array([20, 40, 80, 160])
    errors = [wave_error(M, 0.5) for M in Ms]
    assert estimate_order(1 / Ms, errors) == pytest.approx(2, abs=0.15)


def test_wave_exact_at_courant_one():
    # With C = 1 the 1D leapfrog scheme reproduces d'Alembert's solution exactly.
    assert wave_error(40, 1.0) < 1e-12


def test_wave_rejects_cfl_violation():
    x = np.linspace(0, 1, 11)
    with pytest.raises(ValueError):
        wave_leapfrog(np.sin(np.pi * x), np.zeros_like(x), 0.1, 0.11, 5)


# ---------------------------------------------------------------- Maxwell (Yee)
# Cavity mode between conducting walls:
#   E(x, t) = sin(pi x) cos(pi t),   H(x, t) = cos(pi x) sin(pi t)
T_MAX = 0.75


def yee_error(N, courant):
    x = np.linspace(0, 1, N + 1)
    x_half = (x[:-1] + x[1:]) / 2
    dx = 1 / N
    n_steps = int(round(T_MAX / (courant * dx)))
    dt = T_MAX / n_steps
    E0 = np.sin(np.pi * x)
    H_half = np.cos(np.pi * x_half) * np.sin(np.pi * dt / 2)
    E, _ = maxwell_yee_1d(E0, H_half, dx, dt, n_steps)
    return np.max(np.abs(E - np.sin(np.pi * x) * np.cos(np.pi * T_MAX)))


def test_yee_second_order():
    Ns = np.array([20, 40, 80, 160])
    errors = [yee_error(N, 0.5) for N in Ns]
    assert estimate_order(1 / Ns, errors) == pytest.approx(2, abs=0.15)


def test_yee_history_shape():
    E0 = np.zeros(11)
    _, _, hist = maxwell_yee_1d(E0, np.zeros(10), 0.1, 0.05, 7, history=True)
    assert hist.shape == (8, 11)


def test_yee_rejects_wrong_sizes_and_cfl():
    with pytest.raises(ValueError):
        maxwell_yee_1d(np.zeros(11), np.zeros(11), 0.1, 0.05, 1)  # H has the wrong size
    with pytest.raises(ValueError):
        maxwell_yee_1d(np.zeros(11), np.zeros(10), 0.1, 0.2, 1)  # S = 2 > 1
