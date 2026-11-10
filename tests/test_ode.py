import numpy as np
import pytest

from numerical_methods import ORDERS, estimate_order, explicit_rk, solve


def exponential(t, y):
    return y  # y' = y, exact solution e^t


@pytest.mark.parametrize("method", list(ORDERS))
def test_observed_order_matches_theory(method):
    steps = np.array([20, 40, 80, 160])
    errors = [abs(solve(exponential, (0, 1), 1.0, n, method)[1][-1, 0] - np.e) for n in steps]
    assert estimate_order(1 / steps, errors) == pytest.approx(ORDERS[method], abs=0.15)


def test_rk4_is_accurate():
    _, y = solve(exponential, (0, 1), 1.0, 100, "rk4")
    assert abs(y[-1, 0] - np.e) < 1e-8


def test_system_shapes():
    t, y = solve(lambda t, y: np.array([y[1], -y[0]]), (0, 1), [1.0, 0.0], 10)
    assert t.shape == (11,)
    assert y.shape == (11, 2)


def test_implicit_tableau_is_rejected():
    with pytest.raises(ValueError):
        explicit_rk(exponential, (0, 1), 1.0, 10, [[0.5]], [1.0], [0.5])


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        solve(exponential, (0, 1), 1.0, 10, "nope")
