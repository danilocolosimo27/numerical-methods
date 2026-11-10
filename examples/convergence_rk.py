"""Convergence study of explicit Runge-Kutta methods on the harmonic oscillator.

    x'' = -x,  x(0) = 1, x'(0) = 0   ->   exact solution x(t) = cos t.

Written as a first-order system y = (x, v):  y' = (v, -x).

Run from the project root:   python -m examples.convergence_rk
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numerical_methods import ORDERS, estimate_order, solve


def oscillator(t, y):
    x, v = y
    return np.array([v, -x])


def main():
    T = 2 * np.pi  # one full period
    y0 = [1.0, 0.0]
    exact = np.array([np.cos(T), -np.sin(T)])
    steps = np.array([16, 32, 64, 128, 256, 512])
    h = T / steps

    # Distinct styles: on a linear problem Heun and midpoint coincide exactly,
    # so the dashed line keeps both visible.
    styles = {"euler": "o-", "heun": "s-", "midpoint": "x--", "kutta3": "d-", "rk4": "^-"}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, p in ORDERS.items():
        errors = []
        for n in steps:
            _, y = solve(oscillator, (0.0, T), y0, n, method)
            errors.append(np.linalg.norm(y[-1] - exact))
        errors = np.array(errors)
        p_est = estimate_order(h, errors)
        print(f"{method:>9}: expected order {p}, estimated {p_est:.2f}")
        ax.loglog(h, errors, styles.get(method, "o-"),label=f"{method} (order ≈ {p_est:.2f})")

    ax.set_xlabel("step size $h$")
    ax.set_ylabel(r"error $\|y_N - y(T)\|$")
    ax.set_title("Explicit Runge-Kutta methods: convergence")
    ax.xaxis.set_minor_formatter(plt.NullFormatter())  # avoid overlapping tick labels
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "convergence_rk.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
