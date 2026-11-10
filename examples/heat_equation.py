"""Heat equation u_t = u_xx on [0, 1] with u = 0 at both ends.

Left:  convergence of FTCS and Crank-Nicolson against the exact solution exp(-pi^2 t) sin(pi x).
Right: the stability limit r = dt/dx^2 <= 1/2 of FTCS, seen in practice.

Run from the project root:   python -m examples.heat_equation
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numerical_methods import estimate_order, heat_crank_nicolson, heat_ftcs

T = 0.1


def error(solver, M, dt_rule):
    x = np.linspace(0, 1, M + 1)
    dx = 1 / M
    n_steps = int(np.ceil(T / dt_rule(dx)))
    u = solver(np.sin(np.pi * x), dx, T / n_steps, n_steps)
    return np.max(np.abs(u - np.exp(-np.pi**2 * T) * np.sin(np.pi * x)))


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Convergence
    Ms = np.array([10, 20, 40, 80, 160])
    cases = [
        ("FTCS", "FTCS, $\\Delta t = 0.4\\,\\Delta x^2$", heat_ftcs, lambda dx: 0.4 * dx**2, "o-"),
        ("Crank-Nicolson", "Crank–Nicolson, $\\Delta t = \\Delta x$", heat_crank_nicolson,
         lambda dx: dx, "s-"),
    ]
    for name, label, solver, dt_rule, style in cases:
        errs = np.array([error(solver, M, dt_rule) for M in Ms])
        p = estimate_order(1 / Ms, errs)
        print(f"{name}: estimated order {p:.2f}")
        ax1.loglog(1 / Ms, errs, style, label=f"{label} (order ≈ {p:.2f})")
    ax1.set_xlabel("mesh size $\\Delta x$")
    ax1.set_ylabel("max error at $t = 0.1$")
    ax1.set_title("Convergence")
    ax1.xaxis.set_minor_formatter(plt.NullFormatter())
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    # Stability: a "hat" initial condition contains every Fourier mode,
    # so the unstable high-frequency mode shows up quickly when r > 1/2.
    M = 50
    x = np.linspace(0, 1, M + 1)
    dx = 1 / M
    u0 = 1 - np.abs(2 * x - 1)
    n_steps = 110
    ax2.plot(x, u0, "k:", label="initial condition")
    for r, style in [(0.48, "-"), (0.52, "-")]:
        u = heat_ftcs(u0, dx, r * dx**2, n_steps, check_stability=False)
        ax2.plot(x, u, style, label=f"FTCS, $r = {r}$ after {n_steps} steps")
    ax2.set_ylim(-0.6, 1.2)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$u$")
    ax2.set_title("Stability limit $r = \\Delta t / \\Delta x^2 \\leq 1/2$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "figures" / "heat_equation.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
