"""Wave equation u_tt = u_xx on [0, 1] with fixed ends (a vibrating string).

Left:  convergence of the leapfrog scheme against the standing wave cos(pi t) sin(pi x).
Right: a plucked string. With Courant number C = 1 the scheme is exact on the grid,
       so the corners of d'Alembert's solution stay sharp.

Run from the project root:   python -m examples.wave_equation
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numerical_methods import estimate_order, wave_leapfrog

T = 0.75


def error(M, courant):
    x = np.linspace(0, 1, M + 1)
    dx = 1 / M
    n_steps = int(round(T / (courant * dx)))
    u = wave_leapfrog(np.sin(np.pi * x), np.zeros_like(x), dx, T / n_steps, n_steps)
    return np.max(np.abs(u - np.cos(np.pi * T) * np.sin(np.pi * x)))


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Convergence
    Ms = np.array([20, 40, 80, 160, 320])
    for C, style in [(0.5, "o-"), (0.9, "s-")]:
        errs = np.array([error(M, C) for M in Ms])
        p = estimate_order(1 / Ms, errs)
        print(f"Leapfrog, C = {C}: estimated order {p:.2f}")
        ax1.loglog(1 / Ms, errs, style, label=f"Courant number $C = {C}$ (order ≈ {p:.2f})")
    print(f"Leapfrog, C = 1.0: max error {error(80, 1.0):.1e}  (exact up to rounding)")
    ax1.set_xlabel("mesh size $\\Delta x$")
    ax1.set_ylabel("max error at $t = 0.75$")
    ax1.set_title("Convergence of the leapfrog scheme")
    ax1.xaxis.set_minor_formatter(plt.NullFormatter())
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    # Plucked string: triangle pulled up at x = 0.3, released at rest.
    M = 200
    x = np.linspace(0, 1, M + 1)
    dx = 1 / M
    u0 = np.where(x < 0.3, x / 0.3, (1 - x) / 0.7)
    ax2.plot(x, u0, "k", label="$t = 0$")
    for t in [0.15, 0.35, 0.5, 0.8]:
        n_steps = int(round(t / dx))
        u = wave_leapfrog(u0, np.zeros_like(x), dx, dx, n_steps)  # C = 1
        ax2.plot(x, u, label=f"$t = {t}$")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$u$")
    ax2.set_title("Plucked string ($C = 1$)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "figures" / "wave_equation.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
