"""Maxwell's equations in 1D with the Yee scheme (FDTD), in a cavity with conducting walls.

Left:  space-time diagram of E for a Gaussian pulse. The pulse splits into two halves that
       travel at speed c = 1 and change sign at every reflection on a conducting wall (E = 0 there).
Right: convergence against the exact cavity mode E = sin(pi x) cos(pi t).

Run from the project root:   python -m examples.maxwell_yee
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numerical_methods import estimate_order, maxwell_yee_1d

T = 0.75


def cavity_error(N, courant):
    x = np.linspace(0, 1, N + 1)
    x_half = (x[:-1] + x[1:]) / 2
    dx = 1 / N
    n_steps = int(round(T / (courant * dx)))
    dt = T / n_steps
    E0 = np.sin(np.pi * x)
    H_half = np.cos(np.pi * x_half) * np.sin(np.pi * dt / 2)  # exact H at time dt/2
    E, _ = maxwell_yee_1d(E0, H_half, dx, dt, n_steps)
    return np.max(np.abs(E - np.sin(np.pi * x) * np.cos(np.pi * T)))


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Gaussian pulse, initially at rest (H = 0)
    N = 400
    x = np.linspace(0, 1, N + 1)
    dx = 1 / N
    dt = 0.9 * dx
    t_end = 2.0
    n_steps = int(round(t_end / dt))
    E0 = np.exp(-(((x - 0.5) / 0.04) ** 2))
    _, _, hist = maxwell_yee_1d(E0, np.zeros(N), dx, dt, n_steps, history=True)
    im = ax1.imshow(
        hist, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        extent=[0, 1, 0, n_steps * dt],
    )
    fig.colorbar(im, ax=ax1, label="$E$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$t$")
    ax1.set_title("Pulse in a cavity (Yee scheme, $S = 0.9$)")

    # Convergence on the cavity mode
    Ns = np.array([20, 40, 80, 160, 320])
    errs = np.array([cavity_error(n, 0.5) for n in Ns])
    p = estimate_order(1 / Ns, errs)
    print(f"Yee scheme, S = 0.5: estimated order {p:.2f}")
    ax2.loglog(1 / Ns, errs, "o-", label=f"Courant number $S = 0.5$ (order ≈ {p:.2f})")
    ax2.set_xlabel("mesh size $\\Delta x$")
    ax2.set_ylabel("max error of $E$ at $t = 0.75$")
    ax2.set_title("Convergence on the cavity mode")
    ax2.xaxis.set_minor_formatter(plt.NullFormatter())
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "figures" / "maxwell_yee.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
