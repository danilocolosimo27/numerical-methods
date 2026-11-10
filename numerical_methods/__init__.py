"""Numerical methods for ODEs and PDEs."""

from .heat import heat_crank_nicolson, heat_ftcs, mesh_ratio, solve_tridiagonal
from .maxwell import maxwell_yee_1d
from .ode import ORDERS, TABLEAUS, estimate_order, explicit_rk, solve
from .wave import wave_leapfrog

__all__ = [
    "ORDERS",
    "TABLEAUS",
    "estimate_order",
    "explicit_rk",
    "heat_crank_nicolson",
    "heat_ftcs",
    "maxwell_yee_1d",
    "mesh_ratio",
    "solve",
    "solve_tridiagonal",
    "wave_leapfrog",
]
