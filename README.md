# Numerical Methods for ODEs and PDEs

Clean Python implementations of classical numerical methods for differential equations,
with convergence studies that check each method against its theoretical order.

## Explicit Runge–Kutta methods

For the initial value problem $y'(t) = f(t, y(t))$, $y(t_0) = y_0$, an $s$-stage explicit
Runge–Kutta method with step size $h$ computes

$$
k_i = f\Big(t_n + c_i h,\; y_n + h \sum_{j<i} a_{ij} k_j\Big), \qquad
y_{n+1} = y_n + h \sum_{i=1}^{s} b_i k_i ,
$$

where the coefficients are collected in the Butcher tableau $(A, b, c)$.
A method has order $p$ if the global error satisfies $\|y_N - y(T)\| = O(h^p)$.

The function `explicit_rk` accepts **any** explicit Butcher tableau; the following are built in:

| Method   | Stages | Order |
|----------|:------:|:-----:|
| Euler    | 1      | 1     |
| Heun     | 2      | 2     |
| Midpoint | 2      | 2     |
| Kutta 3  | 3      | 3     |
| RK4      | 4      | 4     |

### Convergence study

Test problem: the harmonic oscillator $x'' = -x$, $x(0)=1$, $x'(0)=0$, with exact solution
$x(t) = \cos t$, integrated over one period $[0, 2\pi]$.
The observed order is the slope of the error in a log–log plot:

![Convergence of Runge–Kutta methods](figures/convergence_rk.png)

Two remarks on the plot:

- **Heun and midpoint coincide.** For a linear problem $y' = \lambda y$ both methods give
  $y_{n+1} = \big(1 + h\lambda + \tfrac{(h\lambda)^2}{2}\big)\, y_n$, the degree-2 Taylor
  polynomial of $e^{h\lambda}$, so their errors are identical.
- **Euler's observed order is slightly above 1.** For the oscillator, explicit Euler multiplies
  the amplitude by $\sqrt{1+h^2}$ at every step, so the error is dominated by the amplitude drift
  $(1+h^2)^{N/2} - 1 \approx e^{hT/2} - 1$.
  For large $h$ this is not yet in the asymptotic regime $O(h)$; the slope tends to 1 as $h \to 0$.

## Usage

```python
import numpy as np
from numerical_methods import solve

# y' = -2 t y,  y(0) = 1   (exact: y = exp(-t^2))
t, y = solve(lambda t, y: -2 * t * y, (0, 2), 1.0, n_steps=100, method="rk4")
```

## Getting started

```bash
pip install -r requirements.txt
python -m examples.convergence_rk   # runs the study and saves the figure
pytest                              # runs the tests
```

## Roadmap

- [x] Explicit Runge–Kutta methods with convergence study
- [ ] Adaptive step size (embedded Runge–Kutta pairs)
- [ ] Implicit methods and stiff problems
- [ ] Finite differences for the heat equation (stability, CFL condition)
- [ ] Wave equation
- [ ] Maxwell's equations with the Yee scheme (FDTD)

## Author

Danilo Colosimo — M.Sc. in Mathematics (University of Calabria & JMU Würzburg)
