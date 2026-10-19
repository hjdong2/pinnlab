# pinnlab: physics-informed neural networks for fluid mechanics, evaluated like a numerical method

pinnlab is a study of physics-informed neural networks (PINNs) in Python (PyTorch, NumPy) for the partial
differential equations (PDEs) of computational fluid dynamics (CFD), starting with Burgers' equation. 
PINNs are evaluated the way one would evaluate any new numerical method: 
checked against exact solutions and published benchmarks, with their
error and cost measured, and with attention to where they struggle.

> **Status:** Phase 1 (Burgers' equation, forward and inverse problems) is complete.
> Phase 2 (steady Navier–Stokes: Kovasznay flow) is next.

## Highlights

- **Verified reference solution:** the exact Cole–Hopf solution reproduces the benchmark of Basdevant et al. (1986),
  $\max_t |u_x(0, t)| = 152.00516$, to all 8 published digits.
- **Forward problem:** relative $L_2$ error **2.5e-4** in float64 (2.2e-3 in float32), below the 6.7e-4 reported by
  Raissi et al. (2019) for this benchmark problem.
- **Inverse problem:** from 2,000 measurements with 1% noise, starting from guesses 2× ($\lambda_1$) and 16× ($\nu$)
  off, the PINN recovers $\lambda_1$ to **0.08%** and $\nu$ to **1.0%**; the data misfit settles at the noise level.
- **Tested:** 10 pytest tests, including a manufactured solution for the PDE residual and the Basdevant benchmark.

## Problem

We study the viscous Burgers' equation

$$u_t + \lambda_1\, u\, u_x = \nu\, u_{xx}, \qquad x \in [-1, 1],\ t \in [0, 1],$$

with $\nu = 0.01/\pi$, initial condition $u(x, 0) = -\sin(\pi x)$ and boundary conditions $u(\pm 1, t) = 0$.
A shock forms at $x = 0$ around $t \approx 0.4$, where the gradient reaches $|u_x| \approx 152$: the kind of sharp
feature where PINNs typically struggle.

- **Forward problem:** $\lambda_1 = 1$ and $\nu = 0.01/\pi$ are known; find $u$. The loss penalizes the PDE residual at
  10,000 random collocation points, the initial condition (256 points) and the boundary conditions (2 × 200 points).
- **Inverse problem:** $\lambda_1$ and $\nu$ are unknown; find them and $u$ from 2,000 noisy measurements of $u$.
  The loss is data misfit + PDE residual; $\lambda_1$ and $\log\nu$ are trained together with the network
  ($\log\nu$ keeps $\nu$ positive and gives the optimizer steps of a relative size).

Setup for all runs: fully connected network, 8 hidden layers × 20 tanh units, inputs scaled to $[-1, 1]$;
5,000 Adam iterations (learning rate $10^{-3}$), then L-BFGS with a strong-Wolfe line search.

## Verification

The reference solution is the ruler for every PINN error reported here, so it is verified before any error is trusted.
It uses the Cole–Hopf transform, which turns Burgers' equation into the heat equation, whose solution is a Gaussian
convolution. After the substitution $\eta = \sqrt{4\nu t}\, z$ the integrals are evaluated with the trapezoidal rule,
which converges spectrally for this smooth, rapidly decaying integrand; log-sum-exp scaling prevents overflow.
The full 256 × 101 evaluation grid takes well under a second (vectorized NumPy).

Independent checks, all in `tests/test_burgers.py`:

- **Published benchmark:** $\max_t |u_x(0, t)| = 152.00516$ at $t \approx 0.51$ (Basdevant et al., 1986), reproduced
  to all 8 digits.
- **Initial condition:** exact at $t = 0$, and also reached through the integral formula at $t = 10^{-6}$.
- **Symmetry and boundaries:** $u(-x, t) = -u(x, t)$ and $u(\pm 1, t) = 0$ to $10^{-12}$.
- **PDE residual (autograd):** checked against hand-derived derivatives of the manufactured solution
  $u = \sin(\pi x)\, e^{-t}$, with deliberately unusual $\lambda_1 = 1.3$, $\nu = 0.07$ so that a wrong coefficient
  cannot hide.
- **Losses and training:** losses are finite and deterministic, every loss term sends gradients to every weight
  (and to $\lambda_1$, $\log\nu$), and training reduces the loss.

## Results

| Run | Relative $L_2$ error of $u$ | Learned coefficients | Training time |
|---|---|---|---|
| Forward, float32 | 2.2e-3 | — | 13 min |
| Forward, float64 | **2.5e-4** | — | 20 min |
| Inverse, float32 (2,000 points, 1% noise) | 1.2e-3 | $\lambda_1 = 1.0008$ (error 0.08%), $\nu = 3.215 \times 10^{-3}$ (error 1.0%) | 6 min |

Relative $L_2$ error over the 256 × 101 reference grid. Times on a 15 W laptop CPU (Intel Core Ultra 7 155U),
2 threads per run; the two forward runs ran at the same time. Each number comes from a single training run, so expect
some variation with the random seed. Raissi et al. (2019) used a slightly different network and number of
initial/boundary points, so the comparison with their 6.7e-4 is indicative rather than exact.

![Forward problem: reference, PINN and pointwise error (top); solution profiles at t = 0.25, 0.5, 0.75 (bottom)](docs/figures/burgers_forward_solution.png)

**Forward problem (float64).** The PINN matches the reference everywhere, including the shock. The remaining error is
concentrated on a thin band along the shock ($x = 0$, $t \gtrsim 0.4$): the steepest gradients are where the PDE
residual is hardest to satisfy, and the PDE term dominates the final loss.

![Inverse problem: loss terms (left) and learned λ₁, ν converging to their true values (right)](docs/figures/burgers_inverse_coefficients.png)

**Inverse problem.** During Adam, $\nu$ decreases steadily in log space; after the switch to L-BFGS it reaches the true
value within about 500 evaluations. The data misfit levels off at $3.70 \times 10^{-5}$, matching the noise variance
$(0.01\,\mathrm{std}(u))^2 = 3.75 \times 10^{-5}$: the network fits the true solution and not the noise, while the PDE
residual keeps decreasing. $\nu$ is harder to identify than $\lambda_1$ because it multiplies $u_{xx}$, which is
significant only inside the thin shock, whereas $u\,u_x$ (the $\lambda_1$ term) is large almost everywhere.

## Performance notes

- **Threads.** For this small network, 2 CPU threads per run were fastest (38 ms per iteration); 14 threads were
  **55× slower** (2.4 s). Each layer is a tiny matrix product, so thread overhead and synchronization outweigh the
  extra cores, made worse by the hybrid performance/efficiency cores of a laptop CPU. Small problems are better served
  by running several jobs side by side (`scripts/bench_threads.py`).
- **Precision.** float64 was 8.6× more accurate than float32 for 1.45× the training time; float32 stops improving once
  the loss changes fall below its ~7 significant digits.

## How to run

CPU only; tested with Python 3.12 and PyTorch 2.14 on Ubuntu (WSL).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

pytest -q                                                               # 10 verification and smoke tests
python -u scripts/run_burgers.py configs/burgers_forward.yaml --quick   # ~15 s pipeline check
python -u scripts/run_burgers.py configs/burgers_forward_f64.yaml       # forward problem, float64
python -u scripts/run_burgers.py configs/burgers_inverse.yaml           # inverse problem
```

Each run writes `results/<name>/` with `config.yaml` (the exact settings used), `metrics.json`, `history.json`,
`model.pt`, `solution.png` and `loss.png`. Options: `--threads N` (default 2), `--out DIR`.

## Repository layout

```
src/pinnlab/
  models.py            fully connected network with input scaling
  train.py             Adam → L-BFGS training loop with logging
  plotting.py          solution / error maps and loss curves
  utils.py             autograd helper, relative L2 error
  problems/burgers.py  PDE residual, exact Cole–Hopf solution, forward and inverse losses
scripts/
  run_burgers.py       config-driven experiment runner
  bench_threads.py     CPU thread-count benchmark
configs/               one YAML file per experiment
tests/                 verification and smoke tests
docs/figures/          figures used in this README
```

## Roadmap

| Phase | Problem | Question |
|---|---|---|
| 1 ✅ | Burgers' equation, forward and inverse | Can a PINN capture a shock, and recover PDE coefficients from noisy data? |
| 2 | Kovasznay flow (steady Navier–Stokes, exact solution) | Does the PINN error converge with network size and collocation density, like mesh refinement? How does error vs. cost compare with a classical solver? |
| 3 | Lid-driven cavity, Re 100 → 1000 (vs. Ghia et al. 1982) | Where do PINNs fail as Re grows, and which fixes help (adaptive loss weights, residual-based sampling, Fourier features)? |
| 4 | Cylinder wake, inverse | Recover velocity and pressure fields from sparse velocity measurements. |

## References

- M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks: A deep learning framework for solving
  forward and inverse problems involving nonlinear partial differential equations*, J. Comput. Phys. 378 (2019).
- C. Basdevant, M. Deville, P. Haldenwang, J. M. Lacroix, J. Ouazzani, R. Peyret, P. Orlandi, A. T. Patera,
  *Spectral and finite difference solutions of the Burgers equation*, Computers & Fluids 14 (1986).
- E. Hopf, *The partial differential equation u_t + u u_x = μ u_xx*, Comm. Pure Appl. Math. 3 (1950);
  J. D. Cole, *On a quasi-linear parabolic equation occurring in aerodynamics*, Quart. Appl. Math. 9 (1951).
- L. N. Trefethen, J. A. C. Weideman, *The exponentially convergent trapezoidal rule*, SIAM Review 56 (2014).
- D. P. Kingma, J. Ba, *Adam: A method for stochastic optimization*, ICLR (2015).
