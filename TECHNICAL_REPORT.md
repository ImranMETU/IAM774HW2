# Forward and inverse heat transfer

**Technical report · November 19, 2026**

On this one-dimensional problem, Crank–Nicolson achieved smaller forward field error at much lower solve cost than PINN training. The inverse PINN recovered thermal diffusivity within 0.74% in five runs. Across the three observation datasets, analytical nonlinear least squares recovered a more accurate temperature field in all three comparisons and a closer diffusivity in two.

## Governing model

On $x,t\in[0,1]$, the nondimensional temperature excess satisfies $u_t=\kappa u_{xx}-\lambda u$, with $u(0,t)=u(1,t)=0$ and $u(x,0)=\sin(\pi x)+a\sin(4\pi x)$. The illustrative coefficients are $\kappa=0.030$, $\lambda=0.875$, and $a=0.60$. Each Dirichlet eigenfunction $\sin(n\pi x)$ has eigenvalue $-(n\pi)^2$. Its amplitude therefore obeys $\dot A_n=-r_n A_n$ with $r_n=\kappa(n\pi)^2+\lambda$. The initial amplitudes give

$$
u(x,t)=e^{-r_1t}\sin(\pi x)+a e^{-r_4t}\sin(4\pi x).
$$

Here $r_1=1.171088$ and $r_4=5.612410$. The fourth mode decays on a time scale of 0.178, so early profiles retain most of the short-wavelength signal.

## Crank–Nicolson benchmark

With centered second differences $D_{xx}$ on the interior, define $A=\kappa D_{xx}-\lambda I$. The update solves $(I-\Delta t A/2)u^{j+1}=(I+\Delta t A/2)u^j$. The solver factors the tridiagonal system once per mesh and advances the full time history. Each refinement halves both step sizes with $\Delta t=\Delta x/2$.

| Spatial intervals | Time steps | Relative L² at t = 1 | Order | Median solve time |
|---:|---:|---:|---:|---:|
| 50 | 100 | 1.8744e-4 | — | 2.16 ms |
| 100 | 200 | 4.6573e-5 | 2.009 | 5.00 ms |
| 200 | 400 | 1.1625e-5 | 2.002 | 14.22 ms |
| 400 | 800 | 2.9052e-6 | 2.001 | 28.30 ms |

Errors in the refinement table use the native spatial nodes at $t=1$. The finest mesh also contains the 101 × 101 comparison nodes and gives a full-grid relative L² error of $1.269\times10^{-5}$. All field relative L² errors are Euclidean norm ratios with equal weight per grid sample.

## Forward PINN

The float64 MLP maps $(x,t)$ through three width-32 tanh layers to one scalar output. The soft loss adds equally weighted means of the squared PDE residual $u_t-\kappa u_{xx}+\lambda u$, boundary error, and initial-condition error. Training uses fixed batches of 1,000 interior points, 50 points at each end, and 200 initial points. Automatic differentiation first obtains $u_x,u_t$ from the output and then differentiates $u_x$ to obtain $u_{xx}$. Both derivative calls retain the computation graph so backpropagation reaches the weights through the residual.

Full-batch Adam runs 6,000 steps with a geometric learning-rate decay from $2\times10^{-3}$ to $2\times10^{-4}$. One strong-Wolfe L-BFGS call follows, with a budget of 1,500 internal iterations. Histories record line-search trials separately from iterations and applied updates. All runs requested one CPU thread.

| Seed | Full-grid relative L² | Relative L² at t = 1 | Max absolute error | Training time |
|---:|---:|---:|---:|---:|
| 67507 | 6.5835e-4 | 2.1733e-3 | 2.4566e-3 | 92.86 s |
| 67508 | 9.1830e-4 | 2.4413e-3 | 3.4765e-3 | 103.78 s |
| 67509 | 7.4620e-4 | 1.9830e-3 | 2.2751e-3 | 96.27 s |

The mean full-grid error was $7.743\times10^{-4}$ with sample SD $1.322\times10^{-4}$. Training losses measure the objective at sampled points. Field errors evaluate against the analytical solution on the common grid, including the prescribed boundary and initial slices. A separate residual check at 10,000 interior points disjoint from training gave RMS values of 0.00300–0.00458. The Crank–Nicolson timings are medians of seven solves and include factorization and time stepping. PINN timings include optimizer training. Inference runtime was unmeasured, so these costs characterize constructing the solutions.

## Inverse estimation

The inverse models train $s=\log\kappa$ jointly with the weights from $\kappa_0=0.10$, with $\lambda,a$ known. Adam uses a constant learning rate of 0.01 for $s$. A fourth mean squared loss fits 100 noisy observations. Their coordinates are uniform on the unit square, with independent Gaussian errors of SD $0.01\,\mathrm{std}(u_{\mathrm{clean}})$ computed with the population convention. The PDE batch combines the observation coordinates and 500 extra interior points. Training uses the prescribed physics, initial data, and noisy readings. The analytical field supplies the synthetic readings and the reference grid. A SciPy nonlinear least-squares fit to the same readings uses the known analytical solution as its model.

| Observation seed | Noise SD | PINN κ | Least-squares κ | PINN field relative L² | Least-squares field relative L² |
|---:|---:|---:|---:|---:|---:|
| 67507 | 0.00240337 | 0.02978088 | 0.02983911 | 1.724e-3 | 1.079e-3 |
| 67508 | 0.00315487 | 0.03001855 | 0.02999201 | 1.148e-3 | 5.341e-5 |
| 67509 | 0.00268008 | 0.03005657 | 0.03007856 | 8.347e-4 | 5.248e-4 |

These paired fits keep the PINN initialization fixed. Observation RMSE is an in-sample fit statistic. On two datasets the PINN had smaller observation RMSE while least squares gave smaller independent field error. For the original inverse experiment, the PINN trained for 83.18 s and the signed parameter errors were −0.7304% for the PINN and −0.5363% for least squares.

## Sensitivity and limitations

With the first observations and extra collocation points fixed, three network initializations produced mean diffusivity 0.02980365 and sample SD $3.171\times10^{-5}$. The corresponding mean for three observation datasets with initialization fixed was 0.02995200, with sample SD $1.494\times10^{-4}$. Changing coordinates and noise draws together gave about 4.7 times the diffusivity spread from changing the weights alone. The five unique inverse runs completed. Initialization seed 67508 showed the largest transient, ranging from 0.00417 to 0.23062 during Adam. L-BFGS reduced the remaining error.

The evidence covers one smooth equation, known loss and initial profile, one noise level, and three seeds per comparison with a shared baseline. The observation experiment varies both measurement placement and noise and cannot isolate those effects individually. The exact model gives least squares a strong reference for this inverse task, while Crank–Nicolson offers the more accurate and cheaper forward solve in the measured setting. A trained PINN provides a differentiable field approximation with joint coefficient estimation. These results do not measure generalization across equations or hardware.

[Archived data and figures](README.md#results) · [Reproduction and audit](REPRODUCIBILITY.md) · [References](README.md#references)
