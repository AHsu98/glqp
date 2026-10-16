## Quadratic Programs + GLM

This package solves optimization problems of the form

$$
\begin{aligned}
\text{minimize}\quad & f(Ax)+\tfrac12x^{\mathsf T}Qx-b^{\mathsf T}x \\
\text{subject to}\quad & Ex = e,\\
& Cx \le c,
\end{aligned}
$$

for positive semidefinite $Q$ and $f$ a separable convex objective, e.g. $f(z)=\sum_{i}f_i(z_i)$
using an interior-point method.

We form the Hessian of $f(Ax)$ explicitly and compute the term $A^{\mathsf T} D A$ efficiently with **sparse\_dot\_mkl**.

All matrices should be supplied as `scipy.sparse.csc_array`; the library converts inputs to this format internally when needed.  
Parts of the implementation were inspired by [[1]](#1), and conversations with its first author were especially helpful, and useful ideas were taken from [[3]](#3), [[4]](#4), [[5]](#5) and [[6]](#6).

#### Requirements
The main additional requirement comes from [sparse_dot_mkl](https://github.com/flatironinstitute/sparse_dot), 
needing the associated MKL dependencies.
These are packaged with Conda, and can be installed into Miniconda with `conda install -c intel mkl`.

In a later update, these may be made optional, but it speeds up formation of the Hessian significantly.

See https://github.com/flatironinstitute/sparse_dot for more details.

QDLDL is the default KKT factorization. The [backend guide](docs/linear-backends.md)
describes optional SuperLU, FERAL, and PARDISO alternatives with explicit
ordering/scaling options and shared verified float64 GMRES refinement.

#### Numerical experiments

- [Standalone C++ QP presolve](docs/presolve.md): optional structural reductions,
  objective offsets, and original primal/dual recovery, available through a
  C ABI, CLI, and Python adapter.
- [Iteration work reuse](docs/iteration-work.md): matvec, transpose, curvature,
  and allocation optimizations with unchanged acceptance rules.
- [Verified linear solves](docs/linear-solves.md) and
  [factorization options](docs/linear-backends.md): separate factor reuse from
  fresh-residual accuracy checks.

The sibling `glqp-bench` repository contains committed-source comparison
campaigns, explicit per-variant presolve/backend settings, and 39 GLM examples,
including large sparse and weighted real-data problems.
Routine tests use eight independent one-thread workers; serial and multithreaded
per-problem profiles are explicit alternatives. The 2026-10-16 checkpoint
retains 169/178 QP successes with byte-identical accepted primal/dual vectors
and 23.7% lower paired geometric mean wall time than the original accepted
float64 GMRES implementation. All 39 GLMs pass under the explicit `glm-v2`
profile; see the [Poisson trial-budget diagnosis](docs/poisson-linesearch.md).
Presolve and backend changes remain opt-in; their gains depend on the problem,
and the presolve guide documents a reproduced convergence regression and an
explicit pass option that avoids it.

#### Convergence tolerance

An `optimal` result requires the existing absolute maximum-norm KKT residual to
be at most `SolverSettings.tol`, together with

```text
sum(abs(y * (c - C @ x))) / max(1, abs(primal_objective)) <= tol
```

The additional condition controls total complementarity using the actual
constraint slack at the returned `x`. A small largest product alone can hide
many accumulated errors; the stored interior slack can also differ slightly
from `c - C @ x`, which matters when dual values are large. Absolute products
prevent cancellation. Problems without inequalities have zero aggregate
complementarity. Nonfinite values cannot satisfy this condition.

This supplements the original absolute KKT test; its tolerance, Newton equations,
and line-search settings are unchanged. Some problems consequently require more
iterations or reach their iteration/time limit before receiving `optimal` status.

#### Historical benchmark
I ran a benchmark on the MM problems [[7]](#7) starting with the experiments and code from the assocated repository for [[1]](#1) in this repo:
https://github.com/AHsu98/glqp-benchmark
That earlier benchmark reported completion on 133/138 with tuned QP settings.
It predates the current independent correctness checks and should not be mixed
with the versioned comparisons in `glqp-bench`.

## References
<a id="1">[1]</a>
Chari, G. M., & Açıkmeşe, B. (2025). QOCO: A Quadratic Objective Conic Optimizer with Custom Solver Generation. arXiv preprint arXiv:2503.12658. Retrieved from https://arxiv.org/abs/2503.12658

<a id="2">[2]</a> 
Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). OSQP: an operator splitting solver for quadratic programs. Mathematical Programming Computation, 12(4), 637–672. doi:10.1007/s12532-020-00179-2

<a id="3">[3]</a> 
Pougkakiotis, S., Gondzio, J. An interior point-proximal method of multipliers for convex quadratic programming. Comput Optim Appl 78, 307–351 (2021). 
https://doi.org/10.1007/s10589-020-00240-9

<a id="4">[4]</a> 
Schwan, R., Jiang, Y., Kuhn, D., & Jones, C. N. (2023). PIQP: A Proximal Interior-Point Quadratic Programming Solver. arXiv [Math.OC]. 
Retrieved from http://arxiv.org/abs/2304.00290

<a id="5">[5]</a> 
Friedlander, M.P., Orban, D. A primal–dual regularized interior-point method for convex quadratic programs. 
Math. Prog. Comp. 4, 71–107 (2012). https://doi.org/10.1007/s12532-012-0035-2

<a id="6">[6]</a> 
Ghannad, A., Orban, D., & Saunders, M. A. (2021). Linear systems arising in interior methods for convex optimization: a symmetric formulation with bounded condition number. 
Optimization Methods and Software, 37(4), 1344–1369. https://doi.org/10.1080/10556788.2021.1965599

<a id="7">[7]</a> 
Maros, I., & Mészáros, C. (1999). A repository of convex quadratic programming problems. 
Optimization Methods and Software, 11(1–4), 671–681. https://doi.org/10.1080/10556789908805768




