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
Parts of the implementation were inspired by [[1]](#1), and conversations with its first author were especially helpful, and useful ideas were taken from [[3]](#3), [[4]](#4) and [[5]](#5).

#### Requirements
The main additional requirement comes from [sparse_dot_mkl](https://github.com/flatironinstitute/sparse_dot), 
needing the associated MKL dependencies.
These are packaged with Conda, and can be installed into Miniconda with `conda install -c intel mkl`.

In a later update, these may be made optional, but it sepeds up formation of the hessian significantly.

See https://github.com/flatironinstitute/sparse_dot for more details.

We solve KKT systems using [qdldl](https://github.com/osqp/qdldl-python)[[2]](#2).

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






