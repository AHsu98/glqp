# Hessian contributions and KKT storage

The iteration loop always requests `get_H(z)`. The objective Hessian is the
stored quadratic matrix `Q` plus an optional GLM contribution. Setup selects a
no-contribution callable when the GLM is absent; it returns `None`, so `get_H`
returns `Q` directly without evaluating a dummy sparse product or adding zeros.
For a GLM, the contribution is evaluated at the current predictor:
`A.T @ diag(f.d2f(z)) @ A`. Callers must treat the returned Hessian as read-only.
There is no separate QP iteration loop or general constant-GLM-Hessian cache.

`KKTWorkspace` owns a full symmetric CSC matrix and the right-hand side for one
solve. Its structure contains the Hessian, both scaled inequality blocks, both
equality blocks, and every diagonal slot, including numerical zeros. Entry maps
allow subsequent assemblies to overwrite values without allocating sparse
blocks. The inequality scaling and proximal regularization equations are
unchanged:

```text
w = sqrt(y + rho)
K = [[H + rho*I,       C.T*diag(w),              E.T],
     [diag(w)*C, -diag(s + rho*y + rho**2),        0],
     [E,                       0,           -rho*I]]
rhs = [-rx, (1/w)*rc - w*rp, -req]
```

The workspace does not mutate supplied matrices. It canonicalizes duplicate
and unsorted sparse entries on private copies when needed. Hessian pattern
changes, including cancellations in GLM curvature, rebuild the structure and
maps; numerical changes with an unchanged pattern reuse the storage. Missing
Hessian diagonal entries are reset before adding `rho`, so shifts cannot
accumulate. Constraint matrices are fixed within one solve. Initialization
discards the old workspace, so matrix replacements between solves are read
afresh (with the original problem dimensions).

Assembly returns borrowed buffers, valid until the next assembly. Internal
consumers finish using them before the next iteration. A caller retaining an
assembly for diagnostics must copy it.

The linear solver still receives the full matrix and applies its numerical
factorization shift separately. Refinement continues to target the requested
matrix, including proximal regularization but excluding the numerical shift.
Backend storage, triangular extraction, factorization, and residual safeguards
are unchanged. See [linear solves](linear-solves.md).
