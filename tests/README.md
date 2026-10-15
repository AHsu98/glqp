# Numerical regression tests

Install GLQP, pytest, and a working MKL runtime, then run from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q
```

When MKL is installed with pip inside a Linux virtual environment, include that
environment's `lib` directory in `LD_LIBRARY_PATH` before starting Python.

These tests use small analytical solutions, independent dense linear solves, and
controlled search directions. They cover zero right-hand sides, changing sparse
patterns, fractional quadratic merit, backtracking and failed searches, finite
trial objectives, warm-start validation, empty inequalities, and reported
objectives. Stopping tests also cover accumulated complementarity, amplification
of slack-consistency errors by large duals, cancellation of signed products, and
objective scaling. Slack-repair tests check that logged residuals and the next
Newton right-hand side match the adjusted iterate. They require no downloaded
benchmark problems.

Performance and the broader known-bug inventory are tracked separately in
[glqp-bench](https://github.com/AHsu98/glqp-bench). Run comparisons between committed
revisions with the same settings, problem inputs, and machine. Passing these
focused tests does not imply that every objective class or benchmark problem is
correct.

`test_linear_accuracy.py` verifies reported residuals against independent matrix
products, retention of a better iterate after a rounded correction, explicit
inaccurate status, bounded refinement, the first-polish trigger and budget,
actual factorization shifts, timeout and
error propagation, and the linear diagnostics exposed in solver results. See
[the linear-solve contract](../docs/linear-solves.md) for the accuracy policy and
backend boundary.

`test_glm_objectives.py` checks Gaussian, logistic, and Poisson gradients and
Hessian diagonals against finite differences, including nonuniform and zero
weights, a single observation, and padded predictor vectors. It also checks the
standard Poisson likelihood scale, ignores extreme padded coordinates, and
compares constrained Gaussian and Poisson solves with independent dense KKT and
scalar reference solutions. Those solve tests verify original KKT conditions and
reported objective values, including the quadratic and linear terms.
