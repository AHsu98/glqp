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
objectives. They require no downloaded benchmark problems.

Performance and the broader known-bug inventory are tracked separately in
[glqp-bench](https://github.com/AHsu98/glqp-bench). Run comparisons between committed
revisions with the same settings, problem inputs, and machine. Passing these
focused tests does not imply that every objective class or benchmark problem is
correct.
