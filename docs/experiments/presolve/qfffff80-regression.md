# QFFFFF80 presolve regression diagnosis

The full 178-problem comparison of solver commit `5b0002e` with and without
presolve retained 169 accepted problems on both sides, but QFFFFF80 slowed
from 0.09277 to 2.67523 seconds. This is an iteration-count regression caused
by the singleton-equality reduction path, not native presolve overhead.
Presolve remains optional; no input-specific rule or default was changed.

The focused diagnostic froze the same solver commit and benchmark commit
`ac314ade1d1e27c43ca36d135b46b50b73f0cbde`. It used three alternating cold
workers per variant, pinned to CPU 8 with one numerical thread, and the same
`benchmark-v1` settings, 60-second limit, solver tolerance `1e-7`, and
original-data validation tolerance `1e-6`. These diagnostic CPU-8 timings
must not be pooled with the normal parallel-8 measurements.

| QFFFFF80 configuration | Variables/equalities/inequalities | Iterations in every repeat | Median wall seconds |
| --- | --- | ---: | ---: |
| No presolve | 854 / 350 / 1028 | 68 | 0.16961 |
| All existing presolve passes | 817 / 313 / 980 | 1751 | 5.34285 |
| `redundant_bounds=False` | 817 / 313 / 991 | 1766 | 5.41041 |
| `singleton_equalities=False` | 854 / 350 / 1028 | 68 | 0.17093 |

The full presolve slowdown reproduces at 31.5 times the baseline wall time
on CPU 8. Its median presolve and postsolve costs are only 1.03 and 0.11 ms.
Keeping the 11 redundant singleton inequalities does not resolve it.
Disabling singleton-equality elimination leaves this problem unchanged,
restores all 68-iteration runs, and returns bitwise-identical original primal
and dual solutions. The existing explicit option provides an opt-out; it
also forgoes the useful equality reductions on other inputs and is not a
new recommended global default.

The equality pass fixes 37 variables: 19 at zero and 18 at nonzero values.
It removes 37 pivot equalities and 37 inequalities that become constant
(19 with zero right-hand side and 18 with positive right-hand side).
The further 11 inequality removals are duplicates exposed by fixing variable
170 to zero. Their original row indices are
`855, 858, 861, 864, 867, 870, 873, 874, 878, 881, 884`.
All fixed linear objective coefficients are zero, and the objective offset
is zero. The retained audit lists every fixed value and removed row.

A further single structural ablation keeps the 817 reduced variables and
313 equalities, disables redundant-bound removal, and restores all 37
constant inequalities at their original row positions. This preserves all
1,028 original inequality positions and their barrier terms. It still takes
**1,785 iterations**, compared with 1,766 when those constant rows are removed
and 68 on the original model. Original-coordinate recovery and independent
KKT validation pass. Its single model-and-solve elapsed time was 5.914 seconds
on CPU 8 with one thread; this is a trajectory diagnostic, not a repeated
performance estimate or a standard parallel-8 result.

The slow trajectory therefore persists after restoring the constant rows;
retaining them alone is not a remedy, and this evidence does not justify
adding an empty-row option. The experiment isolates the singleton-equality
elimination path, but does not identify one particular equality or the
underlying conditioning or trajectory mechanism. No additional heuristic,
option, or numerical code change was made.

The moderate DUAL2/DUAL4 slowdowns from the single-observation campaign did
not reproduce at that size. Both models remain unchanged by presolve, retain
19 and 18 iterations respectively, and give bitwise-identical primal and
both dual arrays in all paired repetitions. Median wall times were
25.879→26.120 ms for DUAL2 (+0.93%) and 22.530→22.667 ms for DUAL4 (+0.61%).

All 30 saved solutions pass an independent replay of original-data KKT
validation and match the recorded residuals exactly. The later constant-row
ablation adds one independently validated solution. The raw experiment is
`glqp-bench/runs/research2-20261016-presolve-slowdown-cpu8`:

- `diagnostic.json`: frozen source identities, protocol, options, records,
  timings, and verified source integrity.
- `driver.py`: reproduces the three cold-worker comparisons.
- `audit.py` and `audit.json`: source/input/record/solution hashes, independent
  residual checks, unchanged-solution comparisons, and removed structure.
- `source/benchmark/runs/{full,keep_bounds,keep_equalities}`: all original
  records, logs, manifests, and numerical solutions.
- `restore_constant_rows.py` and `restored_constant_rows/`: the structural
  ablation driver, exact reduced model, full iteration history, recovered
  solution, source/artifact hashes, and independent audit.

The checked-in [compact evidence](qfffff80-regression.json) retains source,
record, solution, and driver hashes, all timing observations, structural
details, and validation maxima. The [reproduction driver](diagnose_slowdown.py)
accepts `--workspace` and a fresh `--output` directory, and reads the frozen
source identities from the original full campaign before exporting them.
