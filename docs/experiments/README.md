# Experiment ledger

Status: 2026-10-20. This index records implemented changes, completed experiments,
and working decisions. Dated reports remain the authority for their particular
source revisions, settings, observations, and limitations. An option's presence
in the solver does not mean that it has been selected as a default.
See the [accepted algorithm reference](../accepted-algorithm.md) for its exact
equations, control flow, and public versus benchmark settings.

**Keep accepted Newton, legacy outer policies, QDLDL, float64 right-GMRES
refinement, and presolve disabled as the working defaults.** The latest full
Maros–Mészáros comparison finds 129/138 accepted solutions for that configuration
and 127/138 for centered adaptive proximal Newton. The latter has useful large
individual gains, but loses four previously accepted LISWET cases. On the
separately audited 122-problem timing subset it is 19.6% slower geometrically
on the 113 common successes. This supersedes any inference of a general gain
from its earlier eight-problem screen; the screen itself remains valid evidence.
See the [full Newton checkpoint][newton-full].

The subsequent [GLM stress checkpoint][glm-stress] expands the catalog from 39
to 81 without changing prior problem definitions. Accepted source `850a6f5`
repairs weighted logistic tails and zero-weight Poisson evaluation. Both old
and corrected sources pass 76/81 under the repaired independent validator,
with identical counts on every case. Three severe Poisson cases fail their
initial line search; two weak-ridge cases fail only the external gap bound.
These results motivate separate globalization and certification experiments,
not a universal replacement of sparse-MKL Gram formation.

The accepted [cached scalar search](../glm-scalar-search.md) at `af8e5bb` now
replaces `.9` backtracking for eligible GLM descent directions. All 76/81
certificates are retained at native `1e-7`; the three severe Poisson cases now
reach native convergence, but still fail the external gap bound. A matched
severe12 control at native `1e-9` improves 9/12 to 12/12. Initial severe Poisson
searches take two trials instead of exhausting 100, without extra sparse
matvecs. The clean final parallel-8 sweep observes about 2% lower wall time on
76 common successes; this single short-case sweep is not a broad timing claim.
The [checkpoint](../../../glqp-bench/benchmarks/progression/scalar-search-20261018/README.md)
retains both flagged earlier timing attempts, numerical revalidation, and raw
evidence. QP searches, regularization, and barrier updates are unchanged.

The [QP search follow-up](../qp-search-experiments.md) tests scalar proposals,
exact complementarity rejection, and bounded affine residual caching separately.
All retain 12/12 certificates on the selected panel. Scalar proposals leave
the long trajectories unchanged; the bounded cache adds work on QPCBOEI2.
No QP search default changes, and no full-suite coverage claim follows from
this screen. The [four-round checkpoint](../../../glqp-bench/benchmarks/progression/qp-scalar-search-20261018/README.md)
retains the excluded timing attempt and diagnostic parity evidence.

The [full scalar-proposal reliability follow-up](../../../glqp-bench/benchmarks/progression/qp-scalar-full-20261018/README.md)
retains the same 129/138 certificates and nine failures as backtracking, with
residual screening disabled. Full-run timings are excluded after four host
flags. QP restoration/fallback acceptance is preserved; these results do not
validate replacing it with the GLM merit-only descent policy.

The subsequent [search consolidation][search-consolidation] at `f9ceb2e` shares
trial scheduling, cached-ray interfaces, and safeguarded proposals while
retaining explicit acceptance policies and all defaults. All 277 frozen-source
comparisons match `606ae0d` exactly: selected QP modes, all 81 GLMs, severe12 at
native `1e-9`, legacy-search controls, and Newton/PC/proximal small cases.
The check covers 78,922 non-time history rows and full diagnostics on 179 cases;
the same five deliberately exercised legacy/fallback failures also match.
Solver tests: 823. This is untimed structural verification, with no speed or
new full-138-QP coverage claim.

The [weighted Gaussian/Poisson range fix](../weighted-loss-range.md) at
`5573cae` preserves representable weighted values and derivatives outside the
range of their unweighted intermediates. `0f98254` adds opt-in original-design
diagnostics. The [operator audit](../../../glqp-bench/benchmarks/progression/glm-operators-20261019/README.md)
rules out predictor drift as the cause of the five gap failures; manufactured
controls still demonstrate that forming a Gram matrix can erase weak modes.
No normal-equation or fallback policy changes follow from those controls.

The [613-solve stopping study](../../../glqp-bench/benchmarks/progression/glm-stopping-20261019/README.md)
separates outer stopping, inner accuracy, proximal regularization, and the
barrier floor. A certificate veto followed by tighter inner accuracy recovers
81/81 certificates in 1,059 total iterations, versus baseline 76/81 in 1,050;
all previously certified final arrays remain byte-identical. Merely tightening
the linear solve recovers none. A separate strict-tolerance control needs a
constraint-count-aware barrier floor to avoid six stagnation tails. These are
untimed experiments, with no production convergence hook or changed defaults.

The [QP absolute-floor follow-up](../../../glqp-bench/benchmarks/progression/qp-floor-20261020/README.md)
tests `min_mu=min(1e-12,.1*tol/k)` without changing the separate `.01*KKT`
floor. At native `1e-7`, all 14 affected MM cases retain nine certificates
and five timeouts; four unchanged controls also pass. Completed iteration
counts are unchanged. A separately declared strict12 diagnostic at `1e-9`
and shared stagnation window 100 retains 12/12 certificates, improves native
optimal 6→11, and reduces total iterations 979→483. LISWET2 stays stagnant
with identical iterates because its absolute floor never activates.
Primary timings are excluded after a host flag; one clean strict sweep has
wall ratio .59786, scoped to this selected diagnostic panel. No default changes.

## Measurement scopes

| Evidence family | Machine and execution | Scope and interpretation |
| --- | --- | --- |
| Historical correctness, storage, and original float64-GMRES studies | AMD EPYC 9V74 cloud machine; serial cold workers on CPU 0, one numerical thread; five visible CPUs with a four-CPU quota | Full QP campaigns contain 138 MM problems plus 40 inherited applications. Early full campaigns use three repetitions, later storage/GMRES campaigns one. Coverage and means belong to each report's repetition rule. |
| Subsequent implementation, presolve, backend, and outer-policy work | Intel i7-14700; routine `parallel-8`, performance CPUs 0–7, one numerical thread per worker, fixed waves and rotating variant order | Full 178-QP and 138-MM campaigns are distinct. Small backend and algorithm screens are selected cohorts, not full-suite estimates. Serial and multithreaded studies are separately identified. |
| Guarded comparator and later algorithm studies | Same i7-14700 policy, with a two-second launch pause outside measurement and all other agent compute/editing stopped | Host-load audits select the first complete clean whole wave. Failed isolation attempts remain retained. Diagnostic traces and flagged-wave times do not enter timing ratios. A clean subset is explicitly labeled as a subset. |

Unless a report explicitly says otherwise, the native tolerance is `1e-7`, the
independent original-data certificate is `1e-6`, and the solve wall budget is
60 seconds. The competitor comparison also has a separate native-`1e-9` panel;
its common certificate remains `1e-6`. Native stopping definitions differ across
solvers. A passing native status alone is not the benchmark acceptance rule.

Timing ratios use each stated common-success cohort. Failures remain in
performance-profile denominators and never become speedups. Do not multiply
ratios from successive screens, pool absolute times across machines or cohorts,
or interpret warm component profiles as cold complete-solve measurements. The
[execution policy][execution], [isolation policy][isolation], and
[historical progression][history] retain the precise conventions.

## Correctness and implementation

| Work | Result and working decision | Evidence |
| --- | --- | --- |
| Correctness blocks: zero RHS, factor reuse after pattern changes, fractional quadratic merit, backtracking, warm starts, empty inequalities, returned/logged objectives | Retain the fixes. The early blocks alone exposed more independent complementarity failures; their historical coverage losses are preserved rather than presented as performance gains. Subsequent stopping work addresses that mismatch. | [Original two-block results][correctness] |
| Original-slack aggregate complementarity stopping; refresh residuals after slack repair | Retain the original-data stopping check and consistent post-repair residuals. The paired stopping campaign recovers 49 within-budget successes, 116/178 → 165/178, without relaxing validation. The slack refresh is a correctness change; its targeted LISWET10 run incurs more iterations. | [Stopping correction][termination], [slack residual repair][slack] |
| Gaussian and Poisson objective/derivative consistency | Retain Poisson's corrected objective scale and observation-only Gaussian/Poisson curvature handling. The frozen analytical inventory improves from 7/10 to 10/10. This correctness result did not itself establish broad GLM performance. | [GLM correctness checkpoint][glm-fixes] |
| Shared Hessian contribution and preallocated KKT storage | Retain value/storage reuse with pattern checks and fresh coefficients. The historical EPYC comparison preserves all 166 common successful primal/dual arrays and counts; geometric solve time falls 45.5% on that fixed cohort. | [Storage API](../kkt-storage.md), [storage progression][history] |
| Projected iterative refinement → bounded float64 right GMRES | Accepted working refinement strategy: one cycle, at most ten steps, two-pass orthogonalization, stored preconditioned directions, and fresh float64 requested-system residuals. The EPYC paired campaign improves 167/178 → 169/178, with no lost success and a 5.9% geometric wall-time cost on 167 common successes. One gain, CVXQP2_L, is close to the time cutoff. The earlier extended-residual experiment remains separate. | [Linear solve contract](../linear-solves.md), [float64 campaign][gmres-f64], [earlier extended-residual trial][gmres-extended] |
| Five rounds of per-iteration implementation work | Retain accepted-iterate objective/gradient/product reuse, invariant line-search dot products, sparse transpose views, skipped unused rejected-trial work, per-solve constant Gaussian curvature, QDLDL triangular buffers, and shifted-CSC workspace reuse. Fresh accepted-point arithmetic and requested-system residual verification remain protected by parity tests. | [Operation counts and round-by-round profiles](../iteration-work.md), [first implementation checkpoint][research1], [second checkpoint][research2] |
| Direct final implementation comparison | The i7-14700 full 178-QP comparison against original float64-GMRES `3f27e71` preserves 169/178 successes and all 169 accepted primal/dual bytes. Geometric wall/solve time falls 23.7%/26.4%. This remains the last accepted broad implementation gain; it is a direct comparison, not a product of smaller-screen ratios. | [Measured source `4154fa9`, numerical changes through `663374c`][research2] |
| Policy and documentation cleanup | Shared direction clipping, settings validation, and history formatting; named experimental constants and clarified frozen-model ownership. Source `f179f1d` preserves exact numerical output on 112 small QP/GLM cases and seven selected MM trajectories; 556 solver and 104 benchmark integration tests pass. Quiet solves now suppress the old slack-repair print. This is untimed verification, with no performance claim or algorithm promotion. | [Cleanup evidence][cleanup] |
| Shared objective search machinery | One trial controller and safeguarded proposer; cached GLM/legacy merit adapters retain arithmetic and acceptance policies. Source `f9ceb2e` matches all 277 baseline observations exactly, including failed searches, iteration histories, and diagnostics. Accepted-point products remain owned by their trial; defaults are unchanged. | [Implementation map](../accepted-algorithm.md#search-implementation), [parity evidence][search-consolidation] |

The two regularization mechanisms have different contracts. Outer coefficients
modify the requested Newton/proximal equations. Factorization-only `tau_reg`
stabilizes the inverse used by refinement and is excluded from the requested
residual target. Turning `prox_reg` off does not turn `tau_reg` off. Backend
experiments and the later reusable two-RHS implementation preserve this distinction.

## Presolve, linear backends, and benchmark expansion

| Work | Result and working decision | Evidence |
| --- | --- | --- |
| Standalone C++17 presolver and first three implementation rounds | Keep opt-in. Dependency-free library, C ABI, CLI, and Python adapter support exact singleton equalities, fixed bounds, redundant/empty rows, objective offsets, and original primal/dual recovery. Conservative ambiguous-arithmetic fallback, cheaper sparse checks/packing, and recovery work were tested separately. A repeated reduction-heavy 12-QP panel has wall ratio .6653; a later broad 178-QP sweep has .9608 with unchanged 169/178 coverage. These are different cohorts. | [Presolver API](../presolve.md), [initial experiments][research1], [broad follow-up][research2] |
| Local fallback and scalar quadratic elimination | `local_fallback=True` preserves independent certified reductions around ambiguous rows. `separable_quadratics=True` adds certified scalar minima and bound-dual recovery; a further storage pass removes unnecessary recovery tape. Both remain opt-in. Scalar elimination removes 6,546 additional variables across nine QPs. Final full-MM coverage stays 129/138; wall time is 4.5% lower versus presolve off in one sweep, with only about 1% incremental gain over local presolve across the whole common cohort. | [Scalar-pass implementation, compatibility, sanitizer, and timing evidence](presolve/scalar-quadratic-report.md) |
| Presolve counterexamples | Repeated STCQP1/STCQP2 gains of 3.67×/8.12× coexist with an 11% PRIMAL4 scalar-pass slowdown and the approximately 29× QFFFFF80 equality-elimination slowdown. QFFFFF80 changes 68 → 1,751 iterations; retaining singleton equality rows restores 68. Restoring removed constant inequality rows does not cure it. No automatic problem-name switch or default promotion. These are performance regressions, not new certificate losses in the measured full sweep. | [QFFFFF80 diagnosis](presolve/qfffff80-regression.md), [repeated scalar results](presolve/scalar-quadratic-report.md) |
| Interchangeable sparse factors | Keep QDLDL default; expose `qdldl`, `superlu`, `feral`, and `pardiso` behind explicit lifecycle/options interfaces. Pattern changes rebuild factors; fixed-pattern numeric updates reuse storage where supported. All receive the same GLQP projection, right-GMRES, and fresh residual checks. FERAL and pivoted alternatives factor a captured QSCFXM1 system that rejects fresh QDLDL construction; SuperLU exceeds a 90-second BOYD2 diagnostic budget. These captured-system results do not establish full-solver coverage or a universal replacement. | [Backend interface and options](../linear-backends.md), [fixed-system investigation][backends] |
| PARDISO and threading | Useful selected wins, not a universal replacement. The 12-QP end-to-end screen retains 7/12 successes with either backend; PARDISO's common-success wall ratio .6821 is driven by CVXQP2_L and EXDATA, while several other cases slow down. Disabling matching/scaling helps a captured BOYD2 factorization but does not recover its full solve. The small hard-problem threading studies do not justify a general eight-thread configuration. | [Complete backend scope and outcomes][backends], [broader panel][research2] |
| POUNCE investigation | Pinned POUNCE/FERAL sources informed structure/value separation, factor lifecycle, cached scatter storage, and explicit refinement ownership. FERAL was tested through its Python engine inside GLQP. POUNCE itself, its factor-quality escalation/expected-inertia policies, structured Schur/block paths, multi-RHS batching, and licensed MA57 were not measured end to end. | [Pinned implementation review and scope](../linear-backends.md#what-was-learned-from-pounce) |
| GLM suite | Retain 39 canonical Gaussian/logistic/Poisson problems: unconstrained, box, and affine variants; small and larger sparse synthetic data; Airfoil, WDBC, bike counts, and Adult data. The expansion preserves the original 27 canonical hashes. Independent references check selected larger/real-data cases against original KKT and convex dual bounds. These are optimization tests, not predictive-accuracy studies. | [Initial 27 GLMs][research1], [39-case expansion and references][research2] |
| Poisson line-search profile | Keep zero initialization and the original problem definitions. Bike counts first pass the old Armijo search at trial 55; `glm-v2` explicitly raises the cap to 100. The old profile and public default remain unchanged. This is a benchmark-profile change, not an implementation speedup. Centered experiments later use a different residual merit and halving backtracking, so their reduced trial counts are not an isolated cap effect. | [Poisson trace](../poisson-linesearch.md), [centered Newton GLM traces][newton-screen] |

## Outer-rule and proximal experiments

All configurations below remain explicit experiments. Aliases such as
`glqp_candidate` and `glqp_current` change meaning between campaigns; consult
the saved settings and report labels rather than inferring the algorithm from
the alias. The ordinary-Newton default was preserved during PC development.

| Experiment | Finding and decision | Evidence |
| --- | --- | --- |
| Previous-target `0.1*mu` guard | Implemented the intended floor using the previous target; the legacy term referred to the new proposal and was redundant. Both retain 129/138 MM and 39/39 GLM certificates. Geometric iteration counts rise 12.2%/10.5%; YAO improves but LISWET2/6 worsen severely. Keep `barrier_update_guard="legacy"` by default and `"previous"` opt-in. Full-QP and GLM timings are excluded after host flags. | [Guard semantics and diagnostics](barrier-guard.md), [measured guard comparison][guard] |
| Initial target and KKT barrier floor | Complementarity-based initialization removes the worst fixed-100/previous-guard interaction. Mean/legacy retains 22/22 quick QPs/apps and 39/39 GLMs with modest iteration reductions, but worsens LISWET1 and YAO. Smaller floors and combined initialization/floor trials are mixed. Keep fixed initialization and floor .01; frequent floor activation alone does not demonstrate causation. | [Explicit policy controls](outer-policy-controls.md), [two-round comparison][outer] |
| Slack repair and Newton fallback | Guarded repair prevents a measured QPCBOEI2 residual increase but saves only one iteration; a later fallback still increases its residual. Progress-gated/off fallback each lose LISWET1 and QPCBOEI2. Increasing the progress search cap from 40 to 100 rescues neither. Keep legacy repair and finite fallback in the accepted path; retain experimental controls and telemetry. | [Repair and failed-direction evidence][outer] |
| Predictor–corrector with legacy regularization | Opt-in `step_strategy="predictor_corrector"` reuses one verified factorization for affine/combined RHSs, holds the current GLM Hessian fixed across both solves, then holds the selected target through the combined solve and line search while evaluating the actual nonlinear objective. In the 22-selected-QP comparison it retains 22/22 and has a .7430 wall ratio versus accepted Newton. All 39 GLMs pass; all 26 constrained cases use fewer iterations. This was not a full 138-QP PC evaluation, and GLM timings are excluded. | [PC API](predictor-corrector.md), [method derivation](../research/predictor-corrector-design-20261016.md), [PC checkpoint][pc] |
| Removing outer regularization | With PC it sharply reduces iterations on LISWET1/QPCBOEI2/YAO but loses QSHELL, giving 21/22 on the selected QPs. Without PC it has different effects, including long QPCBOEI2/YAO trajectories. Keep regularization enabled by default; do not treat zero-rho gains as a general improvement. | [Factorial and coverage controls][pc] |
| Consistent centered proximal model | Added independent primal/dual blocks, persistent centers, standard unperturbed complementarity, matching fixed-target residual merit, and guarded repair. Controls remove only the slack perturbation or retain the old residual-based coefficient update. Initial centered PC failures motivate a bounded cross term, sufficient-descent centering rescue, and halving backtracking. Centers/shifts stay fixed within each pair/search. These safeguards are experimental, not a convergence theorem. | [Proximal options](proximal-consistency.md), [residual/Jacobian derivation](../research/proximal-consistency-design-20261017.md), [iterations and rejected probes][proximal] |
| Adaptive coefficient continuation with centered PC | Independent caps decrease after inner completion or stalled original residual progress with a substantial center offset. Final selected QPs give 7/8 versus legacy PC's 8/8: QPCBOEI2 improves 3241 → 173, QSHELL worsens 71 → 596, LISWET1 fails. All 39 GLMs pass with 275 total steps versus legacy PC's 280; removing the barrier floor separately gives 241. Keep opt-in; all GLM timings remain excluded. | [Final centered-PC checkpoint][proximal] |
| Centered adaptive Newton without PC | The eight-QP screen gives 7/8 versus accepted Newton's 8/8, with QPCBOEI2 3322 → 111 and YAO 2555 → 328; QSHELL worsens and LISWET1 fails. The centered residual-update controls need 2282/3703 steps on QPCBOEI2/YAO, supporting a role for coefficient continuation independently of PC. All 39 GLMs pass; totals are 421 accepted Newton, 420 centered Newton, 275 centered PC. GLM timings are excluded. | [Matched Newton/PC screen and exact traces][newton-screen] |
| Full centered adaptive Newton check | **Do not promote.** On all 138 MM QPs, accepted Newton passes 129 and centered adaptive Newton 127. Gains: BOYD1/QPILOTNO. Losses: LISWET1/6/7/10. Common-success iteration ratio .949863 across 125 cases. The first-clean-wave timing subset has 122 cases and 113 common successes; its wall ratio is 1.19558, despite fewer iterations on average. This is not a full-suite speed estimate, and the additional per-step cost has not been causally profiled. | [Full coverage, separate profiles, audits, and replay parity][newton-full] |

## Attribution, remaining defects, and next checks

The [same-machine competitor comparison][comparison] tests GLQP, PIQP, Clarabel,
QOCO, and OSQP under a common original-data certificate with separate native
`1e-7` and `1e-9` panels. At native `1e-7`, their accepted MM counts are
129/108/98/120/72; at `1e-9` they are 125/134/117/123/82. These results describe
the frozen versions/options in that report, not universal rankings or identical
native tolerance semantics. Consult the [tolerance audit][tolerances] before
comparing solver-reported status or residuals.

The [speed attribution study][attribution] finds both excessive outer iterations
and substantial surrounding per-iteration work. QDLDL binding calls occupy only
3.6% of QAFIRO and 8.8% of QPCBOEI2 in their serial diagnostic, versus 81.6% of
STCQP2 and 87.0% of CONT-101. The remainder includes real array/sparse operations,
allocation, refinement, assembly, search, and logging; it was not measured as
pure Python interpreter overhead. These diagnostic fractions do not identify
the cause of the latest centered-Newton timing regression.

| Open issue or hypothesis | Established observation | Next discriminating check |
| --- | --- | --- |
| Continuation versus direction accuracy | The exact centered-Newton LISWET1 replay fails at attempt 489 after a dual-cap reduction. Ten-step GMRES is inaccurate; the raw merit slope is already positive, and clipping does not restore descent. PC has a related recorded failure. | Gate or retry coefficient reductions using requested-system and original-coordinate direction accuracy; keep the independent certificate and descent test unchanged. Recheck all lost LISWET cases separately. |
| Other full-suite LISWET losses | LISWET6/7/10 fail common complementarity after line-search termination. Their ordinary benchmark records do not establish the inaccurate-direction mechanism seen on LISWET1. | Capture exact failing states and separate solve error, center offsets, clipping, target choice, and globalization before selecting a remedy. |
| Completed barrier subproblem with an unsolved original problem | For `.5*x^2`, `x<=1`, initial `x=-1,s=2,y=1,mu0=2`, centered Newton rejects an accurate zero direction before advancing the barrier. Legacy Newton advances; changing only the initial target avoids the edge case. | Recognize inner completion and advance the barrier before the next direction, while keeping one target fixed within each solve/search. This is distinct from the measured LISWET1 failure. |
| Legacy merit-slope overflow | A retained LISWET1 capture shows overflow in `-mu/s` and nonfinite slopes for both legacy Newton and PC, including Armijo-labeled acceptances. No arithmetic fix was mixed into those comparisons. | Test overflow-safe slope evaluation and explicit finite acceptance bounds in a separate ablation. See the [captured reproducer][slope]. |
| Clipping and original-coordinate residuals | Diagnostics separate intentional regularization/correction defects from clipping-induced defects. An accurate scaled solve can coexist with a large original complementarity defect; one observed clipping step reduces that defect rather than causing it. | Check individual failing directions and residual scales. Do not attribute every failure to clipping or weaken original-slack stopping solely because stored KKT is small. |
| Centered-Newton cost on ordinary QPs | The full common-success median iteration ratio is 1; the clean timing subset is slower overall even though selected long trajectories improve. | Profile accepted-iterate/trial-gradient work and bookkeeping on representative unchanged-count cases, with numerical parity. The small targeted screen is insufficient for this attribution. |
| Presolve trajectory sensitivity | Equality elimination isolates the QFFFFF80 slowdown; its deeper mechanism is unresolved. PRIMAL4 retains its outer/refinement counts and has slightly less factor fill despite slowing under the scalar pass. | Separate changed numerical trajectory and data-layout/kernel costs with focused, matched probes; retain general pass switches and original-coordinate recovery checks. |
| More ambitious linear/globalization alternatives | Structured Schur methods, POUNCE-style factor-quality escalation, accuracy-aware refinement, and nonlinear stationarity corrections for PC were reviewed but not established as improvements here. | Declare an independent experiment and test complete solves; a fast captured factor or plausible paper mechanism is insufficient. The [ideas review][ideas] distinguishes proposals from completed work. |

The accepted paths and experimental options remain separate. In particular,
ordinary Newton does not use PC's cross-term bound or centering rescue, and
centered globalization is not an isolated change to the legacy line search.
Future experiments should name these combinations explicitly.

## Continuing an experiment

Use the [research workflow][workflow] and [handoff][handoff] in the sibling
benchmark checkout. Freeze committed solver and harness sources, preflight
settings against the exact requested solver revision, and retain explicit
options and all unsuccessful attempts. Routine changes start with focused
parallel-8 cohorts; multithreaded-per-problem work stays on a small hard panel.
Preserve the established QP benchmark stagnation window of 5000 unless changing
it is the declared experiment; public defaults and GLM profiles differ.

Run diagnostic callbacks only after timing stops, compare saved solution bytes
and iteration counts to plain observations, and exclude diagnostic time. The
full Newton study audits 356 records/348 returned solutions and confirms exact
count/status/solution parity on all 80 replay observations; unresolved timing
waves still remain excluded. Clean numerical evidence does not remove a timing
isolation failure.

Links into `glqp-bench` assume sibling checkouts in the standard workspace.
Older reports' statements such as “PC deferred” describe their dated stage;
the later ledger rows record its subsequent implementation. Historical branches,
measurements, and source snapshots are preserved rather than silently relabeled.

[execution]: ../../../glqp-bench/docs/execution.md
[isolation]: ../../../glqp-bench/docs/comparator-isolation.md
[workflow]: ../../../glqp-bench/docs/research-workflow.md
[handoff]: ../../../glqp-bench/docs/handoff.md
[history]: ../../../glqp-bench/benchmarks/progression/overview-20261015/README.md
[correctness]: ../../../glqp-bench/benchmarks/progression/correctness-blocks/README.md
[termination]: ../../../glqp-bench/benchmarks/progression/termination-20261014/README.md
[slack]: ../../../glqp-bench/benchmarks/progression/slack-20261014/README.md
[glm-fixes]: ../../../glqp-bench/benchmarks/progression/glm-fixes-20261015/README.md
[gmres-f64]: ../../../glqp-bench/benchmarks/progression/gmres10-f64-20261016/README.md
[gmres-extended]: ../../../glqp-bench/benchmarks/progression/gmres10-20261015/README.md
[research1]: ../../../glqp-bench/benchmarks/progression/numerical-research-20261016/README.md
[research2]: ../../../glqp-bench/benchmarks/progression/numerical-research-round2-20261016/README.md
[cleanup]: ../../../glqp-bench/benchmarks/progression/cleanup-20261017/README.md
[search-consolidation]: ../../../glqp-bench/benchmarks/progression/search-consolidation-20261018/README.md
[backends]: ../../../glqp-bench/benchmarks/linear-systems/backend-20261016/README.md
[guard]: ../../../glqp-bench/benchmarks/progression/barrier-guard-20261016/README.md
[outer]: ../../../glqp-bench/benchmarks/progression/outer-policy-20261016/README.md
[pc]: ../../../glqp-bench/benchmarks/progression/predictor-corrector-20261016/README.md
[proximal]: ../../../glqp-bench/benchmarks/progression/proximal-consistency-20261017/README.md
[newton-screen]: ../../../glqp-bench/benchmarks/progression/proximal-newton-20261017/README.md
[newton-full]: ../../../glqp-bench/benchmarks/progression/proximal-newton-full-20261017/README.md
[comparison]: ../../../glqp-bench/benchmarks/progression/solver-comparison-20261016/README.md
[tolerances]: ../../../glqp-bench/docs/comparator-tolerances-20261016.md
[attribution]: ../../../glqp-bench/benchmarks/progression/speed-attribution-20261016/README.md
[slope]: ../../../glqp-bench/benchmarks/progression/predictor-corrector-20261016/slope-capture/README.md
[ideas]: ../research/algorithm-improvement-proposals-20261016.md
[glm-stress]: ../../../glqp-bench/benchmarks/progression/glm-stress-20261017/README.md
