"""Primal-dual Newton driver and explicit experimental policy selection.

The accepted algorithm is documented in docs/accepted-algorithm.md. Numerical
linear solves, barrier policies, centered residuals, and observations live in
their own modules; this driver controls when each policy sees an iterate.
"""
from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd
from scipy.sparse import csc_array,block_diag
from scipy.sparse import block_array,eye_array
from sparse_dot_mkl import dot_product_mkl
from .util import (
    Logger,SparseTransposeCache,get_step_size,
    maxnorm,
    print_problem_summary,
    build_solution_summary,factor_and_solve,parse_problem
)

from .linear import LinearSolveError, ShiftedSystemWorkspace
from .linesearch import CachedGLMRay, search_scalar_ray
from .qp_linesearch import QPScalarProposer
from .objective_search import CachedObjectiveRay, search_restoration_ray
from .glm_diagnostics import glm_predictor_diagnostics, glm_hessian_diagnostics
from .kkt import KKTWorkspace
from .barrier import (
    validate_barrier_settings, select_barrier_target, describe_barrier_update,
    initial_barrier_target, describe_initial_target, resolve_barrier_floor,
)
from .outer_policy import (
    DIRECTION_CLIP_THRESHOLD, clip_tiny_directions,
    validate_repair_and_fallback_settings, guarded_slack_repair,
)
from .predictor_corrector import (
    validate_step_strategy, compute_predictor_corrector, describe_held_target,
)
from .proximal import (
    validate_proximal_settings, ProximalState, search_proximal_step,
    guard_proximal_repair,
)
from .diagnostics import (
    observe_iteration, point_diagnostics, direction_diagnostics,
    start_iteration_diagnostics, start_line_search_diagnostics,
    slack_repair_diagnostics, history_diagnostics,
    policy_diagnostics, start_repair_diagnostics,
    proximal_history_fields, record_scalar_search,
)


def _validate_policy_settings(settings):
    """Validate both new settings and mutable settings reused for another solve."""
    validate_barrier_settings(settings)
    validate_repair_and_fallback_settings(settings)
    validate_step_strategy(settings)
    validate_proximal_settings(settings)
    if (not isinstance(settings.glm_linesearch, str)
            or settings.glm_linesearch not in ('backtracking', 'scalar')):
        raise ValueError("glm_linesearch must be 'backtracking' or 'scalar'")
    if (not isinstance(settings.qp_linesearch, str)
            or settings.qp_linesearch not in ('backtracking', 'scalar')):
        raise ValueError("qp_linesearch must be 'backtracking' or 'scalar'")
    if (not isinstance(settings.qp_residual_screen, str)
            or settings.qp_residual_screen not in ('off', 'complementarity', 'bounded')):
        raise ValueError("qp_residual_screen must be 'off', 'complementarity', or 'bounded'")


def _no_glm_hessian(z):
    """An absent GLM contributes no matrix and requires no sparse arithmetic."""
    return None


@dataclass
class SolverSettings():
    """Public defaults retain accepted Newton; experimental policies are opt-in.

    Benchmark profiles override budgets and some globalization settings. See
    docs/accepted-algorithm.md before treating a profile as the public defaults.
    """
    max_precenter:int = 100
    max_iter:int = 200
    # Absolute max-norm KKT tolerance and relative aggregate complementarity.
    tol:float = 1e-8
    boundary_frac:float = 0.99
    gamma:float = 0.5
    # Cap the absolute floor against aggregate complementarity stopping.
    # 'fixed' preserves min_mu exactly, including for deliberate experiments.
    min_mu:float = 1e-12
    min_mu_policy:str = field(default='constraint_count', kw_only=True)
    tau_reg:float =2e-8
    max_linesearch_steps:int = 50
    # Scalar search applies to nonlinear descent with the legacy merit and no
    # Newton fallback. Other globalization paths retain their own contracts.
    glm_linesearch:str = 'scalar'
    # Opt-in QP proposals retain all existing merit/residual/fallback tests.
    qp_linesearch:str = 'backtracking'
    # Reject impossible KKT-decrease trials before sparse constraint products.
    qp_residual_screen:str = 'off'
    # Maximum inner steps in one right-preconditioned GMRES correction cycle.
    max_iterative_refinement:int = 10
    max_time:float = 600.
    max_stagnation:int = 20
    armijo_additive_eps:float = 1e-8
    armijo_param:float = 0.005
    let_newton_cook:float = 0.
    prox_reg:float = 2e-8
    proximal_policy:str = 'legacy'
    prox_dual_reg:float|None = None
    proximal_center_reduction:float = .95
    proximal_inner_tolerance:float = .1
    proximal_corrector_limit:float|None = .5
    proximal_reg_update:str = 'residual'
    linear_solver_backend:str = 'qdldl'
    linear_solver_options:dict|None = None
    # 'previous' gives the intended preceding-target reduction floor.
    # Keep legacy as the default pending the measured LISWET regressions.
    barrier_update_guard:str = 'legacy'
    barrier_reduction_limit:float = 0.1
    barrier_kkt_floor:float = 0.01
    initial_mu_policy:str = 'fixed'
    initial_mu_scale:float = 1.0
    slack_repair:str = 'legacy'
    repair_growth_limit:float = 1.1
    newton_fallback_policy:str = 'finite'
    step_strategy:str = 'newton'
    iteration_diagnostics:bool = False

    def __post_init__(self):
        _validate_policy_settings(self)

@dataclass
class SolverResults():
    """Solution with effective settings and configured/effective floor provenance."""
    settings:SolverSettings
    x:np.ndarray
    y:np.ndarray
    s:np.ndarray
    nu:np.ndarray
    primal:float
    time:float
    cons_viol:float
    feasible:bool
    dual_res:float
    history:pd.DataFrame
    termination_tag:str
    exception:Exception|None
    initialization_linear_solve:dict|None = None
    last_linear_solve:dict|None = None
    diagnostics:dict|None = None
    barrier_floor:dict|None = None

class GLQP():
    def __init__(
        self,
        f=None,A=None,
        Q=None,b=None,
        C=None,c=None,
        E=None,e=None,
        n = None
        ):
        """
        Problem 
        minimize f(Ax) + (1/2) x^T Qx -x^T b
        subject to Ex=e, Cx<=c

        Parameters
        ----------
        f : 
            GLM objective function, sum of scalars
        A : csc_array
            design matrix
        Q : csc_array, optional
            matrix defining quadratic form, defaults to 1e-8 * I
        b : np.ndarray, optional
            tilting vector, defaults to zeros
        C : csc_array, optional
            inequality constraint matrix, defaults to no constraints
        c : np.ndarray, optional
            inequality rhs, defaults to no constraints
        E : csc_array, optional
            equality constraint matrix, defaults to no constraints
        e : np.array, optional
            equality constraint rhs, defaults to no constraints
        """
        (dummy_A,dummy_ineq,
        f,A,
        Q,b,
        C,c,
        E,e,
        m,n,p,k) = parse_problem(f,A,Q,b,C,c,E,e,n)

        self.dummy_A = dummy_A
        self.dummy_ineq = dummy_ineq
        if self.dummy_A is True:
            self.QP_mode = True
        else:
            self.QP_mode = False
        self.f = f
        self.A = A
        self.Q = Q
        self._glm_hessian = (_no_glm_hessian if dummy_A
                             else self._evaluate_glm_hessian)
        self.b = b
        self.C = C
        self.c = c
        self.E = E
        self.e = e
        self.m = m
        self.n = n
        self.p = p
        self.k = k
        self.In = csc_array(eye_array(n))
        self.Ik = csc_array(eye_array(k))
        self.Ip = csc_array(eye_array(p))
        self.reg_shift = block_diag([self.In,-self.Ik,-self.Ip], format='csc')
        self._kkt_workspace = None
        self._shift_workspace = ShiftedSystemWorkspace()
        self._transpose_views = SparseTransposeCache()

    def initialize(self,x0=None,y0 = None,s0 = None,settings = None):

        def warm_start(value, size, name, positive=False):
            value = np.asarray(value, dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite vector with shape ({size},)")
            if positive and np.any(value <= 0):
                raise ValueError(f"{name} must be strictly positive")
            return value.copy()

        if settings is None:
            if self.QP_mode==True:
                settings = SolverSettings(
                    let_newton_cook=0.9
                )
            else:
                settings = SolverSettings()
        _validate_policy_settings(settings)
        settings, self.barrier_floor = resolve_barrier_floor(
            settings, 0 if self.dummy_ineq else self.k)
        self.settings = settings
        self._initial_linear_info = None
        # Problem matrices may have been replaced since the previous solve.
        self._kkt_workspace = None
        self._shift_workspace = ShiftedSystemWorkspace()
        self._transpose_views.clear()

        if x0 is None:
            x = np.zeros(self.n)
        else:
            x = warm_start(x0, self.n, "x0")
        
        if self.E.shape[0]>0:
            G = block_array(
                [
                    [self.In,self._transpose_views.get('E', self.E)],
                    [self.E ,None]
                ],format = 'csc'
            )
            rhs = np.hstack([x,self.e])
            if maxnorm(rhs)<=1e-15:
                #If the rhs is 0, don't do the linear solve
                #Just return the zero solution
                sol = rhs
            else:
                reg_shift = block_diag([0*self.In,-1*self.Ip])
                linear_result = factor_and_solve(
                    G,rhs,
                    reg_shift=reg_shift,
                    init_tau_reg = self.settings.tau_reg,
                    solver = None,
                    target_atol = 1e-12,
                    max_solve_attempts=12,
                    max_refinement_steps=self.settings.max_iterative_refinement,
                    residual_blocks={'primal': slice(0, self.n), 'equality': slice(self.n, None)},
                    backend=self.settings.linear_solver_backend,
                    backend_options=self.settings.linear_solver_options,
                )
                sol, num_refine, solver, lin_rel_error = linear_result
                self._initial_linear_info = linear_result.info
            x = sol[:self.n]
            nu = sol[self.n:]
        else:
            nu = np.ones((0,))
        
        if s0 is None:
            if self.dummy_ineq:
                s = np.ones(self.k)
            else:
                s = np.maximum(self.c - self.C@x,0.01)
        else:
            s = warm_start(s0, self.k, "s0", positive=True)


        if y0 is None:
            if self.dummy_ineq:
                y = self.settings.min_mu*np.ones(self.k)
            else:
                y = np.ones(self.k)
        else:
            y = warm_start(y0, self.k, "y0", positive=True)

        return x,y,s,nu,settings
    
    def KKT_res(self,x,g,y,s,nu):
        return self._kkt_residuals(g, y, s, self._constraint_products(x, y, nu))

    def _constraint_products(self, x, y, nu):
        """Fresh products at one point; callers may retain them for that point."""
        CT = self._transpose_views.get('C', self.C)
        ET = self._transpose_views.get('E', self.E)
        return CT@y, ET@nu, self.C@x, self.E@x

    def _kkt_residuals(self, g, y, s, products):
        """Combine products without changing residual arithmetic or grouping."""
        CTy, ETnu, Cx, Ex = products
        rx = g + CTy + ETnu - self.b
        rp = Cx + s - self.c
        rc = y * s
        req = Ex - self.e
        return rx,rp,rc,req

    def _relative_complementarity(self, x, y, primal):
        """Total complementarity in the original constraints, scaled by objective.

        Small individual products need not have a small sum. Stored slack can
        also differ from c-Cx by a tiny primal residual that large duals amplify.
        Use absolute products to prevent cancellation at slightly infeasible x.
        This supplements, rather than replaces, the absolute KKT stopping test.
        """
        if not np.isfinite(primal):
            return np.inf
        if self.dummy_ineq:
            return 0.
        with np.errstate(over='ignore', invalid='ignore'):
            total = np.sum(np.abs(y * (self.c - self.C@x)))
            residual = total / max(1., abs(primal))
        return float(residual) if np.isfinite(residual) else np.inf
        
    def solve_KKT(
        self,
        x,y,s,H,rx,rp,rc,req,mu,
        tau_reg=None,prox_reg = 1e-7,
        solver = None, *, dual_reg=None, complementarity_reg=None):
        # x and mu remain in the public signature for caller compatibility.

        if tau_reg is None:
            tau_reg = self.settings.tau_reg

        if self._kkt_workspace is None:
            self._kkt_workspace = KKTWorkspace(H, self.C, self.E)
        options = ({} if dual_reg is None and complementarity_reg is None else
                   dict(dual_reg=dual_reg, complementarity_reg=complementarity_reg))
        G, rhs, w = self._kkt_workspace.assemble(H, y, s, rx, rp, rc, req, prox_reg, **options)

        linear_result = factor_and_solve(
            G,rhs,
            reg_shift=self.reg_shift,
            init_tau_reg = tau_reg,
            solver = solver,
            target_atol = 0.05*self.settings.tol,
            max_solve_attempts=12,
            max_refinement_steps=self.settings.max_iterative_refinement,
            residual_blocks={'stationarity': slice(0, self.n),
                             'scaled_inequality': slice(self.n, self.n+self.k),
                             'equality': slice(self.n+self.k, None)},
            backend=self.settings.linear_solver_backend,
            backend_options=self.settings.linear_solver_options,
            workspace=self._shift_workspace,
        )
        sol, num_refine, solver, linsolve_rel_error = linear_result
        self._linear_info = linear_result.info
        # Finite inexact directions remain subject to the existing line search
        # and original KKT stopping tests. Their accuracy status is explicit.
                
        dx = sol[:self.n]
        dy = w*sol[self.n:self.n+self.k]
        ds =  - rp - self.C@dx + (prox_reg if dual_reg is None else dual_reg)*dy
        dnu = sol[self.n+self.k:]

        return dx,ds,dy,dnu,solver,num_refine,linsolve_rel_error

    def solve_predictor_corrector(self, x, y, s, H, rx, rp, req, kkt_res, *,
                                  prox_reg, solver=None, target_override=None,
                                  dual_reg=None, complementarity_reg=None,
                                  residual_safeguard=False, center_offsets=None):
        """Affine/combined RHS solves, with an optional centered-policy rescue."""
        return compute_predictor_corrector(
            self, x, y, s, H, rx, rp, req, kkt_res,
            prox_reg=prox_reg, solver=solver, target_override=target_override,
            dual_reg=dual_reg, complementarity_reg=complementarity_reg,
            residual_safeguard=residual_safeguard, center_offsets=center_offsets)

    
    def get_H(self,z):
        """Combine the fixed quadratic Hessian with optional GLM curvature.

        The returned matrix may be self.Q itself. Callers must not modify it.
        """
        glm_H = self._glm_hessian(z)
        return self.Q if glm_H is None else self.Q + glm_H

    def _evaluate_glm_hessian(self,z):
        D = self.f.d2f(z)[:,None]
        return dot_product_mkl(self._transpose_views.get('A', self.A),
                               csc_array(self.A.multiply(D)))
    
    def solve(
        self,
        x0=None,
        y0=None,
        s0=None,
        mu0 = None,
        verbose = True,
        settings:SolverSettings|None = None,
        *,
        iteration_callback=None,
        ):        
        """Solve, optionally observing initialized and accepted iterates.

        ``iteration_callback(snapshot)`` receives a frozen IterationSnapshot
        with read-only borrowed arrays, valid only during that synchronous
        call. Copy arrays to retain them. The callback must not mutate the
        model/settings or array bases; its return value is ignored. Exceptions
        propagate, and its work counts toward max_time. Iteration diagnostics
        are optional and likewise consume time, but never decide acceptance.

        Settings are copied before resolving the problem-dependent barrier
        floor. Returned ``settings`` contain the effective floor; the separate
        ``barrier_floor`` metadata retains both configured and effective values.
        """
        if iteration_callback is not None and not callable(iteration_callback):
            raise ValueError('iteration_callback must be callable or None')
        solved = False
        near_solved = False
        logger = Logger(verbose=verbose)
        feasible = False
        exception = None

        termination_tag = 'not_optimal'
        x,y,s,nu,settings = self.initialize(x0,y0,s0,settings)
        use_predictor_corrector = (settings.step_strategy == 'predictor_corrector'
                                   and not self.dummy_ineq and self.k > 0)
        diagnostics_enabled = settings.iteration_diagnostics
        qp_residual_norms = None
        diagnostics = (dict(schema_version=1, initial=None, iterations=[],
                            policies=policy_diagnostics(settings), initial_target=None,
                            barrier_floor=self.barrier_floor.copy())
                       if diagnostics_enabled else None)
        if verbose is True:
            k = self.k
            if self.dummy_ineq is True:
                k = 0
            m = self.m
            if self.dummy_A is True:
                m = 0
            print_problem_summary(
                n=self.n,
                m=m,
                p=self.p,
                k=k
            )

        start = time.time()
        # PC chooses its first target only after the affine solve. Until then
        # the initial observation carries a clearly labeled placeholder.
        mu = (settings.min_mu if use_predictor_corrector and mu0 is None else
              initial_barrier_target(s, y, self.dummy_ineq, mu0, settings))
        

        z = np.zeros(self.m) if self.dummy_A else self.A@x
        AT = self._transpose_views.get('A', self.A)
        Qx = self.Q@x
        primal = self.f(z) + 0.5*np.dot(x,Qx) - np.dot(x,self.b)
        
        H = self.get_H(z)
        # A loss may explicitly declare predictor-independent curvature.
        # Keep reuse local to this solve so changed weights/design are read
        # again on the next solve, and public get_H calls remain fresh.
        constant_hessian = self.dummy_A or getattr(self.f, 'hessian_is_constant', False)
        gradf = Qx if self.dummy_A else AT@self.f.d1f(z) + Qx
        rx,rp,rc,req = self.KKT_res(x,gradf,y,s,nu)
        proximal_state = (ProximalState(x,y,nu,rx,rp,req,
                            rho_cap=settings.prox_reg,
                            delta_cap=(settings.prox_reg if settings.prox_dual_reg is None
                                       else settings.prox_dual_reg))
                          if settings.proximal_policy == 'centered' else None)
        kkt_res = maxnorm(np.hstack([rx,rp,rc,req]))
        cons_viol = np.maximum(maxnorm(rp),maxnorm(req))
        initial_point = None
        if diagnostics_enabled:
            initial_point = point_diagnostics(
                self, x, y, s, primal, mu, kkt_res, rx, rp, rc, req)
            if not self.dummy_A:
                initial_point['glm_predictor'] = glm_predictor_diagnostics(self, x, z)
            diagnostics['initial'] = initial_point
            if use_predictor_corrector:
                diagnostics['initial_target'] = dict(
                    policy='predictor_corrector',
                    source='explicit_initial' if mu0 is not None else 'awaiting_predictor',
                    scale=None, mu_bar=float(np.dot(s, y)/self.k),
                    scaled_complementarity=None, min_mu=float(settings.min_mu),
                    target=float(mu), used_for_first_direction=mu0 is not None,
                    selection_timing='predictor_before_direction')
            else:
                diagnostics['initial_target'] = describe_initial_target(
                    s, y, self.dummy_ineq, mu0, mu, settings)
        if iteration_callback is not None:
            observe_iteration(iteration_callback, self, 0, 'initial',
                              x, y, s, nu, primal, mu, kkt_res, initial_point)
        #Perturb to interior complementarity
        rc = rc - mu
        

        
        solver = None
        self._linear_info = None
        for iteration_number in range(settings.max_iter):

            #Put this check at start in case we barely time out later
            if time.time() - start>settings.max_time:
                termination_tag = "max_time"
                break

            if maxnorm(rp)<1e-8 and maxnorm(req)<1e-8:
                feasible = True
            
            # Evaluate the aggregate test only near convergence. Keep the
            # Newton system, line search, and existing absolute KKT test intact.
            relative_comp_res = (self._relative_complementarity(x, y, primal)
                                 if kkt_res <= 100*settings.tol else np.inf)
            near_solved = (kkt_res <= 100*settings.tol
                           and relative_comp_res <= 100*settings.tol)
            
            #Check for convergence
            if kkt_res<=settings.tol and relative_comp_res<=settings.tol:
                solved = True
                break
            
            #Check for stagnation
            if (
                iteration_number>settings.max_stagnation and 
                kkt_res>=0.99*logger.rows[-settings.max_stagnation]['KKT_res']
            ):
                #Little progress in settings.max_stagnation
                termination_tag = "stagnated"
                break
            prox_reg = np.minimum(settings.prox_reg,kkt_res*1e-3)
            linear_options = {}
            dual_reg = prox_reg
            if settings.proximal_policy != 'legacy':
                dual_cap = settings.prox_reg if settings.prox_dual_reg is None else settings.prox_dual_reg
                dual_reg = np.minimum(dual_cap, kkt_res*1e-3)
                if proximal_state is not None and settings.proximal_reg_update == 'progress':
                    prox_reg, dual_reg = proximal_state.select_regularization(prox_reg, dual_reg)
                linear_options = dict(dual_reg=dual_reg, complementarity_reg=0.)
            proximal_model = None if proximal_state is None else proximal_state.freeze(prox_reg, dual_reg)
            linear_rx, linear_rp, linear_req = ((rx,rp,req) if proximal_model is None else
                proximal_model.residuals(x,y,nu,rx,rp,req))
            if diagnostics_enabled:
                iteration_diag = start_iteration_diagnostics(iteration_number+1, prox_reg)
                if proximal_model is not None:
                    iteration_diag['proximal'] = dict(
                        generation=proximal_model.generation, rho=float(prox_reg), delta=float(dual_reg),
                        center_offset_before_inf={key:maxnorm(value) for key,value
                            in proximal_model.offsets(x,y,nu).items()})

            self._linear_info = None
            pc_step = None
            try:
                #Solve KKT
                if use_predictor_corrector:
                    pc_step = self.solve_predictor_corrector(
                        x, y, s, H, linear_rx, linear_rp, linear_req, kkt_res, prox_reg=prox_reg,
                        solver=solver, target_override=mu0 if iteration_number == 0 else None,
                        **linear_options,
                        **({} if proximal_model is None else dict(residual_safeguard=True,
                            center_offsets=proximal_model.offsets(x,y,nu))))
                    dx, ds, dy, dnu = pc_step.dx, pc_step.ds, pc_step.dy, pc_step.dnu
                    solver = pc_step.factorization
                    num_refine = pc_step.refinements
                    self._linear_info = pc_step.combined_info
                    linsolve_rel_error = pc_step.combined_info.relative_error
                    # Both merit and residual progress tests use this same target.
                    mu = pc_step.target
                    rc = y*s - mu
                    if diagnostics_enabled:
                        iteration_diag['predictor_corrector'] = pc_step.decision
                else:
                    dx,ds,dy,dnu,solver,num_refine,linsolve_rel_error = self.solve_KKT(
                        x,y,s,
                        H,
                        linear_rx,linear_rp,rc,linear_req,
                        mu,
                        tau_reg = self.settings.tau_reg,
                        prox_reg = prox_reg,
                        solver = solver,
                        **linear_options,
                        )
            except LinearSolveError as ex:
                termination_tag = "failed_linear_solve"
                exception = ex
                if diagnostics_enabled:
                    iteration_diag['status'] = 'failed_linear_solve'
                    iteration_diag['exception'] = str(ex)
                    diagnostics['iterations'].append(iteration_diag)
                break

            # PC has now selected its target; Newton retains its incoming target.
            # Observe progress against that target before any direction edits.
            perturbed_kkt_before = (maxnorm(np.hstack([rx, rp, rc, req]))
                                    if settings.newton_fallback_policy == 'progress' else None)
            if diagnostics_enabled:
                old_ds, old_dy = ds.copy(), dy.copy()
                iteration_diag['linear_solve'] = (None if self._linear_info is None
                                                  else self._linear_info.history_fields())

            # Tiny boundary components must not block the other components.
            clip_tiny_directions(s, y, ds, dy)
            if diagnostics_enabled:
                iteration_diag['direction'] = direction_diagnostics(
                    self, H, dx, ds, dy, dnu, old_ds, old_dy,
                    y, s, rx, rp, rc, req, prox_reg, DIRECTION_CLIP_THRESHOLD,
                    complementarity_correction=None if pc_step is None else pc_step.correction,
                    **linear_options,
                    **({} if proximal_model is None else dict(
                        center_offsets=proximal_model.offsets(x,y,nu))))
                if not self.dummy_A:
                    iteration_diag['glm_hessian'] = glm_hessian_diagnostics(self, H, z, dx)
            
            boundary_frac = settings.boundary_frac
            tmax = get_step_size(s,ds,y,dy,frac = boundary_frac)
            #Perform a linesearch so we converge on the nonlinear part

            if proximal_model is not None:
                proximal_trial, search_diag = search_proximal_step(
                    self, proximal_model, x=x, y=y, s=s, nu=nu, z=z, Qx=Qx,
                    primal=primal, H=H, residuals=(rx,rp,y*s,req),
                    direction=(dx,ds,dy,dnu), target=mu, tmax=tmax,
                    slope_hint=None if pc_step is None else pc_step.decision['residual_merit_slope'])
                successful = proximal_trial is not None
                if successful:
                    t = proximal_trial.step
                    trial_x, trial_s, trial_y = proximal_trial.x, proximal_trial.s, proximal_trial.y
                    trial_z = proximal_trial.z
                    trial_products = proximal_trial.products
                    trial_glm_gradient = proximal_trial.glm_gradient
                if diagnostics_enabled:
                    iteration_diag['line_search'] = search_diag
            else:
                # Accepted-iterate values were evaluated together below (or at
                # initialization). Only the barrier changes when mu is updated.
                barrier = -mu*np.sum(np.log(s))
                merit0 = primal + barrier

                gx = gradf - self.b
                gs = -mu/s

                dx_gx = np.dot(dx,gx)
                ds_gs = np.dot(ds,gs)
                merit_slope = dx_gx + ds_gs
                if diagnostics_enabled:
                    search_diag = start_line_search_diagnostics(
                        tmax, merit_slope, fallback_policy=settings.newton_fallback_policy,
                        perturbed_kkt_before=perturbed_kkt_before)
                    iteration_diag['line_search'] = search_diag

                t = tmax
                dz = z if self.dummy_A else self.A@dx
                Qdx = self.Q@dx
                xQx = np.dot(x,Qx)
                xQdx = np.dot(x,Qdx)
                dxQdx = np.dot(dx,Qdx)
                # Choose the acceptance policy once. Both objective searches
                # use the same trial controller and safeguarded step proposer.
                scalar_search = (settings.glm_linesearch == 'scalar'
                                 and not self.QP_mode
                                 and np.isfinite(merit_slope) and merit_slope < 0
                                 and (settings.let_newton_cook <= 0
                                      or settings.newton_fallback_policy == 'off'))
                if scalar_search:
                    ray = CachedGLMRay(
                        self.f, z, dz, s, ds, mu, xQx, xQdx, dxQdx,
                        np.dot(x, self.b), np.dot(dx, self.b))

                    def trial_is_valid(step):
                        # No matrix products: the ray checks its loss and slack.
                        # Include free equality multipliers in the finite check.
                        with np.errstate(over='ignore', invalid='ignore'):
                            next_x = x + step*dx
                            next_y = y + step*dy
                            next_nu = nu + step*dnu
                        return bool(np.all(np.isfinite(next_x))
                                    and np.all(np.isfinite(next_y))
                                    and np.all(next_y > 0)
                                    and np.all(np.isfinite(next_nu)))

                    scalar_result = search_scalar_ray(
                        ray, tmax=tmax, merit0=merit0, slope0=merit_slope,
                        armijo_param=settings.armijo_param,
                        additive_eps=settings.armijo_additive_eps,
                        max_trials=settings.max_linesearch_steps,
                        trial_is_valid=trial_is_valid,
                        record_history=diagnostics_enabled)
                    successful = scalar_result.trial is not None
                    trial_products = trial_glm_gradient = None
                    if successful:
                        trial = scalar_result.trial
                        t = trial.step
                        trial_x, trial_y = x + t*dx, y + t*dy
                        trial_s, trial_z = trial.slack, trial.predictor
                    if diagnostics_enabled:
                        record_scalar_search(search_diag, scalar_result)
                else:
                    ray = CachedObjectiveRay(self,
                        point=(x, s, y, nu, z), direction=(dx, ds, dy, dnu),
                        dz=dz, quadratic=(Qx, Qdx, xQx, xQdx, dxQdx), mu=mu,
                        transpose=AT)
                    trial, qp_residual_norms = search_restoration_ray(
                        ray, settings, tmax=tmax, merit0=merit0, slope0=merit_slope,
                        residuals=(rx, rp, req), kkt_res=kkt_res,
                        perturbed_kkt_before=perturbed_kkt_before,
                        residual_norms=qp_residual_norms,
                        diagnostics=search_diag if diagnostics_enabled else None,
                        proposer_factory=QPScalarProposer)
                    successful = trial is not None
                    if successful:
                        t = trial.step
                        trial_x, trial_y = trial.x, trial.y
                        trial_s, trial_z = trial.slack, trial.predictor
                        trial_products = trial.products
                        trial_glm_gradient = trial.glm_gradient

            if not successful:
                termination_tag = "failed_line_search"
                if diagnostics_enabled:
                    iteration_diag['status'] = 'failed_line_search'
                    diagnostics['iterations'].append(iteration_diag)
                break

            if diagnostics_enabled:
                search_diag['accepted_step'] = float(t)
                search_diag['accepted_fraction'] = float(t/tmax)

            #Take step
            x, s, y, z = trial_x, trial_s, trial_y, trial_z
            nu = nu + t*dnu

            Qx = self.Q@x
            primal = self.f(z) + 0.5*np.dot(x,Qx) - np.dot(x,self.b)
            if self.dummy_A:
                gradf = Qx
            else:
                glm_gradient = AT@self.f.d1f(z) if trial_glm_gradient is None else trial_glm_gradient
                gradf = glm_gradient + Qx
            # A residual-accepted trial already evaluated the exact constraint
            # products and likelihood gradient at these same vectors. Reuse
            # only those products, not its approximate Qx+t*Qdx or residual.
            if trial_products is None:
                rx,rp,rc,req = self.KKT_res(x,gradf,y,s,nu)
            else:
                rx,rp,rc,req = self._kkt_residuals(gradf, y, s, trial_products)

            if not constant_hessian:
                H = self.get_H(z)
            kkt_res = np.max([maxnorm(rx),maxnorm(rp),maxnorm(rc),maxnorm(req)])

            # Choose the next target before repair, using the accepted point's
            # current KKT norm. Repair must not silently change this proposal.
            previous_mu = mu
            pre_repair_kkt = kkt_res
            if diagnostics_enabled:
                pre_repair_point = point_diagnostics(
                    self, x, y, s, primal, mu, kkt_res, rx, rp, rc, req)
                pre_repair_s, pre_repair_rp, pre_repair_rc = s.copy(), rp.copy(), rc.copy()
                repair_triggered = False
                repair_policy_diag = start_repair_diagnostics(settings)
            if self.dummy_ineq:
                mu = settings.min_mu
                if diagnostics_enabled:
                    raw_mu = mu
                    mu_est, xi = pre_repair_point['mu_bar'], pre_repair_point['xi']
            else:
                mu_est = np.dot(s,y)/self.k
                xi = np.min(s*y)/mu_est
                if use_predictor_corrector:
                    # Keep the predictor target fixed through this accepted step.
                    # A fresh predictor selects the next target at the next point.
                    raw_mu = mu
                else:
                    raw_mu = (np.minimum((1.02-xi),1)**3)*mu_est
                    mu = select_barrier_target(previous_mu, raw_mu, pre_repair_kkt, settings)

                fix_threshold = 0.1
                if tmax<1e-4 and np.min(s*y)/mu_est<fix_threshold:
                    if diagnostics_enabled:
                        repair_triggered = True
                    if settings.slack_repair == 'off':
                        if diagnostics_enabled:
                            repair_policy_diag.update(decision='disabled', rejection_reason='disabled')
                    else:
                        args_to_fix = (s*y)/mu_est<=fix_threshold
                        if settings.slack_repair == 'legacy' and proximal_model is None:
                            if verbose is True:
                                print("Adjusted")
                            s[args_to_fix] = fix_threshold*mu_est/(y[args_to_fix])
                            # Preserve the accepted legacy update and residual
                            # arithmetic. Its barrier was chosen before repair.
                            rp = self.C@x + s - self.c
                            rc = y*s
                            kkt_res = max(maxnorm(rx), maxnorm(rp), maxnorm(rc), maxnorm(req))
                            if diagnostics_enabled:
                                repair_policy_diag.update(
                                    proposed=True, proposed_count=int(np.count_nonzero(args_to_fix)),
                                    applied=True, decision='legacy_applied',
                                    candidate_primal_inf=float(maxnorm(rp)), candidate_kkt_inf=float(kkt_res),
                                    candidate_finite_positive=bool(np.all(np.isfinite(s)) and np.all(s > 0)
                                        and np.all(np.isfinite(rp)) and np.all(np.isfinite(rc))
                                        and np.isfinite(kkt_res)),
                                    primal_limit=float(max(settings.repair_growth_limit*
                                        pre_repair_point['primal_residual_inf'], settings.tol)),
                                    kkt_limit=float(max(settings.repair_growth_limit*pre_repair_kkt, settings.tol)))
                        else:
                            proposal = guarded_slack_repair(
                                self.C, self.c, x, y, s, rx, rp, req, pre_repair_kkt,
                                mu_est, args_to_fix, fix_threshold, settings.repair_growth_limit, settings.tol)
                            if proximal_model is not None and proposal.accepted:
                                allowed, metrics = guard_proximal_repair(
                                    proximal_model,x,y,nu,rx,rp,rc,req,proposal.rp,proposal.rc,
                                    search_diag['target'],search_diag)
                                proposal.accepted = allowed
                                proposal.metrics.update(metrics, applied=allowed, rejected=not allowed,
                                    rejection_reason=None if allowed else 'proximal_merit_growth',
                                    rejection_reasons=() if allowed else ('proximal_merit_growth',))
                            if proposal.accepted:
                                s, rp, rc, kkt_res = (proposal.s, proposal.rp, proposal.rc,
                                                     proposal.kkt_residual)
                            if diagnostics_enabled:
                                repair_policy_diag.update(proposal.metrics)
                                repair_policy_diag['decision'] = ('guarded_applied' if proposal.accepted
                                                                  else 'guarded_rejected')
                                if proximal_model is not None:
                                    repair_policy_diag['policy'] = 'proximal_merit_guarded'

            proximal_update = None
            if proximal_model is not None:
                proximal_update = proximal_state.update(
                    proximal_model,x,y,nu,rx,rp,rc,req,search_diag['target'],settings)
                if diagnostics_enabled:
                    iteration_diag['proximal'].update(proximal_update)

            accepted_point = None
            if diagnostics_enabled:
                accepted_point = point_diagnostics(
                    self, x, y, s, primal, mu, kkt_res, rx, rp, rc, req)
                if not self.dummy_A:
                    accepted_point['glm_predictor'] = glm_predictor_diagnostics(self, x, z)
                iteration_diag['barrier'] = (describe_held_target(
                    mu, mu_est, xi, pre_repair_kkt, settings) if use_predictor_corrector else
                    describe_barrier_update(
                        previous_mu, raw_mu, mu, mu_est, xi, pre_repair_kkt, settings,
                        dummy=self.dummy_ineq))
                iteration_diag['repair'] = slack_repair_diagnostics(
                    repair_triggered, pre_repair_s, s, pre_repair_rp, rp,
                    pre_repair_rc, rc, pre_repair_point, accepted_point)
                iteration_diag['repair'].update(repair_policy_diag)
                iteration_diag['iterate'] = accepted_point
                iteration_diag['status'] = 'accepted'
                diagnostics['iterations'].append(iteration_diag)
            if iteration_callback is not None:
                observe_iteration(iteration_callback, self, iteration_number+1, 'accepted',
                                  x, y, s, nu, primal, mu, kkt_res, accepted_point)

            comp_res = maxnorm(rc)
            cons_viol = np.maximum(maxnorm(rp),maxnorm(req))

            #Perturb complementarity to new interior parameter
            rc = rc - mu

            elapsed = time.time() - start
            logger.log(
                iter=iteration_number+1,
                primal = primal,
                dual_res = maxnorm(rx),
                cons_viol = cons_viol,
                comp_res = comp_res,
                mu=mu,
                Δx = maxnorm(t*dx),
                step=t,
                KKT_res=kkt_res,
                time=elapsed,
                refine = num_refine,
                **({'lin_rel_res': linsolve_rel_error} if self._linear_info is None
                   else self._linear_info.history_fields()),
                **(history_diagnostics(iteration_diag) if diagnostics_enabled else {}),
                **({} if pc_step is None else pc_step.history_fields()),
                **({} if proximal_update is None else
                   proximal_history_fields(proximal_update, search_diag)),
            )

        termination_tag,message = build_solution_summary(
            solved,
            near_solved,
            termination_tag,
            kkt_res,
            iteration_number,
            time.time() - start,
            exception = exception
        )
        if verbose is True:
            print(message)

        # Report the returned iterate, including when the final step reaches
        # the iteration/time limit or a later line search fails.
        primal = self.f(z if self.dummy_A else self.A@x) + 0.5*np.dot(x,self.Q@x) - np.dot(x,self.b)
        
        results = SolverResults(
            settings,
            x,y,s,nu,
            termination_tag = termination_tag,
            exception = exception,
            primal = primal,
            dual_res = maxnorm(rx),
            cons_viol = np.maximum(maxnorm(rp),maxnorm(req)),
            feasible = feasible,
            time = time.time() - start,
            history = logger.to_dataframe(),
            initialization_linear_solve=(None if self._initial_linear_info is None
                                        else self._initial_linear_info.history_fields()),
            last_linear_solve=(None if self._linear_info is None
                               else self._linear_info.history_fields()),
            diagnostics=diagnostics,
            barrier_floor=self.barrier_floor.copy(),
            )
        return x,results

