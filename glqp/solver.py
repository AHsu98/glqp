import time
import numpy as np
import pandas as pd
from scipy.sparse import csc_array,block_diag
from scipy.sparse import block_array,eye_array
from sparse_dot_mkl import dot_product_mkl
from .util import (
    Logger,get_step_size,
    maxnorm,
    print_problem_summary,
    build_solution_summary,factor_and_solve,parse_problem
)

from .linear import LinearSolveError
from .kkt import KKTWorkspace

from dataclasses import dataclass


def _no_glm_hessian(z):
    """An absent GLM contributes no matrix and requires no sparse arithmetic."""
    return None


@dataclass
class SolverSettings():
    max_precenter:int = 100
    max_iter:int = 200
    # Absolute max-norm KKT tolerance and relative aggregate complementarity.
    tol:float = 1e-8
    boundary_frac:float = 0.99
    gamma:float = 0.5
    min_mu:float = 1e-12
    tau_reg:float =2e-8
    max_linesearch_steps:int = 50
    # Maximum inner steps in one right-preconditioned GMRES correction cycle.
    max_iterative_refinement:int = 10
    max_time:float = 600.
    max_stagnation:int = 20
    armijo_additive_eps:float = 1e-8
    armijo_param:float = 0.005
    let_newton_cook:float = 0.
    prox_reg:float = 2e-8

@dataclass
class SolverResults():
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
        self.reg_shift = block_diag([self.In,-self.Ik,-self.Ip])
        self._kkt_workspace = None

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
        self.settings = settings
        self._initial_linear_info = None
        # Problem matrices may have been replaced since the previous solve.
        self._kkt_workspace = None

        if x0 is None:
            x = np.zeros(self.n)
        else:
            x = warm_start(x0, self.n, "x0")
        
        if self.E.shape[0]>0:
            G = block_array(
                [
                    [self.In,self.E.T],
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
        rx = g + self.C.T@y + self.E.T@nu - self.b
        rp = self.C@x + s - self.c
        rc = y * s
        req = self.E@x - self.e
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
        solver = None):
        #mu,x unused for now

        if tau_reg is None:
            tau_reg = self.settings.tau_reg

        if self._kkt_workspace is None:
            self._kkt_workspace = KKTWorkspace(H, self.C, self.E)
        G, rhs, w = self._kkt_workspace.assemble(H, y, s, rx, rp, rc, req, prox_reg)

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
        )
        sol, num_refine, solver, linsolve_rel_error = linear_result
        self._linear_info = linear_result.info
        # Finite inexact directions remain subject to the existing line search
        # and original KKT stopping tests. Their accuracy status is explicit.
                
        dx = sol[:self.n]
        dy = w*sol[self.n:self.n+self.k]
        ds =  - rp - self.C@dx + prox_reg*dy
        dnu = sol[self.n+self.k:]

        return dx,ds,dy,dnu,solver,num_refine,linsolve_rel_error

    
    def get_H(self,z):
        """Combine the fixed quadratic Hessian with optional GLM curvature.

        The returned matrix may be self.Q itself. Callers must not modify it.
        """
        glm_H = self._glm_hessian(z)
        return self.Q if glm_H is None else self.Q + glm_H

    def _evaluate_glm_hessian(self,z):
        D = self.f.d2f(z)[:,None]
        return dot_product_mkl(self.A.T,csc_array(self.A.multiply(D)))
    
    def solve(
        self,
        x0=None,
        y0=None,
        s0=None,
        mu0 = None,
        verbose = True,
        settings:SolverSettings|None = None
        ):        
        solved = False
        near_solved = False
        logger = Logger(verbose=verbose)
        feasible = False
        exception = None

        termination_tag = 'not_optimal'
        x,y,s,nu,settings = self.initialize(x0,y0,s0,settings)
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
        if mu0 is None:
            if self.dummy_ineq is True:
                mu = settings.min_mu
            else:
                mu = 100.
        else:
            mu = mu0
        

        z = self.A@x
        primal = self.f(z) + (1/2) * x.T@self.Q@x - np.dot(x,self.b)
        
        H = self.get_H(z)
        gradf = self.A.T@self.f.d1f(z) + self.Q@x
        rx,rp,rc,req = self.KKT_res(x,gradf,y,s,nu)
        kkt_res = maxnorm(np.hstack([rx,rp,rc,req]))
        cons_viol = np.maximum(maxnorm(rp),maxnorm(req))
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

            self._linear_info = None
            try:
                #Solve KKT
                dx,ds,dy,dnu,solver,num_refine,linsolve_rel_error = self.solve_KKT(
                    x,y,s,
                    H,
                    rx,rp,rc,req,
                    mu,
                    tau_reg = self.settings.tau_reg,
                    prox_reg = prox_reg,
                    solver = solver
                    )
            except LinearSolveError as ex:
                termination_tag = "failed_linear_solve"
                exception = ex
                break

            #Stop step directions from going too much closer to boundary
            #Freeze as if active set if step points towards boundary,
            #and s or y are already very tiny
            thresh = 1e-14
            #Prevent from stopping others from stepping
            s0_ind = s<=thresh
            y0_ind = y<=thresh
            ds[s0_ind] = np.maximum(-s[s0_ind],ds[s0_ind])
            dy[y0_ind] = np.maximum(-y[y0_ind],dy[y0_ind])
            
            boundary_frac = settings.boundary_frac
            tmax = get_step_size(s,ds,y,dy,frac = boundary_frac)
            #Perform a linesearch so we converge on the nonlinear part

            #Evaluate merit at current point
            Qx = self.Q@x
            primal = self.f(z) + (1/2) * x.T@Qx - np.dot(x,self.b)
            barrier = -mu*np.sum(np.log(s))
            merit0 = primal + barrier

            gx = self.A.T@self.f.d1f(z) + Qx - self.b
            gs = -mu/s

            dx_gx = np.dot(dx,gx)
            ds_gs = np.dot(ds,gs)
            merit_slope = dx_gx + ds_gs

            t = tmax
            dz = self.A@dx
            Qdx = self.Q@dx
            #Enter line search

            #OK with a small additive error in line search
            def merit_line(t):
                primal = (
                    self.f(z+t*dz)
                    + (1/2) * (np.dot(x,Qx) + 2*t*np.dot(x,Qdx) + t*t*np.dot(dx,Qdx))
                        - np.dot(x+t*dx,self.b)
                )
                barrier = -mu*np.sum(np.log(s+t*ds))
                return primal + barrier
            
            successful = False
            for linesearch_step in range(settings.max_linesearch_steps):
                if not np.isfinite(t) or t <= 0:
                    break
                trial_s, trial_y = s + t*ds, y + t*dy
                if (not np.all(np.isfinite(x + t*dx))
                        or not np.all(np.isfinite(trial_s))
                        or not np.all(np.isfinite(trial_y))
                        or np.any(trial_s <= 0) or np.any(trial_y <= 0)):
                    t = 0.9*t
                    continue
                #Check armijo
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    new_merit = merit_line(t)
                if not np.isfinite(new_merit):
                    t = 0.9*t
                    continue
                armijo_satisfied = (
                    new_merit<(
                        merit0 + 
                        settings.armijo_param * t * merit_slope +
                        settings.armijo_additive_eps)
                )
                if armijo_satisfied:
                    successful = True
                    break
                
                #Check KKT residual
                ls_gradf = self.A.T@self.f.d1f(z+t*dz) + Qx + t*Qdx
                ls_kkt_res = maxnorm(np.hstack(
                    self.KKT_res(x+t*dx,ls_gradf,y+t*dy,s+t*ds,nu+t*dnu)
                    ))
                # Preserve residual acceptance for QPs, including restoration
                # from an inconsistent positive slack warm start. For nonlinear
                # GLMs, alternating merit decreases and residual-only steps can
                # cycle: descent directions must pass the merit test there.
                if (self.QP_mode or merit_slope >= 0) and np.isfinite(ls_kkt_res) and (
                    ls_kkt_res<(1-settings.armijo_param*t)*kkt_res
                    or ls_kkt_res<settings.tol
                ):
                    successful = True
                    break
                # Preserve the explicitly configured Newton fallback, but
                # never use it to accept nonfinite or non-interior trials.
                if np.isfinite(ls_kkt_res) and t<settings.let_newton_cook*tmax:
                    successful = True
                    break
                t = 0.9*t

            if not successful:
                termination_tag = "failed_line_search"
                break

            #Take step
            x  =  x + t*dx
            s  =  s + t*ds
            y  =  y + t*dy
            z  =  z + t*dz
            nu = nu + t*dnu

            Qx = self.Q@x
            primal = self.f(z) + 0.5*np.dot(x,Qx) - np.dot(x,self.b)
            gradf = self.A.T@self.f.d1f(z) + Qx
            rx,rp,rc,req = self.KKT_res(x,gradf,y,s,nu)

            H = self.get_H(z)
            kkt_res = np.max([maxnorm(rx),maxnorm(rp),maxnorm(rc),maxnorm(req)])

            #Update mu here
            if self.dummy_ineq:
                mu = settings.min_mu
            else:
                mu_est = np.dot(s,y)/self.k
                xi = np.min(s*y)/mu_est
                mu = (np.minimum((1.02-xi),1)**3)*mu_est

                mu_lower_bound = np.max([mu*0.1,settings.min_mu,0.01*kkt_res])
                mu = np.maximum(mu,mu_lower_bound)

                fix_threshold = 0.1
                if tmax<1e-4 and np.min(s*y)/mu_est<fix_threshold:
                    print("Adjusted")
                    args_to_fix = (s*y)/mu_est<=fix_threshold
                    s[args_to_fix] = fix_threshold*mu_est/(y[args_to_fix])
                    # The repair changes the iterate after the residuals were
                    # evaluated. Refresh both slack-dependent equations before
                    # logging, convergence checks, or the next Newton solve.
                    # Keep the barrier target chosen from the accepted step.
                    rp = self.C@x + s - self.c
                    rc = y*s
                    kkt_res = max(maxnorm(rx), maxnorm(rp), maxnorm(rc), maxnorm(req))

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
                   else self._linear_info.history_fields())
            )

        #TODO: Maybe merge build_solution_summary and SolverResults together
        #into a finish_opt function.
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
        primal = self.f(self.A@x) + 0.5*np.dot(x,self.Q@x) - np.dot(x,self.b)
        
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
            )
        return x,results

