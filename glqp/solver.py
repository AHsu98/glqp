import time
import numpy as np
import pandas as pd
from scipy.sparse import csc_array,block_diag
import qdldl
from scipy.sparse import block_array,eye_array
from sparse_dot_mkl import dot_product_mkl
from .util import (
    Logger,get_step_size,
    maxnorm,
    print_problem_summary,
    build_solution_summary,factor_and_solve,parse_problem
)

from dataclasses import dataclass

@dataclass
class SolverSettings():
    max_precenter:int = 100
    max_iter:int = 200
    tol:float = 1e-8
    safe_boundary_frac:float = 0.99
    greedy_boundary_frac:float = 0.9999
    gamma:float = 0.5
    min_mu:float = 1e-12
    tau_reg:float =5e-9
    max_linesearch_steps:int = 50
    max_iterative_refinement:int = 5
    max_time:float = 600.
    max_stagnation:int = 20
    armijo_additive_eps:float = 1e-8
    armijo_param:float = 0.005
    let_newton_cook:float = 0.5

@dataclass
class SolverResults():
    settings:SolverSettings
    x:np.ndarray
    y:np.ndarray
    s:np.ndarray
    history:pd.DataFrame
    termination_tag:str
    exception:Exception|None

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

    def initialize(self,x0=None,y0 = None,s0 = None):
        if x0 is None:
            x = np.zeros(self.n)
        else:
            x = np.copy(x0)
        
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
                sol,num_refine,solver,lin_rel_error = factor_and_solve(
                    G,rhs,
                    reg_shift=reg_shift,
                    init_tau_reg = self.settings.tau_reg,
                    solver = None,
                    target_atol = 1e-12,
                    max_solve_attempts=10,
                    max_refinement_steps=self.settings.max_iterative_refinement
                )
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
            s = np.copy(s0)


        if y0 is None:
            if self.dummy_ineq:
                y = self.settings.min_mu*np.ones(self.k)
            else:
                y = np.ones(self.k)
        else:
            y = np.copy(y0)
            assert np.min(y)>1e-10

        return x,y,s,nu
    
    def KKT_res(self,x,g,y,s,nu):
        rx = g + self.C.T@y + self.E.T@nu - self.b
        rp = self.C@x + s - self.c
        rc = y * s
        req = self.E@x - self.e
        return rx,rp,rc,req
    
    def solve_KKT(
        self,
        x,y,s,H,rx,rp,rc,req,mu,tau_reg=None,prox_reg = 0.,
        solver = None):
        #mu,x unused for now

        if tau_reg is None:
            tau_reg = self.settings.tau_reg
        #Nesterov-Todd scaling
        # Quasi definite for inequality constraints, 
        # "normal equations" Hessian for GLM part
        w = np.sqrt(y/s)
        wC = self.C.multiply(w[:,None])
        rhs = np.hstack([-rx,-w*rp + (w/y) * rc,-req])
        #Including tau-shift here
        #later may want separate matrix,
        #larger tau shift + iterative refine

        G = block_array(
            [
                [H + prox_reg * self.In,     wC.T,       self.E.T],
                [wC,    -1*self.Ik, None],
                [self.E,None,       None]
            ],format = 'csc'
        )

        sol,num_refine,solver,linsolve_rel_error = factor_and_solve(
            G,rhs,
            reg_shift=self.reg_shift,
            init_tau_reg = tau_reg,
            solver = solver,
            target_atol = 0.05*self.settings.tol,
            max_solve_attempts=10,
            max_refinement_steps=self.settings.max_iterative_refinement
        )
                
        dx = sol[:self.n]
        dy = w*sol[self.n:self.n+self.k]
        ds = -rp - self.C@dx
        dnu = sol[self.n+self.k:]

        return dx,ds,dy,dnu,solver,num_refine,linsolve_rel_error
    
    def get_H(self,z):
        D = self.f.d2f(z)[:,None]
        AtDa = dot_product_mkl(self.A.T,csc_array(self.A.multiply(D)))
        return AtDa + self.Q
    
    def solve(
        self,
        x0=None,
        y0=None,
        s0=None,
        mu0 = None,
        verbose = True,
        settings:SolverSettings|None = None
        ):
        if settings is None:
            settings = SolverSettings()
            #TODO: Different default settings based on QP_mode
        self.settings = settings
        
        solved = False
        near_solved = False
        logger = Logger(verbose=verbose)
        feasible = False
        exception = None

        termination_tag = 'not_optimal'
        x,y,s,nu = self.initialize(x0,y0,s0)
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
        H = self.get_H(z)
        gradf = self.A.T@self.f.d1f(z) + self.Q@x
        rx,rp,rc,req = self.KKT_res(x,gradf,y,s,nu)
        kkt_res = np.max(
                np.abs(np.hstack([rx,rp,rc,req]))
            )
        #Perturb to interior complementarity
        rc = rc - mu

        
        solver = None
        for iteration_number in range(settings.max_iter):

            #Put this check at start in case we barely time out later
            if time.time() - start>settings.max_time:
                termination_tag = "max_time"
                break

            if maxnorm(rp)<1e-8 and maxnorm(req)<1e-8:
                feasible = True
            
            if kkt_res<=100*settings.tol:
                near_solved = True
            
            #Check for convergence
            if kkt_res<=settings.tol:
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
            
            #Give some proximal regularization in primal
            #because we shortcut step acceptance while infeasible
            if feasible is False:
                prox_reg = 0.1 * np.mean(H.diagonal())
            else:
                prox_reg = 0.
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
            except Exception as ex:
                termination_tag = "failed_linear_solve"
                exception = ex
                break
            
            # Allow a greedier step when we have bad complementarity
            if (feasible is True) and maxnorm(rc)>10*maxnorm(rx):
                boundary_frac = settings.greedy_boundary_frac
            else:
                boundary_frac = settings.safe_boundary_frac
            tmax = get_step_size(s,ds,y,dy,frac = boundary_frac)
            #Perform a linesearch on the nonlinear part

            #Evaluate merit at current point
            primal = self.f(z) + (1/2) * x.T@self.Q@x - np.dot(x,self.b)
            barrier = -mu*np.sum(np.log(s))
            merit0 = primal + barrier

            gx = self.A.T@self.f.d1f(z) + self.Q@x - self.b
            gs = -mu/s

            t = tmax
            dz = self.A@dx
            #Check implicit feasibility of f(x+t*dz)
            step_finite = self.f(z + t*dz)<np.inf
            
            #Accept every step if feasible is false and don't nan on the f
            #only enter linesearch if already feasible
            if (feasible is True) or (step_finite is False):
                #OK with a small additive error in line search
                def merit_line(t):
                    primal = self.f(z+t*dz) + (1/2) * (x+t*dx).T@self.Q@(x+t*dx) - np.dot(x+t*dx,self.b)
                    barrier = -mu*np.sum(np.log(s+t*ds))
                    return primal + barrier
                
                successful = False
                for linesearch_step in range(settings.max_linesearch_steps):
                    new_merit = merit_line(t)
                    armijo_satisfied = (
                        new_merit<(
                            merit0 + 
                            settings.armijo_param * t * (np.dot(dx,gx) + np.dot(ds,gs)) + 
                            settings.armijo_additive_eps)
                    )
                    if armijo_satisfied:
                        successful = True
                        break
                    #LET NEWTON COOK??
                    if t<settings.let_newton_cook*tmax:
                        successful = True
                        break
                    else:
                        t = 0.9*t
                if successful ==False:
                    termination_tag = "failed_line_search"
                    break
                            
            #Take step
            x  =  x + t*dx
            s  =  s + t*ds
            y  =  y + t*dy
            z  =  z + t*dz
            nu = nu + t*dnu

            gradf = self.A.T@self.f.d1f(z) + self.Q@x
            rx,rp,rc,req = self.KKT_res(x,gradf,y,s,nu)

            H = self.get_H(z)
            kkt_res = np.max([maxnorm(rx),maxnorm(rp),maxnorm(rc),maxnorm(req)])

            #If we're reasonably close to primal feasibility and 
            # complementarity aggressive mu update
            if  maxnorm(rx)+maxnorm(rc) + maxnorm(rp) + maxnorm(req)<=5*mu+np.minimum(1000 * mu,1000):
                mu_est = np.dot(s,y)/self.k
                xi = np.min(s*y)/mu_est
                #Don't decrease by more than a factor of 100
                mu_lower = np.maximum(mu*0.1,settings.min_mu)
                mu = np.maximum(
                    mu_lower,
                    settings.gamma * 
                    np.minimum(
                    (1-boundary_frac)*(1-xi)/xi + 0.1,2)**3 * mu_est
                    )
            else:
                # Otherwise, perform a modest centering update
                mu_est = np.dot(s,y)/self.k
                mu = 0.9*mu_est

            comp_res = maxnorm(rc)

            #Perturb complementarity to new interior parameter
            rc = rc - mu

            elapsed = time.time() - start
            logger.log(
                iter=iteration_number+1,
                primal = primal,
                dual_res = maxnorm(rx),
                cons_viol = np.maximum(maxnorm(rp),maxnorm(req)),
                comp_res = comp_res,
                mu=mu,
                Δx = maxnorm(t*dx),
                step=t,
                KKT_res=kkt_res,
                time=elapsed,
                refine = num_refine,
                lin_rel_res = linsolve_rel_error
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
        
        results = SolverResults(
            settings,
            x,y,s,
            logger.to_dataframe(),
            termination_tag = termination_tag,
            exception = exception
            )
        return x,results

