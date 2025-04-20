import time
import numpy as np
import pandas as pd
from scipy.sparse import csc_array
import qdldl
from scipy.sparse import block_array,eye_array,tril,triu
from sparse_dot_mkl import dot_product_mkl
from util import PrettyLogger,get_step_size,maxnorm,norm2,print_problem_summary
from warnings import warn

from dataclasses import dataclass

@dataclass
class SolverSettings():
    max_precenter:int = 100
    max_iter:int = 200
    tol:float = 1e-7
    safe_boundary_frac:float = 0.99
    greedy_boundary_frac:float = 0.9999
    gamma:float = 0.5
    min_mu:float = 1e-11
    tau_reg:float =5e-9
    max_linesearch_steps:int = 50

@dataclass
class SolverResults():
    settings:SolverSettings
    x:np.ndarray
    y:np.ndarray
    s:np.ndarray
    history:pd.DataFrame
    convergence_tag:str

class GLMProblem():
    def __init__(
        self,
        f,A,Q,
        C,c,
        b=None,
        E=None,e=None,
        settings = None
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
        Q : csc_array
            matrix defining quadratic form
        C : csc_array
            inequality constraint matrix
        c : np.ndarray
            inequality rhs
        E : csc_array
            equality constraint matrix
        e : 
            equality constraint rhs
        b : np.ndarray, optional
            tilting vector, defaults to zeros
        settings : SolverSettings, optional
            settings for solver, by default SolverSettings(0)
        """
        if settings is None:
            settings = SolverSettings()
        

        self.settings = settings
        
        m = A.shape[0]
        n = A.shape[1]
        assert A.shape[1] == C.shape[1]
        A = csc_array(A)

        #TODO:Separate solver dispatch for unconstrained
        assert C.shape[0] == len(c)
        C = csc_array(C)
        k = C.shape[0]
        self.k = k
        self.c = c

        if E is None:
            assert e is None
            E = csc_array((0,n))
            e = np.zeros((0,))
        else:
            assert e is not None
        assert E.ndim==2
        assert E.shape[1] == n
        assert E.shape[0] == e.shape[0]
        self.E = E
        self.e = e
        p = E.shape[0]
            

        if b is None:
            b = np.zeros(n)
        self.b = b
        
        self.f = f
        self.A = A
        self.Q = Q
        self.C = C
        self.m = m
        self.n = n
        self.p = p
        self.In = csc_array(eye_array(n))
        self.Ik = csc_array(eye_array(k))
        self.Ip = csc_array(eye_array(p))

    def initialize(self,x0=None,y0 = None,s0 = None):
        if x0 is None:
            x = np.zeros(self.n)
        else:
            x = np.copy(x0)
        
        if y0 is None:
            y = np.ones(self.k)
        else:
            y = np.copy(y0)
            assert np.min(y)>1e-10
        if self.E.shape[0]>0:
            G = block_array(
                [
                    [self.In,self.E.T],
                    [self.E ,None]
                ],format = 'csc'
            )
            rhs = np.hstack([x,self.e])
            solver = qdldl.Solver(G)
            x_nu = solver.solve(rhs)
            x = x_nu[:self.n]
            nu = x_nu[self.n:]
        else:
            nu = np.ones((0,))

        
        
        if s0 is None:
            s = np.maximum(self.c - self.C@x,0.01)
        else:
            s = np.copy(s0)
        
        return x,y,s,nu
    
    def KKT_res(self,x,g,y,s,nu):
        rx = g + self.C.T@y + self.E.T@nu - self.b
        rp = self.C@x + s - self.c
        rc = y * s
        req = self.E@x - self.e
        return rx,rp,rc,req
    
    def solve_KKT(
        self,
        x,y,s,nu,H,rx,rp,rc,req,mu,tau_reg=None,
        solver = None):
        #mu,x unused for now

        if tau_reg is None:
            tau_reg = self.settings.tau_reg
        #Nesterov-Todd scaling
        # Quasi definite for inequality constraints, 
        # "normal equations" Hessian for GLM part
        w = np.sqrt(y/s)
        wC = self.C.multiply(w[:,None])
        rhs = np.hstack([-rx,-w*rp + (w/y) * rc, -req])
        #Including tau-shift here
        #later may want separate matrix,
        #larger tau shift + iterative refine

        num_factorization_attempts = 8
        successful = False
        for attempt in range(num_factorization_attempts):
            try:
                G = block_array(
                    [
                        [H+tau_reg*self.In,wC.T        ,self.E.T],
                        [wC               ,-1*self.Ik  ,None],
                        [self.E           ,None         ,-tau_reg * self.Ip]
                    ],format = 'csc'
                )
                G = triu(G,format = 'csc')
                G.sort_indices()
                if solver is None:
                    solver = qdldl.Solver(G,upper = True)
                else:
                    solver.update(G,upper = True)
                successful = True
                break
            except:
                tau_reg = 10*tau_reg
        if successful ==False:
            raise ValueError(f"KKT Factorization Failed with {tau_reg} regularization!")
        sol = solver.solve(rhs)
        dx = sol[:self.n]
        dy = w*sol[self.n:self.n+self.k]
        nu = sol[self.n+self.k:]
        ds = -rp - self.C@dx
        return dx,ds,dy,nu,solver
    
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
        verbose = True
        ):
        solved = False
        near_solved = False
        convergence_tag = 'not_optimal'
        x,y,s,nu = self.initialize(x0,y0,s0)
        if verbose is True:
            print_problem_summary(
                n=self.n,
                m=self.m,
                p=self.E.shape[0],
                k=self.k
            )


        logger = PrettyLogger(verbose=verbose)
        settings = self.settings
        feasible = False
        armijo_param = 0.01

        start = time.time()
        if mu0 is None:
            mu = 100.
        else:
            mu = mu0

        z = self.A@x
        H = self.get_H(z)
        gradf = self.A.T@self.f.d1f(z) + self.Q@x
        rx,rp,rc,req = self.KKT_res(x,gradf,y,s,nu)
        kkt_res = np.max(
                np.abs(np.hstack([rx,rp,rc]))
            )
        #Perturb to interior point
        rc = rc - mu

        
        solver = None    
        for i in range(settings.max_iter):
            if maxnorm(rp)<1e-8:
                feasible = True
            
            if kkt_res<=100*settings.tol:
                near_solved = True
            
            #Check for convergence
            if kkt_res<=settings.tol:
                solved = True
                break
            
            #Check for stagnation
            if i>6 and kkt_res>=0.99*logger.rows[-6]['KKT_res']:
                #Little progress in 5 steps
                convergence_tag = "stagnated"
                warn("Giving up due to stagnation")
                break

            if feasible is False:
                tau_reg = 0.01 * np.mean(H.diagonal())
            else:
                tau_reg = self.settings.tau_reg

            #Solve KKT
            dx,ds,dy,dnu,solver = self.solve_KKT(
                x,y,s,nu,
                H,rx,rp,rc,req,
                mu,tau_reg,solver
                )

            if (feasible is True) and maxnorm(rc)>10*maxnorm(rx):
                boundary_frac = settings.greedy_boundary_frac
            else:
                boundary_frac = settings.safe_boundary_frac
            tmax = get_step_size(s,ds,y,dy,frac = boundary_frac)

            #Linesearch procedure here

            #Set up merit function
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
            if (feasible is True) and (step_finite is True):
                def merit_line(t):
                    primal = self.f(z+t*dz) + (1/2) * (x+t*dx).T@self.Q@(x+t*dx) - np.dot(x,self.b)
                    barrier = -mu*np.sum(np.log(s+t*ds))
                    return primal + barrier
                successful = False
                for linesearch_step in range(settings.max_linesearch_steps):
                    new_merit = merit_line(t)
                    if new_merit<merit0 + armijo_param * t * (np.dot(dx,gx) + np.dot(ds,gs)):
                        successful = True
                        break
                    else:
                        t = 0.9*t
                if successful ==False:
                    convergence_tag = "failed_line_search"
                    warn("Linesearch Failed!")
                    break
            
            #Take step
            x = x + t*dx
            s = s + t*ds
            y = y + t*dy
            nu = nu + t*dnu 
            z = z + t*dz

            gradf = self.A.T@self.f.d1f(z) + self.Q@x
            rx,rp,rc,req = self.KKT_res(x,gradf,y,s,nu)

            H = self.get_H(z)
            kkt_res = np.max([maxnorm(rx),maxnorm(rp),maxnorm(rc),maxnorm(req)])

            #If we're reasonably close to primal feasibility and 
            # complementarity aggressive mu update
            if maxnorm(rc) + maxnorm(rp)<=5*mu+np.minimum(1000 * mu,1000):
                mu_est = np.dot(s,y)/self.k
                xi = np.min(s*y)/mu_est
                #Don't decrease by more than a factor of 100
                mu_lower = np.maximum(mu*0.01,settings.min_mu)
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
                iter=i+1,
                primal = primal,
                dual_res = maxnorm(rx),
                cons_viol = np.maximum(maxnorm(rp),maxnorm(req)),
                comp_res = comp_res,
                mu=mu,
                Δx = maxnorm(t*dx),
                step=t,
                KKT_res=kkt_res,
                time=elapsed,
            )

        if solved ==True:
            convergence_tag = 'optimal'
        else:
            if convergence_tag=='not_optimal':
                convergence_tag = 'maximum_iter'
                warn("Maximum Iterations Reached")
            if near_solved is True:
                convergence_tag = f"near_opt_{convergence_tag}"
        results = SolverResults(settings,x,y,s,logger.to_dataframe(),convergence_tag = convergence_tag)
        return x,results
