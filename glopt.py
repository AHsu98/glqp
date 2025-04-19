import numpy as np
import time
from scipy.sparse import csc_array
import qdldl
from scipy.sparse import block_array,eye_array
from sparse_dot_mkl import dot_product_mkl
from util import PrettyLogger,get_step_size,maxnorm,norm2
from warnings import warn

from dataclasses import dataclass

@dataclass
class SolverSettings():
    max_precenter = 100
    max_iter = 200
    tol = 1e-7
    safe_boundary_frac = 0.99
    greedy_boundary_frac = 0.9999
    gamma = 0.5
    min_mu = 1e-11
    tau_reg =1e-12
    max_linesearch_steps = 50


class GLMProblem():
    def __init__(
        self,
        f,A,Q,C,c,
        b=None,
        settings = None
        ):
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

        if b is None:
            b = np.zeros(n)
        self.b = b
        
        self.f = f
        self.A = A
        self.Q = Q
        self.C = C
        self.m = m
        self.n = n
        self.In = csc_array(eye_array(n))
        self.Ik = csc_array(eye_array(k))
    
    def initialize(self,x0=None,y0 = None,s0 = None):
        if x0 is None:
            x = np.zeros(self.n)
        else:
            x = np.copy(x0)
        
        if y0 is None:
            y = np.ones(self.k)
        else:
            y = np.copy(y0)
            assert np.min(y)>1e-8
        
        if s0 is None:
            s = np.maximum(self.c - self.C@x,0.01)
        else:
            s = np.copy(s0)
        
        return x,y,s
    
    def KKT_res(self,x,g,y,s):
        rx = g + self.C.T@y - self.b
        rp = self.C@x + s - self.c
        rc = y * s
        return rx,rp,rc
    
    def solve_KKT(
        self,
        x,y,s,H,rx,rp,rc,mu,tau_reg=None,
        solver = None):
        #mu,x unused for now

        if tau_reg is None:
            tau_reg = self.settings.tau_reg
        #Nesterov-Todd scaling
        # Quasi definite for inequality constraints, 
        # "normal equations" Hessian for GLM part
        w = np.sqrt(y/s)
        wC = self.C.multiply(w[:,None])
        #Including tiny tau-shift here
        #later may want separate matrix,
        #larger tau shift + iterative refine
        G = block_array(
            [
                [H+self.settings.tau_reg*self.In,wC.T],
                [wC,-1*self.Ik]
            ],format = 'csc'
        )
        rhs = np.hstack([-rx,-w*rp + (w/y) * rc])
        if solver is None:
            solver = qdldl.Solver(G)
        else:
            solver.update(G)
        sol = solver.solve(rhs)
        # linres = rhs - G@sol
        dx = sol[:self.n]
        dy = w*sol[self.n:]
        ds = -rp - self.C@dx
        return dx,ds,dy,solver
    
    def get_H(self,z):
        D = self.f.d2f(z)[:,None]
        AtDa = dot_product_mkl(self.A.T,csc_array(self.A.multiply(D)))
        return AtDa + self.Q
    
    def solve(
        self,
        x0=None,
        y0=None,
        s0=None,
        verbose = True
        ):
        x,y,s = self.initialize(x0,y0,s0)
        if verbose is True:
            print(f"{self.k} constraints")
            print(f"{self.n} variables")
            print(f"{self.m} rows in A")

        logger = PrettyLogger(verbose=verbose)
        settings = self.settings
        feasible = False
        armijo_param = 0.01

        start = time.time()
            
        mu = 100.

        z = self.A@x
        H = self.get_H(z)
        gradf = self.A.T@self.f.d1f(z) + self.Q@x
        rx,rp,rc = self.KKT_res(x,gradf,y,s)
        kkt_res = np.max(
                np.abs(np.hstack([rx,rp,rc]))
            )
        #Perturb to interior point
        rc = rc - mu

        
        solver = None    
        for i in range(settings.max_iter):
            if maxnorm(rp)<1e-8:
                feasible = True
            if kkt_res<=settings.tol:
                break


            if feasible is False:
                tau_reg = 0.1 * np.mean(H.diagonal())
            else:
                tau_reg = self.settings.tau_reg

            #Solve KKT
            dx,ds,dy,solver = self.solve_KKT(x,y,s,H,rx,rp,rc,mu,tau_reg,solver)

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
                    warn("Linesearch Failed!")
                    return x,logger.to_dataframe()
                
            w = np.sqrt(y/s)
            
            #Take step
            x = x + t*dx
            s = s + t*ds
            y = y + t*dy
            z = z + t*dz

            gradf = self.A.T@self.f.d1f(z) + self.Q@x
            rx,rp,rc = self.KKT_res(x,gradf,y,s)
            print("rx",maxnorm(rx))
            print("rp",maxnorm(rp))
            print("rc",maxnorm(rc))

            #Try to steal an extra corrector step
            rhs = np.hstack([-rx,-w*rp + (w/y) * rc])
            sol = solver.solve(rhs)
            # linres = rhs - G@sol
            dx = sol[:self.n]
            dy = w*sol[self.n:]
            ds = -rp - self.C@dx
            tmax = get_step_size(s,ds,y,dy,frac = settings.safe_boundary_frac)
            x = x + tmax*dx
            s = s + tmax*ds
            y = y + tmax*dy
            z = self.A@x
            gradf = self.A.T@self.f.d1f(z) + self.Q@x
            rx,rp,rc = self.KKT_res(x,gradf,y,s)
            print("corrected rx",maxnorm(rx))
            print("corrected rp",maxnorm(rp))
            print("corrected rc",maxnorm(rc))

            #end here


            H = self.get_H(z)
            kkt_res = np.max([maxnorm(rx),maxnorm(rp),maxnorm(rc)])
            #If we're reasonably close to primal feasibility and 
            # complementarity aggressive mu update
            if maxnorm(rc) + maxnorm(rp)<=1000 * mu:
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
                if i==0:
                    #Purely take first estimate
                    m = mu_est
                else:
                    mu = 0.9*np.minimum(mu_est,mu)

            comp_res = maxnorm(rc)

            #Perturb complementarity to new interior parameter
            rc = rc - mu

            elapsed = time.time() - start
            logger.log(
                iter=i+1,
                primal = primal,
                dual_res = maxnorm(rx),
                cons_viol = maxnorm(rp),
                comp_res = comp_res,
                mu=mu,
                Δx = maxnorm(t*dx),
                step=t,
                KKT_res=kkt_res,
                cum_time=elapsed,
            )
        return x,logger.to_dataframe()
