import numpy as np
from glqp.util import maxnorm

def kkt_res(
    x,s,y,nu,
    problem
    ):
    z = problem.A@x
    rx = problem.A.T@(problem.f.d1f(z)) + problem.Q@x+problem.C.T@y + problem.E.T@nu - problem.b
    rp = problem.C@x + s - problem.c
    rc = s * y
    req = problem.E@x - problem.e
    return maxnorm(np.hstack([rx,rp,rc,req]))

def run_problem(problem,settings=None):
    x,sol = problem.solve(verbose = False,settings = settings)
    final_res = kkt_res(sol.x,sol.s,sol.y,sol.nu,problem)
    succeeded = final_res<problem.settings.tol
    return {
        'succeeded':succeeded,
        'run_time':sol.time,
        'termination_tag':sol.termination_tag,
        'iterations':sol.history.shape[0],
        'final_res':final_res,
    }