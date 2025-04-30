import numpy as np
from scipy.sparse import csc_array,csr_array,diags_array
from scipy.sparse import random_array
from scipy.special import expit
from numpy import logaddexp
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_array,eye_array
from sparse_dot_mkl import dot_product_mkl

from glqp.obj import LogisticNLL
from numpy.random import default_rng
import scipy as sp


def build_random_logistic(m,n,k,seed,density = 0.01):
    rng = default_rng(seed)
    samp = lambda size:rng.uniform(low = -0.5,high = 0.5,size = size)
    A = random_array((m,n),density = density,rng = rng,data_sampler = samp)

    x_true = rng.uniform(-0.1,1,n)
    z_true = A@x_true
    w = 100*np.ones(m)
    y = rng.binomial(w.astype(int),expit(z_true))/w

    Q = 1.*diags_array(np.ones(n))
    #Create feasible problem.
    xx = rng.normal(size = n)
    if k>0:
        C = random_array((k,n),
                        density = 0.2,rng = rng,data_sampler = samp)

        c = np.atleast_1d(C@xx) + 0.01
    else:
        C = None
        c = None
    Q = 1e-7*diags_array(np.ones(n))
    f = LogisticNLL(y,w)
    b = np.zeros(n)
    return f,A,Q,C,c,b

def build_logistic_lasso(m,n,seed,lam = 0.1,weight = 1000):
    rng = default_rng(seed)
    samp = lambda size:rng.uniform(low = -0.5,high = 0.5,size = size)
    A = random_array((m,n),density = 0.01,rng = rng,data_sampler = samp)
    x_true = rng.choice([-1,0,1],size = n,p = [0.05,0.9,0.05])
    z_true = A@x_true

    A = block_array(
        [[A,csc_array((m,n))]]
    )
    w = weight*np.ones(m)
    y = rng.binomial(w.astype(int),expit(z_true))/w

    Q = 0.*diags_array(np.ones(2*n))

    In = diags_array(np.ones(n))
    C = block_array(
        [
            [In, -In],
            [-In,-In]
        ]
    )
    c = np.zeros(2*n)
    f = LogisticNLL(y,w)
    b = np.hstack([np.zeros(n),-lam*np.ones(n)])
    return f,A,Q,C,c,b,x_true

def parse_maros(mm_data):
    n = len(mm_data["lb"])
    P = mm_data["Q"]
    c = np.squeeze(mm_data["c"], axis=1)

    Amm = mm_data["A"].tocsc()
    rl = np.squeeze(mm_data["rl"], axis=1)
    ru = np.squeeze(mm_data["ru"], axis=1)
    lb = np.squeeze(mm_data["lb"], axis=1)
    ub = np.squeeze(mm_data["ub"], axis=1)

    eq_idx = np.where(rl == ru)[0]
    ineq_idx = np.where(rl != ru)[0]

    Aeq = Amm[eq_idx]
    beq = rl[eq_idx]

    Aineq = Amm[ineq_idx]
    uineq = ru[ineq_idx]
    lineq = rl[ineq_idx]

    G = sp.sparse.vstack(
        (sp.sparse.identity(n), -sp.sparse.identity(n), Aineq, -Aineq)
    ).tocsc()

    h = np.hstack((ub, -lb, uineq, -lineq))

    # Drop inf
    idx = np.where(h != np.inf)
    G = G[idx]
    h = h[idx]

    m = G.shape[0]
    p = Aeq.shape[0]

    l = m

    Q = P
    b = -1*np.copy(c.astype(float))
    if p>0:
        E = Aeq
        e = beq
    else:
        E = None
        e = None

    if m>0:
        C = G
        c = h
    else:
        C = None
        c = None
    
    return Q,b,C,c,E,e
