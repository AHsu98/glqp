import numpy as np
from scipy.sparse import csc_array,csr_array,diags_array
from scipy.sparse import random_array
from scipy.special import expit
from numpy import logaddexp
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_array,eye_array
from sparse_dot_mkl import dot_product_mkl

from obj import LogisticNLL
from numpy.random import default_rng


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

    C = random_array((k,n),
                    density = 0.2,rng = rng,data_sampler = samp)

    c = C@xx + 0.01
    Q = 1e-7*diags_array(np.ones(n))
    f = LogisticNLL(y,w)
    b = np.zeros(n)
    return f,A,Q,C,c,b

def build_logistic_lasso(m,n,seed,lam = 0.1):
    rng = default_rng(seed)
    samp = lambda size:rng.uniform(low = -0.5,high = 0.5,size = size)
    A = random_array((m,n),density = 0.01,rng = rng,data_sampler = samp)
    x_true = rng.choice([-1,0,1],size = n,p = [0.05,0.9,0.05])
    z_true = A@x_true

    A = block_array(
        [[A,csc_array((m,n))]]
    )
    w = 1000*np.ones(m)
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



