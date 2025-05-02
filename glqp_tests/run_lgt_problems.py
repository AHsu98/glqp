from glqp_tests.build_problems import build_random_logistic,build_logistic_lasso
from glqp_tests.test_tools import run_problem
from glqp import GLQP
import pandas as pd
import numpy as np
from itertools import product

if __name__=='__main__':
    from glqp_tests.build_problems import build_random_logistic,build_logistic_lasso
    from glqp_tests.test_tools import run_problem
    from glqp import GLQP,SolverSettings
    import pandas as pd
    import numpy as np
    from itertools import product

    m = 10000
    n_vals = [10,100,500]
    k_vals = [0,100,200]
    seeds = range(5)
    results = {}
    for n,k,seed in product(n_vals,k_vals,seeds):
        f,A,Q,C,c,b = build_random_logistic(m = m,n = n,k = k,density=0.05, seed = seed)
        problem = GLQP(f=f,A=A,Q=Q,b=b,C=C,c=c)
        res = run_problem(problem)
        name = ("lgt",n,k,seed)
        results[name] = res

    df = pd.DataFrame.from_dict(results,orient = 'index')
    df.index = df.index.set_names(["problem_type",'n','k','seed'])
    if (df['succeeded']!=True).sum()>0:
        print(df[df['succeeded']!=True])

    print(f"{((df['succeeded']==True).sum())} out of {len(df)} logistic regression problems solved in {df['run_time'].sum():.2f}s")

    m = 10000
    n_vals = [10,100,500]
    lam_vals = [0.1,1.,10]
    seeds = range(5)
    results = {}
    for n,lam,seed in product(n_vals,lam_vals,seeds):
        f,A,Q,C,c,b,x_true = build_logistic_lasso(m = m,n = n,lam = lam, seed = seed)
        problem = GLQP(f=f,A=A,Q=Q,b=b,C=C,c=c)
        res = run_problem(problem)
        name = ("lgt",n,k,seed)
        results[name] = res

    df = pd.DataFrame.from_dict(results,orient = 'index')
    df.index = df.index.set_names(["problem_type",'n','k','seed'])
    if (df['succeeded']!=True).sum()>0:
        print(df[df['succeeded']!=True])

    print(f"{((df['succeeded']==True).sum())} out of {len(df)} LASSO logistic regression problems solved in {df['run_time'].sum():.2f}s")    

