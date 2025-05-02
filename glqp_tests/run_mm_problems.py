import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glqp import GLQP
from glqp import SolverSettings
from glqp_tests.build_problems import parse_maros
from glqp_tests.test_tools import run_problem
import scipy as sp
import pickle

from pathlib import Path
import time

if __name__ == "__main__":
    start = time.time()
    directory = Path("mm_problems")
    paths = list(directory.iterdir())
    total = len(paths)
    results = {}
    for i,file_path in enumerate(paths):
        problem_name = file_path.stem
        print(problem_name)
        mat = sp.io.loadmat(file_path)
        Q,b,C,c,E,e = parse_maros(mat)

        settings = SolverSettings(
            tol =1e-7,
            max_iterative_refinement=20,
            max_stagnation=5000,
            max_iter = 10000,
            max_linesearch_steps=40,
            let_newton_cook = 0.5,
            max_time = 1500,
            )
        
        prob = GLQP(Q = Q,b = b,E = E,e = e,C = C,c = c)
        solve_result = run_problem(prob,settings = settings)
        print(solve_result['termination_tag'])
        print(f"Finished {i+1} out of {total}")
        print(f"{time.time() - start} elapsed")
        results[problem_name] = solve_result

    pd.DataFrame(results).T.to_csv("glqp_mm_results.csv")    