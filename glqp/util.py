from collections import OrderedDict
import numpy  as np
import qdldl
from warnings import warn
import pandas as pd
from scipy.sparse import csc_array,eye_array
from .obj import DummyGLM

def norm2(x):
    if len(x)==0:
        return 0.
    else:
        return np.sum(x**2)

def maxnorm(x):
    if len(x)==0:
        return 0.
    else:
        return np.max(np.abs(x))
    
def factor_and_solve(
    G,rhs,reg_shift,
    init_tau_reg,
    solver,
    target_atol = 1e-10,
    max_solve_attempts=20,
    max_refinement_steps = 5,
):
    #Set fixed target_rtol for now
    target_rtol = 1e-6
    tau_increase_factor = 10
    
    succeeded = False
    tau_reg = init_tau_reg
    for i in range(max_solve_attempts):
        try:
            Gshift = G + tau_reg*reg_shift
            if solver is None:
                solver = qdldl.Solver(Gshift)
            else:
                solver.update(Gshift)
            sol = solver.solve(rhs)
            if np.any(np.isnan(sol)):
                raise ValueError(f"NaNs found in solution to linear system with tau = {tau_reg}")
            w = G@sol
            mult = np.dot(rhs,w)/norm2(w)
            sol = sol * np.dot(rhs,w)/norm2(w)
            
            #Start refinement loop
            num_refine = 0
            res = rhs - G@sol
            linsolve_rel_error = np.sqrt(norm2(res)/norm2(rhs))

            #Less stringent condition to perform 1 step of iterative refinement
            #Do 1 step under either condition
            if maxnorm(res)>=0.1*target_atol or linsolve_rel_error>target_rtol:
                d = solver.solve(res)
                w = G@d
                alpha = np.dot(res,w)/norm2(w)
                sol = sol + alpha * d
                res = rhs - G@sol#Fully recompute residual to avoid roundoff
                linsolve_rel_error = np.sqrt(norm2(res)/norm2(rhs))
                num_refine += 1
            #Check if linear solve is terrible
            if linsolve_rel_error>0.98 and (maxnorm(res)>0.98*maxnorm(rhs)):
                raise ValueError(
                    f"""ERROR: Linear solve computed to unacceptable relative L2 error of 
                    {np.sqrt(norm2(res)/norm2(rhs)):.4f}
                    """
                    )
            
            
            #Continue refinement until reaching at least target_atol
            for i in range(1,max_refinement_steps):
                #Refine if either condition is not satisfied
                if (maxnorm(res)>target_atol and linsolve_rel_error>target_rtol):
                    d = solver.solve(res)
                    w = G@d
                    alpha = np.dot(res,w)/norm2(w)
                    sol = sol + alpha * d
                    res = res - alpha * w
                    # res = rhs - G@sol
                    linsolve_rel_error = np.sqrt(norm2(res)/norm2(rhs))
                    num_refine += 1
            if maxnorm(res)>target_atol and linsolve_rel_error>target_rtol:
                if maxnorm(res)>target_atol:
                    warn(f"Poor linear solve: didn't reach target abs tolerance of {target_atol:.3e} in {num_refine} steps")
                if linsolve_rel_error>target_rtol:
                    warn(f"Poor linear solve: didn't reach target rel tolerance of {target_rtol:.3e} in {num_refine} steps")

            succeeded = True
            break
        except Exception as ex:
            last_ex  = ex
            tau_reg = tau_increase_factor*tau_reg
    if succeeded is False:
        warn(f"Failed to solve with attempted reg {tau_reg:.2e} after {max_solve_attempts} attempts")
        raise last_ex
    # if linsolve_rel_error>target_rtol:
    #     print("tau: ",tau_reg)
    #     print("rel error: ",linsolve_rel_error)
    #     print('abs error: ',maxnorm(res))
    #     print("maxnorm(rhs): ",maxnorm(rhs))

    return sol,num_refine,solver,linsolve_rel_error

def get_step_size(s, ds, y, dy,frac = 0.99):
    """
    Returns stepsize
      s + alpha*ds > 0  and  lam + alpha*dlam > 0
    for all components. with safety factor of frac
    """    
    # For s + alpha*ds > 0  =>  alpha < -s[i] / ds[i] for ds[i] < 0
    idx_s_neg = ds < 0
    if np.any(idx_s_neg):
        alpha_s = np.min(-(s[idx_s_neg]) / ds[idx_s_neg])
    else:
        alpha_s = np.inf  # If ds >= 0, it doesn't limit alpha
    
    # For y + alpha*dy > 0  =>  y < -y[i] / dy[i] for dy[i] < 0
    idx_y_neg = dy < 0
    if np.any(idx_y_neg):
        alpha_lam = np.min(-y[idx_y_neg] / dy[idx_y_neg])
    else:
        alpha_lam = np.inf
    
    alpha = min(frac*alpha_s, frac*alpha_lam, 1.0)
    return alpha

def print_problem_summary(n, m, p, k):
    """
      n : number of decision variables
      m : rows in data matrix A
      p : equality constraints (rows in E)
      k : inequality constraints (rows in C)
    """
    line = (
        "| GLQP | "
        f"{'Variables':>3}: {n:<4,} │ "
        f"{'Rows in A':>3}: {m:<4,} │ "
        f"{'Equality Constraints':>3}: {p:<4,} │ "
        f"{'Inequality Constraints':>3}: {k:<4,}"
    )
    bar  = "─" * len(line)
    print(bar)
    print(line)

class Logger:
    """
    table printer and convergence tracker.
      • Create once; call `log(iter=..., mu=..., ...)` each IPM step.
      • All rows are kept in `self.rows` (a list of OrderedDicts).
      • Call `to_dataframe()` at any point to get a pandas DataFrame.
    """

    _LINE_CHAR = "─"
    _COL_SEP   = "│"

    def __init__(self,col_specs=None,verbose = True):
        if col_specs is None:
            col_specs = OrderedDict([
                ("iter",      "{:>4d}"),
                ("primal",    "{:>10.3e}"),
                ("dual_res", "{:>9.2e}"),
                ("cons_viol", "{:>9.2e}"),
                ("comp_res", "{:>9.2e}"),
                ("KKT_res",   "{:>9.2e}"),
                ("mu",        "{:>8.1e}"),
                ("Δx",        "{:>7.1e}"),
                ("step",      "{:>6.1e}"),
                ("refine",    "{:>6d}"),
                ("time",  "{:>6.2f}s"),
                ("lin_rel_res", "{:8.4e}")
            ])
        if not isinstance(col_specs, OrderedDict):
            col_specs = OrderedDict(col_specs)

        self.col_specs     = col_specs
        self._hdr_printed  = False
        self._border       = self._LINE_CHAR * (
            sum(self._col_widths()) + 3 * len(col_specs) + 1
        )

        self.rows: list[OrderedDict] = []
        self.verbose = verbose

    def log(self, **kwargs):
        """
        Print one formatted row *and* append it to self.rows.
        Missing keys show as blanks in the table and as None in storage.
        Extra keys are ignored in printing but kept in storage.
        """
        
        if self.verbose is True:
            if not self._hdr_printed:
                self._print_header()
                self._hdr_printed = True

            fmt_cells = []
            for key, fmt in self.col_specs.items():
                val = kwargs.get(key, "")
                cell = fmt.format(val) if val != "" else " " * self._width(fmt)
                fmt_cells.append(cell)
            row_str = f"{self._COL_SEP} " + f" {self._COL_SEP} ".join(fmt_cells) + f" {self._COL_SEP}"
            print(row_str)

        stored = OrderedDict()
        for key in self.col_specs:                  # preserve column order
            stored[key] = kwargs.get(key, None)     # None if not supplied
        # also keep any extra diagnostics the solver included
        for k, v in kwargs.items():
            if k not in stored:
                stored[k] = v
        self.rows.append(stored)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full history as a pandas DataFrame."""
        return pd.DataFrame(self.rows)

    def _print_header(self):
        hdr_cells = [
            f"{name:^{self._width(fmt)}}" for name, fmt in self.col_specs.items()
        ]
        header = f"{self._COL_SEP} " + f" {self._COL_SEP} ".join(hdr_cells) + f" {self._COL_SEP}"
        print(self._border)
        print(header)
        print(self._border)

    def _width(self, fmt: str) -> int:
        """Width of the formatted string produced by *fmt* for the dummy value 0."""
        return len(fmt.rstrip('s').format(0))

    def _col_widths(self):
        return (self._width(fmt) for fmt in self.col_specs.values())

def build_solution_summary(
    solved,near_solved,termination_tag,KKT_res,iter,elapsed,exception
):
    if solved ==True:
        termination_tag = 'optimal'
        msg = f"Optimal solution after {iter} iterations in {elapsed:.2f}s."
    else:
        if termination_tag=='not_optimal':
            termination_tag = 'maximum_iter'
            warn("Maximum Iterations Reached")
            if near_solved is True:
                termination_tag = f"near_opt_{termination_tag}"
                msg = f"Maximum of {iter} iterations reached in {elapsed:.2f}s. Tolerance was almost achieved."
            else:
                msg = f"Maximum of {iter} iterations reached in {elapsed:.2f}s. "
        
        elif termination_tag=="stagnated":
            warn("Giving up due to stagnation")
            if near_solved is True:
                termination_tag = f"near_opt_{termination_tag}"
                msg = f"Progress stagnated after {iter} iterations in {elapsed:.2f}s. Tolerance was almost achieved."
            else:
                msg = f"Progress stagnated after {iter} iterations in {elapsed:.2f}s."
        
        elif termination_tag=="failed_line_search":
            warn("Failed Line Search")
            if near_solved is True:
                termination_tag = f"near_opt_{termination_tag}"
                msg = f"Failed line search after {iter} iterations in {elapsed:.2f}s. Tolerance was almost achieved."
            else:
                msg = f"Failed line search after {iter} iterations in {elapsed:.2f}s."
        
        elif termination_tag == 'failed_linear_solve':
            #This implies that we have an exception given 
            warn("Failed Linear Solve")
            if near_solved is True:
                msg = f"""Failed linear solve after {iter} iterations in {elapsed:.2f}s. Tolerance was almost achieved. 
                [{exception.__str__()}]
                """

                termination_tag = f"near_opt_{termination_tag}"
                
            else:
                msg = f"{termination_tag} after {iter} iterations in {elapsed:.2f}s."
        
        elif termination_tag == "max_time":
            warn("Solver timed out")
            if near_solved is True:
                termination_tag = f"near_opt_{termination_tag}"
                msg = f"Solver timed out after {iter} iterations in {elapsed:.2f}s. Tolerance was almost achieved."
            else:
                msg = f"Solver timed out after {iter} iterations in {elapsed:.2f}s."


    msg = f"{msg} Final maxnorm KKT residual: {KKT_res:.2e}."
    return termination_tag,msg

def parse_problem(
    f=None,A=None,
    Q=None,b=None,
    C=None,c=None,
    E=None,e=None,
    n = None,
):
    #Figure out dimension of problem
    if n is not None:
        n = n
    elif A is not None:
        n = A.shape[1]
    elif Q is not None:
        n = Q.shape[1]
    elif C is not None:
        n = C.shape[1]
    elif E is not None:
        n = E.shape[1]
    else:
        raise ValueError("Not enough of problem specified, can't determine number of variables.")

    #Setup matrix A
    if A is None:
        A = csc_array((1,n))
        assert f is None, "Cannot have glm function f without A"
        f = DummyGLM()
        dummy_A = True
    else:
        A = csc_array(A)
        assert f is not None, "Need GLM if A is given"
        assert A.ndim==2
        dummy_A = False
    m = A.shape[0]

    #Setup inequality constraints
    if C is None:
        assert c is None
        C = csc_array((1,n))
        c = np.ones((1,))
        dummy_ineq = True
    elif C is not None:
        C = csc_array(C)
        assert C.ndim == 2
        #Need c if C is not None
        assert c is not None, "Need c if inequality matrix C is given"
        dummy_ineq = False
    k = C.shape[0]
    
    #Setup equality constraints
    if E is None:
        assert e is None
        E = csc_array((0,n))
        e = np.zeros((0,))
    elif E is not None:
        E = csc_array(E)
        assert e is not None, "Need e if equality matrix E is given"
    
    #Set up linear tilt
    if b is None:
        b = np.zeros(n)
    
    #Set up quadratic form
    if Q is None:
        Q = 1e-8*csc_array(eye_array(n))
    elif Q is not None:
        Q = csc_array(Q)
        assert Q.shape[0]==Q.shape[1], "Q must be square"
        assert np.min(Q.diagonal())>=0, "Q must be positive semi-definite, diagonal check reveals it cannot be"

    assert (
        A.shape[1] == 
        C.shape[1] == 
        E.shape[1] ==
        len(b)     ==
        Q.shape[1] ==
        Q.shape[0] ==
        n
    ), f"""Implied number of variables inconsistent.
    ncol(A) = {A.shape[1]}, ncol(C) = {C.shape[1]},
    ncol(E) = {E.shape[1]}, len(b) = {len(b)},
    ncol(Q) = {Q.shape[1]}, n = {n}
    """

    assert C.shape[0] == len(c)
    assert E.shape[0] == e.shape[0]
    p = E.shape[0]


    return (
        dummy_A,dummy_ineq,
        f,A,
        Q,b,
        C,c,
        E,e,
        m,n,p,k
    )

