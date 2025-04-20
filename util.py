from collections import OrderedDict
import itertools
import numpy  as np
import pandas as pd

def norm2(x):
    return np.sum(x**2)

def maxnorm(x):
    return np.max(np.abs(x))

def get_step_size(s, ds, y, dy,frac = 0.99):
    """
    Returns stepsize
      s + alpha*ds > 0  and  lam + alpha*dlam > 0
    for all components. with safety factor of frac
    """    
    # For s + alpha*ds > 0  =>  alpha < -s[i] / ds[i] for ds[i] < 0
    idx_s_neg = ds < 0
    if np.any(idx_s_neg):
        alpha_s = np.min(-s[idx_s_neg] / ds[idx_s_neg])
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

class PrettyLogger:
    """
    table printer **and** in‑memory recorder.
      • Create once; call `log(iter=..., mu=..., ...)` each IPM step.
      • All rows are kept in `self.rows` (a list of OrderedDicts).
      • Call `to_dataframe()` at any point to get a pandas DataFrame.
    """

    _LINE_CHAR = "─"
    _COL_SEP   = "│"

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(self,col_specs=None,verbose = True):
        if col_specs is None:
            col_specs = OrderedDict([
                ("iter",      "{:>4d}"),
                ("primal",    "{:>9.4e}"),
                ("dual_res", "{:>9.2e}"),
                ("cons_viol", "{:>9.2e}"),
                ("comp_res", "{:>9.2e}"),
                ("KKT_res",   "{:>9.2e}"),
                ("mu",        "{:>8.1e}"),
                ("Δx",        "{:>9.1e}"),
                ("step",      "{:>6.1e}"),
                ("cum_time",  "{:>8.2f}s"),
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

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def log(self, **kwargs):
        """
        Print one formatted row *and* append it to self.rows.
        Missing keys show as blanks in the table and as None in storage.
        Extra keys are ignored in printing but kept in storage.
        """
        
        # -------- pretty print --------
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

        # -------- store raw values --------
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

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
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
