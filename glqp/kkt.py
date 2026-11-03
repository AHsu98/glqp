"""Reusable full symmetric KKT storage, independent of the linear backend."""
import numpy as np
from scipy.sparse import csc_array


def _canonical(matrix, *, copy=False):
    matrix = csc_array(matrix, copy=copy)
    if not matrix.has_canonical_format:
        if not copy:
            matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()
    return matrix


def _coordinates(matrix):
    return matrix.indices, np.repeat(np.arange(matrix.shape[1]), np.diff(matrix.indptr))


class KKTWorkspace:
    """Own matrix/RHS buffers for one solve with fixed C and E.

    Every assembly overwrites the current values, including proximal shifts.
    Hessian pattern changes rebuild the matrix and its entry maps. Returned
    buffers are borrowed and remain valid only until the next assembly. Input
    matrices are never modified; in particular H may be the objective's Q.
    Numerical factorization shifts and refinement remain the linear solver's concern.
    """

    def __init__(self, H, C, E):
        self.C = _canonical(C, copy=True)
        self.E = _canonical(E, copy=True)
        self.n = H.shape[0]
        self.k, self.p = C.shape[0], E.shape[0]
        if H.shape != (self.n, self.n) or C.shape[1] != self.n or E.shape[1] != self.n:
            raise ValueError("Incompatible KKT block dimensions")
        self.rhs = np.empty(self.n + self.k + self.p)
        self.w = np.empty(self.k)
        self._scaled_C = np.empty(self.C.nnz)
        self._build(_canonical(H))

    def _build(self, H):
        n, k, p = self.n, self.k, self.p
        size = n + k + p
        hr, hc = _coordinates(H)
        cr, cc = _coordinates(self.C)
        er, ec = _coordinates(self.E)
        diagonal = np.arange(size)
        # Include explicit diagonal slots even when H, rho, or a slack is zero.
        rows = np.concatenate([hr, cr + n, cc, er + n + k, ec, diagonal])
        cols = np.concatenate([hc, cc, cr + n, ec, er + n + k, diagonal])
        # COO arrays otherwise promote ordinary 32-bit problems to 64-bit CSC
        # storage. Keep compact indices when dimensions and entry counts fit.
        if max(size, len(rows)) <= np.iinfo(np.int32).max:
            rows = rows.astype(np.int32)
            cols = cols.astype(np.int32)
        self.matrix = csc_array((np.ones(len(rows)), (rows, cols)), shape=(size, size))
        self.matrix.data.fill(0.)
        mr, mc = _coordinates(self.matrix)
        keys = mc * size + mr

        def slots(rows, cols):
            # Linearized coordinates can exceed int32 even though each CSC
            # index fits. Widen before multiplying, including transposed blocks.
            return np.searchsorted(keys, np.asarray(cols, dtype=np.int64) * size + rows)

        self._h_slots = slots(hr, hc)
        self._c_lower = slots(cr + n, cc)
        self._c_upper = slots(cc, cr + n)
        diag_slots = slots(diagonal, diagonal)
        self._primal_diag = diag_slots[:n]
        self._ineq_diag = diag_slots[n:n+k]
        self._eq_diag = diag_slots[n+k:]
        self.matrix.data[slots(er + n + k, ec)] = self.E.data
        self.matrix.data[slots(ec, er + n + k)] = self.E.data
        self._h_indptr = H.indptr.copy()
        self._h_indices = H.indices.copy()

    def assemble(self, H, y, s, rx, rp, rc, req, prox_reg, *,
                 dual_reg=None, complementarity_reg=None):
        """Assemble the symmetric elimination of the four Newton blocks.

        ``prox_reg`` shifts the primal Hessian. ``dual_reg`` shifts both
        constraint equations, and ``complementarity_reg`` shifts the slack
        derivative of complementarity. Each optional shift inherits
        ``prox_reg`` when omitted, preserving the original scalar model.

        For primal, dual, and complementarity shifts ``rho_x, rho_y, rho_s``,
        the uneliminated equations are::

            (H + rho_x I) dx + C.T dy + E.T dnu = -rx
            C dx + ds - rho_y dy = -rp
            (Y + rho_s I) ds + S dy = -rc
            E dx - rho_y dnu = -req

        Proximal centers, when used, belong in the supplied residuals. Standard
        complementarity uses ``complementarity_reg=0``. The returned scaled
        dual direction must be multiplied by ``w``; recover the slack direction
        with ``ds = -rp - C @ dx + rho_y * dy``.
        """
        scalar_model = dual_reg is None and complementarity_reg is None
        rho_y = prox_reg if dual_reg is None else dual_reg
        rho_s = prox_reg if complementarity_reg is None else complementarity_reg
        if not scalar_model:
            for name, value in (("prox_reg", prox_reg), ("dual_reg", rho_y),
                                ("complementarity_reg", rho_s)):
                try:
                    valid = (np.ndim(value) == 0 and np.isreal(value)
                             and np.isfinite(value) and value >= 0)
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError(f"{name} must be a finite nonnegative scalar")
        H = _canonical(H)
        if H.shape != (self.n, self.n):
            raise ValueError("Hessian dimensions changed during a solve")
        if (not np.array_equal(H.indptr, self._h_indptr)
                or not np.array_equal(H.indices, self._h_indices)):
            self._build(H)

        data = self.matrix.data
        # Reset absent Hessian diagonal entries too, so shifts never accumulate.
        data[self._primal_diag] = 0.
        data[self._h_slots] = H.data
        data[self._primal_diag] += prox_reg
        np.add(y, rho_s, out=self.w)
        np.sqrt(self.w, out=self.w)
        np.multiply(self.C.data, self.w[self.C.indices], out=self._scaled_C)
        data[self._c_lower] = self._scaled_C
        data[self._c_upper] = self._scaled_C
        if scalar_model:
            # Keep the accepted scalar model's floating-point arithmetic exact.
            data[self._ineq_diag] = -(s + prox_reg*y + prox_reg**2)
        else:
            data[self._ineq_diag] = -(s + rho_y*y + rho_y*rho_s)
        data[self._eq_diag] = -rho_y

        self.assemble_rhs(rx, rp, rc, req)
        return self.matrix, self.rhs, self.w

    def assemble_rhs(self, rx, rp, rc, req):
        """Replace only the RHS using the weights from the current matrix.

        Call after ``assemble``. This permits sequential predictor/corrector
        solves without rebuilding or changing the borrowed matrix or weights.
        The returned RHS is borrowed until the next RHS or matrix assembly.
        """
        n, k = self.n, self.k
        self.rhs[:n] = -rx
        # Preserve the original arithmetic order in the scaled RHS.
        self.rhs[n:n+k] = (1/self.w)*rc - self.w*rp
        self.rhs[n+k:] = -req
        return self.rhs
