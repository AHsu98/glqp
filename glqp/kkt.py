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

    def assemble(self, H, y, s, rx, rp, rc, req, prox_reg):
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
        np.add(y, prox_reg, out=self.w)
        np.sqrt(self.w, out=self.w)
        np.multiply(self.C.data, self.w[self.C.indices], out=self._scaled_C)
        data[self._c_lower] = self._scaled_C
        data[self._c_upper] = self._scaled_C
        data[self._ineq_diag] = -(s + prox_reg*y + prox_reg**2)
        data[self._eq_diag] = -prox_reg

        n, k = self.n, self.k
        self.rhs[:n] = -rx
        # Preserve the original arithmetic order in the scaled RHS.
        self.rhs[n:n+k] = (1/self.w)*rc - self.w*rp
        self.rhs[n+k:] = -req
        return self.matrix, self.rhs, self.w
