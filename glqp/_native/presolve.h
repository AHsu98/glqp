#ifndef GLQP_PRESOLVE_H
#define GLQP_PRESOLVE_H

/* Standalone C ABI for min 0.5 x'Qx + q'x, Ax=b, Gx<=h.
 * Matrices are full, canonical CSC, with sorted unique row indices, no stored
 * zeros, finite coefficients, and signed 64-bit indices. Q must be symmetric.
 * Inputs are copied; result lifetime is independent of the input buffers.
 * All returned pointers belong to the result and expire at gqp_free().
 * No function throws across this boundary. Thread-safe, no global state.
 */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int64_t rows, cols, nnz;
    const int64_t *indptr, *indices;
    const double *data;
} gqp_matrix;

typedef struct {
    int singleton_equalities;
    int fixed_bounds;
    int redundant_bounds;
    /* Relative deadband for declaring infeasibility; uncertain contradictions
     * return GQP_INCONCLUSIVE and never become valid reductions. */
    double infeasibility_guard;
} gqp_options;

enum gqp_status {
    GQP_UNCHANGED = 0, GQP_REDUCED = 1, GQP_INFEASIBLE = 2,
    GQP_INCONCLUSIVE = 3, GQP_INVALID = 4, GQP_ERROR = 5
};

/* A non-null result is returned even for invalid/infeasible inputs, unless
 * allocation fails. Inconclusive means arithmetic could not certify a
 * decision: the caller must solve the original problem. */
void *gqp_presolve(gqp_matrix Q, const double *q, gqp_matrix A,
                   const double *b, gqp_matrix G, const double *h,
                   gqp_options options);
void gqp_free(void *result);
int gqp_status_code(const void *result);
const char *gqp_message(const void *result);
double gqp_objective_offset(const void *result);
/* matrix: 0=Q, 1=A, 2=G; vector: 0=q, 1=b, 2=h. */
gqp_matrix gqp_get_matrix(const void *result, int which);
const double *gqp_get_vector(const void *result, int which);
/* map: 0=remaining variables, 1=remaining equality rows, 2=inequality rows.
 * stats: 0=fixed variables, 1=removed equalities, 2=removed inequalities,
 * 3=processed singleton rows, 4=propagation visits, 5=original nonzeros. */
const int64_t *gqp_get_map(const void *result, int which);
int64_t gqp_get_stat(const void *result, int which);
/* Reconstruct primal and both dual vectors. Input lengths are reduced
 * dimensions, output lengths original dimensions. Zero lengths permit null.
 * Returns 0 on success, nonzero for invalid/nonfinite input or overflow. */
int gqp_recover(const void *result, const double *x, const double *y,
                const double *z, double *original_x, double *original_y,
                double *original_z);

#ifdef __cplusplus
}
#endif
#endif
