// Standalone C-ABI tests. No Python, BLAS, or testing-framework dependency.
// Run through tools/check_presolve_native.py for ASan + UBSan instrumentation.
#include "presolve.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Index = int64_t;
using Vector = std::vector<double>;
using Dense = std::vector<Vector>;
using Result = std::unique_ptr<void, decltype(&gqp_free)>;
constexpr gqp_options options{1, 1, 1, 1e-12, 0, 0};
int checks = 0;

void require(bool condition, const char *description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

struct Matrix {
    Index rows, cols;
    std::vector<Index> ptr, idx;
    Vector data;

    Matrix(const Dense &dense, Index columns) :
        rows(dense.size()), cols(columns), ptr(columns + 1) {
        for (Index col = 0; col < cols; ++col) {
            for (Index row = 0; row < rows; ++row) {
                require(static_cast<Index>(dense[row].size()) == cols, "Invalid test matrix");
                if (dense[row][col] != 0.) {
                    idx.push_back(row);
                    data.push_back(dense[row][col]);
                }
            }
            ptr[col + 1] = static_cast<Index>(data.size());
        }
    }

    gqp_matrix view() const {
        return {rows, cols, static_cast<Index>(data.size()), ptr.data(), idx.data(), data.data()};
    }
};

Vector multiply(gqp_matrix a, const Vector &x) {
    Vector out(a.rows);
    for (Index col = 0; col < a.cols; ++col)
        for (Index k = a.indptr[col]; k < a.indptr[col + 1]; ++k)
            out[a.indices[k]] += a.data[k] * x[col];
    return out;
}

void add_transpose(gqp_matrix a, const Vector &y, Vector &out) {
    for (Index col = 0; col < a.cols; ++col)
        for (Index k = a.indptr[col]; k < a.indptr[col + 1]; ++k)
            out[col] += a.data[k] * y[a.indices[k]];
}

double objective(gqp_matrix Q, const double *q, const Vector &x) {
    auto product = multiply(Q, x);
    double value = 0.;
    for (size_t j = 0; j < x.size(); ++j) value += x[j] * (.5 * product[j] + q[j]);
    return value;
}

void equal_vectors(const Vector &a, const Vector &b, double tol, const char *message) {
    require(a.size() == b.size(), message);
    for (size_t i = 0; i < a.size(); ++i)
        require(std::isfinite(a[i]) && std::abs(a[i] - b[i]) <= tol, message);
}

void check_known_solution(const Matrix &Q, const Matrix &A, const Matrix &G,
                          const Vector &x, const Vector &y, const Vector &z,
                          const Vector &slack, Index expected_fixed,
                          gqp_options flags = options) {
    Vector b = multiply(A.view(), x), h = multiply(G.view(), x);
    for (size_t i = 0; i < h.size(); ++i) h[i] += slack[i];
    Vector q = multiply(Q.view(), x);
    add_transpose(A.view(), y, q);
    add_transpose(G.view(), z, q);
    for (auto &v : q) v = -v;
    Result result(gqp_presolve(Q.view(), q.data(), A.view(), b.data(), G.view(), h.data(), flags), gqp_free);
    int status = gqp_status_code(result.get());
    require(status == GQP_REDUCED || status == GQP_UNCHANGED, gqp_message(result.get()));
    require(gqp_get_stat(result.get(), 0) == expected_fixed, "Unexpected elimination count");
    auto reduced_Q = gqp_get_matrix(result.get(), 0);
    auto reduced_A = gqp_get_matrix(result.get(), 1);
    auto reduced_G = gqp_get_matrix(result.get(), 2);
    Vector reduced_x(reduced_Q.cols), reduced_y(reduced_A.rows), reduced_z(reduced_G.rows);
    for (Index j = 0; j < reduced_Q.cols; ++j) reduced_x[j] = x[gqp_get_map(result.get(), 0)[j]];
    for (Index j = 0; j < reduced_A.rows; ++j) reduced_y[j] = y[gqp_get_map(result.get(), 1)[j]];
    for (Index j = 0; j < reduced_G.rows; ++j) reduced_z[j] = z[gqp_get_map(result.get(), 2)[j]];
    Vector recovered_x(Q.cols), recovered_y(A.rows), recovered_z(G.rows);
    require(gqp_recover(result.get(), reduced_x.data(), reduced_y.data(), reduced_z.data(),
                         recovered_x.data(), recovered_y.data(), recovered_z.data()) == 0,
            "Recovery failed");
    equal_vectors(recovered_x, x, 2e-12, "Primal recovery changed solution");
    equal_vectors(multiply(A.view(), recovered_x), b, 2e-12, "Original equality violation");
    Vector gx = multiply(G.view(), recovered_x);
    for (Index i = 0; i < G.rows; ++i) {
        require(gx[i] - h[i] <= 2e-12, "Original inequality violation");
        require(recovered_z[i] >= -2e-12, "Negative recovered dual");
        require(std::abs(recovered_z[i] * (gx[i] - h[i])) < 2e-11, "Original complementarity violation");
    }
    Vector gradient = multiply(Q.view(), recovered_x);
    add_transpose(A.view(), recovered_y, gradient);
    add_transpose(G.view(), recovered_z, gradient);
    for (Index j = 0; j < Q.cols; ++j) gradient[j] += q[j];
    equal_vectors(gradient, Vector(Q.cols), 2e-11, "Original stationarity violation");
    double before = objective(Q.view(), q.data(), recovered_x);
    double after = objective(reduced_Q, gqp_get_vector(result.get(), 0), reduced_x)
                   + gqp_objective_offset(result.get());
    require(std::abs(before - after) <= 1e-11 * std::max(1., std::abs(before)), "Wrong objective offset");

    if (!reduced_x.empty()) {
        reduced_x[0] = std::numeric_limits<double>::quiet_NaN();
        require(gqp_recover(result.get(), reduced_x.data(), reduced_y.data(), reduced_z.data(),
                             recovered_x.data(), recovered_y.data(), recovered_z.data()) != 0,
                "Nonfinite recovery input accepted");
        require(gqp_recover(result.get(), nullptr, reduced_y.data(), reduced_z.data(),
                             recovered_x.data(), recovered_y.data(), recovered_z.data()) != 0,
                "Missing primal recovery input accepted");
    }
}

void mixed_cascades() {
    std::mt19937 generator(20261016);
    std::uniform_int_distribution<int> small(-3, 3);
    for (int repetition = 0; repetition < 100; ++repetition) {
        constexpr Index n = 12;
        Dense Q(n, Vector(n));
        for (Index j = 0; j < n; ++j) {
            Q[j][j] = 20.;
            for (Index i = 0; i < j; ++i) Q[j][i] = Q[i][j] = small(generator) * .25;
        }
        Vector x(n);
        for (auto &value : x) value = small(generator);
        Dense A(7, Vector(n)), G(8, Vector(n));
        A[0][0] = 2.;
        A[1][0] = -1.; A[1][1] = 1.;
        A[2][1] = 3.; A[2][2] = -2.;
        A[3][3] = 2.; A[3][5] = 1.;
        A[4][4] = -1.; A[4][6] = 2.;
        A[5][7] = 1.; A[5][8] = -2.; // Retained multi-variable equality.
        // Final equality and inequalities are redundant empty rows.
        G[0][3] = 2.; G[1][3] = -2.;
        G[2][4] = 1.; G[3][4] = -1.;
        G[4][3] = 1.; // Loose redundant singleton bound.
        G[5][9] = 1.; // Active retained inequality with nonzero dual.
        Vector y(7), z(8);
        for (Index i = 0; i < 6; ++i) y[i] = small(generator);
        z[0] = 3.; z[3] = 2.; z[5] = 1.;
        Vector slack{0., 0., 0., 0., 4., 0., 0., 2.};
        auto flags = options;
        flags.local_fallback = repetition % 2;
        check_known_solution(Matrix(Q, n), Matrix(A, n), Matrix(G, n), x, y, z, slack, 7, flags);
    }
}

void empty_and_zero_dimensions() {
    Matrix Q({{2., .5}, {.5, 3.}}, 2), A({{1., 0.}, {2., 1.}}, 2), G({}, 2);
    check_known_solution(Q, A, G, {2., -1.}, {3., 1.}, {}, {}, 2);
    Matrix no_constraints({}, 2);
    check_known_solution(Q, no_constraints, no_constraints, {1., -2.}, {}, {}, {}, 0);
    Matrix no_variables({}, 0);
    Result empty(gqp_presolve(no_variables.view(), nullptr, no_variables.view(), nullptr,
                              no_variables.view(), nullptr, options), gqp_free);
    require(gqp_status_code(empty.get()) == GQP_UNCHANGED, "Empty problem status");
    require(gqp_recover(empty.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 0,
            "Zero-dimensional recovery failed");
    Matrix empty_rows({{}, {}}, 0);
    Vector zero(2), positive{0., 2.};
    Result rows(gqp_presolve(no_variables.view(), nullptr, empty_rows.view(), zero.data(),
                             empty_rows.view(), positive.data(), options), gqp_free);
    require(gqp_status_code(rows.get()) == GQP_REDUCED, "Empty rows not removed");
    Vector y(2), z(2);
    require(gqp_recover(rows.get(), nullptr, nullptr, nullptr, nullptr, y.data(), z.data()) == 0,
            "Empty-row dual recovery failed");
    equal_vectors(y, zero, 0., "Empty equality duals");
    equal_vectors(z, zero, 0., "Empty inequality duals");
    require(gqp_recover(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) != 0,
            "Null result accepted");
    gqp_free(nullptr);
}

void ownership() {
    Result result(nullptr, gqp_free);
    {
        Matrix Q({{2., 1.}, {1., 3.}}, 2), A({{1., 0.}}, 2), G({}, 2);
        Vector q{1., -2.}, b{2.};
        result.reset(gqp_presolve(Q.view(), q.data(), A.view(), b.data(), G.view(), nullptr, options));
    } // Every input buffer is gone; the C API promises independent ownership.
    require(gqp_status_code(result.get()) == GQP_REDUCED, "Ownership fixture did not reduce");
    double reduced_x = 0., x[2], y;
    require(gqp_recover(result.get(), &reduced_x, nullptr, nullptr, x, &y, nullptr) == 0,
            "Recovery borrowed expired input buffers");
    require(x[0] == 2. && x[1] == 0. && y == -5., "Unexpected owned reconstruction");
}

void malformed_inputs() {
    Matrix valid({{1., .5}, {.5, 2.}}, 2), empty({}, 2);
    Vector q{1., 2.};
    auto expect_invalid = [&](gqp_matrix Q, const double *vector, gqp_options flags = options) {
        Result result(gqp_presolve(Q, vector, empty.view(), nullptr, empty.view(), nullptr, flags), gqp_free);
        require(gqp_status_code(result.get()) == GQP_INVALID, "Malformed input was not rejected");
        require(gqp_recover(result.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) != 0,
                "Recovery accepted invalid result");
    };
    expect_invalid(valid.view(), nullptr);
    q[1] = std::numeric_limits<double>::infinity();
    expect_invalid(valid.view(), q.data());
    q[1] = 2.;
    auto view = valid.view(); view.rows = -1; expect_invalid(view, q.data());
    view = valid.view(); view.cols = 1; expect_invalid(view, q.data());
    view = valid.view(); view.nnz = -1; expect_invalid(view, q.data());
    view = valid.view(); view.indptr = nullptr; expect_invalid(view, q.data());
    view = valid.view(); view.indices = nullptr; expect_invalid(view, q.data());
    view = valid.view(); view.data = nullptr; expect_invalid(view, q.data());
    auto bad = valid; bad.ptr[0] = 1; expect_invalid(bad.view(), q.data());
    bad = valid; bad.ptr[1] = -1; expect_invalid(bad.view(), q.data());
    bad = valid; bad.ptr[1] = bad.ptr.back() + 1; expect_invalid(bad.view(), q.data());
    bad = valid; bad.ptr.back() -= 1; expect_invalid(bad.view(), q.data());
    bad = valid; bad.idx[0] = -1; expect_invalid(bad.view(), q.data());
    bad = valid; bad.idx[0] = 2; expect_invalid(bad.view(), q.data());
    bad = valid; bad.idx[1] = bad.idx[0]; expect_invalid(bad.view(), q.data());
    bad = valid; bad.idx[0] = 1; bad.idx[1] = 0; expect_invalid(bad.view(), q.data());
    bad = valid; bad.data[0] = 0.; expect_invalid(bad.view(), q.data());
    bad = valid; bad.data[0] = std::numeric_limits<double>::quiet_NaN(); expect_invalid(bad.view(), q.data());
    bad = valid; bad.data[1] += .25; expect_invalid(bad.view(), q.data());
    expect_invalid(valid.view(), q.data(), {1, 1, 1, -1., 0, 0});
    expect_invalid(valid.view(), q.data(), {1, 1, 1, std::numeric_limits<double>::infinity(), 0, 0});
}

void infeasible_and_uncertain() {
    Matrix Q({{1.}}, 1), empty({}, 1), fixed({{1.}, {-1.}}, 1), zero({{0.}}, 1);
    double q = 0., negative = -1.;
    Vector h{0., -1.};
    Result contradiction(gqp_presolve(Q.view(), &q, empty.view(), nullptr, fixed.view(), h.data(), options), gqp_free);
    require(gqp_status_code(contradiction.get()) == GQP_INFEASIBLE, "Contradictory bounds missed");
    Result empty_inequality(gqp_presolve(Q.view(), &q, empty.view(), nullptr, zero.view(), &negative, options), gqp_free);
    require(gqp_status_code(empty_inequality.get()) == GQP_INFEASIBLE, "Empty contradiction missed");
    Matrix proportional({{3.}, {6.}}, 1);
    Vector b{1., 2.};
    Result uncertain(gqp_presolve(Q.view(), &q, proportional.view(), b.data(), empty.view(), nullptr, options), gqp_free);
    require(gqp_status_code(uncertain.get()) == GQP_INCONCLUSIVE, "Inexact chain should fall back");
    Matrix tiny({{1e-300}}, 1);
    double huge = 1e300;
    Result overflow(gqp_presolve(Q.view(), &q, tiny.view(), &huge, empty.view(), nullptr, options), gqp_free);
    require(gqp_status_code(overflow.get()) == GQP_INCONCLUSIVE, "Overflow should fall back");
}

void local_deferrals() {
    auto local = options;
    local.local_fallback = 1;
    Matrix Q({{4., .5, 1.}, {.5, 3., 0.}, {1., 0., 2.}}, 3);
    Matrix A({{1., 0., 0.}, {1., 3., 0.}, {0., 6., 0.}, {0., 0., 1.}}, 3), G({}, 3);
    check_known_solution(Q, A, G, {2., 1./3., 4.}, {1., 2., -1., 3.}, {}, {}, 2, local);
    Matrix diagonal({{1., 0.}, {0., 1.}}, 2), equality({{0., 1.}}, 2);
    Matrix bounds({{3., 0.}, {-3., 0.}, {6., 0.}}, 2);
    check_known_solution(diagonal, equality, bounds, {1./3., 4.}, {2.}, {0., 0., 0.},
                         {0., 0., 0.}, 1, local);
    Matrix conflicting({{1., 0.}, {-1., 0.}}, 2);
    Vector q{0., 0.}, b{3.}, h{1., -1. - 1e-14};
    Result result(gqp_presolve(diagonal.view(), q.data(), equality.view(), b.data(),
                               conflicting.view(), h.data(), local), gqp_free);
    require(gqp_status_code(result.get()) == GQP_REDUCED, "Local bound deferral failed");
    require(gqp_get_stat(result.get(), 0) == 1 && gqp_get_stat(result.get(), 6) == 1 &&
            gqp_get_stat(result.get(), 7) == 2, "Local bound deferral counts wrong");
    auto kept = gqp_get_matrix(result.get(), 2);
    require(kept.rows == 2 && kept.nnz == 2, "Local deferral dropped conflicting bounds");
    require(gqp_get_vector(result.get(), 2)[1] == h[1], "Local deferral changed conflicting RHS");
}

void scalar_quadratics() {
    auto scalar = options;
    scalar.separable_quadratics = 1;
    Matrix Q({{2., 0., 0.}, {0., 4., 0.}, {0., 0., 8.}}, 3), empty({}, 3);
    check_known_solution(Q, empty, empty, {-.5, 1., -.75}, {}, {}, {}, 3, scalar);
    Matrix G({{-2., 0., 0.}, {-1., 0., 0.}, {0., 2., 0.},
              {0., 1., 0.}, {0., 0., -1.}, {0., 0., 1.}}, 3);
    check_known_solution(Q, empty, G, {-1., 2., .5}, {}, {2., 0., 4., 0., 0., 0.},
                         {0., 1., 0., 1., .5, .5}, 3, scalar);
    Matrix chained_Q({{1., 0.}, {0., 3.}}, 2), A({{1., 0.}}, 2), bound({{1., -1.}}, 2);
    check_known_solution(chained_Q, A, bound, {2., 3.}, {-11.}, {9.}, {0.}, 2, scalar);
    Matrix uncertain_Q({{3., 0.}, {0., 2.}}, 2), uncertain_G({{-3., 0.}}, 2), empty2({}, 2);
    Vector q{1., -2.}, h{1.};
    Result result(gqp_presolve(uncertain_Q.view(), q.data(), empty2.view(), nullptr,
                               uncertain_G.view(), h.data(), scalar), gqp_free);
    require(gqp_status_code(result.get()) == GQP_REDUCED, "Scalar local ambiguity fallback failed");
    require(gqp_get_stat(result.get(), 8) == 1 && gqp_get_stat(result.get(), 9) == 1,
            "Wrong separable scalar statistics");
    require(gqp_get_matrix(result.get(), 2).rows == 1, "Ambiguous scalar bound was dropped");
}
} // namespace

int main() {
    try {
        mixed_cascades();
        empty_and_zero_dimensions();
        ownership();
        malformed_inputs();
        infeasible_and_uncertain();
        local_deferrals();
        scalar_quadratics();
        std::cout << "Native presolver validation passed: " << checks
                  << " assertions, 100 mixed cascades, malformed-input and lifetime cases\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Native presolver validation failed: " << error.what() << '\n';
        return 1;
    }
}
