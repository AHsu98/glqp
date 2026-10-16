#include "presolve.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Index = int64_t;
constexpr double inf = std::numeric_limits<double>::infinity();

struct Matrix {
    Index rows = 0, cols = 0;
    std::vector<Index> ptr{0}, idx;
    std::vector<double> data;
    gqp_matrix view() const {
        return {rows, cols, static_cast<Index>(data.size()), ptr.data(),
                idx.data(), data.data()};
    }
};

Matrix copy_matrix(gqp_matrix source) {
    if (source.rows < 0 || source.cols < 0 || source.nnz < 0 || !source.indptr ||
        source.indptr[0] != 0 || source.indptr[source.cols] != source.nnz ||
        (source.nnz && (!source.indices || !source.data)))
        throw std::invalid_argument("Invalid CSC dimensions or pointers");
    Matrix out;
    out.rows = source.rows;
    out.cols = source.cols;
    out.ptr.assign(source.indptr, source.indptr + source.cols + 1);
    for (Index col = 0; col < source.cols; ++col) {
        if (out.ptr[col] > out.ptr[col + 1] || out.ptr[col] < 0 ||
            out.ptr[col + 1] > source.nnz)
            throw std::invalid_argument("Invalid CSC column pointers");
        Index previous = -1;
        for (Index k = out.ptr[col]; k < out.ptr[col + 1]; ++k) {
            if (source.indices[k] <= previous || source.indices[k] >= source.rows ||
                !std::isfinite(source.data[k]) || source.data[k] == 0.)
                throw std::invalid_argument("CSC coefficients must be finite, nonzero, sorted and unique");
            previous = source.indices[k];
        }
    }
    if (source.nnz) {
        out.idx.assign(source.indices, source.indices + source.nnz);
        out.data.assign(source.data, source.data + source.nnz);
    }
    return out;
}

std::vector<double> copy_vector(const double *source, Index count) {
    if (count && !source) throw std::invalid_argument("Missing vector");
    std::vector<double> out;
    if (count) out.assign(source, source + count);
    for (double value : out)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite vector");
    return out;
}

struct Interval { double lo, hi; };

// Error-free product/sum remainders let exact operations keep point intervals.
// Only inexact operations expand by one representable number. No feasibility
// tolerance is used to declare a variable fixed or a row redundant.
Interval product(double a, double b) {
    double value = a * b;
    if (!std::isfinite(value)) throw std::overflow_error("Overflow during substitution");
    double error = std::fma(a, b, -value);
    bool underflow = a != 0. && b != 0. && std::abs(value) <
        std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon();
    return {error < 0. || underflow ? std::nextafter(value, -inf) : value,
            error > 0. || underflow ? std::nextafter(value, inf) : value};
}

Interval subtract(double a, double b) {
    double value = a - b;
    if (!std::isfinite(value)) throw std::overflow_error("Overflow during substitution");
    double virtual_b = value - a;
    double error = (a - (value - virtual_b)) + (-b - virtual_b);
    return {error < 0. ? std::nextafter(value, -inf) : value,
            error > 0. ? std::nextafter(value, inf) : value};
}

Interval quotient(double a, double b) {
    double value = a / b;
    if (!std::isfinite(value)) throw std::overflow_error("Nonfinite fixed variable");
    double error = std::fma(-value, b, a);
    bool underflow = a != 0. && (std::abs(value) < std::numeric_limits<double>::min() ||
        std::abs(a) < std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon());
    bool below = (error < 0.) != (b < 0.);
    return {(error != 0. && below) || underflow ? std::nextafter(value, -inf) : value,
            (error != 0. && !below) || underflow ? std::nextafter(value, inf) : value};
}

Interval divide(Interval range, double coefficient) {
    if (coefficient > 0.)
        return {quotient(range.lo, coefficient).lo, quotient(range.hi, coefficient).hi};
    return {quotient(range.hi, coefficient).lo, quotient(range.lo, coefficient).hi};
}

struct Pivot {
    Index variable, equality, lower, upper;
    double equality_coefficient, lower_coefficient, upper_coefficient;
};

struct Result {
    int status = GQP_ERROR;
    std::string message;
    std::array<Matrix, 3> original, reduced;
    std::array<std::vector<double>, 3> original_rhs, rhs;
    std::array<std::vector<Index>, 3> maps;
    std::array<Index, 10> stats{};
    std::vector<Pivot> tape;
    std::vector<double> fixed_values;
    // Shared CSR of original A/G, retained for reverse dual reconstruction.
    std::vector<Index> row_ptr, row_col;
    std::vector<double> row_data;
    double offset = 0.;
};

class Presolver {
    Result &r;
    gqp_options options;
    Index n, p, m;
    std::vector<Index> count, single_col, queue, lower_row, upper_row;
    std::vector<uint64_t> coefficient_bits;
    std::vector<double> rhs, lo, hi, singleton_coefficient;
    std::vector<unsigned char> removed, fixed, processed, deferred, ambiguous;

    static uint64_t bits(double value) {
        uint64_t result;
        static_assert(sizeof(result) == sizeof(value));
        std::memcpy(&result, &value, sizeof(value));
        return result;
    }

    void initialize_rows() {
        // XOR preserves the exact surviving column/coefficient when a row
        // becomes singleton, without scanning the row or subtracting floats.
        for (int matrix = 1; matrix <= 2; ++matrix) {
            const auto &a = r.original[matrix];
            for (Index col = 0; col < n; ++col)
                for (Index k = a.ptr[col]; k < a.ptr[col + 1]; ++k) {
                    Index row = a.idx[k] + (matrix == 2 ? p : 0);
                    ++count[row];
                    single_col[row] ^= col;
                    coefficient_bits[row] ^= bits(a.data[k]);
                }
        }
        for (Index row = 0; row < p + m; ++row)
            if (count[row] <= 1) queue.push_back(row);
    }

    void build_recovery_rows() {
        // Only eliminated pivot rows are used by reverse dual recovery. Most
        // inputs eliminate nothing, and never allocate a second sparse view.
        if (r.tape.empty()) return;
        std::vector<unsigned char> pivot_row(p + m);
        for (const auto &pivot : r.tape) {
            if (pivot.equality >= 0) pivot_row[pivot.equality] = 1;
            if (pivot.lower >= 0) pivot_row[p + pivot.lower] = 1;
            if (pivot.upper >= 0) pivot_row[p + pivot.upper] = 1;
        }
        r.row_ptr.assign(p + m + 1, 0);
        for (int matrix = 1; matrix <= 2; ++matrix)
            for (Index row : r.original[matrix].idx) {
                Index global = row + (matrix == 2 ? p : 0);
                if (pivot_row[global]) ++r.row_ptr[global + 1];
            }
        for (Index row = 0; row < p + m; ++row) r.row_ptr[row + 1] += r.row_ptr[row];
        r.row_col.resize(r.row_ptr.back());
        r.row_data.resize(r.row_ptr.back());
        auto next = r.row_ptr;
        for (int matrix = 1; matrix <= 2; ++matrix) {
            const auto &a = r.original[matrix];
            for (Index col = 0; col < n; ++col)
                for (Index k = a.ptr[col]; k < a.ptr[col + 1]; ++k) {
                    Index row = a.idx[k] + (matrix == 2 ? p : 0);
                    if (!pivot_row[row]) continue;
                    Index dest = next[row]++;
                    r.row_col[dest] = col;
                    r.row_data[dest] = a.data[k];
                }
        }
    }

    double coefficient(Index row) const {
        double value;
        std::memcpy(&value, &coefficient_bits[row], sizeof(value));
        return value;
    }

    bool fail(int status, const std::string &message) {
        r.status = status;
        r.message = message;
        return false;
    }

    bool empty(Index row) {
        double original_rhs = row < p ? r.original_rhs[1][row] : r.original_rhs[2][row - p];
        double guard = options.infeasibility_guard * std::max(1., std::abs(original_rhs));
        if (row < p) {
            if (lo[row] > 0. || hi[row] < 0.) {
                if (std::min(std::abs(lo[row]), std::abs(hi[row])) <= guard)
                    return fail(GQP_INCONCLUSIVE, "Empty equality contradiction within guard: retain original problem");
                return fail(GQP_INFEASIBLE, "Inconsistent empty equality after substitution");
            }
            if (lo[row] != 0. || hi[row] != 0.)
                return fail(GQP_INCONCLUSIVE, "Uncertain empty equality: retain original problem");
        } else {
            if (hi[row] < 0.) {
                if (-hi[row] <= guard)
                    return fail(GQP_INCONCLUSIVE, "Empty inequality contradiction within guard: retain original problem");
                return fail(GQP_INFEASIBLE, "Inconsistent empty inequality after substitution");
            }
            if (lo[row] < 0.)
                return fail(GQP_INCONCLUSIVE, "Uncertain empty inequality: retain original problem");
        }
        removed[row] = 1;
        return true;
    }

    Interval substituted_interval(Index row, double coefficient, Interval range) const {
        auto left = product(coefficient, coefficient > 0. ? range.lo : range.hi);
        auto right = product(coefficient, coefficient > 0. ? range.hi : range.lo);
        return {subtract(lo[row], right.hi).lo, subtract(hi[row], left.lo).hi};
    }

    void defer(Index variable) {
        if (!deferred[variable]) ++r.stats[6];
        deferred[variable] = 1;
    }

    bool would_create_ambiguous_row(Index variable, Interval range, const Pivot &pivot) {
        for (int matrix = 1; matrix <= 2; ++matrix) {
            const auto &a = r.original[matrix];
            for (Index k = a.ptr[variable]; k < a.ptr[variable + 1]; ++k) {
                Index row = a.idx[k] + (matrix == 2 ? p : 0);
                if (removed[row] || count[row] != 1 || row == pivot.equality ||
                    (pivot.lower >= 0 && row == p + pivot.lower) ||
                    (pivot.upper >= 0 && row == p + pivot.upper)) continue;
                auto next = substituted_interval(row, a.data[k], range);
                double original_rhs = row < p ? r.original_rhs[1][row] : r.original_rhs[2][row - p];
                double guard = options.infeasibility_guard * std::max(1., std::abs(original_rhs));
                if (row < p) {
                    if (next.lo == 0. && next.hi == 0.) continue;
                    // A contradiction clearly outside the guard remains a
                    // global infeasibility certificate, not an ambiguity.
                    if ((next.lo > 0. || next.hi < 0.) &&
                        std::min(std::abs(next.lo), std::abs(next.hi)) > guard) continue;
                    if (options.local_fallback) ambiguous[row] = 1;
                    return true;
                }
                if (next.lo < 0. && !(next.hi < 0. && -next.hi > guard)) {
                    if (options.local_fallback) ambiguous[row] = 1;
                    return true;
                }
            }
        }
        return false;
    }

    void eliminate(Index variable, double value, Interval range, Pivot pivot) {
        if (!std::isfinite(value)) throw std::overflow_error("Nonfinite fixed variable");
        if (options.local_fallback &&
            (deferred[variable] || would_create_ambiguous_row(variable, range, pivot))) {
            defer(variable);
            return;
        }
        if (pivot.equality >= 0) removed[pivot.equality] = 1;
        if (pivot.lower >= 0) removed[p + pivot.lower] = 1;
        if (pivot.upper >= 0) removed[p + pivot.upper] = 1;
        fixed[variable] = 1;
        r.fixed_values[variable] = value;
        // Interior scalar minima only need primal scattering. Keeping their
        // row-free pivots would force unnecessary CSR construction and
        // stationarity products when no dual multipliers require recovery.
        if (pivot.equality >= 0 || pivot.lower >= 0 || pivot.upper >= 0)
            r.tape.push_back(pivot);
        for (int matrix = 1; matrix <= 2; ++matrix) {
            const auto &a = r.original[matrix];
            for (Index k = a.ptr[variable]; k < a.ptr[variable + 1]; ++k) {
                ++r.stats[4];
                Index row = a.idx[k] + (matrix == 2 ? p : 0);
                if (removed[row]) continue;
                double v = a.data[k];
                auto next = substituted_interval(row, v, range);
                lo[row] = next.lo;
                hi[row] = next.hi;
                // Same evaluation order on every run, without fast-math/FMA contraction.
                rhs[row] -= v * value;
                if (!std::isfinite(rhs[row])) throw std::overflow_error("Nonfinite reduced right-hand side");
                --count[row];
                single_col[row] ^= variable;
                coefficient_bits[row] ^= bits(v);
                if (count[row] <= 1) queue.push_back(row);
            }
        }
    }

    bool singleton_bound(Index row, Index variable, double value) {
        bool upper = value > 0.;
        auto &best = upper ? upper_row[variable] : lower_row[variable];
        Interval candidate = divide({lo[row], hi[row]}, value);
        if (best >= 0 && best != row && !removed[best]) {
            Interval previous = divide({lo[best], hi[best]}, singleton_coefficient[best]);
            bool duplicate = value == singleton_coefficient[best] &&
                lo[row] == hi[row] && lo[best] == hi[best] && rhs[row] == rhs[best];
            bool better = upper ? candidate.hi < previous.lo : candidate.lo > previous.hi;
            bool worse = upper ? candidate.lo > previous.hi : candidate.hi < previous.lo;
            if (options.redundant_bounds && (worse || duplicate)) {
                removed[row] = 1;
                return true;
            }
            if (better) {
                if (options.redundant_bounds) removed[best] = 1;
                best = row;
            }
        } else best = row;

        Index low = lower_row[variable], high = upper_row[variable];
        if (low < 0 || high < 0) return true;
        double a = singleton_coefficient[low], b = singleton_coefficient[high];
        auto low_range = divide({lo[low], hi[low]}, a);
        auto high_range = divide({lo[high], hi[high]}, b);
        if (low_range.lo > high_range.hi) {
            double guard = options.infeasibility_guard * std::max({1., std::abs(low_range.lo), std::abs(high_range.hi)});
            if (low_range.lo - high_range.hi <= guard) {
                if (options.local_fallback) {
                    defer(variable);
                    ambiguous[low] = ambiguous[high] = 1;
                    return true;
                }
                return fail(GQP_INCONCLUSIVE, "Singleton-bound contradiction within guard: retain original problem");
            }
            return fail(GQP_INFEASIBLE, "Inconsistent singleton bounds");
        }
        bool exactly_equal = low_range.lo == low_range.hi && high_range.lo == high_range.hi &&
            low_range.lo == high_range.lo;
        bool opposite_rows = a == -b && rhs[low] == -rhs[high] &&
            lo[low] == hi[low] && lo[high] == hi[high];
        if (options.fixed_bounds && (exactly_equal || opposite_rows)) {
            eliminate(variable, rhs[high] / b, high_range,
                      {variable, -1, low - p, high - p, 0., a, b});
        }
        return true;
    }

    bool independent_quadratic(Index variable) const {
        const auto &Q = r.original[0];
        Index first = Q.ptr[variable];
        // Restrict the first pass to originally diagonal columns. Coupling to
        // earlier fixed variables would need objective coefficient intervals.
        if (Q.ptr[variable + 1] != first + 1 || Q.idx[first] != variable || Q.data[first] <= 0.)
            return false;
        for (int matrix = 1; matrix <= 2; ++matrix) {
            const auto &a = r.original[matrix];
            for (Index k = a.ptr[variable]; k < a.ptr[variable + 1]; ++k) {
                Index row = a.idx[k] + (matrix == 2 ? p : 0);
                if (!removed[row] && (matrix == 1 || count[row] != 1)) return false;
            }
        }
        return true;
    }

    bool choose_scalar_minimum(Index variable, double &value, Interval &range, Pivot &pivot) {
        double diagonal = r.original[0].data[r.original[0].ptr[variable]];
        double linear = r.original_rhs[0][variable];
        range = quotient(-linear, diagonal);
        value = -linear / diagonal;
        Index low = lower_row[variable], high = upper_row[variable];
        Interval lower = low < 0 ? Interval{-inf, -inf} : divide({lo[low], hi[low]}, singleton_coefficient[low]);
        Interval upper = high < 0 ? Interval{inf, inf} : divide({lo[high], hi[high]}, singleton_coefficient[high]);
        if (range.hi < lower.lo) {
            range = lower;
            value = rhs[low] / singleton_coefficient[low];
            pivot.lower = low - p;
            pivot.lower_coefficient = singleton_coefficient[low];
        } else if (range.lo > upper.hi) {
            range = upper;
            value = rhs[high] / singleton_coefficient[high];
            pivot.upper = high - p;
            pivot.upper_coefficient = singleton_coefficient[high];
        } else if (range.lo < lower.hi || range.hi > upper.lo) {
            // Overlap cannot certify whether the bound or objective is active.
            return false;
        }
        // Check every other singleton bound before any row or tape mutation.
        return !would_create_ambiguous_row(variable, range, pivot);
    }

    void minimize_separable_quadratics() {
        for (Index variable = 0; variable < n; ++variable) {
            if (fixed[variable] || !independent_quadratic(variable)) continue;
            if (options.local_fallback && deferred[variable]) {
                ++r.stats[9];
                continue;
            }
            double value;
            Interval range;
            Pivot pivot{variable, -1, -1, -1, 0., 0., 0.};
            bool certified;
            try {
                certified = choose_scalar_minimum(variable, value, range, pivot);
            } catch (const std::overflow_error &) {
                // This optional pass has not modified anything yet. Retain an
                // unrepresentable scalar optimum and keep independent work.
                certified = false;
            }
            if (!certified) {
                ++r.stats[9];
                continue;
            }
            eliminate(variable, value, range, pivot);
            if (fixed[variable]) ++r.stats[8];
        }
    }

    Matrix reduce_matrix(const Matrix &source, const std::vector<Index> &row_map,
                         const std::vector<Index> &column_map) {
        Matrix out;
        out.rows = static_cast<Index>(row_map.size());
        out.cols = static_cast<Index>(column_map.size());
        out.ptr.assign(out.cols + 1, 0);
        std::vector<Index> destination(source.rows, -1);
        for (Index row = 0; row < out.rows; ++row) destination[row_map[row]] = row;
        out.idx.reserve(source.idx.size());
        out.data.reserve(source.data.size());
        for (Index col = 0; col < out.cols; ++col) {
            Index old = column_map[col];
            for (Index k = source.ptr[old]; k < source.ptr[old + 1]; ++k) {
                Index row = destination[source.idx[k]];
                if (row >= 0) {
                    out.idx.push_back(row);
                    out.data.push_back(source.data[k]);
                }
            }
            out.ptr[col + 1] = static_cast<Index>(out.idx.size());
        }
        return out;
    }

    void reconstruct_problem() {
        for (Index j = 0; j < n; ++j) if (!fixed[j]) r.maps[0].push_back(j);
        for (Index row = 0; row < p; ++row) if (!removed[row]) r.maps[1].push_back(row);
        for (Index row = 0; row < m; ++row) if (!removed[p + row]) r.maps[2].push_back(row);
        r.stats[0] = n - static_cast<Index>(r.maps[0].size());
        r.stats[1] = p - static_cast<Index>(r.maps[1].size());
        r.stats[2] = m - static_cast<Index>(r.maps[2].size());
        if (options.local_fallback)
            for (Index row = 0; row < p + m; ++row)
                if (ambiguous[row] && !removed[row]) ++r.stats[7];
        r.status = r.stats[0] || r.stats[1] || r.stats[2] ? GQP_REDUCED : GQP_UNCHANGED;
        r.message = r.status == GQP_REDUCED ? "Certified structural reductions" : "No structural reductions";
        if (r.stats[6]) {
            if (r.status == GQP_UNCHANGED) {
                r.status = GQP_INCONCLUSIVE;
                r.message = "All useful eliminations ambiguous: retain original problem";
                return;
            }
            r.message = "Certified structural reductions; ambiguous variables retained";
        }
        if (r.status == GQP_UNCHANGED) return;
        std::vector<double> gradient = r.original_rhs[0];
        const auto &Q = r.original[0];
        for (Index col = 0; col < n; ++col) if (fixed[col]) {
            double x = r.fixed_values[col];
            for (Index k = Q.ptr[col]; k < Q.ptr[col + 1]; ++k)
                gradient[Q.idx[k]] += Q.data[k] * x;
        }
        for (Index col = 0; col < n; ++col) if (fixed[col])
            r.offset += .5 * r.fixed_values[col] * (gradient[col] + r.original_rhs[0][col]);
        if (!std::isfinite(r.offset)) throw std::overflow_error("Nonfinite objective offset");
        for (Index col : r.maps[0]) {
            if (!std::isfinite(gradient[col])) throw std::overflow_error("Nonfinite reduced objective");
            r.rhs[0].push_back(gradient[col]);
        }
        for (int matrix = 1; matrix <= 2; ++matrix)
            for (Index row : r.maps[matrix]) r.rhs[matrix].push_back(rhs[row + (matrix == 2 ? p : 0)]);
        for (int matrix = 0; matrix < 3; ++matrix)
            r.reduced[matrix] = reduce_matrix(r.original[matrix], r.maps[matrix], r.maps[0]);
        build_recovery_rows();
    }

public:
    Presolver(Result &result, gqp_options flags) : r(result), options(flags),
        n(r.original[0].cols), p(r.original[1].rows), m(r.original[2].rows),
        count(p + m), single_col(p + m), lower_row(n, -1), upper_row(n, -1), coefficient_bits(p + m),
        singleton_coefficient(p + m), removed(p + m), fixed(n), processed(p + m),
        deferred(flags.local_fallback ? n : 0), ambiguous(flags.local_fallback ? p + m : 0) {
        rhs = r.original_rhs[1];
        rhs.insert(rhs.end(), r.original_rhs[2].begin(), r.original_rhs[2].end());
        lo = hi = rhs;
        r.fixed_values.assign(n, 0.);
        queue.reserve(p + m);
    }

    bool propagate() {
        for (size_t head = 0; head < queue.size(); ++head) {
            Index row = queue[head];
            if (removed[row]) continue;
            if (count[row] == 0) {
                if (!empty(row)) return false;
            } else if (!processed[row]) {
                processed[row] = 1;
                ++r.stats[3];
                Index variable = single_col[row];
                double a = coefficient(row);
                singleton_coefficient[row] = a;
                if (row < p && options.singleton_equalities) {
                    eliminate(variable, rhs[row] / a, divide({lo[row], hi[row]}, a),
                              {variable, row, -1, -1, a, 0., 0.});
                } else if (row >= p && !singleton_bound(row, variable, a)) return false;
            }
        }
        queue.clear();
        return true;
    }

    void run() {
        initialize_rows();
        if (!propagate()) return;
        if (options.separable_quadratics) {
            minimize_separable_quadratics();
            if (!propagate()) return;
        }
        reconstruct_problem();
    }
};

void add_transpose(const Matrix &a, const double *dual, std::vector<double> &gradient) {
    for (Index col = 0; col < a.cols; ++col)
        for (Index k = a.ptr[col]; k < a.ptr[col + 1]; ++k)
            gradient[col] += a.data[k] * dual[a.idx[k]];
}

bool finite(const double *values, Index count) {
    if (count && !values) return false;
    for (Index i = 0; i < count; ++i) if (!std::isfinite(values[i])) return false;
    return true;
}
} // namespace

extern "C" {
void *gqp_presolve(gqp_matrix Q, const double *q, gqp_matrix A,
                   const double *b, gqp_matrix G, const double *h, gqp_options options) {
    std::unique_ptr<Result> result;
    try {
        result = std::make_unique<Result>();
        if (!std::isfinite(options.infeasibility_guard) || options.infeasibility_guard < 0.)
            throw std::invalid_argument("Infeasibility guard must be finite and nonnegative");
        if (Q.rows != Q.cols || A.cols != Q.cols || G.cols != Q.cols)
            throw std::invalid_argument("Inconsistent problem dimensions");
        result->original = {copy_matrix(Q), copy_matrix(A), copy_matrix(G)};
        result->original_rhs = {copy_vector(q, Q.cols), copy_vector(b, A.rows), copy_vector(h, G.rows)};
        result->stats[5] = Q.nnz + A.nnz + G.nnz;
        // Encounter each transposed column in sorted order and compare against
        // its CSC cursor: O(nnz+n), without materializing the transpose.
        const auto &a = result->original[0];
        auto next = a.ptr;
        for (Index col = 0; col < a.cols; ++col)
            for (Index k = a.ptr[col]; k < a.ptr[col + 1]; ++k) {
                Index row = a.idx[k], match = next[row]++;
                if (match >= a.ptr[row + 1] || a.idx[match] != col || a.data[match] != a.data[k])
                    throw std::invalid_argument("Q must contain the full exactly symmetric Hessian");
            }
        Presolver(*result, options).run();
    } catch (const std::invalid_argument &error) {
        if (result) { result->status = GQP_INVALID; result->message = error.what(); }
    } catch (const std::overflow_error &error) {
        if (result) { result->status = GQP_INCONCLUSIVE; result->message = error.what(); }
    } catch (const std::exception &error) {
        if (result) { result->status = GQP_ERROR; result->message = error.what(); }
    } catch (...) {
        if (result) { result->status = GQP_ERROR; result->message = "Unknown native error"; }
    }
    return result.release();
}

void gqp_free(void *value) { delete static_cast<Result *>(value); }
int gqp_status_code(const void *value) { return value ? static_cast<const Result *>(value)->status : GQP_ERROR; }
const char *gqp_message(const void *value) {
    return value ? static_cast<const Result *>(value)->message.c_str() : "Allocation failure";
}
double gqp_objective_offset(const void *value) { return static_cast<const Result *>(value)->offset; }
gqp_matrix gqp_get_matrix(const void *value, int which) {
    if (!value || which < 0 || which > 2) return {0, 0, 0, nullptr, nullptr, nullptr};
    const auto &r = *static_cast<const Result *>(value);
    return (r.status == GQP_UNCHANGED ? r.original[which] : r.reduced[which]).view();
}
const double *gqp_get_vector(const void *value, int which) {
    if (!value || which < 0 || which > 2) return nullptr;
    const auto &r = *static_cast<const Result *>(value);
    return (r.status == GQP_UNCHANGED ? r.original_rhs[which] : r.rhs[which]).data();
}
const int64_t *gqp_get_map(const void *value, int which) {
    return !value || which < 0 || which > 2 ? nullptr : static_cast<const Result *>(value)->maps[which].data();
}
int64_t gqp_get_stat(const void *value, int which) {
    return !value || which < 0 || which > 9 ? -1 : static_cast<const Result *>(value)->stats[which];
}

int gqp_recover(const void *value, const double *x, const double *y, const double *z,
                double *original_x, double *original_y, double *original_z) {
    try {
        if (!value) return 1;
        const auto &r = *static_cast<const Result *>(value);
        if (r.status != GQP_REDUCED && r.status != GQP_UNCHANGED) return 1;
        Index n = r.original[0].cols, p = r.original[1].rows, m = r.original[2].rows;
        if (!finite(x, r.maps[0].size()) || !finite(y, r.maps[1].size()) ||
            !finite(z, r.maps[2].size()) || (n && !original_x) ||
            (p && !original_y) || (m && !original_z)) return 1;
        if (r.status == GQP_UNCHANGED) {
            if (n) std::copy(x, x + n, original_x);
            if (p) std::copy(y, y + p, original_y);
            if (m) std::copy(z, z + m, original_z);
            return 0;
        }
        std::copy(r.fixed_values.begin(), r.fixed_values.end(), original_x);
        if (p) std::fill(original_y, original_y + p, 0.);
        if (m) std::fill(original_z, original_z + m, 0.);
        for (size_t j = 0; j < r.maps[0].size(); ++j) original_x[r.maps[0][j]] = x[j];
        for (size_t j = 0; j < r.maps[1].size(); ++j) original_y[r.maps[1][j]] = y[j];
        for (size_t j = 0; j < r.maps[2].size(); ++j) original_z[r.maps[2][j]] = z[j];
        // Removing rows alone never needs a stationarity correction. In that
        // common case the scatter above is the complete inverse operation.
        if (r.tape.empty()) return 0;
        std::vector<double> gradient = r.original_rhs[0];
        const auto &Q = r.original[0];
        for (Index col = 0; col < n; ++col)
            for (Index k = Q.ptr[col]; k < Q.ptr[col + 1]; ++k)
                gradient[Q.idx[k]] += Q.data[k] * original_x[col];
        add_transpose(r.original[1], original_y, gradient);
        add_transpose(r.original[2], original_z, gradient);
        for (auto entry = r.tape.rbegin(); entry != r.tape.rend(); ++entry) {
            Index row;
            double dual;
            if (entry->equality >= 0) {
                row = entry->equality;
                dual = -gradient[entry->variable] / entry->equality_coefficient;
                original_y[row] = dual;
            } else if (entry->lower >= 0 || entry->upper >= 0) {
                bool use_upper = entry->lower < 0 || (entry->upper >= 0 && gradient[entry->variable] < 0.);
                Index local_row = use_upper ? entry->upper : entry->lower;
                double a = use_upper ? entry->upper_coefficient : entry->lower_coefficient;
                dual = -gradient[entry->variable] / a;
                if (dual < 0.) return 2;
                row = p + local_row;
                original_z[local_row] = dual;
            } else continue; // An interior scalar optimum needs no multiplier.
            for (Index k = r.row_ptr[row]; k < r.row_ptr[row + 1]; ++k)
                gradient[r.row_col[k]] += r.row_data[k] * dual;
        }
        return finite(original_x, n) && finite(original_y, p) && finite(original_z, m) ? 0 : 2;
    } catch (...) { return 3; }
}
} // extern C
