// Minimal independent file interface; see docs/presolve.md for GQP1 format.
#include "presolve.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using Index = int64_t;
struct InputMatrix {
    Index rows, cols;
    std::vector<Index> ptr, idx;
    std::vector<double> data;
    gqp_matrix view() const {
        return {rows, cols, static_cast<Index>(data.size()), ptr.data(), idx.data(), data.data()};
    }
};

std::vector<double> read_vector(std::istream &input, Index size) {
    std::vector<double> vector(size);
    for (auto &value : vector) if (!(input >> value)) throw std::runtime_error("Invalid vector");
    return vector;
}

InputMatrix read_matrix(std::istream &input, Index rows, Index cols) {
    Index nnz;
    if (!(input >> nnz) || nnz < 0) throw std::runtime_error("Invalid matrix size");
    std::vector<std::tuple<Index, Index, double>> entries;
    entries.reserve(nnz);
    for (Index k = 0; k < nnz; ++k) {
        Index row, col;
        double value;
        if (!(input >> col >> row >> value) || col < 0 || col >= cols || row < 0 || row >= rows)
            throw std::runtime_error("Invalid coordinate entry");
        entries.emplace_back(col, row, value);
    }
    std::sort(entries.begin(), entries.end());
    InputMatrix result{rows, cols, std::vector<Index>(cols + 1), {}, {}};
    result.idx.reserve(nnz);
    result.data.reserve(nnz);
    for (auto [col, row, value] : entries) {
        ++result.ptr[col + 1];
        result.idx.push_back(row);
        result.data.push_back(value);
    }
    for (Index col = 0; col < cols; ++col) result.ptr[col + 1] += result.ptr[col];
    return result;
}

void write_vector(const double *values, Index size) {
    for (Index k = 0; k < size; ++k) std::cout << values[k] << (k + 1 == size ? '\n' : ' ');
    if (!size) std::cout << '\n';
}

void write_matrix(gqp_matrix matrix) {
    std::cout << matrix.nnz << '\n';
    for (Index col = 0; col < matrix.cols; ++col)
        for (Index k = matrix.indptr[col]; k < matrix.indptr[col + 1]; ++k)
            std::cout << col << ' ' << matrix.indices[k] << ' ' << matrix.data[k] << '\n';
}
} // namespace

int main(int argc, char **argv) {
    try {
        gqp_options options{1, 1, 1, 1e-12, 0, 0};
        std::string filename;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--separable-quadratics") options.separable_quadratics = 1;
            else if (arg == "--local-fallback") options.local_fallback = 1;
            else if (arg == "--no-singleton-equalities") options.singleton_equalities = 0;
            else if (arg == "--no-fixed-bounds") options.fixed_bounds = 0;
            else if (arg == "--no-redundant-bounds") options.redundant_bounds = 0;
            else if (arg == "--infeasibility-guard" && i + 1 < argc) {
                std::string value = argv[++i];
                size_t used = 0;
                options.infeasibility_guard = std::stod(value, &used);
                if (used != value.size()) throw std::runtime_error("Invalid infeasibility guard");
            } else if (arg == "--help") {
                std::cout << "Usage: glqp-presolve [options] [problem.gqp|-] > reduced.gqp\n"
                          << "  --separable-quadratics  --local-fallback\n"
                          << "  --no-singleton-equalities  --no-fixed-bounds  --no-redundant-bounds\n"
                          << "  --infeasibility-guard NUMBER\n";
                return 0;
            } else if (arg.rfind("--", 0) == 0 || !filename.empty()) {
                throw std::runtime_error("Unknown option or extra input; see --help");
            } else filename = arg;
        }
        std::ifstream file;
        bool from_file = !filename.empty() && filename != "-";
        if (from_file) {
            file.open(filename);
            if (!file) throw std::runtime_error("Cannot open input");
        }
        std::istream &input = from_file ? file : std::cin;
        std::string version;
        Index n, p, m;
        double offset;
        if (!(input >> version >> n >> p >> m >> offset) || version != "GQP1" || n < 0 || p < 0 || m < 0)
            throw std::runtime_error("Invalid GQP1 header");
        auto q = read_vector(input, n), b = read_vector(input, p), h = read_vector(input, m);
        auto Q = read_matrix(input, n, n), A = read_matrix(input, p, n), G = read_matrix(input, m, n);
        std::unique_ptr<void, decltype(&gqp_free)> result(
            gqp_presolve(Q.view(), q.data(), A.view(), b.data(), G.view(), h.data(), options), gqp_free);
        int status = gqp_status_code(result.get());
        std::cerr << gqp_message(result.get()) << '\n';
        if (status > GQP_REDUCED) return status;
        auto reduced_Q = gqp_get_matrix(result.get(), 0);
        auto reduced_A = gqp_get_matrix(result.get(), 1);
        auto reduced_G = gqp_get_matrix(result.get(), 2);
        std::cout << std::setprecision(17) << "GQP1\n" << reduced_Q.cols << ' ' << reduced_A.rows
                  << ' ' << reduced_G.rows << ' ' << offset + gqp_objective_offset(result.get()) << '\n';
        write_vector(gqp_get_vector(result.get(), 0), reduced_Q.cols);
        write_vector(gqp_get_vector(result.get(), 1), reduced_A.rows);
        write_vector(gqp_get_vector(result.get(), 2), reduced_G.rows);
        write_matrix(reduced_Q);
        write_matrix(reduced_A);
        write_matrix(reduced_G);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 4;
    }
}
