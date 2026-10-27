// Based on qdldl-python 0.1.9.post1; see upstream.json and LICENSE.
// The numeric QDLDL/AMD kernels are unchanged. This private binding validates
// ownership and propagates native failures; it never changes solve arithmetic.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include "qdldl.hpp"

namespace py = pybind11;
using namespace py::literals;
using FloatArray = py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast>;

struct UpperMatrix {
    QDLDL_int n;
    IndexArray pointers, indices;
    FloatArray values;
};

static UpperMatrix checked_upper(py::object matrix, bool upper) {
    auto sparse = py::module::import("scipy.sparse");
    if (!py::cast<bool>(sparse.attr("isspmatrix_csc")(matrix))) {
        matrix = sparse.attr("csc_matrix")(matrix);
    }
    auto shape = matrix.attr("shape").cast<py::tuple>();
    auto n = shape[0].cast<QDLDL_int>();
    if (n <= 0 || n != shape[1].cast<QDLDL_int>()) {
        throw py::value_error("QDLDL matrix must be nonempty and square");
    }
    matrix.attr("check_format")("full_check"_a=true);
    if (!upper) matrix = sparse.attr("triu")(matrix, "format"_a="csc");
    UpperMatrix result{n, matrix.attr("indptr").cast<IndexArray>(),
        matrix.attr("indices").cast<IndexArray>(), matrix.attr("data").cast<FloatArray>()};
    const auto* Ap = result.pointers.data();
    const auto* Ai = result.indices.data();
    const auto* Ax = result.values.data();
    auto nnz = result.values.size();
    if (result.pointers.ndim() != 1 || result.pointers.size() != n+1 ||
        result.indices.ndim() != 1 || result.indices.size() != nnz ||
        result.values.ndim() != 1 || !nnz || Ap[0] != 0 || Ap[n] != nnz) {
        throw py::value_error("Invalid QDLDL CSC storage lengths");
    }
    for (QDLDL_int column=0; column<n; ++column) {
        if (Ap[column] < 0 || Ap[column] > Ap[column+1] || Ap[column+1] > nnz) {
            throw py::value_error("Invalid QDLDL CSC column pointers");
        }
        QDLDL_int previous = -1;
        for (QDLDL_int position=Ap[column]; position<Ap[column+1]; ++position) {
            if (Ai[position] <= previous || Ai[position] > column) {
                throw py::value_error("QDLDL requires canonical upper-triangular CSC storage");
            }
            if (!std::isfinite(Ax[position])) throw py::value_error("QDLDL matrix must be finite");
            previous = Ai[position];
        }
    }
    return result;
}

class PySolver {
    std::unique_ptr<glqp_checked_qdldl::Solver> solver;
    std::vector<QDLDL_int> pointers, indices;
public:
    PySolver(py::object matrix, bool upper) {
        auto input = checked_upper(matrix, upper);
        pointers.assign(input.pointers.data(), input.pointers.data()+input.pointers.size());
        indices.assign(input.indices.data(), input.indices.data()+input.indices.size());
        py::gil_scoped_release release;
        solver.reset(new glqp_checked_qdldl::Solver(input.n,
            const_cast<QDLDL_int*>(input.pointers.data()), const_cast<QDLDL_int*>(input.indices.data()),
            const_cast<QDLDL_float*>(input.values.data())));
    }

    void update(py::object matrix, bool upper) {
        // Public binding checks the complete supplied pattern. GLQP's adapter
        // owns fixed CSC storage and uses _update_values after matches().
        auto input = checked_upper(matrix, upper);
        if (input.n != solver->nx || input.pointers.size() != static_cast<py::ssize_t>(pointers.size()) ||
            input.indices.size() != static_cast<py::ssize_t>(indices.size()) ||
            !std::equal(pointers.begin(), pointers.end(), input.pointers.data()) ||
            !std::equal(indices.begin(), indices.end(), input.indices.data())) {
            throw py::value_error("QDLDL update requires the original shape and sparse pattern");
        }
        update_values(input.values);
    }

    void update_values(FloatArray values) {
        // Private adapter seam: callers own the fixed pattern and have already
        // checked finite coefficients. Do not rescan or export its factors.
        if (values.ndim() != 1 || values.size() != solver->nnz) {
            throw py::value_error("QDLDL update values must match the original storage length");
        }
        py::gil_scoped_release release;
        solver->update(const_cast<QDLDL_float*>(values.data()));
    }

    py::array solve(FloatArray rhs) {
        if (rhs.ndim() != 1 || rhs.size() != solver->nx) {
            throw py::value_error("QDLDL RHS must be a vector matching the matrix size");
        }
        std::unique_ptr<QDLDL_float[]> solution;
        {
            py::gil_scoped_release release;
            solution.reset(solver->solve(const_cast<QDLDL_float*>(rhs.data())));
        }
        // Preserve the upstream copy into an owning NumPy array. Finite-result
        // verification remains in glqp.linear, including GMRES corrections.
        return py::array(solver->nx, solution.get());
    }

    py::tuple factors() {
        solver->require_valid();
        auto sparse = py::module::import("scipy.sparse");
        auto n = solver->nx;
        auto count = solver->sum_Lnz;
        py::array Lp(n+1, solver->get_Lp()), Li(count, solver->get_Li()), Lx(count, solver->get_Lx());
        auto L = sparse.attr("csc_matrix")(py::make_tuple(Lx,Li,Lp), py::make_tuple(n,n));
        py::array D(n,solver->get_D()), P(n,solver->get_P());
        return py::make_tuple(L,D,P);
    }
};

PYBIND11_MODULE(_glqp_qdldl_checked, module) {
    module.doc() = "GLQP private checked QDLDL 0.1.9.post1 binding";
    py::class_<PySolver>(module,"Solver",py::module_local())
        .def(py::init<py::object,bool>(),py::arg("A"),py::arg("upper")=false)
        .def("update",&PySolver::update,py::arg("Anew"),py::arg("upper")=false)
        .def("_update_values",&PySolver::update_values)
        .def("solve",&PySolver::solve)
        .def("factors",&PySolver::factors);
}
