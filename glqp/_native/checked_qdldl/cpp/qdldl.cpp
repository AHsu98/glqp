#include "qdldl.hpp"
#include <memory>

using namespace glqp_checked_qdldl;



Solver::Solver(QDLDL_int n, QDLDL_int * Ap, QDLDL_int *Ai, QDLDL_float * Ax){
    // Construction can fail after allocating work arrays. Release them even
    // though the destructor is not called for an incomplete C++ object.
    try {

	// Dimension
	nx = n;
	nnz = Ap[nx];

	// Elimination tree
	etree  = new QDLDL_int[n];
	Lnz = new QDLDL_int[n];

	// L factors
	Lp = new QDLDL_int[n + 1];

	// D
	D = new QDLDL_float[n];
	Dinv = new QDLDL_float[n];

	// Workspace
	iwork = new QDLDL_int[3 * n];
	bwork = new QDLDL_bool[n];
	fwork = new QDLDL_float[n];

	// Permutation
	P = new QDLDL_int[n];
	Pinv = new QDLDL_int[n];

	// Permutation
	QDLDL_int amd_status = amd_l_order(nx, Ap, Ai, P, NULL, NULL);
	if (amd_status < 0)
		throw std::runtime_error(std::string("Error in AMD computation ") + std::to_string(amd_status));

	pinv(P, Pinv, n); // Compute inverse permutation

	// Allocate elements of A permuted
	Aperm_p = new QDLDL_int[n+1];
	Aperm_i = new QDLDL_int[nnz];
	Aperm_x = new QDLDL_float[nnz];
	A2Aperm = new QDLDL_int[nnz];
	std::unique_ptr<QDLDL_int[]> work_perm(new QDLDL_int[n]());

	// Permute A
	symperm(n, Ap, Ai, Ax, Aperm_p, Aperm_i, Aperm_x, Pinv, A2Aperm, work_perm.get());

	// Compute elimination tree
    sum_Lnz = QDLDL_etree(n, Aperm_p, Aperm_i, iwork, Lnz, etree);

	if (sum_Lnz < 0)
		throw std::runtime_error(std::string("Error in computing elimination tree. Matrix not properly upper-triangular, sum_Lnz = ") + std::to_string(sum_Lnz));

	// Allocate factor
	Li = new QDLDL_int[sum_Lnz];
	Lx = new QDLDL_float[sum_Lnz];


	// Compute numeric factorization
	int factor_status = QDLDL_factor(nx, Aperm_p, Aperm_i, Aperm_x,
			Lp, Li, Lx,
			D, Dinv, Lnz,
			etree, bwork, iwork, fwork);

	if (factor_status < 0){
		throw std::runtime_error(std::string("Error in matric factorization. Input matrix is not quasi-definite, factor_status = ") + std::to_string(factor_status));
	}


    valid = true;
    } catch (...) {
        release();
        throw;
    }

}


QDLDL_float * Solver::get_D(){
    return D;
}

QDLDL_int * Solver::get_P(){
    return P;
}

QDLDL_int * Solver::get_Lp(){
    return Lp;
}

QDLDL_int * Solver::get_Li(){
    return Li;
}

QDLDL_float * Solver::get_Lx(){
    return Lx;
}

QDLDL_float * Solver::solve(QDLDL_float * b){
    require_valid();

	auto * x = new QDLDL_float[nx];
	auto work = new QDLDL_float[nx];

    permute_x(nx, work, b, P);
    QDLDL_solve(nx, Lp, Li, Lx, Dinv, work);
    permutet_x(nx, x, work, P);

	delete [] work;
	return x;

}



void Solver::update(QDLDL_float * Anew_x){
    // A failed factorization leaves partial factors. Never solve with them.
    valid = false;

	// Update matrix
	update_A(nnz, Aperm_x, Anew_x, A2Aperm);

	// Compute numeric factorization
    const int factor_status = QDLDL_factor(nx, Aperm_p, Aperm_i, Aperm_x,
			     Lp, Li, Lx,
				 D, Dinv, Lnz,
				 etree, bwork, iwork, fwork);
    if (factor_status < 0) {
        throw std::runtime_error("QDLDL update failed, factor_status = " + std::to_string(factor_status));
    }
    valid = true;

}

void Solver::require_valid() const {
    if (!valid) {
        throw std::runtime_error("QDLDL factor is invalid after a failed numeric update");
    }
}

Solver::~Solver() { release(); }

void Solver::release() noexcept {

	delete [] Lp;
	delete [] Li;
	delete [] Lx;
	delete [] D;
	delete [] Dinv;
	delete [] P;
	delete [] Pinv;
	delete [] etree;
	delete [] Lnz;
	delete [] iwork;
	delete [] bwork;
	delete [] fwork;
	delete [] Aperm_p;
	delete [] Aperm_i;
	delete [] Aperm_x;
	delete [] A2Aperm;

}
