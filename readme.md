## Quadratic Probrams + GLM
This package solves optimization problems of the form

minimize f(Ax) + (1/2) x^T Qx -x^T b
subject to Ex=e, Cx<=c

using an interior point method. 

We form the hessian of f(Ax) explicitly, using sparse_dot_mkl to compute A^T D A
efficiently. 

All matrices we use should be specified as csc_array, and we internally attempt to 
convert them to csc_array. 

#### Requirements
The main additional requirement comes from [sparse_dot_mkl](https://github.com/flatironinstitute/sparse_dot), 
needing the associated MKL dependencies.
These are packaged with Conda, and can be installed into Miniconda with `conda install -c intel mkl`.

In a later update, these may be made optional, but it sepeds up formation of the hessian significantly.

See https://github.com/flatironinstitute/sparse_dot for more details.

We solve KKT systems using [qdldl](https://github.com/osqp/qdldl-python)[[1]](#1).

## References
<a id="1">[1]</a> 
Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). OSQP: an operator splitting solver for quadratic programs. Mathematical Programming Computation, 12(4), 637–672. doi:10.1007/s12532-020-00179-2