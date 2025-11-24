# XKBlas.jl
Julia binding for XKBlas: a portable multi-GPU BLAS library with support host memory coherence

See https://gitlab.inria.fr/xkblas/dev/tree/v2.0 for XKBlas details

See `deps/README.md` for developing

See `examples/kernel-abstractions.jl` for an example

See `examples/krylov-cg.jl` for using with Krylov.jl

# TODO
- Fix deadlocks occuring in the Julia runtime when compiling from a foreign thread -- or find a work-around
