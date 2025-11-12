# launch using
# ```
#   export LBT_FORCE_F2C=plain
#   export LBT_FORCE_RETSTYLE=normal
#   export LBT_FORCE_INTERFACE=ilp64
#   julia examples/krylov-cg.jl
# ```

using Krylov

using LinearAlgebra # norm
using SparseArrays  # spdiagm

# Symmetric and positive definite systems.
function symmetric_definite(n::Int, FC)
  α = FC <: Complex ? FC(im) : one(FC)
  A = spdiagm(-1 => α * ones(FC, n-1), 0 => 4 * ones(FC, n), 1 => conj(α) * ones(FC, n-1))
  b = A * FC[1:n;]
  return A, b
end

# Run CG
cg_tol = 1.0e-6
n = 128
FC=Float64 # (Float64, ComplexF64)

# TODO: one extra line with XKBlas.jl to have it run on GPU
include("../src/krylov-synchronous.jl")

for i in 1:5
    XKBlas.memory_invalidate_caches()
    A, b = symmetric_definite(n, FC)
    @time begin
        (x, stats) = cg(A, b, itmax=5*n)
    end
    r = b - A * x
    resid = norm(r) / norm(b)
    @assert resid ≤ cg_tol "Failure"
    println("Success")
end

# TODO: extra line with CUDA.jl
# using CUDA
#
# # run once so julia compiles
# for i in 1:5
#     A, b = symmetric_definite(n, FC)
#     @time begin
#         A_gpu = CuMatrix(A)
#         b_gpu = CuVector(b)
#         (x, stats) = cg(A_gpu, b_gpu, itmax=5*n)
#     end
#     r = b_gpu - A_gpu * x
#     resid = norm(r) / norm(b)
#     @assert resid ≤ cg_tol "Failure"
#     println("Success")
# end
