using Krylov

using LinearAlgebra # norm
using SparseArrays  # spdiagm
using SparseMatricesCSR

# Symmetric and positive definite systems.
function symmetric_definite(n::Int, FC)
    α = FC <: Complex ? FC(im) : one(FC)
    A_csc = spdiagm(-1 => α * ones(FC, n-1), 0 => 4 * ones(FC, n), 1 => conj(α) * ones(FC, n-1))
    b = A_csc * FC[1:n;]
    A_csr = SparseMatrixCSR(A_csc)
    return A_csr, b
end

# Run CG
cg_tol = 1.0e-6
n = 128
FC=Float64 # (Float64, ComplexF64)

# TODO: one extra line with XKBlas.jl to have it run on GPU
include("../../src/KrylovBackend/synchronous.jl")

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
