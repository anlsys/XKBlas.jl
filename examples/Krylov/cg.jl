using LinearAlgebra # norm
using SparseArrays  # spdiagm
using SparseMatricesCSR

const T = Float64 # (Float64, ComplexF64)

# Symmetric and positive definite systems.
function symmetric_definite(n::Int, T)
    α = T <: Complex ? T(im) : one(T)
    A_csc = spdiagm(
        -1 => α * ones(T, n-1),
         0 => 4 * ones(T, n),
         1 => conj(α) * ones(T, n-1)
    )
    y = A_csc * T[1:n;]
    return A_csc, y
end

include("./overrides.jl")                       # TODO

# ----------------------------
# Command-line arguments
# ----------------------------
# Usage:
#   julia script.jl [n] [ts]
#
# Defaults:
#   n  = 4
#   ts = 2
n  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
ts = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2

# Run CG
cg_tol = 1.0e-6

A, y = symmetric_definite(n, T)
A = SparseMatrixCSR(A)

XKBlas.set_tile_parameter(ts)                   # TODO
@time begin
    (x, stats) = Krylov.cg(A, y, itmax = 5*n)
    XKBlas.memory_coherent_sync(x)              # TODO
end

r = y - A * x
println("       x is $(x)")
println("       y is $(y)")
println("residual is $(r)")

resid = norm(r) / norm(y)
@assert resid ≤ cg_tol "Failure"
println("Success")
