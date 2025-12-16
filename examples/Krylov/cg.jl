using LinearAlgebra # norm
using SparseArrays  # spdiagm
using SparseMatricesCSR

const T = Float64 # (Float64, ComplexF64)

# Symmetric and positive definite systems.
function symmetric_definite(n::Int, T)
    α = T <: Complex ? T(im) : one(T)
    A_csc = spdiagm(-1 => α * ones(T, n-1), 0 => 4 * ones(T, n), 1 => conj(α) * ones(T, n-1))
    y = A_csc * T[1:n;]
    return A_csc, y
end

include("./overrides.jl")                       # TODO

# Run CG
cg_tol = 1.0e-6
n=64
ts=32
A, y = symmetric_definite(n, T)
A = SparseMatrixCSR(A)

XKBlas.set_tile_parameter(ts)                   # TODO
@time begin
    (x, stats) = Krylov.cg(A, y, itmax=5*n)
    XKBlas.memory_coherent_sync(x)              # TODO
end
r = y - A * x
println("residual is $(r)")
resid = norm(r) / norm(y)
@assert resid ≤ cg_tol "Failure"
println("Success")
