using Krylov

using LinearAlgebra # norm
using SparseArrays  # spdiagm
using SparseMatricesCSR

# TODO: one extra line with XKBlas.jl to have it run on GPU
include("../../src/KrylovBackend/synchronous.jl")

const T = Float64 # (Float64, ComplexF64)

# Symmetric and positive definite systems.
function symmetric_definite(n::Int)
    α = T <: Complex ? T(im) : one(T)
    A_csc = spdiagm(-1 => α * ones(T, n-1), 0 => 4 * ones(T, n), 1 => conj(α) * ones(T, n-1))
    y = A_csc * T[1:n;]
    return A_csc, y
end

n=4
A, x = symmetric_definite(n)
y = Vector{Int}(undef, n)
A = SparseMatrixCSR(A)
Krylov.kmul!(y, A, x)
display(LinearAlgebra.Matrix(A))
display(x)
display(y)
