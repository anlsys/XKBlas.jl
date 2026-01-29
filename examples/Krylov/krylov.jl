# ----------------------------
# Command-line arguments
# ----------------------------
# Usage:
#   julia script.jl [fname] [n] [ts] [use_xkblas]
#
# Defaults:
#   fname      = cg
#   n          = 4
#   ts         = 2
#   use_xkblas = false

fname      = length(ARGS) >= 1 ?               ARGS[1]  : "cg"
n          = length(ARGS) >= 2 ? parse(Int,    ARGS[2]) : 4
ts         = length(ARGS) >= 3 ? parse(Int,    ARGS[3]) : 2
use_xkblas = length(ARGS) >= 4 ? parse(Bool,   ARGS[4]) : false

using LinearAlgebra # norm
using SparseArrays  # spdiagm
using SparseMatricesCSR
using Krylov
using XK

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

tolerance = 1.0e-6
A, y = symmetric_definite(n, T)
A = SparseMatrixCSR(A)
f = getproperty(Krylov, Symbol(fname))

# If using XKBlas
if use_xkblas
    # redefine krylov_utils.jl API with XKVector/XKMatrix
    include("./overrides.jl")

    # use them types to dispatch on XK.XBLAS routines
    A = XKSparseMatrixCSR(A)
    y = XKVector(y)

    # set tile parameter
    XK.set_tile_parameter(ts)
end

println("Running fname=$(fname), with n=$(n) of tile size ts=$(ts) $(use_xkblas ? "" : "not") using XKBLAS")

# Run
@time begin
    (x, stats) = f(A, y, itmax = 5*n)
end

# Write back
if use_xkblas
    XK.memory_coherent_sync(x)
end

print(stats)
println(typeof(x))
println(typeof(y))
println(typeof(A))

# Checking result
r = y - A * x
println("       x is $(x)")
println("       y is $(y)")
println("residual is $(r)")

resid = norm(r)
@assert resid ≤ tolerance "Failure"
println("Success")
