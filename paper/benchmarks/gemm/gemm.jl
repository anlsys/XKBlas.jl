# ----------------------------
# Command-line arguments
# ----------------------------
# Usage:
#   julia script.jl [n]
#
# Defaults:
#   n  = 4

using LinearAlgebra, Random, XK
using Base: Matrix

# Run parameters
const T = Float32
n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
m = k = n
ts = 2_048
lda, ldb, ldc = m, k, m
alpha = T(1.0)
beta = T(0.0)
transA = transB = XK.BLAS.NO_TRANS

# Create host matrices
A = rand(m, k) # Matrix{T}(undef, m, k)
B = rand(k, n) # Matrix{T}(undef, k, n)
C = rand(m, n) # Matrix{T}(undef, m, n)

# Execution on all available devices - 3 flavors:
XK.BLAS.memory_register(A)
XK.BLAS.memory_register(B)
XK.BLAS.memory_register(C)

println(A)
println(B)
println(alpha*A*B+beta*C)

# (1) Blocking call, and automatically write-back
XK.BLAS.set_tile_size(ts)
XK.BLAS.gemm(
    transA, transB,
    alpha, A, B,
    beta,  C
)

println(C)
