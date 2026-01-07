using LinearAlgebra, Random, XK

# ----------------------------
# Command-line arguments
# ----------------------------
# Usage:
#   julia script.jl [n]
#
# Defaults:
#   n  = 4

const T = Float64

# Create empty vectors
n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
alpha = T(0.2)

# Set tile parameter
ngpus = XK.get_ngpus()
ts = div(n, ngpus)
XK.set_tile_parameter(ts)

# Initialize memory
x = Vector{T}(undef, n)
y = Vector{T}(undef, n)
XK.BLAS.ext.fill(n, x, 1.0)
XK.BLAS.ext.fill(n, y, 0.5)
XK.BLAS.axpy(n, alpha, x, 1, y, 1)
println(y)
