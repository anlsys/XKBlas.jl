using LinearAlgebra, Random, XKLas
const XK = XKLas

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
XK.BLAS.ext.fill(n, y, 1.0)

# Pin memory
XK.memory_register(x)
XK.memory_register(y)

for i in 1:16
    @time begin
        XK.BLAS.axpy(n, alpha, x, 1, y, 1)
    end
    XK.memory_invalidate_caches()
end
