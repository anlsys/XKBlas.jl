using LinearAlgebra, Random, XKLas
const XK = XKLas

n = 16
x = [Float32(0.0) for _ in 1:n]
value = Float32(42.0)

XK.BLAS.ext.fill(n, x, value)
XK.sync()

println(x)
