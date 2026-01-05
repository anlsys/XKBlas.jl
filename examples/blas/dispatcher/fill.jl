using LinearAlgebra, Random
using XKLas

n = 16
x = [Float32(0.0) for _ in 1:n]
value = Float32(42.0)

XKLas.fill(n, x, value)
XKLas.sync()

println(x)
