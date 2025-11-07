using LinearAlgebra, Random
using XKBlas

n = 16
x = [Float32(0.0) for _ in 1:n]
value = Float32(42.0)

XKBlas.init()
XKBlas.fill(n, x, value)
XKBlas.sync()

XKBlas.deinit()

println(x)
