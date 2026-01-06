using XKLas
const XK = XKLas

n=4
ts=2

const T = Float64
x = rand(T, n)
y = Vector{T}(undef, n)

XK.BLAS.set_tile_parameter(ts)
XK.BLAS.ext.copy(n, x, 1, y, 1)

println(x)
println(y)
