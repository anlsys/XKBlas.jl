using XKLas

n=4
ts=2

const T = Float64
x = rand(T, n)
y = Vector{T}(undef, n)

XKLas.set_tile_parameter(ts)
XKLas.copy(n, x, 1, y, 1)

println(x)
println(y)
