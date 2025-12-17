using XKBlas

n=4
ts=2

const T = Float64
x = rand(T, n)
y = Vector{T}(undef, n)

XKBlas.set_tile_parameter(ts)
XKBlas.copy(n, x, 1, y, 1)

println(x)
println(y)
