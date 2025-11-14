using Base.Libc
using KernelAbstractions
using XKBlas
const XK = XKBlas

##################
# Kernel example #
##################

# Declare kernels, and its memory accesses
@kernel unsafe_indices=true function vector_add(a, b, c)
    i = @index(Local, Linear)
    @inbounds c[i] = a[i] + b[i]
end

const T = Float64

# A kernelabstraction task format
const vector_add_format = XK.KA.Format(

    # the kernelabstraction's kernel
    vector_add,

    # the grid launch configuration
    (a, b, c) -> (length(a), 1, 1),

    # the amount of shared memory
    (a, b, c) -> 0,

    # the accessed memory
    (a::AbstractVector{T}) -> XK.Access(XK.ACCESS_MODE_R, a),
    (b::AbstractVector{T}) -> XK.Access(XK.ACCESS_MODE_R, b),
    (c::AbstractVector{T}) -> XK.Access(XK.ACCESS_MODE_W, c)
)

#####################
# Execution example #
#####################

n = 4
a = rand(T, n)
b = rand(T, n)
c = Vector{T}(undef, n)

# Spawn a task that executes the kernel, and a task that reads back onto the host
XK.KA.device_async(vector_add_format, a, b, c)
XK.memory_coherent_async(c)
XK.sync()

# Check
apb = a + b
println("  a = $a")
println("  b = $b")
println("a+b = $apb")
println("  c = $c")
@assert isapprox(apb, c; atol=1e-6)
