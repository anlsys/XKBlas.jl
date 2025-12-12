using Base.Libc
using KernelAbstractions
using XKBlas
const XK = XKBlas

#########################
# Custom Kernel example #
#########################

@XK.KA.kernel function vector_add(a, b, c, n)
    i = @XK.KA.tid
    if i <= n
        c[i] = a[i] + b[i]
    end
end

const T = Float64

# A kernelabstraction task format
const vector_add_format = XK.KA.Format(

    # the XK.KA kernel
    vector_add,

    # the kernel parameters
    (a::AbstractVector{T}) -> XK.Access(XK.ACCESS_MODE_R, a),
    (b::AbstractVector{T}) -> XK.Access(XK.ACCESS_MODE_R, b),
    (c::AbstractVector{T}) -> XK.Access(XK.ACCESS_MODE_W, c),
    (n::Int)               -> n,

    # Optional launcher options
    launcher = XK.KA.Launcher(

        # Optional grid size
        threads       = (a, b, c, n) -> (n, 1, 1),

        # Optional shared memory size
        shared_memory = (a, b, c, n) -> 0
    )
)

#####################
# Execution example #
#####################

# This is just any host virtual memory
# XKRT/XKBlas will automatically replicate to devices
n = 4
a = rand(T, n)
b = rand(T, n)
c = Vector{T}(undef, n)

# Spawn a task that executes the kernel
XK.KA.device_async(vector_add_format, a, b, c, n)

# Spawn a task that reads onto the host, to write-back from the previous kernel execution
XK.memory_coherent_async(c)

# Wait for tasks completion
XK.sync()

# Check
apb = a + b
println("  a = $a")
println("  b = $b")
println("a+b (Julia)  = $apb")
println("  c (XKBLAS) = $c")
@assert isapprox(apb, c; atol=1e-6)
