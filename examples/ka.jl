using Base.Libc
using KernelAbstractions
using XKBlas
const XK = XKBlas

#########################
# Custom Kernel example #
#########################

# Declare kernels, and its memory accesses

# KA
# @kernel unsafe_indices=true function vector_add(a, b, c)
#     i = @index(Local, Linear)
#     @inbounds c[i] = a[i] + b[i]
# end

using CUDA  # or ROCm, oneAPI, etc.
using CUDA.CUDAKernels

function vector_add(a, b, c, n)
    i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if i <= n
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

const T = Float64

# A kernelabstraction task format
const vector_add_format = XK.KA.Format(

    # the kernelabstraction's kernel
    vector_add,

    # the number of threads
    (a, b, c, n) -> (n, 1, 1),

    # the amount of shared memory
    (a, b, c, n) -> 0,

    # the kernel parameters
    (a::AbstractVector{T}) -> XK.Access(XK.ACCESS_MODE_R, a),
    (b::AbstractVector{T}) -> XK.Access(XK.ACCESS_MODE_R, b),
    (c::AbstractVector{T}) -> XK.Access(XK.ACCESS_MODE_W, c),
    (n::Int)               -> n
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
