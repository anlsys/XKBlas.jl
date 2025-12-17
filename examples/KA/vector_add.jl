using Base.Libc
using XKBlas
const XK = XKBlas

#########################
# Custom Kernel example #
#########################

# All of this is type-agnostic, it is instanciated and compiled on-line with passed instances

# vector addition kernel, typ
@XK.KA.kernel function vector_add(a, b, c, n)
    i = @XK.KA.tid
    if i <= n
        c[i] = a[i] + b[i]
    end
end

# An XK.KA format
const vector_add_format = XK.KA.Format(

    # the XK.KA kernel
    vector_add,

    # the task accesses
    (a, b, c, n) -> (XK.Access(XK.ACCESS_MODE_R, a),
                     XK.Access(XK.ACCESS_MODE_R, b),
                     XK.Access(XK.ACCESS_MODE_W, c),
                     XK.Value),

    # Optional grid size
    threads       = (a, b, c, n) -> (n, 1, 1),

    # Optional shared memory size
    shared_memory = (a, b, c, n) -> 0
)

#####################
# Execution example #
#####################

const T = Float64

# This is just any host virtual memory
# XKRT automatically replicates to devices
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
