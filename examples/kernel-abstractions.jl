using Base.Libc
using KernelAbstractions
using XKBlas

# Kernel to execute and its accesses
@kernel function vector_add(a, b, c)
    i = @index(Global)
    @inbounds c[i] = a[i] + b[i]
end

const vector_add_format = XKBlas.KA.Format(
    vector_add,
    (a) -> Access(ACCESS_MODE_R, Segment(pointer(a), pointer(a) + length(a))),
    (b) -> Access(ACCESS_MODE_R, Segment(pointer(b), pointer(b) + length(b))),
    (c) -> Access(ACCESS_MODE_W, Segment(pointer(c), pointer(c) + length(c)))
)

# Create three vectors
n = 4
a = rand(Float64, n)
b = rand(Float64, n)
c = Vector{Float64}(undef, n)

# Spawn a task that executes the kernel, and a task that reads back onto the host
XKBlas.KA.device_async(vector_add_format, a, b, c)
XKBlas.memory_coherent_async(c)
XKBlas.sync()

println("c is coherent on the host")
