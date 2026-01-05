using LinearAlgebra, Random
using XKLas
const XK = XKLas

#################
# Problem setup #
#################

const T = Float32

# This is host memory, XKRT/XKLas will replicate to devices
n = 3 #32768
m, n, k = n, n, n
A = [T(rand()) for _ in 1:(m*k)]
B = [T(rand()) for _ in 1:(k*n)]
C = [T(0.0)    for _ in 1:(m*n)]
alpha = T(1.0)
beta  = T(0.0)
lda, ldb, ldc = m, k, m
transA, transB = XK.CblasNoTrans, XK.CblasNoTrans

@time begin

    # Run a gemm, it is automatically tiled and distributed to available GPUs
    # see https://gitlab.inria.fr/xkblas/dev/-/tree/v2.0/
    XK.gemm_async(
        transA, transB,
        m, n, k,
        alpha,
        A, lda,
        B, ldb,
        beta,
        C, ldc
    )

    # Write back to host memory
    XK.memory_matrix_coherent_async(C, ldc, m, n, sizeof(T))

    # wait for completion
    XK.sync()
end

# Print XKblas and Julia-native results
if (n <= 64)
    println("XKLas A = ", reshape(A, m, k))
    println("XKLas B = ", reshape(B, k, n))
    println("XKLas C = ", reshape(C, m, n))

    C_julia = reshape(A, m, k) * reshape(B, k, n)
    println(" Julia C = ", C_julia)
end
