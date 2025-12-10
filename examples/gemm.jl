using LinearAlgebra, Random
using XKBlas
const XK = XKBlas

#################
# Problem setup #
#################

# This is host memory, XKRT/XKBlas will replicate to devices
n = 3 #32768
m, n, k = n, n, n
A = [Float32(rand()) for _ in 1:(m*k)]
B = [Float32(rand()) for _ in 1:(k*n)]
C = [Float32(0.0)    for _ in 1:(m*n)]
alpha_vec = [Float32(1.0)]
beta_vec  = [Float32(0.0)]
lda, ldb, ldc = m, k, m
transA, transB = XK.CblasNoTrans, XK.CblasNoTrans

@time begin

    # Run a gemm, it is automatically tiled and distributed to available GPUs
    # see https://gitlab.inria.fr/xkblas/dev/-/tree/v2.0/
    XK.gemm_async(
        transA, transB,
        m, n, k,
        alpha_vec,
        A, lda,
        B, ldb,
        beta_vec,
        C, ldc
    )

    # Write back to host memory
    XK.memory_matrix_coherent_async(C, ldc, m, n, sizeof(Float32))

    # wait for completion
    XK.sync()
end

# Print XKblas and Julia-native results
if (n <= 64)
    println("XKBlas A = ", reshape(A, m, k))
    println("XKBlas B = ", reshape(B, k, n))
    println("XKBlas C = ", reshape(C, m, n))

    C_julia = reshape(A, m, k) * reshape(B, k, n)
    println(" Julia C = ", C_julia)
end
