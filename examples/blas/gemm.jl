using LinearAlgebra, Random, XK

#################
# Problem setup #
#################

const T = Float32

# This is host memory, XKRT/XK will replicate to devices
n = 3 #32768
m, n, k = n, n, n
A = [T(rand()) for _ in 1:(m*k)]
B = [T(rand()) for _ in 1:(k*n)]
C = [T(0.0)    for _ in 1:(m*n)]
alpha = T(1.0)
beta  = T(0.0)
lda, ldb, ldc = m, k, m
transA, transB = XK.BLAS.NO_TRANS, XK.BLAS.NO_TRANS

@time begin

    # Run a gemm, it is automatically tiled and distributed to available GPUs
    # see https://gitlab.inria.fr/xkblas/dev/-/tree/v2.0/
    XK.BLAS.gemm_async(
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
    println("XK A = ", reshape(A, m, k))
    println("XK B = ", reshape(B, k, n))
    println("XK C = ", reshape(C, m, n))

    C_julia = reshape(A, m, k) * reshape(B, k, n)
    println(" Julia C = ", C_julia)
end
