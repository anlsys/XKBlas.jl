using LinearAlgebra, Random, XK

TYPE = Float32

# Problem setup
n = 4
m, n, k = n, n, n
A = [TYPE(rand()) for _ in 1:(m*k)]
B = [TYPE(rand()) for _ in 1:(k*n)]
C = [TYPE(0.0)    for _ in 1:(m*n)]

alpha_vec = TYPE(1.0)
beta_vec  = TYPE(0.0)

lda, ldb, ldc = m, k, m

transA, transB = XK.NO_TRANS, XK.NO_TRANS

@time begin
    XK.gemm_async(
        transA, transB,
        m, n, k,
        alpha_vec,
        A, lda,
        B, ldb,
        beta_vec,
        C, ldc
    )
    XK.memory_coherent_async(C, ldc, m, n)
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
