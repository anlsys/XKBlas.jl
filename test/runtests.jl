using XKBlas
using Test

@testset "XKBlas.jl" begin
    # init xkblas
    XKBlas.init()

    # Problem setup
    n = 128
    m, n, k = n, n, n
    A = [Float32(rand()) for _ in 1:(m*k)]
    B = [Float32(rand()) for _ in 1:(k*n)]
    C = [Float32(0.0)    for _ in 1:(m*n)]

    alpha_vec = [Float32(1.0)]
    beta_vec  = [Float32(0.0)]

    lda, ldb, ldc = m, k, m

    transA, transB = XKBlas.CblasNoTrans, XKBlas.CblasNoTrans

    # Run an XKBlas sequence
    XKBlas.sgemm_async(
        transA, transB,
        m, n, k,
        alpha_vec,
        A, lda,
        B, ldb,
        beta_vec,
        C, ldc
    )
    XKBlas.memory_matrix_coherent_async(C, ldc, m, n, sizeof(Float32))
    XKBlas.sync()

    # TODO: check result

    # deinit xkblas
    XKBlas.deinit()
end
