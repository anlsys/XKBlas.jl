using Test, XK

@testset "XK.jl" begin
    # init xkblas
    XK.init()

    # Problem setup
    n = 128
    m, n, k = n, n, n
    A = [Float32(rand()) for _ in 1:(m*k)]
    B = [Float32(rand()) for _ in 1:(k*n)]
    C = [Float32(0.0)    for _ in 1:(m*n)]

    alpha_vec = [Float32(1.0)]
    beta_vec  = [Float32(0.0)]

    lda, ldb, ldc = m, k, m

    transA, transB = XK.CblasNoTrans, XK.CblasNoTrans

    # Run an XK sequence
    XK.sgemm_async(
        transA, transB,
        m, n, k,
        alpha_vec,
        A, lda,
        B, ldb,
        beta_vec,
        C, ldc
    )
    XK.memory_matrix_coherent_async(C, ldc, m, n, sizeof(Float32))
    XK.sync()

    # TODO: check result

    # deinit xkblas
    XK.deinit()
end
