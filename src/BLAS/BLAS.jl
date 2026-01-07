# Kernels (see `xkblas/xkblas.hpp` and convert C++ prototypes)

module BLAS

    import ..XKLas
    const XK = XKLas

    # Enums #

    const ROW_MAJOR = XK.CblasRowMajor
    const COL_MAJOR = XK.CblasColMajor

    const NO_TRANS   = XK.CblasNoTrans
    const TRANS      = XK.CblasTrans
    const CONJ_TRANS = XK.CblasConjTrans

    const UPPER = XK.CblasUpper
    const LOWER = XK.CblasLower

    const NON_UNIT = XK.CblasNonUnit
    const UNIT     = XK.CblasUnit

    const LEFT  = XK.CblasLeft
    const RIGHT = XK.CblasRight

    const SPARSE_CSR = XK.CblasSparseCSR
    const SPARSE_CSC = XK.CblasSparseCSC
    const SPARSE_COO = XK.CblasSparseCOO
    const SPARSE_BSR = XK.CblasSparseBSR
    const SPARSE_ELL = XK.CblasSparseELL
    const SPARSE_DIA = XK.CblasSparseDIA

    ### Level 1 ###

    axpy(      n, alpha::ComplexF32, x, incx, y, incy)  = XK.caxpy(      n, Ref(alpha), x, incx, y, incy)
    axpy(      n, alpha::ComplexF64, x, incx, y, incy)  = XK.zaxpy(      n, Ref(alpha), x, incx, y, incy)
    axpy(      n, alpha::Float32,    x, incx, y, incy)  = XK.saxpy(      n, Ref(alpha), x, incx, y, incy)
    axpy(      n, alpha::Float64,    x, incx, y, incy)  = XK.daxpy(      n, Ref(alpha), x, incx, y, incy)
    axpy_async(n, alpha::ComplexF32, x, incx, y, incy)  = XK.caxpy_async(n, Ref(alpha), x, incx, y, incy)
    axpy_async(n, alpha::ComplexF64, x, incx, y, incy)  = XK.zaxpy_async(n, Ref(alpha), x, incx, y, incy)
    axpy_async(n, alpha::Float32,    x, incx, y, incy)  = XK.saxpy_async(n, Ref(alpha), x, incx, y, incy)
    axpy_async(n, alpha::Float64,    x, incx, y, incy)  = XK.daxpy_async(n, Ref(alpha), x, incx, y, incy)
    axpy_sync( n, alpha::ComplexF32, x, incx, y, incy)  = XK.caxpy_sync( n, Ref(alpha), x, incx, y, incy)
    axpy_sync( n, alpha::ComplexF64, x, incx, y, incy)  = XK.zaxpy_sync( n, Ref(alpha), x, incx, y, incy)
    axpy_sync( n, alpha::Float32,    x, incx, y, incy)  = XK.saxpy_sync( n, Ref(alpha), x, incx, y, incy)
    axpy_sync( n, alpha::Float64,    x, incx, y, incy)  = XK.daxpy_sync( n, Ref(alpha), x, incx, y, incy)

    # TODO: complex version not supported yet, but they will need to change the dispatcher
    dot(      n, x, incx, y, incy, result::Ref{Float32}) = XK.sdot(      n, x, incx, y, incy, result)
    dot(      n, x, incx, y, incy, result::Ref{Float64}) = XK.ddot(      n, x, incx, y, incy, result)
    dot_async(n, x, incx, y, incy, result::Ref{Float32}) = XK.sdot_async(n, x, incx, y, incy, result)
    dot_async(n, x, incx, y, incy, result::Ref{Float64}) = XK.ddot_async(n, x, incx, y, incy, result)
    dot_sync( n, x, incx, y, incy, result::Ref{Float32}) = XK.sdot_sync( n, x, incx, y, incy, result)
    dot_sync( n, x, incx, y, incy, result::Ref{Float64}) = XK.ddot_sync( n, x, incx, y, incy, result)

    ### Level 2 ###

    gemv(      transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = XK.sgemv(      transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv(      transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = XK.dgemv(      transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv_async(transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = XK.sgemv_async(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv_async(transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = XK.dgemv_async(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv_sync( transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = XK.sgemv_sync( transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv_sync( transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = XK.dgemv_sync( transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)

    gemv(      transA, alpha, A::AbstractMatrix, x, beta, y) = XK.gemv(transA, size(A, 1), size(A, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(x), stride(x, 1), beta, pointer(y), stride(y, 1))
    gemv_async(transA, alpha, A::AbstractMatrix, x, beta, y) = XK.gemv(transA, size(A, 1), size(A, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(x), stride(x, 1), beta, pointer(y), stride(y, 1))
    gemv_sync( transA, alpha, A::AbstractMatrix, x, beta, y) = XK.gemv(transA, size(A, 1), size(A, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(x), stride(x, 1), beta, pointer(y), stride(y, 1))

    ### Level 3 ###

    gemm(      transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm(      transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm(      transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm(      transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_async(transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_async(transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_async(transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_async(transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_sync( transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_sync( transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_sync( transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_sync( transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    gemm(      transA, transB, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)  = XK.gemm(transA, transB, size(C, 1), size(C, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(B), stride(B, 2), beta, pointer(C), stride(C, 2))
    gemm_async(transA, transB, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)  = XK.gemm(transA, transB, size(C, 1), size(C, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(B), stride(B, 2), beta, pointer(C), stride(C, 2))
    gemm_sync( transA, transB, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)  = XK.gemm(transA, transB, size(C, 1), size(C, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(B), stride(B, 2), beta, pointer(C), stride(C, 2))

    herk(      uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.cherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk(      uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk(      uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.sherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk(      uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.cherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_async(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.sherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_async(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_sync( uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.cherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_sync( uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_sync( uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.sherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_sync( uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    symm(      side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm(      side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm(      side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm(      side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_async(side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_async(side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_async(side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_async(side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_sync( side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_sync( side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_sync( side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_sync( side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    syr2k(      uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k(      uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k(      uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k(      uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_async(uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_async(uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_sync( uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_sync( uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_sync( uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_sync( uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    syrk(      uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.csyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk(      uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zsyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk(      uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.ssyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk(      uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dsyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.csyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zsyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_async(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.ssyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_async(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dsyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_sync( uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.csyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_sync( uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zsyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_sync( uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.ssyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_sync( uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dsyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    trmm(      side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = XK.ctrmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm(      side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = XK.ztrmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm(      side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = XK.strmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm(      side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = XK.dtrmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_async(side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = XK.ctrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_async(side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = XK.ztrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_async(side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = XK.strmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_async(side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = XK.dtrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_sync( side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = XK.ctrmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_sync( side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = XK.ztrmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_sync( side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = XK.strmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_sync( side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = XK.dtrmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)

    # SPMV
    spmv(      alpha::ComplexF32, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.cspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv(      alpha::ComplexF64, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.zspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv(      alpha::Float32,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.sspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv(      alpha::Float64,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.dspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_async(alpha::ComplexF32, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.cspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_async(alpha::ComplexF64, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.zspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_async(alpha::Float32,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.sspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_async(alpha::Float64,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.dspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_sync( alpha::ComplexF32, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.cspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_sync( alpha::ComplexF64, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.zspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_sync( alpha::Float32,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.sspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_sync( alpha::Float64,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.dspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)

    module ext

        import ..XK

        ### Level 1 ###

        axpby(      n, alpha::ComplexF32, x, incx, beta, y, incy) = XK.caxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby(      n, alpha::ComplexF64, x, incx, beta, y, incy) = XK.zaxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby(      n, alpha::Float32,    x, incx, beta, y, incy) = XK.saxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby(      n, alpha::Float64,    x, incx, beta, y, incy) = XK.daxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_async(n, alpha::ComplexF32, x, incx, beta, y, incy) = XK.caxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_async(n, alpha::ComplexF64, x, incx, beta, y, incy) = XK.zaxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_async(n, alpha::Float32,    x, incx, beta, y, incy) = XK.saxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_async(n, alpha::Float64,    x, incx, beta, y, incy) = XK.daxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_sync( n, alpha::ComplexF32, x, incx, beta, y, incy) = XK.caxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_sync( n, alpha::ComplexF64, x, incx, beta, y, incy) = XK.zaxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_sync( n, alpha::Float32,    x, incx, beta, y, incy) = XK.saxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_sync( n, alpha::Float64,    x, incx, beta, y, incy) = XK.daxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)

        copy(      n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = XK.ccopy(      n, x, incx, y, incy)
        copy(      n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = XK.zcopy(      n, x, incx, y, incy)
        copy(      n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = XK.scopy(      n, x, incx, y, incy)
        copy(      n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = XK.dcopy(      n, x, incx, y, incy)
        copy_async(n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = XK.ccopy_async(n, x, incx, y, incy)
        copy_async(n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = XK.zcopy_async(n, x, incx, y, incy)
        copy_async(n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = XK.scopy_async(n, x, incx, y, incy)
        copy_async(n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = XK.dcopy_async(n, x, incx, y, incy)
        copy_sync( n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = XK.ccopy_sync( n, x, incx, y, incy)
        copy_sync( n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = XK.zcopy_sync( n, x, incx, y, incy)
        copy_sync( n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = XK.scopy_sync( n, x, incx, y, incy)
        copy_sync( n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = XK.dcopy_sync( n, x, incx, y, incy)

        fill(      n, x, value::ComplexF32) = XK.cfill(      n, x, value)
        fill(      n, x, value::ComplexF64) = XK.zfill(      n, x, value)
        fill(      n, x, value::Float32   ) = XK.sfill(      n, x, value)
        fill(      n, x, value::Float64   ) = XK.dfill(      n, x, value)
        fill_async(n, x, value::ComplexF32) = XK.cfill_async(n, x, value)
        fill_async(n, x, value::ComplexF64) = XK.zfill_async(n, x, value)
        fill_async(n, x, value::Float32   ) = XK.sfill_async(n, x, value)
        fill_async(n, x, value::Float64   ) = XK.dfill_async(n, x, value)
        fill_sync( n, x, value::ComplexF32) = XK.cfill_sync( n, x, value)
        fill_sync( n, x, value::ComplexF64) = XK.zfill_sync( n, x, value)
        fill_sync( n, x, value::Float32   ) = XK.sfill_sync( n, x, value)
        fill_sync( n, x, value::Float64   ) = XK.dfill_sync( n, x, value)

        # TODO: complex version not supported yet, but they will need to change the dispatcher
        scal(      n, alpha::Float32, x, incx) = XK.sscal(      n, Ref(alpha), x, incx)
        scal(      n, alpha::Float64, x, incx) = XK.dscal(      n, Ref(alpha), x, incx)
        scal_async(n, alpha::Float32, x, incx) = XK.sscal_async(n, Ref(alpha), x, incx)
        scal_async(n, alpha::Float64, x, incx) = XK.dscal_async(n, Ref(alpha), x, incx)
        scal_sync( n, alpha::Float32, x, incx) = XK.sscal_sync( n, Ref(alpha), x, incx)
        scal_sync( n, alpha::Float64, x, incx) = XK.dscal_sync( n, Ref(alpha), x, incx)

        ### Level 3 ###

        gemmt(      uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt(      uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt(      uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt(      uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_async(uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_async(uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_async(uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_async(uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_sync( uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_sync( uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_sync( uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_sync( uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    end

end
