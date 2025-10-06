using CEnum: CEnum, @cenum

function init()
    @ccall libxkblas.xkblas_init()::Cint
end

function get_device_count(count)
    @ccall libxkblas.xkblas_get_device_count(count::Ptr{Cint})::Cint
end

function sync()
    @ccall libxkblas.xkblas_sync()::Cvoid
end

function deinit()
    @ccall libxkblas.xkblas_deinit()::Cvoid
end

function memory_segment_coherent_async(ptr, size)
    @ccall libxkblas.xkblas_memory_segment_coherent_async(ptr::Ptr{Cvoid}, size::Csize_t)::Cvoid
end

function memory_matrix_coherent_async(ptr, ld, m, n, sizeof_type)
    @ccall libxkblas.xkblas_memory_matrix_coherent_async(ptr::Ptr{Cvoid}, ld::Csize_t, m::Csize_t, n::Csize_t, sizeof_type::Csize_t)::Cvoid
end

function host_async(func, args)
    @ccall libxkblas.xkblas_host_async(func::Ptr{Cvoid}, args::Ptr{Cvoid})::Cvoid
end

function unified_alloc(size)
    @ccall libxkblas.xkblas_unified_alloc(size::Csize_t)::Ptr{Cvoid}
end

function unified_free(ptr, size)
    @ccall libxkblas.xkblas_unified_free(ptr::Ptr{Cvoid}, size::Csize_t)::Cvoid
end

function host_alloc(size)
    @ccall libxkblas.xkblas_host_alloc(size::Csize_t)::Ptr{Cvoid}
end

function host_free(ptr, size)
    @ccall libxkblas.xkblas_host_free(ptr::Ptr{Cvoid}, size::Csize_t)::Cvoid
end

@cenum xkblas_mode_math_t::UInt32 begin
    XKBLAS_DEFAULT_MATH = 0
    XKBLAS_TENSOR_OP_MATH = 1
end

function set_modemath(mode)
    @ccall libxkblas.xkblas_set_modemath(mode::xkblas_mode_math_t)::Cvoid
end

function register_memory(ptr, sz)
    @ccall libxkblas.xkblas_register_memory(ptr::Ptr{Cvoid}, sz::UInt64)::Cint
end

function unregister_memory(ptr, sz)
    @ccall libxkblas.xkblas_unregister_memory(ptr::Ptr{Cvoid}, sz::UInt64)::Cint
end

function memory_register_tiled_async(ptr, sz, n)
    @ccall libxkblas.xkblas_memory_register_tiled_async(ptr::Ptr{Cvoid}, sz::Csize_t, n::Cint)::Cint
end

function memory_unregister_tiled_async(ptr, sz, n)
    @ccall libxkblas.xkblas_memory_unregister_tiled_async(ptr::Ptr{Cvoid}, sz::Csize_t, n::Cint)::Cint
end

function memory_touch_tiled_async(ptr, sz, n)
    @ccall libxkblas.xkblas_memory_touch_tiled_async(ptr::Ptr{Cvoid}, sz::Csize_t, n::Cint)::Cint
end

function get_ngpus()
    @ccall libxkblas.xkblas_get_ngpus()::Cint
end

function get_nanotime()
    @ccall libxkblas.xkblas_get_nanotime()::UInt64
end

"""
    xkblas_malloc(size)

///////////////////////////////
"""
function malloc(size)
    @ccall libxkblas.xkblas_malloc(size::Csize_t)::Ptr{Cvoid}
end

function free(ptr, size)
    @ccall libxkblas.xkblas_free(ptr::Ptr{Cvoid}, size::Csize_t)::Cvoid
end

function set_param(nb, p)
    @ccall libxkblas.xkblas_set_param(nb::Csize_t, p::Csize_t)::Cvoid
end

function finalize()
    @ccall libxkblas.xkblas_finalize()::Cvoid
end

function memory_invalidate_caches()
    @ccall libxkblas.xkblas_memory_invalidate_caches()::Cvoid
end

function register_memory_async(ptr, sz)
    @ccall libxkblas.xkblas_register_memory_async(ptr::Ptr{Cvoid}, sz::UInt64)::UInt64
end

function unregister_memory_async(ptr, sz)
    @ccall libxkblas.xkblas_unregister_memory_async(ptr::Ptr{Cvoid}, sz::UInt64)::Cint
end

function register_memory_waitall()
    @ccall libxkblas.xkblas_register_memory_waitall()::Cint
end

function memory_coherent_async(uplo, memflag, M, N, A, LD, eltsize)
    @ccall libxkblas.xkblas_memory_coherent_async(uplo::Cint, memflag::Cint, M::Csize_t, N::Csize_t, A::Ptr{Cvoid}, LD::Csize_t, eltsize::Csize_t)::Cint
end

const Complex32_t = ComplexF32

const Complex64_t = ComplexF32

const CFloat64_t = Cdouble

function saxpby_async(n, alpha, x, incx, beta, y, incy)
    @ccall libxkblas.xkblas_saxpby_async(n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)::Cint
end

function saxpy_async(n, alpha, x, incx, y, incy)
    @ccall libxkblas.xkblas_saxpy_async(n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)::Cint
end

function sdot_async(n, x, incx, y, incy, result)
    @ccall libxkblas.xkblas_sdot_async(n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, result::Ptr{Cfloat})::Cint
end

# no prototype is found for this function at skernels.h:52:9, please use with caution
function sdivcopy_async()
    @ccall libxkblas.xkblas_sdivcopy_async()::Cint
end

function sfill(n, x, v)
    @ccall libxkblas.xkblas_sfill(n::Cint, x::Ptr{Cfloat}, v::Cfloat)::Cint
end

function snrm2_async(n, x, result)
    @ccall libxkblas.xkblas_snrm2_async(n::Cint, x::Ptr{Cfloat}, result::Ptr{Cfloat})::Cint
end

# no prototype is found for this function at skernels.h:58:9, please use with caution
function sscalcopy_async()
    @ccall libxkblas.xkblas_sscalcopy_async()::Cint
end

function sscal_async(n, alpha, x, incx)
    @ccall libxkblas.xkblas_sscal_async(n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint)::Cint
end

function scopyscale_async(m, n, should_copy, IW, D, ldd, L, ldl, U, ldu)
    @ccall libxkblas.xkblas_scopyscale_async(m::Cint, n::Cint, should_copy::Cint, IW::Ptr{Cint}, D::Ptr{Cfloat}, ldd::Cint, L::Ptr{Cfloat}, ldl::Cint, U::Ptr{Cfloat}, ldu::Cint)::Cint
end

function sgemv_async(transA, m, n, alpha, A, lda, x, incx, beta, y, incy)
    @ccall libxkblas.xkblas_sgemv_async(transA::Cint, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)::Cint
end

function sgemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_sgemm_async(transA::Cint, transB::Cint, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::Cint
end

function sgemmt_async(uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_sgemmt_async(uplo::Cint, transA::Cint, transB::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::Cint
end

function sherk_async(uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.xkblas_sherk_async(uplo::Cint, transA::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::Cint
end

function ssyrk_async(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.xkblas_ssyrk_async(uplo::Cint, trans::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::Cint
end

function strsm_async(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    @ccall libxkblas.xkblas_strsm_async(side::Cint, uplo::Cint, transA::Cint, diag::Cint, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint)::Cint
end

function strmm_async(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    @ccall libxkblas.xkblas_strmm_async(side::Cint, uplo::Cint, transA::Cint, diag::Cint, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint)::Cint
end

function ssyr2k_async(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_ssyr2k_async(uplo::Cint, trans::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::Cint
end

function ssymm_async(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_ssymm_async(side::Cint, uplo::Cint, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::Cint
end

function spotrf_async(uplo, n, A, lda)
    @ccall libxkblas.xkblas_spotrf_async(uplo::Cint, n::Cint, A::Ptr{Cfloat}, lda::Cint)::Cint
end

function sspmv_async(alpha, transA, index_base, index_type, nrows, ncols, nnz, format, csr_row_offsets, csr_col_indices, csr_values, X, beta, Y)
    @ccall libxkblas.xkblas_sspmv_async(alpha::Ptr{Cfloat}, transA::Cint, index_base::Cint, index_type::Cint, nrows::Cint, ncols::Cint, nnz::Cint, format::Cint, csr_row_offsets::Ptr{Cvoid}, csr_col_indices::Ptr{Cvoid}, csr_values::Ptr{Cfloat}, X::Ptr{Cfloat}, beta::Ptr{Cfloat}, Y::Ptr{Cfloat})::Cint
end

function daxpby_async(n, alpha, x, incx, beta, y, incy)
    @ccall libxkblas.xkblas_daxpby_async(n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)::Cint
end

function daxpy_async(n, alpha, x, incx, y, incy)
    @ccall libxkblas.xkblas_daxpy_async(n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)::Cint
end

function ddot_async(n, x, incx, y, incy, result)
    @ccall libxkblas.xkblas_ddot_async(n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, result::Ptr{Cdouble})::Cint
end

# no prototype is found for this function at dkernels.h:52:9, please use with caution
function ddivcopy_async()
    @ccall libxkblas.xkblas_ddivcopy_async()::Cint
end

function dfill(n, x, v)
    @ccall libxkblas.xkblas_dfill(n::Cint, x::Ptr{Cdouble}, v::Cdouble)::Cint
end

function dnrm2_async(n, x, result)
    @ccall libxkblas.xkblas_dnrm2_async(n::Cint, x::Ptr{Cdouble}, result::Ptr{Cfloat})::Cint
end

# no prototype is found for this function at dkernels.h:58:9, please use with caution
function dscalcopy_async()
    @ccall libxkblas.xkblas_dscalcopy_async()::Cint
end

function dscal_async(n, alpha, x, incx)
    @ccall libxkblas.xkblas_dscal_async(n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint)::Cint
end

function dcopyscale_async(m, n, should_copy, IW, D, ldd, L, ldl, U, ldu)
    @ccall libxkblas.xkblas_dcopyscale_async(m::Cint, n::Cint, should_copy::Cint, IW::Ptr{Cint}, D::Ptr{Cdouble}, ldd::Cint, L::Ptr{Cdouble}, ldl::Cint, U::Ptr{Cdouble}, ldu::Cint)::Cint
end

function dgemv_async(transA, m, n, alpha, A, lda, x, incx, beta, y, incy)
    @ccall libxkblas.xkblas_dgemv_async(transA::Cint, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)::Cint
end

function dgemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_dgemm_async(transA::Cint, transB::Cint, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::Cint
end

function dgemmt_async(uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_dgemmt_async(uplo::Cint, transA::Cint, transB::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::Cint
end

function dherk_async(uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.xkblas_dherk_async(uplo::Cint, transA::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::Cint
end

function dsyrk_async(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.xkblas_dsyrk_async(uplo::Cint, trans::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::Cint
end

function dtrsm_async(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    @ccall libxkblas.xkblas_dtrsm_async(side::Cint, uplo::Cint, transA::Cint, diag::Cint, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint)::Cint
end

function dtrmm_async(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    @ccall libxkblas.xkblas_dtrmm_async(side::Cint, uplo::Cint, transA::Cint, diag::Cint, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint)::Cint
end

function dsyr2k_async(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_dsyr2k_async(uplo::Cint, trans::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::Cint
end

function dsymm_async(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_dsymm_async(side::Cint, uplo::Cint, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::Cint
end

function dpotrf_async(uplo, n, A, lda)
    @ccall libxkblas.xkblas_dpotrf_async(uplo::Cint, n::Cint, A::Ptr{Cdouble}, lda::Cint)::Cint
end

function dspmv_async(alpha, transA, index_base, index_type, nrows, ncols, nnz, format, csr_row_offsets, csr_col_indices, csr_values, X, beta, Y)
    @ccall libxkblas.xkblas_dspmv_async(alpha::Ptr{Cdouble}, transA::Cint, index_base::Cint, index_type::Cint, nrows::Cint, ncols::Cint, nnz::Cint, format::Cint, csr_row_offsets::Ptr{Cvoid}, csr_col_indices::Ptr{Cvoid}, csr_values::Ptr{Cdouble}, X::Ptr{Cdouble}, beta::Ptr{Cdouble}, Y::Ptr{Cdouble})::Cint
end

function caxpby_async(n, alpha, x, incx, beta, y, incy)
    @ccall libxkblas.xkblas_caxpby_async(n::Cint, alpha::Ptr{ComplexF32}, x::Ptr{ComplexF32}, incx::Cint, beta::Ptr{ComplexF32}, y::Ptr{ComplexF32}, incy::Cint)::Cint
end

function caxpy_async(n, alpha, x, incx, y, incy)
    @ccall libxkblas.xkblas_caxpy_async(n::Cint, alpha::Ptr{ComplexF32}, x::Ptr{ComplexF32}, incx::Cint, y::Ptr{ComplexF32}, incy::Cint)::Cint
end

function cdot_async(n, x, incx, y, incy, result)
    @ccall libxkblas.xkblas_cdot_async(n::Cint, x::Ptr{ComplexF32}, incx::Cint, y::Ptr{ComplexF32}, incy::Cint, result::Ptr{ComplexF32})::Cint
end

# no prototype is found for this function at ckernels.h:52:9, please use with caution
function cdivcopy_async()
    @ccall libxkblas.xkblas_cdivcopy_async()::Cint
end

function cfill(n, x, v)
    @ccall libxkblas.xkblas_cfill(n::Cint, x::Ptr{ComplexF32}, v::ComplexF32)::Cint
end

function cnrm2_async(n, x, result)
    @ccall libxkblas.xkblas_cnrm2_async(n::Cint, x::Ptr{ComplexF32}, result::Ptr{Cfloat})::Cint
end

# no prototype is found for this function at ckernels.h:58:9, please use with caution
function cscalcopy_async()
    @ccall libxkblas.xkblas_cscalcopy_async()::Cint
end

function cscal_async(n, alpha, x, incx)
    @ccall libxkblas.xkblas_cscal_async(n::Cint, alpha::Ptr{ComplexF32}, x::Ptr{ComplexF32}, incx::Cint)::Cint
end

function ccopyscale_async(m, n, should_copy, IW, D, ldd, L, ldl, U, ldu)
    @ccall libxkblas.xkblas_ccopyscale_async(m::Cint, n::Cint, should_copy::Cint, IW::Ptr{Cint}, D::Ptr{ComplexF32}, ldd::Cint, L::Ptr{ComplexF32}, ldl::Cint, U::Ptr{ComplexF32}, ldu::Cint)::Cint
end

function cgemv_async(transA, m, n, alpha, A, lda, x, incx, beta, y, incy)
    @ccall libxkblas.xkblas_cgemv_async(transA::Cint, m::Cint, n::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, x::Ptr{ComplexF32}, incx::Cint, beta::Ptr{ComplexF32}, y::Ptr{ComplexF32}, incy::Cint)::Cint
end

function cgemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_cgemm_async(transA::Cint, transB::Cint, m::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function cgemmt_async(uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_cgemmt_async(uplo::Cint, transA::Cint, transB::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function cherk_async(uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.xkblas_cherk_async(uplo::Cint, transA::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function csyrk_async(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.xkblas_csyrk_async(uplo::Cint, trans::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function ctrsm_async(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    @ccall libxkblas.xkblas_ctrsm_async(side::Cint, uplo::Cint, transA::Cint, diag::Cint, m::Cint, n::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint)::Cint
end

function ctrmm_async(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    @ccall libxkblas.xkblas_ctrmm_async(side::Cint, uplo::Cint, transA::Cint, diag::Cint, m::Cint, n::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint)::Cint
end

function csyr2k_async(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_csyr2k_async(uplo::Cint, trans::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function csymm_async(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_csymm_async(side::Cint, uplo::Cint, m::Cint, n::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function cpotrf_async(uplo, n, A, lda)
    @ccall libxkblas.xkblas_cpotrf_async(uplo::Cint, n::Cint, A::Ptr{ComplexF32}, lda::Cint)::Cint
end

function cspmv_async(alpha, transA, index_base, index_type, nrows, ncols, nnz, format, csr_row_offsets, csr_col_indices, csr_values, X, beta, Y)
    @ccall libxkblas.xkblas_cspmv_async(alpha::Ptr{ComplexF32}, transA::Cint, index_base::Cint, index_type::Cint, nrows::Cint, ncols::Cint, nnz::Cint, format::Cint, csr_row_offsets::Ptr{Cvoid}, csr_col_indices::Ptr{Cvoid}, csr_values::Ptr{ComplexF32}, X::Ptr{ComplexF32}, beta::Ptr{ComplexF32}, Y::Ptr{ComplexF32})::Cint
end

function zaxpby_async(n, alpha, x, incx, beta, y, incy)
    @ccall libxkblas.xkblas_zaxpby_async(n::Cint, alpha::Ptr{ComplexF32}, x::Ptr{ComplexF32}, incx::Cint, beta::Ptr{ComplexF32}, y::Ptr{ComplexF32}, incy::Cint)::Cint
end

function zaxpy_async(n, alpha, x, incx, y, incy)
    @ccall libxkblas.xkblas_zaxpy_async(n::Cint, alpha::Ptr{ComplexF32}, x::Ptr{ComplexF32}, incx::Cint, y::Ptr{ComplexF32}, incy::Cint)::Cint
end

function zdot_async(n, x, incx, y, incy, result)
    @ccall libxkblas.xkblas_zdot_async(n::Cint, x::Ptr{ComplexF32}, incx::Cint, y::Ptr{ComplexF32}, incy::Cint, result::Ptr{ComplexF32})::Cint
end

# no prototype is found for this function at zkernels.h:52:9, please use with caution
function zdivcopy_async()
    @ccall libxkblas.xkblas_zdivcopy_async()::Cint
end

function zfill(n, x, v)
    @ccall libxkblas.xkblas_zfill(n::Cint, x::Ptr{ComplexF32}, v::ComplexF32)::Cint
end

function znrm2_async(n, x, result)
    @ccall libxkblas.xkblas_znrm2_async(n::Cint, x::Ptr{ComplexF32}, result::Ptr{Cfloat})::Cint
end

# no prototype is found for this function at zkernels.h:58:9, please use with caution
function zscalcopy_async()
    @ccall libxkblas.xkblas_zscalcopy_async()::Cint
end

function zscal_async(n, alpha, x, incx)
    @ccall libxkblas.xkblas_zscal_async(n::Cint, alpha::Ptr{ComplexF32}, x::Ptr{ComplexF32}, incx::Cint)::Cint
end

function zcopyscale_async(m, n, should_copy, IW, D, ldd, L, ldl, U, ldu)
    @ccall libxkblas.xkblas_zcopyscale_async(m::Cint, n::Cint, should_copy::Cint, IW::Ptr{Cint}, D::Ptr{ComplexF32}, ldd::Cint, L::Ptr{ComplexF32}, ldl::Cint, U::Ptr{ComplexF32}, ldu::Cint)::Cint
end

function zgemv_async(transA, m, n, alpha, A, lda, x, incx, beta, y, incy)
    @ccall libxkblas.xkblas_zgemv_async(transA::Cint, m::Cint, n::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, x::Ptr{ComplexF32}, incx::Cint, beta::Ptr{ComplexF32}, y::Ptr{ComplexF32}, incy::Cint)::Cint
end

function zgemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_zgemm_async(transA::Cint, transB::Cint, m::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function zgemmt_async(uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_zgemmt_async(uplo::Cint, transA::Cint, transB::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function zherk_async(uplo, transA, n, k, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.xkblas_zherk_async(uplo::Cint, transA::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function zsyrk_async(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.xkblas_zsyrk_async(uplo::Cint, trans::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function ztrsm_async(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    @ccall libxkblas.xkblas_ztrsm_async(side::Cint, uplo::Cint, transA::Cint, diag::Cint, m::Cint, n::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint)::Cint
end

function ztrmm_async(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
    @ccall libxkblas.xkblas_ztrmm_async(side::Cint, uplo::Cint, transA::Cint, diag::Cint, m::Cint, n::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint)::Cint
end

function zsyr2k_async(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_zsyr2k_async(uplo::Cint, trans::Cint, n::Cint, k::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function zsymm_async(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.xkblas_zsymm_async(side::Cint, uplo::Cint, m::Cint, n::Cint, alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Cint, B::Ptr{ComplexF32}, ldb::Cint, beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Cint)::Cint
end

function zpotrf_async(uplo, n, A, lda)
    @ccall libxkblas.xkblas_zpotrf_async(uplo::Cint, n::Cint, A::Ptr{ComplexF32}, lda::Cint)::Cint
end

function zspmv_async(alpha, transA, index_base, index_type, nrows, ncols, nnz, format, csr_row_offsets, csr_col_indices, csr_values, X, beta, Y)
    @ccall libxkblas.xkblas_zspmv_async(alpha::Ptr{ComplexF32}, transA::Cint, index_base::Cint, index_type::Cint, nrows::Cint, ncols::Cint, nnz::Cint, format::Cint, csr_row_offsets::Ptr{Cvoid}, csr_col_indices::Ptr{Cvoid}, csr_values::Ptr{ComplexF32}, X::Ptr{ComplexF32}, beta::Ptr{ComplexF32}, Y::Ptr{ComplexF32})::Cint
end

@cenum CBLAS_ORDER::UInt32 begin
    CblasRowMajor = 101
    CblasColMajor = 102
end

@cenum CBLAS_TRANSPOSE::UInt32 begin
    CblasNoTrans = 111
    CblasTrans = 112
    CblasConjTrans = 113
end

@cenum CBLAS_UPLO::UInt32 begin
    CblasUpper = 121
    CblasLower = 122
end

@cenum CBLAS_DIAG::UInt32 begin
    CblasNonUnit = 131
    CblasUnit = 132
end

@cenum CBLAS_SIDE::UInt32 begin
    CblasLeft = 141
    CblasRight = 142
end

function cblas_sdsdot(N, alpha, X, incX, Y, incY)
    @ccall libxkblas.cblas_sdsdot(N::Cint, alpha::Cfloat, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint)::Cfloat
end

function cblas_dsdot(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_dsdot(N::Cint, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint)::Cdouble
end

function cblas_sdot(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_sdot(N::Cint, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint)::Cfloat
end

function cblas_ddot(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_ddot(N::Cint, X::Ptr{Cdouble}, incX::Cint, Y::Ptr{Cdouble}, incY::Cint)::Cdouble
end

function cblas_cdotu_sub(N, X, incX, Y, incY, dotu)
    @ccall libxkblas.cblas_cdotu_sub(N::Cint, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, dotu::Ptr{Cvoid})::Cvoid
end

function cblas_cdotc_sub(N, X, incX, Y, incY, dotc)
    @ccall libxkblas.cblas_cdotc_sub(N::Cint, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, dotc::Ptr{Cvoid})::Cvoid
end

function cblas_zdotu_sub(N, X, incX, Y, incY, dotu)
    @ccall libxkblas.cblas_zdotu_sub(N::Cint, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, dotu::Ptr{Cvoid})::Cvoid
end

function cblas_zdotc_sub(N, X, incX, Y, incY, dotc)
    @ccall libxkblas.cblas_zdotc_sub(N::Cint, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, dotc::Ptr{Cvoid})::Cvoid
end

function cblas_snrm2(N, X, incX)
    @ccall libxkblas.cblas_snrm2(N::Cint, X::Ptr{Cfloat}, incX::Cint)::Cfloat
end

function cblas_sasum(N, X, incX)
    @ccall libxkblas.cblas_sasum(N::Cint, X::Ptr{Cfloat}, incX::Cint)::Cfloat
end

function cblas_dnrm2(N, X, incX)
    @ccall libxkblas.cblas_dnrm2(N::Cint, X::Ptr{Cdouble}, incX::Cint)::Cdouble
end

function cblas_dasum(N, X, incX)
    @ccall libxkblas.cblas_dasum(N::Cint, X::Ptr{Cdouble}, incX::Cint)::Cdouble
end

function cblas_scnrm2(N, X, incX)
    @ccall libxkblas.cblas_scnrm2(N::Cint, X::Ptr{Cvoid}, incX::Cint)::Cfloat
end

function cblas_scasum(N, X, incX)
    @ccall libxkblas.cblas_scasum(N::Cint, X::Ptr{Cvoid}, incX::Cint)::Cfloat
end

function cblas_dznrm2(N, X, incX)
    @ccall libxkblas.cblas_dznrm2(N::Cint, X::Ptr{Cvoid}, incX::Cint)::Cdouble
end

function cblas_dzasum(N, X, incX)
    @ccall libxkblas.cblas_dzasum(N::Cint, X::Ptr{Cvoid}, incX::Cint)::Cdouble
end

function cblas_isamax(N, X, incX)
    @ccall libxkblas.cblas_isamax(N::Cint, X::Ptr{Cfloat}, incX::Cint)::Csize_t
end

function cblas_idamax(N, X, incX)
    @ccall libxkblas.cblas_idamax(N::Cint, X::Ptr{Cdouble}, incX::Cint)::Csize_t
end

function cblas_icamax(N, X, incX)
    @ccall libxkblas.cblas_icamax(N::Cint, X::Ptr{Cvoid}, incX::Cint)::Csize_t
end

function cblas_izamax(N, X, incX)
    @ccall libxkblas.cblas_izamax(N::Cint, X::Ptr{Cvoid}, incX::Cint)::Csize_t
end

function cblas_sswap(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_sswap(N::Cint, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint)::Cvoid
end

function cblas_scopy(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_scopy(N::Cint, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint)::Cvoid
end

function cblas_saxpy(N, alpha, X, incX, Y, incY)
    @ccall libxkblas.cblas_saxpy(N::Cint, alpha::Cfloat, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint)::Cvoid
end

function cblas_dswap(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_dswap(N::Cint, X::Ptr{Cdouble}, incX::Cint, Y::Ptr{Cdouble}, incY::Cint)::Cvoid
end

function cblas_dcopy(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_dcopy(N::Cint, X::Ptr{Cdouble}, incX::Cint, Y::Ptr{Cdouble}, incY::Cint)::Cvoid
end

function cblas_daxpy(N, alpha, X, incX, Y, incY)
    @ccall libxkblas.cblas_daxpy(N::Cint, alpha::Cdouble, X::Ptr{Cdouble}, incX::Cint, Y::Ptr{Cdouble}, incY::Cint)::Cvoid
end

function cblas_cswap(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_cswap(N::Cint, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_ccopy(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_ccopy(N::Cint, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_caxpy(N, alpha, X, incX, Y, incY)
    @ccall libxkblas.cblas_caxpy(N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_zswap(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_zswap(N::Cint, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_zcopy(N, X, incX, Y, incY)
    @ccall libxkblas.cblas_zcopy(N::Cint, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_zaxpy(N, alpha, X, incX, Y, incY)
    @ccall libxkblas.cblas_zaxpy(N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_srotg(a, b, c, s)
    @ccall libxkblas.cblas_srotg(a::Ptr{Cfloat}, b::Ptr{Cfloat}, c::Ptr{Cfloat}, s::Ptr{Cfloat})::Cvoid
end

function cblas_srotmg(d1, d2, b1, b2, P)
    @ccall libxkblas.cblas_srotmg(d1::Ptr{Cfloat}, d2::Ptr{Cfloat}, b1::Ptr{Cfloat}, b2::Cfloat, P::Ptr{Cfloat})::Cvoid
end

function cblas_srot(N, X, incX, Y, incY, c, s)
    @ccall libxkblas.cblas_srot(N::Cint, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint, c::Cfloat, s::Cfloat)::Cvoid
end

function cblas_srotm(N, X, incX, Y, incY, P)
    @ccall libxkblas.cblas_srotm(N::Cint, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint, P::Ptr{Cfloat})::Cvoid
end

function cblas_drotg(a, b, c, s)
    @ccall libxkblas.cblas_drotg(a::Ptr{Cdouble}, b::Ptr{Cdouble}, c::Ptr{Cdouble}, s::Ptr{Cdouble})::Cvoid
end

function cblas_drotmg(d1, d2, b1, b2, P)
    @ccall libxkblas.cblas_drotmg(d1::Ptr{Cdouble}, d2::Ptr{Cdouble}, b1::Ptr{Cdouble}, b2::Cdouble, P::Ptr{Cdouble})::Cvoid
end

function cblas_drot(N, X, incX, Y, incY, c, s)
    @ccall libxkblas.cblas_drot(N::Cint, X::Ptr{Cdouble}, incX::Cint, Y::Ptr{Cdouble}, incY::Cint, c::Cdouble, s::Cdouble)::Cvoid
end

function cblas_drotm(N, X, incX, Y, incY, P)
    @ccall libxkblas.cblas_drotm(N::Cint, X::Ptr{Cdouble}, incX::Cint, Y::Ptr{Cdouble}, incY::Cint, P::Ptr{Cdouble})::Cvoid
end

function cblas_sscal(N, alpha, X, incX)
    @ccall libxkblas.cblas_sscal(N::Cint, alpha::Cfloat, X::Ptr{Cfloat}, incX::Cint)::Cvoid
end

function cblas_dscal(N, alpha, X, incX)
    @ccall libxkblas.cblas_dscal(N::Cint, alpha::Cdouble, X::Ptr{Cdouble}, incX::Cint)::Cvoid
end

function cblas_cscal(N, alpha, X, incX)
    @ccall libxkblas.cblas_cscal(N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_zscal(N, alpha, X, incX)
    @ccall libxkblas.cblas_zscal(N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_csscal(N, alpha, X, incX)
    @ccall libxkblas.cblas_csscal(N::Cint, alpha::Cfloat, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_zdscal(N, alpha, X, incX)
    @ccall libxkblas.cblas_zdscal(N::Cint, alpha::Cdouble, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_sgemv(order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, M::Cint, N::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, X::Ptr{Cfloat}, incX::Cint, beta::Cfloat, Y::Ptr{Cfloat}, incY::Cint)::Cvoid
end

function cblas_sgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_sgbmv(order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, M::Cint, N::Cint, KL::Cint, KU::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, X::Ptr{Cfloat}, incX::Cint, beta::Cfloat, Y::Ptr{Cfloat}, incY::Cint)::Cvoid
end

function cblas_strmv(order, Uplo, TransA, Diag, N, A, lda, X, incX)
    @ccall libxkblas.cblas_strmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, A::Ptr{Cfloat}, lda::Cint, X::Ptr{Cfloat}, incX::Cint)::Cvoid
end

function cblas_stbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)
    @ccall libxkblas.cblas_stbmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, K::Cint, A::Ptr{Cfloat}, lda::Cint, X::Ptr{Cfloat}, incX::Cint)::Cvoid
end

function cblas_stpmv(order, Uplo, TransA, Diag, N, Ap, X, incX)
    @ccall libxkblas.cblas_stpmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, Ap::Ptr{Cfloat}, X::Ptr{Cfloat}, incX::Cint)::Cvoid
end

function cblas_strsv(order, Uplo, TransA, Diag, N, A, lda, X, incX)
    @ccall libxkblas.cblas_strsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, A::Ptr{Cfloat}, lda::Cint, X::Ptr{Cfloat}, incX::Cint)::Cvoid
end

function cblas_stbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)
    @ccall libxkblas.cblas_stbsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, K::Cint, A::Ptr{Cfloat}, lda::Cint, X::Ptr{Cfloat}, incX::Cint)::Cvoid
end

function cblas_stpsv(order, Uplo, TransA, Diag, N, Ap, X, incX)
    @ccall libxkblas.cblas_stpsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, Ap::Ptr{Cfloat}, X::Ptr{Cfloat}, incX::Cint)::Cvoid
end

function cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_dgemv(order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, M::Cint, N::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, X::Ptr{Cdouble}, incX::Cint, beta::Cdouble, Y::Ptr{Cdouble}, incY::Cint)::Cvoid
end

function cblas_dgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_dgbmv(order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, M::Cint, N::Cint, KL::Cint, KU::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, X::Ptr{Cdouble}, incX::Cint, beta::Cdouble, Y::Ptr{Cdouble}, incY::Cint)::Cvoid
end

function cblas_dtrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX)
    @ccall libxkblas.cblas_dtrmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, A::Ptr{Cdouble}, lda::Cint, X::Ptr{Cdouble}, incX::Cint)::Cvoid
end

function cblas_dtbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)
    @ccall libxkblas.cblas_dtbmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, K::Cint, A::Ptr{Cdouble}, lda::Cint, X::Ptr{Cdouble}, incX::Cint)::Cvoid
end

function cblas_dtpmv(order, Uplo, TransA, Diag, N, Ap, X, incX)
    @ccall libxkblas.cblas_dtpmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, Ap::Ptr{Cdouble}, X::Ptr{Cdouble}, incX::Cint)::Cvoid
end

function cblas_dtrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX)
    @ccall libxkblas.cblas_dtrsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, A::Ptr{Cdouble}, lda::Cint, X::Ptr{Cdouble}, incX::Cint)::Cvoid
end

function cblas_dtbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)
    @ccall libxkblas.cblas_dtbsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, K::Cint, A::Ptr{Cdouble}, lda::Cint, X::Ptr{Cdouble}, incX::Cint)::Cvoid
end

function cblas_dtpsv(order, Uplo, TransA, Diag, N, Ap, X, incX)
    @ccall libxkblas.cblas_dtpsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, Ap::Ptr{Cdouble}, X::Ptr{Cdouble}, incX::Cint)::Cvoid
end

function cblas_cgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_cgemv(order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_cgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_cgbmv(order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, M::Cint, N::Cint, KL::Cint, KU::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_ctrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX)
    @ccall libxkblas.cblas_ctrmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ctbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)
    @ccall libxkblas.cblas_ctbmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, K::Cint, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ctpmv(order, Uplo, TransA, Diag, N, Ap, X, incX)
    @ccall libxkblas.cblas_ctpmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, Ap::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ctrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX)
    @ccall libxkblas.cblas_ctrsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ctbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)
    @ccall libxkblas.cblas_ctbsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, K::Cint, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ctpsv(order, Uplo, TransA, Diag, N, Ap, X, incX)
    @ccall libxkblas.cblas_ctpsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, Ap::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_zgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_zgemv(order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_zgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_zgbmv(order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, M::Cint, N::Cint, KL::Cint, KU::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_ztrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX)
    @ccall libxkblas.cblas_ztrmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ztbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)
    @ccall libxkblas.cblas_ztbmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, K::Cint, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ztpmv(order, Uplo, TransA, Diag, N, Ap, X, incX)
    @ccall libxkblas.cblas_ztpmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, Ap::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ztrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX)
    @ccall libxkblas.cblas_ztrsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ztbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX)
    @ccall libxkblas.cblas_ztbsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, K::Cint, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ztpsv(order, Uplo, TransA, Diag, N, Ap, X, incX)
    @ccall libxkblas.cblas_ztpsv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, N::Cint, Ap::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint)::Cvoid
end

function cblas_ssymv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_ssymv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, X::Ptr{Cfloat}, incX::Cint, beta::Cfloat, Y::Ptr{Cfloat}, incY::Cint)::Cvoid
end

function cblas_ssbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_ssbmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, K::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, X::Ptr{Cfloat}, incX::Cint, beta::Cfloat, Y::Ptr{Cfloat}, incY::Cint)::Cvoid
end

function cblas_sspmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_sspmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cfloat, Ap::Ptr{Cfloat}, X::Ptr{Cfloat}, incX::Cint, beta::Cfloat, Y::Ptr{Cfloat}, incY::Cint)::Cvoid
end

function cblas_sger(order, M, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_sger(order::CBLAS_ORDER, M::Cint, N::Cint, alpha::Cfloat, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

function cblas_ssyr(order, Uplo, N, alpha, X, incX, A, lda)
    @ccall libxkblas.cblas_ssyr(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cfloat, X::Ptr{Cfloat}, incX::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

function cblas_sspr(order, Uplo, N, alpha, X, incX, Ap)
    @ccall libxkblas.cblas_sspr(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cfloat, X::Ptr{Cfloat}, incX::Cint, Ap::Ptr{Cfloat})::Cvoid
end

function cblas_ssyr2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_ssyr2(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cfloat, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint, A::Ptr{Cfloat}, lda::Cint)::Cvoid
end

function cblas_sspr2(order, Uplo, N, alpha, X, incX, Y, incY, A)
    @ccall libxkblas.cblas_sspr2(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cfloat, X::Ptr{Cfloat}, incX::Cint, Y::Ptr{Cfloat}, incY::Cint, A::Ptr{Cfloat})::Cvoid
end

function cblas_dsymv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_dsymv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, X::Ptr{Cdouble}, incX::Cint, beta::Cdouble, Y::Ptr{Cdouble}, incY::Cint)::Cvoid
end

function cblas_dsbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_dsbmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, K::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, X::Ptr{Cdouble}, incX::Cint, beta::Cdouble, Y::Ptr{Cdouble}, incY::Cint)::Cvoid
end

function cblas_dspmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_dspmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cdouble, Ap::Ptr{Cdouble}, X::Ptr{Cdouble}, incX::Cint, beta::Cdouble, Y::Ptr{Cdouble}, incY::Cint)::Cvoid
end

function cblas_dger(order, M, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_dger(order::CBLAS_ORDER, M::Cint, N::Cint, alpha::Cdouble, X::Ptr{Cdouble}, incX::Cint, Y::Ptr{Cdouble}, incY::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

function cblas_dsyr(order, Uplo, N, alpha, X, incX, A, lda)
    @ccall libxkblas.cblas_dsyr(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cdouble, X::Ptr{Cdouble}, incX::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

function cblas_dspr(order, Uplo, N, alpha, X, incX, Ap)
    @ccall libxkblas.cblas_dspr(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cdouble, X::Ptr{Cdouble}, incX::Cint, Ap::Ptr{Cdouble})::Cvoid
end

function cblas_dsyr2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_dsyr2(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cdouble, X::Ptr{Cdouble}, incX::Cint, Y::Ptr{Cdouble}, incY::Cint, A::Ptr{Cdouble}, lda::Cint)::Cvoid
end

function cblas_dspr2(order, Uplo, N, alpha, X, incX, Y, incY, A)
    @ccall libxkblas.cblas_dspr2(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cdouble, X::Ptr{Cdouble}, incX::Cint, Y::Ptr{Cdouble}, incY::Cint, A::Ptr{Cdouble})::Cvoid
end

function cblas_chemv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_chemv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_chbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_chbmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_chpmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_chpmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Ptr{Cvoid}, Ap::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_cgeru(order, M, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_cgeru(order::CBLAS_ORDER, M::Cint, N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, A::Ptr{Cvoid}, lda::Cint)::Cvoid
end

function cblas_cgerc(order, M, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_cgerc(order::CBLAS_ORDER, M::Cint, N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, A::Ptr{Cvoid}, lda::Cint)::Cvoid
end

function cblas_cher(order, Uplo, N, alpha, X, incX, A, lda)
    @ccall libxkblas.cblas_cher(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cfloat, X::Ptr{Cvoid}, incX::Cint, A::Ptr{Cvoid}, lda::Cint)::Cvoid
end

function cblas_chpr(order, Uplo, N, alpha, X, incX, A)
    @ccall libxkblas.cblas_chpr(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cfloat, X::Ptr{Cvoid}, incX::Cint, A::Ptr{Cvoid})::Cvoid
end

function cblas_cher2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_cher2(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, A::Ptr{Cvoid}, lda::Cint)::Cvoid
end

function cblas_chpr2(order, Uplo, N, alpha, X, incX, Y, incY, Ap)
    @ccall libxkblas.cblas_chpr2(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, Ap::Ptr{Cvoid})::Cvoid
end

function cblas_zhemv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_zhemv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_zhbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_zhbmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_zhpmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY)
    @ccall libxkblas.cblas_zhpmv(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Ptr{Cvoid}, Ap::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, beta::Ptr{Cvoid}, Y::Ptr{Cvoid}, incY::Cint)::Cvoid
end

function cblas_zgeru(order, M, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_zgeru(order::CBLAS_ORDER, M::Cint, N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, A::Ptr{Cvoid}, lda::Cint)::Cvoid
end

function cblas_zgerc(order, M, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_zgerc(order::CBLAS_ORDER, M::Cint, N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, A::Ptr{Cvoid}, lda::Cint)::Cvoid
end

function cblas_zher(order, Uplo, N, alpha, X, incX, A, lda)
    @ccall libxkblas.cblas_zher(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cdouble, X::Ptr{Cvoid}, incX::Cint, A::Ptr{Cvoid}, lda::Cint)::Cvoid
end

function cblas_zhpr(order, Uplo, N, alpha, X, incX, A)
    @ccall libxkblas.cblas_zhpr(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Cdouble, X::Ptr{Cvoid}, incX::Cint, A::Ptr{Cvoid})::Cvoid
end

function cblas_zher2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda)
    @ccall libxkblas.cblas_zher2(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, A::Ptr{Cvoid}, lda::Cint)::Cvoid
end

function cblas_zhpr2(order, Uplo, N, alpha, X, incX, Y, incY, Ap)
    @ccall libxkblas.cblas_zhpr2(order::CBLAS_ORDER, Uplo::CBLAS_UPLO, N::Cint, alpha::Ptr{Cvoid}, X::Ptr{Cvoid}, incX::Cint, Y::Ptr{Cvoid}, incY::Cint, Ap::Ptr{Cvoid})::Cvoid
end

function cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_sgemm(Order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, TransB::CBLAS_TRANSPOSE, M::Cint, N::Cint, K::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Cfloat, C::Ptr{Cfloat}, ldc::Cint)::Cvoid
end

function cblas_ssymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_ssymm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, M::Cint, N::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Cfloat, C::Ptr{Cfloat}, ldc::Cint)::Cvoid
end

function cblas_ssyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.cblas_ssyrk(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, beta::Cfloat, C::Ptr{Cfloat}, ldc::Cint)::Cvoid
end

function cblas_ssyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_ssyr2k(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Cfloat, C::Ptr{Cfloat}, ldc::Cint)::Cvoid
end

function cblas_strmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)
    @ccall libxkblas.cblas_strmm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, M::Cint, N::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint)::Cvoid
end

function cblas_strsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)
    @ccall libxkblas.cblas_strsm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, M::Cint, N::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint)::Cvoid
end

function cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_dgemm(Order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, TransB::CBLAS_TRANSPOSE, M::Cint, N::Cint, K::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Cdouble, C::Ptr{Cdouble}, ldc::Cint)::Cvoid
end

function cblas_dsymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_dsymm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, M::Cint, N::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Cdouble, C::Ptr{Cdouble}, ldc::Cint)::Cvoid
end

function cblas_dsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.cblas_dsyrk(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, beta::Cdouble, C::Ptr{Cdouble}, ldc::Cint)::Cvoid
end

function cblas_dsyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_dsyr2k(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Cdouble, C::Ptr{Cdouble}, ldc::Cint)::Cvoid
end

function cblas_dtrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)
    @ccall libxkblas.cblas_dtrmm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, M::Cint, N::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint)::Cvoid
end

function cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)
    @ccall libxkblas.cblas_dtrsm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, M::Cint, N::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint)::Cvoid
end

function cblas_cgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_cgemm(Order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, TransB::CBLAS_TRANSPOSE, M::Cint, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_csymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_csymm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_csyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.cblas_csyrk(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_csyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_csyr2k(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_ctrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)
    @ccall libxkblas.cblas_ctrmm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint)::Cvoid
end

function cblas_ctrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)
    @ccall libxkblas.cblas_ctrsm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint)::Cvoid
end

function cblas_zgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_zgemm(Order::CBLAS_ORDER, TransA::CBLAS_TRANSPOSE, TransB::CBLAS_TRANSPOSE, M::Cint, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_zsymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_zsymm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_zsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.cblas_zsyrk(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_zsyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_zsyr2k(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_ztrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)
    @ccall libxkblas.cblas_ztrmm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint)::Cvoid
end

function cblas_ztrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb)
    @ccall libxkblas.cblas_ztrsm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, TransA::CBLAS_TRANSPOSE, Diag::CBLAS_DIAG, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint)::Cvoid
end

function cblas_chemm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_chemm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_cherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.cblas_cherk(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Cfloat, A::Ptr{Cvoid}, lda::Cint, beta::Cfloat, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_cher2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_cher2k(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Cfloat, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_zhemm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_zhemm(Order::CBLAS_ORDER, Side::CBLAS_SIDE, Uplo::CBLAS_UPLO, M::Cint, N::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_zherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc)
    @ccall libxkblas.cblas_zherk(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Cdouble, A::Ptr{Cvoid}, lda::Cint, beta::Cdouble, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

function cblas_zher2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    @ccall libxkblas.cblas_zher2k(Order::CBLAS_ORDER, Uplo::CBLAS_UPLO, Trans::CBLAS_TRANSPOSE, N::Cint, K::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, beta::Cdouble, C::Ptr{Cvoid}, ldc::Cint)::Cvoid
end

@cenum CBLAS_SPARSE::UInt32 begin
    CblasSparseCSR = 141
    CblasSparseCSC = 142
    CblasSparseCOO = 143
    CblasSparseBSR = 144
    CblasSparseELL = 145
    CblasSparseDIA = 146
end

function f2c_trans(trans)
    @ccall libxkblas.f2c_trans(trans::Cstring)::CBLAS_TRANSPOSE
end

function blas2cblas_trans(trans)
    @ccall libxkblas.xkblas_blas2cblas_trans(trans::Cstring)::Cint
end

function blas2cblas_side(side)
    @ccall libxkblas.xkblas_blas2cblas_side(side::Cstring)::Cint
end

function blas2cblas_fill(uplo)
    @ccall libxkblas.xkblas_blas2cblas_fill(uplo::Cstring)::Cint
end

function blas2cblas_diag(diag)
    @ccall libxkblas.xkblas_blas2cblas_diag(diag::Cstring)::Cint
end

const CBLAS_INDEX = size_t

FMULS_GEMV(__m, __n) = Float64(__m) * Float64(__n) + 2.0 * Float64(__m)

FADDS_GEMV(__m, __n) = Float64(__m) * Float64(__n)

FMULS_SYMV(__n) = FMULS_GEMV(__n, __n)

FADDS_SYMV(__n) = FADDS_GEMV(__n, __n)

const FMULS_HEMV = FMULS_SYMV

const FADDS_HEMV = FADDS_SYMV

FMULS_GEMM(__m, __n, __k) = Float64(__m) * Float64(__n) * Float64(__k)

FADDS_GEMM(__m, __n, __k) = Float64(__m) * Float64(__n) * Float64(__k)

FMULS_SYMM(__side, __m, __n) = if __side == CblasLeft
        FMULS_GEMM(__m, __m, __n)
    else
        FMULS_GEMM(__m, __n, __n)
    end

FADDS_SYMM(__side, __m, __n) = if __side == CblasLeft
        FADDS_GEMM(__m, __m, __n)
    else
        FADDS_GEMM(__m, __n, __n)
    end

const FMULS_HEMM = FMULS_SYMM

const FADDS_HEMM = FADDS_SYMM

FMULS_SYRK(__k, __n) = 0.5 * Float64(__k) * Float64(__n) * (Float64(__n) + 1.0)

FADDS_SYRK(__k, __n) = 0.5 * Float64(__k) * Float64(__n) * (Float64(__n) + 1.0)

const FMULS_HERK = FMULS_SYRK

const FADDS_HERK = FADDS_SYRK

FMULS_SYR2K(__k, __n) = Float64(__k) * Float64(__n) * Float64(__n)

FADDS_SYR2K(__k, __n) = Float64(__k) * Float64(__n) * Float64(__n) + Float64(__n)

const FMULS_HER2K = FMULS_SYR2K

const FADDS_HER2K = FADDS_SYR2K

FMULS_TRMM_2(__m, __n) = 0.5 * Float64(__n) * Float64(__m) * (Float64(__m) + 1.0)

FADDS_TRMM_2(__m, __n) = 0.5 * Float64(__n) * Float64(__m) * (Float64(__m) - 1.0)

FMULS_TRMM(__side, __m, __n) = if __side == CblasLeft
        FMULS_TRMM_2(__m, __n)
    else
        FMULS_TRMM_2(__n, __m)
    end

FADDS_TRMM(__side, __m, __n) = if __side == CblasLeft
        FADDS_TRMM_2(__m, __n)
    else
        FADDS_TRMM_2(__n, __m)
    end

const FMULS_TRSM = FMULS_TRMM

const FADDS_TRSM = FMULS_TRMM

FMULS_GETRF(__m, __n) = if __m < __n
        0.5 * Float64(__m) * (Float64(__m) * ((Float64(__n) - (1.0 ÷ 3.0) * __m) - 1.0) + Float64(__n)) + (2.0 ÷ 3.0) * __m
    else
        0.5 * Float64(__n) * (Float64(__n) * ((Float64(__m) - (1.0 ÷ 3.0) * __n) - 1.0) + Float64(__m)) + (2.0 ÷ 3.0) * __n
    end

FADDS_GETRF(__m, __n) = if __m < __n
        0.5 * Float64(__m) * (Float64(__m) * (Float64(__n) - (1.0 ÷ 3.0) * __m) - Float64(__n)) + (1.0 ÷ 6.0) * __m
    else
        0.5 * Float64(__n) * (Float64(__n) * (Float64(__m) - (1.0 ÷ 3.0) * __n) - Float64(__m)) + (1.0 ÷ 6.0) * __n
    end

FMULS_GETRI(__n) = Float64(__n) * (5.0 ÷ 6.0 + Float64(__n) * ((2.0 ÷ 3.0) * Float64(__n) + 0.5))

FADDS_GETRI(__n) = Float64(__n) * (5.0 ÷ 6.0 + Float64(__n) * ((2.0 ÷ 3.0) * Float64(__n) - 1.5))

FMULS_GETRS(__n, __nrhs) = Float64(__nrhs) * Float64(__n) * Float64(__n)

FADDS_GETRS(__n, __nrhs) = Float64(__nrhs) * Float64(__n) * (Float64(__n) - 1.0)

FMULS_POTRF(__n) = Float64(__n) * (((1.0 ÷ 6.0) * Float64(__n) + 0.5) * Float64(__n) + 1.0 ÷ 3.0)

FADDS_POTRF(__n) = Float64(__n) * (((1.0 ÷ 6.0) * Float64(__n)) * Float64(__n) - 1.0 ÷ 6.0)

FMULS_POTRI(__n) = Float64(__n) * (2.0 ÷ 3.0 + Float64(__n) * ((1.0 ÷ 3.0) * Float64(__n) + 1.0))

FADDS_POTRI(__n) = Float64(__n) * (1.0 ÷ 6.0 + Float64(__n) * ((1.0 ÷ 3.0) * Float64(__n) - 0.5))

FMULS_POTRS(__n, __nrhs) = Float64(__nrhs) * Float64(__n) * (Float64(__n) + 1.0)

FADDS_POTRS(__n, __nrhs) = Float64(__nrhs) * Float64(__n) * (Float64(__n) - 1.0)

FMULS_SYTRF(__n) = Float64(__n) * (((1.0 ÷ 6.0) * Float64(__n) + 0.5) * Float64(__n) + 1.0 ÷ 3.0)

FADDS_SYTRF(__n) = Float64(__n) * (((1.0 ÷ 6.0) * Float64(__n)) * Float64(__n) - 1.0 ÷ 6.0)

FMULS_GEQRF(__m, __n) = if __m > __n
        Float64(__n) * (Float64(__n) * ((0.5 - (1.0 ÷ 3.0) * Float64(__n)) + Float64(__m)) + Float64(__m) + 23.0 ÷ 6.0)
    else
        Float64(__m) * (Float64(__m) * ((-0.5 - (1.0 ÷ 3.0) * Float64(__m)) + Float64(__n)) + 2.0 * Float64(__n) + 23.0 ÷ 6.0)
    end

FADDS_GEQRF(__m, __n) = if __m > __n
        Float64(__n) * (Float64(__n) * ((0.5 - (1.0 ÷ 3.0) * Float64(__n)) + Float64(__m)) + 5.0 ÷ 6.0)
    else
        Float64(__m) * (Float64(__m) * ((-0.5 - (1.0 ÷ 3.0) * Float64(__m)) + Float64(__n)) + Float64(__n) + 5.0 ÷ 6.0)
    end

FMULS_GEQLF(__m, __n) = FMULS_GEQRF(__m, __n)

FADDS_GEQLF(__m, __n) = FADDS_GEQRF(__m, __n)

FMULS_GERQF(__m, __n) = if __m > __n
        Float64(__n) * (Float64(__n) * ((0.5 - (1.0 ÷ 3.0) * Float64(__n)) + Float64(__m)) + Float64(__m) + 29.0 ÷ 6.0)
    else
        Float64(__m) * (Float64(__m) * ((-0.5 - (1.0 ÷ 3.0) * Float64(__m)) + Float64(__n)) + 2.0 * Float64(__n) + 29.0 ÷ 6.0)
    end

FADDS_GERQF(__m, __n) = if __m > __n
        Float64(__n) * (Float64(__n) * ((-0.5 - (1.0 ÷ 3.0) * Float64(__n)) + Float64(__m)) + Float64(__m) + 5.0 ÷ 6.0)
    else
        Float64(__m) * (Float64(__m) * ((0.5 - (1.0 ÷ 3.0) * Float64(__m)) + Float64(__n)) + +5.0 ÷ 6.0)
    end

FMULS_GELQF(__m, __n) = FMULS_GERQF(__m, __n)

FADDS_GELQF(__m, __n) = FADDS_GERQF(__m, __n)

FMULS_UNGQR(__m, __n, __k) = Float64(__k) * (((2.0 * Float64(__m) * Float64(__n) + 2.0 * Float64(__n)) - 5.0 ÷ 3.0) + Float64(__k) * (((2.0 ÷ 3.0) * Float64(__k) - (Float64(__m) + Float64(__n))) - 1.0))

FADDS_UNGQR(__m, __n, __k) = Float64(__k) * (((2.0 * Float64(__m) * Float64(__n) + Float64(__n)) - Float64(__m)) + 1.0 ÷ 3.0 + Float64(__k) * ((2.0 ÷ 3.0) * Float64(__k) - (Float64(__m) + Float64(__n))))

const FMULS_UNGQL = FMULS_UNGQR

const FMULS_ORGQR = FMULS_UNGQR

const FMULS_ORGQL = FMULS_UNGQR

const FADDS_UNGQL = FADDS_UNGQR

const FADDS_ORGQR = FADDS_UNGQR

const FADDS_ORGQL = FADDS_UNGQR

FMULS_UNGRQ(__m, __n, __k) = Float64(__k) * (((2.0 * Float64(__m) * Float64(__n) + Float64(__m) + Float64(__n)) - 2.0 ÷ 3.0) + Float64(__k) * (((2.0 ÷ 3.0) * Float64(__k) - (Float64(__m) + Float64(__n))) - 1.0))

FADDS_UNGRQ(__m, __n, __k) = Float64(__k) * (((2.0 * Float64(__m) * Float64(__n) + Float64(__m)) - Float64(__n)) + 1.0 ÷ 3.0 + Float64(__k) * ((2.0 ÷ 3.0) * Float64(__k) - (Float64(__m) + Float64(__n))))

const FMULS_UNGLQ = FMULS_UNGRQ

const FMULS_ORGRQ = FMULS_UNGRQ

const FMULS_ORGLQ = FMULS_UNGRQ

const FADDS_UNGLQ = FADDS_UNGRQ

const FADDS_ORGRQ = FADDS_UNGRQ

const FADDS_ORGLQ = FADDS_UNGRQ

FMULS_GEQRS(__m, __n, __nrhs) = Float64(__nrhs) * (Float64(__n) * ((2.0 * Float64(__m) - 0.5 * Float64(__n)) + 2.5))

FADDS_GEQRS(__m, __n, __nrhs) = Float64(__nrhs) * (Float64(__n) * ((2.0 * Float64(__m) - 0.5 * Float64(__n)) + 0.5))

FMULS_TRTRI(__n) = Float64(__n) * (Float64(__n) * ((1.0 ÷ 6.0) * Float64(__n) + 0.5) + 1.0 ÷ 3.0)

FADDS_TRTRI(__n) = Float64(__n) * (Float64(__n) * ((1.0 ÷ 6.0) * Float64(__n) - 0.5) + 1.0 ÷ 3.0)

FMULS_GEHRD(__n) = Float64(__n) * (Float64(__n) * ((5.0 ÷ 3.0) * Float64(__n) + 0.5) - 7.0 ÷ 6.0) - 13.0

FADDS_GEHRD(__n) = Float64(__n) * (Float64(__n) * ((5.0 ÷ 3.0) * Float64(__n) - 1.0) - 2.0 ÷ 3.0) - 8.0

FMULS_SYTRD(__n) = Float64(__n) * (Float64(__n) * ((2.0 ÷ 3.0) * Float64(__n) + 2.5) - 1.0 ÷ 6.0) - 15.0

FADDS_SYTRD(__n) = Float64(__n) * (Float64(__n) * ((2.0 ÷ 3.0) * Float64(__n) + 1.0) - 8.0 ÷ 3.0) - 4.0

const FMULS_HETRD = FMULS_SYTRD

const FADDS_HETRD = FADDS_SYTRD

FMULS_GEBRD(__m, __n) = if __m >= __n
        Float64(__n) * (Float64(__n) * ((2.0 * Float64(__m) - (2.0 ÷ 3.0) * Float64(__n)) + 2.0) + 20.0 ÷ 3.0)
    else
        Float64(__m) * (Float64(__m) * ((2.0 * Float64(__n) - (2.0 ÷ 3.0) * Float64(__m)) + 2.0) + 20.0 ÷ 3.0)
    end

FADDS_GEBRD(__m, __n) = if __m >= __n
        Float64(__n) * ((Float64(__n) * ((2.0 * Float64(__m) - (2.0 ÷ 3.0) * Float64(__n)) + 1.0) - Float64(__m)) + 5.0 ÷ 3.0)
    else
        Float64(__m) * ((Float64(__m) * ((2.0 * Float64(__n) - (2.0 ÷ 3.0) * Float64(__m)) + 1.0) - Float64(__n)) + 5.0 ÷ 3.0)
    end

FLOPS_ZGEMV(__m, __n) = 6.0 * FMULS_GEMV(__m, __n) + 2.0 * FADDS_GEMV(__m, __n)

FLOPS_CGEMV(__m, __n) = 6.0 * FMULS_GEMV(__m, __n) + 2.0 * FADDS_GEMV(__m, __n)

FLOPS_DGEMV(__m, __n) = FMULS_GEMV(__m, __n) + FADDS_GEMV(__m, __n)

FLOPS_SGEMV(__m, __n) = FMULS_GEMV(__m, __n) + FADDS_GEMV(__m, __n)

FLOPS_ZHEMV(__n) = 6.0 * FMULS_HEMV(__n) + 2.0 * FADDS_HEMV(__n)

FLOPS_CHEMV(__n) = 6.0 * FMULS_HEMV(__n) + 2.0 * FADDS_HEMV(__n)

FLOPS_ZSYMV(__n) = 6.0 * FMULS_SYMV(__n) + 2.0 * FADDS_SYMV(__n)

FLOPS_CSYMV(__n) = 6.0 * FMULS_SYMV(__n) + 2.0 * FADDS_SYMV(__n)

FLOPS_DSYMV(__n) = FMULS_SYMV(__n) + FADDS_SYMV(__n)

FLOPS_SSYMV(__n) = FMULS_SYMV(__n) + FADDS_SYMV(__n)

FLOPS_ZGEMM(__m, __n, __k) = 6.0 * FMULS_GEMM(__m, __n, __k) + 2.0 * FADDS_GEMM(__m, __n, __k)

FLOPS_CGEMM(__m, __n, __k) = 6.0 * FMULS_GEMM(__m, __n, __k) + 2.0 * FADDS_GEMM(__m, __n, __k)

FLOPS_DGEMM(__m, __n, __k) = FMULS_GEMM(__m, __n, __k) + FADDS_GEMM(__m, __n, __k)

FLOPS_SGEMM(__m, __n, __k) = FMULS_GEMM(__m, __n, __k) + FADDS_GEMM(__m, __n, __k)

FLOPS_ZHEMM(__side, __m, __n) = 6.0 * FMULS_HEMM(__side, __m, __n) + 2.0 * FADDS_HEMM(__side, __m, __n)

FLOPS_CHEMM(__side, __m, __n) = 6.0 * FMULS_HEMM(__side, __m, __n) + 2.0 * FADDS_HEMM(__side, __m, __n)

FLOPS_ZSYMM(__side, __m, __n) = 6.0 * FMULS_SYMM(__side, __m, __n) + 2.0 * FADDS_SYMM(__side, __m, __n)

FLOPS_CSYMM(__side, __m, __n) = 6.0 * FMULS_SYMM(__side, __m, __n) + 2.0 * FADDS_SYMM(__side, __m, __n)

FLOPS_DSYMM(__side, __m, __n) = FMULS_SYMM(__side, __m, __n) + FADDS_SYMM(__side, __m, __n)

FLOPS_SSYMM(__side, __m, __n) = FMULS_SYMM(__side, __m, __n) + FADDS_SYMM(__side, __m, __n)

FLOPS_ZHERK(__k, __n) = 6.0 * FMULS_HERK(__k, __n) + 2.0 * FADDS_HERK(__k, __n)

FLOPS_CHERK(__k, __n) = 6.0 * FMULS_HERK(__k, __n) + 2.0 * FADDS_HERK(__k, __n)

FLOPS_ZSYRK(__k, __n) = 6.0 * FMULS_SYRK(__k, __n) + 2.0 * FADDS_SYRK(__k, __n)

FLOPS_CSYRK(__k, __n) = 6.0 * FMULS_SYRK(__k, __n) + 2.0 * FADDS_SYRK(__k, __n)

FLOPS_DSYRK(__k, __n) = FMULS_SYRK(__k, __n) + FADDS_SYRK(__k, __n)

FLOPS_SSYRK(__k, __n) = FMULS_SYRK(__k, __n) + FADDS_SYRK(__k, __n)

FLOPS_ZHER2K(__k, __n) = 6.0 * FMULS_HER2K(__k, __n) + 2.0 * FADDS_HER2K(__k, __n)

FLOPS_CHER2K(__k, __n) = 6.0 * FMULS_HER2K(__k, __n) + 2.0 * FADDS_HER2K(__k, __n)

FLOPS_ZSYR2K(__k, __n) = 6.0 * FMULS_SYR2K(__k, __n) + 2.0 * FADDS_SYR2K(__k, __n)

FLOPS_CSYR2K(__k, __n) = 6.0 * FMULS_SYR2K(__k, __n) + 2.0 * FADDS_SYR2K(__k, __n)

FLOPS_DSYR2K(__k, __n) = FMULS_SYR2K(__k, __n) + FADDS_SYR2K(__k, __n)

FLOPS_SSYR2K(__k, __n) = FMULS_SYR2K(__k, __n) + FADDS_SYR2K(__k, __n)

FLOPS_ZTRMM(__side, __m, __n) = 6.0 * FMULS_TRMM(__side, __m, __n) + 2.0 * FADDS_TRMM(__side, __m, __n)

FLOPS_CTRMM(__side, __m, __n) = 6.0 * FMULS_TRMM(__side, __m, __n) + 2.0 * FADDS_TRMM(__side, __m, __n)

FLOPS_DTRMM(__side, __m, __n) = FMULS_TRMM(__side, __m, __n) + FADDS_TRMM(__side, __m, __n)

FLOPS_STRMM(__side, __m, __n) = FMULS_TRMM(__side, __m, __n) + FADDS_TRMM(__side, __m, __n)

FLOPS_ZTRSM(__side, __m, __n) = 6.0 * FMULS_TRSM(__side, __m, __n) + 2.0 * FADDS_TRSM(__side, __m, __n)

FLOPS_CTRSM(__side, __m, __n) = 6.0 * FMULS_TRSM(__side, __m, __n) + 2.0 * FADDS_TRSM(__side, __m, __n)

FLOPS_DTRSM(__side, __m, __n) = FMULS_TRSM(__side, __m, __n) + FADDS_TRSM(__side, __m, __n)

FLOPS_STRSM(__side, __m, __n) = FMULS_TRSM(__side, __m, __n) + FADDS_TRSM(__side, __m, __n)

FLOPS_ZGETRF(__m, __n) = 6.0 * FMULS_GETRF(__m, __n) + 2.0 * FADDS_GETRF(__m, __n)

FLOPS_CGETRF(__m, __n) = 6.0 * FMULS_GETRF(__m, __n) + 2.0 * FADDS_GETRF(__m, __n)

FLOPS_DGETRF(__m, __n) = FMULS_GETRF(__m, __n) + FADDS_GETRF(__m, __n)

FLOPS_SGETRF(__m, __n) = FMULS_GETRF(__m, __n) + FADDS_GETRF(__m, __n)

FLOPS_ZGETRI(__n) = 6.0 * FMULS_GETRI(__n) + 2.0 * FADDS_GETRI(__n)

FLOPS_CGETRI(__n) = 6.0 * FMULS_GETRI(__n) + 2.0 * FADDS_GETRI(__n)

FLOPS_DGETRI(__n) = FMULS_GETRI(__n) + FADDS_GETRI(__n)

FLOPS_SGETRI(__n) = FMULS_GETRI(__n) + FADDS_GETRI(__n)

FLOPS_ZGETRS(__n, __nrhs) = 6.0 * FMULS_GETRS(__n, __nrhs) + 2.0 * FADDS_GETRS(__n, __nrhs)

FLOPS_CGETRS(__n, __nrhs) = 6.0 * FMULS_GETRS(__n, __nrhs) + 2.0 * FADDS_GETRS(__n, __nrhs)

FLOPS_DGETRS(__n, __nrhs) = FMULS_GETRS(__n, __nrhs) + FADDS_GETRS(__n, __nrhs)

FLOPS_SGETRS(__n, __nrhs) = FMULS_GETRS(__n, __nrhs) + FADDS_GETRS(__n, __nrhs)

FLOPS_ZPOTRF(__n) = 6.0 * FMULS_POTRF(__n) + 2.0 * FADDS_POTRF(__n)

FLOPS_CPOTRF(__n) = 6.0 * FMULS_POTRF(__n) + 2.0 * FADDS_POTRF(__n)

FLOPS_DPOTRF(__n) = FMULS_POTRF(__n) + FADDS_POTRF(__n)

FLOPS_SPOTRF(__n) = FMULS_POTRF(__n) + FADDS_POTRF(__n)

FLOPS_ZPOTRI(__n) = 6.0 * FMULS_POTRI(__n) + 2.0 * FADDS_POTRI(__n)

FLOPS_CPOTRI(__n) = 6.0 * FMULS_POTRI(__n) + 2.0 * FADDS_POTRI(__n)

FLOPS_DPOTRI(__n) = FMULS_POTRI(__n) + FADDS_POTRI(__n)

FLOPS_SPOTRI(__n) = FMULS_POTRI(__n) + FADDS_POTRI(__n)

FLOPS_ZPOTRS(__n, __nrhs) = 6.0 * FMULS_POTRS(__n, __nrhs) + 2.0 * FADDS_POTRS(__n, __nrhs)

FLOPS_CPOTRS(__n, __nrhs) = 6.0 * FMULS_POTRS(__n, __nrhs) + 2.0 * FADDS_POTRS(__n, __nrhs)

FLOPS_DPOTRS(__n, __nrhs) = FMULS_POTRS(__n, __nrhs) + FADDS_POTRS(__n, __nrhs)

FLOPS_SPOTRS(__n, __nrhs) = FMULS_POTRS(__n, __nrhs) + FADDS_POTRS(__n, __nrhs)

FLOPS_ZGEQRF(__m, __n) = 6.0 * FMULS_GEQRF(__m, __n) + 2.0 * FADDS_GEQRF(__m, __n)

FLOPS_CGEQRF(__m, __n) = 6.0 * FMULS_GEQRF(__m, __n) + 2.0 * FADDS_GEQRF(__m, __n)

FLOPS_DGEQRF(__m, __n) = FMULS_GEQRF(__m, __n) + FADDS_GEQRF(__m, __n)

FLOPS_SGEQRF(__m, __n) = FMULS_GEQRF(__m, __n) + FADDS_GEQRF(__m, __n)

FLOPS_ZGEQLF(__m, __n) = 6.0 * FMULS_GEQLF(__m, __n) + 2.0 * FADDS_GEQLF(__m, __n)

FLOPS_CGEQLF(__m, __n) = 6.0 * FMULS_GEQLF(__m, __n) + 2.0 * FADDS_GEQLF(__m, __n)

FLOPS_DGEQLF(__m, __n) = FMULS_GEQLF(__m, __n) + FADDS_GEQLF(__m, __n)

FLOPS_SGEQLF(__m, __n) = FMULS_GEQLF(__m, __n) + FADDS_GEQLF(__m, __n)

FLOPS_ZGERQF(__m, __n) = 6.0 * FMULS_GERQF(__m, __n) + 2.0 * FADDS_GERQF(__m, __n)

FLOPS_CGERQF(__m, __n) = 6.0 * FMULS_GERQF(__m, __n) + 2.0 * FADDS_GERQF(__m, __n)

FLOPS_DGERQF(__m, __n) = FMULS_GERQF(__m, __n) + FADDS_GERQF(__m, __n)

FLOPS_SGERQF(__m, __n) = FMULS_GERQF(__m, __n) + FADDS_GERQF(__m, __n)

FLOPS_ZGELQF(__m, __n) = 6.0 * FMULS_GELQF(__m, __n) + 2.0 * FADDS_GELQF(__m, __n)

FLOPS_CGELQF(__m, __n) = 6.0 * FMULS_GELQF(__m, __n) + 2.0 * FADDS_GELQF(__m, __n)

FLOPS_DGELQF(__m, __n) = FMULS_GELQF(__m, __n) + FADDS_GELQF(__m, __n)

FLOPS_SGELQF(__m, __n) = FMULS_GELQF(__m, __n) + FADDS_GELQF(__m, __n)

FLOPS_ZUNGQR(__m, __n, __k) = 6.0 * FMULS_UNGQR(__m, __n, __k) + 2.0 * FADDS_UNGQR(__m, __n, __k)

FLOPS_CUNGQR(__m, __n, __k) = 6.0 * FMULS_UNGQR(__m, __n, __k) + 2.0 * FADDS_UNGQR(__m, __n, __k)

FLOPS_DUNGQR(__m, __n, __k) = FMULS_UNGQR(__m, __n, __k) + FADDS_UNGQR(__m, __n, __k)

FLOPS_SUNGQR(__m, __n, __k) = FMULS_UNGQR(__m, __n, __k) + FADDS_UNGQR(__m, __n, __k)

FLOPS_ZUNGQL(__m, __n, __k) = 6.0 * FMULS_UNGQL(__m, __n, __k) + 2.0 * FADDS_UNGQL(__m, __n, __k)

FLOPS_CUNGQL(__m, __n, __k) = 6.0 * FMULS_UNGQL(__m, __n, __k) + 2.0 * FADDS_UNGQL(__m, __n, __k)

FLOPS_DUNGQL(__m, __n, __k) = FMULS_UNGQL(__m, __n, __k) + FADDS_UNGQL(__m, __n, __k)

FLOPS_SUNGQL(__m, __n, __k) = FMULS_UNGQL(__m, __n, __k) + FADDS_UNGQL(__m, __n, __k)

FLOPS_ZORGQR(__m, __n, __k) = 6.0 * FMULS_ORGQR(__m, __n, __k) + 2.0 * FADDS_ORGQR(__m, __n, __k)

FLOPS_CORGQR(__m, __n, __k) = 6.0 * FMULS_ORGQR(__m, __n, __k) + 2.0 * FADDS_ORGQR(__m, __n, __k)

FLOPS_DORGQR(__m, __n, __k) = FMULS_ORGQR(__m, __n, __k) + FADDS_ORGQR(__m, __n, __k)

FLOPS_SORGQR(__m, __n, __k) = FMULS_ORGQR(__m, __n, __k) + FADDS_ORGQR(__m, __n, __k)

FLOPS_ZORGQL(__m, __n, __k) = 6.0 * FMULS_ORGQL(__m, __n, __k) + 2.0 * FADDS_ORGQL(__m, __n, __k)

FLOPS_CORGQL(__m, __n, __k) = 6.0 * FMULS_ORGQL(__m, __n, __k) + 2.0 * FADDS_ORGQL(__m, __n, __k)

FLOPS_DORGQL(__m, __n, __k) = FMULS_ORGQL(__m, __n, __k) + FADDS_ORGQL(__m, __n, __k)

FLOPS_SORGQL(__m, __n, __k) = FMULS_ORGQL(__m, __n, __k) + FADDS_ORGQL(__m, __n, __k)

FLOPS_ZUNGRQ(__m, __n, __k) = 6.0 * FMULS_UNGRQ(__m, __n, __k) + 2.0 * FADDS_UNGRQ(__m, __n, __k)

FLOPS_CUNGRQ(__m, __n, __k) = 6.0 * FMULS_UNGRQ(__m, __n, __k) + 2.0 * FADDS_UNGRQ(__m, __n, __k)

FLOPS_DUNGRQ(__m, __n, __k) = FMULS_UNGRQ(__m, __n, __k) + FADDS_UNGRQ(__m, __n, __k)

FLOPS_SUNGRQ(__m, __n, __k) = FMULS_UNGRQ(__m, __n, __k) + FADDS_UNGRQ(__m, __n, __k)

FLOPS_ZUNGLQ(__m, __n, __k) = 6.0 * FMULS_UNGLQ(__m, __n, __k) + 2.0 * FADDS_UNGLQ(__m, __n, __k)

FLOPS_CUNGLQ(__m, __n, __k) = 6.0 * FMULS_UNGLQ(__m, __n, __k) + 2.0 * FADDS_UNGLQ(__m, __n, __k)

FLOPS_DUNGLQ(__m, __n, __k) = FMULS_UNGLQ(__m, __n, __k) + FADDS_UNGLQ(__m, __n, __k)

FLOPS_SUNGLQ(__m, __n, __k) = FMULS_UNGLQ(__m, __n, __k) + FADDS_UNGLQ(__m, __n, __k)

FLOPS_ZORGRQ(__m, __n, __k) = 6.0 * FMULS_ORGRQ(__m, __n, __k) + 2.0 * FADDS_ORGRQ(__m, __n, __k)

FLOPS_CORGRQ(__m, __n, __k) = 6.0 * FMULS_ORGRQ(__m, __n, __k) + 2.0 * FADDS_ORGRQ(__m, __n, __k)

FLOPS_DORGRQ(__m, __n, __k) = FMULS_ORGRQ(__m, __n, __k) + FADDS_ORGRQ(__m, __n, __k)

FLOPS_SORGRQ(__m, __n, __k) = FMULS_ORGRQ(__m, __n, __k) + FADDS_ORGRQ(__m, __n, __k)

FLOPS_ZORGLQ(__m, __n, __k) = 6.0 * FMULS_ORGLQ(__m, __n, __k) + 2.0 * FADDS_ORGLQ(__m, __n, __k)

FLOPS_CORGLQ(__m, __n, __k) = 6.0 * FMULS_ORGLQ(__m, __n, __k) + 2.0 * FADDS_ORGLQ(__m, __n, __k)

FLOPS_DORGLQ(__m, __n, __k) = FMULS_ORGLQ(__m, __n, __k) + FADDS_ORGLQ(__m, __n, __k)

FLOPS_SORGLQ(__m, __n, __k) = FMULS_ORGLQ(__m, __n, __k) + FADDS_ORGLQ(__m, __n, __k)

FLOPS_ZGEQRS(__m, __n, __nrhs) = 6.0 * FMULS_GEQRS(__m, __n, __nrhs) + 2.0 * FADDS_GEQRS(__m, __n, __nrhs)

FLOPS_CGEQRS(__m, __n, __nrhs) = 6.0 * FMULS_GEQRS(__m, __n, __nrhs) + 2.0 * FADDS_GEQRS(__m, __n, __nrhs)

FLOPS_DGEQRS(__m, __n, __nrhs) = FMULS_GEQRS(__m, __n, __nrhs) + FADDS_GEQRS(__m, __n, __nrhs)

FLOPS_SGEQRS(__m, __n, __nrhs) = FMULS_GEQRS(__m, __n, __nrhs) + FADDS_GEQRS(__m, __n, __nrhs)

FLOPS_ZTRTRI(__n) = 6.0 * FMULS_TRTRI(__n) + 2.0 * FADDS_TRTRI(__n)

FLOPS_CTRTRI(__n) = 6.0 * FMULS_TRTRI(__n) + 2.0 * FADDS_TRTRI(__n)

FLOPS_DTRTRI(__n) = FMULS_TRTRI(__n) + FADDS_TRTRI(__n)

FLOPS_STRTRI(__n) = FMULS_TRTRI(__n) + FADDS_TRTRI(__n)

FLOPS_ZGEHRD(__n) = 6.0 * FMULS_GEHRD(__n) + 2.0 * FADDS_GEHRD(__n)

FLOPS_CGEHRD(__n) = 6.0 * FMULS_GEHRD(__n) + 2.0 * FADDS_GEHRD(__n)

FLOPS_DGEHRD(__n) = FMULS_GEHRD(__n) + FADDS_GEHRD(__n)

FLOPS_SGEHRD(__n) = FMULS_GEHRD(__n) + FADDS_GEHRD(__n)

FLOPS_ZHETRD(__n) = 6.0 * FMULS_HETRD(__n) + 2.0 * FADDS_HETRD(__n)

FLOPS_CHETRD(__n) = 6.0 * FMULS_HETRD(__n) + 2.0 * FADDS_HETRD(__n)

FLOPS_ZSYTRD(__n) = 6.0 * FMULS_SYTRD(__n) + 2.0 * FADDS_SYTRD(__n)

FLOPS_CSYTRD(__n) = 6.0 * FMULS_SYTRD(__n) + 2.0 * FADDS_SYTRD(__n)

FLOPS_DSYTRD(__n) = FMULS_SYTRD(__n) + FADDS_SYTRD(__n)

FLOPS_SSYTRD(__n) = FMULS_SYTRD(__n) + FADDS_SYTRD(__n)

FLOPS_ZGEBRD(__m, __n) = 6.0 * FMULS_GEBRD(__m, __n) + 2.0 * FADDS_GEBRD(__m, __n)

FLOPS_CGEBRD(__m, __n) = 6.0 * FMULS_GEBRD(__m, __n) + 2.0 * FADDS_GEBRD(__m, __n)

FLOPS_DGEBRD(__m, __n) = FMULS_GEBRD(__m, __n) + FADDS_GEBRD(__m, __n)

FLOPS_SGEBRD(__m, __n) = FMULS_GEBRD(__m, __n) + FADDS_GEBRD(__m, __n)

FMULS_LANGE(__m, __n) = Float64(__m) * Float64(__n)

FADDS_LANGE(__m, __n) = Float64(__m) * Float64(__n)

DATA_MAT(m, n) = Float64(m) * Float64(n)

DATA__GEMM(m, n, k) = DATA_MAT(m, n) + DATA_MAT(m, k) + DATA_MAT(k, n)

# Skipping MacroDefinition: DATA_ZGEMM ( m , n , k ) ( 1.0 * sizeof ( double _Complex ) * DATA__GEMM ( ( m ) , ( n ) , ( k ) ) )

# Skipping MacroDefinition: DATA_CGEMM ( m , n , k ) ( 1.0 * sizeof ( float _Complex ) * DATA__GEMM ( ( m ) , ( n ) , ( k ) ) )

DATA_DGEMM(m, n, k) = 1.0 * sizeof(Float64) * DATA__GEMM(m, n, k)

DATA_SGEMM(m, n, k) = 1.0 * sizeof(Float32) * DATA__GEMM(m, n, k)

FLOPS_ZGEMMT(n, k) = 0.5 * FLOPS_ZGEMM(n, n, k)

FLOPS_CGEMMT(n, k) = 0.5 * FLOPS_CGEMM(n, n, k)

FLOPS_DGEMMT(n, k) = 0.5 * FLOPS_DGEMM(n, n, k)

FLOPS_SGEMMT(n, k) = 0.5 * FLOPS_SGEMM(n, n, k)

DATA__GEMMT(n, k) = 0.5 * DATA_MAT(n, n) + DATA_MAT(n, k) + DATA_MAT(k, n)

# Skipping MacroDefinition: DATA_ZGEMMT ( n , k ) ( 1.0 * sizeof ( double _Complex ) * DATA__GEMMT ( ( n ) , ( k ) ) )

# Skipping MacroDefinition: DATA_CGEMMT ( n , k ) ( 1.0 * sizeof ( float _Complex ) * DATA__GEMMT ( ( n ) , ( k ) ) )

DATA_DGEMMT(n, k) = 1.0 * sizeof(Float64) * DATA__GEMMT(n, k)

DATA_SGEMMT(n, k) = 1.0 * sizeof(Float32) * DATA__GEMMT(n, k)

FLOPS_ZCOPYSCALE(m, n) = 4.0 * m * n

FLOPS_CCOPYSCALE(m, n) = 4.0 * m * n

FLOPS_DCOPYSCALE(m, n) = 1.0 * m * n

FLOPS_SCOPYSCALE(m, n) = 1.0 * m * n

DATA__COPYSCALE(m, n) = DATA_MAT(n, n) + DATA_MAT(n, m) + DATA_MAT(m, n)

# Skipping MacroDefinition: DATA_ZCOPYSCALE ( m , n ) ( 1.0 * sizeof ( double _Complex ) * DATA__COPYSCALE ( ( n ) , ( m ) ) )

# Skipping MacroDefinition: DATA_CCOPYSCALE ( m , n ) ( 1.0 * sizeof ( float _Complex ) * DATA__COPYSCALE ( ( n ) , ( m ) ) )

DATA_DCOPYSCALE(m, n) = 1.0 * sizeof(Float64) * DATA__COPYSCALE(n, m)

DATA_SCOPYSCALE(m, n) = 1.0 * sizeof(Float32) * DATA__COPYSCALE(n, m)

DATA__TRSM(s, m, n) = if s == CblasLeft
        0.5 * DATA_MAT(m, m) + 2 * DATA_MAT(m, n)
    else
        0.5 * DATA_MAT(n, n) + 2 * DATA_MAT(n, m)
    end

# Skipping MacroDefinition: DATA_ZTRSM ( s , m , n ) ( 1.0 * sizeof ( double _Complex ) * DATA__TRSM ( ( s ) , ( m ) , ( n ) ) )

# Skipping MacroDefinition: DATA_CTRSM ( s , m , n ) ( 1.0 * sizeof ( float _Complex ) * DATA__TRSM ( ( s ) , ( m ) , ( n ) ) )

DATA_DTRSM(s, m, n) = 1.0 * sizeof(Float64) * DATA__TRSM(s, m, n)

DATA_STRSM(s, m, n) = 1.0 * sizeof(Float32) * DATA__TRSM(s, m, n)

const DATA_ZTRMM = DATA_ZTRSM

const DATA_CTRMM = DATA_CTRSM

const DATA_DTRMM = DATA_DTRSM

const DATA_STRMM = DATA_STRSM

DATA_ZSYMM(s, m, n) = if s == CblasLeft
        DATA_ZGEMM(m, m, n)
    else
        DATA_ZGEMM(m, n, n)
    end

DATA_CSYMM(s, m, n) = if s == CblasLeft
        DATA_CGEMM(m, m, n)
    else
        DATA_CGEMM(m, n, n)
    end

DATA_DSYMM(s, m, n) = if s == CblasLeft
        DATA_DGEMM(m, m, n)
    else
        DATA_DGEMM(m, n, n)
    end

DATA_SSYMM(s, m, n) = if s == CblasLeft
        DATA_SGEMM(m, m, n)
    else
        DATA_SGEMM(m, n, n)
    end

DATA__SYRK(n, k) = 0.5 * DATA_MAT(n, n) + DATA_MAT(n, k)

# Skipping MacroDefinition: DATA_ZSYRK ( n , k ) ( 1.0 * sizeof ( double _Complex ) * DATA__SYRK ( ( n ) , ( k ) ) )

# Skipping MacroDefinition: DATA_CSYRK ( n , k ) ( 1.0 * sizeof ( float _Complex ) * DATA__SYRK ( ( n ) , ( k ) ) )

DATA_DSYRK(n, k) = 1.0 * sizeof(Float64) * DATA__SYRK(n, k)

DATA_SSYRK(n, k) = 1.0 * sizeof(Float32) * DATA__SYRK(n, k)

DATA__SYR2K(n, k) = 0.5 * DATA_MAT(n, n) + 2 * DATA_MAT(n, k)

# Skipping MacroDefinition: DATA_ZSYR2K ( n , k ) ( 1.0 * sizeof ( double _Complex ) * DATA__SYR2K ( ( n ) , ( k ) ) )

# Skipping MacroDefinition: DATA_CSYR2K ( n , k ) ( 1.0 * sizeof ( float _Complex ) * DATA__SYR2K ( ( n ) , ( k ) ) )

DATA_DSYR2K(n, k) = 1.0 * sizeof(Float64) * DATA__SYR2K(n, k)

DATA_SSYR2K(n, k) = 1.0 * sizeof(Float32) * DATA__SYR2K(n, k)

DATA_ZHEMM(s, m, n) = if s == CblasLeft
        DATA_ZGEMM(m, m, n)
    else
        DATA_ZGEMM(m, n, n)
    end

DATA_CHEMM(s, m, n) = if s == CblasLeft
        DATA_CGEMM(m, m, n)
    else
        DATA_CGEMM(m, n, n)
    end

DATA_DHEMM(s, m, n) = if s == CblasLeft
        DATA_DGEMM(m, m, n)
    else
        DATA_DGEMM(m, n, n)
    end

DATA_SHEMM(s, m, n) = if s == CblasLeft
        DATA_SGEMM(m, m, n)
    else
        DATA_SGEMM(m, n, n)
    end

DATA_ZHERK(n, k) = DATA_ZSYRK(n, k)

DATA_CHERK(n, k) = DATA_CSYRK(n, k)

DATA_ZHER2K(n, k) = DATA_ZSYR2K(n, k)

DATA_CHER2K(n, k) = DATA_CSYR2K(n, k)

# exports
const PREFIXES = [""]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

