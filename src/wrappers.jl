####################
# HOST ASYNC TASKS #
####################

# global table to keep Refs alive
const _host_async_refs = IdDict{Ptr{Cvoid}, Any}()

function _host_async_trampoline(fptr::Ptr{Cvoid})
    args = unsafe_pointer_to_objref(fptr)
    args[]()
    delete!(_host_async_refs, fptr)
    return
end

function host_async(body::Function; set_accesses::Union{Function,Nothing}=nothing)
    fptr = @cfunction(_host_async_trampoline, Cvoid, (Ptr{Cvoid},))
    args = Ref(body)
    _host_async_refs[fptr] = args  # preserve Ref until trampoline executed

    accesses = xkrt_access_t[]
    if set_accesses !== nothing
        set_accesses(accesses)
    end

    local len = length(accesses)
    if len == 0
        XKBlas.host_async(fptr, args)
    else
        XKBlas.host_with_accesses_async(fptr, args, pointer(accesses), Cint(len))
    end
end

function host_async(set_accesses::Function, body::Function)
    return XKBlas.host_async(body, set_accesses=set_accesses)
end

# Helper constructor for xkrt_access_t
function Access(
    mode::xkrt_access_mode_t,
    region::Union{xkrt_handle_t, xkrt_segment_t, xkrt_matrix_t};
    scope::Union{xkrt_access_scope_t, Nothing}=nothing,
    concurrency::Union{xkrt_access_concurrency_t, Nothing}=nothing
)
    access     = xkrt_access_t(ntuple(_ -> 0x00, 80))
    access_ref = Ref(access)
    access_ptr = Base.unsafe_convert(Ptr{xkrt_access_t}, access_ref)

    # set mode
    @assert mode != nothing
    access_ptr.mode = mode

    # set type
    if region isa xkrt_handle_t
        access_ptr.type = ACCESS_TYPE_HANDLE
        access_ptr.region.handle = region
    elseif region isa xkrt_segment_t
        access_ptr.type = ACCESS_TYPE_SEGMENT
        access_ptr.region.segment = region
    elseif region isa xkrt_matrix_t
        access_ptr.type = ACCESS_TYPE_MATRIX
        access_ptr.region.matrix = region
    else
        error("Unknown region type: $(typeof(region))")
    end

    # set scope
    if scope == nothing
        scope = ACCESS_SCOPE_NONUNIFIED
    end
    access_ptr.scope = scope

    # set concurrency
    if concurrency == nothing
        concurrency = ACCESS_CONCURRENCY_SEQUENTIAL
    end
    access_ptr.concurrency = concurrency

    return access_ref[]
end

const Handle  = xkrt_handle_t;
const Segment = xkrt_segment_t;
const Matrix  = xkrt_matrix_t;

########################
# Dispatcher for types #
########################

# Memory routines

memory_coherent_async(x)            = memory_segment_coherent_async(x, length(x))
memory_coherent_async(x, n)         = memory_segment_coherent_async(x, n*sizeof(eltype(x)))
memory_coherent_async(A, lda, m, n) = memory_matrix_coherent_async(A, lda, m, n, sizeof(eltype(A)))

# Kernels (see `xkblas/xkblas.hpp` and convert C++ prototypes)

### Level 1 ###

axpy(n, alpha::Float32,    x, incx, y, incy)  = saxpy(n, Ref(alpha), x, incx, y, incy)
axpy(n, alpha::Float64,    x, incx, y, incy)  = daxpy(n, Ref(alpha), x, incx, y, incy)
axpy(n, alpha::ComplexF32, x, incx, y, incy)  = caxpy(n, Ref(alpha), x, incx, y, incy)
axpy(n, alpha::ComplexF64, x, incx, y, incy)  = zaxpy(n, Ref(alpha), x, incx, y, incy)

axpy_lazy(n, alpha::Float32,    x, incx, y, incy)  = saxpy_lazy(n, Ref(alpha), x, incx, y, incy)
axpy_lazy(n, alpha::Float64,    x, incx, y, incy)  = daxpy_lazy(n, Ref(alpha), x, incx, y, incy)
axpy_lazy(n, alpha::ComplexF32, x, incx, y, incy)  = caxpy_lazy(n, Ref(alpha), x, incx, y, incy)
axpy_lazy(n, alpha::ComplexF64, x, incx, y, incy)  = zaxpy_lazy(n, Ref(alpha), x, incx, y, incy)

axpy_async(n, alpha::Float32,    x, incx, y, incy)  = saxpy_async(n, Ref(alpha), x, incx, y, incy)
axpy_async(n, alpha::Float64,    x, incx, y, incy)  = daxpy_async(n, Ref(alpha), x, incx, y, incy)
axpy_async(n, alpha::ComplexF32, x, incx, y, incy)  = caxpy_async(n, Ref(alpha), x, incx, y, incy)
axpy_async(n, alpha::ComplexF64, x, incx, y, incy)  = zaxpy_async(n, Ref(alpha), x, incx, y, incy)

axpby(n, alpha::Float32,    x, incx, beta, y, incy) = saxpby(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby(n, alpha::Float64,    x, incx, beta, y, incy) = daxpby(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby(n, alpha::ComplexF32, x, incx, beta, y, incy) = caxpby(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby(n, alpha::ComplexF64, x, incx, beta, y, incy) = zaxpby(n, Ref(alpha), x, incx, Ref(beta), y, incy)

axpby_lazy(n, alpha::Float32,    x, incx, beta, y, incy) = saxpby_lazy(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_lazy(n, alpha::Float64,    x, incx, beta, y, incy) = daxpby_lazy(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_lazy(n, alpha::ComplexF32, x, incx, beta, y, incy) = caxpby_lazy(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_lazy(n, alpha::ComplexF64, x, incx, beta, y, incy) = zaxpby_lazy(n, Ref(alpha), x, incx, Ref(beta), y, incy)

axpby_async(n, alpha::Float32,    x, incx, beta, y, incy) = saxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_async(n, alpha::Float64,    x, incx, beta, y, incy) = daxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_async(n, alpha::ComplexF32, x, incx, beta, y, incy) = caxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_async(n, alpha::ComplexF64, x, incx, beta, y, incy) = zaxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)

copy(n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = scopy(n, x, incx, y, incy)
copy(n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = dcopy(n, x, incx, y, incy)
copy(n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = ccopy(n, x, incx, y, incy)
copy(n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = zcopy(n, x, incx, y, incy)

copy_lazy(n, x::AbstractVector{Float32},   incx, y::AbstractVector{Float32}   , incy)  = scopy_lazy(n, x, incx, y, incy)
copy_lazy(n, x::AbstractVector{Float64},   incx, y::AbstractVector{Float64}   , incy)  = dcopy_lazy(n, x, incx, y, incy)
copy_lazy(n, x::AbstractVector{ComplexF32},incx, y::AbstractVector{ComplexF32}, incy)  = ccopy_lazy(n, x, incx, y, incy)
copy_lazy(n, x::AbstractVector{ComplexF64},incx, y::AbstractVector{ComplexF64}, incy)  = zcopy_lazy(n, x, incx, y, incy)

copy_async(n, x::AbstractVector{Float32},   incx, y::AbstractVector{Float32}   , incy)  = scopy_async(n, x, incx, y, incy)
copy_async(n, x::AbstractVector{Float64},   incx, y::AbstractVector{Float64}   , incy)  = dcopy_async(n, x, incx, y, incy)
copy_async(n, x::AbstractVector{ComplexF32},incx, y::AbstractVector{ComplexF32}, incy)  = ccopy_async(n, x, incx, y, incy)
copy_async(n, x::AbstractVector{ComplexF64},incx, y::AbstractVector{ComplexF64}, incy)  = zcopy_async(n, x, incx, y, incy)

# TODO: complex version not supported yet, but they will need to change the dispatcher
dot(n, x, incx, y, incy, result::Ref{Float32}) = sdot(n, x, incx, y, incy, result)
dot(n, x, incx, y, incy, result::Ref{Float64}) = ddot(n, x, incx, y, incy, result)

dot_lazy(n, x, incx, y, incy, result::Ref{Float32}) = sdot_lazy(n, x, incx, y, incy, result)
dot_lazy(n, x, incx, y, incy, result::Ref{Float64}) = ddot_lazy(n, x, incx, y, incy, result)

dot_async(n, x, incx, y, incy, result::Ref{Float32}) = sdot_async(n, x, incx, y, incy, result)
dot_async(n, x, incx, y, incy, result::Ref{Float64}) = ddot_async(n, x, incx, y, incy, result)

fill(n, x, value::Float32   ) = sfill(n, x, value)
fill(n, x, value::Float64   ) = dfill(n, x, value)
fill(n, x, value::ComplexF32) = cfill(n, x, value)
fill(n, x, value::ComplexF64) = zfill(n, x, value)

fill_lazy(n, x, value::Float32   ) = sfill_lazy(n, x, value)
fill_lazy(n, x, value::Float64   ) = dfill_lazy(n, x, value)
fill_lazy(n, x, value::ComplexF32) = cfill_lazy(n, x, value)
fill_lazy(n, x, value::ComplexF64) = zfill_lazy(n, x, value)

fill_async(n, x, value::Float32   ) = sfill_async(n, x, value)
fill_async(n, x, value::Float64   ) = dfill_async(n, x, value)
fill_async(n, x, value::ComplexF32) = cfill_async(n, x, value)
fill_async(n, x, value::ComplexF64) = zfill_async(n, x, value)

# TODO: complex version not supported yet, but they will need to change the dispatcher
scal(n, alpha::Float32, x, incx) = sscal_async(n, Ref(alpha), x, incx)
scal(n, alpha::Float64, x, incx) = dscal_async(n, Ref(alpha), x, incx)

scal_lazy(n, alpha::Float32, x, incx) = sscal_lazy(n, Ref(alpha), x, incx)
scal_lazy(n, alpha::Float64, x, incx) = dscal_lazy(n, Ref(alpha), x, incx)

scal_async(n, alpha::Float32, x, incx) = sscal_async(n, Ref(alpha), x, incx)
scal_async(n, alpha::Float64, x, incx) = dscal_async(n, Ref(alpha), x, incx)

### Level 2 ###

gemv(transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = sgemv(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
gemv(transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = dgemv(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)

gemv_lazy(transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = sgemv_lazy(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
gemv_lazy(transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = dgemv_lazy(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)

gemv_async(transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = sgemv_async(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
gemv_async(transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = dgemv_async(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)

### Level 3 ###

gemm(transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemm(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm(transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemm(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm(transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemm(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm(transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemm(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

gemm_lazy(transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemm_lazy(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_lazy(transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemm_lazy(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_lazy(transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemm_lazy(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_lazy(transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemm_lazy(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

gemm_async(transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_async(transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_async(transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_async(transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

gemmt_async(uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_async(uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_async(uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_async(uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

herk(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = sherk(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dherk(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = cherk(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zherk(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

herk_async(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = sherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_async(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = cherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

herk_lazy(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = sherk_lazy(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_lazy(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dherk_lazy(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_lazy(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = cherk_lazy(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_lazy(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zherk_lazy(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

symm(side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssymm(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm(side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsymm(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm(side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csymm(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm(side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsymm(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

symm_lazy(side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssymm_lazy(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_lazy(side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsymm_lazy(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_lazy(side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csymm_lazy(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_lazy(side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsymm_lazy(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

symm_async(side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_async(side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_async(side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_async(side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

syr2k(uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssyr2k(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k(uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsyr2k(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k(uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csyr2k(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k(uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsyr2k(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

syr2k_lazy(uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssyr2k_lazy(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_lazy(uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsyr2k_lazy(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_lazy(uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csyr2k_lazy(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_lazy(uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsyr2k_lazy(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

syr2k_async(uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_async(uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

syrk(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = ssyrk(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dsyrk(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = csyrk(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zsyrk(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

syrk_lazy(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = ssyrk_lazy(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_lazy(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dsyrk_lazy(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_lazy(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = csyrk_lazy(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_lazy(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zsyrk_lazy(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

syrk_async(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = ssyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_async(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dsyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = csyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zsyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

trmm(side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = strmm(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm(side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = dtrmm(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm(side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = ctrmm(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm(side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = ztrmm(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)

trmm_lazy(side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = strmm_lazy(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_lazy(side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = dtrmm_lazy(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_lazy(side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = ctrmm_lazy(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_lazy(side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = ztrmm_lazy(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)

trmm_async(side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = strmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_async(side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = dtrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_async(side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = ctrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_async(side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = ztrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)

# Export symbols
for name in names(@__MODULE__; all=true)
    @eval export $name
end
