####################
# HOST ASYNC TASKS #
####################

# global table to keep Refs alive
const _host_async_refs = IdDict{Ptr{Cvoid}, Any}()

function _async_trampoline(fptr::Ptr{Cvoid})
    args = unsafe_pointer_to_objref(fptr)
    GC.enable(false)    # disable GC to avoid deadlocks if 'args[]()' ends-up calling the Julia runtime
    args[]()
    GC.enable(true)
    delete!(_host_async_refs, fptr)
    return
end

#
#   Generic API to spawn an XKRT task
#   The 'may_run_julia_runtime' indicates whether the task may enter the Julia runtime.
#   In such case,
#       - it is made detachable
#       - it will be executed by a Julia thread.
#   see https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/#Thread-safety
#
function device_async(
    device_global_id::xkrt_device_global_id_t,
    fmt_or_func::Union{xkrt_task_format_id_t, Function};
    set_accesses::Union{Function,Nothing}=nothing,
    args=C_NULL,
    args_size=0,
    detach_ctr_initial=nothing
)
    accesses = xkrt_access_t[]
    if set_accesses !== nothing
        set_accesses(accesses)
    end
    local AC = length(accesses)
    local ocr_access = UNSPECIFIED_TASK_ACCESS

    local flags = TASK_FLAG_ZERO
    detach_ctr_initial != nothing  && (flags |= TASK_FLAG_DETACHABLE)
    (AC > 0)                       && (flags |= TASK_FLAG_DEPENDENT)
    (true)                         && (flags |= TASK_FLAG_DEVICE)

    runtime = XKBlas.xkrt_runtime_get()

    if fmt_or_func isa xkrt_task_format_id_t
        XKBlas.xkrt_task_spawn_generic(
            runtime,
            device_global_id,
            flags,
            fmt_or_func::xkrt_task_format_id_t,
            args, args_size,
            pointer(accesses), Cint(AC),
            ocr_access,
            detach_ctr_initial
        )
    elseif fmt_or_func isa Function
        fptr = @cfunction(_async_trampoline, Cvoid, (Ptr{Cvoid},))
        args = Ref(fmt_or_func::Function)
        _host_async_refs[fptr] = args  # preserve Ref until trampoline executed
        XKBlas.Logger.fatal("TODO: 2 args here, the function body and user parameter")
        # XKBlas.async_generic_with_format(
        #     device_global_id,
        #     fmt_or_func::xkrt_task_format_id_t,
        #     flags,
        #     args, args_size,
        #     pointer(accesses), Cint(AC),
        #     detach_ctr_initial
        # )
    end
end

function host_async(body::Function; set_accesses::Union{Function,Nothing}=nothing)
    return device_async(HOST_DEVICE_GLOBAL_ID, body, set_accesses=set_accesses)
end

function host_async(set_accesses::Function, body::Function)
    return device_async(HOST_DEVICE_GLOBAL_ID, body, set_accesses=set_accesses)
end

@enum ArgumentPassMode begin
    Value
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

# Access Types and regions
const Handle  = xkrt_handle_t;
const Segment = xkrt_segment_t;
const Matrix  = xkrt_matrix_t;

# Wrappers
function Access(
    mode::xkrt_access_mode_t,
    vec::AbstractVector;
    scope::Union{xkrt_access_scope_t, Nothing}=nothing,
    concurrency::Union{xkrt_access_concurrency_t, Nothing}=nothing
)
    return Access(mode, Segment(pointer(vec), pointer(vec) + length(vec) * Base.elsize(vec)))
end

# Set tile parameters for all kernels
function set_tile_parameter(ts)
    XKBlas.set_param(ts, 0)
end

########################
# Dispatcher for types #
########################

# Memory routines

memory_coherent_async(A, lda, m, n) = memory_matrix_coherent_async(A, lda, m, n, sizeof(eltype(A)))
memory_coherent_sync(A, lda, m, n)  = memory_matrix_coherent_async(A, lda, m, n, sizeof(eltype(A)))

memory_coherent_sync(x)             = memory_segment_coherent_async(x, length(x)*sizeof(eltype(x)))
memory_coherent_async(x)            = memory_segment_coherent_async(x, length(x)*sizeof(eltype(x)))

memory_coherent_async(x, n)         = memory_segment_coherent_async(x, n*sizeof(eltype(x)))
memory_coherent_sync(x, n)          = memory_segment_coherent_async(x, n*sizeof(eltype(x)))

# Kernels (see `xkblas/xkblas.hpp` and convert C++ prototypes)

### Level 1 ###

axpy(      n, alpha::ComplexF32, x, incx, y, incy)  = caxpy(      n, Ref(alpha), x, incx, y, incy)
axpy(      n, alpha::ComplexF64, x, incx, y, incy)  = zaxpy(      n, Ref(alpha), x, incx, y, incy)
axpy(      n, alpha::Float32,    x, incx, y, incy)  = saxpy(      n, Ref(alpha), x, incx, y, incy)
axpy(      n, alpha::Float64,    x, incx, y, incy)  = daxpy(      n, Ref(alpha), x, incx, y, incy)
axpy_async(n, alpha::ComplexF32, x, incx, y, incy)  = caxpy_async(n, Ref(alpha), x, incx, y, incy)
axpy_async(n, alpha::ComplexF64, x, incx, y, incy)  = zaxpy_async(n, Ref(alpha), x, incx, y, incy)
axpy_async(n, alpha::Float32,    x, incx, y, incy)  = saxpy_async(n, Ref(alpha), x, incx, y, incy)
axpy_async(n, alpha::Float64,    x, incx, y, incy)  = daxpy_async(n, Ref(alpha), x, incx, y, incy)
axpy_sync( n, alpha::ComplexF32, x, incx, y, incy)  = caxpy_sync( n, Ref(alpha), x, incx, y, incy)
axpy_sync( n, alpha::ComplexF64, x, incx, y, incy)  = zaxpy_sync( n, Ref(alpha), x, incx, y, incy)
axpy_sync( n, alpha::Float32,    x, incx, y, incy)  = saxpy_sync( n, Ref(alpha), x, incx, y, incy)
axpy_sync( n, alpha::Float64,    x, incx, y, incy)  = daxpy_sync( n, Ref(alpha), x, incx, y, incy)

axpby(      n, alpha::ComplexF32, x, incx, beta, y, incy) = caxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby(      n, alpha::ComplexF64, x, incx, beta, y, incy) = zaxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby(      n, alpha::Float32,    x, incx, beta, y, incy) = saxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby(      n, alpha::Float64,    x, incx, beta, y, incy) = daxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_async(n, alpha::ComplexF32, x, incx, beta, y, incy) = caxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_async(n, alpha::ComplexF64, x, incx, beta, y, incy) = zaxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_async(n, alpha::Float32,    x, incx, beta, y, incy) = saxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_async(n, alpha::Float64,    x, incx, beta, y, incy) = daxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_sync( n, alpha::ComplexF32, x, incx, beta, y, incy) = caxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_sync( n, alpha::ComplexF64, x, incx, beta, y, incy) = zaxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_sync( n, alpha::Float32,    x, incx, beta, y, incy) = saxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)
axpby_sync( n, alpha::Float64,    x, incx, beta, y, incy) = daxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)

copy(      n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = ccopy(      n, x, incx, y, incy)
copy(      n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = zcopy(      n, x, incx, y, incy)
copy(      n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = scopy(      n, x, incx, y, incy)
copy(      n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = dcopy(      n, x, incx, y, incy)
copy_async(n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = ccopy_async(n, x, incx, y, incy)
copy_async(n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = zcopy_async(n, x, incx, y, incy)
copy_async(n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = scopy_async(n, x, incx, y, incy)
copy_async(n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = dcopy_async(n, x, incx, y, incy)
copy_sync( n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = ccopy_sync( n, x, incx, y, incy)
copy_sync( n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = zcopy_sync( n, x, incx, y, incy)
copy_sync( n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = scopy_sync( n, x, incx, y, incy)
copy_sync( n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = dcopy_sync( n, x, incx, y, incy)

# TODO: complex version not supported yet, but they will need to change the dispatcher
dot(      n, x, incx, y, incy, result::Ref{Float32}) = sdot(      n, x, incx, y, incy, result)
dot(      n, x, incx, y, incy, result::Ref{Float64}) = ddot(      n, x, incx, y, incy, result)
dot_async(n, x, incx, y, incy, result::Ref{Float32}) = sdot_async(n, x, incx, y, incy, result)
dot_async(n, x, incx, y, incy, result::Ref{Float64}) = ddot_async(n, x, incx, y, incy, result)
dot_sync( n, x, incx, y, incy, result::Ref{Float32}) = sdot_sync( n, x, incx, y, incy, result)
dot_sync( n, x, incx, y, incy, result::Ref{Float64}) = ddot_sync( n, x, incx, y, incy, result)

fill(      n, x, value::ComplexF32) = cfill(      n, x, value)
fill(      n, x, value::ComplexF64) = zfill(      n, x, value)
fill(      n, x, value::Float32   ) = sfill(      n, x, value)
fill(      n, x, value::Float64   ) = dfill(      n, x, value)
fill_async(n, x, value::ComplexF32) = cfill_async(n, x, value)
fill_async(n, x, value::ComplexF64) = zfill_async(n, x, value)
fill_async(n, x, value::Float32   ) = sfill_async(n, x, value)
fill_async(n, x, value::Float64   ) = dfill_async(n, x, value)
fill_sync( n, x, value::ComplexF32) = cfill_sync( n, x, value)
fill_sync( n, x, value::ComplexF64) = zfill_sync( n, x, value)
fill_sync( n, x, value::Float32   ) = sfill_sync( n, x, value)
fill_sync( n, x, value::Float64   ) = dfill_sync( n, x, value)

# TODO: complex version not supported yet, but they will need to change the dispatcher
scal(      n, alpha::Float32, x, incx) = sscal(      n, Ref(alpha), x, incx)
scal(      n, alpha::Float64, x, incx) = dscal(      n, Ref(alpha), x, incx)
scal_async(n, alpha::Float32, x, incx) = sscal_async(n, Ref(alpha), x, incx)
scal_async(n, alpha::Float64, x, incx) = dscal_async(n, Ref(alpha), x, incx)
scal_sync( n, alpha::Float32, x, incx) = sscal_sync( n, Ref(alpha), x, incx)
scal_sync( n, alpha::Float64, x, incx) = dscal_sync( n, Ref(alpha), x, incx)

### Level 2 ###

gemv(      transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = sgemv(      transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
gemv(      transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = dgemv(      transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
gemv_async(transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = sgemv_async(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
gemv_async(transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = dgemv_async(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
gemv_sync( transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = sgemv_sync( transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
gemv_sync( transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = dgemv_sync( transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)

### Level 3 ###

gemm(      transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm(      transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm(      transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm(      transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_async(transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_async(transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_async(transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_async(transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_sync( transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_sync( transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_sync( transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemm_sync( transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

gemmt(      uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt(      uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt(      uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt(      uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_async(uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_async(uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_async(uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_async(uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_sync( uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = cgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_sync( uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_sync( uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = sgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
gemmt_sync( uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

herk(      uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = cherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk(      uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk(      uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = sherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk(      uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = cherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_async(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = sherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_async(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_sync( uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = cherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_sync( uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_sync( uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = sherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
herk_sync( uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

symm(      side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm(      side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm(      side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm(      side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_async(side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_async(side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_async(side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_async(side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_sync( side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_sync( side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_sync( side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
symm_sync( side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

syr2k(      uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k(      uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k(      uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k(      uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_async(uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_async(uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_sync( uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = csyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_sync( uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = zsyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_sync( uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = ssyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syr2k_sync( uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = dsyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

syrk(      uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = csyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk(      uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zsyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk(      uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = ssyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk(      uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dsyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = csyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zsyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_async(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = ssyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_async(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dsyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_sync( uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = csyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_sync( uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = zsyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_sync( uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = ssyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
syrk_sync( uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = dsyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

trmm(      side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = ctrmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm(      side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = ztrmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm(      side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = strmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm(      side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = dtrmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_async(side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = ctrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_async(side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = ztrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_async(side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = strmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_async(side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = dtrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_sync( side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = ctrmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_sync( side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = ztrmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_sync( side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = strmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
trmm_sync( side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = dtrmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)

# SPMV
spmv(      alpha::ComplexF32, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = cspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv(      alpha::ComplexF64, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = zspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv(      alpha::Float32,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = sspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv(      alpha::Float64,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = dspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv_async(alpha::ComplexF32, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = cspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv_async(alpha::ComplexF64, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = zspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv_async(alpha::Float32,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = sspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv_async(alpha::Float64,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = dspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv_sync( alpha::ComplexF32, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = cspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv_sync( alpha::ComplexF64, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = zspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv_sync( alpha::Float32,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = sspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
spmv_sync( alpha::Float64,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = dspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)

# Export symbols
for name in names(@__MODULE__; all=true)
    @eval export $name
end
