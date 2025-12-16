using XKBlas

using Krylov
import Krylov: FloatOrComplex

using SparseMatricesCSR

function kdotr(n::Integer, x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
    r = Ref{T}(0)
    XKBlas.dot_sync(n, x, 1, y, 1, r)
    return r[]
end

function knrm2(n::Integer, x::Vector{T}) where T <: AbstractFloat
    # TODO: xkblas do not yet support nrm2, use dot
    # r = Ref{T}(0)
    # XKBlas.nrm2(n, x, 1, r)
    # return r[]
    return kdotr(n, x, x)
end

Krylov.kdotr(n::Integer, x::AbstractVector{Complex{T}}, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = kdotr(n, x, y)
Krylov.kdotr(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: AbstractFloat = kdotr(n, x, y)

Krylov.kaxpy!(n::Integer, a::T, x::Vector{T}, y::Vector{T}) where T <: BLAS.BlasFloat = XKBlas.axpy_sync(n, a, x, 1, y, 1)
Krylov.kaxpy!(n::Integer, a::T, x::AbstractVector{T}, y::AbstractVector{T}) where T <: FloatOrComplex = XKBlas.axpy_sync(n, a, x, 1, y, 1)
Krylov.kaxpy!(n::Integer, a::T, x::AbstractVector{Complex{T}}, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.axpy_sync(n, a, x, 1, y, 1)

Krylov.kaxpby!(n::Integer, a::T, x::Vector{T}, b::T, y::Vector{T}) where T <: BLAS.BlasFloat = XKBlas.axpby_sync(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, a::T, x::AbstractVector{T}, b::T, y::AbstractVector{T}) where T <: FloatOrComplex = XKBlas.axpby_sync(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, a::T, x::AbstractVector{Complex{T}}, t::Complex{T}, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.axpby_sync(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, s::Complex{T}, x::AbstractVector{Complex{T}}, b::T, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.axpby_sync(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, a::T, x::AbstractVector{Complex{T}}, b::T, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.axpby_sync(n, a, x, 1, b, y, 1)

Krylov.kcopy!(n::Integer, y::Vector{T}, x::Vector{T}) where {T<:BLAS.BlasFloat} = XKBlas.copy_sync(n, x, 1, y, 1)
Krylov.kcopy!(n::Integer, y::AbstractVector, x::AbstractVector) = XKBlas.copy_sync(n, x, 1, y, 1)

# no need to impl
Krylov.kdiv!(n::Integer, x::AbstractVector{Complex{T}}, s::T) where T <: AbstractFloat = Krylov.kscal!(n, one(T) / s, x)
Krylov.kdiv!(n::Integer, x::AbstractVector{T}, s::T) where T <: FloatOrComplex = Krylov.kscal!(n, one(T) / s, x)

Krylov.kdivcopy!(n::Integer, y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::T) where T <: AbstractFloat = XKBlas.divcopy_sync(n, x, 1, y, 1, s)
Krylov.kdivcopy!(n::Integer, y::AbstractVector{T}, x::AbstractVector{T}, s::T) where T <: FloatOrComplex = XKBlas.divcopy_sync(n, x, 1, y, 1, s)

Krylov.kdot(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: FloatOrComplex = XKBlas.dot_sync(x, 1, y, 1)
Krylov.kdot(n::Integer, x::Vector{T}, y::Vector{T}) where T <: BLAS.BlasComplex = XKBlas.dot_sync(x, 1, y, 1)
Krylov.kdot(n::Integer, x::Vector{T}, y::Vector{T}) where T <: BLAS.BlasReal = XKBlas.dot_sync(x, 1, y, 1)

Krylov.kfill!(x::AbstractArray{T}, val::T) where T <: FloatOrComplex = XKBlas.fill_sync(length(x), x, val)

Krylov.knorm(n::Integer, x::AbstractVector{T}) where T <: FloatOrComplex = knrm2(n, x)
Krylov.knorm(n::Integer, x::Vector{T}) where T <: BLAS.BlasFloat = knrm2(n, x)

# TODO
# knorm_elliptic(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: FloatOrComplex = (x === y) ? knorm(n, x) : kdotr(n, x, y) |> sqrt

# TODO
# kmul and ldiv

Krylov.kscal!(n::Integer, s::T, x::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.scal_sync(n, s, x, 1)
Krylov.kscal!(n::Integer, s::T, x::AbstractVector{T}) where T <: FloatOrComplex = XKBlas.scal_sync(n, s, x, 1)
Krylov.kscal!(n::Integer, s::T, x::Vector{T}) where T <: BLAS.BlasFloat = XKBlas.scal_sync(n, s, x, 1)

Krylov.kscalcopy!(n::Integer, y::AbstractVector{Complex{T}}, s::T, x::AbstractVector{Complex{T}}) where T <: AbstractFloat = Krylov.kaxpby!(n, s, x, T(0), y)
Krylov.kscalcopy!(n::Integer, y::AbstractVector{T}, s::T, x::AbstractVector{T}) where T <: FloatOrComplex = Krylov.kaxpby!(n, s, x, T(0), y)

Krylov.kmul!(y::AbstractVector{T}, P::SparseMatrixCSR, x::AbstractVector{T}) where T <: FloatOrComplex = XKBlas.spmv_sync(T(1.0), CblasNoTrans, size(P, 1), size(P, 2), length(P.nzval), CblasSparseCSR, P.rowptr, P.colval, P.nzval, x, T(0.0), y)
