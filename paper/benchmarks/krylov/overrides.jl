using Krylov, SparseMatricesCSR, XK
import Krylov: FloatOrComplex

function xkdot(n::Integer, x::Vector{T}, y::Vector{T}) where T <: FloatOrComplex
    r = Ref{T}(0)
    XK.BLAS.dot_sync(n, x, 1, y, 1, r)
    return r[]
end

function xkdotr(n::Integer, x::Vector{T}, y::Vector{T}) where T <: FloatOrComplex
    r = Ref{T}(0)
    # TODO: i think this is wrong for complexP::SparseMatrixCSR
    XK.BLAS.dot_sync(n, x, 1, y, 1, r)
    return r[]
end

# TODO
# struct XKArray{T, N} <: AbstractArray{T,N}
#     data::Array{T, N}
# end
#
# const XKVector{T} = XKArray{T, 1}
# const XKMatrix{T} = XKArray{T, 2}
#
# # TODO: this should work - Alexis do it thanks :-)
# #   mat = XKSparseMatrix(mat_csc)
# #   mat = XKSparseMatrix(mat_csr)
# struct XKSparseMatrix{T MAT_T}
#     data::SparseMatrixCSR
# end

function xknrm2(n::Integer, x::Vector{T}) where T <: AbstractFloat
    return sqrt(xkdotr(n, x, x))
end

Krylov.kdotr(n::Integer, x::AbstractVector{Complex{T}}, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = xkdotr(n, x, y)
Krylov.kdotr(n::Integer, x::AbstractVector{T},          y::AbstractVector{T})          where T <: AbstractFloat = xkdotr(n, x, y)

Krylov.kdot(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: FloatOrComplex   = xkdot(n, x, y)
Krylov.kdot(n::Integer, x::Vector{T},         y::Vector{T})         where T <: BLAS.BlasComplex = xkdot(n, x, y)
Krylov.kdot(n::Integer, x::Vector{T},         y::Vector{T})         where T <: BLAS.BlasReal    = xkdot(n, x, y)

Krylov.kaxpy!(n::Integer, a::T, x::Vector{T},                  y::Vector{T})                  where T <: BLAS.BlasFloat = XK.BLAS.axpy_sync(n, a, x, 1, y, 1)
Krylov.kaxpy!(n::Integer, a::T, x::AbstractVector{T},          y::AbstractVector{T})          where T <: FloatOrComplex = XK.BLAS.axpy_sync(n, a, x, 1, y, 1)
Krylov.kaxpy!(n::Integer, a::T, x::AbstractVector{Complex{T}}, y::AbstractVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.axpy_sync(n, a, x, 1, y, 1)

Krylov.kaxpby!(n::Integer, a::T,          x::Vector{T},                  b::T,          y::Vector{T})                  where T <: BLAS.BlasFloat = XK.BLAS.ext.axpby_sync(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, a::T,          x::AbstractVector{T},          b::T,          y::AbstractVector{T})          where T <: FloatOrComplex = XK.BLAS.ext.axpby_sync(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, a::T,          x::AbstractVector{Complex{T}}, t::Complex{T}, y::AbstractVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.axpby_sync(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, s::Complex{T}, x::AbstractVector{Complex{T}}, b::T,          y::AbstractVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.axpby_sync(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, a::T,          x::AbstractVector{Complex{T}}, b::T,          y::AbstractVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.axpby_sync(n, a, x, 1, b, y, 1)

Krylov.kcopy!(n::Integer, y::Vector{T},      x::Vector{T})      where {T<:BLAS.BlasFloat} = XK.BLAS.ext.copy_sync(n, x, 1, y, 1)
Krylov.kcopy!(n::Integer, y::AbstractVector, x::AbstractVector)                           = XK.BLAS.ext.copy_sync(n, x, 1, y, 1)

# no need to impl
Krylov.kdiv!(n::Integer, x::AbstractVector{Complex{T}}, s::T) where T <: AbstractFloat  = Krylov.kscal!(n, one(T) / s, x)
Krylov.kdiv!(n::Integer, x::AbstractVector{T},          s::T) where T <: FloatOrComplex = Krylov.kscal!(n, one(T) / s, x)

Krylov.kdivcopy!(n::Integer, y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::T) where T <: AbstractFloat  = Krylov.kscalcopy!(n, y, one(T)/s, x)
Krylov.kdivcopy!(n::Integer, y::AbstractVector{T},          x::AbstractVector{T},          s::T) where T <: FloatOrComplex = Krylov.kscalcopy!(n, y, one(T)/s, x)

Krylov.kfill!(x::AbstractArray{T}, val::T) where T <: FloatOrComplex = XK.BLAS.ext.fill_sync(length(x), x, val)

Krylov.knorm(n::Integer, x::AbstractVector{T}) where T <: FloatOrComplex = xknrm2(n, x)
Krylov.knorm(n::Integer, x::Vector{T})         where T <: BLAS.BlasFloat = xknrm2(n, x)

# TODO
# knorm_elliptic(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: FloatOrComplex = (x === y) ? knorm(n, x) : kdotr(n, x, y) |> sqrt

# TODO
# kmul and ldiv

Krylov.kscal!(n::Integer, s::T, x::AbstractVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.scal_sync(n, s, x, 1)
Krylov.kscal!(n::Integer, s::T, x::AbstractVector{T})          where T <: FloatOrComplex = XK.BLAS.ext.scal_sync(n, s, x, 1)
Krylov.kscal!(n::Integer, s::T, x::Vector{T})                  where T <: BLAS.BlasFloat = XK.BLAS.ext.scal_sync(n, s, x, 1)

Krylov.kscalcopy!(n::Integer, y::AbstractVector{Complex{T}}, s::T, x::AbstractVector{Complex{T}}) where T <: AbstractFloat  = Krylov.kaxpby!(n, s, x, T(0), y)
Krylov.kscalcopy!(n::Integer, y::AbstractVector{T},          s::T, x::AbstractVector{T})          where T <: FloatOrComplex = Krylov.kaxpby!(n, s, x, T(0), y)

Krylov.kmul!(y::AbstractVector{T}, P::SparseMatrixCSR, x::AbstractVector{T}) where T <: FloatOrComplex = XK.BLAS.spmv_sync(T(1.0), XK.BLAS.NO_TRANS, size(P, 1), size(P, 2), length(P.nzval), XK.BLAS.SPARSE_CSR, P.rowptr, P.colval, P.nzval, x, T(0.0), y)
