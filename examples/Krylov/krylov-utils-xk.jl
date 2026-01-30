using Krylov, SparseMatricesCSR, XK
import Krylov: FloatOrComplex

#############################################################################
# Wrap XK.BLAS routines to return the scalar result from write-back address #
#############################################################################

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

function xknrm2(n::Integer, x::Vector{T}) where T <: AbstractFloat
    return sqrt(real(xkdotr(n, x, x)))
end

#######################################################
# Dispatch routines over XKVector/XKMatrix to XK.BLAS #
#######################################################

# TODO: why do we need both ?
Krylov.kaxpy!(n::Integer, a::T, x::XKVector{T},          y::XKVector{T})          where T <: FloatOrComplex = XK.BLAS.axpy_sync(n, a, x.data, 1, y.data, 1)
Krylov.kaxpy!(n::Integer, a::T, x::XKVector{Complex{T}}, y::XKVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.axpy_sync(n, a, x.data, 1, y.data, 1)

# TODO: same, why do we need all this?
Krylov.kaxpby!(n::Integer, a::T,          x::XKVector{T},          b::T,          y::XKVector{T})          where T <: FloatOrComplex = XK.BLAS.ext.axpby_sync(n, a, x.data, 1, b, y.data, 1)
Krylov.kaxpby!(n::Integer, a::T,          x::XKVector{Complex{T}}, t::Complex{T}, y::XKVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.axpby_sync(n, a, x.data, 1, b, y.data, 1)
Krylov.kaxpby!(n::Integer, a::Complex{T}, x::XKVector{Complex{T}}, b::T,          y::XKVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.axpby_sync(n, a, x.data, 1, b, y.data, 1)
Krylov.kaxpby!(n::Integer, a::T,          x::XKVector{Complex{T}}, b::T,          y::XKVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.axpby_sync(n, a, x.data, 1, b, y.data, 1)

Krylov.kdot( n::Integer, x::XKVector{T}, y::XKVector{T}) where T <: FloatOrComplex = xkdot( n, x.data, y.data)
Krylov.kdotr(n::Integer, x::XKVector{T}, y::XKVector{T}) where T <: FloatOrComplex = xkdotr(n, x.data, y.data)

# TODO: do we need other types ?
Krylov.kfill!(x::XKVector{T}, val::T) where T <: FloatOrComplex = XK.BLAS.ext.fill_sync(length(x.data), x.data, val)

# TODO: why BLAS.BlasFloat? not FloatOrComplex ?
Krylov.kscal!(n::Integer, s::T, x::XKVector{T}) where T <: BLAS.BlasFloat = XK.BLAS.ext.scal_sync(n, s, x.data, 1)

# TODO: added an XKSparseMatrix too here probably
Krylov.kmul!(y::XKVector{T}, P::XKSparseMatrixCSR, x::XKVector{T}) where T <: FloatOrComplex =
    XK.BLAS.spmv_sync(T(1.0), XK.BLAS.NO_TRANS, size(P.data, 1), size(P.data, 2), length(P.data.nzval), XK.BLAS.SPARSE_CSR, P.data.rowptr, P.data.colval, P.data.nzval, x.data, T(0.0), y.data)

Krylov.kmul!(y::XKVector{T}, P::XKMatrix, x::XKVector{T}) where T <: FloatOrComplex =
    XK.BLAS.gemv_sync(XK.BLAS.NO_TRANS, size(P.data, 1), size(P.data, 2), T(1.0), p.data, stride(p.data, 2), x.data, 1, T(0.0), y.data, 1)

# TODO: why BLAS.BlasFloat
Krylov.kcopy!(n::Integer, y::XKVector{T}, x::XKVector{T}) where {T<:BLAS.BlasFloat} = XK.BLAS.ext.copy_sync(n, x.data, 1, y.data, 1)

# TODO: why BLAS.BlasFloat
Krylov.knorm(n::Integer, x::XKVector{T}) where T <: BLAS.BlasFloat = xknrm2(n, x.data)

#################################################
# The following APIs fallback to other routines #
#################################################

Krylov.kscalcopy!(n::Integer, y::XKVector{T}, s::T, x::XKVector{T}) where T <: FloatOrComplex = Krylov.kaxpby!(n, s, x, T(0), y)
Krylov.kdivcopy!( n::Integer, y::XKVector{T}, x::XKVector{T}, s::T) where T <: FloatOrComplex = Krylov.kscalcopy!(n, y, one(T)/s, x)

# TODO: why both ?
Krylov.kdiv!(n::Integer, x::XKVector{Complex{T}}, s::T) where T <: AbstractFloat  = Krylov.kscal!(n, one(T) / s, x)
Krylov.kdiv!(n::Integer, x::XKVector{T}, s::T)          where T <: FloatOrComplex = Krylov.kscal!(n, one(T) / s, x)

# TODO
# kldiv and kref!
