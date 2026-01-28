using Krylov, SparseMatricesCSR, XK
import Krylov: FloatOrComplex

function xkdot(n::Integer, x::Vector{T}, y::Vector{T}) where T <: FloatOrComplex
    r = Ref{T}(0)
    # Alexis: Romain you need to call dotc here for complex numbers
    XK.BLAS.dot_sync(n, x, 1, y, 1, r)
    return r[]
end

function xknrm2(n::Integer, x::Vector{T}) where T <: AbstractFloat
    return sqrt(real(xkdotr(n, x, x)))
end

Krylov.kaxpy!(n::Integer, a::T, x::XKVector{T}, y::XKVector{T}) where T <: FloatOrComplex = XK.BLAS.axpy_sync(n, a, x.data, 1, y.data, 1)
Krylov.kaxpy!(n::Integer, a::T, x::XKVector{Complex{T}}, y::XKVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.axpy_sync(n, a, x.data, 1, y.data, 1)

Krylov.kaxpby!(n::Integer, a::T,          x::XKVector{T},          b::T,          y::XKVector{T})          where T <: FloatOrComplex = XK.BLAS.ext.axpby_sync(n, a, x.data, 1, b, y.data, 1)
Krylov.kaxpby!(n::Integer, a::T,          x::XKVector{Complex{T}}, t::Complex{T}, y::XKVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.axpby_sync(n, a, x.data, 1, b, y.data, 1)
Krylov.kaxpby!(n::Integer, s::Complex{T}, x::XKVector{Complex{T}}, b::T,          y::XKVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.axpby_sync(n, a, x.data, 1, b, y.data, 1)
Krylov.kaxpby!(n::Integer, a::T,          x::XKVector{Complex{T}}, b::T,          y::XKVector{Complex{T}}) where T <: AbstractFloat  = XK.BLAS.ext.axpby_sync(n, a, x.data, 1, b, y.data, 1)

Krylov.kfill!(x::XKVector{T}, val::T) where T <: FloatOrComplex = XK.BLAS.ext.fill_sync(length(x), x.data, val)

Krylov.kscal!(n::Integer, s::T, x::XKVector{T}) where T <: BLAS.BlasFloat = XK.BLAS.ext.scal_sync(n, s, x.data, 1)

Krylov.kmul!(y::XKVector{T}, P::SparseMatrixCSR, x::XKVector{T}) where T <: FloatOrComplex = XK.BLAS.spmv_sync(T(1.0), XK.BLAS.NO_TRANS, size(P, 1), size(P, 2), length(P.nzval), XK.BLAS.SPARSE_CSR, P.rowptr, P.colval, P.nzval, x.data, T(0.0), y.data)

Krylov.kcopy!(n::Integer, y::XKVector{T}, x::XKVector{T}) where {T<:BLAS.BlasFloat} = XK.BLAS.ext.copy_sync(n, x.data, 1, y.data, 1)

Krylov.kdot(n::Integer, x::XKVector{T}, y::XKVector{T}) where T <: FloatOrComplex = xkdot(n, x.data, y.data)

Krylov.knorm(n::Integer, x::XKVector{T}) where T <: BLAS.BlasFloat = xknrm2(n, x.data)

# We don't call XK.BLAS in the following routines
Krylov.kscalcopy!(n::Integer, y::XKVector{T}, s::T, x::XKVector{T}) where T <: FloatOrComplex = Krylov.kaxpby!(n, s, x, T(0), y)

Krylov.kdivcopy!(n::Integer, y::XKVector{T}, x::XKVector{T}, s::T) where T <: FloatOrComplex = Krylov.kscalcopy!(n, y, one(T)/s, x)

Krylov.kdiv!(n::Integer, x::XKVector{Complex{T}}, s::T) where T <: AbstractFloat  = Krylov.kscal!(n, one(T) / s, x)
Krylov.kdiv!(n::Integer, x::XKVector{T}, s::T) where T <: FloatOrComplex = Krylov.kscal!(n, one(T) / s, x)

# TODO
# kldiv and kref!
