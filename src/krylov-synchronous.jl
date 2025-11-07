# gotta init XKBlas
using XKBlas

# Init xkblas
XKBlas.init()

# Deinit XKBLAS
function cleanup()
    XKBlas.deinit()
end
atexit(cleanup)

# Overload Krylov interfaces
using Krylov
import Krylov: FloatOrComplex

function kdotr(n::Integer, x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
    r = Ref{T}(0)
    XKBlas.dot(n, x, 1, y, 1, r)
    return r[]
end
Krylov.kdotr(n::Integer, x::AbstractVector{Complex{T}}, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = kdotr(n, x, y)
Krylov.kdotr(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: AbstractFloat = kdotr(n, x, y)

Krylov.kaxpy!(n::Integer, a::T, x::Vector{T}, y::Vector{T}) where T <: BLAS.BlasFloat = XKBlas.axpy(n, a, x, 1, y, 1)
Krylov.kaxpy!(n::Integer, a::T, x::AbstractVector{T}, y::AbstractVector{T}) where T <: FloatOrComplex = XKBlas.axpy(n, a, x, 1, y, 1)
Krylov.kaxpy!(n::Integer, a::T, x::AbstractVector{Complex{T}}, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.axpy(n, a, x, 1, y, 1)

Krylov.kaxpby!(n::Integer, a::T, x::Vector{T}, b::T, y::Vector{T}) where T <: BLAS.BlasFloat = XKBlas.axpby(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, a::T, x::AbstractVector{T}, b::T, y::AbstractVector{T}) where T <: FloatOrComplex = XKBlas.axpby(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, a::T, x::AbstractVector{Complex{T}}, t::Complex{T}, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.axpby(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, s::Complex{T}, x::AbstractVector{Complex{T}}, b::T, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.axpby(n, a, x, 1, b, y, 1)
Krylov.kaxpby!(n::Integer, a::T, x::AbstractVector{Complex{T}}, b::T, y::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.axpby(n, a, x, 1, b, y, 1)

Krylov.kcopy!(n::Integer, y::Vector{T}, x::Vector{T}) where {T<:BLAS.BlasFloat} = XKBlas.copy(n, x, 1, y, 1)
Krylov.kcopy!(n::Integer, y::AbstractVector, x::AbstractVector) = XKBlas.copy(n, x, 1, y, 1)

# no need to impl
Krylov.kdiv!(n::Integer, x::AbstractVector{Complex{T}}, s::T) where T <: AbstractFloat = kscal!(n, one(T) / s, x)
Krylov.kdiv!(n::Integer, x::AbstractVector{T}, s::T) where T <: FloatOrComplex = kscal!(n, one(T) / s, x)

Krylov.kdivcopy!(n::Integer, y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::T) where T <: AbstractFloat = XKBlas.divcopy(n, x, 1, y, 1, s)
Krylov.kdivcopy!(n::Integer, y::AbstractVector{T}, x::AbstractVector{T}, s::T) where T <: FloatOrComplex = XKBlas.divcopy(n, x, 1, y, 1, s)

Krylov.kdot(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: FloatOrComplex = XKBlas.dot(x, 1, y, 1)
Krylov.kdot(n::Integer, x::Vector{T}, y::Vector{T}) where T <: BLAS.BlasComplex = XKBlas.dot(x, 1, y, 1)
Krylov.kdot(n::Integer, x::Vector{T}, y::Vector{T}) where T <: BLAS.BlasReal = XKBlas.dot(x, 1, y, 1)

Krylov.kfill!(x::AbstractArray{T}, val::T) where T <: FloatOrComplex = XKBlas.fill(length(x), x, val)

# TODO
# knorm(n::Integer, x::AbstractVector{T}) where T <: FloatOrComplex = norm(x)
# knorm(n::Integer, x::Vector{T}) where T <: BLAS.BlasFloat = BLAS.nrm2(n, x, 1)

# TODO
# knorm_elliptic(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: FloatOrComplex = (x === y) ? knorm(n, x) : kdotr(n, x, y) |> sqrt

Krylov.kscal!(n::Integer, s::T, x::AbstractVector{Complex{T}}) where T <: AbstractFloat = XKBlas.scal(n, s, x, 1)
Krylov.kscal!(n::Integer, s::T, x::AbstractVector{T}) where T <: FloatOrComplex = XKBlas.scal(n, s, x, 1)
Krylov.kscal!(n::Integer, s::T, x::Vector{T}) where T <: BLAS.BlasFloat = XKBlas.scal(n, s, x, 1)

Krylov.kscalcopy!(n::Integer, y::AbstractVector{Complex{T}}, s::T, x::AbstractVector{Complex{T}}) where T <: AbstractFloat = Krylov.kaxpby!(n, s, x, T(0), y)
Krylov.kscalcopy!(n::Integer, y::AbstractVector{T}, s::T, x::AbstractVector{T}) where T <: FloatOrComplex = Krylov.kaxpby!(n, s, x, T(0), y)
