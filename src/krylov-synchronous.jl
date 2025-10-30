# gotta init XKBlas
using XKBlas

# Overload Krylov interfaces
using Krylov
import Krylov: FloatOrComplex

function Krylov.kdotr(n::Integer, x::Vector{T}, y::Vector{T}) where {T<:BLAS.BlasFloat}
    r = Ref{T}(0)
    XKBlas.dot(n, x, 1, y, 1, r)
    return r[]
end

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
