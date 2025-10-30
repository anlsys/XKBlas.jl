# gotta init XKBlas
using XKBlas

# Overload Krylov interfaces
using Krylov

function Krylov.kdotr(n::Integer, x::Vector{T}, y::Vector{T}) where {T<:BLAS.BlasFloat}
    r = Ref{T}(0)
    XKBlas.dot(n, x, 1, y, 1, r)
    return r[]
end

function Krylov.kaxpy!(n::Integer, a, x::Vector{T}, y::Vector{T}) where {T<:BLAS.BlasFloat}
    return XKBlas.axpy(n, a, x, 1, y, 1)
end

function Krylov.kaxpby!(n::Integer, a, x::Vector{T}, b, y::Vector{T}) where {T<:BLAS.BlasFloat}
    return XKBlas.axpby(n, a, x, 1, b, y, 1)
end

function Krylov.kcopy!(n::Integer, y::Vector{T}, x::Vector{T}) where {T<:BLAS.BlasFloat}
    return XKBlas.copy(n, x, 1, y, 1)
end
