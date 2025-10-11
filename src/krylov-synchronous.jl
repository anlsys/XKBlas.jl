# gotta init XKBlas
using XKBlas

# Overload Krylov interfaces
using Krylov

function Krylov.kdotr(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: AbstractFloat
    r = Ref{T}(0)
    XKBlas.dot(n, x, 1, y, 1, r)
    return r[]
end

function Krylov.kaxpy!(n::Integer, a, x::AbstractVector{T}, y::AbstractVector{T}) where T <: AbstractFloat
    return XKBlas.axpy(n, a, x, 1, y, 1)
end

function Krylov.kaxpby!(n::Integer, a, x::AbstractVector{T}, b, y::AbstractVector{T}) where T <: AbstractFloat
    return XKBlas.axpby(n, a, x, 1, b, y, 1)
end

function Krylov.kcopy!(n::Integer, y::AbstractVector{T}, x::AbstractVector{T}) where T <: AbstractFloat
    return XKBlas.copy(n, x, y)
end
