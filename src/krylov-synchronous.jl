# gotta init XKBlas
using XKBlas

# Overload Krylov interfaces
using Krylov
import Krylov.FloatOrComplex

function Krylov.kdotr(n::Integer, x::AbstractVector{T}, y::AbstractVector{T}) where T <: AbstractFloat
    r = Ref{T}(0)
    XKBlas.dot(n, x, 1, y, 1, r)
    return r[]
end
