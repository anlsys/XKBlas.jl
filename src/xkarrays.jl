# Dependencies
using SparseMatricesCSR

# Exports
export XKArray, XKVector, XKMatrix, XKSparseMatrixCSR

# Make XKObject an AbstractArray subtype
mutable struct XKObject{Td,T,N} <: AbstractArray{T,N}
    data::Td
end

# Constructor that infers T and N from the data
XKObject(data::Array{T,N}) where {T,N} = XKObject{Array{T,N},T,N}(data)
XKObject(data::SparseMatrixCSR{Bi,T}) where {Bi,T} = XKObject{SparseMatrixCSR{Bi,T},T,2}(data)

Base.eltype(A::XKObject)                                   = eltype(A.data)
Base.length(A::XKObject)                                   = length(A.data)
Base.size(A::XKObject)                                     = size(A.data)
Base.getindex(A::XKObject, inds...)                        = getindex(A.data, inds...)
Base.setindex!(A::XKObject, val, inds...)                  = setindex!(A.data, val, inds...)
Base.pointer(A::XKObject)                                  = pointer(A.data)
Base.cconvert(::Type{Ptr{T}}, A::XKObject) where {T}       = A.data
Base.unsafe_convert(::Type{Ptr{T}}, A::XKObject) where {T} = Base.unsafe_convert(Ptr{T}, A.data)

const XKArray{T,N}         = XKObject{Array{T,N},T,N}
const XKVector{T}          = XKObject{Vector{T},T,1}
const XKMatrix{T}          = XKObject{Matrix{T},T,2}
const XKSparseMatrixCSR{T} = XKObject{SparseMatrixCSR,T,2}

XKArray(data::Array{T,N}) where {T,N}    = XKObject(data)
XKVector(data::Vector{T}) where {T}      = XKObject(data)
XKMatrix(data::Matrix{T}) where {T}      = XKObject(data)
XKSparseMatrixCSR(data::SparseMatrixCSR) = XKObject(data)

# type but not dimensionality specified
XKObject{Td,T,N}(::UndefInitializer, dims::Dims{N})           where {Td,T,N} = XKObject{Td,T,N}(Td(undef, dims))
XKObject{Td,T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {Td,T,N} = XKObject{Td,T,N}(undef, Dims{N}(dims))
XKObject{Td,T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {Td,T,N} = XKObject{Td,T,N}(undef, Dims{N}(dims))
