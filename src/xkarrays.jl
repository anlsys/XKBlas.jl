# Dependencies
using SparseMatricesCSR

# Exports
export XKArray, XKVector, XKMatrix, XKSparseMatrixCSR

# Make XKObject an AbstractArray subtype
mutable struct XKObject{T,N,Td} <: AbstractArray{T,N}
    data::Td
end

# Constructor that infers T and N from the data
XKObject(data::Array{T,N})            where {T,N}  = XKObject{T,N,Array{T,N}}(data)
XKObject(data::SparseMatrixCSR{Bi,T}) where {Bi,T} = XKObject{T,2,SparseMatrixCSR{Bi,T}}(data)

# Base functions forwarding
Base.eltype(A::XKObject)                                   = eltype(A.data)
Base.length(A::XKObject)                                   = length(A.data)
Base.size(A::XKObject)                                     = size(A.data)
Base.getindex(A::XKObject, inds...)                        = getindex(A.data, inds...)
Base.setindex!(A::XKObject, val, inds...)                  = setindex!(A.data, val, inds...)
Base.pointer(A::XKObject)                                  = pointer(A.data)
Base.cconvert(::Type{Ptr{T}}, A::XKObject) where {T}       = Base.cconvert(Ptr{T}, A.data)
Base.unsafe_convert(::Type{Ptr{T}}, A::XKObject) where {T} = Base.unsafe_convert(Ptr{T}, A.data)

# Types
const XKArray{T,N}                          = XKObject{T,N,Array{T,N}}
const XKVector{T}                           = XKObject{T,1,Vector{T}}
const XKMatrix{T}                           = XKObject{T,2,Matrix{T}}
const XKSparseMatrixCSR{Bi,Tv,Ti<:Integer}  = XKObject{Tv,2,SparseMatrixCSR{Bi,Tv,Ti}}

# Constructors
XKArray(data::Array{T,N})                          where {T,N}               = XKObject{T,N,Array{T,N}}(data)
XKVector(data::Vector{T})                          where {T}                 = XKObject{T,1,Vector{T}}(data)
XKMatrix(data::Matrix{T})                          where {T}                 = XKObject{T,2,Matrix{T}}(data)
XKSparseMatrixCSR(data::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti<:Integer} = XKObject{Tv,2,SparseMatrixCSR{Bi,Tv,Ti}}(data)

# type but not dimensionality specified
XKObject{T,N,Td}(::UndefInitializer, dims::Dims{N})           where {T,N,Td} = XKObject{T,N,Td}(Td(undef, dims))
XKObject{T,N,Td}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,Td} = XKObject{T,N,Td}(undef, Dims{N}(dims))
XKObject{T,N,Td}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,Td} = XKObject{T,N,Td}(undef, Dims{N}(dims))
