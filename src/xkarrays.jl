export XKArray, XKVector, XKMatrix, XKVecOrMat

mutable struct XKArray{T,N} <: AbstractArray{T,N}
  data::Array{T,N}
end

const XKVector{T} = XKArray{T,1}
const XKMatrix{T} = XKArray{T,2}
const XKVecOrMat{T} = Union{XKVector{T},XKMatrix{T}}

XKArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} = XKArray{T,N}(Array{T,N}(undef, dims))

# type and dimensionality specified
XKArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} = XKArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
XKArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} = XKArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# type but not dimensionality specified
XKArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} = XKArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
XKArray{T}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} = XKArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

Base.similar(A::XKArray{T,N}) where {T,N} = XKArray{T,N}(undef, size(A))
Base.similar(A::XKArray{T}, dims::Base.Dims{N}) where {T,N} = XKArray{T,N}(undef, dims)
Base.similar(A::XKArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = XKArray{T,N}(undef, dims)

Base.eltype(A::XKArray) = eltype(A.data)
Base.size(A::XKArray) = size(A.data)
Base.getindex(A::XKArray, inds...) = getindex(A.data, inds...)
