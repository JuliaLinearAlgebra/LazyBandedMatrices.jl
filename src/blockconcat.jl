##########
# BlockVcat
##########

struct BlockVcat{T, N, Arrays} <: AbstractBlockArray{T,N}
    arrays::Arrays
    function BlockVcat{T,N,Arrays}(arrays::Arrays) where {T,N,Arrays}
        blockisequal(axes.(arrays,2)...) || throw(ArgumentError("Blocks must match"))
        new{T,N,Arrays}(arrays)
    end
end

BlockVcat{T,N}(arrays::AbstractArray...) where {T,N} =
    BlockVcat{T,N,typeof(arrays)}(arrays)
BlockVcat{T}(arrays::AbstractArray{<:Any,N}...) where {T,N} =
    BlockVcat{T,N}(arrays...)
BlockVcat(arrays::AbstractArray...) =
    BlockVcat{mapreduce(eltype, promote_type, arrays)}(arrays...)

axes(b::BlockVcat{<:Any,1}) = (blockedrange(SVector(length.(b.arrays)...)),)
axes(b::BlockVcat{<:Any,2}) = (blockedrange(SVector(size.(b.arrays,1)...)),axes(b.arrays[1],2))

getblock(b::BlockVcat{<:Any,1}, k::Integer) = b.arrays[k]
getindex(b::BlockVcat{<:Any,1}, Kk::BlockIndex{1}) = getblock(b,Int(block(Kk)))[blockindex(Kk)]
getindex(b::BlockVcat{<:Any,1}, k::Integer) = b[findblockindex(axes(b,1), k)]

_viewifblocked(::OneTo, a, j) = a
_viewifblocked(_, a, j) = view(a, Block(1,j))
_viewifblocked(a, j) = _viewifblocked(axes(a,2), a, j)
getblock(b::BlockVcat{<:Any,2}, k::Integer, j::Integer) = _viewifblocked(b.arrays[k], j)
getindex(b::BlockVcat{<:Any,2}, Kk::BlockIndex{1}, Jj::BlockIndex{1}) = getblock(b,Int(block(Kk)), Int(block(Jj)))[blockindex(Kk), blockindex(Jj)]
getindex(b::BlockVcat{<:Any,2}, k::Integer, j::Integer) = b[findblockindex(axes(b,1),k), findblockindex(axes(b,2),j)]

MemoryLayout(::Type{<:BlockVcat}) = ApplyLayout{typeof(vcat)}()
arguments(::ApplyLayout{typeof(vcat)}, b::BlockVcat) = b.arrays

sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractVector, ::Tuple{<:BlockedUnitRange}) =
    BlockVcat(arguments(lay, V)...)

sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:BlockedUnitRange}) =
    BlockVcat(arguments(lay, V)...)
sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:AbstractUnitRange}) =
    BlockVcat(arguments(lay, V)...)
sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, ::Tuple{<:AbstractUnitRange,<:BlockedUnitRange}) =
    BlockVcat(arguments(lay, V)...)

LazyArrays._vcat_sub_arguments(lay::ApplyLayout{typeof(vcat)}, A, V, kr::BlockSlice{<:BlockRange{1}}) =
    arguments(lay, A)[Int.(kr.block)]

function arguments(lay::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:Block1}}})
    kr, jr = parentindices(V)
    @assert jr.block ≡ Block(1)
    arguments(lay, parent(V))[Int.(kr.block)]
end

##########
# BlockHcat
##########

struct BlockHcat{T, Arrays} <: AbstractBlockMatrix{T}
    arrays::Arrays
    function BlockHcat{T,Arrays}(arrays::Arrays) where {T,Arrays}
        blockisequal(axes.(arrays,1)...) || throw(ArgumentError("Blocks must match"))
        new{T,Arrays}(arrays)
    end
end

BlockHcat{T}(arrays::AbstractArray...) where T =
    BlockHcat{T,typeof(arrays)}(arrays)
BlockHcat(arrays::AbstractArray...) =
    BlockHcat{mapreduce(eltype, promote_type, arrays)}(arrays...)

axes(b::BlockHcat{<:Any}) = (axes(b.arrays[1],1),blockedrange(SVector(size.(b.arrays,2)...)))

_hcat_viewifblocked(::OneTo, a::AbstractMatrix, k) = a
_hcat_viewifblocked(::OneTo, a::AbstractVector, k) = a
_hcat_viewifblocked(_, a::AbstractMatrix, k) = view(a, Block(k,1))
_hcat_viewifblocked(_, a::AbstractVector, k) = view(a, Block(k))
_hcat_viewifblocked(a, k) = _hcat_viewifblocked(axes(a,1), a, k)
getblock(b::BlockHcat{<:Any}, k::Integer, j::Integer) = _hcat_viewifblocked(b.arrays[j], k)
getindex(b::BlockHcat{<:Any}, Kk::BlockIndex{1}, Jj::BlockIndex{1}) = getblock(b,Int(block(Kk)), Int(block(Jj)))[blockindex(Kk), blockindex(Jj)]
getindex(b::BlockHcat{<:Any}, k::Integer, j::Integer) = b[findblockindex(axes(b,1),k), findblockindex(axes(b,2),j)]

MemoryLayout(::Type{<:BlockHcat}) = ApplyLayout{typeof(hcat)}()
arguments(::ApplyLayout{typeof(hcat)}, b::BlockHcat) = b.arrays

sub_materialize(lay::ApplyLayout{typeof(hcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:BlockedUnitRange}) =
    BlockHcat(arguments(lay, V)...)
sub_materialize(lay::ApplyLayout{typeof(hcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:AbstractUnitRange}) =
    BlockHcat(arguments(lay, V)...)
sub_materialize(lay::ApplyLayout{typeof(hcat)}, V::AbstractMatrix, ::Tuple{<:AbstractUnitRange,<:BlockedUnitRange}) =
    BlockHcat(arguments(lay, V)...)

function arguments(lay::ApplyLayout{typeof(hcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:Block1}}})
    kr, jr = parentindices(V)
    @assert kr.block ≡ Block(1)
    arguments(lay, parent(V))[Int.(jr.block)]
end

for adj in (:adjoint, :transpose)
    @eval begin
        $adj(A::BlockHcat{T}) where T = BlockVcat{T}(map($adj,A.arrays)...)
        $adj(A::BlockVcat{T,2}) where T = BlockHcat{T}(map($adj,A.arrays)...)
    end
end

#############
# BlockApplyArray
#############

"""
   BlockBroadcastArray(f, A, B, C...)

is a block array corresponding to `f.(blocks(A), blocks(B), ...)`,
except if `A` is scalar. 
"""

struct BlockBroadcastArray{T, N, FF, Args} <: AbstractBlockArray{T, N}
    f::FF
    args::Args
end

const BlockBroadcastVector{T, FF, Args} = BlockBroadcastArray{T, 1, FF, Args}
const BlockBroadcastMatrix{T, FF, Args} = BlockBroadcastArray{T, 2, FF, Args}

_blocks(A::AbstractArray) = blocks(A)
_blocks(A::Number) = A


BlockBroadcastArray{T,N}(f, args...) where {T,N} =
    BlockBroadcastArray{T,N,typeof(f),typeof(args)}(f, args)
BlockBroadcastArray{T}(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Tuple{Vararg{Any,N}},<:Any,<:Tuple}) where {T,N} = 
    BlockBroadcastArray{T,N}(bc.f, bc.args...)
BlockBroadcastArray(bc::Broadcasted) = 
    BlockBroadcastArray{eltype(Base.Broadcast.combine_eltypes(bc.f, bc.args))}(bc)
BlockBroadcastArray(f, args...) = BlockBroadcastArray(instantiate(broadcasted(f, args...)))



_block_vcat_axes(ax...) = BlockArrays._BlockedUnitRange(1,+(map(blocklasts,ax)...))

_block_interlace_axes(::Int, ax::Tuple{OneTo{Int}}...) = _block_vcat_axes(ax...)

function _block_interlace_axes(nbc::Int, ax::NTuple{2,OneTo{Int}}...)
    n,m = max(length.(first.(ax))...),max(length.(last.(ax))...)
    (blockedrange(Fill(length(ax) ÷ nbc, n)),blockedrange(Fill(mod1(length(ax),nbc), m)))
end

axes(A::BlockBroadcastVector{<:Any,typeof(vcat)}) = (_block_vcat_axes(axes.(A.args,1)...),)
axes(A::BlockBroadcastMatrix{<:Any,typeof(hcat)}) = (axes(A.args[1],1), _block_vcat_axes(axes.(A.args,2)...))

axes(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}) = _block_interlace_axes(A.args[1], map(axes,A.args[2:end])...)
# size(A::BlockBroadcastArray) = map(length, axes(A))

function getindex(A::BlockBroadcastVector{<:Any,typeof(vcat)}, k::Int)
    K = findblockindex(axes(A,1), k)
    A.args[blockindex(K)][Int(block(K))]
end

function getindex(A::BlockBroadcastMatrix{<:Any,typeof(hcat)}, k::Int, j::Int)
    J = findblockindex(axes(A,2), j)
    A.args[blockindex(J)][k,Int(block(J))]
end

function getindex(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}, k::Int, j::Int)
    K,J = findblockindex.(axes(A), (k,j))
    A.args[(blockindex(K)-1)*A.args[1] + blockindex(J)+1][Int(block(K)), Int(block(J))]
end

BlockArrays.getblock(A::BlockBroadcastVector{<:Any,typeof(vcat)}, k::Int) = Vcat(getindex.(A.args, k)...)
BlockArrays.getblock(A::BlockBroadcastMatrix{<:Any,typeof(hcat)}, k::Int, j::Int) = Hcat(getindex.(A.args, k, j)...)
BlockArrays.getblock(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}, k::Int, j::Int) = hvcat(A.args[1], getindex.(A.args[2:end], k, j)...)
# blockbandwidths(A::BlockBroadcastArray{<:Any,2}) = max.(map(bandwidths,A.args)...)
# subblockbandwidths(A::BlockBroadcastArray{<:Any,2}) = length(axes(A,1)[Block(1)]),length(axes(A,2)[Block(2)])

# blockinterlacelayout(_...) = UnknownLayout()
# blockinterlacelayout(::Union{ZerosLayout,AbstractBandedLayout}...) = BlockBandedLayout()
# MemoryLayout(::Type{<:BlockBroadcastArray{<:Any,2,Arrays}}) where Arrays = blockinterlacelayout(LazyArrays.tuple_type_memorylayouts(Arrays)...)