###
# PseudoBlockArray apply
###

arguments(LAY, A::PseudoBlockArray) = arguments(LAY, A.blocks)


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

BlockVcat{T,N}(arrays::AbstractArray...) where {T,N} = BlockVcat{T,N,typeof(arrays)}(arrays)
BlockVcat{T}(arrays::AbstractArray{<:Any,N}...) where {T,N} = BlockVcat{T,N}(arrays...)
BlockVcat(arrays::AbstractArray...) = BlockVcat{mapreduce(eltype, promote_type, arrays)}(arrays...)

# all 
_vcat_axes(ax::OneTo{Int}...) = blockedrange(SVector(map(length,ax)...))
_vcat_axes(ax...) = blockedrange(vcat(map(blocklengths,ax)...))
axes(b::BlockVcat{<:Any,1}) = (_vcat_axes(axes.(b.arrays,1)...),)
axes(b::BlockVcat{<:Any,2}) = (_vcat_axes(axes.(b.arrays,1)...),axes(b.arrays[1],2))


_findvcatblock(k) = throw(BoundsError())
function _findvcatblock(k, a, b...)
    n = blocklength(a)
    k ≤ n && return view(a, Block(k))
    _findvcatblock(k - n, b...)
end
viewblock(b::BlockVcat{<:Any,1}, k::Block{1}) = _findvcatblock(Int(k), b.arrays...)
getindex(b::BlockVcat{<:Any,1}, Kk::BlockIndex{1}) = view(b,block(Kk))[blockindex(Kk)]
getindex(b::BlockVcat{<:Any,1}, k::Integer) = b[findblockindex(axes(b,1), k)]

_viewifblocked(::OneTo, a, j) = a
_viewifblocked(_, a, j) = view(a, Block(1,j))
_viewifblocked(a, j) = _viewifblocked(axes(a,2), a, j)
function viewblock(b::BlockVcat{<:Any,2}, kj::Block{2})
    k,j = kj.n
    _viewifblocked(b.arrays[k], j)
end
getindex(b::BlockVcat{<:Any,2}, Kk::BlockIndex{1}, Jj::BlockIndex{1}) = view(b,block(Kk), block(Jj))[blockindex(Kk), blockindex(Jj)]
getindex(b::BlockVcat{<:Any,2}, k::Integer, j::Integer) = b[findblockindex(axes(b,1),k), findblockindex(axes(b,2),j)]

MemoryLayout(::Type{<:BlockVcat}) = ApplyLayout{typeof(vcat)}()
arguments(::ApplyLayout{typeof(vcat)}, b::BlockVcat) = b.arrays


sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractVector, ::Tuple{<:BlockedUnitRange}) =
    BlockVcat(arguments(lay, V)...)

sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:BlockedUnitRange}) = BlockVcat(arguments(lay, V)...)
sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:AbstractUnitRange}) = BlockVcat(arguments(lay, V)...)

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

BlockHcat{T}(arrays::AbstractArray...) where T = BlockHcat{T,typeof(arrays)}(arrays)
BlockHcat(arrays::AbstractArray...) = BlockHcat{mapreduce(eltype, promote_type, arrays)}(arrays...)

axes(b::BlockHcat) = (axes(b.arrays[1],1),_vcat_axes(axes.(b.arrays,2)...))
axes(b::BlockHcat{<:Any, <:Tuple{Vararg{AbstractVector}}}) = (axes(b.arrays[1],1),blockedrange(Ones{Int}(length(b.arrays))))

_hcat_viewifblocked(::OneTo, a::AbstractMatrix, k) = a
_hcat_viewifblocked(::OneTo, a::AbstractVector, k) = a
_hcat_viewifblocked(_, a::AbstractMatrix, k) = view(a, Block(k,1))
_hcat_viewifblocked(_, a::AbstractVector, k) = view(a, Block(k))
_hcat_viewifblocked(a, k) = _hcat_viewifblocked(axes(a,1), a, k)
function viewblock(b::BlockHcat, kj::Block{2})
    k,j = kj.n
    _hcat_viewifblocked(b.arrays[j], k)
end
getindex(b::BlockHcat, Kk::BlockIndex{1}, Jj::BlockIndex{1}) = view(b,block(Kk), block(Jj))[blockindex(Kk), blockindex(Jj)]
getindex(b::BlockHcat, k::Integer, j::Integer) = b[findblockindex(axes(b,1),k), findblockindex(axes(b,2),j)]

MemoryLayout(::Type{<:BlockHcat}) = ApplyLayout{typeof(hcat)}()
arguments(::ApplyLayout{typeof(hcat)}, b::BlockHcat) = b.arrays

sub_materialize(lay::ApplyLayout{typeof(hcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:BlockedUnitRange}) = BlockHcat(arguments(lay, V)...)
sub_materialize(lay::ApplyLayout{typeof(hcat)}, V::AbstractMatrix, ::Tuple{<:AbstractUnitRange,<:BlockedUnitRange}) = BlockHcat(arguments(lay, V)...)

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


##########
# BlockHvcat
##########

struct BlockHvcat{T, Args} <: AbstractBlockMatrix{T}
    n::Int
    args::Args
    function BlockHvcat{T,Args}(n::Int, args::Args) where {T,Args}
        new{T,Args}(n, args)
    end
end

BlockHvcat{T}(n::Int, args...) where T = BlockHvcat{T,typeof(args)}(n, args)
BlockHvcat(n::Int, args...) = BlockHvcat{mapreduce(eltype, promote_type, args)}(n, args...)

axes(b::BlockHvcat) = (_vcat_axes(axes.(b.args[1:b.n:end],1)...),_vcat_axes(axes.(b.args[1:b.n],2)...))


# _hvcat_viewifblocked(::OneTo, a::AbstractMatrix, k) = a
# _hvcat_viewifblocked(::OneTo, a::AbstractVector, k) = a
# _hvcat_viewifblocked(_, a::AbstractMatrix, k) = view(a, Block(k,1))
# _hvcat_viewifblocked(_, a::AbstractVector, k) = view(a, Block(k))
# _hvcat_viewifblocked(a, k) = _hvcat_viewifblocked(axes(a,1), a, k)


function viewblock(b::BlockHvcat, kj::Block{2})
    k,j = kj.n
    b.args[b.n*(k-1) + j] # TODO: blocked
end
getindex(b::BlockHvcat, Kk::BlockIndex{1}, Jj::BlockIndex{1}) = view(b,block(Kk), block(Jj))[blockindex(Kk), blockindex(Jj)]
getindex(b::BlockHvcat, k::Integer, j::Integer) = b[findblockindex(axes(b,1),k), findblockindex(axes(b,2),j)]

# MemoryLayout(::Type{<:BlockHvcat}) = ApplyLayout{typeof(hvcat)}()
# arguments(::ApplyLayout{typeof(hvcat)}, b::BlockHvcat) = b.args

# sub_materialize(lay::ApplyLayout{typeof(hvcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:BlockedUnitRange}) = BlockHvcat(arguments(lay, V)...)
# sub_materialize(lay::ApplyLayout{typeof(hvcat)}, V::AbstractMatrix, ::Tuple{<:AbstractUnitRange,<:BlockedUnitRange}) = BlockHvcat(arguments(lay, V)...)

# function arguments(lay::ApplyLayout{typeof(hvcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:Block1}}})
#     kr, jr = parentindices(V)
#     @assert kr.block ≡ Block(1)
#     arguments(lay, parent(V))[Int.(jr.block)]
# end


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


BlockBroadcastArray{T,N}(f, args...) where {T,N} = BlockBroadcastArray{T,N,typeof(f),typeof(args)}(f, args)
BlockBroadcastArray{T}(bc::Broadcasted) where T = BlockBroadcastArray{T}(bc.f, bc.args...)
BlockBroadcastArray(bc::Broadcasted) = BlockBroadcastArray{eltype(Base.Broadcast.combine_eltypes(bc.f, bc.args))}(bc)
BlockBroadcastArray(f, args...) = BlockBroadcastArray(instantiate(broadcasted(f, args...)))

BlockBroadcastArray{T}(::typeof(hcat), args...) where T = BlockBroadcastMatrix{T}(hcat, args...)
BlockBroadcastArray{T}(::typeof(vcat), args::AbstractVector...) where T = BlockBroadcastVector{T}(vcat, args...)
BlockBroadcastArray{T}(::typeof(vcat), args...) where T = BlockBroadcastMatrix{T}(vcat, args...)
BlockBroadcastArray{T}(::typeof(hvcat), args...) where T = BlockBroadcastMatrix{T}(hvcat, args...)


_block_vcat_axes(ax...) = BlockArrays._BlockedUnitRange(1,+(map(blocklasts,ax)...))

_block_interlace_axes(::Int, ax::Tuple{BlockedUnitRange{OneTo{Int}}}...) = _block_vcat_axes(ax...)

function _block_interlace_axes(nbc::Int, ax::NTuple{2,BlockedUnitRange{OneTo{Int}}}...)
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

viewblock(A::BlockBroadcastVector{<:Any,typeof(vcat)}, k::Block{1}) = Vcat(getindex.(A.args, Int(k))...)
viewblock(A::BlockBroadcastMatrix{<:Any,typeof(hcat)}, kj::Block{2}) = Hcat(getindex.(A.args, kj.n...)...)
viewblock(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}, kj::Block{2}) = hvcat(A.args[1], getindex.(A.args[2:end], kj.n...)...)
blockbandwidths(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}) = max.(map(blockbandwidths,Base.tail(A.args))...)

function subblockbandwidths(B::BlockBroadcastMatrix{<:Any,typeof(hvcat)})
    p = B.args[1]
    ret = p .* subblockbandwidths(B.args[2]) # initialise with first
    shft = 0
    rw = 0
    for a in Base.tail(Base.tail(B.args))
        shft += 1
        if shft == p
            # next row
            rw += 1
            shft = -rw
        end
        ret = max.(ret, p .* subblockbandwidths(a) .+ (-shft,shft))
    end
    ret
end

blockinterlacelayout(_...) = UnknownLayout()
blockinterlacelayout(::Union{ZerosLayout,AbstractBandedLayout}...) = BlockBandedLayout()
MemoryLayout(::Type{<:BlockBroadcastMatrix{<:Any,typeof(hvcat),Arrays}}) where Arrays = blockinterlacelayout(Base.tail(LazyArrays.tuple_type_memorylayouts(Arrays))...)

# temporary hack, need to think of how to flag as lazy for infinite case.
MemoryLayout(::Type{<:BlockBroadcastMatrix{<:Any,typeof(hcat),Arrays}}) where Arrays = LazyLayout()

##
# special for unitblocks
blockbandwidths(A::PseudoBlockMatrix{<:Any,<:Any,<:NTuple{2,BlockedUnitRange{<:AbstractUnitRange{Int}}}}) = bandwidths(A.blocks)
subblockbandwidths(A::PseudoBlockMatrix{<:Any,<:Any,<:NTuple{2,BlockedUnitRange{<:AbstractUnitRange{Int}}}}) = (0,0)


###
# work around bug in dat
###

LazyArrays._lazy_getindex(dat::PseudoBlockArray, kr::UnitRange) = view(dat.blocks,kr)
LazyArrays._lazy_getindex(dat::PseudoBlockArray, kr::OneTo) = view(dat.blocks,kr)