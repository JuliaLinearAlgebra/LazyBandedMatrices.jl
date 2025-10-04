

##########
# BlockVcat
##########

struct BlockVcat{T, N, Arrays, Axes} <: AbstractBlockArray{T,N}
    arrays::Arrays
    axes::Axes
end

function BlockVcat{T,N,Arrays}(arrays::Arrays) where {T,N,Arrays}
    length(arrays) == 1 || blockisequal(axes.(arrays,2)...) || throw(ArgumentError("Blocks must match"))
    ax = _block_vcat_axes(Val{N}(), arrays...)
    BlockVcat{T,N,Arrays, typeof(ax)}(arrays, ax)
end

BlockVcat{T,N}(arrays::AbstractArray...) where {T,N} = BlockVcat{T,N,typeof(arrays)}(arrays)
BlockVcat{T}(arrays::AbstractArray{<:Any,N}...) where {T,N} = BlockVcat{T,N}(arrays...)
BlockVcat(arrays::AbstractArray...) = BlockVcat{mapreduce(eltype, promote_type, arrays)}(arrays...)
blockvcat(a) = a
blockvcat(a, b...) = BlockVcat(a, b...)


# use integers if possible
@inline _blocklengths(ax::OneTo) = length(ax)
@inline _blocklengths(ax) = blocklengths(ax)

@inline __vcat_axes(bls::Integer...) = blockedrange(SVector(bls...))
@inline __vcat_axes(bls...) = blockedrange(Vcat{Int}(bls...))
@inline _vcat_axes_args_1() = ()
@inline _vcat_axes_args_1(a, arrys...) = (_blocklengths(axes(a,1)), _vcat_axes_args_1(arrys...)...)
@inline _vcat_axes_args_2() = ()
@inline _vcat_axes_args_2(a, arrys...) = (_blocklengths(axes(a,2)), _vcat_axes_args_2(arrys...)...)
_vcat_axes_1(a...) = __vcat_axes(_vcat_axes_args_1(a...)...)
_vcat_axes_2(a...) = __vcat_axes(_vcat_axes_args_2(a...)...)
@inline _block_vcat_axes(::Val{1}, a...) = (_vcat_axes_1(a...),)
@inline _block_vcat_axes(::Val{2}, a, b...) = (_vcat_axes_1(a, b...),axes(a,2))

axes(b::BlockVcat) = b.axes

copy(b::BlockVcat{T,N}) where {T,N} = BlockVcat{T,N}(map(copy, b.arrays)...)
copy(b::AdjOrTrans{<:Any,<:BlockVcat}) = copy(parent(b))'
AbstractArray{T}(B::BlockVcat{<:Any,N}) where {T,N} = BlockVcat{T,N}(map(AbstractArray{T}, B.arrays)...)
AbstractArray{T,N}(B::BlockVcat{<:Any,N}) where {T,N} = BlockVcat{T,N}(map(AbstractArray{T}, B.arrays)...)
convert(::Type{AbstractArray{T}}, B::BlockVcat{<:Any,N}) where {T,N} = BlockVcat{T,N}(convert.(AbstractArray{T}, B.arrays)...)
convert(::Type{AbstractArray{T,N}}, B::BlockVcat{<:Any,N}) where {T,N} = BlockVcat{T,N}(convert.(AbstractArray{T}, B.arrays)...)
convert(::Type{AbstractArray{T}}, B::BlockVcat{T,N}) where {T,N} = B
convert(::Type{AbstractArray{T,N}}, B::BlockVcat{T,N}) where {T,N} = B


# avoid making naive view
_viewifblocked(::Tuple{Vararg{OneTo}}, a, kj::Block{1}) = a
_viewifblocked(::Tuple{Vararg{OneTo}}, a, kj::Block{2}) = a
_viewifblocked(::Tuple{OneTo}, a::AbstractVector, kj::Block{2}) = a
_viewifblocked(_, a, kj::Block) = view(a, kj)
function _viewifblocked(_, a::AbstractVector, kj::Block{2})
    k,j = kj.n
    @assert j == 1
    view(a, Block(k))
end
_viewifblocked(a, kj) = _viewifblocked(axes(a), a, kj)

_findvcatblock(k) = throw(BoundsError())

function _findvcatblock(k::Block{1}, a, b...)
    n = Block(blocklength(a))
    k ≤ n && return _viewifblocked(a, k)
    _findvcatblock(k - n, b...)
end

function _findvcatblock(kj::Block{2}, a, b...)
    k,j= kj.n
    n = blocksize(a,1)
    k ≤ n && return _viewifblocked(a, kj)
    _findvcatblock(Block(k-n, j), b...)
end

viewblock(b::BlockVcat, k::Block) = _findvcatblock(k, b.arrays...)
getindex(b::BlockVcat, Kk::BlockIndex{1}) = view(b,block(Kk))[Kk.α...]
getindex(b::BlockVcat, Kk::BlockIndex{2}) = view(b,block(Kk))[Kk.α...]
getindex(b::BlockVcat{<:Any,1}, k::Integer) = b[findblockindex(axes(b,1), k)]
getindex(b::BlockVcat{<:Any,2}, k::Integer, j::Integer) = b[findblockindex(axes(b,1),k), findblockindex(axes(b,2),j)]

MemoryLayout(::Type{<:BlockVcat}) = ApplyLayout{typeof(vcat)}()
arguments(::ApplyLayout{typeof(vcat)}, b::BlockVcat) = b.arrays


##########
# BlockHcat
##########

struct BlockHcat{T, Arrays, Axes} <: AbstractBlockMatrix{T}
    arrays::Arrays
    axes::Axes
end

function BlockHcat{T,Arrays}(arrays::Arrays) where {T,Arrays}
    length(arrays) == 1 || blockisequal(axes.(arrays,1)...) || throw(ArgumentError("Blocks must match"))
    ax = _block_hcat_axes(arrays...)
    BlockHcat{T,Arrays,typeof(ax)}(arrays, ax)
end

BlockHcat{T}(arrays::AbstractArray...) where T = BlockHcat{T,typeof(arrays)}(arrays)
BlockHcat(arrays::AbstractArray...) = BlockHcat{mapreduce(eltype, promote_type, arrays)}(arrays...)
blockhcat(a) = a
blockhcat(a, b...) = BlockHcat(a, b...)

_block_hcat_axes(a, b...) = (axes(a,1),_vcat_axes_2(a, b...))
_block_hcat_axes(a::AbstractVector, b::AbstractVector...) = (axes(a,1),blockedrange(Ones{Int}(length(b)+1)))
axes(b::BlockHcat) = b.axes

copy(b::BlockHcat{T}) where T = BlockHcat{T}(map(copy, b.arrays)...)
copy(b::AdjOrTrans{<:Any,<:BlockHcat}) = copy(parent(b))'
AbstractArray{T}(B::BlockHcat) where T = BlockHcat{T}(map(AbstractArray{T}, B.arrays)...)
AbstractMatrix{T}(B::BlockHcat) where T = BlockHcat{T}(map(AbstractArray{T}, B.arrays)...)
convert(::Type{AbstractArray{T}}, B::BlockHcat) where T = BlockHcat{T}(convert.(AbstractArray{T}, B.arrays)...)
convert(::Type{AbstractMatrix{T}}, B::BlockHcat) where T = BlockHcat{T}(convert.(AbstractArray{T}, B.arrays)...)
convert(::Type{AbstractArray{T}}, B::BlockHcat{T}) where T = B
convert(::Type{AbstractMatrix{T}}, B::BlockHcat{T}) where T = B

_blocksize2(a::AbstractVector) = 1
_blocksize2(a) = blocksize(a,2)

function _findhcatblock(kj::Block{2}, a, b...)
    k,j = kj.n
    n = _blocksize2(a)
    j ≤ n && return _viewifblocked(a, kj)
    _findhcatblock(Block(k, j-n), b...)
end

viewblock(b::BlockHcat, kj::Block{2}) = _findhcatblock(kj, b.arrays...)
getindex(b::BlockHcat, Kk::BlockIndex{1}, Jj::BlockIndex{1}) = view(b,block(Kk), block(Jj))[blockindex(Kk), blockindex(Jj)]
getindex(b::BlockHcat, k::Integer, j::Integer) = b[findblockindex(axes(b,1),k), findblockindex(axes(b,2),j)]

_blockhcat_getindex_args(::Tuple{}, _, ::Tuple{}) = ()
@inline _blockhcat_getindex_args(args::Tuple, kr, jrs::Tuple) = (args[1][kr, jrs[1]], _blockhcat_getindex_args(tail(args), kr, tail(jrs))...)


function _blockhcat_blockrange_getindex(A, KR, JR)
    args = A.arrays
    sz = blocksize.(args,2)
    sjr = intersect.(LazyArrays._argsindices(sz), Ref(Int.(JR)))
    sjr2 = broadcast((a,b) -> a .- b .+ 1, sjr, LazyArrays._vcat_firstinds(sz))
    BlockHcat(_blockhcat_getindex_args(args, KR, broadcast(jr -> Block.(jr), sjr2))...)
end

getindex(A::BlockHcat, ::Colon, JR::BlockRange{1}) = _blockhcat_blockrange_getindex(A, :, JR)
getindex(A::BlockHcat, KR::BlockRange{1}, JR::BlockRange{1}) = _blockhcat_blockrange_getindex(A, KR, JR)

arguments(::ApplyLayout{typeof(hcat)}, b::BlockHcat) = b.arrays

sub_materialize(lay::ApplyLayout{typeof(hcat)}, V::AbstractMatrix, ::Tuple{<:AbstractBlockedUnitRange,<:AbstractBlockedUnitRange}) = blockhcat(arguments(lay, V)...)
sub_materialize(lay::ApplyLayout{typeof(hcat)}, V::AbstractMatrix, ::Tuple{<:AbstractBlockedUnitRange,<:AbstractUnitRange}) = blockhcat(arguments(lay, V)...)
sub_materialize(lay::ApplyLayout{typeof(hcat)}, V::AbstractMatrix, ::Tuple{<:AbstractUnitRange,<:AbstractBlockedUnitRange}) = blockhcat(arguments(lay, V)...)

sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractVector, ::Tuple{<:AbstractBlockedUnitRange}) = blockvcat(sub_materialize.(arguments(lay, V))...)
sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, ::Tuple{<:AbstractBlockedUnitRange,<:AbstractBlockedUnitRange}) = blockvcat(sub_materialize.(arguments(lay, V))...)
sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, ::Tuple{<:AbstractBlockedUnitRange,<:AbstractUnitRange}) = blockvcat(sub_materialize.(arguments(lay, V))...)



for adj in (:adjoint, :transpose)
    @eval begin
        $adj(A::BlockHcat{T}) where T = BlockVcat{T}(map($adj,A.arrays)...)
        $adj(A::BlockVcat{T,2}) where T = BlockHcat{T}(map($adj,A.arrays)...)
    end
end


####
# copyto!
####

sublayout(::ApplyBlockBandedLayout{typeof(hcat)}, ::Type{<:Tuple{<:BlockSlice{<:BlockRange1}, <:BlockSlice{<:BlockRange1}}}) = ApplyBlockBandedLayout{typeof(hcat)}()

_copyto!(_, LAY::ApplyBlockBandedLayout{typeof(hcat)}, dest::AbstractMatrix, H::AbstractMatrix) =
    block_hcat_copyto!(dest, arguments(LAY,H)...)
function block_hcat_copyto!(dest::AbstractMatrix, arrays...)
    nrows = blocksize(dest, 1)
    ncols = 0
    dense = true
    for a in arrays
        dense &= isa(a,Array)
        nd = ndims(a)
        ncols += (nd==2 ? blocksize(a,2) : 1)
    end

    nrows == blocksize(first(arrays),1) || throw(DimensionMismatch("Destination rows must match"))
    ncols == blocksize(dest,2) || throw(DimensionMismatch("Destination columns must match"))

    pos = 1
    for a in arrays
        p1 = pos+(isa(a,AbstractMatrix) ? blocksize(a, 2) : 1)-1
        copyto!(view(dest,:, Block.(pos:p1)), a)
        pos = p1+1
    end
    return dest
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

axes(b::BlockHvcat) = (_vcat_axes_1(b.args[1:b.n:end]...),_vcat_axes_2(b.args[1:b.n]...))

copy(b::BlockHvcat{T}) where T = BlockHvcat{T}(b.n, map(copy, b.args)...)
copy(b::AdjOrTrans{<:Any,<:BlockHvcat}) = copy(parent(b))'

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

# sub_materialize(lay::ApplyLayout{typeof(hvcat)}, V::AbstractMatrix, ::Tuple{<:AbstractBlockedUnitRange,<:AbstractBlockedUnitRange}) = BlockHvcat(arguments(lay, V)...)
# sub_materialize(lay::ApplyLayout{typeof(hvcat)}, V::AbstractMatrix, ::Tuple{<:AbstractUnitRange,<:AbstractBlockedUnitRange}) = BlockHvcat(arguments(lay, V)...)

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


blockbroadcast_eltype(f, args...) = eltype(Base.Broadcast.combine_eltypes(f, args))

BlockBroadcastArray(bc::Broadcasted) = BlockBroadcastArray{blockbroadcast_eltype(bc.f, bc.args...)}(bc)
BlockBroadcastArray(f, args...) = BlockBroadcastArray(instantiate(broadcasted(f, args...)))

BlockBroadcastArray{T}(::typeof(hcat), args...) where T = BlockBroadcastMatrix{T}(hcat, args...)
BlockBroadcastArray{T}(::typeof(vcat), args::AbstractVector...) where T = BlockBroadcastVector{T}(vcat, args...)
BlockBroadcastArray{T}(::typeof(vcat), args...) where T = BlockBroadcastMatrix{T}(vcat, args...)
BlockBroadcastArray{T}(::typeof(hvcat), args...) where T = BlockBroadcastMatrix{T}(hvcat, args...)
BlockBroadcastArray{T}(::typeof(Diagonal), args...) where T = BlockBroadcastMatrix{T}(Diagonal, args...)


_block_vcat_axes(ax...) = BlockArrays.BlockedOneTo(+(map(blocklasts,ax)...))

_block_interlace_axes(::Int, ax::Tuple{BlockedOneTo{Int,OneTo{Int}}}...) = _block_vcat_axes(ax...)

function _block_interlace_axes(nbc::Int, ax::NTuple{2,BlockedOneTo{Int,OneTo{Int}}}...)
    n,m = max(length.(first.(ax))...),max(length.(last.(ax))...)
    (blockedrange(Fill(length(ax) ÷ nbc, n)),blockedrange(Fill(mod1(length(ax),nbc), m)))
end

blockbroadcast_eltype(::typeof(hvcat), ::Integer, args::AbstractArray...) = promote_type(map(eltype, args)...)

axes(A::BlockBroadcastVector{<:Any,typeof(vcat)}) = (_block_vcat_axes(axes.(A.args,1)...),)
axes(A::BlockBroadcastMatrix{<:Any,typeof(hcat)}) = (axes(A.args[1],1), _block_vcat_axes(axes.(A.args,2)...))

axes(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}) = _block_interlace_axes(A.args[1], map(axes,A.args[2:end])...)
axes(A::BlockBroadcastMatrix{<:Any,typeof(Diagonal)}) = (_block_vcat_axes(axes.(A.args,1)...), _block_vcat_axes(axes.(A.args,2)...))
# size(A::BlockBroadcastArray) = map(length, axes(A))

copy(b::BlockBroadcastArray{T,N}) where {T,N} = BlockBroadcastArray{T,N}(b.f, map(copy, b.args)...)
copy(b::AdjOrTrans{<:Any,<:BlockBroadcastArray}) = copy(parent(b))'

AbstractArray{T}(B::BlockBroadcastArray{<:Any,N}) where {T,N} = BlockBroadcastArray{T,N}(B.f, map(AbstractArray{T}, B.args)...)
AbstractArray{T,N}(B::BlockBroadcastArray{<:Any,N}) where {T,N} = BlockBroadcastArray{T,N}(B.f, map(AbstractArray{T}, B.args)...)
convert(::Type{AbstractArray{T}}, B::BlockBroadcastArray{<:Any,N}) where {T,N} = BlockBroadcastArray{T,N}(B.f, convert.(AbstractArray{T}, B.args)...)
convert(::Type{AbstractArray{T,N}}, B::BlockBroadcastArray{<:Any,N}) where {T,N} = BlockBroadcastArray{T,N}(B.f, convert.(AbstractArray{T}, B.args)...)
convert(::Type{AbstractArray{T}}, B::BlockBroadcastArray{T,N}) where {T,N} = B
convert(::Type{AbstractArray{T,N}}, B::BlockBroadcastArray{T,N}) where {T,N} = B

AbstractArray{T}(B::BlockBroadcastArray{<:Any,N,typeof(hvcat)}) where {T,N} = BlockBroadcastArray{T,N}(B.f, first(B.args), map(AbstractArray{T}, tail(B.args))...)
AbstractArray{T,N}(B::BlockBroadcastArray{<:Any,N,typeof(hvcat)}) where {T,N} = BlockBroadcastArray{T,N}(B.f, first(B.args), map(AbstractArray{T}, tail(B.args))...)
convert(::Type{AbstractArray{T}}, B::BlockBroadcastArray{<:Any,N,typeof(hvcat)}) where {T,N} = BlockBroadcastArray{T,N}(B.f, first(B.args), convert.(AbstractArray{T}, tail(B.args))...)
convert(::Type{AbstractArray{T,N}}, B::BlockBroadcastArray{<:Any,N,typeof(hvcat)}) where {T,N} = BlockBroadcastArray{T,N}(B.f, first(B.args), convert.(AbstractArray{T}, tail(B.args))...)
convert(::Type{AbstractArray{T}}, B::BlockBroadcastArray{T,N,typeof(hvcat)}) where {T,N} = B
convert(::Type{AbstractArray{T,N}}, B::BlockBroadcastArray{T,N,typeof(hvcat)}) where {T,N} = B

Base.BroadcastStyle(::Type{<:BlockBroadcastArray{T,N}}) where {T,N} = LazyArrayStyle{N}()
Base.BroadcastStyle(::Type{<:BlockVcat{T,N}}) where {T,N} = LazyArrayStyle{N}()
Base.BroadcastStyle(::Type{<:BlockHcat}) = LazyArrayStyle{2}()
Base.BroadcastStyle(::Type{<:BlockHvcat}) = LazyArrayStyle{2}()

Base.BroadcastStyle(::Type{<:AdjOrTrans{<:Any,<:BlockBroadcastArray}}) = LazyArrayStyle{2}()
Base.BroadcastStyle(::Type{<:AdjOrTrans{<:Any,<:BlockVcat}}) = LazyArrayStyle{2}()
Base.BroadcastStyle(::Type{<:AdjOrTrans{<:Any,<:BlockHcat}}) = LazyArrayStyle{2}()
Base.BroadcastStyle(::Type{<:AdjOrTrans{<:Any,<:BlockHvcat}}) = LazyArrayStyle{2}()


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

function getindex(A::BlockBroadcastMatrix{T,typeof(Diagonal)}, k::Int, j::Int) where T
    K,J = findblockindex.(axes(A), (k,j))
    blockindex(K) == blockindex(J) || return zero(T)
    A.args[blockindex(K)][Int(block(K)),Int(block(J))]
end

_view_blockvec(A, k) = view(A, k)
_view_blockvec(A::AbstractVector, k::Block{2}) = view(A, Block(k.n[1]))

viewblock(A::BlockBroadcastVector{<:Any,typeof(vcat)}, k::Block{1}) = Vcat(getindex.(A.args, Int(k))...)
viewblock(A::BlockBroadcastMatrix{<:Any,typeof(hcat)}, kj::Block{2}) = Hcat(_view_blockvec.(A.args, Ref(kj))...)
viewblock(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}, kj::Block{2}) = ApplyArray(hvcat, first(A.args), getindex.(tail(A.args), kj.n...)...)


###
# blockbandwidths
###
blockbandwidths(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}) = max.(map(blockbandwidths,Base.tail(A.args))...)
blockbandwidths(A::BlockBroadcastMatrix{<:Any,typeof(Diagonal)}) = max.(map(blockbandwidths,A.args)...)
subblockbandwidths(A::BlockBroadcastMatrix{<:Any,typeof(Diagonal)}) = (0,0)

function blockbandwidths(M::BlockVcat{<:Any,2})
    cs = tuple(0, _cumsum(blocksize.(M.arrays[1:end-1],1)...)...) # cumsum of sizes
    (maximum(cs .+ blockbandwidth.(M.arrays,1)), maximum(blockbandwidth.(M.arrays,2) .- cs))
end
isblockbanded(M::BlockVcat{<:Any,2}) = all(isblockbanded, M.arrays)

function blockbandwidths(M::BlockHcat)
    cs = tuple(0, _cumsum(blocksize.(M.arrays[1:end-1],2)...)...) # cumsum of sizes
    (maximum(blockbandwidth.(M.arrays,1) .- cs), maximum(blockbandwidth.(M.arrays,2) .+ cs))
end
isblockbanded(M::BlockHcat) = all(isblockbanded, M.arrays)

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


###
# MemoryLayout
#
# sometimes we get block/banded when concatenting block/banded matrices
###

blockhcatlayout(_...) = ApplyLayout{typeof(hcat)}()
# at the moment we just support hcat for a special case of a subview of Eye concatenated with a block banded.
# This can be generalised later as needed
blockhcatlayout(::AbstractBandedLayout, ::AbstractBlockBandedLayout) = ApplyBlockBandedLayout{typeof(hcat)}()
MemoryLayout(::Type{<:BlockHcat{<:Any,Arrays}}) where Arrays = blockhcatlayout(LazyArrays.tuple_type_memorylayouts(Arrays)...)

struct BlockBandedInterlaceLayout <: AbstractLazyBlockBandedLayout end

sublayout(::BlockBandedInterlaceLayout, ::Type{<:NTuple{2,BlockSlices}}) = BlockBandedInterlaceLayout()

arguments(::BlockBandedInterlaceLayout, A::BlockBroadcastMatrix{<:Any,typeof(vcat)}) = A.args

# avoid extra types, since we are using int indexing for now...
# TODO: rewrite when other block sizes are allowed
deblock(A::BlockedArray) = A.blocks
deblock(A::Zeros{T}) where T = Zeros{T}(size(A)...)
function arguments(::BlockBandedInterlaceLayout, A::SubArray)
    P = parent(A)
    args = arguments(BlockBandedInterlaceLayout(), P)
    KR,JR = parentindices(A)
    kr,jr = Int.(KR.block),Int.(JR.block)
    tuple(first(args), view.(deblock.(tail(args)), Ref(kr), Ref(jr))...)
end

# function _copyto!(::BlockBandedColumnMajor, ::BlockBandedInterlaceLayout, dest::AbstractMatrix, src::AbstractMatrix)
#     args = arguments(BlockBandedInterlaceLayout(), src)
#     N,M = blocksize(dest)

# end

blockinterlacelayout(_...) = LazyLayout()
blockinterlacelayout(::Union{ZerosLayout,AbstractPaddedLayout,AbstractBandedLayout}...) = BlockBandedInterlaceLayout()

MemoryLayout(::Type{<:BlockBroadcastMatrix{<:Any,typeof(hvcat),Arrays}}) where Arrays = blockinterlacelayout(Base.tail(LazyArrays.tuple_type_memorylayouts(Arrays))...)

# temporary hack, need to think of how to flag as lazy for infinite case.
MemoryLayout(::Type{<:BlockBroadcastMatrix{<:Any,typeof(hcat),Arrays}}) where Arrays = LazyLayout()

MemoryLayout(::Type{<:BlockBroadcastMatrix{<:Any,typeof(Diagonal),Arrays}}) where Arrays = LazyBandedBlockBandedLayout()




###
# block col/rowsupport
###
blockcolsupport(M::BlockVcat, j) = first(blockcolsupport(first(M.arrays),j)):(Block(blocksize(BlockVcat(Base.front(M.arrays)...),1))+last(blockcolsupport(last(M.arrays),j)))
blockrowsupport(M::BlockHcat, k) = first(blockrowsupport(first(M.arrays),k)):(Block(blocksize(BlockHcat(Base.front(M.arrays)...),1))+last(blockrowsupport(last(M.arrays),k)))
function blockcolsupport(H::BlockHcat, J::Block{1})
    j = Integer(J)
    for A in arguments(H)
        n = blocksize(A,2)
        j ≤ n && return blockcolsupport(A, Block(j))
        j -= n
    end
    return Block.(1:0)
end

function blockrowsupport(H::BlockVcat, K::Block{1})
    k = Integer(K)
    for A in arguments(H)
        n = blocksize(A,1)
        k ≤ n && return blockrowsupport(A, Block(k))
        k -= n
    end
    return Block.(1:0)
end

blockcolsupport(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}, j) = Block.(convexunion(colsupport.(tail(A.args), Ref(Int.(j)))...))
blockrowsupport(A::BlockBroadcastMatrix{<:Any,typeof(hvcat)}, k) = Block.(convexunion(rowsupport.(tail(A.args), Ref(Int.(k)))...))

blockcolsupport(A::BlockBroadcastVector{<:Any,typeof(vcat)}, j) = Block.(convexunion(colsupport.(tail(A.args), Ref(Int.(j)))...))

blockbroadcastlayout(FF, args...) = UnknownLayout()
blockbroadcastlayout(::Type{typeof(vcat)}, ::PaddedColumns...) = PaddedColumns{UnknownLayout}()

function paddeddata(B::BlockBroadcastVector{T,typeof(vcat)}) where T
    dats = map(paddeddata,B.args)
    N = max(map(length,dats)...)
    all(length.(dats) .== N) || error("differening padded lengths not supported")
    BlockBroadcastVector{T}(vcat, dats...)
end

MemoryLayout(::Type{BlockBroadcastArray{T,N,FF,Args}}) where {T,N,FF,Args} = blockbroadcastlayout(FF, tuple_type_memorylayouts(Args)...)

resize!(c::BlockBroadcastVector{T,typeof(vcat)}, N::Block{1}) where T = BlockBroadcastVector{T}(vcat, resize!.(c.args, N)...)

####
# BlockVec
####

# support LazyArrays v2.8 where BlockVec is moved
if isdefined(LazyBandedMatrices.LazyArraysBlockArraysExt, :BlockVec)
    const BlockVec = LazyBandedMatrices.LazyArraysBlockArraysExt.BlockVec
else
    const BlockVec{T, M<:AbstractMatrix{T}} = ApplyVector{T, typeof(blockvec), <:Tuple{M}}

    BlockVec{T}(M::AbstractMatrix{T}) where T = ApplyVector{T}(blockvec, M)
    BlockVec(M::AbstractMatrix{T}) where T = BlockVec{T}(M)
    axes(b::BlockVec) = (blockedrange(Fill(size(b.args[1])...)),)
    size(b::BlockVec) = (length(b.args[1]),)

    view(b::BlockVec, K::Block{1}) = view(b.args[1], :, Int(K))
    Base.@propagate_inbounds getindex(b::BlockVec, k::Int) = b.args[1][k]
    Base.@propagate_inbounds setindex!(b::BlockVec, v, k::Int) = setindex!(b.args[1], v, k)

    _resize!(A::AbstractMatrix, m, n) = A[1:m, 1:n]
    _resize!(At::Transpose, m, n) = transpose(transpose(At)[1:n, 1:m])
    _resize!(Ac::Adjoint, m, n) = (Ac')[1:n, 1:m]'
    resize!(b::BlockVec, K::Block{1}) = BlockVec(_resize!(b.args[1], size(b.args[1],1), Int(K)))

    applylayout(::Type{typeof(blockvec)}, ::AbstractPaddedLayout) = PaddedColumns{ApplyLayout{typeof(blockvec)}}()
    paddeddata(b::BlockVec) = BlockVec(paddeddata(b.args[1]))
end


####
# summary
####

function BlockArrays._show_typeof(io::IO, B::BlockHcat{T}) where T
    print(io, "BlockHcat{$T}")
end

function BlockArrays._show_typeof(io::IO, B::BlockVcat{T}) where T
    print(io, "BlockVcat{$T}")
end