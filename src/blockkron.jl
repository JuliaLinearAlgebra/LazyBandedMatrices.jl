const OneToCumsum = RangeCumsum{Int,OneTo{Int}}
BlockArrays.sortedunion(a::OneToCumsum, ::OneToCumsum) = a
function BlockArrays.sortedunion(a::RangeCumsum{<:Any,<:AbstractRange}, b::RangeCumsum{<:Any,<:AbstractRange})
    @assert a == b
    a
end

###
# Block
###

Base.in(K::Block, B::BroadcastVector{<:Block,Type{Block}}) = Int(K) in B.args[1]


###
# BlockBanded
###

bandedblockbandedbroadcaststyle(::LazyArrayStyle{2}) = LazyArrayStyle{2}()
bandedblockbandedcolumns(::LazyLayout) = BandedBlockBandedColumns{LazyLayout}()
bandedblockbandedcolumns(::ApplyLayout) = BandedBlockBandedColumns{LazyLayout}()
bandedblockbandedcolumns(::BroadcastLayout) = BandedBlockBandedColumns{LazyLayout}()



"""
    DiagTrav(A::AbstractMatrix)

converts a matrix to a block vector by traversing the anti-diagonals.
"""
struct DiagTrav{T, N, AA<:AbstractArray{T,N}} <: AbstractBlockVector{T}
    array::AA
    function DiagTrav{T, N, AA}(array::AA) where {T, N, AA<:AbstractArray{T,N}}
        ==(size(array)...) || error("array must be square")
        new{T,N,AA}(array)
    end
end
DiagTrav{T,N}(A::AbstractArray) where {T,N} = DiagTrav{T,N,typeof(A)}(A)
DiagTrav{T}(A::AbstractArray{<:Any,N}) where {T,N} = DiagTrav{T,N}(A)
DiagTrav(A::AbstractArray{T}) where T = DiagTrav{T}(A)

axes(A::DiagTrav{<:Any,2}) = (blockedrange(oneto(size(A.array,1))),)
axes(A::DiagTrav{<:Any,3}) = (blockedrange(cumsum(oneto(size(A.array,1)))),)

struct DiagTravLayout{Lay} <: AbstractBlockLayout end
MemoryLayout(::Type{<:DiagTrav{T, N, AA}}) where {T,N,AA} = DiagTravLayout{typeof(MemoryLayout(AA))}()

function blockcolsupport(A::DiagTrav{<:Any,2}, _)
    cs = colsupport(A.array)
    rs = rowsupport(A.array)
    Block.(max(first(cs),first(rs)):max(last(cs),last(rs)))
end

function colsupport(A::DiagTrav{<:Any,2}, _)
    bs = blockcolsupport(A)
    sum(1:Int(first(bs)-1))+1:sum(1:Int(last(bs)))
end


function getindex(A::DiagTrav{<:Any,2}, K::Block{1})
    @boundscheck checkbounds(A, K)
    _diagtravgetindex(MemoryLayout(A.array), A.array, K)
end

function _diagtravgetindex(_, A::AbstractMatrix, K::Block{1})
    k = Int(K)
    [A[k-j+1,j] for j = 1:k]
end


function _diagtravgetindex(::AbstractStridedLayout, A::AbstractMatrix, K::Block{1})
    k = Int(K)
    st = stride(A,2)
    A[range(k; step=st-1, length=k)]
end

function Base.view(A::DiagTrav{<:Any,2,<:Matrix}, K::Block{1})
    k = Int(K)
    st = stride(A.array,2)
    view(A.array,range(k; step=st-1, length=k))
end

function _diagtravgetindex(::PaddedLayout{<:AbstractStridedLayout}, A::AbstractMatrix{T}, K::Block{1}) where T
    k = Int(K)
    P = paddeddata(A)
    m,_ = size(P)
    st = stride(P,2)
    if k ≤ m
        P[range(k; step=st-1, length=k)]
    else
        [Zeros{T}(k-m); P[range(m+(k-m)*st; step=st-1, length=2m-k)]; Zeros{T}(k-m)]
    end
end

function getindex(A::DiagTrav{T,3}, K::Block{1}) where T
    k = Int(K)
    m,n,p = size(A.array)
    @assert m == n == p
    st = stride(A.array,2)
    st3 = stride(A.array,3)
    ret = T[]
    for j = 0:k-1
        append!(ret, view(A.array, range(j*st + k-j; step=st3-st, length=j+1)))
    end
    ret
end

getindex(A::DiagTrav, k::Int) = A[findblockindex(axes(A,1), k)]

function resize!(A::DiagTrav{<:Any,2}, K::Block{1})
    k = Int(K)
    DiagTrav(A.array[1:k, 1:k])
end

function Base._maximum(f, a::DiagTrav, ::Colon; kws...)
    # avoid zeros
    KR = blockaxes(a,1)
    ret = maximum(f, view(a,KR[1]))
    for K = KR[2]:KR[end]
        ret = max(ret, maximum(f, view(a,K)))
    end
    ret
end

struct InvDiagTrav{T, AA<:AbstractVector{T}} <: LayoutMatrix{T}
    vector::AA
end

size(A::InvDiagTrav) = (blocksize(A.vector,1),blocksize(A.vector,1))

function getindex(A::InvDiagTrav{T}, k::Int, j::Int)  where T
    if k+j-1 ≤ blocksize(A.vector,1)
        A.vector[Block(k+j-1)][j]
    else
        zero(T)
    end
end

struct KronTrav{T, N, AA<:Tuple{Vararg{AbstractArray{T,N}}}, AXES} <: AbstractBlockArray{T, N}
    args::AA
    axes::AXES
end

KronTrav(A::AbstractArray{T,N}...) where {T,N} =
    KronTrav(A, map(_krontrav_axes, map(axes,A)...))

function _krontrav_axes(A::OneTo{Int}, B::OneTo{Int})
    m,n = length(A), length(B)
    mn = min(m,n)
    blockedrange(Vcat(OneTo(mn), Fill(mn,max(m,n)-mn)))
end

function _krontrav_axes(A::OneTo{Int}, B::OneTo{Int}, C::OneTo{Int})
    m,n,ν = length(A), length(B), length(C)
    @assert m == n == ν
    blockedrange(RangeCumsum(OneTo(m)))
end
copy(K::KronTrav) = KronTrav(map(copy,K.args), K.axes)
axes(A::KronTrav) = A.axes

function getindex(M::KronTrav{<:Any,1}, K::Block{1})
    A,B = M.args
    m,n = length(A), length(B)
    mn = min(m,n)
    k = Int(K)
    if k ≤ mn
        A[1:k] .* B[k:-1:1]
    elseif m < n
        A .* B[k:-1:(k-m+1)]
    else # n < m
        A[(k-n+1):k] .* B[end:-1:1]
    end
end

function _krontrav_getindex(K::Block{2}, A, B)
    m,n = size(A), size(B)
    @assert m == n
    k,j = K.n
    # Following is equivalent to A[1:k,1:j] .* B[k:-1:1,j:-1:1]
    # layout_getindex to avoid SparseArrays from Diagonal
    # rot180 to preserve bandedness
    layout_getindex(A,1:k,1:j) .* rot180(layout_getindex(B,1:k,1:j))
end

function _krontrav_getindex(Kin::Block{2}, A, B, C)
    k,j = Kin.n
    AB = KronTrav(A, B)[Block.(1:k), Block.(1:j)]
    C̃ = rot180(layout_getindex(C,1:k,1:j))
    for J = Block.(1:j), K = blockcolsupport(AB, J)
        lmul!(C̃[Int(K), Int(J)], view(AB, K, J))
    end
    AB
end

getindex(M::KronTrav{<:Any,2}, K::Block{2}) = _krontrav_getindex(K, M.args...)


getindex(A::KronTrav{<:Any,N}, kj::Vararg{Int,N}) where N =
    A[findblockindex.(axes(A), kj)...]

# A.A[1:k,1:j] has A_l,A_u
# A.B[k:-1:1,j:-1:1] has bandwidths (B_u + k-j, B_l + j-k)

_krontrav_subblockbandwidths(A, B) = bandwidths(A)
_krontrav_subblockbandwidths(A, B, C) = bandwidths(KronTrav(A, B))
_krontrav_blockbandwidths(A...) = broadcast(+, map(bandwidths,A)...)

subblockbandwidths(A::KronTrav) = _krontrav_subblockbandwidths(A.args...)
blockbandwidths(A::KronTrav) = _krontrav_blockbandwidths(A.args...)


isblockbanded(A::KronTrav) = all(isbanded, A.args)
isbandedblockbanded(A::KronTrav) = isblockbanded(A) && length(A.args) == 2

convert(::Type{B}, A::KronTrav{<:Any,2}) where B<:BandedBlockBandedMatrix = convert(B, BandedBlockBandedMatrix(A))

struct KronTravBandedBlockBandedLayout <: AbstractBandedBlockBandedLayout end
struct KronTravLayout{M} <: AbstractBlockLayout end



krontravlayout(::Vararg{Any,M}) where M = KronTravLayout{M}()
krontravlayout(::AbstractBandedLayout, ::AbstractBandedLayout) = KronTravBandedBlockBandedLayout()
MemoryLayout(::Type{KronTrav{T,N,AA,AXIS}}) where {T,N,AA,AXIS} = krontravlayout(tuple_type_memorylayouts(AA)...)


sublayout(::KronTravBandedBlockBandedLayout, ::Type{<:NTuple{2,BlockSlice1}}) = BroadcastBandedLayout{typeof(*)}()
sublayout(::KronTravLayout{2}, ::Type{<:NTuple{2,BlockSlice1}}) = BroadcastLayout{typeof(*)}()

sublayout(::KronTravLayout{M}, ::Type{<:NTuple{2,BlockSlice{BlockRange{1,Tuple{OneTo{Int}}}}}}) where M = KronTravLayout{M}()
sublayout(::KronTravLayout{2}, ::Type{<:NTuple{2,BlockSlice{BlockRange{1,Tuple{OneTo{Int}}}}}}) = KronTravLayout{2}()
sublayout(::KronTravLayout{2}, ::Type{<:NTuple{2,BlockSlice{<:BlockRange1}}}) = BlockLayout{UnknownLayout,BroadcastLayout{typeof(*)}}()
sublayout(::KronTravBandedBlockBandedLayout, ::Type{<:NTuple{2,BlockSlice{BlockRange{1,Tuple{OneTo{Int}}}}}}) = KronTravBandedBlockBandedLayout()

sub_materialize(::Union{KronTravLayout,KronTravBandedBlockBandedLayout}, V) = KronTrav(map(sub_materialize, krontravargs(V))...)


krontravargs(K::KronTrav) = K.args
function krontravargs(V::SubArray)
    KR,JR = parentindices(V)
    m,n = Int(KR.block[end]), Int(JR.block[end])
    view.(krontravargs(parent(V)), Ref(OneTo(m)), Ref(OneTo(n)))
end


call(b::BroadcastLayout{typeof(*)}, a::KronTrav) = *
call(b::BroadcastBandedLayout{typeof(*)}, a::SubArray) = *

function _broadcast_sub_arguments(::Union{KronTravLayout{2},KronTravBandedBlockBandedLayout}, M, V)
    K,J = parentindices(V)
    k,j = Int(K.block),Int(J.block)
    @assert length(krontravargs(M)) == 2
    A,B = krontravargs(M)
    view(A,1:k,1:j), ApplyMatrix(rot180,view(B,1:k,1:j))
end


krontavbroadcaststyle(::BandedStyle, ::BandedStyle) = BandedBlockBandedStyle()
krontavbroadcaststyle(::BandedStyle, ::StructuredMatrixStyle{<:Diagonal}) = BandedBlockBandedStyle()
krontavbroadcaststyle(::StructuredMatrixStyle{<:Diagonal}, ::BandedStyle) = BandedBlockBandedStyle()
krontavbroadcaststyle(::Union{BandedStyle,StructuredMatrixStyle{<:Diagonal}}...) = BlockBandedStyle()
krontavbroadcaststyle(::LazyArrayStyle{2}, ::BandedStyle, ::LazyArrayStyle{2}...) = LazyArrayStyle{2}()
krontavbroadcaststyle(::BandedStyle, ::LazyArrayStyle{2}, ::LazyArrayStyle{2}...) = LazyArrayStyle{2}()
krontavbroadcaststyle(::LazyArrayStyle{2}, ::LazyArrayStyle{2}, ::LazyArrayStyle{2}...) = LazyArrayStyle{2}()

tuple_type_broadcaststyle(::Type{Tuple{}}) = ()
tuple_type_broadcaststyle(T::Type{<:Tuple{A,Vararg{Any}}}) where A = 
    tuple(BroadcastStyle(A), tuple_type_broadcaststyle(tuple_type_tail(T))...)
BroadcastStyle(::Type{KronTrav{T,N,AA,AXIS}}) where {T,N,AA,AXIS} =
    krontavbroadcaststyle(tuple_type_broadcaststyle(AA)...)

# mul(L::KronTrav, M::KronTrav) = KronTrav((L.args .* M.args)...)
