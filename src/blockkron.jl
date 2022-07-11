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
end

function axes(A::DiagTrav{<:Any,2})
    m,n = size(A.array)
    mn = min(m,n)
    (blockedrange(Vcat(OneTo(mn), Fill(mn,max(m,n)-mn))),)
end

function axes(A::DiagTrav{<:Any,3})
    m,n,p = size(A.array)
    @assert m == n == p
    (blockedrange(cumsum(OneTo(m))),)
end


function getindex(A::DiagTrav{<:Any,2}, K::Block{1})
    k = Int(K)
    m,n = size(A.array)
    mn = min(m,n)
    st = stride(A.array,2)
    if k ≤ m
        A.array[range(k; step=st-1, length=min(k,mn))]
    else
        A.array[range(m+(k-m)*st; step=st-1, length=min(k,mn))]
    end
end

function getindex(A::DiagTrav{T,3}, K::Block{1}) where T
    k = Int(K)
    m,n,p = size(A.array)
    @assert m == n == p
    st = stride(A.array,2)
    st3 = stride(A.array,3)
    ret = A.array[range(k; step=st-1, length=k)]
    for j = 1:k-1
        append!(ret, view(A.array, range(j*st3 + (k-j); step=st-1, length=k-j)))
    end
    ret
end

getindex(A::DiagTrav, k::Int) = A[findblockindex(axes(A,1), k)]

function resize!(A::DiagTrav{<:Any,2}, K::Block{1})
    k = Int(K)
    DiagTrav(A.array[1:k, 1:k])
end

struct InvDiagTrav{T, AA<:AbstractVector{T}} <: AbstractMatrix{T}
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

KronTrav(A::AbstractArray{T,N}...) where {T,V,N} =
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
    for j̃ = 1:j, k̃ = 1:k
        AB[Block(k̃), Block(j̃)] .*= C̃[k̃, j̃]
    end
    AB
end

getindex(M::KronTrav{<:Any,2}, K::Block{2}) = _krontrav_getindex(K, M.args...)


getindex(A::KronTrav{<:Any,N}, kj::Vararg{Int,N}) where N =
    A[findblockindex.(axes(A), kj)...]

# A.A[1:k,1:j] has A_l,A_u
# A.B[k:-1:1,j:-1:1] has bandwidths (B_u + k-j, B_l + j-k)
subblockbandwidths(A::KronTrav) = bandwidths(first(A.args))
blockbandwidths(A::KronTrav) = broadcast(+, map(bandwidths,A.args)...)
isblockbanded(A::KronTrav) = all(isbanded, A.args)
isbandedblockbanded(A::KronTrav) = isblockbanded(A)

struct KronTravBandedBlockBandedLayout <: AbstractBandedBlockBandedLayout end

krontravlayout(_...) = UnknownLayout()
krontravlayout(::AbstractBandedLayout, ::AbstractBandedLayout) = KronTravBandedBlockBandedLayout()
MemoryLayout(::Type{KronTrav{T,N,AA,AXIS}}) where {T,N,AA,AXIS} = krontravlayout(tuple_type_memorylayouts(AA)...)


sublayout(::KronTravBandedBlockBandedLayout, ::Type{<:NTuple{2,BlockSlice1}}) = BroadcastBandedLayout{typeof(*)}()

call(b::BroadcastLayout{typeof(*)}, a::KronTrav) = *
call(b::BroadcastBandedLayout{typeof(*)}, a::SubArray) = *

function _broadcast_sub_arguments(::KronTravBandedBlockBandedLayout, M, V)
    K,J = parentindices(V)
    k,j = Int(K.block),Int(J.block)
    @assert length(M.args) == 2
    A,B = M.args
    view(A,1:k,1:j), ApplyMatrix(rot180,view(B,1:k,1:j))
end


krontavbroadcaststyle(::BandedStyle, ::BandedStyle) = BandedBlockBandedStyle()
krontavbroadcaststyle(::BandedStyle, ::StructuredMatrixStyle{<:Diagonal}) = BandedBlockBandedStyle()
krontavbroadcaststyle(::StructuredMatrixStyle{<:Diagonal}, ::BandedStyle) = BandedBlockBandedStyle()
krontavbroadcaststyle(::LazyArrayStyle{2}, ::BandedStyle) = LazyArrayStyle{2}()
krontavbroadcaststyle(::BandedStyle, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()
krontavbroadcaststyle(::LazyArrayStyle{2}, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()

tuple_type_broadcaststyle(::Type{Tuple{}}) = ()
tuple_type_broadcaststyle(T::Type{<:Tuple{A,Vararg{Any}}}) where A = 
    tuple(BroadcastStyle(A), tuple_type_broadcaststyle(tuple_type_tail(T))...)
BroadcastStyle(::Type{KronTrav{T,N,AA,AXIS}}) where {T,N,AA,AXIS} =
    krontavbroadcaststyle(tuple_type_broadcaststyle(AA)...)

mul(L::KronTrav, M::KronTrav) = KronTrav((L.args .* M.args)...)