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


isblockbanded(K::BlockKron) = isbanded(first(K.args))
isbandedblockbanded(K::BlockKron) = all(isbanded, K.args)
blockbandwidths(K::BlockKron) = bandwidths(first(K.args))
subblockbandwidths(K::BlockKron) = bandwidths(last(K.args))


"""
    diagtrav(A::AbstractMatrix)
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

struct KronTrav{T, N, AA<:AbstractArray{T,N}, BB<:AbstractArray{T,N}, AXES} <: AbstractBlockArray{T, N}
    A::AA
    B::BB
    axes::AXES
end

KronTrav(A::AbstractArray{T,N}, B::AbstractArray{V,N}, axes) where {T,V,N} =
    KronTrav{promote_type(T,V),N,typeof(A),typeof(B),typeof(axes)}(A, B, axes)

KronTrav(A::AbstractArray{T,N}, B::AbstractArray{V,N}) where {T,V,N} =
    KronTrav(A, B, _krontrav_axes(axes(A), axes(B)))

function _krontrav_axes(A::NTuple{N,OneTo{Int}}, B::NTuple{N,OneTo{Int}}) where N
    m,n = length.(A), length.(B)
    mn = min.(m,n)
    @. blockedrange(Vcat(OneTo(mn), Fill(mn,max(m,n)-mn)))
end

axes(A::KronTrav) = A.axes

function getindex(A::KronTrav{<:Any,1}, K::Block{1})
    m,n = length(A.A), length(A.B)
    mn = min(m,n)
    k = Int(K)
    if k ≤ mn
        A.A[1:k] .* A.B[k:-1:1]
    elseif m < n
        A.A .* A.B[k:-1:(k-m+1)]
    else # n < m
        A.A[(k-n+1):k] .* A.B[end:-1:1]
    end
end



function getindex(A::KronTrav{<:Any,2}, K::Block{2})
    m,n = size(A.A), size(A.B)
    @assert m == n
    k,j = K.n
    # Following is equivalent to A.A[1:k,1:j] .* A.B[k:-1:1,j:-1:1]
    # layout_getindex to avoid SparseArrays from Diagonal
    # rot180 to preserve bandedness
    layout_getindex(A.A,1:k,1:j) .* rot180(layout_getindex(A.B,1:k,1:j))
end
getindex(A::KronTrav{<:Any,N}, kj::Vararg{Int,N}) where N =
    A[findblockindex.(axes(A), kj)...]

# A.A[1:k,1:j] has A_l,A_u
# A.B[k:-1:1,j:-1:1] has bandwidths (B_u + k-j, B_l + j-k)
subblockbandwidths(A::KronTrav) = bandwidths(A.A)
blockbandwidths(A::KronTrav) = bandwidths(A.A) .+ bandwidths(A.B)
isblockbanded(A::KronTrav) = isbanded(A.A) && isbanded(A.B)
isbandedblockbanded(A::KronTrav) = isbanded(A.A) && isbanded(A.B)

struct KronTravBandedBlockBandedLayout <: AbstractBandedBlockBandedLayout end

krontravlayout(_, _) = UnknownLayout()
krontravlayout(::AbstractBandedLayout, ::AbstractBandedLayout) = KronTravBandedBlockBandedLayout()
MemoryLayout(::Type{KronTrav{T,N,AA,BB,AXIS}}) where {T,N,AA,BB,AXIS} = krontravlayout(MemoryLayout(AA), MemoryLayout(BB))


sublayout(::KronTravBandedBlockBandedLayout, ::Type{NTuple{2,BlockSlice1}}) = BroadcastBandedLayout{typeof(*)}()

call(b::BroadcastLayout{typeof(*)}, a::KronTrav) = *
call(b::BroadcastBandedLayout{typeof(*)}, a::SubArray) = *

function _broadcast_sub_arguments(::KronTravBandedBlockBandedLayout, A, V)
    K,J = parentindices(V)
    k,j = Int(K.block),Int(J.block)
    view(A.A,1:k,1:j), ApplyMatrix(rot180,view(A.B,1:k,1:j))
end


krontavbroadcaststyle(::BandedStyle, ::BandedStyle) = BandedBlockBandedStyle()
krontavbroadcaststyle(::BandedStyle, ::StructuredMatrixStyle{<:Diagonal}) = BandedBlockBandedStyle()
krontavbroadcaststyle(::StructuredMatrixStyle{<:Diagonal}, ::BandedStyle) = BandedBlockBandedStyle()
krontavbroadcaststyle(::LazyArrayStyle{2}, ::BandedStyle) = LazyArrayStyle{2}()
krontavbroadcaststyle(::BandedStyle, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()
krontavbroadcaststyle(::LazyArrayStyle{2}, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()
BroadcastStyle(::Type{KronTrav{T,N,AA,BB,AXIS}}) where {T,N,AA,BB,AXIS} =
    krontavbroadcaststyle(BroadcastStyle(AA), BroadcastStyle(BB))
