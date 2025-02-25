

###
# Block
###

function _DiagTrav end

"""
    DiagTrav(A::AbstractMatrix)

converts a matrix to a block vector by traversing the anti-diagonals.
"""
struct DiagTrav{T, N, AA<:AbstractArray{T,N}} <: AbstractBlockVector{T}
    array::AA
    global function _DiagTrav(array::AA) where {T, N, AA<:AbstractArray{T,N}}
        new{T,N,AA}(array)
    end
end
DiagTrav{T,N,AA}(A::AA) where {T, N, AA<:AbstractArray{T,N}} = _DiagTrav(zero_bottomright(A))

DiagTrav{T,N}(A::AbstractArray) where {T,N} = DiagTrav{T,N,typeof(A)}(A)
DiagTrav{T}(A::AbstractArray{<:Any,N}) where {T,N} = DiagTrav{T,N}(A)
DiagTrav(A::AbstractArray{T}) where T = DiagTrav{T}(A)


zero_bottomright(array) = zero_bottomright(array, axes(array))
zero_bottomright!(array) = zero_bottomright!(array, axes(array))

function zero_bottomright(X::AbstractMatrix, _)
    m,n = size(X)
    μ = max(m,n)
    for j in rowsupport(X), k = (μ-j+2:m) ∩ colsupport(X,j)
        iszero(X[k,j]) || return zero_bottomright!(copy(X))
    end
    X
end

function zero_bottomright!(X::AbstractMatrix{T}, _) where T
    m,n = size(X)
    μ = max(m,n)
    for j in rowsupport(X), k = (μ-j+2:m) ∩ colsupport(X,j)
        X[k,j] = zero(T)
    end
    X
end

function zero_bottomright(X::AbstractArray{<:Any,3}, _)
    m,n,p = size(X)
    @assert m == n == p
    for ℓ = 0:n-1, j=0:n-1, k=max(0,n-(ℓ+j)):n-1
        iszero(X[k+1,j+1,ℓ+1]) || return zero_bottomright!(copy(X))
    end
    X
end

function zero_bottomright!(X::AbstractArray{T,3}, _) where T
    m,n,p = size(X)
    @assert m == n == p
    for ℓ = 0:n-1, j=0:n-1, k=max(0,n-(ℓ+j)):n-1
        X[k+1,j+1,ℓ+1] = zero(T)
    end
    X
end



function _krontrav_axes(A, B)
    m,n = length(A), length(B)
    mn = min(m,n)
    blockedrange(Vcat(oneto(mn), Fill(mn,max(m,n)-mn)))
end

function _krontrav_axes(A, B, C)
    m,n,ν = length(A), length(B), length(C)
    @assert m == n == ν
    blockedrange(RangeCumsum(oneto(m)))
end

axes(A::DiagTrav) = (_krontrav_axes(axes(A.array)...),)

copy(A::DiagTrav) = DiagTrav(copy(A.array))

similar(A::DiagTrav, ::Type{T}) where T = DiagTrav(similar(A.array, T))

struct DiagTravLayout{Lay} <: AbstractBlockLayout end
MemoryLayout(::Type{<:DiagTrav{T, N, AA}}) where {T,N,AA} = DiagTravLayout{typeof(MemoryLayout(AA))}()

function blockcolsupport(A::DiagTrav{<:Any,2}, _)
    cs = colsupport(A.array)
    rs = rowsupport(A.array)
    Block.(max(first(cs),first(rs)):min(max(size(A.array)...), last(cs)+last(rs)-1))
end

function colsupport(A::DiagTrav{<:Any,2}, _)
    bs = blockcolsupport(A)
    axes(A,1)[bs]
end


Base.@propagate_inbounds function getindex(A::DiagTrav, Kk::BlockIndex{1})
    @boundscheck checkbounds(A, Kk)
    _diagtravgetindex(MemoryLayout(A.array), A.array, Kk)
end

function _diagtravgetindex(_, A::AbstractMatrix, Kk::BlockIndex{1})
    K,j = Int(block(Kk)), blockindex(Kk)
    m,n = size(A)
    j = max(0,K-m)+j
    A[K-j+1,j]
end

Base.@propagate_inbounds function setindex!(A::DiagTrav, v, Kk::BlockIndex{1})
    @boundscheck checkbounds(A, Kk)
    _diagtravsetindex!(MemoryLayout(A.array), A.array, v, Kk)
    A
end

function _diagtravsetindex!(_, A, v, Kk::BlockIndex{1})
    K,j = Int(block(Kk)), blockindex(Kk)
    m,n = size(A)
    j = max(0,K-m)+j
    A[K-j+1,j] = v
end

Base.@propagate_inbounds function getindex(A::DiagTrav, K::Block{1})
    @boundscheck checkbounds(A, K)
    _diagtravgetindex(MemoryLayout(A.array), A.array, K)
end

function _diagtravgetindex(_, A::AbstractMatrix, K::Block{1})
    k = Int(K)
    m,n = size(A)
    [A[k-j+1,j] for j = max(1,k-m+1):min(k,n)]
end


_diagtravgetindex(::AbstractStridedLayout, A::AbstractMatrix, K::Block{1}) = layout_getindex(_DiagTrav(A), K)

function _diagtravview(::AbstractStridedLayout, A::AbstractMatrix, K::Block{1})
    k = Int(K)
    st = stride(A,2)
    m,n = size(A)
    mn = min(m,n)
    if k ≤ m
        view(A,range(k; step=max(1,st-1), length=min(k,mn)))
    else
        view(A,range(m+(k-m)*st; step=max(1,st-1), length=min(k,mn)))
    end
end

Base.view(A::DiagTrav, K::Block{1}) = _diagtravview(MemoryLayout(A.array), A.array, K)

_diagtravview(_, A::AbstractArray, K::Block{1}) = Base.invoke(view, Tuple{AbstractArray, Any}, DiagTrav(A), K)

function _diagtravgetindex(::AbstractPaddedLayout{<:AbstractStridedLayout}, A::AbstractMatrix{T}, K::Block{1}) where T
    k = Int(K)
    P = paddeddata(A)
    m,n = size(P)
    M,N = size(A)
    mn = min(m,n)
    st = stride(P,2)
    # TODO: not really a view...
    if k ≤ m
        [Zeros{T}(k-m); view(P,StepRangeLen(k, st-1, max(0, min(k,n)))); Zeros{T}(max(0,k-n))]
    else
        [Zeros{T}(min(k,M)-m); view(P,StepRangeLen(m+(k-m)*st, st-1, max(0,m+n-k))); Zeros{T}(max(0,k-n))]
    end
end


function _diagtravgetindex(::AbstractStridedLayout, A::AbstractArray{T,3}, K::Block{1}) where T
    k = Int(K)
    m,n,p = size(A)
    @assert m == n == p
    st = stride(A,2)
    st3 = stride(A,3)
    ret = T[]
    for j = 0:k-1
        append!(ret, view(A, range(j*st + k-j; step=st3-st, length=j+1))) # this matches lexigraphical order
    end
    ret
end

# TODO: don't build the block
_diagtravgetindex(lay, A::AbstractArray{T,3}, Kk::BlockIndex{1}) where T = _diagtravgetindex(lay, A, block(Kk))[blockindex(Kk)]

getindex(A::DiagTrav, k::Int) = A[findblockindex(axes(A,1), k)]
setindex!(A::DiagTrav, v, k::Int) = A[findblockindex(axes(A,1), k)] = v

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

pad(c, ax...) = PaddedArray(c, ax)

copy(M::Mul{<:ApplyBandedBlockBandedLayout{typeof(*)},<:DiagTravLayout{<:AbstractPaddedLayout}}) =
    copy(Mul{ApplyLayout{typeof(*)},UnknownLayout}(M.A, M.B))

function copy(M::Mul{<:LazyBlockBandedLayouts,<:DiagTravLayout})
    M.A * Vector(M.B)
end

LazyArrays.simplifiable(::Mul{<:LazyBlockBandedLayouts,<:DiagTravLayout{<:AbstractPaddedLayout}}) = Val(true)
function copy(M::Mul{<:LazyBlockBandedLayouts,<:DiagTravLayout{<:AbstractPaddedLayout}})
    A,B = M.A, M.B
    dat = paddeddata(B.array)
    N = 2max(size(dat)...)-1
    P = DiagTrav(convert(Matrix, B.array[1:N,1:N]))
    JR = blockaxes(P,1)
    KR = Block.(OneTo(Int(blockcolsupport(A,JR)[end])))
    pad(A[KR,JR] * P, axes(A,1))
end

struct InvDiagTrav{T, AA<:AbstractVector{T}} <: LayoutMatrix{T}
    vector::AA
end

size(A::InvDiagTrav) = (blocksize(A.vector,1),blocksize(A.vector,1))

function getindex(A::InvDiagTrav{T}, k::Int, j::Int)  where T
    if k+j-1 ≤ blocksize(A.vector,1)
        A.vector[Block(k+j-1)][j]
    else
        zero(T)
    end
end

invdiagtrav(a) = InvDiagTrav(a)
invdiagtrav(a::DiagTrav) = a.array

-(A::DiagTrav) = DiagTrav(-A.array)

for op in (:+, :-)
    @eval $op(A::DiagTrav, B::DiagTrav) = DiagTrav($op(A.array, B.array))
end

for op in (:*, :/)
    @eval $op(A::DiagTrav, k::Number) = DiagTrav($op(A.array, k))
end

for op in (:*, :\)
    @eval $op(k::Number, A::DiagTrav) = DiagTrav($op(k, A.array))
end

"""
    KronTrav(A,B,C...)

represents a kronecker product but where the coefficients are ordered as in `DiagTrav`.
For example, `KronTrav(A,B) * DiagTrav(X)` represents the same operator as `B*X*A'`, though
the bottom of the right of `X` is ignored.
"""
struct KronTrav{T, N, AA<:Tuple{Vararg{AbstractArray{T,N}}}, AXES} <: AbstractBlockArray{T, N}
    args::AA
    axes::AXES
end

KronTrav{T}(A::AbstractArray{T,N}...) where {T,N} = KronTrav(A, map(_krontrav_axes, map(axes,A)...))
KronTrav{T}(A::AbstractArray{T}...) where T = error("All arrays must have same dimensions")
KronTrav(A::AbstractArray{T}...) where T = KronTrav{T}(A...)    
KronTrav{T}(A::AbstractArray...) where T = KronTrav{T}(convert.(AbstractArray{T}, A)...)
KronTrav(A::AbstractArray...) = KronTrav{mapreduce(eltype, promote_type, A)}(A...)


copy(K::KronTrav) = KronTrav(map(copy,K.args), K.axes)
axes(A::KronTrav) = A.axes



function _krontrav_getindex(K::Block{1}, A, B)
    m,n = length(A), length(B)
    mn = min(m,n)
    k = Int(K)
    if k ≤ mn
        A[1:k] .* B[k:-1:1]
    elseif m < n
        A .* B[k:-1:(k-m+1)]
    else # n < m
        A[(k-n+1):k] .* B[end:-1:1]
    end
end



function _krontrav_getindex(K::Block{1}, A, B, C)
    @assert length(A) == length(B) == length(C) # TODO: generalise

    # make a tuple corresponding to lexigraphical order
    ret = Vector{promote_type(eltype(A),eltype(B),eltype(C))}()
    n = Int(K)
    for k = 1:n, j=1:k
        push!(ret, C[n-k+1]B[k-j+1]A[j])
    end
    ret
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

getindex(M::KronTrav{<:Any,1}, K::Block{1}) = _krontrav_getindex(K, M.args...)
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

const KronTravLayouts = Union{KronTravBandedBlockBandedLayout, KronTravLayout}


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

const BandedStyles = Union{BandedStyle, StructuredMatrixStyle{<:Diagonal}, StructuredMatrixStyle{<:Tridiagonal}, StructuredMatrixStyle{<:LinearAlgebra.Tridiagonal}}

krontravbroadcaststyle(::BandedStyles, ::BandedStyles) = BandedBlockBandedStyle()
krontravbroadcaststyle(::BandedStyles...) = BlockBandedStyle()
krontravbroadcaststyle(::LazyArrayStyle{2}, ::BandedStyles, ::LazyArrayStyle{2}...) = LazyArrayStyle{2}()
krontravbroadcaststyle(::BandedStyles, ::LazyArrayStyle{2}, ::LazyArrayStyle{2}...) = LazyArrayStyle{2}()
krontravbroadcaststyle(::LazyArrayStyle{2}, ::LazyArrayStyle{2}, ::LazyArrayStyle{2}...) = LazyArrayStyle{2}()

tuple_type_broadcaststyle(::Type{Tuple{}}) = ()
tuple_type_broadcaststyle(T::Type{<:Tuple{A,Vararg{Any}}}) where A = 
    tuple(BroadcastStyle(A), tuple_type_broadcaststyle(tuple_type_tail(T))...)
BroadcastStyle(::Type{KronTrav{T,N,AA,AXIS}}) where {T,N,AA,AXIS} =
    krontravbroadcaststyle(tuple_type_broadcaststyle(AA)...)

# mul(L::KronTrav, M::KronTrav) = KronTrav((L.args .* M.args)...)


###
# algebra
###

*(a::Number, b::KronTrav) = KronTrav(a*first(b.args), tail(b.args)...)
*(a::KronTrav, b::Number) = KronTrav(first(a.args)*b, tail(a.args)...)


function copy(M::Mul{<:KronTravLayouts, <:DiagTravLayout})
    K,x = M.A,M.B
    A,B = K.args
    _krontrav_mul_diagtrav(K.args, invdiagtrav(x), eltype(M))
end

_krontrav_mul_diagtrav((A,B), X::AbstractMatrix, ::Type{T}) where T = DiagTrav(convert(AbstractMatrix{T}, B*X*A'))
function _krontrav_mul_diagtrav((A,B,C), X::AbstractArray{<:Any,3}, ::Type{T}) where T
    m,n,p = size(X)
    @assert m == n == p
    Y = similar(X, T)
    Z = similar(X, T)
    for k = 1:n, j=1:n mul!(view(Y,k,j,:),A,view(X,k,j,:)) end
    for k = 1:n, j=1:n mul!(view(Z,k,:,j),B,view(Y,k,:,j)) end
    for k = 1:n, j=1:n mul!(view(Y,:,k,j),C,view(Z,:,k,j)) end
    DiagTrav(Y)
end

krontrav_materialize_layout(_, K) = BlockedArray(K)
krontrav_materialize_layout(::AbstractBandedBlockBandedLayout, K) = BandedBlockBandedMatrix(K)
krontrav_materialize(K) = krontrav_materialize_layout(MemoryLayout(K), K)
krontrav(A...) = krontrav_materialize(KronTrav(A...))


function krontrav(a::SquareEye{T}, b::SquareEye{V}) where {T,V}
    size(a) == size(b) || throw(ArgumentError("size must match"))
    SquareEye{promote_type(T,V)}((blockedrange(oneto(size(a,1))),))
end
# C = α*B*X*A' + β*C