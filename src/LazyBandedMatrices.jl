module LazyBandedMatrices
using BandedMatrices, BlockBandedMatrices, BlockArrays, LazyArrays,
        ArrayLayouts, MatrixFactorizations, LinearAlgebra, Base

import MatrixFactorizations: ql, ql!, QLPackedQ, QRPackedQ, reflector!, reflectorApply!,
            QLPackedQLayout, QRPackedQLayout, AdjQLPackedQLayout, AdjQRPackedQLayout

import Base: BroadcastStyle, similar, OneTo, copy, *, axes, size, getindex
import Base.Broadcast: Broadcasted, broadcasted
import LinearAlgebra: kron, hcat, vcat, AdjOrTrans, AbstractTriangular, BlasFloat, BlasComplex, BlasReal,
                        lmul!, rmul!, checksquare, StructuredMatrixStyle

import ArrayLayouts: materialize!, colsupport, rowsupport, MatMulVecAdd, require_one_based_indexing,
                    sublayout, transposelayout, _copyto!, MemoryLayout, AbstractQLayout, 
                    OnesLayout, DualLayout, mulreduce
import LazyArrays: LazyArrayStyle, combine_mul_styles, PaddedLayout,
                        broadcastlayout, applylayout, arguments, _mul_arguments, call,
                        LazyArrayApplyStyle, ApplyArrayBroadcastStyle, ApplyStyle,
                        LazyLayout, AbstractLazyLayout, ApplyLayout, BroadcastLayout, CachedVector, AbstractInvLayout,
                        _mat_mul_arguments, paddeddata, sub_paddeddata, sub_materialize, lazymaterialize,
                        MulMatrix, Mul, CachedMatrix, CachedArray, cachedlayout, _cache,
                        resizedata!, applybroadcaststyle, _broadcastarray2broadcasted,
                        LazyMatrix, LazyVector, LazyArray, MulAddStyle, _broadcast_sub_arguments,
                        _mul_args_colsupport, _mul_args_rowsupport, _islazy
import BandedMatrices: bandedcolumns, bandwidths, isbanded, AbstractBandedLayout,
                        prodbandwidths, BandedStyle, BandedColumns, BandedRows, BandedLayout,
                        AbstractBandedMatrix, BandedSubBandedMatrix, BandedStyle, _bnds,
                        banded_rowsupport, banded_colsupport, _BandedMatrix, bandeddata,
                        banded_qr_lmul!, banded_qr_rmul!, _banded_broadcast!
import BlockBandedMatrices: BlockSlice, Block1, AbstractBlockBandedLayout,
                        isblockbanded, isbandedblockbanded, blockbandwidths,
                        bandedblockbandedbroadcaststyle, bandedblockbandedcolumns,
                        BandedBlockBandedColumns, BlockBandedColumns,
                        subblockbandwidths, BandedBlockBandedMatrix, BlockBandedMatrix,
                        AbstractBandedBlockBandedLayout, BandedBlockBandedStyle,
                        blockcolsupport, BlockRange1, blockrowsupport, BlockIndexRange1
import BlockArrays: blockbroadcaststyle, BlockSlice1, BlockLayout

export DiagTrav, KronTrav, blockkron

BroadcastStyle(::LazyArrayStyle{1}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{1}) = LazyArrayStyle{2}()
BroadcastStyle(::LazyArrayStyle{2}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()

bandedcolumns(::AbstractLazyLayout) = BandedColumns{LazyLayout}()
bandedcolumns(::DualLayout{<:AbstractLazyLayout}) = BandedColumns{LazyLayout}()

abstract type AbstractLazyBandedLayout <: AbstractBandedLayout end
abstract type AbstractLazyBlockBandedLayout <: AbstractBlockBandedLayout end
abstract type AbstractLazyBandedBlockBandedLayout <: AbstractBandedBlockBandedLayout end

struct LazyBandedLayout <: AbstractLazyBandedLayout end

sublayout(::AbstractLazyBandedLayout, ::Type{<:NTuple{2,AbstractUnitRange}}) = LazyBandedLayout()


BroadcastStyle(M::ApplyArrayBroadcastStyle{2}, ::BandedStyle) = M
BroadcastStyle(::BandedStyle, M::ApplyArrayBroadcastStyle{2}) = M


bandwidths(M::Applied{<:Any,typeof(*)}) = min.(_bnds(M), prodbandwidths(M.args...))

function bandwidths(L::ApplyMatrix{<:Any,typeof(\)})
    A,B = arguments(L)
    l,u = bandwidths(A)
    if l == u == 0
        bandwidths(B)
    elseif l == 0
        (bandwidth(B,1), size(L,2)-1)
    elseif u == 0
        (size(L,1)-1,bandwidth(B,2))
    else
        (size(L,1)-1 , size(L,2)-1)
    end
end


isbanded(K::Kron{<:Any,2}) = all(isbanded, K.args)
function bandwidths(K::Kron{<:Any,2})
    A,B = K.args
    (size(B,1)*bandwidth(A,1) + max(0,size(B,1)-size(B,2))*size(A,1)   + bandwidth(B,1),
        size(B,2)*bandwidth(A,2) + max(0,size(B,2)-size(B,1))*size(A,2) + bandwidth(B,2))
end

const BandedMatrixTypes = (:AbstractBandedMatrix, :(AdjOrTrans{<:Any,<:AbstractBandedMatrix}),
                                    :(AbstractTriangular{<:Any, <:AbstractBandedMatrix}),
                                    :(Symmetric{<:Any, <:AbstractBandedMatrix}))

const OtherBandedMatrixTypes = (:Zeros, :Eye, :Diagonal, :SymTridiagonal)

for T1 in BandedMatrixTypes, T2 in BandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

for T1 in BandedMatrixTypes, T2 in OtherBandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

for T1 in OtherBandedMatrixTypes, T2 in BandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

###
# Columns as padded
# This is ommitted as it changes the behaviour of slicing B[:,4]
###

# sublayout(::AbstractBandedLayout, ::Type{<:Tuple{KR,Integer}}) where {KR<:AbstractUnitRange{Int}} = 
#     sublayout(PaddedLayout{UnknownLayout}(), Tuple{KR})
# sublayout(::AbstractBandedLayout, ::Type{<:Tuple{Integer,JR}}) where {JR<:AbstractUnitRange{Int}} = 
#     sublayout(PaddedLayout{UnknownLayout}(), Tuple{JR})

# function sub_paddeddata(::BandedColumns, S::SubArray{T,1,<:AbstractMatrix,<:Tuple{AbstractUnitRange{Int},Integer}}) where T
#     P = parent(S)
#     (kr,j) = parentindices(S)
#     data = bandeddata(P)
#     l,u = bandwidths(P)
#     Vcat(Zeros{T}(max(0,j-u-1)), view(data, (kr .- j .+ (u+1)) ∩ axes(data,1), j))
# end


###
# Specialised multiplication for arrays padded for zeros
# needed for ∞-dimensional banded linear algebra
###

function paddeddata(P::PseudoBlockVector)
    C = P.blocks
    data = paddeddata(C)
    ax = axes(P,1)
    N = findblock(ax,length(data))
    n = last(ax[N])
    if n ≠ length(data)
        resizedata!(C,n)
    end
    PseudoBlockVector(data, (ax[Block(1):N],))
end

function sub_materialize(::PaddedLayout, v::AbstractVector{T}, ax::Tuple{<:BlockedUnitRange}) where T
    dat = paddeddata(v)
    PseudoBlockVector(Vcat(dat, Zeros{T}(length(v) - length(dat))), ax)
end

function similar(M::MulAdd{<:AbstractBandedLayout,<:PaddedLayout}, ::Type{T}, axes) where T
    A,x = M.A,M.B
    xf = paddeddata(x)
    n = max(0,min(length(xf) + bandwidth(A,1),length(M)))
    Vcat(Vector{T}(undef, n), Zeros{T}(size(A,1)-n))
end

function materialize!(M::MatMulVecAdd{<:AbstractBandedLayout,<:PaddedLayout,<:PaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())

    x̃ = paddeddata(x)
    resizedata!(y, min(length(M),length(x̃)+bandwidth(A,1)))
    ỹ = paddeddata(y)

    if length(ỹ) < min(length(M),length(x̃)+bandwidth(A,1))
        # its ok if the entries are actually zero
        for k = max(1,length(x̃)-bandwidth(A,1)):length(x̃)
            iszero(x̃[k]) || throw(ArgumentError("Cannot assign non-zero entries to Zero"))
        end
    end

    materialize!(MulAdd(α, view(A, axes(ỹ,1), axes(x̃,1)) , x̃, β, ỹ))
    y
end

# (vec .* mat) * B is typically faster as vec .* (mat * b)
_broadcast_banded_padded_mul((A1,A2)::Tuple{<:AbstractVector,<:AbstractMatrix}, B) = A1 .* mul(A2, B)
_broadcast_banded_padded_mul(Aargs, B) = copy(mulreduce(Mul(BroadcastArray(*, Aargs...), B)))

const AllBlockBandedLayout = Union{AbstractBlockBandedLayout,BlockLayout{<:AbstractBandedLayout}}

function similar(Ml::MulAdd{<:AllBlockBandedLayout,<:PaddedLayout}, ::Type{T}, _) where T
    A,x = Ml.A,Ml.B
    xf = paddeddata(x)
    ax1,ax2 = axes(A)
    N = findblock(ax2,length(xf))
    M = last(blockcolsupport(A,N))
    m = last(ax1[M]) # number of non-zero entries
    PseudoBlockVector(Vcat(Vector{T}(undef, m), Zeros{T}(length(ax1)-m)), (ax1,))
end

function materialize!(M::MatMulVecAdd{<:AllBlockBandedLayout,<:PaddedLayout,<:PaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())

    if !blockisequal(axes(A,2), axes(x,1))
        return muladd!(α, A, PseudoBlockVector(x, (axes(A,2),)), β, y)
    end

    ỹ = paddeddata(y)
    x̃ = paddeddata(x)

    muladd!(α, view(A, blockaxes(ỹ,1), blockaxes(x̃,1)) , x̃, β, ỹ)
    y
end


###
# MulMatrix
###

bandwidths(M::MulMatrix) = bandwidths(Applied(M))
isbanded(M::Applied{<:Any,typeof(*)}) = all(isbanded, M.args)
isbanded(M::MulMatrix) = isbanded(Applied(M))

###
# ApplyBanded
###

struct ApplyBandedLayout{F} <: AbstractLazyBandedLayout end
struct ApplyBlockBandedLayout{F} <: AbstractLazyBlockBandedLayout end
struct ApplyBandedBlockBandedLayout{F} <: AbstractLazyBandedBlockBandedLayout end
StructuredApplyLayouts{F} = Union{ApplyBandedLayout{F},ApplyBlockBandedLayout{F},ApplyBandedBlockBandedLayout{F}}
ApplyLayouts{F} = Union{ApplyLayout{F},ApplyBandedLayout{F},ApplyBlockBandedLayout{F},ApplyBandedBlockBandedLayout{F}}


arguments(::ApplyBandedLayout{F}, A) where F = arguments(ApplyLayout{F}(), A)
sublayout(::ApplyBandedLayout{F}, A) where F = sublayout(ApplyLayout{F}(), A)
@inline _islazy(::StructuredApplyLayouts) = Val(true)

# The following catches the arguments machinery to work for BlockRange
# see LazyArrays.jl/src/mul.jl

_mul_args_colsupport(a, kr::BlockRange) = blockcolsupport(a, kr)
_mul_args_rowsupport(a, kr::BlockRange) = blockrowsupport(a, kr)
_mul_args_colsupport(a, kr::Block) = blockcolsupport(a, kr)
_mul_args_rowsupport(a, kr::Block) = blockrowsupport(a, kr)
_mat_mul_arguments(args, (kr,jr)::Tuple{BlockSlice,BlockSlice}) = _mat_mul_arguments(args, (kr.block, jr.block))

arguments(::ApplyBlockBandedLayout{F}, A) where F = arguments(ApplyLayout{F}(), A)
arguments(::ApplyBandedBlockBandedLayout{F}, A) where F = arguments(ApplyLayout{F}(), A)

sublayout(::ApplyBlockBandedLayout{F}, A) where F = sublayout(ApplyLayout{F}(), A)
sublayout(::ApplyBandedBlockBandedLayout{F}, A) where F = sublayout(ApplyLayout{F}(), A)

sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{Block1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{BlockIndexRange1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{Block1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{BlockIndexRange1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{BlockRange1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{BlockRange1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{Block1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{BlockIndexRange1}}}) where F = BandedBlockBandedLayout()

applylayout(::Type{typeof(*)}, ::AbstractBandedLayout...) = ApplyBandedLayout{typeof(*)}()
applylayout(::Type{typeof(*)}, ::AllBlockBandedLayout...) = ApplyBlockBandedLayout{typeof(*)}()
applylayout(::Type{typeof(*)}, ::AbstractBandedBlockBandedLayout...) = ApplyBandedBlockBandedLayout{typeof(*)}()

applybroadcaststyle(::Type{<:AbstractMatrix}, ::ApplyBandedLayout{typeof(*)}) = BandedStyle()

@inline colsupport(::ApplyBandedLayout{typeof(*)}, A, j) = banded_colsupport(A, j)
@inline rowsupport(::ApplyBandedLayout{typeof(*)}, A, j) = banded_rowsupport(A, j)
@inline _mul_arguments(::ApplyBandedLayout{typeof(*)}, A) = arguments(A)



prodblockbandwidths(A) = blockbandwidths(A)
prodblockbandwidths() = (0,0)
prodblockbandwidths(A...) = broadcast(+, blockbandwidths.(A)...)

prodsubblockbandwidths(A) = subblockbandwidths(A)
prodsubblockbandwidths() = (0,0)
prodsubblockbandwidths(A...) = broadcast(+, subblockbandwidths.(A)...)

blockbandwidths(M::MulMatrix) = prodblockbandwidths(M.args...)
subblockbandwidths(M::MulMatrix) = prodsubblockbandwidths(M.args...)

mulreduce(M::Mul{<:StructuredApplyLayouts{F},<:StructuredApplyLayouts{G}}) where {F,G} = Mul{ApplyLayout{F},ApplyLayout{G}}(M.A, M.B)
mulreduce(M::Mul{<:StructuredApplyLayouts{F},D}) where {F,D} = Mul{ApplyLayout{F},D}(M.A, M.B)
mulreduce(M::Mul{D,<:StructuredApplyLayouts{F}}) where {F,D} = Mul{D,ApplyLayout{F}}(M.A, M.B)
mulreduce(M::Mul{<:StructuredApplyLayouts{F},<:DiagonalLayout}) where F = Rmul(M)
mulreduce(M::Mul{<:DiagonalLayout,<:StructuredApplyLayouts{F}}) where F = Lmul(M)


###
# BroadcastMatrix
###

bandwidths(M::BroadcastMatrix) = bandwidths(broadcasted(M))
# TODO: Generalize
for op in (:+, :-)
    @eval begin
        blockbandwidths(M::BroadcastMatrix{<:Any,typeof($op)}) =
            broadcast(max, map(blockbandwidths,arguments(M))...)
        subblockbandwidths(M::BroadcastMatrix{<:Any,typeof($op)}) =
            broadcast(max, map(subblockbandwidths,arguments(M))...)
    end
end

for func in (:blockbandwidths, :subblockbandwidths)
    @eval begin
        $func(M::BroadcastMatrix{<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}}) = $func(M.args[2])
        $func(M::BroadcastMatrix{<:Any,typeof(*),<:Tuple{<:AbstractMatrix,<:Number}}) = $func(M.args[1])
    end
end

isbanded(M::BroadcastMatrix) = isbanded(broadcasted(M))

struct BroadcastBandedLayout{F} <: AbstractLazyBandedLayout end
struct BroadcastBlockBandedLayout{F} <: AbstractLazyBlockBandedLayout end
struct BroadcastBandedBlockBandedLayout{F} <: AbstractLazyBandedBlockBandedLayout end

StructuredBroadcastLayouts{F} = Union{BroadcastBandedLayout{F},BroadcastBlockBandedLayout{F},BroadcastBandedBlockBandedLayout{F}}
BroadcastLayouts{F} = Union{BroadcastLayout{F},StructuredBroadcastLayouts{F}}


blockbandwidths(B::BroadcastMatrix) = blockbandwidths(broadcasted(B))
subblockbandwidths(B::BroadcastMatrix) = subblockbandwidths(broadcasted(B))

BroadcastLayout(::BroadcastBandedLayout{F}) where F = BroadcastLayout{F}()

broadcastlayout(::Type{F}, ::AbstractBandedLayout) where F = BroadcastBandedLayout{F}()
# functions that satisfy f(0,0) == 0

for op in (:*, :/, :\, :+, :-)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AllBlockBandedLayout, ::AllBlockBandedLayout) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedBlockBandedLayout, ::AbstractBandedBlockBandedLayout) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end
for op in (:*, :/)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::Any) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AllBlockBandedLayout, ::Any) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedBlockBandedLayout, ::Any) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end
for op in (:*, :\)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::Any, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::Any, ::AllBlockBandedLayout) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::Any, ::AbstractBandedBlockBandedLayout) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end


sublayout(LAY::BroadcastBlockBandedLayout, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{BlockRange1}}}) = LAY
sublayout(LAY::BroadcastBandedBlockBandedLayout, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{BlockRange1}}}) = LAY


@inline colsupport(::BroadcastBandedLayout, A, j) = banded_colsupport(A, j)
@inline rowsupport(::BroadcastBandedLayout, A, j) = banded_rowsupport(A, j)

_broadcastarray2broadcasted(::StructuredBroadcastLayouts{F}, A) where F = _broadcastarray2broadcasted(BroadcastLayout{F}(), A)
_broadcastarray2broadcasted(::StructuredBroadcastLayouts{F}, A::BroadcastArray) where F = _broadcastarray2broadcasted(BroadcastLayout{F}(), A)

_copyto!(::AbstractBandedLayout, ::BroadcastBandedLayout, dest::AbstractMatrix, bc::AbstractMatrix) =
    copyto!(dest, _broadcastarray2broadcasted(bc))

_copyto!(_, ::BroadcastBandedLayout, dest::AbstractMatrix, bc::AbstractMatrix) =
    copyto!(dest, _broadcastarray2broadcasted(bc))

_banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractMatrix{T},AbstractMatrix{V}}, _, ::Tuple{<:Any,ApplyBandedLayout{typeof(*)}}) where {T,V} =
    broadcast!(f, dest, BandedMatrix(A), BandedMatrix(B))

copy(M::Mul{BroadcastBandedLayout{typeof(*)}, <:PaddedLayout}) = _broadcast_banded_padded_mul(arguments(BroadcastBandedLayout{typeof(*)}(), M.A), M.B)

function _cache(::AllBlockBandedLayout, A::AbstractMatrix{T}) where T
    kr,jr = axes(A)
    CachedArray(BlockBandedMatrix{T}(undef, (kr[Block.(1:0)], jr[Block.(1:0)]), blockbandwidths(A)), A)
end
###
# copyto!
###

_BandedMatrix(::ApplyBandedLayout{typeof(*)}, V::AbstractMatrix{T}) where T = 
    copyto!(BandedMatrix{T}(undef, axes(V), bandwidths(V)), V)

_broadcast_BandedMatrix(a::AbstractMatrix) = BandedMatrix(a)
_broadcast_BandedMatrix(a) = a
_broadcast_BandedBlockBandedMatrix(a::AbstractMatrix) = BandedBlockBandedMatrix(a)
_broadcast_BandedBlockBandedMatrix(a) = a

for op in (:+, :-, :*)
    @eval begin
        @inline _BandedMatrix(::BroadcastBandedLayout{typeof($op)}, V::AbstractMatrix)::BandedMatrix = broadcast($op, map(_broadcast_BandedMatrix,arguments(V))...)
        _copyto!(::AbstractBandedLayout, ::BroadcastBandedLayout{typeof($op)}, dest::AbstractMatrix, src::AbstractMatrix) =
            broadcast!($op, dest, map(_broadcast_BandedMatrix, arguments(src))...)
        _copyto!(::AbstractBandedBlockBandedLayout, ::BroadcastBandedBlockBandedLayout{typeof($op)}, dest::AbstractMatrix, src::AbstractMatrix) =
            broadcast!($op, dest, map(_broadcast_BandedBlockBandedMatrix, arguments(src))...)
    end
end

_mulbanded_copyto!(dest, a) = copyto!(dest, a)
_mulbanded_copyto!(dest::AbstractArray{T}, a, b) where T = muladd!(one(T), a, b, zero(T), dest)
_mulbanded_copyto!(dest::AbstractArray{T}, a, b, c, d...) where T = _mulbanded_copyto!(dest, mul(a,b), c, d...)

_mulbanded_BandedMatrix(A, _) = A
_mulbanded_BandedMatrix(A, ::NTuple{2,OneTo{Int}}) = BandedMatrix(A)
_mulbanded_BandedMatrix(A) = _mulbanded_BandedMatrix(A, axes(A))

_copyto!(::AbstractBandedLayout, ::ApplyBandedLayout{typeof(*)}, dest::AbstractMatrix, src::AbstractMatrix) =
    _mulbanded_copyto!(dest, map(_mulbanded_BandedMatrix,arguments(src))...)

_mulbanded_BandedBlockBandedMatrix(A, _) = A
_mulbanded_BandedBlockBandedMatrix(A, ::NTuple{2,OneTo{Int}}) = BandedBlockBandedMatrix(A)
_mulbanded_BandedBlockBandedMatrix(A) = _mulbanded_BandedBlockBandedMatrix(A, axes(A))

_copyto!(::AbstractBandedBlockBandedLayout, ::ApplyBandedBlockBandedLayout{typeof(*)}, dest::AbstractMatrix, src::AbstractMatrix) =
    _mulbanded_copyto!(dest, map(_mulbanded_BandedBlockBandedMatrix,arguments(src))...)


arguments(::BroadcastBandedLayout{F}, V::SubArray) where F = _broadcast_sub_arguments(parent(V), V)
arguments(::BroadcastBandedBlockBandedLayout, V::SubArray) = _broadcast_sub_arguments(parent(V), V)


call(b::BroadcastBandedLayout, a) = call(BroadcastLayout(b), a)
call(b::BroadcastBandedLayout, a::SubArray) = call(BroadcastLayout(b), a)

sublayout(M::ApplyBandedLayout{typeof(*)}, ::Type{<:NTuple{2,AbstractUnitRange}}) = M
sublayout(M::BroadcastBandedLayout, ::Type{<:NTuple{2,AbstractUnitRange}}) = M

transposelayout(b::BroadcastBandedLayout) = b
arguments(b::BroadcastBandedLayout, A::AdjOrTrans) where F = arguments(BroadcastLayout(b), A)

sublayout(M::ApplyBlockBandedLayout{typeof(*)}, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{BlockRange1}}}) = M
sublayout(M::ApplyBandedBlockBandedLayout{typeof(*)}, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{BlockRange1}}}) = M


######
# Concat banded matrix
######

# cumsum for tuples
_cumsum(a) = a
_cumsum(a, b...) = tuple(a, (a .+ _cumsum(b...))...)


function bandwidths(M::Vcat{<:Any,2})
    cs = tuple(0, _cumsum(size.(M.args[1:end-1],1)...)...) # cumsum of sizes
    (maximum(cs .+ bandwidth.(M.args,1)), maximum(bandwidth.(M.args,2) .- cs))
end
isbanded(M::Vcat) = all(isbanded, M.args)

function bandwidths(M::Hcat)
    cs = tuple(0, _cumsum(size.(M.args[1:end-1],2)...)...) # cumsum of sizes
    (maximum(bandwidth.(M.args,1) .- cs), maximum(bandwidth.(M.args,2) .+ cs))
end
isbanded(M::Hcat) = all(isbanded, M.args)


const HcatBandedMatrix{T,N} = Hcat{T,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}
const VcatBandedMatrix{T,N} = Vcat{T,2,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}

BroadcastStyle(::Type{HcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()
BroadcastStyle(::Type{VcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()

hcat(A::BandedMatrix...) = BandedMatrix(Hcat(A...))
hcat(A::BandedMatrix, B::AbstractMatrix...) = Matrix(Hcat(A, B...))

vcat(A::BandedMatrix...) = BandedMatrix(Vcat(A...))
vcat(A::BandedMatrix, B::AbstractMatrix...) = Matrix(Vcat(A, B...))



#######
# CachedArray
#######

cachedlayout(::BandedColumns{DenseColumnMajor}, ::AbstractBandedLayout) = BandedColumns{DenseColumnMajor}()
bandwidths(B::CachedMatrix) = bandwidths(B.data)
isbanded(B::CachedMatrix) = isbanded(B.data)

function bandeddata(A::CachedMatrix)
    resizedata!(A, size(A)...)
    bandeddata(A.data)
end

function bandeddata(B::SubArray{<:Any,2,<:CachedMatrix})
    A = parent(B)
    kr,jr = parentindices(B)
    resizedata!(A, maximum(kr), maximum(jr))
    bandeddata(view(A.data,kr,jr))
end

function resize(A::BandedMatrix, n::Integer, m::Integer)
    l,u = bandwidths(A)
    _BandedMatrix(reshape(resize!(vec(bandeddata(A)), (l+u+1)*m), l+u+1, m), n, l,u)
end
function resize(A::BandedSubBandedMatrix, n::Integer, m::Integer)
    l,u = bandwidths(A)
    _BandedMatrix(reshape(resize!(vec(copy(bandeddata(A))), (l+u+1)*m), l+u+1, m), n, l,u)
end
function resize(A::BlockSkylineMatrix{T}, ax::NTuple{2,AbstractUnitRange{Int}}) where T
    l,u = blockbandwidths(A)
    ret = BlockBandedMatrix{T}(undef, ax, (l,u))
    ret.data[1:length(A.data)] .= A.data
    ret
end

function resizedata!(::BandedColumns{DenseColumnMajor}, _, B::AbstractMatrix{T}, n::Integer, m::Integer) where T<:Number
    (n ≤ 0 || m ≤ 0) && return B
    @boundscheck checkbounds(Bool, B, n, m) || throw(ArgumentError("Cannot resize to ($n,$m) which is beyond size $(size(B))"))

    # increase size of array if necessary
    olddata = B.data
    ν,μ = B.datasize
    n,m = max(ν,n), max(μ,m)

    if (ν,μ) ≠ (n,m)
        l,u = bandwidths(B.array)
        λ,ω = bandwidths(B.data)
        if n ≥ size(B.data,1) || m ≥ size(B.data,2)
            M = 2*max(m,n+u)
            B.data = resize(olddata, M+λ, M)
        end
        if ν > 0 # upper-right
            kr = max(1,μ+1-ω):ν
            jr = μ+1:min(m,ν+ω)
            if !isempty(kr) && !isempty(jr)
                view(B.data, kr, jr) .= B.array[kr, jr]
            end
        end
        view(B.data, ν+1:n, μ+1:m) .= B.array[ν+1:n, μ+1:m]
        if μ > 0
            kr = ν+1:min(n,μ+λ)
            jr = max(1,ν+1-λ):μ
            if !isempty(kr) && !isempty(jr)
                view(B.data, kr, jr) .= B.array[kr, jr]
            end
        end
        B.datasize = (n,m)
    end

    B
end

resizedata!(laydat::BlockBandedColumns{<:AbstractColumnMajor}, layarr, B::AbstractMatrix, n::Integer, m::Integer) =
    resizedata!(laydat, layarr, B, findblock.(axes(B), (n,m))...)

function resizedata!(::BlockBandedColumns{<:AbstractColumnMajor}, _, B::AbstractMatrix{T}, N::Block{1}, M::Block{1}) where T<:Number
    (Int(N) ≤ 0 || Int(M) ≤ 0) && return B
    @boundscheck (N in blockaxes(B,1) && M in blockaxes(B,2)) || throw(ArgumentError("Cannot resize to ($N,$M) which is beyond size $(blocksize(B))"))


    N_max, M_max = Block.(blocksize(B))
    # increase size of array if necessary
    olddata = B.data
    ν,μ = B.datasize
    N_old = ν == 0 ? Block(0) : findblock(axes(B)[1], ν)
    M_old = μ == 0 ? Block(0) : findblock(axes(B)[2], μ)
    N,M = max(N_old,N),max(M_old,M)

    n,m = last.(getindex.(axes(B), (N,M)))


    if (ν,μ) ≠ (n,m)
        l,u = blockbandwidths(B.array)
        λ,ω = blockbandwidths(B.data)
        if Int(N) > blocksize(B.data,1) || Int(M) > blocksize(B.data,2)
            M̃ = 2*max(M,N+u)
            B.data = resize(olddata, (axes(B)[1][Block(1):min(M̃+λ,M_max)], axes(B)[2][Block(1):min(M̃,N_max)]))
        end
        if ν > 0 # upper-right
            KR = max(Block(1),M_old+1-ω):N_old
            JR = M_old+1:min(M,N_old+ω)
            if !isempty(KR) && !isempty(JR)
                copyto!(view(B.data, KR, JR), B.array[KR, JR])
            end
        end
        view(B.data, N_old+1:N, M_old+1:M) .= B.array[N_old+1:N, M_old+1:M]
        if μ > 0
            KR = N_old+1:min(N,M_old+λ)
            JR = max(Block(1),N_old+1-λ):M_old
            if !isempty(KR) && !isempty(JR)
                view(B.data, KR, JR) .= B.array[KR, JR]
            end
        end
        B.datasize = (n,m)
    end

    B
end

include("bandedql.jl")
include("blockkron.jl")

###
# Concat and rot ArrayLayouts
###

applylayout(::Type{typeof(vcat)}, ::ZerosLayout, ::AbstractBandedLayout) = ApplyBandedLayout{typeof(vcat)}()
sublayout(::ApplyBandedLayout{typeof(vcat)}, ::Type{<:NTuple{2,AbstractUnitRange}}) where J = ApplyBandedLayout{typeof(vcat)}()

applylayout(::Type{typeof(rot180)}, ::BandedColumns{LAY}) where LAY =
    BandedColumns{typeof(sublayout(LAY(), NTuple{2,StepRange{Int,Int}}))}()

applylayout(::Type{typeof(rot180)}, ::AbstractBandedLayout) =
    ApplyBandedLayout{typeof(rot180)}()

call(::ApplyBandedLayout{typeof(*)}, A::ApplyMatrix{<:Any,typeof(rot180)}) = *
applylayout(::Type{typeof(rot180)}, ::ApplyBandedLayout{typeof(*)}) = ApplyBandedLayout{typeof(*)}()
arguments(::ApplyBandedLayout{typeof(*)}, A::ApplyMatrix{<:Any,typeof(rot180)}) = ApplyMatrix.(rot180, arguments(A.args...))


bandwidths(R::ApplyMatrix{<:Any,typeof(rot180)}) = bandwidths(Applied(R))
function bandwidths(R::Applied{<:Any,typeof(rot180)})
    m,n = size(R)
    sh = m-n
    l,u = bandwidths(arguments(R)[1])
    u+sh,l-sh
end

bandeddata(R::ApplyMatrix{<:Any,typeof(rot180)}) =
    @view bandeddata(arguments(R)[1])[end:-1:1,end:-1:1]


# leave lazy banded matrices lazy when multiplying.
# overload copy as overloading `mulreduce` requires `copyto!` overloads
# Should probably be redesigned in a trait-based way, but hard to see how to do this

BandedLazyLayouts = Union{AbstractLazyBandedLayout, BandedColumns{LazyLayout}, BandedRows{LazyLayout},
                TriangularLayout{UPLO,UNIT,BandedRows{LazyLayout}} where {UPLO,UNIT},
                TriangularLayout{UPLO,UNIT,BandedColumns{LazyLayout}} where {UPLO,UNIT}}

StructuredLazyLayouts = Union{BandedLazyLayouts,
                BlockBandedColumns{LazyLayout}, BandedBlockBandedColumns{LazyLayout}, BlockLayout{LazyLayout},
                BlockLayout{TridiagonalLayout{LazyLayout}}, BlockLayout{DiagonalLayout{LazyLayout}}, 
                BlockLayout{BidiagonalLayout{LazyLayout}}, BlockLayout{SymTridiagonalLayout{LazyLayout}},
                AbstractLazyBlockBandedLayout, AbstractLazyBandedBlockBandedLayout}


@inline _islazy(::StructuredLazyLayouts) = Val(true)

copy(M::Mul{<:StructuredLazyLayouts, <:StructuredLazyLayouts}) = lazymaterialize(M)
copy(M::Mul{<:StructuredLazyLayouts}) = lazymaterialize(M)
copy(M::Mul{<:Any, <:StructuredLazyLayouts}) = lazymaterialize(M)
copy(M::Mul{<:StructuredLazyLayouts, <:AbstractLazyLayout}) = lazymaterialize(M)
copy(M::Mul{<:AbstractLazyLayout, <:StructuredLazyLayouts}) = lazymaterialize(M)
copy(M::Mul{<:StructuredLazyLayouts, <:DiagonalLayout}) = lazymaterialize(M)
copy(M::Mul{<:DiagonalLayout, <:StructuredLazyLayouts}) = lazymaterialize(M)
copy(M::Mul{<:StructuredLazyLayouts, <:DiagonalLayout{<:OnesLayout}}) = copy(Rmul(M))
copy(M::Mul{<:DiagonalLayout{<:OnesLayout}, <:StructuredLazyLayouts}) = copy(Lmul(M))
copy(M::Mul{<:StructuredApplyLayouts{typeof(*)},<:StructuredApplyLayouts{typeof(*)}}) = lazymaterialize(*, arguments(M.A)..., arguments(M.B)...)
copy(M::Mul{<:StructuredApplyLayouts{typeof(*)},<:StructuredLazyLayouts}) = lazymaterialize(*, arguments(M.A)..., M.B)
copy(M::Mul{<:StructuredLazyLayouts,<:StructuredApplyLayouts{typeof(*)}}) = lazymaterialize(*, M.A, arguments(M.B)...)
copy(M::Mul{<:StructuredApplyLayouts{typeof(*)},<:BroadcastLayouts}) = lazymaterialize(*, arguments(M.A)..., M.B)
copy(M::Mul{<:BroadcastLayouts,<:StructuredApplyLayouts{typeof(*)}}) = lazymaterialize(*, M.A, arguments(M.B)...)
copy(M::Mul{ApplyLayout{typeof(*)},<:StructuredLazyLayouts}) = lazymaterialize(*, arguments(M.A)..., M.B)
copy(M::Mul{<:StructuredLazyLayouts,ApplyLayout{typeof(*)}}) = lazymaterialize(*, M.A, arguments(M.B)...)
copy(M::Mul{ApplyLayout{typeof(*)},<:BroadcastLayouts}) = lazymaterialize(*, arguments(M.A)..., M.B)
copy(M::Mul{<:BroadcastLayouts,ApplyLayout{typeof(*)}}) = lazymaterialize(*, M.A, arguments(M.B)...)
copy(M::Mul{<:AbstractInvLayout,<:StructuredLazyLayouts}) = ArrayLayouts.ldiv(pinv(M.A), M.B)

## padded copy
mulreduce(M::Mul{<:StructuredLazyLayouts, <:PaddedLayout}) = MulAdd(M)
mulreduce(M::Mul{<:StructuredApplyLayouts{F}, D}) where {F,D<:PaddedLayout} = Mul{ApplyLayout{F},D}(M.A, M.B)
# need to overload copy due to above
copy(M::Mul{<:StructuredLazyLayouts, <:PaddedLayout}) = copy(mulreduce(M))

##
# support Inf Block ranges
broadcasted(::LazyArrayStyle{1}, ::Type{Block}, r::AbstractUnitRange) = Block(first(r)):Block(last(r))
broadcasted(::LazyArrayStyle{1}, ::Type{Int}, block_range::BlockRange{1}) = first(block_range.indices)
broadcasted(::LazyArrayStyle{0}, ::Type{Int}, block::Block{1}) = Int(block)

end