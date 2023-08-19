module LazyBandedMatrices
using ArrayLayouts: symmetriclayout
using BandedMatrices, BlockBandedMatrices, BlockArrays, LazyArrays,
        ArrayLayouts, MatrixFactorizations, Base, StaticArrays

import LinearAlgebra

import MatrixFactorizations: ql, ql!, QLPackedQ, QRPackedQ, reflector!, reflectorApply!,
            QLPackedQLayout, QRPackedQLayout, AdjQLPackedQLayout, AdjQRPackedQLayout

import Base: BroadcastStyle, similar, OneTo, oneto, copy, *, axes, size, getindex, tail, convert, resize!, tuple_type_tail, view
import Base.Broadcast: Broadcasted, broadcasted, instantiate
import LinearAlgebra: kron, hcat, vcat, AdjOrTrans, AbstractTriangular, BlasFloat, BlasComplex, BlasReal,
                        lmul!, rmul!, checksquare, StructuredMatrixStyle, adjoint, transpose,
                        Symmetric, Hermitian, Adjoint, Transpose, Diagonal, eigvals, eigen, pinv

import ArrayLayouts: materialize!, colsupport, rowsupport, MatMulVecAdd, MatMulMatAdd, require_one_based_indexing,
                    sublayout, transposelayout, conjlayout, _copyto!, MemoryLayout, AbstractQLayout, 
                    OnesLayout, DualLayout, mulreduce, _inv, symtridiagonallayout, tridiagonallayout, bidiagonallayout,
                    bidiagonaluplo, diagonaldata, subdiagonaldata, supdiagonaldata, mul,
                    symmetriclayout, hermitianlayout, _fill_lmul!, _copy_oftype
import LazyArrays: LazyArrayStyle, combine_mul_styles, PaddedLayout,
                        broadcastlayout, applylayout, arguments, _mul_arguments, call,
                        LazyArrayApplyStyle, ApplyArrayBroadcastStyle, ApplyStyle,
                        LazyLayout, AbstractLazyLayout, ApplyLayout, BroadcastLayout, CachedVector, AbstractInvLayout,
                        _mat_mul_arguments, paddeddata, paddeddata_axes, sub_paddeddata, sub_materialize, lazymaterialize,
                        MulMatrix, Mul, CachedMatrix, CachedArray, AbstractCachedMatrix, AbstractCachedArray, cachedlayout, _cache,
                        resizedata!, applybroadcaststyle, _broadcastarray2broadcasted,
                        LazyMatrix, LazyVector, LazyArray, MulAddStyle, _broadcast_sub_arguments,
                        _mul_args_colsupport, _mul_args_rowsupport, _islazy, simplifiable, simplify, convexunion, tuple_type_memorylayouts,
                        PaddedArray, DualOrPaddedLayout, layout_broadcasted
import BandedMatrices: bandedcolumns, bandwidths, isbanded, AbstractBandedLayout,
                        prodbandwidths, BandedStyle, BandedColumns, BandedRows, BandedLayout,
                        AbstractBandedMatrix, BandedSubBandedMatrix, BandedStyle, _bnds,
                        banded_rowsupport, banded_colsupport, _BandedMatrix, bandeddata,
                        banded_qr_lmul!, banded_qr_rmul!, _banded_broadcast!, bandedbroadcaststyle
import BlockBandedMatrices: BlockSlice, Block1, AbstractBlockBandedLayout,
                        isblockbanded, isbandedblockbanded, blockbandwidths,
                        bandedblockbandedbroadcaststyle, bandedblockbandedcolumns,
                        BandedBlockBandedColumns, BlockBandedColumns, BlockBandedRows, BandedBlockBandedRows,
                        subblockbandwidths, BandedBlockBandedMatrix, BlockBandedMatrix, BlockBandedLayout,
                        AbstractBandedBlockBandedLayout, BandedBlockBandedLayout, BandedBlockBandedStyle, BlockBandedStyle,
                        blockcolsupport, BlockRange1, blockrowsupport, BlockIndexRange1,
                        BlockBandedColumnMajor
import BlockArrays: BlockSlice1, BlockLayout, AbstractBlockStyle, block, blockindex, BlockKron, viewblock, blocks, BlockSlices, AbstractBlockLayout, blockvec

# for bidiag/tridiag
import Base: -, +, *, /, \, ==, AbstractMatrix, Matrix, Array, size, conj, real, imag, copy,
            iszero, isone, one, zero, getindex, setindex!, copyto!, fill, fill!, promote_rule, show, print_matrix, permutedims
import LinearAlgebra: transpose, adjoint, istriu, istril, isdiag, tril!, triu!, det, logabsdet,
                        symmetric, symmetric_type, diag, issymmetric, UniformScaling,
                        LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular, char_uplo

if VERSION ≥ v"1.11.0-DEV.21"
    using LinearAlgebra: UpperOrLowerTriangular
else
    const UpperOrLowerTriangular{T,S} = Union{LinearAlgebra.UpperTriangular{T,S},
                                              LinearAlgebra.UnitUpperTriangular{T,S},
                                              LinearAlgebra.LowerTriangular{T,S},
                                              LinearAlgebra.UnitLowerTriangular{T,S}}
end

export DiagTrav, KronTrav, blockkron, BlockKron, BlockBroadcastArray, BlockVcat, BlockHcat, BlockHvcat, unitblocks

include("tridiag.jl")
include("bidiag.jl")
include("special.jl")

abstract type AbstractLazyBandedLayout <: AbstractBandedLayout end
struct LazyBandedLayout <: AbstractLazyBandedLayout end
sublayout(::AbstractLazyBandedLayout, ::Type{<:NTuple{2,AbstractUnitRange}}) = LazyBandedLayout()
symmetriclayout(::AbstractLazyBandedLayout) = SymmetricLayout{LazyBandedLayout}()
hermitianlayout(::Type{<:Real}, ::AbstractLazyBandedLayout) = SymmetricLayout{LazyBandedLayout}()
hermitianlayout(::Type{<:Complex}, ::AbstractLazyBandedLayout) = HermitianLayout{LazyBandedLayout}()

bandedbroadcaststyle(::LazyArrayStyle) = LazyArrayStyle{2}()

BroadcastStyle(::LazyArrayStyle{1}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{1}) = LazyArrayStyle{2}()
BroadcastStyle(::LazyArrayStyle{2}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()

BroadcastStyle(::LazyArrayStyle{N}, ::AbstractBlockStyle{N}) where N = LazyArrayStyle{N}()
BroadcastStyle(::AbstractBlockStyle{N}, ::LazyArrayStyle{N}) where N = LazyArrayStyle{N}()

bandedcolumns(::AbstractLazyLayout) = BandedColumns{LazyLayout}()
bandedcolumns(::DualLayout{<:AbstractLazyLayout}) = BandedColumns{LazyLayout}()


abstract type AbstractLazyBlockBandedLayout <: AbstractBlockBandedLayout end
abstract type AbstractLazyBandedBlockBandedLayout <: AbstractBandedBlockBandedLayout end

struct LazyBlockBandedLayout <: AbstractLazyBlockBandedLayout end
struct LazyBandedBlockBandedLayout <: AbstractLazyBandedBlockBandedLayout end

const StructuredLayoutTypes{Lay} = Union{SymmetricLayout{Lay}, HermitianLayout{Lay}, TriangularLayout{'L','N',Lay}, TriangularLayout{'U','N',Lay}, TriangularLayout{'L','U',Lay}, TriangularLayout{'U','U',Lay}}

const BandedLayouts = Union{AbstractBandedLayout, StructuredLayoutTypes{<:AbstractBandedLayout}, DualOrPaddedLayout}
const BlockBandedLayouts = Union{AbstractBlockBandedLayout, BlockLayout{<:AbstractBandedLayout}, StructuredLayoutTypes{<:AbstractBlockBandedLayout}}
const BandedBlockBandedLayouts = Union{AbstractBandedBlockBandedLayout,DiagonalLayout{<:AbstractBlockLayout}, StructuredLayoutTypes{<:AbstractBandedBlockBandedLayout}}


const LazyBandedBlockBandedLayouts = Union{AbstractLazyBandedBlockBandedLayout,BandedBlockBandedColumns{<:AbstractLazyLayout}, BandedBlockBandedRows{<:AbstractLazyLayout}, StructuredLayoutTypes{<:AbstractLazyBandedBlockBandedLayout}}


BroadcastStyle(M::ApplyArrayBroadcastStyle{2}, ::BandedStyle) = M
BroadcastStyle(::BandedStyle, M::ApplyArrayBroadcastStyle{2}) = M

transposelayout(::AbstractLazyBandedBlockBandedLayout) = LazyBandedBlockBandedLayout()
transposelayout(::AbstractLazyBlockBandedLayout) = LazyBlockBandedLayout()
conjlayout(::Type{<:Complex}, ::AbstractLazyBandedBlockBandedLayout) = LazyBandedBlockBandedLayout()
conjlayout(::Type{<:Complex}, ::AbstractLazyBlockBandedLayout) = LazyBlockBandedLayout()

symmetriclayout(::LazyBandedBlockBandedLayouts) = LazyBandedBlockBandedLayout()
hermitianlayout(_, ::LazyBandedBlockBandedLayouts) = LazyBandedBlockBandedLayout()

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

function bandwidths(L::ApplyMatrix{<:Any,typeof(inv)})
    A, = arguments(L)
    l,u = bandwidths(A)
    l == u == 0 && return (0,0)
    m,n = size(A)
    l == 0 && return (0,n-1)
    u == 0 && return (m-1,0)
    (m-1 , n-1)
end

function colsupport(::AbstractInvLayout{<:AbstractBandedLayout}, A, j)
    l,u = bandwidths(A)
    l == 0 && u == 0 && return first(j):last(j)
    m,_ = size(A)
    l == 0 && return 1:last(j)
    u == 0 && return first(j):m
    1:m
end

function rowsupport(::AbstractInvLayout{<:AbstractBandedLayout}, A, k)
    l,u = bandwidths(A)
    l == 0 && u == 0 && return first(k):last(k)
    _,n = size(A)
    l == 0 && return first(k):n
    u == 0 && return 1:last(k)
    1:n
end


isbanded(K::Kron{<:Any,2}) = all(isbanded, K.args)
function bandwidths(K::Kron{<:Any,2})
    A,B = K.args
    (size(B,1)*bandwidth(A,1) + max(0,size(B,1)-size(B,2))*size(A,1)   + bandwidth(B,1),
        size(B,2)*bandwidth(A,2) + max(0,size(B,2)-size(B,1))*size(A,2) + bandwidth(B,2))
end

const BandedMatrixTypes = (:AbstractBandedMatrix, :(AdjOrTrans{<:Any,<:AbstractBandedMatrix}),
                                    :(UpperOrLowerTriangular{<:Any, <:AbstractBandedMatrix}),
                                    :(Symmetric{<:Any, <:AbstractBandedMatrix}))

const OtherBandedMatrixTypes = (:Zeros, :Eye, :Diagonal, :(LinearAlgebra.SymTridiagonal))

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
# it's activated in InfiniteLinearAlgebra
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



_makevec(data::AbstractVector) = data
_makevec(data::Number) = [data]

# make sure data is big enough for blocksize
function _block_paddeddata(C::CachedVector, data::AbstractVector, n)
    if n > length(data)
        resizedata!(C,n)
        data = paddeddata(C)
    end
    _makevec(data)
end

_block_paddeddata(C, data::Union{Number,AbstractVector}, n) = Vcat(data, Zeros{eltype(data)}(n-length(data)))
_block_paddeddata(C, data::Union{Number,AbstractMatrix}, n, m) = PaddedArray(data, n, m)

function resizedata!(P::PseudoBlockVector, n::Integer)
    ax = axes(P,1)
    N = findblock(ax,n)
    resizedata!(P.blocks, last(ax[N]))
    P
end

function paddeddata(P::PseudoBlockVector)
    C = P.blocks
    ax = axes(P,1)
    data = paddeddata(C)
    N = findblock(ax,max(length(data),1))
    n = last(ax[N])
    PseudoBlockVector(_block_paddeddata(C, data, n), (ax[Block(1):N],))
end

function paddeddata_axes((ax,)::Tuple{BlockedUnitRange}, A)
    data = A.args[2]
    N = findblock(ax,max(length(data),1))
    n = last(ax[N])
    PseudoBlockVector(_block_paddeddata(nothing, data, n), (ax[Block(1):N],))
end

function paddeddata(P::PseudoBlockMatrix)
    C = P.blocks
    ax,bx = axes(P)
    data = paddeddata(C)
    N = findblock(ax,max(size(data,1),1))
    M = findblock(bx,max(size(data,2),1))
    n,m = last(ax[N]),last(bx[M])
    PseudoBlockArray(_block_paddeddata(C, data, n, m), (ax[Block(1):N],bx[Block(1):M]))
end

blockcolsupport(::PaddedLayout, A, j) = Block.(OneTo(blocksize(paddeddata(A),1)))
blockrowsupport(::PaddedLayout, A, k) = Block.(OneTo(blocksize(paddeddata(A),2)))

function sub_materialize(::PaddedLayout, v::AbstractVector{T}, ax::Tuple{<:BlockedUnitRange}) where T
    dat = paddeddata(v)
    PseudoBlockVector(Vcat(sub_materialize(dat), Zeros{T}(length(v) - length(dat))), ax)
end

function sub_materialize(::PaddedLayout, V::AbstractMatrix{T}, ::Tuple{BlockedUnitRange,AbstractUnitRange}) where T
    dat = paddeddata(V)
    ApplyMatrix{T}(setindex, Zeros{T}(axes(V)), sub_materialize(dat), axes(dat)...)
end

function sub_materialize(::PaddedLayout, V::AbstractMatrix{T}, ::Tuple{BlockedUnitRange,BlockedUnitRange}) where T
    dat = paddeddata(V)
    ApplyMatrix{T}(setindex, Zeros{T}(axes(V)), sub_materialize(dat), axes(dat)...)
end

function sub_materialize(::PaddedLayout, V::AbstractMatrix{T}, ::Tuple{AbstractUnitRange,BlockedUnitRange}) where T
    dat = paddeddata(V)
    ApplyMatrix{T}(setindex, Zeros{T}(axes(V)), sub_materialize(dat), axes(dat)...)
end

function similar(M::MulAdd{<:BandedLayouts,<:PaddedLayout}, ::Type{T}, axes::Tuple{Any}) where T
    A,x = M.A,M.B
    xf = paddeddata(x)
    n = max(0,min(length(xf) + bandwidth(A,1),length(M)))
    Vcat(Vector{T}(undef, n), Zeros{T}(size(A,1)-n))
end

function similar(M::MulAdd{<:BandedLayouts,<:PaddedLayout}, ::Type{T}, axes::Tuple{Any,Any}) where T
    A,x = M.A,M.B
    xf = paddeddata(x)
    m = max(0,min(size(xf,1) + bandwidth(A,1),size(M,1)))
    n = size(xf,2)
    PaddedArray(Matrix{T}(undef, m, n), size(A,1), size(x,2))
end

function materialize!(M::MatMulVecAdd{<:BandedLayouts,<:PaddedLayout,<:PaddedLayout})
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

    muladd!(α, view(A, axes(ỹ,1), axes(x̃,1)) , x̃, β, ỹ)
    y
end

function materialize!(M::MatMulMatAdd{<:BandedLayouts,<:PaddedLayout,<:PaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    size(y) == (size(A,1),size(x,2)) || throw(DimensionMismatch())
    size(x,1) == size(A,2) || throw(DimensionMismatch())

    x̃ = paddeddata(x)
    resizedata!(y, min(size(M,1),size(x̃,1)+bandwidth(A,1)), min(size(M,2),size(x̃,2)))
    ỹ = paddeddata(y)

    if size(ỹ,1) < min(size(M,1),size(x̃,1)+bandwidth(A,1))
        # its ok if the entries are actually zero
        for j = 1:size(x̃,2), k = max(1,size(ỹ,1)-bandwidth(A,1)+1):size(x̃,1)
            iszero(x̃[k,j]) || throw(ArgumentError("Cannot assign non-zero entry $k,$j to zero"))
        end
    end

    muladd!(α, view(A, axes(ỹ,1), axes(x̃,1)) , x̃, β, view(ỹ,:,axes(x̃,2)))
    _fill_lmul!(β, view(ỹ,:,size(x̃,2)+1:size(ỹ,2)))
    y
end

# (vec .* mat) * B is typically faster as vec .* (mat * b)
_broadcast_banded_padded_mul((A1,A2)::Tuple{<:AbstractVector,<:AbstractMatrix}, B) = A1 .* mul(A2, B)
_broadcast_banded_padded_mul(Aargs, B) = copy(mulreduce(Mul(BroadcastArray(*, Aargs...), B)))

_block_last(b::Block) = b
_block_last(b::AbstractVector{<:Block}) = last(b)
function similar(Ml::MulAdd{<:BlockBandedLayouts,<:PaddedLayout}, ::Type{T}, _) where T
    A,x = Ml.A,Ml.B
    xf = paddeddata(x)
    ax1,ax2 = axes(A)
    N = findblock(ax2,length(xf))
    M = _block_last(blockcolsupport(A,N))
    isfinite(Integer(M)) || error("cannot multiply matrix with infinite block support")
    m = last(ax1[M]) # number of non-zero entries
    c = cache(Zeros{T}(length(ax1)))
    resizedata!(c, m)
    PseudoBlockVector(c, (ax1,))
end

function materialize!(M::MatMulVecAdd{<:BlockBandedLayouts,<:PaddedLayout,<:PaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())
    
    ỹ = paddeddata(y)

    if !blockisequal(axes(A,2), axes(x,1))
        x̃2 = paddeddata(x)
        muladd!(α, view(A, axes(ỹ,1), axes(x̃2,1)), x̃2, β, ỹ)
    else
        x̃ = paddeddata(x)
        muladd!(α, view(A, axes(ỹ,1), axes(x̃,1)), x̃, β, ỹ)
    end
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
sublayout(::ApplyBandedLayout, ::Type{<:NTuple{2,AbstractUnitRange}}) = LazyBandedLayout()
LazyArrays._mul_arguments(::StructuredApplyLayouts{F}, A) where F = LazyArrays._mul_arguments(ApplyLayout{F}(), A)
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

sublayout(::ApplyBlockBandedLayout, ::Type{<:Tuple{<:BlockSlice{<:BlockRange1}, <:BlockSlice{<:BlockRange1}}}) = BlockBandedLayout()

sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{Block1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice{<:BlockIndexRange1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice{Block1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{<:BlockIndexRange1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{<:BlockRange1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{Block1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockIndexRange1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice{<:BlockRange1}}}) where F = BandedBlockBandedLayout()

applylayout(::Type{typeof(*)}, ::BandedLayouts...) = ApplyBandedLayout{typeof(*)}()
applylayout(::Type{typeof(*)}, ::BlockBandedLayouts...) = ApplyBlockBandedLayout{typeof(*)}()
applylayout(::Type{typeof(*)}, ::BandedBlockBandedLayouts...) = ApplyBandedBlockBandedLayout{typeof(*)}()


applybroadcaststyle(::Type{<:AbstractMatrix}, ::ApplyBandedLayout) = LazyArrayStyle{2}()
applybroadcaststyle(::Type{<:AbstractMatrix}, ::ApplyBlockBandedLayout) = LazyArrayStyle{2}()
applybroadcaststyle(::Type{<:AbstractMatrix}, ::ApplyBandedBlockBandedLayout) = LazyArrayStyle{2}()

BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{BlockSlice{<:Any,<:BlockedUnitRange},Vararg{Any}}} = applybroadcaststyle(M, MemoryLayout(M))
BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{BlockSlice{<:Any,<:BlockedUnitRange},BlockSlice{<:Any,<:BlockedUnitRange},Vararg{Any}}} = applybroadcaststyle(M, MemoryLayout(M))
BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{Any,BlockSlice{<:Any,<:BlockedUnitRange},Vararg{Any}}} = applybroadcaststyle(M, MemoryLayout(M))

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

isbanded(M::BroadcastMatrix) = all(isfinite, bandwidths(M))

struct BroadcastBandedLayout{F} <: AbstractLazyBandedLayout end
struct BroadcastBlockBandedLayout{F} <: AbstractLazyBlockBandedLayout end
struct BroadcastBandedBlockBandedLayout{F} <: AbstractLazyBandedBlockBandedLayout end

StructuredBroadcastLayouts{F} = Union{BroadcastBandedLayout{F},BroadcastBlockBandedLayout{F},BroadcastBandedBlockBandedLayout{F}}
BroadcastLayouts{F} = Union{BroadcastLayout{F},StructuredBroadcastLayouts{F}}


blockbandwidths(B::BroadcastMatrix) = blockbandwidths(broadcasted(B))
subblockbandwidths(B::BroadcastMatrix) = subblockbandwidths(broadcasted(B))

BroadcastLayout(::BroadcastBandedLayout{F}) where F = BroadcastLayout{F}()

broadcastlayout(::Type{F}, ::AbstractBandedLayout) where F = BroadcastBandedLayout{F}()
broadcastlayout(::Type{F}, ::AbstractBlockBandedLayout) where F = BroadcastBlockBandedLayout{F}()
broadcastlayout(::Type{F}, ::AbstractBandedBlockBandedLayout) where F = BroadcastBandedBlockBandedLayout{F}()
# functions that satisfy f(0,0) == 0

for op in (:+, :-)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::PaddedLayout) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::PaddedLayout, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
    end
end

for op in (:*, :/, :\, :+, :-)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::BlockBandedLayouts, ::BlockBandedLayouts) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::BandedBlockBandedLayouts, ::BandedBlockBandedLayouts) = BroadcastBandedBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::DiagonalLayout, ::AbstractBlockBandedLayout) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AbstractBlockBandedLayout, ::DiagonalLayout) = BroadcastBlockBandedLayout{typeof($op)}()
    end
end
for op in (:*, :/)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::Any) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::BlockBandedLayouts, ::Any) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::BandedBlockBandedLayouts, ::Any) = BroadcastBandedBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::BandedBlockBandedLayouts, ::DiagonalLayout) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end
for op in (:*, :\)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::Any, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::Any, ::BlockBandedLayouts) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::Any, ::BandedBlockBandedLayouts) = BroadcastBandedBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::DiagonalLayout, ::BandedBlockBandedLayouts) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end


sublayout(LAY::BroadcastBlockBandedLayout, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) = LAY
sublayout(LAY::BroadcastBandedBlockBandedLayout, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) = LAY


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

broadcasted(::LazyArrayStyle, ::typeof(*), c::Number, A::BandedMatrix) = _BandedMatrix(c .* A.data, A.raxis, A.l, A.u)
broadcasted(::LazyArrayStyle, ::typeof(*), A::BandedMatrix, c::Number) = _BandedMatrix(A.data .* c, A.raxis, A.l, A.u)
broadcasted(::LazyArrayStyle, ::typeof(\), c::Number, A::BandedMatrix) = _BandedMatrix(c .\ A.data, A.raxis, A.l, A.u)
broadcasted(::LazyArrayStyle, ::typeof(/), A::BandedMatrix, c::Number) = _BandedMatrix(A.data ./ c, A.raxis, A.l, A.u)


copy(M::Mul{BroadcastBandedLayout{typeof(*)}, <:PaddedLayout}) = _broadcast_banded_padded_mul(arguments(BroadcastBandedLayout{typeof(*)}(), M.A), M.B)

function _cache(::BlockBandedLayouts, A::AbstractMatrix{T}) where T
    kr,jr = axes(A)
    CachedArray(BlockBandedMatrix{T}(undef, (kr[Block.(1:0)], jr[Block.(1:0)]), blockbandwidths(A)), A)
end
###
# copyto!
###

_BandedMatrix(::ApplyBandedLayout{typeof(*)}, V::AbstractMatrix{T}) where T = 
    copyto!(BandedMatrix{T}(undef, axes(V), bandwidths(V)), V)
_BandedMatrix(::BroadcastBandedLayout, V::AbstractMatrix{T}) where T = 
    copyto!(BandedMatrix{T}(undef, axes(V), bandwidths(V)), broadcasted(V))

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
_mulbanded_BandedBlockBandedMatrix(A, ::NTuple{2,Int}) = BandedBlockBandedMatrix(A)
_mulbanded_BandedBlockBandedMatrix(A) = _mulbanded_BandedBlockBandedMatrix(A, size(A))

_copyto!(::AbstractBandedBlockBandedLayout, ::ApplyBandedBlockBandedLayout{typeof(*)}, dest::AbstractMatrix, src::AbstractMatrix) =
    _mulbanded_copyto!(dest, map(_mulbanded_BandedBlockBandedMatrix,arguments(src))...)


function getindex(A::ApplyMatrix{<:Any,typeof(*)}, kr::BlockRange{1}, jr::BlockRange{1})
    args = A.args
    kjr = intersect.(LazyArrays._mul_args_rows(kr, args...), LazyArrays._mul_args_cols(jr, reverse(args)...))
    *(map(getindex, args, (kr, kjr...), (kjr..., jr))...)
end

arguments(::BroadcastBandedLayout{F}, V::SubArray) where F = _broadcast_sub_arguments(V)
arguments(::BroadcastBandedBlockBandedLayout, V::SubArray) = _broadcast_sub_arguments(V)


call(b::BroadcastBandedLayout, a) = call(BroadcastLayout(b), a)
call(b::BroadcastBandedLayout, a::SubArray) = call(BroadcastLayout(b), a)
call(lay::BroadcastLayout, a::PseudoBlockArray) = call(lay, a.blocks)

sublayout(M::ApplyBandedLayout{typeof(*)}, ::Type{<:NTuple{2,AbstractUnitRange}}) = M
sublayout(M::BroadcastBandedLayout, ::Type{<:NTuple{2,AbstractUnitRange}}) = M

transposelayout(b::BroadcastBandedLayout) = b
arguments(b::BroadcastBandedLayout, A::AdjOrTrans) = arguments(BroadcastLayout(b), A)

sublayout(M::ApplyBlockBandedLayout{typeof(*)}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) = M
sublayout(M::ApplyBandedBlockBandedLayout{typeof(*)}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) = M


######
# Concat banded matrix
######


const ZerosLayouts = Union{ZerosLayout,DualLayout{ZerosLayout}}
const ScalarOrZerosLayouts = Union{ScalarLayout,ZerosLayouts}
const ScalarOrBandedLayouts = Union{ScalarOrZerosLayouts,BandedLayouts}

for op in (:hcat, :vcat)
    @eval begin
        applylayout(::Type{typeof($op)}, ::A, ::ZerosLayout) where A<:ScalarOrBandedLayouts = PaddedLayout{A}()
        applylayout(::Type{typeof($op)}, ::A, ::ZerosLayout) where A<:ScalarOrZerosLayouts = PaddedLayout{A}()
        applylayout(::Type{typeof($op)}, ::A, ::PaddedLayout) where A<:ScalarOrBandedLayouts = PaddedLayout{ApplyLayout{typeof($op)}}()
        applylayout(::Type{typeof($op)}, ::ScalarOrBandedLayouts...) = ApplyBandedLayout{typeof($op)}()
        applylayout(::Type{typeof($op)}, ::ScalarOrZerosLayouts...) = ApplyLayout{typeof($op)}()
        sublayout(::ApplyBandedLayout{typeof($op)}, ::Type{<:NTuple{2,AbstractUnitRange}}) = ApplyBandedLayout{typeof($op)}()
    end
end

applylayout(::Type{typeof(hvcat)}, _, ::ScalarOrBandedLayouts...)= ApplyBandedLayout{typeof(hvcat)}()


# cumsum for tuples
_cumsum(a) = a
_cumsum(a, b...) = tuple(a, (a .+ _cumsum(b...))...)

_bandwidth(a::Number, n) = iszero(a) ? bandwidth(Zeros{typeof(a)}(1,1),n) : 0
_bandwidth(a, n) = bandwidth(a, n)

_bandwidths(a::Number) = iszero(a) ? bandwidths(Zeros{typeof(a)}(1,1)) : (0,0)
_bandwidths(a) = bandwidths(a)

function bandwidths(M::Vcat{<:Any,2})
    cs = tuple(0, _cumsum(size.(M.args[1:end-1],1)...)...) # cumsum of sizes
    (maximum(cs .+ _bandwidth.(M.args,1)), maximum(_bandwidth.(M.args,2) .- cs))
end
isbanded(M::Vcat) = all(isbanded, M.args)

function bandwidths(M::Hcat)
    cs = tuple(0, _cumsum(size.(M.args[1:end-1],2)...)...) # cumsum of sizes
    (maximum(_bandwidth.(M.args,1) .- cs), maximum(_bandwidth.(M.args,2) .+ cs))
end
isbanded(M::Hcat) = all(isbanded, M.args)

function bandwidths(M::ApplyMatrix{<:Any,typeof(hvcat),<:Tuple{Int,Vararg{Any}}})
    N = first(M.args)
    args = tail(M.args)
    @assert length(args) == N^2
    rs = tuple(0, _cumsum(size.(args[1:N:end-2N+1],1)...)...) # cumsum of sizes
    cs = tuple(0, _cumsum(size.(args[1:N-1],2)...)...) # cumsum of sizes

    l,u = _bandwidth(args[1],1)::Int,_bandwidth(args[1],2)::Int
    for K = 1:N, J = 1:N
        if !(K == J == 1)
            λ,μ = _bandwidth(args[J+N*(K-1)],1),_bandwidth(args[J+N*(K-1)],2)
            if λ ≥ -μ # don't do anything if bandwidths are empty
                l = max(l,λ + rs[K] - cs[J])::Int
                u = max(u,μ + cs[K] - rs[J])::Int
            end
        end
    end
    l,u
end

# just support padded for now
bandwidths(::PaddedLayout, A) = _bandwidths(paddeddata(A))
isbanded(::PaddedLayout, A) = true # always treat as banded



const HcatBandedMatrix{T,N} = Hcat{T,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}
const VcatBandedMatrix{T,N} = Vcat{T,2,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}

BroadcastStyle(::Type{HcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()
BroadcastStyle(::Type{VcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()

Base.typed_hcat(::Type{T}, A::BandedMatrix, B::BandedMatrix...) where T = BandedMatrix{T}(Hcat{T}(A, B...))
Base.typed_hcat(::Type{T}, A::BandedMatrix, B::AbstractVecOrMat...) where T = Matrix{T}(Hcat{T}(A, B...))

Base.typed_vcat(::Type{T}, A::BandedMatrix...) where T = BandedMatrix{T}(Vcat{T}(A...))
Base.typed_vcat(::Type{T}, A::BandedMatrix, B::AbstractVecOrMat...) where T = Matrix{T}(Vcat{T}(A, B...))


# layout_broadcasted(lay, ::ApplyBandedLayout{typeof(vcat)}, op, A::AbstractVector, B::AbstractVector) = layout_broadcasted(lay, ApplyLayout{typeof(vcat)}(), op,A, B)
# layout_broadcasted(::ApplyBandedLayout{typeof(vcat)}, lay, op, A::AbstractVector, B::AbstractVector) = layout_broadcasted(ApplyLayout{typeof(vcat)}(), lay, op,A, B)

LazyArrays._vcat_sub_arguments(::ApplyBandedLayout{typeof(vcat)}, A, V) = LazyArrays._vcat_sub_arguments(ApplyLayout{typeof(vcat)}(), A, V)

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

function resizedata!(laydat::BlockBandedColumns{<:AbstractColumnMajor}, layarr, B::AbstractMatrix, n::Integer, m::Integer)
    ν,μ = B.datasize
    n ≤ ν && m ≤ μ || resizedata!(laydat, layarr, B, findblock.(axes(B), (n,m))...)
end

resizedata!(lay1, lay2, B::AbstractMatrix, N::Block{2}) = resizedata!(lay1, lay2, B, Block.(N.n)...)

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
        isempty(N_old+1:N) || isempty(M_old+1:M) || copyto!(view(B.data, N_old+1:N, M_old+1:M), B.array[N_old+1:N, M_old+1:M])
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

# Use memory laout for sub-blocks
@inline function Base.getindex(A::AbstractCachedMatrix, K::Block{1}, J::Block{1})
    @boundscheck checkbounds(A, K, J)
    resizedata!(A, K, J)
    A.data[K, J]
end
@inline Base.getindex(A::AbstractCachedMatrix, kr::Colon, jr::Block{1}) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline Base.getindex(A::AbstractCachedMatrix, kr::Block{1}, jr::Colon) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline Base.getindex(A::AbstractCachedMatrix, kr::Block{1}, jr::AbstractVector) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline Base.getindex(A::AbstractCachedArray{T,N}, kr::Block{1}, jrs...) where {T,N} = ArrayLayouts.layout_getindex(A, kr, jrs...)
@inline function Base.getindex(A::AbstractCachedArray{T,N}, block::Block{N}) where {T,N}
    @boundscheck checkbounds(A, block)
    resizedata!(A, block)
    A.data[block]
end

@inline Base.getindex(A::AbstractCachedMatrix, kr::AbstractVector, jr::Block) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline Base.getindex(A::AbstractCachedMatrix, kr::BlockRange{1}, jr::BlockRange{1}) = ArrayLayouts.layout_getindex(A, kr, jr)
include("bandedql.jl")
include("blockconcat.jl")
include("blockkron.jl")

###
# Concat and rot ArrayLayouts
###

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
                TriangularLayout{UPLO,UNIT,BandedColumns{LazyLayout}} where {UPLO,UNIT},
                SymTridiagonalLayout{LazyLayout}, BidiagonalLayout{LazyLayout}, TridiagonalLayout{LazyLayout},
                SymmetricLayout{BandedColumns{LazyLayout}}, HermitianLayout{BandedColumns{LazyLayout}}}

StructuredLazyLayouts = Union{BandedLazyLayouts,
                BlockBandedColumns{LazyLayout}, BandedBlockBandedColumns{LazyLayout},
                BlockBandedRows{LazyLayout},BandedBlockBandedRows{LazyLayout},
                BlockLayout{LazyLayout},
                BlockLayout{TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}}, BlockLayout{DiagonalLayout{LazyLayout}}, 
                BlockLayout{BidiagonalLayout{LazyLayout,LazyLayout}}, BlockLayout{SymTridiagonalLayout{LazyLayout,LazyLayout}},
                BlockLayout{LazyBandedLayout},
                AbstractLazyBlockBandedLayout, LazyBandedBlockBandedLayouts}


@inline _islazy(::StructuredLazyLayouts) = Val(true)

copy(M::Mul{<:StructuredLazyLayouts, <:StructuredLazyLayouts}) = simplify(M)
copy(M::Mul{<:StructuredLazyLayouts}) = simplify(M)
copy(M::Mul{<:Any, <:StructuredLazyLayouts}) = simplify(M)
copy(M::Mul{<:StructuredLazyLayouts, <:AbstractLazyLayout}) = simplify(M)
copy(M::Mul{<:AbstractLazyLayout, <:StructuredLazyLayouts}) = simplify(M)
copy(M::Mul{<:StructuredLazyLayouts, <:DiagonalLayout}) = simplify(M)
copy(M::Mul{<:DiagonalLayout, <:StructuredLazyLayouts}) = simplify(M)


copy(M::Mul{<:Union{ZerosLayout,DualLayout{ZerosLayout}}, <:StructuredLazyLayouts}) = copy(mulreduce(M))
copy(M::Mul{<:StructuredLazyLayouts, <:Union{ZerosLayout,DualLayout{ZerosLayout}}}) = copy(mulreduce(M))

simplifiable(::Mul{<:StructuredLazyLayouts, <:DiagonalLayout{<:OnesLayout}}) = Val(true)
simplifiable(::Mul{<:DiagonalLayout{<:OnesLayout}, <:StructuredLazyLayouts}) = Val(true)
copy(M::Mul{<:StructuredLazyLayouts, <:DiagonalLayout{<:OnesLayout}}) = _copy_oftype(M.A, eltype(M))
copy(M::Mul{<:DiagonalLayout{<:OnesLayout}, <:StructuredLazyLayouts}) = _copy_oftype(M.B, eltype(M))

copy(M::Mul{<:DiagonalLayout{<:AbstractFillLayout}, <:StructuredLazyLayouts}) = copy(mulreduce(M))
copy(M::Mul{<:StructuredLazyLayouts, <:DiagonalLayout{<:AbstractFillLayout}}) = copy(mulreduce(M))

copy(M::Mul{<:StructuredApplyLayouts{typeof(*)},<:StructuredApplyLayouts{typeof(*)}}) = simplify(M)
copy(M::Mul{<:StructuredApplyLayouts{typeof(*)},<:StructuredLazyLayouts}) = simplify(M)
copy(M::Mul{<:StructuredLazyLayouts,<:StructuredApplyLayouts{typeof(*)}}) = simplify(M)
copy(M::Mul{<:StructuredApplyLayouts{typeof(*)},<:BroadcastLayouts}) = simplify(M)
copy(M::Mul{<:BroadcastLayouts,<:StructuredApplyLayouts{typeof(*)}}) = simplify(M)
copy(M::Mul{BroadcastLayout{typeof(*)},<:StructuredApplyLayouts{typeof(*)}}) = simplify(M)
copy(M::Mul{ApplyLayout{typeof(*)},<:StructuredLazyLayouts}) = simplify(M)
copy(M::Mul{<:StructuredLazyLayouts,ApplyLayout{typeof(*)}}) = simplify(M)
copy(M::Mul{ApplyLayout{typeof(*)},<:BroadcastLayouts}) = simplify(M)
copy(M::Mul{<:BroadcastLayouts,ApplyLayout{typeof(*)}}) = simplify(M)

copy(M::Mul{<:AbstractInvLayout, <:StructuredApplyLayouts{typeof(*)}}) = simplify(M)
simplifiable(::Mul{<:AbstractInvLayout, <:StructuredLazyLayouts}) = Val(false)
copy(M::Mul{<:AbstractInvLayout, <:StructuredLazyLayouts}) = simplify(M)


copy(L::Ldiv{<:StructuredLazyLayouts, <:StructuredLazyLayouts}) = lazymaterialize(\, L.A, L.B)

# TODO: this is type piracy
function colsupport(lay::ApplyLayout{typeof(\)}, L, j)
    A,B = arguments(lay, L)
    l,u = bandwidths(A)
    cs = colsupport(B,j)
    m = size(L,1)
    l == u == 0 && return cs
    l == 0 && return 1:last(cs)
    u == 0 && return first(cs):m
    1:m
end

function rowsupport(lay::ApplyLayout{typeof(\)}, L, k)
    A,B = arguments(lay, L)
    l,u = bandwidths(A)
    cs = rowsupport(B,k)
    m = size(L,1)
    l == u == 0 && return cs
    l == 0 && return first(cs):m
    u == 0 && return 1:last(cs)
    1:m
end

copy(M::Mul{ApplyLayout{typeof(\)}, <:StructuredLazyLayouts}) = lazymaterialize(*, M.A, M.B)
copy(M::Mul{BroadcastLayout{typeof(*)}, <:StructuredLazyLayouts}) = lazymaterialize(*, M.A, M.B)

## padded copy
mulreduce(M::Mul{<:StructuredLazyLayouts, <:Union{PaddedLayout,AbstractStridedLayout}}) = MulAdd(M)
mulreduce(M::Mul{<:StructuredApplyLayouts{F}, D}) where {F,D<:Union{PaddedLayout,AbstractStridedLayout}} = Mul{ApplyLayout{F},D}(M.A, M.B)
# need to overload copy due to above
copy(M::Mul{<:StructuredLazyLayouts, <:Union{PaddedLayout,AbstractStridedLayout}}) = copy(mulreduce(M))
copy(M::Mul{<:AbstractInvLayout{<:BandedLazyLayouts}, <:Union{PaddedLayout,AbstractStridedLayout}}) = ArrayLayouts.ldiv(pinv(M.A), M.B)
copy(M::Mul{<:BandedLazyLayouts, <:Union{PaddedLayout,AbstractStridedLayout}}) = copy(mulreduce(M))
simplifiable(::Mul{<:StructuredLazyLayouts, <:Union{PaddedLayout,AbstractStridedLayout}}) = Val(true)


copy(L::Ldiv{ApplyBandedLayout{typeof(*)}, Lay}) where Lay = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
copy(L::Ldiv{ApplyBandedLayout{typeof(*)}, Lay}) where Lay<:StructuredLazyLayouts = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
_inv(::StructuredLazyLayouts, _, A) = ApplyArray(inv, A)

##
# support Inf Block ranges
broadcasted(::LazyArrayStyle{1}, ::Type{Block}, r::AbstractUnitRange) = Block(first(r)):Block(last(r))
broadcasted(::LazyArrayStyle{1}, ::Type{Int}, block_range::BlockRange{1}) = first(block_range.indices)
broadcasted(::LazyArrayStyle{0}, ::Type{Int}, block::Block{1}) = Int(block)


####
# Band getindex
####

function getindex(bc::BroadcastArray{<:Any,2,<:Any,<:NTuple{2,AbstractMatrix}}, b::Band)
    A,B = bc.args
    bc.f.(A[b],B[b])
end
function getindex(bc::BroadcastArray{<:Any,2,<:Any,<:Tuple{Number,AbstractMatrix}}, b::Band)
    a,B = bc.args
    bc.f.(a,B[b])
end
function getindex(bc::BroadcastArray{<:Any,2,<:Any,<:Tuple{AbstractMatrix,Number}}, b::Band)
    A,c = bc.args
    bc.f.(A[b],c)
end

# useful for turning Array into block array
unitblocks(a::AbstractArray) = PseudoBlockArray(a, Ones{Int}.(axes(a))...)
unitblocks(a::OneTo) = blockedrange(Ones{Int}(length(a)))
unitblocks(a::AbstractUnitRange) = BlockArrays._BlockedUnitRange(first(a),(first(a)-1) .+ BlockArrays._blocklengths2blocklasts(Ones{Int}(length(a))))

end
