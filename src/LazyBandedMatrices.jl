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


include("blockconcat.jl")
include("blockkron.jl")

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
copy(M::Mul{<:Union{PaddedLayout,AbstractStridedLayout}, <:BandedLazyLayouts}) = copy(mulreduce(M))
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
