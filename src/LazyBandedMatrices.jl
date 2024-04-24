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
                    symmetriclayout, hermitianlayout, _copy_oftype
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
                        BandedStyle, BandedColumns, BandedRows, BandedLayout,
                        AbstractBandedMatrix, BandedSubBandedMatrix, BandedStyle,
                        _BandedMatrix, bandeddata,
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

end
