module LazyBandedMatrices
using ArrayLayouts: symmetriclayout
using BandedMatrices, BlockBandedMatrices, BlockArrays, LazyArrays,
        ArrayLayouts, MatrixFactorizations, Base, StaticArrays, LinearAlgebra

import Base: -, +, *, /, \, ==, AbstractMatrix, Matrix, Array, size, conj, real, imag, copy,
            iszero, isone, one, zero, getindex, setindex!, copyto!, fill, fill!, promote_rule, show, print_matrix, permutedims,
            OneTo, oneto, require_one_based_indexing, similar, convert, axes, tail, tuple_type_tail, view, resize!
import Base.Broadcast: Broadcasted, BroadcastStyle, broadcasted, instantiate
import LinearAlgebra: transpose, adjoint, istriu, istril, isdiag, tril!, triu!, det, logabsdet,
                        symmetric, symmetric_type, diag, issymmetric, UniformScaling, char_uplo,
                        AbstractTriangular, AdjOrTrans, StructuredMatrixStyle

import ArrayLayouts: MemoryLayout, bidiagonallayout, bidiagonaluplo, diagonaldata, supdiagonaldata, subdiagonaldata,
                     symtridiagonallayout, tridiagonallayout, symmetriclayout,
                     colsupport, rowsupport, sublayout, sub_materialize
import LazyArrays: ApplyLayout, AbstractPaddedLayout, PaddedLayout, PaddedColumns, BroadcastLayout, LazyArrayStyle, LazyLayout,
                   arguments, call, tuple_type_memorylayouts, paddeddata, _broadcast_sub_arguments, resizedata!,
                   _cumsum, convexunion, applylayout
import BandedMatrices: AbstractBandedMatrix, BandedStyle, bandwidths, isbanded
import BlockBandedMatrices: AbstractBlockBandedLayout, AbstractBandedBlockBandedLayout, BlockRange1, Block1, blockbandwidths, subblockbandwidths,
                             BlockBandedStyle, BandedBlockBandedStyle, isblockbanded, isbandedblockbanded
import BlockArrays: BlockSlices, BlockSlice1, BlockSlice, blockvec, AbstractBlockLayout, blockcolsupport, blockrowsupport, BlockLayout, block, blockindex, viewblock, AbstractBlockedUnitRange

LazyArraysBandedMatricesExt = Base.get_extension(LazyArrays, :LazyArraysBandedMatricesExt)
LazyArraysBlockBandedMatricesExt = Base.get_extension(LazyArrays, :LazyArraysBlockBandedMatricesExt)

BroadcastBandedLayout = LazyArraysBandedMatricesExt.BroadcastBandedLayout
AbstractLazyBandedLayout = LazyArraysBandedMatricesExt.AbstractLazyBandedLayout
ApplyBandedLayout = LazyArraysBandedMatricesExt.ApplyBandedLayout
LazyBandedLayout = LazyArraysBandedMatricesExt.LazyBandedLayout
AbstractLazyBlockBandedLayout = LazyArraysBlockBandedMatricesExt.AbstractLazyBlockBandedLayout
BroadcastBandedBlockBandedLayout = LazyArraysBlockBandedMatricesExt.BroadcastBandedBlockBandedLayout
ApplyBlockBandedLayout = LazyArraysBlockBandedMatricesExt.ApplyBlockBandedLayout
LazyBandedBlockBandedLayout = LazyArraysBlockBandedMatricesExt.LazyBandedBlockBandedLayout
AbstractLazyBandedBlockBandedLayout = LazyArraysBlockBandedMatricesExt.AbstractLazyBandedBlockBandedLayout




export DiagTrav, KronTrav, blockkron, BlockKron, BlockBroadcastArray, BlockVcat, BlockHcat, BlockHvcat, unitblocks

include("tridiag.jl")
include("bidiag.jl")
include("special.jl")

# useful for turning Array into block array
unitblocks(a::AbstractArray) = PseudoBlockArray(a, Ones{Int}.(axes(a))...)
unitblocks(a::OneTo) = blockedrange(Ones{Int}(length(a)))
unitblocks(a::AbstractUnitRange) = BlockArrays._BlockedUnitRange(first(a), (first(a)-1) .+ BlockArrays._blocklengths2blocklasts(Ones{Int}(length(a))))



include("blockconcat.jl")
include("blockkron.jl")

end
