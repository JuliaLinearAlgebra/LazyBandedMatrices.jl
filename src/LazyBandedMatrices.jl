module LazyBandedMatrices
using ArrayLayouts: symmetriclayout
using BandedMatrices, BlockBandedMatrices, BlockArrays, LazyArrays,
        ArrayLayouts, MatrixFactorizations, Base, StaticArrays, LinearAlgebra

# for bidiag/tridiag
import Base: -, +, *, /, \, ==, AbstractMatrix, Matrix, Array, size, conj, real, imag, copy,
            iszero, isone, one, zero, getindex, setindex!, copyto!, fill, fill!, promote_rule, show, print_matrix, permutedims,
            OneTo
import Base.Broadcast: Broadcasted
import LinearAlgebra: transpose, adjoint, istriu, istril, isdiag, tril!, triu!, det, logabsdet,
                        symmetric, symmetric_type, diag, issymmetric, UniformScaling, char_uplo,
                        AbstractTriangular, AdjOrTrans, StructuredMatrixStyle
import LazyArrays: ApplyLayout, AbstractPaddedLayout, BroadcastLayout, LazyArrayStyle
import BandedMatrices: AbstractBandedMatrix, BandedStyle
import BlockBandedMatrices: AbstractBlockBandedLayout, AbstractBandedBlockBandedLayout, BlockRange1, Block1
import BlockArrays: BlockSlices, BlockSlice1, BlockSlice, blockvec, AbstractBlockLayout

LazyArraysBandedMatricesExt = Base.get_extension(LazyArrays, :LazyArraysBandedMatricesExt)
LazyArraysBlockBandedMatricesExt = Base.get_extension(LazyArrays, :LazyArraysBlockBandedMatricesExt)

BroadcastBandedLayout = LazyArraysBandedMatricesExt.BroadcastBandedLayout
AbstractLazyBlockBandedLayout = LazyArraysBlockBandedMatricesExt.AbstractLazyBlockBandedLayout


export DiagTrav, KronTrav, blockkron, BlockKron, BlockBroadcastArray, BlockVcat, BlockHcat, BlockHvcat, unitblocks

include("tridiag.jl")
include("bidiag.jl")
include("special.jl")

# useful for turning Array into block array
unitblocks(a::AbstractArray) = PseudoBlockArray(a, Ones{Int}.(axes(a))...)
unitblocks(a::OneTo) = blockedrange(Ones{Int}(length(a)))
unitblocks(a::AbstractUnitRange) = BlockArrays._BlockedUnitRange(first(a),(first(a)-1) .+ BlockArrays._blocklengths2blocklasts(Ones{Int}(length(a))))



include("blockconcat.jl")
include("blockkron.jl")

end
