module LazyBandedMatrices
using ArrayLayouts: symmetriclayout
using BandedMatrices, BlockBandedMatrices, BlockArrays, LazyArrays,
        ArrayLayouts, MatrixFactorizations, Base, StaticArrays

# for bidiag/tridiag
import Base: -, +, *, /, \, ==, AbstractMatrix, Matrix, Array, size, conj, real, imag, copy,
            iszero, isone, one, zero, getindex, setindex!, copyto!, fill, fill!, promote_rule, show, print_matrix, permutedims
import LinearAlgebra: transpose, adjoint, istriu, istril, isdiag, tril!, triu!, det, logabsdet,
                        symmetric, symmetric_type, diag, issymmetric, UniformScaling, char_uplo



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
