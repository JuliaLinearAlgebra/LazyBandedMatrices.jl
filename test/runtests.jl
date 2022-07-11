using LazyBandedMatrices, BlockBandedMatrices, BandedMatrices, LazyArrays, BlockArrays,
            ArrayLayouts, MatrixFactorizations, Random, Test
import LinearAlgebra
import LinearAlgebra: qr, rmul!, lmul!
import LazyArrays: Applied, resizedata!, FillLayout, MulStyle, arguments, colsupport, rowsupport, LazyLayout, ApplyStyle, 
                    PaddedLayout, paddeddata, call, ApplyLayout, LazyArrayStyle, simplifiable
import LazyBandedMatrices: VcatBandedMatrix, BroadcastBlockBandedLayout, BroadcastBandedLayout, 
                    ApplyBandedLayout, ApplyBlockBandedLayout, ApplyBandedBlockBandedLayout, BlockKron, LazyBandedLayout, BroadcastBandedBlockBandedLayout
import BandedMatrices: BandedStyle, _BandedMatrix, AbstractBandedMatrix, BandedRows, BandedColumns
import ArrayLayouts: StridedLayout, OnesLayout
import BlockArrays: blockcolsupport, blockrowsupport

Random.seed!(0)

struct MyLazyArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end


Base.size(A::MyLazyArray) = size(A.data)
Base.getindex(A::MyLazyArray, j::Int...) = A.data[j...]
LazyArrays.MemoryLayout(::Type{<:MyLazyArray}) = LazyLayout()
Base.BroadcastStyle(::Type{<:MyLazyArray{<:Any,N}}) where N = LazyArrayStyle{N}()
LinearAlgebra.factorize(A::MyLazyArray) = factorize(A.data)

include("test_tridiag.jl")
include("test_bidiag.jl")
include("test_special.jl")
include("test_banded.jl")
include("test_block.jl")




@testset "Misc" begin

    @testset "Diagonal interface" begin
        n = 10
        h = 1/n
        D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2
        D_xx = BandedBlockBandedMatrix(BlockKron(D², Eye(n)))

        D = Diagonal(randn(n^2))
        @test D_xx + D isa BandedBlockBandedMatrix
        @test blockbandwidths(D_xx + D) == blockbandwidths(D_xx)
        @test subblockbandwidths(D_xx + D) == subblockbandwidths(D_xx)
        @test D_xx + D == Matrix(D_xx) + D

        @test D_xx - D isa BandedBlockBandedMatrix
        @test blockbandwidths(D_xx - D) == blockbandwidths(D_xx)
        @test subblockbandwidths(D_xx - D) == subblockbandwidths(D_xx)
        @test D_xx - D == Matrix(D_xx) - D

        @test D_xx*D == Matrix(D_xx)*D
        @test D_xx*D isa BandedBlockBandedMatrix
        @test blockbandwidths(D_xx*D) == blockbandwidths(D_xx)
        @test subblockbandwidths(D_xx*D) == subblockbandwidths(D_xx)

        @test D*D_xx == D*Matrix(D_xx)
        @test D*D_xx isa BandedBlockBandedMatrix
        @test blockbandwidths(D*D_xx) == blockbandwidths(D_xx)
        @test subblockbandwidths(D*D_xx) == subblockbandwidths(D_xx)
    end

    # @testset "Padded columns" begin
    #     B = brand(8,8,1,2)
    #     v = view(B,:,4)
    #     w = view(B,3,:)
    #     @test MemoryLayout(v) isa PaddedLayout
    #     @test_broken MemoryLayout(w) isa PaddedLayout
    #     @test paddeddata(v) isa Vcat
    #     paddeddata(v) == B[:,4]
    # end
end



include("test_blockkron.jl")
include("test_blockconcat.jl")
