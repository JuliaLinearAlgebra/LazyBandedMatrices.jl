module TestMisc

using FillArrays
using BlockBandedMatrices
using BandedMatrices
using BlockArrays
using Test
using LazyArrays
import LazyArrays: PaddedLayout, paddeddata, call
using LinearAlgebra
using LazyBandedMatrices

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

    @testset "Block broadcast" begin
        a = PseudoBlockArray(BroadcastArray(exp, randn(5)), [3,2])
        @test call(a) == exp
    end

    @testset "unitblocks" begin
        @test blockbandwidths(unitblocks(Diagonal(1:5))) == (0,0)
    end
end

end # module
