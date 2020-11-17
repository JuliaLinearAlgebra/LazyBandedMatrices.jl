using LazyBandedMatrices, BlockArrays, Test
import LazyBandedMatrices: BlockInterlace

@testset "BlockInterlace" begin
    @testset "vector" begin
        N = 1000
        a = 1:N
        b = 11:10+N
        A = BlockInterlace(a, b)
        @test A[Block(1)] == PseudoBlockArray(A)[Block(1)] == [1,11]
        @test A[Block(N)] == PseudoBlockArray(A)[Block(N)] == [1000,1010]
    end
    @testset  "matrix" begin
        a = randn(2,3)
        b = randn(2,3)
        c = randn(2,3)
        d = randn(2,3)
        e = randn(2,3)
        f = randn(2,3)

        A = BlockInterlace(2, a, b, c, d, e, f)
        @test blocksize(A) == (2,3)
        @test A[Block(1,1)] == [a[1] b[1]; c[1] d[1]; e[1] f[1]]
    end
end