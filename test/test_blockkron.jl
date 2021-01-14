using LazyBandedMatrices, FillArrays, BandedMatrices, BlockBandedMatrices, BlockArrays, ArrayLayouts, LazyArrays, Test
import BlockBandedMatrices: isbandedblockbanded, BandedBlockBandedStyle, BandedLayout
import LazyBandedMatrices: KronTravBandedBlockBandedLayout, BroadcastBandedLayout, arguments, FillLayout, OnesLayout, call
import BandedMatrices: BandedColumns

@testset "DiagTrav" begin
    A = [1 2 3; 4 5 6; 7 8 9]
    @test DiagTrav(A) == [1, 4, 2, 7, 5, 3]
    A = [1 2 3; 4 5 6]
    @test DiagTrav(A) == [1, 4, 2, 5, 3]
    A = [1 2; 3 4; 5 6]
    @test DiagTrav(A) == [1, 3, 2, 5, 4]

    A = DiagTrav(randn(3,3,3))
    @test A[Block(1)] == A[1:1,1,1]
    @test A[Block(2)] == [A.array[2,1,1], A.array[1,2,1], A.array[1,1,2]]
    @test A[Block(3)] == [A.array[3,1,1], A.array[2,2,1], A.array[1,3,1],
                          A.array[2,1,2], A.array[1,2,2], A.array[1,1,3]]
    @test A == [A[Block(1)]; A[Block(2)]; A[Block(3)]]
end

@testset "BlockKron" begin
    n = 4
    Δ = BandedMatrix(1 => Ones(n-1), 0 => Fill(-2,n), -1 => Ones(n-1))
    A = BlockKron(Δ, Eye(n))
    @test isblockbanded(A)
    @test isbandedblockbanded(A)
    @test blockbandwidths(A) == (1,1)
    @test BandedBlockBandedMatrix(A) == A
end

@testset "KronTrav" begin
    @testset "vector" begin
        a = [1,2,3]
        b = [4,5,6]
        c = [7,8]
        @test KronTrav(a,b) == DiagTrav(b*a')
        @test KronTrav(a,c) == DiagTrav(c*a')
        @test KronTrav(c,a) == DiagTrav(a*c')
    end

    @testset "matrix" begin
        A = [1 2; 3 4]
        B = [5 6; 7 8]
        K = KronTrav(A,B)

        X = [9 10; 11 0]
        @test K*DiagTrav(X) == DiagTrav(B*X*A')

    end

    @testset "banded" begin
        n = 4
        Δ = BandedMatrix(1 => Ones(n-1), 0 => Fill(-2,n), -1 => Ones(n-1))
        A = KronTrav(Δ, Eye(n))
        B = KronTrav(Eye(n), Δ)

        X = triu!(randn(n,n))[:,end:-1:1]
        @test A * DiagTrav(X) == DiagTrav(X * Δ')
        @test B * DiagTrav(X) == DiagTrav(Δ * X)

        @test blockbandwidths(A) == blockbandwidths(B) == (1,1)
        @test subblockbandwidths(A) == (1,1)
        @test subblockbandwidths(B) == (0,0)
        @test isblockbanded(A)
        @test isbandedblockbanded(A)
        @test BandedBlockBandedMatrix(A) == A
        @test MemoryLayout(A) isa KronTravBandedBlockBandedLayout


        @test Base.BroadcastStyle(typeof(A)) isa BandedBlockBandedStyle
        @test A+B isa BandedBlockBandedMatrix
        @test A+B == Matrix(A) + Matrix(B)
    end

    @testset "Views" begin
        n = 4
        Δ = BandedMatrix(1 => Ones(n-1), 0 => Fill(-2,n), -1 => Ones(n-1))
        A = KronTrav(Δ, Eye(n))

        V = view(A, Block(2,2))
        @test MemoryLayout(V) isa BroadcastBandedLayout{typeof(*)}
        @test call(V) == *
        @test MemoryLayout.(arguments(V)) isa Tuple{BandedColumns{DenseColumnMajor},BandedColumns{OnesLayout}}
        @test BandedMatrix(V) == V == A[Block(2,2)]

        V = view(A, Block.(2:4), Block.(1:4))
        @test blockbandwidths(V) == (0,2)
        @test subblockbandwidths(V) == (1,1)
        @test BandedBlockBandedMatrix(V) == A[Block.(2:4), Block.(1:4)]
        @test A[Block.(2:4), Block.(1:4)] isa BandedBlockBandedMatrix
    end
end
