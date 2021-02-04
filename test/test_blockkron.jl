using LazyBandedMatrices, FillArrays, BandedMatrices, BlockBandedMatrices, BlockArrays, ArrayLayouts, LazyArrays, Test
import BlockBandedMatrices: isbandedblockbanded, BandedBlockBandedStyle, BandedLayout
import LazyBandedMatrices: KronTravBandedBlockBandedLayout, BroadcastBandedLayout, arguments, FillLayout, OnesLayout, call
import BandedMatrices: BandedColumns

@testset "Banded kron" begin
    A = brand(5,5,2,2)
    B = brand(2,2,1,0)
    K = kron(A,B)
    @test K isa BandedMatrix
    @test bandwidths(K) == (5,4)
    @test Matrix(K) == kron(Matrix(A), Matrix(B))

    A = brand(3,4,1,1)
    B = brand(3,2,1,0)
    K = kron(A,B)
    @test K isa BandedMatrix
    @test bandwidths(K) == (7,2)
    @test Matrix(K) ≈ kron(Matrix(A), Matrix(B))
    K = kron(B,A)
    @test Matrix(K) ≈ kron(Matrix(B), Matrix(A))

    K = kron(A, B')
    K isa BandedMatrix
    @test Matrix(K) ≈ kron(Matrix(A), Matrix(B'))
    K = kron(A', B)
    K isa BandedMatrix
    @test Matrix(K) ≈ kron(Matrix(A'), Matrix(B))
    K = kron(A', B')
    K isa BandedMatrix
    @test Matrix(K) ≈ kron(Matrix(A'), Matrix(B'))

    A = brand(5,6,2,2)
    B = brand(3,2,1,0)
    K = kron(A,B)
    @test K isa BandedMatrix
    @test bandwidths(K) == (12,4)
    @test Matrix(K) ≈ kron(Matrix(A), Matrix(B))

    n = 10; h = 1/n
    D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))
    D_xx = kron(D², Eye(n))
    @test D_xx isa BandedMatrix
    @test bandwidths(D_xx) == (10,10)
    D_yy = kron(Eye(n), D²)
    @test D_yy isa BandedMatrix
    @test bandwidths(D_yy) == (1,1)
    Δ = D_xx + D_yy
    @test Δ isa BandedMatrix
    @test bandwidths(Δ) == (10,10)

    @testset "#87" begin
        @test kron(Diagonal([1,2,3]), Eye(3)) isa Diagonal{Float64,Vector{Float64}}
    end
end

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
