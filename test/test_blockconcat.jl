using LazyBandedMatrices, BlockArrays, StaticArrays, FillArrays, LazyArrays, Test
import LazyBandedMatrices: BlockInterlace

@testset "BlockVcat" begin
    a = BlockVcat(1:5, 10:12, 14:15)
    @test axes(a,1) ≡ blockedrange(SVector(5,3))
    @test a[Block(1)] ≡ 1:5
    @test a == [1:5; 10:12; 14:15]
    @test a[Block.(1:2)] ≡ BlockVcat(1:5, 10:12)
    @test a[:] == a[1:size(a,1)] == a
    @test a[1:10] isa Vcat

    A = BlockVcat(randn(2,3), randn(3,3))
    @test axes(A,2) ≡ Base.OneTo(3)
    @test A[Block(1,1)] == A.arrays[1]
    @test A[Block.(1:2),Block(1)] == A
    @test A[Block.(1:2),Block(1)] isa typeof(A)

    a = PseudoBlockArray(1:5, SVector(1,3))
    b = PseudoBlockArray(2:6, SVector(1,3))
    A = BlockVcat(a', b')
    @test axes(A,1) ≡ blockedrange(SVector(1,1))
    @test axes(A,2) ≡ axes(a,1)
    @test A[Block(1,1)] == a[Block(1)]'
    @test A[Block(2,2)] == b[Block(2)]'
    @test A == [a'; b']

    @testset "triangle recurrence" begin
        N = 10_000
        a = b = c = 0.0
        n = mortar(Fill.(Base.OneTo(N),Base.OneTo(N)))
        k = mortar(Base.OneTo.(Base.OneTo(N)))
        dat = BlockVcat(
            ((k .+ (c-1)) .* ( k .- n .- 1 ) ./ (2k .+ (b+c-1)))',
            (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))'
            )
    end
end


@testset "BlockHcat" begin
    a = BlockHcat(1:5, 10:14)
    @test axes(a,2) ≡ blockedrange(SVector(1,1))
    @test a[Block(1,1)] ≡ 1:5
    @test a == [1:5 10:14]
    # @test a[:,Block.(1:2)] ≡ BlockHcat(1:5, 10:14)
    # @test a[:] == a[1:size(a,1)] == a
    @test a[1:5,1:2] isa Vcat

    A = BlockHcat(randn(3,2), randn(3,3))
    @test axes(A,1) ≡ Base.OneTo(3)
    @test A[Block(1,1)] == A.arrays[1]
    @test A[Block(1),Block.(1:2)] == A

    a = PseudoBlockArray(1:5, SVector(1,3))
    b = PseudoBlockArray(2:6, SVector(1,3))
    A = BlockHcat(a, b)
    @test axes(A,2) ≡ blockedrange(SVector(1,1))
    @test axes(A,1) ≡ axes(a,1)
    @test A[Block(1,1)] == a[Block(1)]
    @test A[Block(2,2)] == b[Block(2)]
    @test A == [a b]

    @testset "triangle recurrence" begin
        N = 1_000
        a = b = c = 0.0
        n = mortar(BroadcastArray(Fill,Base.OneTo(N),Base.OneTo(N)))
        k = mortar(BroadcastArray(Base.OneTo,Base.OneTo(N)))
        A = BlockHcat(
            BroadcastArray((n,k,a,b,c) -> (k + c - 1)*(k-n-1) / (2k+b+c-1), n, k, a, b, c),
            BroadcastArray((n,k,a,b,c) -> k*(k-n-a) / (2k+b+c-1), n, k, a, b, c))
        dest = PseudoBlockArray{Float64}(undef, axes(A))
        @test copyto!(dest, A) == A;
        @test @allocated(copyto!(dest, A)) ≤ 1500
    end
end


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