using LazyBandedMatrices, BlockBandedMatrices, BlockArrays, StaticArrays, FillArrays, LazyArrays, ArrayLayouts, Test
import LazyBandedMatrices: BlockBroadcastArray

@testset "BlockVcat" begin
    a = BlockVcat(1:5, 10:12, 14:15)
    @test axes(a,1) ≡ blockedrange(SVector(5,3,2))
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
        N = 1000
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
    @test axes(a,2) ≡ blockedrange(Ones{Int}(2))
    @test a[Block(1,1)] ≡ 1:5
    @test a == [1:5 10:14]
    @test_broken a[:,Block.(1:2)] ≡ BlockHcat(1:5, 10:14)
    @test a[:] == a[1:length(a)] == vec(a)

    A = BlockHcat(randn(3,2), randn(3,3))
    @test axes(A,1) ≡ Base.OneTo(3)
    @test A[Block(1,1)] == A.arrays[1]
    @test A[Block(1),Block.(1:2)] == A

    a = PseudoBlockArray(1:5, SVector(1,3))
    b = PseudoBlockArray(2:6, SVector(1,3))
    A = BlockHcat(a, b)
    @test axes(A,2) ≡ blockedrange(Ones{Int}(2))
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
            BroadcastVector((n,k,bc1,abc) -> (n + k +  bc1) / (2n + abc), n, k, b+c-1, a+b+c),
            BroadcastVector((n,k,abc) -> (n + k +  abc) / (2n + abc), n, k, a+b+c)
            )
        dest = PseudoBlockArray{Float64}(undef, axes(A))
        @test copyto!(dest, A) == A;
        @test @allocated(copyto!(dest, A)) ≤ 2800
        # dest = BlockArray{Float64}(undef, axes(A))
        # @time copyto!(dest, A);

        dest = PseudoBlockArray{Float64}(undef, axes(A'))
        @test copyto!(dest, A') == A'
        @test @allocated(copyto!(dest, A')) ≤ 2200
        

        Rx = BlockBandedMatrices._BandedBlockBandedMatrix(A', axes(k,1), (0,1), (0,0))
        dest = BandedBlockBandedMatrix{Float64}(undef, axes(Rx), (0,1), (0,0))
        @test copyto!(dest, Rx) == BandedBlockBandedMatrix(Rx)

        Vx = view(Rx, Block.(1:N), Block.(1:N))
        @test MemoryLayout(Vx) isa BlockBandedMatrices.BandedBlockBandedColumns
        # TODO: Fast materialization
    end
end


@testset "Interlace" begin
    @testset "vcat" begin
        N = 1000
        a = 1:N
        b = 11:10+N
        a, b = PseudoBlockArray(a,Ones{Int}(length(a))), PseudoBlockArray(b,Ones{Int}(length(b)))
        A = BlockBroadcastArray(vcat, a, b)
        @test axes(A,1) isa BlockedUnitRange{StepRange{Int,Int}}
        @test @allocated(axes(A)) ≤ 50
        @test A[Block(1)] == PseudoBlockArray(A)[Block(1)] == [1,11]
        @test A[Block(N)] == PseudoBlockArray(A)[Block(N)] == [1000,1010]
    end
    @testset "hcat" begin
        N = 1000
        a = 1:N
        b = 11:10+N
        a, b = PseudoBlockArray(a,Ones{Int}(length(a))), PseudoBlockArray(b,Ones{Int}(length(b)))
        A = BlockBroadcastArray(hcat, a', b')
        @test axes(A,2) isa BlockedUnitRange{StepRange{Int,Int}}
        @test @allocated(axes(A)) ≤ 50
        @test A[Block(1,1)] == PseudoBlockArray(A)[Block(1,1)] == [1 11]
        @test A[Block(1,N)] == PseudoBlockArray(A)[Block(1,N)] == [1000 1010]
    end
    @testset  "hvcat" begin
        a = unitblocks(randn(2,3))
        b = unitblocks(randn(2,3))
        c = unitblocks(randn(2,3))
        d = unitblocks(randn(2,3))
        e = unitblocks(randn(2,3))
        f = unitblocks(randn(2,3))

        A = BlockBroadcastArray(hvcat, 2, a, b, c, d, e, f)
        @test MemoryLayout(A) isa UnknownLayout
        @test blocksize(A) == (2,3)
        @test A[Block(1,1)] == [a[1] b[1]; c[1] d[1]; e[1] f[1]]
    end
end