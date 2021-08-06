using LazyBandedMatrices, BlockBandedMatrices, BlockArrays, StaticArrays, FillArrays, LazyArrays, ArrayLayouts, BandedMatrices, Test
import LazyBandedMatrices: BlockBroadcastArray, ApplyLayout
import LinearAlgebra: Adjoint ,Transpose

@testset "unitblocks" begin
    a = unitblocks(Base.OneTo(5))
    @test a == 1:5
    @test blockaxes(a,1) == Block.(1:5)

    a = unitblocks(2:5)
    @test a == 2:5
    @test blockaxes(a,1) == Block.(1:4)
end

@testset "BlockVcat" begin
    @testset "vec vcat" begin
        a = BlockVcat(1:5, 10:12, 14:15)
        @test @inferred(axes(a)) ≡ (blockedrange(SVector(5,3,2)),)
        @test @inferred(a[Block(1)]) ≡ 1:5
        @test a == [1:5; 10:12; 14:15]
        @test a[Block.(1:2)] ≡ BlockVcat(1:5, 10:12)
        @test a[:] == a[1:size(a,1)] == a
        @test a[1:10] isa Vcat
        @test a[Block(1)[1:2]] ≡ 1:2
        @test a[3] ≡ 3

        @test copy(a) ≡ convert(AbstractArray{Int},a) ≡ convert(AbstractVector{Int},a) ≡ a
        @test AbstractArray{Float64}(a) == AbstractVector{Float64}(a) == convert(AbstractArray{Float64},a) == convert(AbstractVector{Float64},a) == a
        @test copy(a') ≡ a'
    end
    @testset "mat vcat" begin
        A = BlockVcat(randn(2,3), randn(3,3))
        @test axes(A,2) ≡ Base.OneTo(3)
        @test A[Block(1,1)] == A.arrays[1]
        @test A[Block.(1:2),Block(1)] == A
        @test A[Block.(1:2),Block(1)] isa typeof(A)

        @test A[Block.(1:2), Block.(1:1)] == A
        @test A[Block.(1:2), Block.(1:1)] isa BlockVcat
        @test A[Block.(1:2), 1:2] == A[:,1:2]
        @test A[Block.(1:2), 1:2] isa BlockVcat

        @test convert(AbstractArray{Float64},A) == convert(AbstractMatrix{Float64},A) == A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractMatrix{Float64}(A) == A
        @test copy(A') == A'
    end
    @testset "block vec vcat" begin
        a = PseudoBlockArray(1:5, SVector(1,3))
        b = PseudoBlockArray(2:6, SVector(1,3))

        c = BlockVcat(a,b)
        @test c == [a; b]
        @test c[Block(2)] == a[Block(2)]
        @test c[Block(3)] == b[Block(1)]
        @test c[Block.(2:3)] == [a[Block(2)]; b[Block(1)]]
        @test c[Block.(2:3),Block(1)] == reshape([a[Block(2)]; b[Block(1)]], 4, 1)
        @test copy(c) ≡ convert(AbstractArray{Int},c) ≡ convert(AbstractVector{Int},c) ≡ c
        @test AbstractArray{Float64}(c) == AbstractVector{Float64}(c) == c
        @test copy(c') ≡ c'

        A = BlockVcat(a', b')
        @test axes(A,1) ≡ blockedrange(SVector(1,1))
        @test axes(A,2) ≡ axes(a,1)
        @test A[Block(1,1)] == a[Block(1)]'
        @test A[Block(2,2)] == b[Block(2)]'
        @test A == [a'; b']
        @test convert(AbstractArray{Int},A) ≡ convert(AbstractMatrix{Int},A) ≡ A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractMatrix{Float64}(A) == A
        @test copy(A') == A'
    end

    @testset "block mat vcat" begin
        A = PseudoBlockArray(randn(3,2), [1,2], [1,1])
        B = PseudoBlockArray(randn(4,2), [3,1], [1,1])
        V = BlockVcat(A, B)
        @test V == [A; B]
        @test V[Block(3,1)] == B[Block(1,1)]
        @test convert(AbstractArray{Float64},V) ≡ convert(AbstractMatrix{Float64},V) ≡ V
        @test copy(V) == AbstractArray{Float64}(V) == AbstractMatrix{Float64}(V) == V
        @test copy(A') == A'

        @test V[Block.(2:3), Block.(1:2)] == [A[Block(2),Block.(1:2)]; B[Block(1),Block.(1:2)]]
    end

    @testset "triangle recurrence" begin
        N = 1000
        a = b = c = 0.0
        n = mortar(Fill.(Base.OneTo(N),Base.OneTo(N)))
        k = mortar(Base.OneTo.(Base.OneTo(N)))
        dat = BlockVcat(
            ((k .+ (c-1)) .* ( k .- n .- 1 ) ./ (2k .+ (b+c-1)))',
            (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))'
            )
        @test dat[:,Block.(1:3)] isa BlockVcat
    end

    @testset "PseudoBlockArray" begin
        a = PseudoBlockArray(Vcat(randn(3), 1:3), [3,3])
        @test MemoryLayout(a) isa LazyArrays.ApplyLayout{typeof(vcat)}
        @test LazyArrays.arguments(a) == LazyArrays.arguments(a.blocks)

        b = PseudoBlockArray(Vcat(randn(3), Zeros(3)), [3,3])
        @test LazyArrays.paddeddata(view(b, 1:4)) == LazyArrays.paddeddata(view(b, Base.OneTo(4))) == b[1:3]
    end
end


@testset "BlockHcat" begin
    @testset "vec hcat" begin
        a = BlockHcat(1:5, 10:14)
        @test axes(a,2) ≡ blockedrange(Ones{Int}(2))
        @test a[Block(1,1)] ≡ 1:5
        @test a == [1:5 10:14]
        @test_broken a[:,Block.(1:2)] ≡ BlockHcat(1:5, 10:14)
        @test a[:] == a[1:length(a)] == vec(a)
        @test copy(a) ≡ convert(AbstractArray{Int},a) ≡ convert(AbstractMatrix{Int},a) ≡ a
        @test AbstractArray{Float64}(a) == AbstractMatrix{Float64}(a) == convert(AbstractArray{Float64},a) == convert(AbstractMatrix{Float64},a) == a
        @test copy(a') == a'
    end

    @testset "mat hcat" begin
        A = BlockHcat(randn(3,2), randn(3,3))
        @test axes(A,1) ≡ Base.OneTo(3)
        @test A[Block(1,1)] == A.arrays[1]
        @test A[Block(1),Block.(1:2)] == A
        @test A[Block(1),Block.(1:2)] isa BlockHcat
        @test convert(AbstractArray{Float64},A) ≡ convert(AbstractMatrix{Float64},A) ≡ A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractMatrix{Float64}(A) == A
        @test copy(A') == copy(Adjoint(A)) == A'
    end

    @testset "block vec hcat" begin
        a = PseudoBlockArray(1:4, SVector(1,3))
        b = PseudoBlockArray(2:5, SVector(1,3))
        A = BlockHcat(a, b)
        @test axes(A,2) ≡ blockedrange(Ones{Int}(2))
        @test axes(A,1) ≡ axes(a,1)
        @test A[Block(1,1)] == a[Block(1)]
        @test A[Block(2,2)] == b[Block(2)]
        @test A[Block(2), Block.(1:2)] == [a[Block(2)] b[Block(2)]]
        @test A[Block.(1:2),Block.(2)] == reshape(b,4,1)
        @test A[Block.(1:2), Block.(1:1)] == reshape(a,4,1)

        @test A == [a b]
        @test convert(AbstractArray{Int},A) ≡ convert(AbstractMatrix{Int},A) ≡ A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractMatrix{Float64}(A) == A
        @test copy(A') == A'
    end

    @testset "block mat hcat" begin
        A = PseudoBlockArray(randn(3,2), [1,2], [1,1])
        B = PseudoBlockArray(randn(3,3), [1,2], [2,1])
        H = BlockHcat(A, B)
        @test H == [A B]
        @test H[Block(1,3)] == B[Block(1,1)]

        @test H[Block(2),Block.(2:3)] == H[Block.(2:2),Block.(2:3)]  == [A[Block(2,2)] B[Block(2,1)]]
        @test H[Block.(2:2),Block(3)] == H[Block(2,3)]
        @test convert(AbstractArray{Float64},H) ≡ convert(AbstractMatrix{Float64},H) ≡ H
        @test copy(H) == AbstractArray{Float64}(H) == AbstractMatrix{Float64}(H) == H
        @test copy(H') == H'
    end

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
        @test (A')[Block(2,3)] == A[Block(3,2)]'
        @test copyto!(dest, A') == A'
        @test @allocated(copyto!(dest, A')) ≤ 2600


        Rx = BlockBandedMatrices._BandedBlockBandedMatrix(A', axes(k,1), (0,1), (0,0))
        dest = BandedBlockBandedMatrix{Float64}(undef, axes(Rx), (0,1), (0,0))
        @test copyto!(dest, Rx) == BandedBlockBandedMatrix(Rx)

        Vx = view(Rx, Block.(1:N), Block.(1:N))
        @test MemoryLayout(Vx) isa BlockBandedMatrices.BandedBlockBandedColumns

        V = view(A, Block.(1:5),:)
        @test MemoryLayout(V) isa ApplyLayout{typeof(hcat)}
        HV = A[Block.(1:5),:]
        @test HV == V
        # TODO: Fast materialization
    end

    # @testset "Block-mat hcat" begin
    #     A = BlockHcat(PseudoBlockArray(randn(6,4), [4,2], [2,2]), PseudoBlockArray(randn(6,3), [4,2], [2,1]))
    #     A[:,Block.(1:2)]
    # end

    @testset "adjtrans" begin
        A = BlockHcat(randn(3,2), randn(3,3))
        @test A' == transpose(A) == BlockArray(A)'
        @test A' isa BlockVcat
        @test transpose(A) isa BlockVcat
        @test A'' == A
        @test A'' isa BlockHcat
    end
end

@testset "BlockHvcat" begin
    A = randn(2,2)
    B = randn(2,3)
    C = randn(3,2)
    D = randn(3,3)

    H = BlockHvcat(2, A, B, C, D)
    @test H == [A B; C D]

    @test convert(AbstractArray{Float64},H) ≡ convert(AbstractMatrix{Float64},H) ≡ H
    @test copy(H) == AbstractArray{Float64}(H) == AbstractMatrix{Float64}(H) == H
    @test copy(H') == H'
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
        @test A[Block(1)] == PseudoBlockArray(A)[Block(1)] == [A[1],A[2]] == [1,11]
        @test A[Block(N)] == PseudoBlockArray(A)[Block(N)] == [1000,1010]
        @test convert(AbstractArray{Int},A) ≡ convert(AbstractVector{Int},A) ≡ A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractVector{Float64}(A) == convert(AbstractArray{Float64},A) == convert(AbstractVector{Float64},A) == A
        @test copy(A') == A'
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
        @test convert(AbstractArray{Int},A) ≡ convert(AbstractMatrix{Int},A) ≡ A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractMatrix{Float64}(A) == A
        @test copy(A') == A'
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
        @test A[1,1] == a[1,1]

        @test convert(AbstractArray{Float64},A) ≡ convert(AbstractMatrix{Float64},A) ≡ A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractMatrix{Float64}(A) == A
        @test convert(AbstractArray{BigFloat}, A) == convert(AbstractMatrix{BigFloat}, A) == A
        @test copy(A') == A'

        @testset "Banded" begin
            a = unitblocks(brand(5,4,1,2))
            z = Zeros(axes(a))
            @test blockbandwidths(a) == (1,2)
            @test subblockbandwidths(a) == (0,0)
            A = BlockBroadcastArray(hvcat, 2, a, z, z, a)
            @test blockbandwidths(A) == (1,2)
            @test subblockbandwidths(A) == (0,0)
            @test A[Block.(1:2),Block.(1:2)] isa BlockSkylineMatrix
        end
    end

    @testset "Diagonal" begin
        A = unitblocks(brand(5,5,1,2))
        B = unitblocks(brand(5,5,2,1))
        C = BlockBroadcastArray{Float64}(Diagonal, A, B)
        @test blockisequal(axes(C),(blockedrange(Fill(2,5)),blockedrange(Fill(2,5))))
        for k = 1:5, j = 1:5
            @test C[Block(k,j)] == Diagonal([A[k,j],B[k,j]])
        end
        @test blockbandwidths(C) == (2,2)
        @test subblockbandwidths(C) == (0,0)
    end
end
