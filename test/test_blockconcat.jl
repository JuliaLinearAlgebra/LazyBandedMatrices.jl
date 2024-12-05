module TestBlockConcat

using LazyBandedMatrices, BlockBandedMatrices, BlockArrays, StaticArrays, FillArrays, LazyArrays, ArrayLayouts, BandedMatrices, Test
import LazyBandedMatrices: BlockBroadcastArray, blockcolsupport, blockrowsupport, arguments, paddeddata, resizedata!, BlockVec
import BlockArrays: blockvec
using LinearAlgebra
import LazyArrays: resizedata!, arguments, colsupport, rowsupport, LazyLayout,
                    PaddedLayout, PaddedColumns, paddeddata, ApplyLayout, PaddedArray


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
        @test blocklengths(axes(a[Block.(1:2)],1)) == [5,3]
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
        @test blocklengths(axes(A[Block.(1:2),Block(1)],1)) == [2,3]
        @test blocklengths(axes(A[Block.(1:2),Block(1)],2)) == [3]

        @test A[Block.(1:2), Block.(1:1)] == A
        @test A[Block.(1:2), Block.(1:1)] isa BlockVcat
        @test A[Block.(1:2), 1:2] == A[:,1:2]
        @test A[Block.(1:2), 1:2] isa BlockVcat

        @test convert(AbstractArray{Float64},A) == convert(AbstractMatrix{Float64},A) == A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractMatrix{Float64}(A) == A
        @test copy(A') == A'
    end
    @testset "block vec vcat" begin
        a = BlockedArray(1:5, SVector(1,3))
        b = BlockedArray(2:6, SVector(1,3))

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
        A = BlockedArray(randn(3,2), [1,2], [1,1])
        B = BlockedArray(randn(4,2), [3,1], [1,1])
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

    @testset "BlockedArray" begin
        a = BlockedArray(Vcat(randn(3), 1:3), [3,3])
        @test MemoryLayout(a) isa LazyArrays.ApplyLayout{typeof(vcat)}
        @test LazyArrays.arguments(a) == LazyArrays.arguments(a.blocks)

        b = BlockedArray(Vcat(randn(3), Zeros(3)), [3,3])
        @test paddeddata(view(b, 1:4)) == paddeddata(view(b, Base.OneTo(4))) == b[1:3]

        c = BlockedArray(cache(Zeros(6)), 1:3);
        c[2] = 2
        @test blocksize(paddeddata(c)) == (2,)
        @test paddeddata(c)[Block(2)] == [2.0,0.0]
        resizedata!(c,4);
        @test blocksize(paddeddata(c)) == (3,)

        dat = randn(3,3)
        A = BlockedArray(PaddedArray(dat, 6,6), 1:3, 1:3)
        @test paddeddata(A) == dat
    end

    @testset "blockcol/rowsupport" begin
        B = BlockBandedMatrix(randn(10,10),1:4,1:4,(1,0))
        V = BlockVcat(B, B)
        @test blockrowsupport(V, Block(3)) == blockrowsupport(V, Block(7)) == Block.(2:3)
        @test blockcolsupport(V, Block(1)) == Block.(1:6)
        @test blockbandwidths(V) == (5,0)
        @test isblockbanded(V)
    end

    @testset "use int when possible" begin
        a = BlockedArray(1:5, SVector(1,3))
        b = BlockVcat(1:2, a, 3:4)
        @test axes(b,1) == blockedrange([2, 1, 3, 2])
    end

    @testset "broadcast" begin
        A = BlockVcat(randn(2,3), randn(1,3))
        @test A + I isa BroadcastArray

        a = BlockVcat(randn(2), randn(3))
        @test a' .+ 1 isa BroadcastArray
    end

    @testset "padded different sizes" begin
        @test MemoryLayout(Vcat(Hcat([1,2], Zeros(5)), Hcat([1,2,3], Zeros(4)))) isa ApplyLayout{typeof(vcat)}
    end

    @testset "show" begin
        a = BlockVcat(1:2, 1:3)
        @test summary(a) == "2-blocked 5-element BlockVcat{$Int}"
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
        a = BlockedArray(1:4, SVector(1,3))
        b = BlockedArray(2:5, SVector(1,3))
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
        A = BlockedArray(randn(3,2), [1,2], [1,1])
        B = BlockedArray(randn(3,3), [1,2], [2,1])
        H = BlockHcat(A, B)
        @test H == [A B]
        @test H[Block(1,3)] == B[Block(1,1)]

        @test H[Block(2),Block.(2:3)] == H[Block.(2:2),Block.(2:3)]  == [A[Block(2,2)] B[Block(2,1)]]
        @test H[Block.(2:2),Block(3)] == H[Block(2,3)]
        @test H[:, Block.(1:4)] == H[Block.(1:2), Block.(1:4)] == H[Block.(1:2), :] == H[:,:] == H
        @test blocksize(H[:, Block.(1:4)])  == blocksize(H[Block.(1:2), Block.(1:4)]) == blocksize(H[Block.(1:2), :]) == blocksize(H[:,:]) == blocksize(H)
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
        dest = BlockedArray{Float64}(undef, axes(A))
        @test copyto!(dest, A) == A;
        @test @allocated(copyto!(dest, A)) ≤ 2800
        # dest = BlockArray{Float64}(undef, axes(A))
        # @time copyto!(dest, A);

        dest = BlockedArray{Float64}(undef, axes(A'))
        @test (A')[Block(2,3)] == A[Block(3,2)]'
        @test copyto!(dest, A') ≈ A'
        @test @allocated(copyto!(dest, A')) ≤ 2600


        Rx = BlockBandedMatrices._BandedBlockBandedMatrix(A', axes(k,1), (0,1), (0,0))
        dest = BandedBlockBandedMatrix{Float64}(undef, axes(Rx), (0,1), (0,0))
        @test copyto!(dest, Rx) ≈ BandedBlockBandedMatrix(Rx)

        Vx = view(Rx, Block.(1:N), Block.(1:N))
        @test MemoryLayout(Vx) isa BlockBandedMatrices.BandedBlockBandedColumns

        V = view(A, Block.(1:5),:)
        @test MemoryLayout(V) isa ApplyLayout{typeof(hcat)}
        HV = A[Block.(1:5),:]
        @test HV == V
        # TODO: Fast materialization
    end

    # @testset "Block-mat hcat" begin
    #     A = BlockHcat(BlockedArray(randn(6,4), [4,2], [2,2]), BlockedArray(randn(6,3), [4,2], [2,1]))
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

    @testset "blockcol/rowsupport" begin
        B = BlockBandedMatrix(randn(10,10),1:4,1:4,(1,0))
        H = BlockHcat(B, B)
        @test blockcolsupport(H, Block(2)) == blockcolsupport(H, Block(6)) == Block.(2:3)
        @test blockrowsupport(H, Block(1)) == Block.(1:5)
        @test blockbandwidths(H) == (1,4)
        @test isblockbanded(H)
        @test copyto!(similar(H), H) == copy(H) == H
    end

    @testset "Eye BlockHcat" begin
        B = BlockBandedMatrix(randn(10,10),1:4,1:4,(1,0))
        H = BlockHcat(Eye((axes(B,1),))[:,Block(1)], B)
        @test MemoryLayout(H) isa LazyBandedMatrices.ApplyBlockBandedLayout{typeof(hcat)}
        @test H[Block.(1:4), Block.(1:5)] == H == copy(H)
        @test copyto!(similar(H), H) == H
    end

    @testset "broadcast" begin
        A = BlockHcat(randn(3,2), randn(3,1))
        @test A + I isa BroadcastArray
        @test Adjoint(A) + I isa BroadcastArray
    end

    @testset "show" begin
        a = BlockHcat(1:3, 1:3)
        @test summary(a) == "1×2-blocked 3×2 BlockHcat{$Int}"
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

    @test H + I isa BroadcastArray
    @test H' + I isa BroadcastArray
end


@testset "Interlace" begin
    @testset "vcat" begin
        N = 1000
        a = 1:N
        b = 11:10+N
        a, b = BlockedArray(a,Ones{Int}(length(a))), BlockedArray(b,Ones{Int}(length(b)))
        A = BlockBroadcastArray(vcat, a, b)
        @test axes(A,1) isa BlockedOneTo{Int,StepRangeLen{Int,Int,Int,Int}}

        @test @allocated(axes(A)) ≤ 50
        @test A[Block(1)] == BlockedArray(A)[Block(1)] == [A[1],A[2]] == [1,11]
        @test A[Block(N)] == BlockedArray(A)[Block(N)] == [1000,1010]
        @test convert(AbstractArray{Int},A) ≡ convert(AbstractVector{Int},A) ≡ A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractVector{Float64}(A) == convert(AbstractArray{Float64},A) == convert(AbstractVector{Float64},A) == A
        @test copy(A') == A'

        @test A .+ A isa BroadcastArray
        @test A' .+ A isa BroadcastArray

        @testset "padded" begin
            a = Vcat(randn(3), Zeros(10))
            b = Vcat(randn(4), Zeros(9))
            C = BlockBroadcastArray(vcat,unitblocks(a),unitblocks(a))
            @test C[1:2:end] == a
            @test C[2:2:end] == a

            # differening data sizes not supported yet
            @test_throws ErrorException paddeddata(BlockBroadcastArray(vcat,unitblocks(a),unitblocks(b)))
        end

        @testset "resize!" begin
            N = 10
            a, b = BlockedArray(randn(N),Ones{Int}(N)), BlockedArray(randn(N),Ones{Int}(randn(N)))
            A = BlockBroadcastArray(vcat, a, b)
            Ã = resize!(A, Block(3))
            @test Ã == A[1:6]
            @test_throws BoundsError A[7]
            @test !isassigned(A,7)
        end
    end
    @testset "hcat" begin
        N = 1000
        a = 1:N
        b = 11:10+N
        a, b = BlockedArray(a,Ones{Int}(length(a))), BlockedArray(b,Ones{Int}(length(b)))
        A = BlockBroadcastArray(hcat, a', b')
        @test axes(A,2) isa BlockedOneTo{Int,StepRangeLen{Int,Int,Int,Int}}
        @test @allocated(axes(A)) ≤ 70
        @test A[Block(1,1)] == BlockedArray(A)[Block(1,1)] == [1 11]
        @test A[Block(1,N)] == BlockedArray(A)[Block(1,N)] == [1000 1010]
        @test convert(AbstractArray{Int},A) ≡ convert(AbstractMatrix{Int},A) ≡ A
        @test copy(A) == AbstractArray{Float64}(A) == AbstractMatrix{Float64}(A) == A
        @test copy(A') == A'

        v = BlockVector(randn(3), 1:2)
        H = BlockBroadcastArray(hcat, v, v)
        @test H[Block(2,1)] == [v[Block(2)] v[Block(2)]]
    end
    @testset  "hvcat" begin
        a = unitblocks(randn(2,3))
        b = unitblocks(randn(2,3))
        c = unitblocks(randn(2,3))
        d = unitblocks(randn(2,3))
        e = unitblocks(randn(2,3))
        f = unitblocks(randn(2,3))

        A = BlockBroadcastArray(hvcat, 2, a, b, c, d, e, f)
        @test MemoryLayout(A) isa LazyLayout
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

            @test blockcolsupport(A, Block(2)) == Block.(1:3)
            @test blockrowsupport(A, Block(3)) == Block.(2:4)


            V = view(A, Block.(1:3),Block.(1:3))
            @test MemoryLayout(V) isa LazyBandedMatrices.BlockBandedInterlaceLayout
            @test arguments(V) == (2,a[1:3,1:3],z[1:3,1:3],z[1:3,1:3],a[1:3,1:3])
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

@testset "BlockVec" begin
    X = randn(5,4)
    b = BlockVec(X)
    @test size(b) == (20,)
    @test length(b) == 20
    @test MemoryLayout(b) isa ApplyLayout{typeof(blockvec)}
    @test b == vec(X)
    @test view(b, Block(3)) ≡ view(X, :, 3)
    @test b[Block(3)] isa Vector
    b[5] = 6
    @test X[5] == 6
    @test resize!(b, Block(2)) == b[Block.(1:2)]

    c = BlockVec(X')
    @test c == vec(X')
    @test view(c, Block(3)) ≡ view(X', :, 3)
    @test resize!(c, Block(2)) == c[Block.(1:2)]

    c = BlockVec(transpose(X))
    @test c == vec(transpose(X))
    @test view(c, Block(3)) ≡ view(transpose(X), :, 3)
    @test resize!(c, Block(2)) == c[Block.(1:2)]

    X = cache(Zeros(5,6));
    X[1,1] = 2
    c = BlockVec(X);
    @test MemoryLayout(c) isa PaddedColumns
    @test paddeddata(c) isa BlockVec
    @test paddeddata(c) == [2]
end

end # module
