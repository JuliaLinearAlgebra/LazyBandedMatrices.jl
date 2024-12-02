using LazyBandedMatrices, InfiniteArrays, ArrayLayouts, LazyArrays, BlockArrays, BandedMatrices, BlockBandedMatrices, Test
using InfiniteArrays: TridiagonalToeplitzLayout, BidiagonalToeplitzLayout, TridiagonalToeplitzLayout
using Base: oneto
using BlockArrays: blockcolsupport
using LazyArrays: arguments

const InfiniteArraysBlockArraysExt = Base.get_extension(InfiniteArrays, :InfiniteArraysBlockArraysExt)
const LazyBandedMatricesInfiniteArraysExt = Base.get_extension(LazyBandedMatrices, :LazyBandedMatricesInfiniteArraysExt)

const OneToInfBlocks = InfiniteArraysBlockArraysExt.OneToInfBlocks
const InfKronTravBandedBlockBandedLayout = LazyBandedMatricesInfiniteArraysExt.InfKronTravBandedBlockBandedLayout

@testset "∞ LazyBandedMatrices" begin
    @test MemoryLayout(LazyBandedMatrices.Tridiagonal(Fill(1,∞), Zeros(∞), Fill(3,∞))) isa TridiagonalToeplitzLayout
    @test MemoryLayout(LazyBandedMatrices.Bidiagonal(Fill(1,∞), Zeros(∞), :U)) isa BidiagonalToeplitzLayout
    @test MemoryLayout(LazyBandedMatrices.SymTridiagonal(Fill(1,∞), Zeros(∞))) isa TridiagonalToeplitzLayout

    T = LazyBandedMatrices.Tridiagonal(Fill(1,∞), Zeros(∞), Fill(3,∞))
    @test T[2:∞,3:∞] isa SubArray
    @test exp.(T) isa BroadcastMatrix
    @test exp.(T)[2:∞,3:∞][1:10,1:10] == exp.(T[2:∞,3:∞])[1:10,1:10] == exp.(T[2:11,3:12])
    @test exp.(T)[2:∞,3:∞] isa BroadcastMatrix
    @test exp.(T[2:∞,3:∞]) isa BroadcastMatrix

    B = LazyBandedMatrices.Bidiagonal(Fill(1,∞), Zeros(∞), :U)
    @test B[2:∞,3:∞] isa SubArray
    @test exp.(B) isa BroadcastMatrix
    @test exp.(B)[2:∞,3:∞][1:10,1:10] == exp.(B[2:∞,3:∞])[1:10,1:10] == exp.(B[2:11,3:12])
    @test exp.(B)[2:∞,3:∞] isa BroadcastMatrix

    @testset "Diagonal{Fill} * Bidiagonal" begin
        A, B = Diagonal(Fill(2,∞)) , LazyBandedMatrices.Bidiagonal(exp.(1:∞), exp.(1:∞), :L)
        @test (A*B)[1:10,1:10] ≈ (B*A)[1:10,1:10] ≈ 2B[1:10,1:10]
    end

    @testset "∞-unit blocks" begin
        @test unitblocks(oneto(∞)) ≡ blockedrange(Ones{Int}(∞))
        @test unitblocks(2:∞) == 2:∞

        @test unitblocks(oneto(∞))[Block.(2:∞)] == 2:∞
    end

    @testset "concat" begin
        a = unitblocks(1:∞)
        b = exp.(a)
        c = BlockBroadcastArray(vcat, a, b)
        @test length(c) == ∞
        @test blocksize(c) == (∞,)
        @test c[Block(5)] == [a[5], b[5]]

        A = unitblocks(BandedMatrix(0 => 1:∞, 1 => Fill(2.0, ∞), -1 => Fill(3.0, ∞)))
        B = BlockBroadcastArray(hvcat, 2, A, Zeros(axes(A)), Zeros(axes(A)), A)
        @test B[Block(3, 3)] == [A[3, 3] 0; 0 A[3, 3]]
        @test B[Block(3, 4)] == [A[3, 4] 0; 0 A[3, 4]]
        @test B[Block(3, 5)] == [A[3, 5] 0; 0 A[3, 5]]
        @test blockbandwidths(B) == (1, 1)
        @test subblockbandwidths(B) == (0, 0)
        @test B[Block.(1:10), Block.(1:10)] isa BlockSkylineMatrix

        C = BlockBroadcastArray(hvcat, 2, A, A, A, A)
        @test C[Block(3, 3)] == fill(A[3, 3], 2, 2)
        @test C[Block(3, 4)] == fill(A[3, 4], 2, 2)
        @test C[Block(3, 5)] == fill(A[3, 5], 2, 2)
        @test blockbandwidths(C) == (1, 1)
        @test subblockbandwidths(C) == (1, 1)
        @test B[Block.(1:10), Block.(1:10)] isa BlockSkylineMatrix
    end

    @testset "DiagTrav" begin
        C = zeros(∞,∞);
        C[1:3,1:4] .= [1 2 3 4; 5 6 7 8; 9 10 11 12]
        A = DiagTrav(C)
        @test blockcolsupport(A) == Block.(1:6)
        @test A[Block.(1:7)] == [1; 5; 2; 9; 6; 3; 0; 10; 7; 4; 0; 0; 11; 8; 0; 0; 0; 0; 12; zeros(9)]

        C = zeros(∞,4);
        C[1:3,1:4] .= [1 2 3 4; 5 6 7 8; 9 10 11 12]
        A = DiagTrav(C)
        @test A[Block.(1:7)] == [1; 5; 2; 9; 6; 3; 0; 10; 7; 4; 0; 0; 11; 8; 0; 0; 0; 12; zeros(4)]

        C = zeros(3,∞);
        C[1:3,1:4] .= [1 2 3 4; 5 6 7 8; 9 10 11 12]
        A = DiagTrav(C)
        @test A[Block.(1:7)] == [1; 5; 2; 9; 6; 3; 10; 7; 4; 11; 8; 0; 12; zeros(5)]
    end

    @testset "KronTrav" begin
        Δ = BandedMatrix(1 => Ones(∞), -1 => Ones(∞)) / 2
        A = KronTrav(Δ - 2I, Eye(∞))
        @test axes(A, 1) isa OneToInfBlocks
        V = view(A, Block.(Base.OneTo(3)), Block.(Base.OneTo(3)))

        @test MemoryLayout(A) isa InfKronTravBandedBlockBandedLayout
        @test MemoryLayout(V) isa LazyBandedMatrices.KronTravBandedBlockBandedLayout

        @test A[Block.(Base.OneTo(3)), Block.(Base.OneTo(3))] isa KronTrav

        u = A * [1; zeros(∞)]
        @test u[1:3] == A[1:3, 1]
        @test bandwidths(view(A, Block(1, 1))) == (1, 1)

        @test A*A isa KronTrav
        @test (A*A)[Block.(Base.OneTo(3)), Block.(Base.OneTo(3))] ≈ A[Block.(1:3), Block.(1:4)]A[Block.(1:4), Block.(1:3)]
    end

    @testset "BlockHcat copyto!" begin
        n = mortar(Fill.(oneto(∞), oneto(∞)))
        k = mortar(Base.OneTo.(oneto(∞)))

        a = b = c = 0.0
        dat = BlockHcat(
            BroadcastArray((n, k, b, bc1) -> (k + b - 1) * (n + k + bc1) / (2k + bc1), n, k, b, b + c - 1),
            BroadcastArray((n, k, abc, bc, bc1) -> (n + k + abc) * (k + bc) / (2k + bc1), n, k, a + b + c, b + c, b + c - 1)
        )
        N = 1000
        KR = Block.(Base.OneTo(N))
        V = view(dat, Block.(Base.OneTo(N)), :)
        @test MemoryLayout(V) isa LazyArrays.ApplyLayout{typeof(hcat)}
        @test BlockedArray(V)[Block.(1:5), :] == dat[Block.(1:5), :]
        V = view(dat', :, Block.(Base.OneTo(N)))
        @test MemoryLayout(V) isa LazyArrays.ApplyLayout{typeof(vcat)}
        a = dat.arrays[1]'
        N = 100
        KR = Block.(Base.OneTo(N))
        v = view(a, :, KR)
        @time r = BlockedArray(v)
        @test v == r
    end

    @testset "Symmetric" begin
        k = mortar(Base.OneTo.(oneto(∞)))
        n = mortar(Fill.(oneto(∞), oneto(∞)))

        dat = BlockHcat(
            BlockBroadcastArray(hcat, float.(k), Zeros((axes(n, 1),)), float.(n)),
            Zeros((axes(n, 1), Base.OneTo(3))),
            Zeros((axes(n, 1), Base.OneTo(3))))
        M = BlockBandedMatrices._BandedBlockBandedMatrix(dat', axes(k, 1), (1, 1), (1, 1))
        Ms = Symmetric(M)
        @test blockbandwidths(M) == (1, 1)
        @test blockbandwidths(Ms) == (1, 1)
        @test Ms[Block.(1:5), Block.(1:5)] == Symmetric(M[Block.(1:5), Block.(1:5)])
        @test Ms[Block.(1:5), Block.(1:5)] isa BandedBlockBandedMatrix

        b = [ones(10); zeros(∞)]
        @test (Ms * b)[Block.(1:6)] == Ms[Block.(1:6), Block.(1:4)]*ones(10)
        @test ((Ms * Ms) * b)[Block.(1:6)] == (Ms * (Ms * b))[Block.(1:6)]
        @test ((Ms + Ms) * b)[Block.(1:6)] == (2*(Ms * b))[Block.(1:6)]

        dat = BlockBroadcastArray(hcat, float.(k), Zeros((axes(n, 1),)), float.(n))
        M = BlockBandedMatrices._BandedBlockBandedMatrix(dat', axes(k, 1), (-1, 1), (1, 1))
        Ms = Symmetric(M)
        @test Symmetric((M+M)[Block.(1:10), Block.(1:10)]) == (Ms+Ms)[Block.(1:10), Block.(1:10)]
    end

    @testset "BlockBidiagonal" begin
        B = mortar(LazyBandedMatrices.Bidiagonal(Fill([1 2; 3 4], ∞), Fill([1 2; 3 4], ∞), :U));
        #TODO: copy BlockBidiagonal code from BlockBandedMatrices to LazyBandedMatrices
        @test B[Block(2, 3)] == [1 2; 3 4]
        @test_broken B[Block(1, 3)] == Zeros(2, 2)
    end

    @testset "KronTrav" begin
        Δ = BandedMatrix(1 => Ones(∞) / 2, -1 => Ones(∞))
        A = KronTrav(Δ, Eye(∞))
        @test A[Block(100, 101)] isa BandedMatrix
        @test A[Block(100, 100)] isa BandedMatrix
        @test A[Block.(1:5), Block.(1:5)] isa BandedBlockBandedMatrix
        B = KronTrav(Eye(∞), Δ)
        @test B[Block(100, 101)] isa BandedMatrix
        @test B[Block(100, 100)] isa BandedMatrix
        V = view(A + B, Block.(1:5), Block.(1:5))
        @test MemoryLayout(typeof(V)) isa BroadcastBandedBlockBandedLayout{typeof(+)}
        @test arguments(V) == (A[Block.(1:5), Block.(1:5)], B[Block.(1:5), Block.(1:5)])
        @test (A+B)[Block.(1:5), Block.(1:5)] == A[Block.(1:5), Block.(1:5)] + B[Block.(1:5), Block.(1:5)]

        @test blockbandwidths(A + B) == (1, 1)
        @test blockbandwidths(2A) == (1, 1)
        @test blockbandwidths(2 * (A + B)) == (1, 1)

        @test subblockbandwidths(A + B) == (1, 1)
        @test subblockbandwidths(2A) == (1, 1)
        @test subblockbandwidths(2 * (A + B)) == (1, 1)
    end

    @testset "BlockTridiagonal" begin
        T = mortar(LazyBandedMatrices.Tridiagonal(Fill([1 2; 3 4], ∞), Fill([1 2; 3 4], ∞), Fill([1 2; 3 4], ∞)));
        #TODO: copy BlockBidiagonal code from BlockBandedMatrices to LazyBandedMatrices
        @test T[Block(2, 2)] == [1 2; 3 4]
        @test_broken T[Block(1, 3)] == Zeros(2, 2)
    end
end