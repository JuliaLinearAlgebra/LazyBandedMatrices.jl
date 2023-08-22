using LazyBandedMatrices, LazyArrays, BlockBandedMatrices, BlockArrays, Test
using LazyArrays: paddeddata
using BlockArrays: blockcolsupport

@testset "Block" begin
    @testset "LazyBlock" begin
        @test Block(5) in BroadcastVector(Block, [1,3,5])
        @test Base.broadcasted(LazyArrayStyle{1}(), Block, 1:5) ≡ Block.(1:5)
        @test Base.broadcasted(LazyArrayStyle{1}(), Int, Block.(1:5)) ≡ 1:5
        @test Base.broadcasted(LazyArrayStyle{0}(), Int, Block(1)) ≡ 1
    end

    @testset "LazyBlockArray Triangle Recurrences" begin
        N = 1000
        n = mortar(BroadcastArray(Fill,Base.OneTo(N),Base.OneTo(N)))
        k = mortar(BroadcastArray(Base.OneTo,Base.OneTo(N)))

        @test view(n, Block(5)) ≡ Fill(5,5)
        @test view(k,Block(5)) ≡ Base.OneTo(5)
        a = b = c = 0.0
        # for some reason the following was causing major slowdown. I think it 
        # went pass a limit to Base.Broadcast.flatten which caused `bc.f` to have a strange type.
        # bc = Base.Broadcast.instantiate(Base.broadcasted(/, Base.broadcasted(*, k, Base.broadcasted(-, Base.broadcasted(-, k, n), a)), Base.broadcasted(+, 2k, b+c-1)))

        bc = Base.Broadcast.instantiate(Base.broadcasted((k,n,a,b,c) -> k * (k-n-a) / (2k+(b+c-1)), k, n, a, b, c))
        @test axes(n,1) ≡ axes(k,1) ≡ axes(bc)[1] ≡ blockedrange(Base.OneTo(N))
        u = (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))
        @test u == (Vector(k) .* (Vector(k) .- Vector(n) .- a) ./ (2Vector(k) .+ (b+c-1)))
        @test copyto!(u, bc) == (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))
        @test @allocated(copyto!(u, bc)) ≤ 1000 
        # not clear why allocatinos so high: all allocations are coming from checking
        # axes

        u = PseudoBlockArray{Float64}(undef, collect(1:N))
        @test copyto!(u, bc) == (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))
        @test @allocated(copyto!(u, bc)) ≤ 1000
    end

    @testset "BlockBanded and padded" begin
        A = BlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,0)); A.data .= randn.();
        D = mortar(Diagonal([randn(k,k) for k=1:4]))
        c = Vcat(randn(3), Zeros(7))
        b = PseudoBlockVector(c, (axes(A,2),))
        @test MemoryLayout(b) isa PaddedLayout
        @test MemoryLayout(A*b) isa PaddedLayout
        @test MemoryLayout(A*c) isa PaddedLayout
        @test A*b ≈ A*c ≈ Matrix(A)*Vector(b)
        @test D*b ≈ D*c ≈ Matrix(D)*Vector(b)

        @test b[Block.(2:3)] isa PseudoBlockVector{Float64,<:ApplyArray}
        @test MemoryLayout(b[Block.(2:3)]) isa PaddedLayout
        @test b[Block.(2:3)] == b[2:6]
    end

     
    @testset "block padded" begin
        c = PseudoBlockVector(Vcat(1, Zeros(5)), 1:3)
        @test paddeddata(c) == [1]
        @test paddeddata(c) isa PseudoBlockVector
        @test blockcolsupport(c) == Block.(1:1)
        C = PseudoBlockArray(Vcat(randn(2,3), Zeros(4,3)), 1:3, [1,2])
        @test blockcolsupport(C) == Block.(1:2)
        @test blockrowsupport(C) == Block.(1:2)

        @test C[Block.(1:2),1:3] == C[Block.(1:2),Block.(1:2)] == C[1:3,Block.(1:2)] == C[1:3,1:3]

        H = PseudoBlockArray(Hcat(1, Zeros(1,5)), [1], 1:3)
        @test paddeddata(H) == Ones(1,1)
    end

    @testset "MulBlockBanded" begin
        A = BlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,0)); A.data .= randn.();
        B = BlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,1)); B.data .= randn.();
        M = ApplyMatrix(*, A, B)
        @test blockbandwidths(M) == (2,1)
        @test MemoryLayout(M) isa ApplyBlockBandedLayout{typeof(*)}
        @test Base.BroadcastStyle(typeof(M)) isa LazyArrayStyle{2}
        @test BlockBandedMatrix(M) ≈ A*B
        @test arguments(M) == (A,B)
        V = view(M, Block.(1:2), Block.(1:2))
        @test MemoryLayout(V) isa ApplyBlockBandedLayout{typeof(*)}
        @test arguments(V) == (A[Block.(1:2),Block.(1:2)], B[Block.(1:2),Block.(1:2)])
        @test M[Block.(1:2), Block.(1:2)] isa BlockBandedMatrix
    end
    @testset "MulBandedBlockBanded" begin
        A = BandedBlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,0), (1,0)); A.data .= randn.();
        B = BandedBlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,1), (1,1)); B.data .= randn.();
        M = ApplyMatrix(*, A, B)
        @test blockbandwidths(M) == (2,1)
        @test subblockbandwidths(M) == (2,1)
        @test MemoryLayout(M) isa ApplyBandedBlockBandedLayout{typeof(*)}
        @test Base.BroadcastStyle(typeof(M)) isa LazyArrayStyle{2}
        @test BandedBlockBandedMatrix(M) ≈ BlockBandedMatrix(M) ≈ A*B
        @test arguments(M) == (A,B)
        V = view(M, Block.(1:2), Block.(1:2))
        @test MemoryLayout(V) isa ApplyBandedBlockBandedLayout{typeof(*)}
        @test arguments(V) == (A[Block.(1:2),Block.(1:2)], B[Block.(1:2),Block.(1:2)])
        @test M[Block.(1:2), Block.(1:2)] isa BandedBlockBandedMatrix
        V = view(M, 1:3, 1:3)
        @test MemoryLayout(V) isa ApplyLayout{typeof(*)}
        @test arguments(V) == (A[1:3,1:3], B[1:3,1:3])
        @test M[1:3, 1:3] ≈ (A*B)[1:3,1:3]

        @test M[Block(2)[1:2],Block(2)[1:2]] isa BandedMatrix
        @test M[Block(2)[1:2],Block(2)] isa BandedMatrix
        @test M[Block(2),Block(2)[1:2]] isa BandedMatrix
        @test M[Block.(1:2), Block.(2:3)] isa BandedBlockBandedMatrix
        @test M[Block(2),Block.(2:3)] isa PseudoBlockArray
        @test M[Block.(2:3),Block(2)] isa PseudoBlockArray
        @test M[Block.(2:3),Block(2)[1:2]] isa PseudoBlockArray
        @test M[Block(2)[1:2],Block.(2:3)] isa PseudoBlockArray
    end

    @testset "BroadcastMatrix" begin
        @testset "BroadcastBlockBanded" begin
            A = BlockBandedMatrix(randn(6,6),1:3,1:3,(1,1))
            B = BroadcastMatrix(*, 2, A)
            @test blockbandwidths(B) == (1,1)
            @test MemoryLayout(B) == BroadcastBlockBandedLayout{typeof(*)}()
            @test BandedBlockBandedMatrix(B) == B == copyto!(BandedBlockBandedMatrix(B), B) == 2*B.args[2]
            @test MemoryLayout(B') isa LazyBandedMatrices.LazyBlockBandedLayout
            @test BlockBandedMatrix(B') == B'
    
            C = BroadcastMatrix(*, 2, im*A)
            @test MemoryLayout(C') isa LazyBandedMatrices.LazyBlockBandedLayout
            @test MemoryLayout(transpose(C)) isa LazyBandedMatrices.LazyBlockBandedLayout
    
            E = BroadcastMatrix(*, A, 2)
            @test MemoryLayout(E) == BroadcastBlockBandedLayout{typeof(*)}()
    
            
            D = Diagonal(PseudoBlockArray(randn(6),1:3))
            @test MemoryLayout(BroadcastMatrix(*, A, D)) isa BroadcastBlockBandedLayout{typeof(*)}
            @test MemoryLayout(BroadcastMatrix(*, D, A)) isa BroadcastBlockBandedLayout{typeof(*)}
    
            F = BroadcastMatrix(*, A, A)
            @test MemoryLayout(F) == BroadcastBlockBandedLayout{typeof(*)}()
        end
        @testset "BroadcastBandedBlockBanded" begin
            A = BandedBlockBandedMatrix(randn(6,6),1:3,1:3,(1,1),(1,1))
    
            B = BroadcastMatrix(*, 2, A)
            @test blockbandwidths(B) == (1,1)
            @test subblockbandwidths(B) == (1,1)
            @test MemoryLayout(B) == BroadcastBandedBlockBandedLayout{typeof(*)}()
            @test BandedBlockBandedMatrix(B) == B == copyto!(BandedBlockBandedMatrix(B), B) == 2*B.args[2]
            @test MemoryLayout(B') isa LazyBandedMatrices.LazyBandedBlockBandedLayout
            @test BandedBlockBandedMatrix(B') == B'
            @test MemoryLayout(Symmetric(B)) isa LazyBandedMatrices.LazyBandedBlockBandedLayout
            @test MemoryLayout(Hermitian(B)) isa LazyBandedMatrices.LazyBandedBlockBandedLayout
    
            C = BroadcastMatrix(*, 2, im*A)
            @test MemoryLayout(C') isa LazyBandedMatrices.LazyBandedBlockBandedLayout
            @test MemoryLayout(transpose(C)) isa LazyBandedMatrices.LazyBandedBlockBandedLayout
    
            E = BroadcastMatrix(*, A, 2)
            @test MemoryLayout(E) == BroadcastBandedBlockBandedLayout{typeof(*)}()
    
            D = Diagonal(PseudoBlockArray(randn(6),1:3))
            @test MemoryLayout(BroadcastMatrix(*, A, D)) isa BroadcastBandedBlockBandedLayout{typeof(*)}
            @test MemoryLayout(BroadcastMatrix(*, D, A)) isa BroadcastBandedBlockBandedLayout{typeof(*)}
    
            F = BroadcastMatrix(*, Ones(axes(A,1)), A)
            @test blockbandwidths(F) == (1,1)
            @test subblockbandwidths(F) == (1,1)
            @test F == A
        end
    end

    @testset "Padded Block" begin
        b = PseudoBlockArray(cache(Zeros(55)),1:10);
        b[10] = 5;
        @test MemoryLayout(b) isa PaddedLayout{DenseColumnMajor}
        @test paddeddata(b) isa PseudoBlockVector
        @test paddeddata(b) == [zeros(9); 5]
    end

    @testset "Lazy block" begin
        b = PseudoBlockVector(randn(5),[2,3])
        c = BroadcastVector(exp,1:5)
        @test c .* b isa BroadcastVector
        @test b .* c isa BroadcastVector
        @test (c .* b)[Block(1)] == c[1:2] .* b[Block(1)]
    end

    @testset "Apply block indexing" begin
        b = PseudoBlockVector(randn(5),[2,3])
        a = ApplyArray(+, b, b)

        @test exp.(view(a,Block.(1:2))) == exp.(a)

        B = BandedBlockBandedMatrix(randn(6,6),1:3,1:3,(1,1),(1,1))
        A = ApplyArray(+, B, B)
        @test exp.(view(A,Block.(1:3),Block.(1:3))) == exp.(A)
        @test exp.(view(A,Block.(1:3),2)) == exp.(A)[Block.(1:3),2]
        @test exp.(view(A,2,Block.(1:3))) == exp.(A)[2,Block.(1:3)]
    end

    @testset "Blockbandwidths" begin
        @test blockbandwidths(unitblocks(Diagonal(1:5))) == (0,0)
    end

    @testset "PaddedArray" begin
        p = PaddedArray(1:5, (blockedrange(1:4),))
        @test paddeddata(p) == [1:5; 0]
        @test blocksize(paddeddata(p),1) == 3
    end
end