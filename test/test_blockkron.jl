module TestBlockKron

using LazyBandedMatrices, FillArrays, BandedMatrices, BlockBandedMatrices, BlockArrays, ArrayLayouts, LazyArrays, Test
using LinearAlgebra
import BlockBandedMatrices: isbandedblockbanded, isbanded, BandedBlockBandedStyle, BandedLayout
import LazyBandedMatrices: KronTravBandedBlockBandedLayout, BroadcastBandedLayout, BroadcastBandedBlockBandedLayout, arguments, call, blockcolsupport, InvDiagTrav, invdiagtrav
import ArrayLayouts: FillLayout, OnesLayout
import LazyArrays: resizedata!, FillLayout, arguments, colsupport, call, LazyArrayStyle
import BandedMatrices: BandedColumns

struct MyLazyArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end


Base.size(A::MyLazyArray) = size(A.data)
Base.getindex(A::MyLazyArray, j::Int...) = A.data[j...]
LazyArrays.MemoryLayout(::Type{<:MyLazyArray}) = LazyLayout()
Base.BroadcastStyle(::Type{<:MyLazyArray{<:Any,N}}) where N = LazyArrayStyle{N}()
LinearAlgebra.factorize(A::MyLazyArray) = factorize(A.data)


@testset "Kron" begin
    @testset "DiagTrav" begin
        A = [1 2 3; 4 5 6; 7 8 9]
        @test DiagTrav(A) == Vector(DiagTrav(A)) == [1, 4, 2, 7, 5, 3]
        @test resize!(DiagTrav(A), Block(2)) == [1, 4,2]
        @test maximum(abs, DiagTrav(A)) == 7
        @test copy(DiagTrav(A)) == DiagTrav(A)

        A = [1 2 3; 4 5 6]
        @test DiagTrav(A) == [1, 4, 2, 5, 3]
        A = [1 2; 3 4; 5 6]
        @test DiagTrav(A) == [1, 3, 2, 5, 4]

        @test resize!(DiagTrav(A), Block(2)) == [1, 3,2]

        A = DiagTrav(randn(3,3,3))
        @test A[Block(1)] == A[1:1,1,1]
        @test A[Block(2)] == [A.array[2,1,1], A.array[1,2,1], A.array[1,1,2]]
        @test A[Block(3)] == [A.array[3,1,1], A.array[2,2,1], A.array[2,1,2],
                            A.array[1,3,1], A.array[1,2,2], A.array[1,1,3]]
        @test A == [A[Block(1)]; A[Block(2)]; A[Block(3)]]

        A = reshape(1:9,3,3)'
        @test DiagTrav(A) == Vector(DiagTrav(A)) == [1, 4, 2, 7, 5, 3]
        A = reshape(1:12,3,4)'
        @test DiagTrav(A) == [1, 4, 2, 7, 5, 3, 10, 8, 6]
        A = reshape(1:12,3,4)
        @test DiagTrav(A) == [1, 2, 4, 3, 5, 7, 6, 8, 10]

        C = cache(Zeros(10,10));
        C[1:3,1:3] .= [1 2 3; 4 5 6; 7 8 9];
        @test blockcolsupport(DiagTrav(C)) == Block.(1:5)
        @test DiagTrav(C) == [1; 4; 2; 7; 5; 3; 0; 8; 6; 0; 0; 0; 9; zeros(42)]
        C = cache(Zeros(5,6));
        C[1:3,1:4] .= [1 2 3 4; 4 5 6 4; 7 8 9 4];
        @test DiagTrav(C) == [1; 4; 2; 7; 5; 3; 0; 8; 6; 4; 0; 0; 9; 4; 0; 0; 0; 4; 0; 0]
        C = cache(Zeros(6,5));
        C[1:3,1:4] .= [1 2 3 4; 4 5 6 4; 7 8 9 4];
        @test DiagTrav(C) == [1; 4; 2; 7; 5; 3; 0; 8; 6; 4; 0; 0; 9; 4; 0; 0; 0; 0; 4; 0]

        a = DiagTrav(ones(1,1))
        @test a == [1]
        a = DiagTrav(ones(1,3))
        @test a == ones(3)
    end

    @testset "InvDiagTrav" begin
        A = [1 2 3; 4 5 6; 7 8 9]
        @test invdiagtrav(BlockedVector(DiagTrav(A))) == [1 2 3; 4 5 0; 7 0 0]
        @test invdiagtrav(DiagTrav(A)) == A
    end

    @testset "BlockKron" begin
        n = 4
        Δ = BandedMatrix(1 => Ones(n-1), 0 => Fill(-2,n), -1 => Ones(n-1))
        A = BlockKron(Δ, Eye(n))
        @test isblockbanded(A)
        @test isbandedblockbanded(A)
        @test blockbandwidths(A) == (1,1)
        @test BandedBlockBandedMatrix(A) == A

        A = BlockKron(Δ, Eye(n), Eye(n))
        @test isblockbanded(A)
        @test blockbandwidths(A) == (1,1)
        @test subblockbandwidths(A) == (0,0)
        @test BandedBlockBandedMatrix(A) == A

        B = BlockKron(Eye(n), Δ, Eye(n))
        @test isblockbanded(B)
        @test blockbandwidths(B) == (0,0)
        @test_broken subblockbandwidths(B) == (4,4)
        @test_broken BandedBlockBandedMatrix(B) == B
    end

    @testset "KronTrav" begin
        @testset "vector" begin
            a = [1,2,3]
            b = [4,5,6]
            c = [7,8]
            @test KronTrav(a,b) == DiagTrav(b*a')
            @test KronTrav(a,c) == [7,8,14,16,21]
            @test KronTrav(c,a) == [7,14,8,21,16]
        end

        @testset "matrix" begin
            A = [1 2; 3 4]
            B = [5 6; 7 8]
            K = KronTrav(A,B)

            @test copy(K) == K

            X = [9 10; 11 0]
            @test K*DiagTrav(X) == DiagTrav(B*X*A')

            @test K[Block.(Base.OneTo(2)), Block.(Base.OneTo(2))] == K[Block.(1:2), Block.(1:2)] == K
        end 

        @testset "tensor" begin
            A = [1 2;
                 3 4]
            B = [5 6;
                 7 8]
            C = [2 4;
                 6 8]
            K = KronTrav(A,B,C)
            @test K == [1*5*2 1*5*4 1*6*2 2*5*2;
                        1*5*6 1*5*8 1*6*6 2*5*6;
                        1*7*2 1*7*4 1*8*2 2*7*2;
                        3*5*2 3*5*4 3*6*2 4*5*2]

            @test K == KronTrav(A, B, 1.0C)

            @test K == K[Block.(Base.OneTo(2)), Block.(Base.OneTo(2))] == K[Block.(1:2),Block.(1:2)]

            n = 2
            X = Array(reshape(1:n^3, n, n, n))
            X[2,2,1] = X[1,2,2] = X[2,1,2] = X[2,2,2] = 0
            Y = similar(X)
            for k = 1:n, j=1:n Y[k,j,:] = A*X[k,j,:] end
            for k = 1:n, j=1:n Y[k,:,j] = B*Y[k,:,j] end
            for k = 1:n, j=1:n Y[:,k,j] = C*Y[:,k,j] end
            @test K*DiagTrav(X) ≈ DiagTrav(Y)

            n = 3
            A,B,C = randn(n,n), randn(n,n), randn(n,n)
            K = KronTrav(A,B,C)

            X = randn(n,n,n)
            for ℓ = 0:n-1, j=0:n-1, k=max(0,n-(ℓ+j)):n-1
                X[k+1,j+1,ℓ+1] = 0
            end
            Y = float(similar(X))
            for k = 1:n, j=1:n Y[k,j,:] = A*X[k,j,:] end
            for k = 1:n, j=1:n Y[k,:,j] = B*Y[k,:,j] end
            for k = 1:n, j=1:n Y[:,k,j] = C*Y[:,k,j] end
            @test K*DiagTrav(X) ≈ DiagTrav(Y)

            @test_throws ErrorException KronTrav(randn(3), randn(3,2))
        end

        @testset "banded" begin
            @testset "2D" begin
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

                @test A[Block.(Base.OneTo(3)), Block.(Base.OneTo(3))] isa KronTrav
            end
            @testset "3D" begin
                n = 4
                D² = BandedMatrix(1 => Ones(n-1), 0 => Fill(-2,n), -1 => Ones(n-1))
                A = KronTrav(D², Eye(n), Eye(n))
                B = KronTrav(Eye(n), D², Eye(n))
                C = KronTrav(Eye(n), Eye(n), D²)
                @test blockbandwidths(A) == blockbandwidths(B) == blockbandwidths(C) == (1,1)

                @test A == KronTrav(map(Matrix, A.args)...) == BandedBlockBandedMatrix(A)

                Δ = A + B + C

                X = randn(n, n, n)
                for ℓ = 0:n-1, j=0:n-1, k=max(0,n-(ℓ+j)):n-1
                    X[k+1,j+1,ℓ+1] = 0
                end

                Y = similar(X)
                for k = 1:n, j=1:n Y[k,j,:] = D²*X[k,j,:] end
                @test A*DiagTrav(X) ≈ DiagTrav(Y)

                for k = 1:n, j=1:n Y[k,:,j] += D²*X[k,:,j] end
                for k = 1:n, j=1:n Y[:,k,j] += D²*X[:,k,j] end
                @test Δ*DiagTrav(X) ≈ DiagTrav(Y)
            end
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

        @testset "Lazy" begin
            n = 5
            A = MyLazyArray(randn(n,n))
            B = brand(n,n,1,1)
            @test Base.BroadcastStyle(typeof(KronTrav(A,B))) isa LazyArrayStyle{2}
            @test Base.BroadcastStyle(typeof(KronTrav(B,A))) isa LazyArrayStyle{2}
            @test Base.BroadcastStyle(typeof(KronTrav(A,A))) isa LazyArrayStyle{2}
        end

        @testset "Mul" begin
            n = 4
            Δ = BandedMatrix(1 => Ones(n-1), 0 => Fill(-2,n), -1 => Ones(n-1))
            A = KronTrav(Δ, Eye(n))
            @test A^2 == Matrix(A)^2
        end

        @testset "algebra" begin
            n = 4
            Δ = BandedMatrix(1 => Ones(n-1), 0 => Fill(-2,n), -1 => Ones(n-1))
            A = KronTrav(Δ, Eye(n))
            @test 2A == A*2 == 2Matrix(A)
            @test 2A isa KronTrav
            @test A*2 isa KronTrav
        end
    end


    @testset "banded-block-banded Kron" begin
        n = 4
        h = 1/n
        D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2

        D_xx = BandedBlockBandedMatrix(BlockKron(D², Eye(n)))
        D_yy = BandedBlockBandedMatrix(BlockKron(Eye(n),D²))
        Δ = BroadcastArray(+, D_xx, D_yy)
        @test MemoryLayout(Δ) isa BroadcastBandedBlockBandedLayout
        @test blockbandwidths(Δ) == (1,1)
        @test subblockbandwidths(Δ) == (1,1)
        @test Δ[Block.(1:4),Block.(2:4)] == Δ[:,5:end]

        @testset "irradic indexing" begin
            B = cache(Δ);
            resizedata!(B,16,1);
            # we do blockwise
            @test B.data[1:16,1:4] == Δ[1:16,1:4]
            resizedata!(B,1,16);
            @test B.datasize == (16,16)
            @test B.data == Δ
        end

        @testset "Cache-block indexing" begin
            B = cache(Δ);
            @test B[Block(1,1)] == B[Block(1),Block(1)] == Δ[Block(1,1)]
            @test B[Block.(1:3),Block.(1:2)] == Δ[Block.(1:3),Block.(1:2)]
            @test B[:, Block(1)] == Δ[:, Block(1)]
            @test B[Block(1), :] == Δ[Block(1), :]
            @test B[Block(1), [1,2,3]] == Δ[Block(1), [1,2,3]]
            @test B[[1,2,3], Block(1)] == Δ[[1,2,3], Block(1)]
        end
    
        @testset "resizedata!" begin
            B = cache(Δ);
            resizedata!(B,5,5);
            @test B.data[1:5,1:5] == Δ[1:5,1:5]
            @test B == Δ
        end
    end
end

end # module
