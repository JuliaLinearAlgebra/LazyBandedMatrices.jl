# This file based on a part of Julia, LinearAlgebra/test/tridiag.jl. License is MIT: https://julialang.org/license

module TestTridiagonal

using Test, SparseArrays, Random, LazyBandedMatrices, FillArrays, LazyArrays
import LinearAlgebra
# need to avoid confusion with LinearAlgebra.Tridiagonal
import LinearAlgebra: tril!, triu!, tril, triu, det, logabsdet, diag, isdiag, istriu, istril, I, dot,
                        UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular,
                        Symmetric, Hermitian, Diagonal, diagm, eigen, eigvals
import LazyBandedMatrices: SymTridiagonal, Tridiagonal

@testset "Tridigonal" begin
    @testset for elty in (Float32, Float64, ComplexF32, ComplexF64, Int)
        n = 12 #Size of matrix problem to test
        Random.seed!(123)
        if elty == Int
            Random.seed!(61516384)
            d = rand(1:100, n)
            dl = -rand(0:10, n-1)
            du = -rand(0:10, n-1)
            v = rand(1:100, n)
            B = rand(1:100, n, 2)
            a = rand(1:100, n-1)
            b = rand(1:100, n)
            c = rand(1:100, n-1)
        else
            d = convert(Vector{elty}, 1 .+ randn(n))
            dl = convert(Vector{elty}, randn(n - 1))
            du = convert(Vector{elty}, randn(n - 1))
            v = convert(Vector{elty}, randn(n))
            B = convert(Matrix{elty}, randn(n, 2))
            a = convert(Vector{elty}, randn(n - 1))
            b = convert(Vector{elty}, randn(n))
            c = convert(Vector{elty}, randn(n - 1))
            if elty <: Complex
                a += im*convert(Vector{elty}, randn(n - 1))
                b += im*convert(Vector{elty}, randn(n))
                c += im*convert(Vector{elty}, randn(n - 1))
            end
        end
        @test_throws DimensionMismatch SymTridiagonal(dl, fill(elty(1), n+1))
        @test SymTridiagonal(ones(n, n)) == SymTridiagonal(ones(n), ones(n-1))
        @test_throws ArgumentError Tridiagonal(dl, dl, dl)
        @test_throws ArgumentError convert(SymTridiagonal{elty}, Tridiagonal(dl, d, du))

        if elty != Int
            @testset "issue #1490" begin
                @test det(fill(elty(1),3,3)) ≈ zero(elty) atol=3*eps(real(one(elty)))
                @test det(SymTridiagonal(elty[],elty[])) == one(elty)
            end
        end

        @testset "constructor" begin
            for (x, y) in ((d, dl), (GenericArray(d), GenericArray(dl)))
                ST = (SymTridiagonal(x, y))::SymTridiagonal{elty, typeof(x), typeof(y)}
                @test ST == Matrix(ST)
                @test ST.dv === x
                @test ST.ev === y
                TT = (Tridiagonal(y, x, y))::Tridiagonal{elty, typeof(y), typeof(x), typeof(y)}
                @test TT == Matrix(TT)
                @test TT.dl === y
                @test TT.d  === x
                @test TT.du === y
            end
            ST = SymTridiagonal{elty}([1,2,3,4], [1,2,3])
            @test eltype(ST) == elty
            @test SymTridiagonal{elty, Vector{elty}, Vector{elty}}(ST) === ST
            @test SymTridiagonal{Int64, Vector{Int64}, Vector{Int64}}(ST) isa SymTridiagonal{Int64, Vector{Int64}, Vector{Int64}}
            TT = Tridiagonal{elty}([1,2,3], [1,2,3,4], [1,2,3])
            @test eltype(TT) == elty
            ST = SymTridiagonal{elty,Vector{elty},Vector{elty}}(d, GenericArray(dl))
            @test isa(ST, SymTridiagonal{elty,Vector{elty},Vector{elty}})
            TT = Tridiagonal{elty,Vector{elty},Vector{elty},Vector{elty}}(GenericArray(dl), d, GenericArray(dl))
            @test isa(TT, Tridiagonal{elty,Vector{elty},Vector{elty},Vector{elty}})
            @test SymTridiagonal(d, GenericArray(dl)) == SymTridiagonal(d, dl)
            @test SymTridiagonal(GenericArray(d), dl) == SymTridiagonal(d, dl)
            @test Tridiagonal(GenericArray(dl), d, GenericArray(dl)) == Tridiagonal(dl, d, dl)
            @test Tridiagonal(dl, GenericArray(d), dl) == Tridiagonal(dl, d, dl)
            @test SymTridiagonal{elty}(d, GenericArray(dl)) == SymTridiagonal{elty}(d, dl)
            @test Tridiagonal{elty}(GenericArray(dl), d,GenericArray(dl)) == Tridiagonal{elty}(dl, d, dl)
            STI = SymTridiagonal([1,2,3,4], [1,2,3])
            TTI = Tridiagonal([1,2,3], [1,2,3,4], [1,2,3])
            @test SymTridiagonal(STI) === STI
            @test Tridiagonal(TTI)    === TTI
            @test isa(SymTridiagonal{elty}(STI), SymTridiagonal{elty})
            @test isa(Tridiagonal{elty}(TTI), Tridiagonal{elty})
        end
        @testset "interconversion of Tridiagonal and SymTridiagonal" begin
            @test Tridiagonal(dl, d, dl) == SymTridiagonal(d, dl)
            @test SymTridiagonal(d, dl) == Tridiagonal(dl, d, dl)
            @test Tridiagonal(dl, d, du) + Tridiagonal(du, d, dl) == SymTridiagonal(2d, dl+du)
            @test SymTridiagonal(d, dl) + Tridiagonal(dl, d, du) == Tridiagonal(dl + dl, d+d, dl+du)
            @test convert(SymTridiagonal,Tridiagonal(SymTridiagonal(d, dl))) == SymTridiagonal(d, dl)
            @test Array(convert(SymTridiagonal{ComplexF32},Tridiagonal(SymTridiagonal(d, dl)))) == convert(Matrix{ComplexF32}, SymTridiagonal(d, dl))
        end
        @testset "tril/triu" begin
            zerosd = fill!(similar(d), 0)
            zerosdl = fill!(similar(dl), 0)
            zerosdu = fill!(similar(du), 0)
            @test_throws ArgumentError tril!(SymTridiagonal(d, dl), -n - 2)
            @test_throws ArgumentError tril!(SymTridiagonal(d, dl), n)
            @test_throws ArgumentError tril!(Tridiagonal(dl, d, du), -n - 2)
            @test_throws ArgumentError tril!(Tridiagonal(dl, d, du), n)
            @test tril(SymTridiagonal(d,dl))    == Tridiagonal(dl,d,zerosdl)
            @test tril(SymTridiagonal(d,dl),1)  == Tridiagonal(dl,d,dl)
            @test tril(SymTridiagonal(d,dl),-1) == Tridiagonal(dl,zerosd,zerosdl)
            @test tril(SymTridiagonal(d,dl),-2) == Tridiagonal(zerosdl,zerosd,zerosdl)
            @test tril(Tridiagonal(dl,d,du))    == Tridiagonal(dl,d,zerosdu)
            @test tril(Tridiagonal(dl,d,du),1)  == Tridiagonal(dl,d,du)
            @test tril(Tridiagonal(dl,d,du),-1) == Tridiagonal(dl,zerosd,zerosdu)
            @test tril(Tridiagonal(dl,d,du),-2) == Tridiagonal(zerosdl,zerosd,zerosdu)

            @test_throws ArgumentError triu!(SymTridiagonal(d, dl), -n)
            @test_throws ArgumentError triu!(SymTridiagonal(d, dl), n + 2)
            @test_throws ArgumentError triu!(Tridiagonal(dl, d, du), -n)
            @test_throws ArgumentError triu!(Tridiagonal(dl, d, du), n + 2)
            @test triu(SymTridiagonal(d,dl))    == Tridiagonal(zerosdl,d,dl)
            @test triu(SymTridiagonal(d,dl),-1) == Tridiagonal(dl,d,dl)
            @test triu(SymTridiagonal(d,dl),1)  == Tridiagonal(zerosdl,zerosd,dl)
            @test triu(SymTridiagonal(d,dl),2)  == Tridiagonal(zerosdl,zerosd,zerosdl)
            @test triu(Tridiagonal(dl,d,du))    == Tridiagonal(zerosdl,d,du)
            @test triu(Tridiagonal(dl,d,du),-1) == Tridiagonal(dl,d,du)
            @test triu(Tridiagonal(dl,d,du),1)  == Tridiagonal(zerosdl,zerosd,du)
            @test triu(Tridiagonal(dl,d,du),2)  == Tridiagonal(zerosdl,zerosd,zerosdu)

            @test !istril(SymTridiagonal(d,dl))
            @test istril(SymTridiagonal(d,zerosdl))
            @test !istril(SymTridiagonal(d,dl),-2)
            @test !istriu(SymTridiagonal(d,dl))
            @test istriu(SymTridiagonal(d,zerosdl))
            @test !istriu(SymTridiagonal(d,dl),2)
            @test istriu(Tridiagonal(zerosdl,d,du))
            @test !istriu(Tridiagonal(dl,d,zerosdu))
            @test istriu(Tridiagonal(zerosdl,zerosd,du),1)
            @test !istriu(Tridiagonal(dl,d,zerosdu),2)
            @test istril(Tridiagonal(dl,d,zerosdu))
            @test !istril(Tridiagonal(zerosdl,d,du))
            @test istril(Tridiagonal(dl,zerosd,zerosdu),-1)
            @test !istril(Tridiagonal(dl,d,zerosdu),-2)

            @test isdiag(SymTridiagonal(d,zerosdl))
            @test !isdiag(SymTridiagonal(d,dl))
            @test isdiag(Tridiagonal(zerosdl,d,zerosdu))
            @test !isdiag(Tridiagonal(dl,d,zerosdu))
            @test !isdiag(Tridiagonal(zerosdl,d,du))
            @test !isdiag(Tridiagonal(dl,d,du))
        end

        @testset "iszero and isone" begin
            Tzero = Tridiagonal(zeros(elty, 9), zeros(elty, 10), zeros(elty, 9))
            Tone = Tridiagonal(zeros(elty, 9), ones(elty, 10), zeros(elty, 9))
            Tmix = Tridiagonal(zeros(elty, 9), zeros(elty, 10), zeros(elty, 9))
            Tmix[end, end] = one(elty)

            Szero = SymTridiagonal(zeros(elty, 10), zeros(elty, 9))
            Sone = SymTridiagonal(ones(elty, 10), zeros(elty, 9))
            Smix = SymTridiagonal(zeros(elty, 10), zeros(elty, 9))
            Smix[end, end] = one(elty)

            @test iszero(Tzero)
            @test !isone(Tzero)
            @test !iszero(Tone)
            @test isone(Tone)
            @test !iszero(Tmix)
            @test !isone(Tmix)

            @test iszero(Szero)
            @test !isone(Szero)
            @test !iszero(Sone)
            @test isone(Sone)
            @test !iszero(Smix)
            @test !isone(Smix)
        end

        @testset for mat_type in (Tridiagonal, SymTridiagonal)
            A = mat_type == Tridiagonal ? mat_type(dl, d, du) : mat_type(d, dl)
            fA = map(elty <: Complex ? ComplexF64 : Float64, Array(A))
            @testset "similar, size, and copyto!" begin
                B = similar(A)
                @test size(B) == size(A)
                if mat_type == Tridiagonal # doesn't work for SymTridiagonal yet
                    copyto!(B, A)
                    @test B == A
                end
                @test isa(similar(A), mat_type{elty})
                @test isa(similar(A, Int), mat_type{Int})
                @test size(A, 3) == 1
                @test size(A, 1) == n
                @test size(A) == (n, n)
                @test_throws ArgumentError size(A, 0)
            end
            @testset "getindex" begin
                @test_throws BoundsError A[n + 1, 1]
                @test_throws BoundsError A[1, n + 1]
                @test A[1, n] == convert(elty, 0.0)
                @test A[1, 1] == d[1]
            end
            @testset "setindex!" begin
                @test_throws BoundsError A[n + 1, 1] = 0 # test bounds check
                @test_throws BoundsError A[1, n + 1] = 0 # test bounds check
                @test_throws ArgumentError A[1, 3]   = 1 # test assignment off the main/sub/super diagonal
                if mat_type == Tridiagonal
                    @test (A[3, 3] = A[3, 3]; A == fA) # test assignment on the main diagonal
                    @test (A[3, 2] = A[3, 2]; A == fA) # test assignment on the subdiagonaldataonal
                    @test (A[2, 3] = A[2, 3]; A == fA) # test assignment on the superdiagonal
                    @test ((A[1, 3] = 0) == 0; A == fA) # test zero assignment off the main/sub/super diagonal
                else # mat_type is SymTridiagonal
                    @test ((A[3, 3] = A[3, 3]) == A[3, 3]; A == fA) # test assignment on the main diagonal
                    @test_throws ArgumentError A[3, 2] = 1 # test assignment on the subdiagonaldataonal
                    @test_throws ArgumentError A[2, 3] = 1 # test assignment on the superdiagonal
                end
            end
            @testset "diag" begin
                @test (@inferred diag(A))::typeof(d) == d
                @test (@inferred diag(A, 0))::typeof(d) == d
                @test (@inferred diag(A, 1))::typeof(d) == (mat_type == Tridiagonal ? du : dl)
                @test (@inferred diag(A, -1))::typeof(d) == dl
                @test (@inferred diag(A, n-1))::typeof(d) == zeros(elty, 1)
                @test_throws ArgumentError diag(A, -n - 1)
                @test_throws ArgumentError diag(A, n + 1)
                GA = mat_type == Tridiagonal ? mat_type(GenericArray.((dl, d, du))...) : mat_type(GenericArray.((d, dl))...)
                @test (@inferred diag(GA))::typeof(GenericArray(d)) == GenericArray(d)
                @test (@inferred diag(GA, -1))::typeof(GenericArray(d)) == GenericArray(dl)
            end
            @testset "Idempotent tests" begin
                for func in (conj, transpose, adjoint)
                    @test func(func(A)) == A
                end
            end
            if elty != Int
                @testset "Simple unary functions" begin
                    for func in (det, inv)
                        @test func(A) ≈ func(fA) atol=n^2*sqrt(eps(real(one(elty))))
                    end
                end
            end
            ds = mat_type == Tridiagonal ? (dl, d, du) : (d, dl)
            for f in (real, imag)
                @test f(A)::mat_type == mat_type(map(f, ds)...)
            end
            # if elty <: Real
            #     for f in (round, trunc, floor, ceil)
            #         fds = [f.(d) for d in ds]
            #         @test f.(A)::mat_type == mat_type(fds...)
            #         @test f.(Int, A)::mat_type == f.(Int, fA)
            #     end
            # end
            # fds = [abs.(d) for d in ds]
            # @test abs.(A)::mat_type == mat_type(fds...)
            @testset "Multiplication with strided matrix/vector" begin
                @test (x = fill(1.,n); A*x ≈ Array(A)*x)
                @test (X = fill(1.,n,2); A*X ≈ Array(A)*X)
            end
            @testset "Binary operations" begin
                B = mat_type == Tridiagonal ? mat_type(a, b, c) : mat_type(b, a)
                fB = map(elty <: Complex ? ComplexF64 : Float64, Array(B))
                for op in (+, -, *)
                    @test Array(op(A, B)) ≈ op(fA, fB)
                end
                α = rand(elty)
                @test Array(α*A) ≈ α*Array(A)
                @test Array(A*α) ≈ Array(A)*α
                @test Array(A/α) ≈ Array(A)/α

                @testset "Matmul with Triangular types" begin
                    @test A*LinearAlgebra.UnitUpperTriangular(Matrix(1.0I, n, n)) ≈ fA
                    @test A*LinearAlgebra.UnitLowerTriangular(Matrix(1.0I, n, n)) ≈ fA
                    @test A*UpperTriangular(Matrix(1.0I, n, n)) ≈ fA
                    @test A*LowerTriangular(Matrix(1.0I, n, n)) ≈ fA
                end
                @testset "mul! errors" begin
                    Cnn, Cnm, Cmn = Matrix{elty}.(undef, ((n,n), (n,n+1), (n+1,n)))
                    @test_throws DimensionMismatch LinearAlgebra.mul!(Cnn,A,Cnm)
                    @test_throws DimensionMismatch LinearAlgebra.mul!(Cnn,A,Cmn)
                    @test_throws DimensionMismatch LinearAlgebra.mul!(Cnn,B,Cmn)
                    @test_throws DimensionMismatch LinearAlgebra.mul!(Cmn,B,Cnn)
                    @test_throws DimensionMismatch LinearAlgebra.mul!(Cnm,B,Cnn)
                end
            end
            @testset "Negation" begin
                mA = -A
                @test mA isa mat_type
                @test -mA == A
            end
            if mat_type == SymTridiagonal
                @testset "Tridiagonal/SymTridiagonal mixing ops" begin
                    B = convert(Tridiagonal{elty}, A)
                    @test B == A
                    @test B + A == A + B
                    @test B - A == A - B
                end
            end
            @testset "generalized dot" begin
                x = fill(convert(elty, 1), n)
                y = fill(convert(elty, 1), n)
                @test dot(x, A, y) ≈ dot(A'x, y)
            end
        end
    end

    @testset "SymTridiagonal block matrix" begin
        M = [1 2; 2 4]
        n = 5
        A = SymTridiagonal(fill(M, n), fill(M, n-1))
        @test @inferred A[1,1] == Symmetric(M)
        @test @inferred A[1,2] == M
        @test @inferred A[2,1] == transpose(M)
        @test @inferred diag(A, 1) == fill(M, n-1)
        @test @inferred diag(A, 0) == fill(Symmetric(M), n)
        @test @inferred diag(A, -1) == fill(transpose(M), n-1)
        @test_throws ArgumentError diag(A, -2)
        @test_throws ArgumentError diag(A, 2)
        @test_throws ArgumentError diag(A, n+1)
        @test_throws ArgumentError diag(A, -n-1)
    end

    @testset "Issue 12068" begin
        @test SymTridiagonal([1, 2], [0])^3 == [1 0; 0 8]
    end

    @testset "convert for SymTridiagonal" begin
        STF32 = SymTridiagonal{Float32}(fill(1f0, 5), fill(1f0, 4))
        @test convert(SymTridiagonal{Float64}, STF32)::SymTridiagonal{Float64} == STF32
        @test convert(AbstractMatrix{Float64}, STF32)::SymTridiagonal{Float64} == STF32
    end

    @testset "constructors from matrix" begin
        @test SymTridiagonal([1 2 3; 2 5 6; 0 6 9]) == [1 2 0; 2 5 6; 0 6 9]
        @test Tridiagonal([1 2 3; 4 5 6; 7 8 9]) == [1 2 0; 4 5 6; 0 8 9]
    end

    @testset "constructors with range and other abstract vectors" begin
        @test SymTridiagonal(1:3, 1:2) == [1 1 0; 1 2 2; 0 2 3]
        @test Tridiagonal(4:5, 1:3, 1:2) == [1 1 0; 4 2 2; 0 5 3]
    end

    @testset "Issue #26994 (and the empty case)" begin
        T = SymTridiagonal([1.0],[3.0])
        x = ones(1)
        @test T*x == ones(1)
        @test SymTridiagonal(ones(0), ones(0)) * ones(0, 2) == ones(0, 2)
    end

    @testset "Issue 29630" begin
        function central_difference_discretization(N; dfunc = x -> 12x^2 - 2N^2,
                                                dufunc = x -> N^2 + 4N*x,
                                                dlfunc = x -> N^2 - 4N*x,
                                                bfunc = x -> 114ℯ^-x * (1 + 3x),
                                                b0 = 0, bf = 57/ℯ,
                                                x0 = 0, xf = 1)
            h = 1/N
            d, du, dl, b = map(dfunc, (x0+h):h:(xf-h)), map(dufunc, (x0+h):h:(xf-2h)),
                        map(dlfunc, (x0+2h):h:(xf-h)), map(bfunc, (x0+h):h:(xf-h))
            b[1] -= dlfunc(x0)*b0     # subtract the boundary term
            b[end] -= dufunc(xf)*bf   # subtract the boundary term
            Tridiagonal(dl, d, du), b
        end

        A90, b90 = central_difference_discretization(90)

        @test A90\b90 ≈ inv(A90)*b90
    end

    @testset "sum, mapreduce" begin
        T = Tridiagonal([1,2], [1,2,3], [7,8])
        Tdense = Matrix(T)
        S = SymTridiagonal([1,2,3], [1,2])
        Sdense = Matrix(S)
        @test sum(T) == 24
        @test sum(S) == 12
        @test_throws ArgumentError sum(T, dims=0)
        @test sum(T, dims=1) == sum(Tdense, dims=1)
        @test sum(T, dims=2) == sum(Tdense, dims=2)
        @test sum(T, dims=3) == sum(Tdense, dims=3)
        @test typeof(sum(T, dims=1)) == typeof(sum(Tdense, dims=1))
        @test mapreduce(one, min, T, dims=1) == mapreduce(one, min, Tdense, dims=1)
        @test mapreduce(one, min, T, dims=2) == mapreduce(one, min, Tdense, dims=2)
        @test mapreduce(one, min, T, dims=3) == mapreduce(one, min, Tdense, dims=3)
        @test typeof(mapreduce(one, min, T, dims=1)) == typeof(mapreduce(one, min, Tdense, dims=1))
        @test mapreduce(zero, max, T, dims=1) == mapreduce(zero, max, Tdense, dims=1)
        @test mapreduce(zero, max, T, dims=2) == mapreduce(zero, max, Tdense, dims=2)
        @test mapreduce(zero, max, T, dims=3) == mapreduce(zero, max, Tdense, dims=3)
        @test typeof(mapreduce(zero, max, T, dims=1)) == typeof(mapreduce(zero, max, Tdense, dims=1))
        @test_throws ArgumentError sum(S, dims=0)
        @test sum(S, dims=1) == sum(Sdense, dims=1)
        @test sum(S, dims=2) == sum(Sdense, dims=2)
        @test sum(S, dims=3) == sum(Sdense, dims=3)
        @test typeof(sum(S, dims=1)) == typeof(sum(Sdense, dims=1))
        @test mapreduce(one, min, S, dims=1) == mapreduce(one, min, Sdense, dims=1)
        @test mapreduce(one, min, S, dims=2) == mapreduce(one, min, Sdense, dims=2)
        @test mapreduce(one, min, S, dims=3) == mapreduce(one, min, Sdense, dims=3)
        @test typeof(mapreduce(one, min, S, dims=1)) == typeof(mapreduce(one, min, Sdense, dims=1))
        @test mapreduce(zero, max, S, dims=1) == mapreduce(zero, max, Sdense, dims=1)
        @test mapreduce(zero, max, S, dims=2) == mapreduce(zero, max, Sdense, dims=2)
        @test mapreduce(zero, max, S, dims=3) == mapreduce(zero, max, Sdense, dims=3)
        @test typeof(mapreduce(zero, max, S, dims=1)) == typeof(mapreduce(zero, max, Sdense, dims=1))

        T = Tridiagonal(Int[], Int[], Int[])
        Tdense = Matrix(T)
        S = SymTridiagonal(Int[], Int[])
        Sdense = Matrix(S)
        @test sum(T) == 0
        @test sum(S) == 0
        @test_throws ArgumentError sum(T, dims=0)
        @test sum(T, dims=1) == sum(Tdense, dims=1)
        @test sum(T, dims=2) == sum(Tdense, dims=2)
        @test sum(T, dims=3) == sum(Tdense, dims=3)
        @test typeof(sum(T, dims=1)) == typeof(sum(Tdense, dims=1))
        @test_throws ArgumentError sum(S, dims=0)
        @test sum(S, dims=1) == sum(Sdense, dims=1)
        @test sum(S, dims=2) == sum(Sdense, dims=2)
        @test sum(S, dims=3) == sum(Sdense, dims=3)
        @test typeof(sum(S, dims=1)) == typeof(sum(Sdense, dims=1))

        T = Tridiagonal(Int[], Int[2], Int[])
        Tdense = Matrix(T)
        S = SymTridiagonal(Int[2], Int[])
        Sdense = Matrix(S)
        @test sum(T) == 2
        @test sum(S) == 2
        @test_throws ArgumentError sum(T, dims=0)
        @test sum(T, dims=1) == sum(Tdense, dims=1)
        @test sum(T, dims=2) == sum(Tdense, dims=2)
        @test sum(T, dims=3) == sum(Tdense, dims=3)
        @test typeof(sum(T, dims=1)) == typeof(sum(Tdense, dims=1))
        @test_throws ArgumentError sum(S, dims=0)
        @test sum(S, dims=1) == sum(Sdense, dims=1)
        @test sum(S, dims=2) == sum(Sdense, dims=2)
        @test sum(S, dims=3) == sum(Sdense, dims=3)
        @test typeof(sum(S, dims=1)) == typeof(sum(Sdense, dims=1))
    end

    @testset "Issue #28994 (sum of Tridigonal and UniformScaling)" begin
        dl = [1., 1.]
        d = [-2., -2., -2.]
        T = Tridiagonal(dl, d, dl)
        S = SymTridiagonal(T)

        @test diag(T + 2I) == zero(d)
        @test diag(S + 2I) == zero(d)
    end

    @testset "convert Tridiagonal to SymTridiagonal error" begin
        du = rand(Float64, 4)
        d  = rand(Float64, 5)
        dl = rand(Float64, 4)
        T = Tridiagonal(dl, d, du)
        @test_throws ArgumentError SymTridiagonal{Float32}(T)
    end

    @testset "eigen" begin
        A = SymTridiagonal(Zeros(5), 1:4)
        Ã = LinearAlgebra.SymTridiagonal(zeros(5), collect(1.0:4))
        @test eigvals(A) == eigvals(Ã)
        @test eigen(A) == eigen(Ã)
    end

    @testset "Mul" begin
        A = randn(5,5)
        B = randn(5,5)
        @test SymTridiagonal(ApplyArray(*, A, B)) ≈ SymTridiagonal(A*B)
    end

    @testset "Broadcast" begin
        
    end
end # testset

end # module TestTridiagonal
