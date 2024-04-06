# This file is based on LinearAlgebra/test/special.jl a part of Julia. License is MIT: https://julialang.org/license

module TestSpecial

using Test, SparseArrays, Random, LazyBandedMatrices
import LazyBandedMatrices: Bidiagonal, Tridiagonal, SymTridiagonal
import LinearAlgebra
import LinearAlgebra: Diagonal, UpperTriangular, LowerTriangular, triu, Symmetric
import LinearAlgebra: UniformScaling

@testset "Tri/Bidiagonal special" begin
    n= 10 #Size of matrix to test
    Random.seed!(1)

    @testset "Interconversion between special matrix types" begin
        a = [1.0:n;]
        A = Diagonal(a)
        @testset for newtype in [Diagonal, Bidiagonal, SymTridiagonal, Tridiagonal, Matrix]
            @test Matrix(convert(newtype, A)) == Matrix(A)
            @test Matrix(convert(newtype, Diagonal(GenericArray(a)))) == Matrix(A)
        end

        @testset for isupper in (true, false)
            A = Bidiagonal(a, [1.0:n-1;], ifelse(isupper, :U, :L))
            for newtype in [Bidiagonal, Tridiagonal, Matrix]
            @test Matrix(convert(newtype, A)) == Matrix(A)
            @test Matrix(newtype(A)) == Matrix(A)
            end
            @test_throws ArgumentError convert(SymTridiagonal, A)
            tritype = isupper ? UpperTriangular : LowerTriangular
            @test Matrix(tritype(A)) == Matrix(A)

            A = Bidiagonal(a, zeros(n-1), ifelse(isupper, :U, :L)) #morally Diagonal
            for newtype in [Diagonal, Bidiagonal, SymTridiagonal, Tridiagonal, Matrix]
            @test Matrix(convert(newtype, A)) == Matrix(A)
            @test Matrix(newtype(A)) == Matrix(A)
            end
            @test Matrix(tritype(A)) == Matrix(A)
        end

        A = SymTridiagonal(a, [1.0:n-1;])
        for newtype in [Tridiagonal, Matrix]
            @test Matrix(convert(newtype, A)) == Matrix(A)
        end
        for newtype in [Diagonal, Bidiagonal]
            @test_throws ArgumentError convert(newtype,A)
        end
        A = SymTridiagonal(a, zeros(n-1))
        @test Matrix(convert(Bidiagonal,A)) == Matrix(A)

        A = Tridiagonal(zeros(n-1), [1.0:n;], zeros(n-1)) #morally Diagonal
        for newtype in [Diagonal, Bidiagonal, SymTridiagonal, Matrix]
        @test Matrix(convert(newtype, A)) == Matrix(A)
        end
        A = Tridiagonal(fill(1., n-1), [1.0:n;], fill(1., n-1)) #not morally Diagonal
        for newtype in [SymTridiagonal, Matrix]
        @test Matrix(convert(newtype, A)) == Matrix(A)
        end
        for newtype in [Diagonal, Bidiagonal]
            @test_throws ArgumentError convert(newtype,A)
        end
        A = Tridiagonal(zeros(n-1), [1.0:n;], fill(1., n-1)) #not morally Diagonal
        @test Matrix(convert(Bidiagonal, A)) == Matrix(A)
        A = UpperTriangular(Tridiagonal(zeros(n-1), [1.0:n;], fill(1., n-1)))
        @test Matrix(convert(Bidiagonal, A)) == Matrix(A)
        A = Tridiagonal(fill(1., n-1), [1.0:n;], zeros(n-1)) #not morally Diagonal
        @test Matrix(convert(Bidiagonal, A)) == Matrix(A)
        A = LowerTriangular(Tridiagonal(fill(1., n-1), [1.0:n;], zeros(n-1)))
        @test Matrix(convert(Bidiagonal, A)) == Matrix(A)
        @test_throws ArgumentError convert(SymTridiagonal,A)

        A = LowerTriangular(Matrix(Diagonal(a))) #morally Diagonal
        for newtype in [Diagonal, Bidiagonal, SymTridiagonal, LowerTriangular, Matrix]
            @test Matrix(convert(newtype, A)) == Matrix(A)
        end
        A = UpperTriangular(Matrix(Diagonal(a))) #morally Diagonal
        for newtype in [Diagonal, Bidiagonal, SymTridiagonal, UpperTriangular, Matrix]
            @test Matrix(convert(newtype, A)) == Matrix(A)
        end
        A = UpperTriangular(triu(rand(n,n)))
        for newtype in [Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal]
            @test_throws ArgumentError convert(newtype,A)
        end


        # test operations/constructors (not conversions) permitted in the docs
        dl = [1., 1.]
        d = [-2., -2., -2.]
        T = Tridiagonal(dl, d, -dl)
        S = SymTridiagonal(d, dl)
        Bu = Bidiagonal(d, dl, :U)
        Bl = Bidiagonal(d, dl, :L)
        D = Diagonal(d)
        M = [-2. 0. 0.; 1. -2. 0.; -1. 1. -2.]
        U = UpperTriangular(M)
        L = LowerTriangular(Matrix(M'))

        for A in (T, S, Bu, Bl, D, U, L, M)
            Adense = Matrix(A)
            B = Symmetric(A)
            Bdense = Matrix(B)
            for (C,Cdense) in ((A,Adense), (B,Bdense))
                @test Diagonal(C) == Diagonal(Cdense)
                @test Bidiagonal(C, :U) == Bidiagonal(Cdense, :U)
                @test Bidiagonal(C, :L) == Bidiagonal(Cdense, :L)
                @test Tridiagonal(C) == Tridiagonal(Cdense)
                @test UpperTriangular(C) == UpperTriangular(Cdense)
                @test LowerTriangular(C) == LowerTriangular(Cdense)
            end
        end
    end

    @testset "Binary ops among special types" begin
        a=[1.0:n;]
        A=Diagonal(a)
        Spectypes = [Diagonal, Bidiagonal, Tridiagonal, Matrix]
        for (idx, type1) in enumerate(Spectypes)
            for type2 in Spectypes
            B = convert(type1,A)
            C = convert(type2,A)
            @test Matrix(B + C) ≈ Matrix(A + A)
            @test Matrix(B - C) ≈ Matrix(A - A)
        end
        end
        B = SymTridiagonal(a, fill(1., n-1))
        for Spectype in [Diagonal, Bidiagonal, Tridiagonal, Matrix]
            @test Matrix(B + convert(Spectype,A)) ≈ Matrix(B + A)
            @test Matrix(convert(Spectype,A) + B) ≈ Matrix(B + A)
            @test Matrix(B - convert(Spectype,A)) ≈ Matrix(B - A)
            @test Matrix(convert(Spectype,A) - B) ≈ Matrix(A - B)
        end

        C = rand(n,n)
        for TriType in [LinearAlgebra.UnitLowerTriangular, LinearAlgebra.UnitUpperTriangular, UpperTriangular, LowerTriangular]
            D = TriType(C)
            for Spectype in [Diagonal, Bidiagonal, Tridiagonal, Matrix]
                @test Matrix(D + convert(Spectype,A)) ≈ Matrix(D + A)
                @test Matrix(convert(Spectype,A) + D) ≈ Matrix(A + D)
                @test Matrix(D - convert(Spectype,A)) ≈ Matrix(D - A)
                @test Matrix(convert(Spectype,A) - D) ≈ Matrix(A - D)
            end
        end

        UpTri = UpperTriangular(rand(20,20))
        LoTri = LowerTriangular(rand(20,20))
        Diag = Diagonal(rand(20,20))
        Tridiag = Tridiagonal(rand(20, 20))
        UpBi = Bidiagonal(rand(20,20), :U)
        LoBi = Bidiagonal(rand(20,20), :L)
        Sym = SymTridiagonal(rand(20), rand(19))
        Dense = rand(20, 20)
        mats = [UpTri, LoTri, Diag, Tridiag, UpBi, LoBi, Sym, Dense]

        for op in (+,-,*)
            for A in mats
                for B in mats
                    @test (op)(A, B) ≈ (op)(Matrix(A), Matrix(B)) ≈ Matrix((op)(A, B))
                end
            end
        end
    end

    @testset "+ and - among structured matrices with different container types" begin
        diag = 1:5
        offdiag = 1:4
        uniformscalingmats = [UniformScaling(3), UniformScaling(1.0), UniformScaling(3//5), UniformScaling(ComplexF64(1.3, 3.5))]
        mats = [Diagonal(diag), Bidiagonal(diag, offdiag, 'U'), Bidiagonal(diag, offdiag, 'L'), Tridiagonal(offdiag, diag, offdiag), SymTridiagonal(diag, offdiag)]
        for T in [ComplexF64, Int64, Rational{Int64}, Float64]
            push!(mats, Diagonal(Vector{T}(diag)))
            push!(mats, Bidiagonal(Vector{T}(diag), Vector{T}(offdiag), 'U'))
            push!(mats, Bidiagonal(Vector{T}(diag), Vector{T}(offdiag), 'L'))
            push!(mats, Tridiagonal(Vector{T}(offdiag), Vector{T}(diag), Vector{T}(offdiag)))
            push!(mats, SymTridiagonal(Vector{T}(diag), Vector{T}(offdiag)))
        end

        for op in (+,*) # to do: fix when operation is - and the matrix has a range as the underlying representation and we get a step size of 0.
            for A in mats
                for B in mats
                    @test (op)(A, B) ≈ (op)(Matrix(A), Matrix(B)) ≈ Matrix((op)(A, B))
                end
            end
        end
        for op in (+,-)
            for A in mats
                for B in uniformscalingmats
                    @test (op)(A, B) ≈ (op)(Matrix(A), B) ≈ Matrix((op)(A, B))
                    @test (op)(B, A) ≈ (op)(B, Matrix(A)) ≈ Matrix((op)(B, A))
                end
            end
        end
    end


    @testset "zero and one for structured matrices" begin
        for elty in (Int64, Float64, ComplexF64)
            D = Diagonal(rand(elty, 10))
            Bu = Bidiagonal(rand(elty, 10), rand(elty, 9), 'U')
            Bl = Bidiagonal(rand(elty, 10), rand(elty, 9), 'L')
            T = Tridiagonal(rand(elty, 9),rand(elty, 10), rand(elty, 9))
            S = SymTridiagonal(rand(elty, 10), rand(elty, 9))
            mats = [D, Bu, Bl, T, S]
            for A in mats
                @test iszero(zero(A))
                @test isone(one(A))
                @test zero(A) == zero(Matrix(A))
                @test one(A) == one(Matrix(A))
            end

            @test zero(D) isa Diagonal
            @test one(D) isa Diagonal

            @test zero(Bu) isa Bidiagonal
            @test one(Bu) isa Bidiagonal
            @test zero(Bl) isa Bidiagonal
            @test one(Bl) isa Bidiagonal
            @test zero(Bu).uplo == one(Bu).uplo == Bu.uplo
            @test zero(Bl).uplo == one(Bl).uplo == Bl.uplo

            @test zero(T) isa Tridiagonal
            @test one(T) isa Tridiagonal
            @test zero(S) isa SymTridiagonal
            @test one(S) isa SymTridiagonal
        end

        # ranges
        D = Diagonal(1:10)
        Bu = Bidiagonal(1:10, 1:9, 'U')
        Bl = Bidiagonal(1:10, 1:9, 'L')
        T = Tridiagonal(1:9, 1:10, 1:9)
        S = SymTridiagonal(1:10, 1:9)
        mats = [D, Bu, Bl, T, S]
        for A in mats
            @test iszero(zero(A))
            @test isone(one(A))
            @test zero(A) == zero(Matrix(A))
            @test one(A) == one(Matrix(A))
        end

        @test zero(D) isa Diagonal
        @test one(D) isa Diagonal

        @test zero(Bu) isa Bidiagonal
        @test one(Bu) isa Bidiagonal
        @test zero(Bl) isa Bidiagonal
        @test one(Bl) isa Bidiagonal
        @test zero(Bu).uplo == one(Bu).uplo == Bu.uplo
        @test zero(Bl).uplo == one(Bl).uplo == Bl.uplo

        @test zero(T) isa Tridiagonal
        @test one(T) isa Tridiagonal
        @test zero(S) isa SymTridiagonal
        @test one(S) isa SymTridiagonal
    end

    @testset "== for structured matrices" begin
        diag = rand(10)
        offdiag = rand(9)
        D = Diagonal(rand(10))
        Bup = Bidiagonal(diag, offdiag, 'U')
        Blo = Bidiagonal(diag, offdiag, 'L')
        Bupd = Bidiagonal(diag, zeros(9), 'U')
        Blod = Bidiagonal(diag, zeros(9), 'L')
        T = Tridiagonal(offdiag, diag, offdiag)
        Td = Tridiagonal(zeros(9), diag, zeros(9))
        Tu = Tridiagonal(zeros(9), diag, offdiag)
        Tl = Tridiagonal(offdiag, diag, zeros(9))
        S = SymTridiagonal(diag, offdiag)
        Sd = SymTridiagonal(diag, zeros(9))

        mats = [D, Bup, Blo, Bupd, Blod, T, Td, Tu, Tl, S, Sd]

        for a in mats
            for b in mats
                @test (a == b) == (Matrix(a) == Matrix(b)) == (b == a) == (Matrix(b) == Matrix(a))
            end
        end
    end
end # testset

end # module TestSpecial
