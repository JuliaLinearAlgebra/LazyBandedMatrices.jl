using LazyBandedMatrices, BlockBandedMatrices, BandedMatrices, LazyArrays, 
            ArrayLayouts, MatrixFactorizations, LinearAlgebra, Random, Test
import LazyArrays: Applied, resizedata!, FillLayout, MulAddStyle
import LazyBandedMatrices: MulBandedLayout, VcatBandedMatrix, BroadcastBandedLayout
import BandedMatrices: BandedStyle, _BandedMatrix, AbstractBandedMatrix

Random.seed!(0)

struct PseudoBandedMatrix{T} <: AbstractMatrix{T}
    data::Array{T}
    l::Int
    u::Int
end

Base.size(A::PseudoBandedMatrix) = size(A.data)
function Base.getindex(A::PseudoBandedMatrix, j::Int, k::Int)
    l, u = bandwidths(A)
    if -l ≤ k-j ≤ u
        A.data[j, k]
    else
        zero(eltype(A.data))
    end
end
function Base.setindex!(A::PseudoBandedMatrix, v, j::Int, k::Int)
    l, u = bandwidths(A)
    if -l ≤ k-j ≤ u
        A.data[j, k] = v
    else
        error("out of band.")
    end
end

struct PseudoBandedLayout <: AbstractBandedLayout end
Base.BroadcastStyle(::Type{<:PseudoBandedMatrix}) = BandedStyle()
BandedMatrices.MemoryLayout(::Type{<:PseudoBandedMatrix}) = PseudoBandedLayout()
BandedMatrices.isbanded(::PseudoBandedMatrix) = true
BandedMatrices.bandwidths(A::PseudoBandedMatrix) = (A.l , A.u)
BandedMatrices.inbands_getindex(A::PseudoBandedMatrix, j::Int, k::Int) = A.data[j, k]
BandedMatrices.inbands_setindex!(A::PseudoBandedMatrix, v, j::Int, k::Int) = setindex!(A.data, v, j, k)
LinearAlgebra.fill!(A::PseudoBandedMatrix, v) = fill!(A.data,v)

@testset "Mul" begin
    A = PseudoBandedMatrix(rand(5, 4), 1, 2)
    B = PseudoBandedMatrix(rand(4, 4), 2, 3)
    C = PseudoBandedMatrix(zeros(5, 4), 3, 4)
    D = zeros(5, 4)

    @test (C .= Mul(A, B)) ≈ (D .= Mul(A, B)) ≈ A*B
end

@testset "Vcat Zeros special case" begin
    A = _BandedMatrix((1:10)', 10, -1,1)
    x = Vcat(1:3, Zeros(10-3))
    @test A*x isa Vcat{Float64,1,<:Tuple{<:Vector,<:Zeros}}
    @test length((A*x).args[1]) == length(x.args[1]) + bandwidth(A,1) == 2
    @test A*x == A*Vector(x)

    A = _BandedMatrix(randn(3,10), 10, 1,1)
    x = Vcat(randn(10), Zeros(0))
    @test A*x isa Vcat{Float64,1,<:Tuple{<:Vector,<:Zeros}}
    @test length((A*x).args[1]) == 10
    @test A*x ≈ A*Vector(x)
end

@testset "MulMatrix" begin
    A = brand(6,5,0,1)
    B = brand(5,5,1,0)
    M = ApplyArray(*,A,B)

    @test isbanded(M) && isbanded(Applied(M))
    @test bandwidths(M) == bandwidths(Applied(M))
    @test BandedMatrix(M) == A*B
    @test MemoryLayout(typeof(M)) isa MulBandedLayout
    @test colsupport(M,1) == colsupport(Applied(M),1) == 1:2
    @test rowsupport(M,1) == rowsupport(Applied(M),1) == 1:2

    @test Base.BroadcastStyle(typeof(M)) isa BandedStyle
    @test M .+ A isa BandedMatrix

    V = view(M,1:4,1:4)
    @test bandwidths(V) == (1,1)
    @test MemoryLayout(typeof(V)) == MemoryLayout(typeof(M))
    @test M[1:4,1:4] isa BandedMatrix

    A = brand(5,5,0,1)
    B = brand(6,5,1,0)
    @test_throws DimensionMismatch ApplyArray(*,A,B)

    A = brand(6,5,0,1)
    B = brand(5,5,1,0)
    C = brand(5,6,2,2)
    M = Mul(A,B,C)
    @test @inferred(eltype(M)) == Float64
    @test bandwidths(M) == (3,3)
    @test M[1,1] ≈ (A*B*C)[1,1]

    M = @inferred(ApplyArray(*,A,B,C))
    @test @inferred(eltype(M)) == Float64
    @test bandwidths(M) == (3,3)
    @test BandedMatrix(M) ≈ A*B*C

    M = ApplyArray(*, A, Zeros(5))
    @test colsupport(M,1) == colsupport(Applied(M),1)
    @test_skip colsupport(M,1) == 1:0
end

@testset "Cat" begin
    A = brand(6,5,2,1)
    H = Hcat(A,A)
    @test H[1,1] == applied(hcat,A,A)[1,1] == A[1,1]
    @test isbanded(H)
    @test bandwidths(H) == (2,6)
    @test BandedMatrix(H) == BandedMatrix(H,(2,6)) == hcat(A,A) == hcat(A,Matrix(A)) == 
            hcat(Matrix(A),A) == hcat(Matrix(A),Matrix(A))
    @test hcat(A,A) isa BandedMatrix
    @test hcat(A,Matrix(A)) isa Matrix
    @test hcat(Matrix(A),A) isa Matrix
    @test isone.(H) isa BandedMatrix
    @test bandwidths(isone.(H)) == (2,6)
    @test Base.replace_in_print_matrix(H,4,1,"0") == "⋅"

    H = Hcat(A,A,A)
    @test isbanded(H)
    @test bandwidths(H) == (2,11)
    @test BandedMatrix(H) == hcat(A,A,A) == hcat(A,Matrix(A),A) == hcat(Matrix(A),A,A) == 
            hcat(Matrix(A),Matrix(A),A) == hcat(Matrix(A),Matrix(A),Matrix(A))
    @test hcat(A,A,A) isa BandedMatrix 
    @test isone.(H) isa BandedMatrix
    @test bandwidths(isone.(H)) == (2,11)
    
    V = Vcat(A,A)
    @test V isa VcatBandedMatrix
    @test isbanded(V)
    @test bandwidths(V) == (8,1)
    @test BandedMatrix(V) == vcat(A,A) == vcat(A,Matrix(A)) == vcat(Matrix(A),A) == vcat(Matrix(A),Matrix(A))
    @test vcat(A,A) isa BandedMatrix
    @test vcat(A,Matrix(A)) isa Matrix
    @test vcat(Matrix(A),A) isa Matrix
    @test isone.(V) isa BandedMatrix
    @test bandwidths(isone.(V)) == (8,1)
    @test Base.replace_in_print_matrix(V,1,3,"0") == "⋅"

    V = Vcat(A,A,A)
    @test bandwidths(V) == (14,1)
    @test BandedMatrix(V) == vcat(A,A,A) == vcat(A,Matrix(A),A) == vcat(Matrix(A),A,A) == 
            vcat(Matrix(A),Matrix(A),A) == vcat(Matrix(A),Matrix(A),Matrix(A))
    @test vcat(A,A,A) isa BandedMatrix 
    @test isone.(V) isa BandedMatrix
    @test bandwidths(isone.(V)) == (14,1)
end

@testset "BroadcastMatrix" begin
    A = BroadcastMatrix(*, brand(5,5,1,2), brand(5,5,2,1))
    @test eltype(A) == Float64
    @test bandwidths(A) == (1,1)
    @test LazyArrays.colsupport(A, 1) == 1:2
    @test A == broadcast(*, A.args...)

    B = BroadcastMatrix(+, brand(5,5,1,2), 2)
    @test B == broadcast(+, B.args...)

    C = BroadcastMatrix(+, brand(5,5,1,2), brand(5,5,3,1))
    @test bandwidths(C) == (3,2)
    @test MemoryLayout(typeof(C)) == BroadcastBandedLayout{typeof(+)}()
    @test isbanded(C) == true
    @test BandedMatrix(C) == C
end

@testset "Cache" begin
    A = _BandedMatrix(Fill(1,3,10_000), 10_000, 1, 1)
    C = cache(A);
    @test C.data isa BandedMatrix{Int,Matrix{Int},Base.OneTo{Int}}
    @test colsupport(C,1) == rowsupport(C,1) == 1:2
    @test bandwidths(C) == bandwidths(A) == (1,1)
    @test isbanded(C) 
    resizedata!(C,1,1);
    @test C[1:10,1:10] == A[1:10,1:10]
    @test C[1:10,1:10] isa BandedMatrix
end

@testset "NaN Bug" begin
    C = BandedMatrix{Float64}(undef, (1,2), (0,2)); C.data .= NaN;
    A = brand(1,1,0,1)
    B = brand(1,2,0,2)
    C .= Mul(A,B)
    @test C == A*B

    C.data .= NaN
    C .= @~ 1.0 * A*B + 0.0 * C
    @test C == A*B
end

@testset "Applied" begin
    A = brand(5,5,1,2)
    @test applied(*,Symmetric(A),A) isa Applied{MulAddStyle}
    B = apply(*,A,A,A)
    @test B isa BandedMatrix
    @test all(B .=== (A*A)*A)
    @test bandwidths(B) == (3,4)
end

@testset "QL tests" begin
    for T in (Float64,ComplexF64,Float32,ComplexF32)
        A=brand(T,10,10,3,2)
        Q,L=ql(A)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        @test Matrix(Q)*Matrix(L) ≈ A
        b=rand(T,10)
        @test mul!(similar(b),Q,mul!(similar(b),Q',b)) ≈ b
        for j=1:size(A,2)
            @test Q' * A[:,j] ≈ L[:,j]
        end

        A=brand(T,14,10,3,2)
        Q,L=ql(A)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        @test_broken Matrix(Q)*Matrix(L) ≈ A

        for k=1:size(A,1),j=1:size(A,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        A=brand(T,10,14,3,2)
        Q,L=ql(A)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        @test Matrix(Q)*Matrix(L) ≈ A

        for k=1:size(Q,1),j=1:size(Q,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        A=brand(T,10,14,3,6)
        Q,L=ql(A)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        @test Matrix(Q)*Matrix(L) ≈ A

        for k=1:size(Q,1),j=1:size(Q,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        A=brand(T,100,100,3,4)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        b=rand(T,100)
        @test ql(A)\b ≈ Matrix(A)\b
        b=rand(T,100,2)
        @test ql(A)\b ≈ Matrix(A)\b
        @test_throws DimensionMismatch ql(A) \ randn(3)
        @test_throws DimensionMismatch ql(A).Q'randn(3)

        A=brand(T,102,100,3,4)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        b=rand(T,102)
        @test_broken ql(A)\b ≈ Matrix(A)\b
        b=rand(T,102,2)
        @test_broken ql(A)\b ≈ Matrix(A)\b
        @test_throws DimensionMismatch ql(A) \ randn(3)
        @test_throws DimensionMismatch ql(A).Q'randn(3)

        A=brand(T,100,102,3,4)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        b=rand(T,100)
        @test_broken ql(A)\b ≈ Matrix(A)\b

        A = Tridiagonal(randn(T,99), randn(T,100), randn(T,99))
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        b=rand(T,100)
        @test ql(A)\b ≈ Matrix(A)\b
        b=rand(T,100,2)
        @test ql(A)\b ≈ Matrix(A)\b
        @test_throws DimensionMismatch ql(A) \ randn(3)
        @test_throws DimensionMismatch ql(A).Q'randn(3)
    end

    @testset "lmul!/rmul!" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            A = brand(T,100,100,3,4)
            Q,R = qr(A)
            x = randn(T,100)
            b = randn(T,100,2)
            @test lmul!(Q, copy(x)) ≈ Matrix(Q)*x
            @test lmul!(Q, copy(b)) ≈ Matrix(Q)*b
            @test lmul!(Q', copy(x)) ≈ Matrix(Q)'*x
            @test lmul!(Q', copy(b)) ≈ Matrix(Q)'*b
            c = randn(T,2,100)
            @test rmul!(copy(c), Q) ≈ c*Matrix(Q)
            @test rmul!(copy(c), Q') ≈ c*Matrix(Q')

            A = brand(T,100,100,3,4)
            Q,L = ql(A)
            x = randn(T,100)
            b = randn(T,100,2)
            @test lmul!(Q, copy(x)) ≈ Matrix(Q)*x
            @test lmul!(Q, copy(b)) ≈ Matrix(Q)*b
            @test lmul!(Q', copy(x)) ≈ Matrix(Q)'*x
            @test lmul!(Q', copy(b)) ≈ Matrix(Q)'*b
            c = randn(T,2,100)
            @test rmul!(copy(c), Q) ≈ c*Matrix(Q)
            @test rmul!(copy(c), Q') ≈ c*Matrix(Q')
        end
    end

    @testset "Mixed types" begin
        A=brand(10,10,3,2)
        b=rand(ComplexF64,10)
        Q,L=ql(A)
        @test L\(Q'*b) ≈ ql(A)\b ≈ Matrix(A)\b
        @test Q*L ≈ A

        A=brand(ComplexF64,10,10,3,2)
        b=rand(10)
        Q,L=ql(A)
        @test Q*L ≈ A
        @test L\(Q'*b) ≈ ql(A)\b ≈ Matrix(A)\b

        A = BandedMatrix{Int}(undef, (2,1), (4,4))
        A.data .= 1:length(A.data)
        Q, L = ql(A)
        @test_broken Q*L ≈ A
    end
end

@testset "kron" begin
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


struct FiniteDifference{T} <: AbstractBandedMatrix{T}
    n::Int
end

FiniteDifference(n) = FiniteDifference{Float64}(n)

Base.getindex(F::FiniteDifference{T}, k::Int, j::Int) where T =
    if k == j
        -2*one(T)*F.n^2
    elseif abs(k-j) == 1
        one(T)*F.n^2
    else
        zero(T)
    end

BandedMatrices.bandwidths(F::FiniteDifference) = (1,1)
Base.size(F::FiniteDifference) = (F.n,F.n)

@testset "Misc" begin
    @testset "Block banded Kron" begin
        n = 10
        h = 1/n
        D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2

        @time D_xx = BandedBlockBandedMatrix(Kron(D², Eye(n)))
        @time D_yy = BandedBlockBandedMatrix(Kron(Eye(n),D²))
        @time Δ = D_xx + D_yy

        @test Δ isa BandedBlockBandedMatrix
        @test blockbandwidths(Δ) == subblockbandwidths(Δ) == (1,1)
        @test Δ == kron(Matrix(D²), Matrix(I,n,n)) + kron(Matrix(I,n,n), Matrix(D²))

        n = 10
        D² = FiniteDifference(n)
        D̃_xx = Kron(D², Eye(n))
        @test blockbandwidths(D̃_xx) == (1,1)
        @test subblockbandwidths(D̃_xx) == (0,0)

        V = view(D̃_xx, Block(1,1))
        @test bandwidths(V) == (0,0)

        @test BandedBlockBandedMatrix(D̃_xx) ≈ D_xx

        D̃_yy = Kron(Eye(n), D²)
        @test blockbandwidths(D̃_yy) == (0,0)
        @test subblockbandwidths(D̃_yy) == (1,1)

        V = view(D̃_yy, Block(1,1))
        @test bandwidths(V) == (1,1)

        @test BandedBlockBandedMatrix(D̃_yy) ≈ D_yy
    end

    @testset "Diagonal interface" begin
        n = 10
        h = 1/n
        D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2
        D_xx = BandedBlockBandedMatrix(Kron(D², Eye(n)))

        D = Diagonal(randn(n^2))
        @test D_xx + D isa BandedBlockBandedMatrix
        @test blockbandwidths(D_xx + D) == blockbandwidths(D_xx)
        @test subblockbandwidths(D_xx + D) == subblockbandwidths(D_xx)
        @test D_xx + D == Matrix(D_xx) + D

        @test D_xx - D isa BandedBlockBandedMatrix
        @test blockbandwidths(D_xx - D) == blockbandwidths(D_xx)
        @test subblockbandwidths(D_xx - D) == subblockbandwidths(D_xx)
        @test D_xx - D == Matrix(D_xx) - D

        @test D_xx*D == Matrix(D_xx)*D
        @test D_xx*D isa BandedBlockBandedMatrix
        @test blockbandwidths(D_xx*D) == blockbandwidths(D_xx)
        @test subblockbandwidths(D_xx*D) == subblockbandwidths(D_xx)

        @test D*D_xx == D*Matrix(D_xx)
        @test D*D_xx isa BandedBlockBandedMatrix
        @test blockbandwidths(D*D_xx) == blockbandwidths(D_xx)
        @test subblockbandwidths(D*D_xx) == subblockbandwidths(D_xx)
    end
end