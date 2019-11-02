using LazyBandedMatrices, BandedMatrices, LazyArrays, ArrayLayouts, Test
import LazyArrays: Applied, resizedata!, FillLayout


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
    A = BandedMatrices._BandedMatrix((1:10)', 10, -1,1)
    x = Vcat(1:3, Zeros(10-3))
    @test A*x isa Vcat{Float64,1,<:Tuple{<:Vector,<:Zeros}}
    @test length((A*x).args[1]) == length(x.args[1]) + bandwidth(A,1) == 2
    @test A*x == A*Vector(x)

    A = BandedMatrices._BandedMatrix(randn(3,10), 10, 1,1)
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
    MemoryLayout(typeof(M))
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
    @test isbanded(H)
    @test bandwidths(H) == (2,6)
    @test BandedMatrix(H) == hcat(A,A) == hcat(A,Matrix(A)) == hcat(Matrix(A),A) == hcat(Matrix(A),Matrix(A))
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
    @test V isa BandedMatrices.VcatBandedMatrix
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
