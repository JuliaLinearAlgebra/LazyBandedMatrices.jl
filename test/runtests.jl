using LazyBandedMatrices, BlockBandedMatrices, BandedMatrices, LazyArrays, BlockArrays,
            ArrayLayouts, MatrixFactorizations, Random, Test
import LinearAlgebra
import LinearAlgebra: qr, rmul!, lmul!
import LazyArrays: Applied, resizedata!, FillLayout, MulStyle, arguments, colsupport, rowsupport, LazyLayout, ApplyStyle, PaddedLayout, paddeddata, call, ApplyLayout, LazyArrayStyle
import LazyBandedMatrices: VcatBandedMatrix, BroadcastBlockBandedLayout, BroadcastBandedLayout, 
                    ApplyBandedLayout, ApplyBlockBandedLayout, ApplyBandedBlockBandedLayout, BlockKron, LazyBandedLayout, BroadcastBandedBlockBandedLayout
import BandedMatrices: BandedStyle, _BandedMatrix, AbstractBandedMatrix, BandedRows, BandedColumns
import ArrayLayouts: StridedLayout, OnesLayout

Random.seed!(0)

include("test_tridiag.jl")
include("test_bidiag.jl")
include("test_special.jl")

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
ArrayLayouts.lmul!(β::Number, A::PseudoBandedMatrix) = (lmul!(β, A.data); A)
LinearAlgebra.lmul!(β::Number, A::PseudoBandedMatrix) = (lmul!(β, A.data); A)

struct MyLazyArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end


Base.size(A::MyLazyArray) = size(A.data)
Base.getindex(A::MyLazyArray, j::Int...) = A.data[j...]
LazyArrays.MemoryLayout(::Type{<:MyLazyArray}) = LazyLayout()
LinearAlgebra.factorize(A::MyLazyArray) = factorize(A.data)


@testset "LazyBlock" begin
    @test Block(5) in BroadcastVector(Block, [1,3,5])
    @test Base.broadcasted(LazyArrayStyle{1}(), Block, 1:5) ≡ Block.(1:5)
    @test Base.broadcasted(LazyArrayStyle{1}(), Int, Block.(1:5)) ≡ 1:5
    @test Base.broadcasted(LazyArrayStyle{0}(), Int, Block(1)) ≡ 1
end


@testset "Padded" begin
    @testset "Banded padded" begin
        A = _BandedMatrix((1:10)', 10, -1,1)
        x = Vcat(1:3, Zeros(10-3))
        @test MemoryLayout(x) isa PaddedLayout
        @test A*x isa Vcat{Float64,1,<:Tuple{<:Vector,<:Zeros}}
        @test length((A*x).args[1]) == length(x.args[1]) + bandwidth(A,1) == 2
        @test A*x == A*Vector(x)

        A = _BandedMatrix(randn(3,10), 10, 1,1)
        x = Vcat(randn(10), Zeros(0))
        @test A*x isa Vcat{Float64,1,<:Tuple{<:Vector,<:Zeros}}
        @test length((A*x).args[1]) == 10
        @test A*x ≈ A*Vector(x)

        A = Vcat(Zeros(1,10), brand(9,10,0,2))
        @test bandwidths(A) == (1,1)
        @test BandedMatrix(A) == Array(A) == A

        A = Hcat(Zeros(5,2), brand(5,5,1,1))
        @test bandwidths(A) == (-1,3)
        @test BandedMatrix(A) == Array(A) == A
    end
    @testset "BlockBanded and padded" begin
        A = BlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,0)); A.data .= randn.();
        c = Vcat(randn(3), Zeros(7))
        b = PseudoBlockVector(c, (axes(A,2),))
        @test MemoryLayout(b) isa PaddedLayout
        @test MemoryLayout(A*b) isa PaddedLayout
        @test MemoryLayout(A*c) isa PaddedLayout
        @test A*b ≈ A*c ≈ Matrix(A)*Vector(b)

        @test b[Block.(2:3)] isa PseudoBlockVector{Float64,<:ApplyArray}
        @test MemoryLayout(b[Block.(2:3)]) isa PaddedLayout
        @test b[Block.(2:3)] == b[2:6]
    end

    @testset "BroadcastBanded * Padded" begin
        A = BroadcastArray(*, randn(5), brand(5,5,1,2))
        @test axes(A) == (Base.OneTo(5), Base.OneTo(5))
        B = BroadcastArray(*, randn(5,5), brand(5,5,1,2))
        b = Vcat(randn(2), Zeros(3))
        @test A*b ≈ Matrix(A)b
        @test B*b ≈ Matrix(B)b
    end

    @testset "Apply * Banded" begin
        B = brand(5,5,2,1)
        A = ApplyArray(*, B, B)
        @test A * Vcat([1,2], Zeros(3)) ≈ B*B*[1,2,0,0,0]
    end

    @testset "block padded" begin
        c = PseudoBlockVector(Vcat(1, Zeros(5)), 1:3)
        @test paddeddata(c) == [1]
        @test paddeddata(c) isa PseudoBlockVector
    end
end

@testset "MulMatrix" begin
    @testset "MulBanded" begin
        A = brand(6,5,0,1)
        B = brand(5,5,1,0)

        M = ApplyArray(*, A)
        @test BandedMatrix(M) == copyto!(similar(A), M) == A

        M = ApplyArray(*,A,B)
        @test isbanded(M) && isbanded(Applied(M))
        @test bandwidths(M) == bandwidths(Applied(M))
        @test BandedMatrix(M) == A*B == copyto!(BandedMatrix(M), M)
        @test MemoryLayout(typeof(M)) isa ApplyBandedLayout{typeof(*)}
        @test arguments(M) == (A,B)
        @test call(M) == *
        @test colsupport(M,1) == colsupport(Applied(M),1) == 1:2
        @test rowsupport(M,1) == rowsupport(Applied(M),1) == 1:2

        @test Base.BroadcastStyle(typeof(M)) isa BandedStyle
        @test M .+ A isa BandedMatrix
        @test M .+ A == M .+ Matrix(A) == Matrix(A) .+ M

        V = view(M,1:4,1:4)
        @test bandwidths(V) == (1,1)
        @test MemoryLayout(typeof(V)) == MemoryLayout(typeof(M))
        @test M[1:4,1:4] isa BandedMatrix
        @test colsupport(V,1) == 1:2
        @test rowsupport(V,1) == 1:2

        MemoryLayout(view(M, [1,3], [2,3])) isa ApplyLayout{typeof(*)}

        A = brand(5,5,0,1)
        B = brand(6,5,1,0)
        @test_throws DimensionMismatch ApplyArray(*,A,B)

        A = brand(6,5,0,1)
        B = brand(5,5,1,0)
        C = brand(5,6,2,2)
        M = applied(*,A,B,C)
        @test @inferred(eltype(M)) == Float64
        @test bandwidths(M) == (3,3)
        @test M[1,1] ≈ (A*B*C)[1,1]

        M = @inferred(ApplyArray(*,A,B,C))
        @test @inferred(eltype(M)) == Float64
        @test bandwidths(M) == (3,3)
        @test BandedMatrix(M) ≈ A*B*C ≈ copyto!(BandedMatrix(M), M)

        M = ApplyArray(*, A, Zeros(5))
        @test colsupport(M,1) == colsupport(Applied(M),1)
        @test_skip colsupport(M,1) == 1:0

        @testset "inv" begin
            A = brand(6,5,0,1)
            B = brand(5,5,1,0)
            C = randn(6,2)
            M = ApplyArray(*,A,B)
            @test M \ C ≈ Matrix(M) \ C
        end
    end
    @testset "MulBlockBanded" begin
        A = BlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,0)); A.data .= randn.();
        B = BlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,1)); B.data .= randn.();
        M = ApplyMatrix(*, A, B)
        @test blockbandwidths(M) == (2,1)
        @test MemoryLayout(M) isa ApplyBlockBandedLayout{typeof(*)}
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
    @testset "Psuedo Mul" begin
        A = PseudoBandedMatrix(rand(5, 4), 1, 2)
        B = PseudoBandedMatrix(rand(4, 4), 2, 3)
        C = PseudoBandedMatrix(zeros(5, 4), 3, 4)
        D = zeros(5, 4)

        @test (C .= applied(*, A, B)) ≈ (D .= applied(*, A, B)) ≈ A*B
    end
    @testset "MulStyle" begin
        A = brand(5,5,0,1)
        B = brand(5,5,1,0)
        C = BroadcastMatrix(*, A, 2)
        M = ApplyArray(*,A,B)
        @test M^2 isa ApplyMatrix{Float64,typeof(*)}
        @test M*C isa ApplyMatrix{Float64,typeof(*)}
        @test C*M isa ApplyMatrix{Float64,typeof(*)}
    end
end

@testset "InvMatrix" begin
    D = brand(5,5,0,0)
    L = brand(5,5,2,0)
    U = brand(5,5,0,2)
    B = brand(5,5,1,2)

    @test bandwidths(ApplyMatrix(inv,D)) == (0,0)
    @test bandwidths(ApplyMatrix(inv,L)) == (4,0)
    @test bandwidths(ApplyMatrix(inv,U)) == (0,4)
    @test bandwidths(ApplyMatrix(inv,B)) == (4,4)

    @test colsupport(ApplyMatrix(inv,D) ,3) == 3:3
    @test colsupport(ApplyMatrix(inv,L), 3) == 3:5
    @test colsupport(ApplyMatrix(inv,U), 3) == 1:3
    @test colsupport(ApplyMatrix(inv,B), 3) == 1:5

    @test rowsupport(ApplyMatrix(inv,D) ,3) == 3:3
    @test rowsupport(ApplyMatrix(inv,L), 3) == 1:3
    @test rowsupport(ApplyMatrix(inv,U), 3) == 3:5
    @test rowsupport(ApplyMatrix(inv,B), 3) == 1:5

    @test bandwidths(ApplyMatrix(\,D,B)) == (1,2)
    @test bandwidths(ApplyMatrix(\,L,B)) == (4,2)
    @test bandwidths(ApplyMatrix(\,U,B)) == (1,4)
    @test bandwidths(ApplyMatrix(\,B,B)) == (4,4)

    @test colsupport(ApplyMatrix(\,D,B), 3) == 1:4
    @test colsupport(ApplyMatrix(\,L,B), 4) == 2:5
    @test colsupport(ApplyMatrix(\,U,B), 3) == 1:4
    @test colsupport(ApplyMatrix(\,B,B), 3) == 1:5

    @test rowsupport(ApplyMatrix(\,D,B), 3) == 2:5
    @test rowsupport(ApplyMatrix(\,L,B), 2) == 1:4
    @test rowsupport(ApplyMatrix(\,U,B), 3) == 2:5
    @test rowsupport(ApplyMatrix(\,B,B), 3) == 1:5
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
    @test @inferred(colsupport(H,1)) == 1:3
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
    @testset "BroadcastBanded" begin
        A = BroadcastMatrix(*, brand(5,5,1,2), brand(5,5,2,1))
        @test eltype(A) == Float64
        @test bandwidths(A) == (1,1)
        @test colsupport(A, 1) == 1:2
        @test rowsupport(A, 1) == 1:2
        @test A == broadcast(*, A.args...)
        @test MemoryLayout(typeof(A)) isa BroadcastBandedLayout{typeof(*)}

        @test MemoryLayout(typeof(A')) isa BroadcastBandedLayout{typeof(*)}
        @test bandwidths(A') == (1,1)
        @test colsupport(A',1) == rowsupport(A', 1) == 1:2
        @test A' == BroadcastArray(A') == Array(A)'

        V = view(A, 2:3, 3:5)
        @test MemoryLayout(typeof(V)) isa BroadcastBandedLayout{typeof(*)}
        @test bandwidths(V) == (1,0)
        @test colsupport(V,1) == 1:2
        @test V == BroadcastArray(V) == Array(A)[2:3,3:5]
        @test bandwidths(view(A,2:4,3:5)) == (2,0)

        V = view(A, 2:3, 3:5)'
        @test MemoryLayout(V) isa BroadcastBandedLayout{typeof(*)}
        @test bandwidths(V) == (0,1)
        @test colsupport(V,1) == 1:1
        @test V == BroadcastArray(V) == Array(A)[2:3,3:5]'

        B = BroadcastMatrix(+, brand(5,5,1,2), 2)
        @test B == broadcast(+, B.args...)

        C = BroadcastMatrix(+, brand(5,5,1,2), brand(5,5,3,1))
        @test bandwidths(C) == (3,2)
        @test MemoryLayout(C) == BroadcastBandedLayout{typeof(+)}()
        @test isbanded(C) == true
        @test BandedMatrix(C) == C == copyto!(BandedMatrix(C), C)

        D = BroadcastMatrix(*, 2, brand(5,5,1,2))
        @test bandwidths(D) == (1,2)
        @test MemoryLayout(D) == BroadcastBandedLayout{typeof(*)}()
        @test isbanded(D) == true
        @test BandedMatrix(D) == D == copyto!(BandedMatrix(D), D) == 2*D.args[2]

        @testset "band" begin
            @test A[band(0)] == Matrix(A)[band(0)]
            @test B[band(0)] == Matrix(B)[band(0)]
            @test C[band(0)] == Matrix(C)[band(0)]
            @test D[band(0)] == Matrix(D)[band(0)]
        end
    end
    @testset "BroadcastBlockBanded" begin
        A = BlockBandedMatrix(randn(6,6),1:3,1:3,(1,1))
        B = BroadcastMatrix(*, 2, A)
        @test blockbandwidths(B) == (1,1)
        @test MemoryLayout(B) == BroadcastBlockBandedLayout{typeof(*)}()
        @test BandedBlockBandedMatrix(B) == B == copyto!(BandedBlockBandedMatrix(B), B) == 2*B.args[2]

        C = BroadcastMatrix(*, A, 2)
        @test MemoryLayout(C) == BroadcastBlockBandedLayout{typeof(*)}()

        
        D = Diagonal(PseudoBlockArray(randn(5),1:3))
        @test MemoryLayout(BroadcastMatrix(*, A, D)) isa BroadcastBlockBandedLayout{typeof(*)}

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

        E = BroadcastMatrix(*, A, 2)
        @test MemoryLayout(E) == BroadcastBandedBlockBandedLayout{typeof(*)}()

        D = Diagonal(PseudoBlockArray(randn(5),1:3))
        @test MemoryLayout(BroadcastMatrix(*, A, D)) isa BroadcastBandedBlockBandedLayout{typeof(*)}

        F = BroadcastMatrix(*, Ones(axes(A,1)), A)
        @test blockbandwidths(F) == (1,1)
        @test subblockbandwidths(F) == (1,1)
        @test F == A
    end
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
    C .= applied(*, A,B)
    @test C == A*B

    C.data .= NaN
    C .= @~ 1.0 * A*B + 0.0 * C
    @test C == A*B
end

@testset "Applied" begin
    A = brand(5,5,1,2)
    @test applied(*,Symmetric(A),A) isa Applied{MulStyle}
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

        A = LinearAlgebra.Tridiagonal(randn(T,99), randn(T,100), randn(T,99))
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

        @time D_xx = BandedBlockBandedMatrix(BlockKron(D², Eye(n)))
        @time D_yy = BandedBlockBandedMatrix(BlockKron(Eye(n),D²))
        @test D_xx == blockkron(D², Eye(n))
        @time Δ = D_xx + D_yy

        @test Δ isa BandedBlockBandedMatrix
        @test blockbandwidths(Δ) == subblockbandwidths(Δ) == (1,1)
        @test Δ == blockkron(Matrix(D²), Matrix(I,n,n)) + blockkron(Matrix(I,n,n), Matrix(D²))

        n = 10
        D² = FiniteDifference(n)
        D̃_xx = BlockKron(D², Eye(n))
        @test blockbandwidths(D̃_xx) == (1,1)
        @test subblockbandwidths(D̃_xx) == (0,0)

        V = view(D̃_xx, Block(1,1))
        @test bandwidths(V) == (0,0)

        @test BandedBlockBandedMatrix(D̃_xx) ≈ D_xx

        D̃_yy = BlockKron(Eye(n), D²)
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
        D_xx = BandedBlockBandedMatrix(BlockKron(D², Eye(n)))

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

    @testset "Banded Vcat" begin
        A = Vcat(Zeros(1,10), brand(9,10,1,1))
        @test isbanded(A)
        @test bandwidths(A) == (2,0)
        @test MemoryLayout(typeof(A)) isa ApplyBandedLayout{typeof(vcat)}
        @test BandedMatrix(A) == Array(A) == A
        @test A*A isa MulMatrix
        @test A*A ≈ BandedMatrix(A)*A ≈ A*BandedMatrix(A) ≈ BandedMatrix(A*A)
        @test A[1:5,1:5] isa BandedMatrix
    end

    @testset "resize" begin
        A = brand(4,5,1,1)
        @test LazyBandedMatrices.resize(A,6,5)[1:4,1:5] == A
        @test LazyBandedMatrices.resize(view(A,2:3,2:5),5,5) isa BandedMatrix
        @test LazyBandedMatrices.resize(view(A,2:3,2:5),5,5)[1:2,1:4] == A[2:3,2:5]
    end

    @testset "Lazy banded * Padded" begin
        A = _BandedMatrix(Vcat(BroadcastArray(exp, 1:5)', Ones(1,5)), 5, 1, 0)
        @test MemoryLayout(A) isa BandedColumns{LazyLayout}
        x = Vcat([1,2], Zeros(3))
        @test A*x isa Vcat
        @test A*A*x isa Vcat
    end

    @testset "Lazy banded" begin
        A = _BandedMatrix(Ones{Int}(1,10),10,0,0)'
        B = _BandedMatrix((-2:-2:-20)', 10,-1,1)
        C = Diagonal( BroadcastVector(/, 2, (1:2:20)))
        C̃ = _BandedMatrix(BroadcastArray(/, 2, (1:2:20)'), 10, -1, 1)
        D = MyLazyArray(randn(10,10))
        M = ApplyArray(*,A,A)
        M̃ = ApplyArray(*,randn(10,10),randn(10,10))
        @test MemoryLayout(A) isa BandedRows{OnesLayout}
        @test MemoryLayout(B) isa BandedColumns{UnknownLayout}
        @test MemoryLayout(C) isa DiagonalLayout{LazyLayout}
        @test MemoryLayout(C̃) isa BandedColumns{LazyLayout}
        BC = BroadcastArray(*, B, permutedims(MyLazyArray(Array(C.diag))))
        @test MemoryLayout(BC) isa BroadcastBandedLayout
        @test A*BC isa MulMatrix
        @test BC*B isa MulMatrix
        @test BC*BC isa MulMatrix
        @test C*C̃ isa MulMatrix
        @test C̃*C isa MulMatrix
        @test C̃*D isa MulMatrix
        @test D*C̃ isa MulMatrix
        @test C̃*M isa MulMatrix
        @test M*C̃ isa MulMatrix
        @test C̃*M̃ isa MulMatrix
        @test M̃*C̃ isa MulMatrix
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

    @testset "Padded Block" begin
        b = PseudoBlockArray(cache(Zeros(55)),1:10);
        b[10] = 5;
        @test MemoryLayout(b) isa PaddedLayout{DenseColumnMajor}
        @test paddeddata(b) isa PseudoBlockVector
        @test paddeddata(b) == [zeros(9); 5]
    end

    # @testset "Padded columns" begin
    #     B = brand(8,8,1,2)
    #     v = view(B,:,4)
    #     w = view(B,3,:)
    #     @test MemoryLayout(v) isa PaddedLayout
    #     @test_broken MemoryLayout(w) isa PaddedLayout
    #     @test paddeddata(v) isa Vcat
    #     paddeddata(v) == B[:,4]
    # end

    @testset "Banded rot" begin
        A = brand(5,5,1,2)
        R = ApplyArray(rot180, A)
        @test MemoryLayout(R) isa BandedColumns{StridedLayout}
        @test bandwidths(R) == (2,1)
        @test BandedMatrix(R) == R == rot180(Matrix(A)) == rot180(A)

        A = brand(5,4,1,2)
        R = ApplyArray(rot180, A)
        @test MemoryLayout(R) isa BandedColumns{StridedLayout}
        @test bandwidths(R) == (3,0)
        @test BandedMatrix(R) == R == rot180(Matrix(A)) == rot180(A)
        
        A = brand(5,6,1,-1)
        R = ApplyArray(rot180, A)
        @test MemoryLayout(R) isa BandedColumns{StridedLayout}
        @test bandwidths(R) == (-2,2)
        @test BandedMatrix(R) == R == rot180(Matrix(A)) == rot180(A)

        A = brand(6,5,-1,1)
        R = ApplyArray(rot180, A)
        @test MemoryLayout(R) isa BandedColumns{StridedLayout}
        @test bandwidths(R) == (2,-2)
        @test BandedMatrix(R) == R == rot180(Matrix(A)) == rot180(A)

        B = brand(5,4,1,1)
        R = ApplyArray(rot180, ApplyArray(*, A, B))
        MemoryLayout(R) isa ApplyBandedLayout{typeof(*)}
        @test bandwidths(R) == (4,-2)
        @test R == rot180(A*B)
    end

    @testset "invlayout * structured banded (#21)" begin
        A = randn(5,5)
        B = BroadcastArray(*, brand(5,5,1,1), 2)
        @test A * B ≈ A * Matrix(B)
        @test A \ B ≈ A \ Matrix(B)
    end

    @testset "Triangular bandwidths" begin
        B = brand(5,5,1,2)
        @test bandwidths(ApplyArray(\, Diagonal(randn(5)), B)) == (1,2)
        @test bandwidths(ApplyArray(\, UpperTriangular(randn(5,5)), B)) == (1,4)
        @test bandwidths(ApplyArray(\, LowerTriangular(randn(5,5)), B)) == (4,2)
        @test bandwidths(ApplyArray(\, randn(5,5), B)) == (4,4)
    end

    @testset "Lazy block" begin
        b = PseudoBlockVector(randn(5),[2,3])
        c = BroadcastVector(exp,1:5)
        @test c .* b isa BroadcastVector
        @test b .* c isa BroadcastVector
        @test (c .* b)[Block(1)] == c[1:2] .* b[Block(1)]
    end

    @testset "Concat bandwidths" begin
        @test bandwidths(Hcat(1,randn(1,5))) == (0,5)
        @test bandwidths(Vcat(1,randn(5,1))) == (5,0)
    end
end

@testset "QR" begin
    A = brand(100_000,100_000,1,1)
    F = qr(A)
    b = Vcat([1,2,3],Zeros(size(A,1)-3))
    @test F.Q'b == apply(*,F.Q',b)
end



include("test_blockkron.jl")
include("test_blockconcat.jl")
