using LazyBandedMatrices, FillArrays, BandedMatrices, BlockArrays, Test

@testset "DiagTrav" begin
    A = [1 2 3; 4 5 6; 7 8 9]
    @test DiagTrav(A) == [1, 4, 2, 7, 5, 3]
    A = [1 2 3; 4 5 6]
    @test DiagTrav(A) == [1, 4, 2, 5, 3]
    A = [1 2; 3 4; 5 6]
    @test DiagTrav(A) == [1, 3, 2, 5, 4]

    A = DiagTrav(randn(3,3,3))
    @test A[Block(1)] == A[1:1,1,1]
    @test A[Block(2)] == [A.array[2,1,1], A.array[1,2,1], A.array[1,1,2]]
    @test A[Block(3)] == [A.array[3,1,1], A.array[2,2,1], A.array[1,3,1], 
                          A.array[2,1,2], A.array[1,2,2], A.array[1,1,3]]
    @test A == [A[Block(1)]; A[Block(2)]; A[Block(3)]]
end
@testset "KronTrav" begin
    a = [1,2,3]
    b = [4,5,6]
    c = [7,8]
    @test KronTrav(a,b) == DiagTrav(b*a')
    @test KronTrav(a,c) == DiagTrav(c*a')
    @test KronTrav(c,a) == DiagTrav(a*c')

    A = [1 2; 3 4]
    B = [5 6; 7 8]
    K = KronTrav(A,B) 

    X = [9 10; 11 0]
    @test K*DiagTrav(X) == DiagTrav(B*X*A')

    n = 4
    Δ = BandedMatrix(1 => Ones(n-1), 0 => Fill(-2,n), -1 => Ones(n-1))
    X = triu!(randn(n,n))[:,end:-1:1]
    @test KronTrav(Δ, Eye(n)) * DiagTrav(X) == DiagTrav(X * Δ')
    @test KronTrav(Eye(n), Δ) * DiagTrav(X) == DiagTrav(Δ * X)
end