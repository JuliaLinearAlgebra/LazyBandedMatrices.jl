using LazyBandedMatrices, Test

@testset "test DiagTrav" begin
    A = [1 2 3; 4 5 6; 7 8 9]
    @test DiagTrav(A) == [1, 4, 2, 7, 5, 3]
end