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
ends