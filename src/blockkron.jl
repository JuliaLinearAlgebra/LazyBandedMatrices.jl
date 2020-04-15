
"""
    blockvec(A::AbstractMatrix)

is like `vec(A)` but includes block structure to represent the columns.
"""
blockvec(A::AbstractMatrix) = PseudoBlockVector(vec(A), Fill(size(A,1), size(A,2)))


"""
    diagtrav(A::AbstractMatrix)
"""
struct DiagTrav{T, AA<:AbstractMatrix} <: AbstractBlockVector{T}
    matrix::AA
    function DiagTrav{T,AA}(matrix::AA) where {T,AA<:AbstractMatrix}
        checksquare(matrix)
        new{T,AA}(matrix)
    end
end

DiagTrav(A::AA) where AA<:AbstractMatrix{T} where T = DiagTrav{T,AA}(A)

axes(A::DiagTrav) = (blockedrange(Base.OneTo(size(A.matrix,1))),)

getindex(A::DiagTrav, K::Block{1}) = A.matrix[range(Int(K); step=size(A.matrix,1)-1, length=Int(K))]
getindex(A::DiagTrav, k::Int) = A[findblockindex(axes(A,1), k)]