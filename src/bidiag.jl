# This file is based on LinearAlgebra/src/bidiag.jl, a part of Julia. License is MIT: https://julialang.org/license

# Bidiagonal matrices
struct Bidiagonal{T,DV<:AbstractVector{T},EV<:AbstractVector{T}} <: AbstractBandedMatrix{T}
    dv::DV      # diagonal
    ev::EV      # sub/super diagonal
    uplo::Char # upper bidiagonal ('U') or lower ('L')
    function Bidiagonal{T,DV,EV}(dv, ev, uplo::AbstractChar) where {T,DV<:AbstractVector{T},EV<:AbstractVector{T}}
        require_one_based_indexing(dv, ev)
        if length(ev) != max(length(dv)-1, 0)
            throw(DimensionMismatch("length of diagonal vector is $(length(dv)), length of off-diagonal vector is $(length(ev))"))
        end
        new{T,DV,EV}(dv, ev, uplo)
    end
end
function Bidiagonal{T,DV,EV}(dv, ev, uplo::Symbol) where {T,DV<:AbstractVector{T},EV<:AbstractVector{T}}
    Bidiagonal{T,DV,EV}(dv, ev, char_uplo(uplo))
end
function Bidiagonal{T}(dv::AbstractVector, ev::AbstractVector, uplo::Union{Symbol,AbstractChar}) where {T}
    Bidiagonal(convert(AbstractVector{T}, dv)::AbstractVector{T},
               convert(AbstractVector{T}, ev)::AbstractVector{T},
               uplo)
end

"""
    Bidiagonal(dv::V, ev::V, uplo::Symbol) where V <: AbstractVector

Constructs an upper (`uplo=:U`) or lower (`uplo=:L`) bidiagonal matrix using the
given diagonal (`dv`) and off-diagonal (`ev`) vectors. The result is of type `Bidiagonal`
and provides efficient specialized linear solvers, but may be converted into a regular
matrix with [`convert(Array, _)`](@ref) (or `Array(_)` for short). The length of `ev`
must be one less than the length of `dv`.

# Examples
```jldoctest
julia> dv = [1, 2, 3, 4]
4-element Vector{Int64}:
 1
 2
 3
 4

julia> ev = [7, 8, 9]
3-element Vector{Int64}:
 7
 8
 9

julia> Bu = Bidiagonal(dv, ev, :U) # ev is on the first superdiagonal
4×4 Bidiagonal{Int64, Vector{Int64}}:
 1  7  ⋅  ⋅
 ⋅  2  8  ⋅
 ⋅  ⋅  3  9
 ⋅  ⋅  ⋅  4

julia> Bl = Bidiagonal(dv, ev, :L) # ev is on the first subdiagonal
4×4 Bidiagonal{Int64, Vector{Int64}}:
 1  ⋅  ⋅  ⋅
 7  2  ⋅  ⋅
 ⋅  8  3  ⋅
 ⋅  ⋅  9  4
```
"""
function Bidiagonal(dv::DV, ev::EV, uplo::Symbol) where {T,DV<:AbstractVector{T},EV<:AbstractVector{T}}
    Bidiagonal{T,DV,EV}(dv, ev, char_uplo(uplo))
end
function Bidiagonal(dv::DV, ev::EV, uplo::AbstractChar) where {T,DV<:AbstractVector{T},EV<:AbstractVector{T}}
    Bidiagonal{T,DV,EV}(dv, ev, uplo)
end


"""
    Bidiagonal(A, uplo::Symbol)

Construct a `Bidiagonal` matrix from the main diagonal of `A` and
its first super- (if `uplo=:U`) or sub-diagonal (if `uplo=:L`).

# Examples
```jldoctest
julia> A = [1 1 1 1; 2 2 2 2; 3 3 3 3; 4 4 4 4]
4×4 Matrix{Int64}:
 1  1  1  1
 2  2  2  2
 3  3  3  3
 4  4  4  4

julia> Bidiagonal(A, :U) # contains the main diagonal and first superdiagonal of A
4×4 Bidiagonal{Int64, Vector{Int64}}:
 1  1  ⋅  ⋅
 ⋅  2  2  ⋅
 ⋅  ⋅  3  3
 ⋅  ⋅  ⋅  4

julia> Bidiagonal(A, :L) # contains the main diagonal and first subdiagonal of A
4×4 Bidiagonal{Int64, Vector{Int64}}:
 1  ⋅  ⋅  ⋅
 2  2  ⋅  ⋅
 ⋅  3  3  ⋅
 ⋅  ⋅  4  4
```
"""
function Bidiagonal(A::AbstractMatrix, uplo::Symbol)
    Bidiagonal(diag(A, 0), diag(A, uplo === :U ? 1 : -1), uplo)
end

Bidiagonal(A::Bidiagonal) = A
Bidiagonal{T}(A::Bidiagonal{T}) where {T} = A
Bidiagonal{T}(A::Bidiagonal) where {T} = Bidiagonal{T}(A.dv, A.ev, A.uplo)

function getindex(A::Bidiagonal{T}, i::Integer, j::Integer) where T
    if !((1 <= i <= size(A,2)) && (1 <= j <= size(A,2)))
        throw(BoundsError(A,(i,j)))
    end
    if i == j
        return A.dv[i]
    elseif A.uplo == 'U' && (i == j - 1)
        return A.ev[i]
    elseif A.uplo == 'L' && (i == j + 1)
        return A.ev[j]
    else
        return zero(T)
    end
end

function setindex!(A::Bidiagonal, x, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    if i == j
        @inbounds A.dv[i] = x
    elseif A.uplo == 'U' && (i == j - 1)
        @inbounds A.ev[i] = x
    elseif A.uplo == 'L' && (i == j + 1)
        @inbounds A.ev[j] = x
    elseif !iszero(x)
        throw(ArgumentError(string("cannot set entry ($i, $j) off the ",
            "$(istriu(A) ? "upper" : "lower") bidiagonal band to a nonzero value ($x)")))
    end
    return x
end

## structured matrix methods ##
function Base.replace_in_print_matrix(A::Bidiagonal,i::Integer,j::Integer,s::AbstractString)
    if A.uplo == 'U'
        i==j || i==j-1 ? s : Base.replace_with_centered_mark(s)
    else
        i==j || i==j+1 ? s : Base.replace_with_centered_mark(s)
    end
end

#Converting from Bidiagonal to dense Matrix
function Matrix{T}(A::Bidiagonal) where T
    n = size(A, 1)
    B = zeros(T, n, n)
    if n == 0
        return B
    end
    for i = 1:n - 1
        B[i,i] = A.dv[i]
        if A.uplo == 'U'
            B[i, i + 1] = A.ev[i]
        else
            B[i + 1, i] = A.ev[i]
        end
    end
    B[n,n] = A.dv[n]
    return B
end
Matrix(A::Bidiagonal{T}) where {T} = Matrix{T}(A)
Array(A::Bidiagonal) = Matrix(A)
promote_rule(::Type{Matrix{T}}, ::Type{<:Bidiagonal{S}}) where {T,S} =
    @isdefined(T) && @isdefined(S) ? Matrix{promote_type(T,S)} : Matrix
promote_rule(::Type{Matrix}, ::Type{<:Bidiagonal}) = Matrix

#Converting from Bidiagonal to Tridiagonal
function Tridiagonal{T}(A::Bidiagonal) where T
    dv = convert(AbstractVector{T}, A.dv)
    ev = convert(AbstractVector{T}, A.ev)
    z = fill!(similar(ev), zero(T))
    A.uplo == 'U' ? Tridiagonal(z, dv, ev) : Tridiagonal(ev, dv, z)
end
promote_rule(::Type{<:Tridiagonal{T}}, ::Type{<:Bidiagonal{S}}) where {T,S} =
    @isdefined(T) && @isdefined(S) ? Tridiagonal{promote_type(T,S)} : Tridiagonal
promote_rule(::Type{<:Tridiagonal}, ::Type{<:Bidiagonal}) = Tridiagonal

# When asked to convert Bidiagonal to AbstractMatrix{T}, preserve structure by converting to Bidiagonal{T} <: AbstractMatrix{T}
AbstractMatrix{T}(A::Bidiagonal) where {T} = convert(Bidiagonal{T}, A)

convert(T::Type{<:Bidiagonal}, m::AbstractMatrix) = m isa T ? m : T(m)

# For B<:Bidiagonal, similar(B[, neweltype]) should yield a Bidiagonal matrix.
# On the other hand, similar(B, [neweltype,] shape...) should yield a sparse matrix.
# The first method below effects the former, and the second the latter.
similar(B::Bidiagonal, ::Type{T}) where {T} = Bidiagonal(similar(B.dv, T), similar(B.ev, T), B.uplo)
# The method below is moved to SparseArrays for now
# similar(B::Bidiagonal, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = spzeros(T, dims...)



####################
# Generic routines #
####################

function show(io::IO, M::Bidiagonal)
    # TODO: make this readable and one-line
    summary(io, M); println(io, ":")
    print(io, " diag:")
    print_matrix(io, (M.dv)')
    print(io, M.uplo == 'U' ? "\n super:" : "\n sub:")
    print_matrix(io, (M.ev)')
end

size(M::Bidiagonal) = (length(M.dv), length(M.dv))
function size(M::Bidiagonal, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    elseif d <= 2
        return length(M.dv)
    else
        return 1
    end
end

#Elementary operations
for func in (:conj, :copy, :real, :imag)
    @eval ($func)(M::Bidiagonal) = Bidiagonal(($func)(M.dv), ($func)(M.ev), M.uplo)
end

adjoint(B::Bidiagonal) = Adjoint(B)
transpose(B::Bidiagonal) = Transpose(B)
adjoint(B::Bidiagonal{<:Real}) = Bidiagonal(B.dv, B.ev, B.uplo == 'U' ? :L : :U)
transpose(B::Bidiagonal{<:Number}) = Bidiagonal(B.dv, B.ev, B.uplo == 'U' ? :L : :U)
function Base.copy(aB::Adjoint{<:Any,<:Bidiagonal})
    B = aB.parent
    return Bidiagonal(map(x -> copy.(adjoint.(x)), (B.dv, B.ev))..., B.uplo == 'U' ? :L : :U)
end
function Base.copy(tB::Transpose{<:Any,<:Bidiagonal})
    B = tB.parent
    return Bidiagonal(map(x -> copy.(transpose.(x)), (B.dv, B.ev))..., B.uplo == 'U' ? :L : :U)
end

iszero(M::Bidiagonal) = iszero(M.dv) && iszero(M.ev)
isone(M::Bidiagonal) = all(isone, M.dv) && iszero(M.ev)
function istriu(M::Bidiagonal, k::Integer=0)
    if M.uplo == 'U'
        if k <= 0
            return true
        elseif k == 1
            return iszero(M.dv)
        else # k >= 2
            return iszero(M.dv) && iszero(M.ev)
        end
    else # M.uplo == 'L'
        if k <= -1
            return true
        elseif k == 0
            return iszero(M.ev)
        else # k >= 1
            return iszero(M.ev) && iszero(M.dv)
        end
    end
end
function istril(M::Bidiagonal, k::Integer=0)
    if M.uplo == 'U'
        if k >= 1
            return true
        elseif k == 0
            return iszero(M.ev)
        else # k <= -1
            return iszero(M.ev) && iszero(M.dv)
        end
    else # M.uplo == 'L'
        if k >= 0
            return true
        elseif k == -1
            return iszero(M.dv)
        else # k <= -2
            return iszero(M.dv) && iszero(M.ev)
        end
    end
end
isdiag(M::Bidiagonal) = iszero(M.ev)

function tril!(M::Bidiagonal, k::Integer=0)
    n = length(M.dv)
    if !(-n - 1 <= k <= n - 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-n - 1) and at most $(n - 1) in an $n-by-$n matrix")))
    elseif M.uplo == 'U' && k < 0
        fill!(M.dv,0)
        fill!(M.ev,0)
    elseif k < -1
        fill!(M.dv,0)
        fill!(M.ev,0)
    elseif M.uplo == 'U' && k == 0
        fill!(M.ev,0)
    elseif M.uplo == 'L' && k == -1
        fill!(M.dv,0)
    end
    return M
end

function triu!(M::Bidiagonal, k::Integer=0)
    n = length(M.dv)
    if !(-n + 1 <= k <= n + 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least",
            "$(-n + 1) and at most $(n + 1) in an $n-by-$n matrix")))
    elseif M.uplo == 'L' && k > 0
        fill!(M.dv,0)
        fill!(M.ev,0)
    elseif k > 1
        fill!(M.dv,0)
        fill!(M.ev,0)
    elseif M.uplo == 'L' && k == 0
        fill!(M.ev,0)
    elseif M.uplo == 'U' && k == 1
        fill!(M.dv,0)
    end
    return M
end

function diag(M::Bidiagonal, n::Integer=0)
    # every branch call similar(..., ::Int) to make sure the
    # same vector type is returned independent of n
    if n == 0
        return copyto!(similar(M.dv, length(M.dv)), M.dv)
    elseif (n == 1 && M.uplo == 'U') ||  (n == -1 && M.uplo == 'L')
        return copyto!(similar(M.ev, length(M.ev)), M.ev)
    elseif -size(M,1) <= n <= size(M,1)
        return fill!(similar(M.dv, size(M,1)-abs(n)), 0)
    else
        throw(ArgumentError(string("requested diagonal, $n, must be at least $(-size(M, 1)) ",
            "and at most $(size(M, 2)) for an $(size(M, 1))-by-$(size(M, 2)) matrix")))
    end
end

function +(A::Bidiagonal, B::Bidiagonal)
    if A.uplo == B.uplo || length(A.dv) == 0
        Bidiagonal(A.dv+B.dv, A.ev+B.ev, A.uplo)
    else
        newdv = A.dv+B.dv
        Tridiagonal((A.uplo == 'U' ? (typeof(newdv)(B.ev), newdv, typeof(newdv)(A.ev)) : (typeof(newdv)(A.ev), newdv, typeof(newdv)(B.ev)))...)
    end
end

function -(A::Bidiagonal, B::Bidiagonal)
    if A.uplo == B.uplo || length(A.dv) == 0
        Bidiagonal(A.dv-B.dv, A.ev-B.ev, A.uplo)
    else
        newdv = A.dv-B.dv
        Tridiagonal((A.uplo == 'U' ? (typeof(newdv)(-B.ev), newdv, typeof(newdv)(A.ev)) : (typeof(newdv)(A.ev), newdv, typeof(newdv)(-B.ev)))...)
    end
end

-(A::Bidiagonal)=Bidiagonal(-A.dv,-A.ev,A.uplo)
*(A::Bidiagonal, B::Number) = Bidiagonal(A.dv*B, A.ev*B, A.uplo)
*(B::Number, A::Bidiagonal) = A*B
/(A::Bidiagonal, B::Number) = Bidiagonal(A.dv/B, A.ev/B, A.uplo)

function ==(A::Bidiagonal, B::Bidiagonal)
    if A.uplo == B.uplo
        return A.dv == B.dv && A.ev == B.ev
    else
        return iszero(A.ev) && iszero(B.ev) && A.dv == B.dv
    end
end


function check_A_mul_B!_sizes(C, A, B)
    require_one_based_indexing(C)
    require_one_based_indexing(A)
    require_one_based_indexing(B)
    nA, mA = size(A)
    nB, mB = size(B)
    nC, mC = size(C)
    if nA != nC
        throw(DimensionMismatch("sizes size(A)=$(size(A)) and size(C) = $(size(C)) must match at first entry."))
    elseif mA != nB
        throw(DimensionMismatch("second entry of size(A)=$(size(A)) and first entry of size(B) = $(size(B)) must match."))
    elseif mB != mC
        throw(DimensionMismatch("sizes size(B)=$(size(B)) and size(C) = $(size(C)) must match at first second entry."))
    end
end

# function to get the internally stored vectors for Bidiagonal and [Sym]Tridiagonal
# to avoid allocations in A_mul_B_td! below (#24324, #24578)
_diag(A::Tridiagonal, k) = k == -1 ? A.dl : k == 0 ? A.d : A.du
_diag(A::SymTridiagonal, k) = k == 0 ? A.dv : A.ev
function _diag(A::Bidiagonal, k)
    if k == 0
        return A.dv
    elseif (A.uplo == 'L' && k == -1) || (A.uplo == 'U' && k == 1)
        return A.ev
    else
        return diag(A, k)
    end
end


factorize(A::Bidiagonal) = A

Base._sum(A::Bidiagonal, ::Colon) = sum(A.dv) + sum(A.ev)
function Base._sum(A::Bidiagonal, dims::Integer)
    res = Base.reducedim_initarray(A, dims, zero(eltype(A)))
    n = length(A.dv)
    if n == 0
        # Just to be sure. This shouldn't happen since there is a check whether
        # length(A.dv) == length(A.ev) + 1 in the constructor.
        return res
    elseif n == 1
        res[1] = A.dv[1]
        return res
    end
    @inbounds begin
        if (dims == 1 && A.uplo == 'U') || (dims == 2 && A.uplo == 'L')
            res[1] = A.dv[1]
            for i = 2:length(A.dv)
                res[i] = A.ev[i-1] + A.dv[i]
            end
        elseif (dims == 1 && A.uplo == 'L') || (dims == 2 && A.uplo == 'U')
            for i = 1:length(A.dv)-1
                res[i] = A.ev[i] + A.dv[i]
            end
            res[end] = A.dv[end]
        elseif dims >= 3
            if A.uplo == 'U'
                for i = 1:length(A.dv)-1
                    res[i,i]   = A.dv[i]
                    res[i,i+1] = A.ev[i]
                end
            else
                for i = 1:length(A.dv)-1
                    res[i,i]   = A.dv[i]
                    res[i+1,i] = A.ev[i]
                end
            end
            res[end,end] = A.dv[end]
        end
    end
    res
end
