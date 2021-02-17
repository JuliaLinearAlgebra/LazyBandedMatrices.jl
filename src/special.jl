# This file is based on a part of Julia LinearAlgebra/src/special.jl. License is MIT: https://julialang.org/license

# Methods operating on different special matrix types


# Usually, reducedim_initarray calls similar, which yields a sparse matrix for a
# Diagonal/Bidiagonal/Tridiagonal/SymTridiagonal matrix. However, reducedim should
# yield a dense vector to increase performance.
Base.reducedim_initarray(A::Union{Bidiagonal,Tridiagonal,SymTridiagonal}, region, init, ::Type{R}) where {R} = fill(convert(R, init), Base.reduced_indices(A,region))


# Interconversion between special matrix types

# conversions from Diagonal to other special matrix types
Bidiagonal(A::Diagonal) = Bidiagonal(A.diag, fill!(similar(A.diag, length(A.diag)-1), 0), :U)
SymTridiagonal(A::Diagonal) = SymTridiagonal(A.diag, fill!(similar(A.diag, length(A.diag)-1), 0))
Tridiagonal(A::Diagonal) = Tridiagonal(fill!(similar(A.diag, length(A.diag)-1), 0), A.diag,
                                       fill!(similar(A.diag, length(A.diag)-1), 0))

# conversions from Bidiagonal to other special matrix types
Diagonal(A::Bidiagonal) = Diagonal(A.dv)
SymTridiagonal(A::Bidiagonal) =
    iszero(A.ev) ? SymTridiagonal(A.dv, A.ev) :
        throw(ArgumentError("matrix cannot be represented as SymTridiagonal"))
Tridiagonal(A::Bidiagonal) =
    Tridiagonal(A.uplo == 'U' ? fill!(similar(A.ev), 0) : A.ev, A.dv,
                A.uplo == 'U' ? A.ev : fill!(similar(A.ev), 0))

# conversions from SymTridiagonal to other special matrix types
Diagonal(A::SymTridiagonal) = Diagonal(A.dv)
Bidiagonal(A::SymTridiagonal) =
    iszero(A.ev) ? Bidiagonal(A.dv, A.ev, :U) :
        throw(ArgumentError("matrix cannot be represented as Bidiagonal"))
Tridiagonal(A::SymTridiagonal) =
    Tridiagonal(copy(A.ev), A.dv, A.ev)

# conversions from Tridiagonal to other special matrix types
Diagonal(A::Tridiagonal) = Diagonal(A.d)
Bidiagonal(A::Tridiagonal) =
    iszero(A.dl) ? Bidiagonal(A.d, A.du, :U) :
    iszero(A.du) ? Bidiagonal(A.d, A.dl, :L) :
        throw(ArgumentError("matrix cannot be represented as Bidiagonal"))

# conversions from AbstractTriangular to special matrix types
Bidiagonal(A::AbstractTriangular) =
    LinearAlgebra.isbanded(A, 0, 1) ? Bidiagonal(diag(A, 0), diag(A,  1), :U) : # is upper bidiagonal
    LinearAlgebra.isbanded(A, -1, 0) ? Bidiagonal(diag(A, 0), diag(A, -1), :L) : # is lower bidiagonal
        throw(ArgumentError("matrix cannot be represented as Bidiagonal"))

const ConvertibleSpecialMatrix = Union{Diagonal,Bidiagonal,SymTridiagonal,Tridiagonal,AbstractTriangular}

convert(T::Type{<:Diagonal},       m::Union{Bidiagonal,SymTridiagonal,Tridiagonal}) = m isa T ? m :
    isdiag(m) ? T(m) : throw(ArgumentError("matrix cannot be represented as Diagonal"))
convert(T::Type{<:SymTridiagonal}, m::ConvertibleSpecialMatrix) = m isa T ? m :
    issymmetric(m) && LinearAlgebra.isbanded(m, -1, 1) ? T(m) : throw(ArgumentError("matrix cannot be represented as SymTridiagonal"))
convert(T::Type{<:Tridiagonal},    m::ConvertibleSpecialMatrix) = m isa T ? m :
    LinearAlgebra.isbanded(m, -1, 1) ? T(m) : throw(ArgumentError("matrix cannot be represented as Tridiagonal"))

convert(T::Type{<:LowerTriangular}, m::Bidiagonal) = m isa T ? m :
    istril(m) ? T(m) : throw(ArgumentError("matrix cannot be represented as LowerTriangular"))
convert(T::Type{<:UpperTriangular}, m::Bidiagonal) = m isa T ? m :
    istriu(m) ? T(m) : throw(ArgumentError("matrix cannot be represented as UpperTriangular"))


# specialized +/- for structured matrices. If these are removed, it falls
# back to broadcasting which has ~2-10x speed regressions.
# For the other structure matrix pairs, broadcasting works well.

# For structured matrix types with different non-zero diagonals the underlying
# representations must be promoted to the same type.
# For example, in Diagonal + Bidiagonal only the main diagonal is touched so
# the off diagonal could be a different type after the operation resulting in
# an error. See issue #28994

function (+)(A::Bidiagonal, B::Diagonal)
    newdv = A.dv + B.diag
    Bidiagonal(newdv, A.ev, A.uplo)
end

function (-)(A::Bidiagonal, B::Diagonal)
    newdv = A.dv - B.diag
    Bidiagonal(newdv, A.ev, A.uplo)
end

function (+)(A::Diagonal, B::Bidiagonal)
    newdv = A.diag + B.dv
    Bidiagonal(newdv, B.ev, B.uplo)
end

function (-)(A::Diagonal, B::Bidiagonal)
    newdv = A.diag-B.dv
    Bidiagonal(newdv, -B.ev, B.uplo)
end

function (+)(A::Diagonal, B::SymTridiagonal)
    newdv = A.diag+B.dv
    SymTridiagonal(A.diag+B.dv, B.ev)
end

function (-)(A::Diagonal, B::SymTridiagonal)
    newdv = A.diag-B.dv
    SymTridiagonal(newdv, -B.ev)
end

function (+)(A::SymTridiagonal, B::Diagonal)
    newdv = A.dv+B.diag
    SymTridiagonal(newdv, A.ev)
end

function (-)(A::SymTridiagonal, B::Diagonal)
    newdv = A.dv-B.diag
    SymTridiagonal(newdv, A.ev)
end

# this set doesn't have the aforementioned problem

+(A::Tridiagonal, B::SymTridiagonal) = Tridiagonal(A.dl+B.ev, A.d+B.dv, A.du+B.ev)
-(A::Tridiagonal, B::SymTridiagonal) = Tridiagonal(A.dl-B.ev, A.d-B.dv, A.du-B.ev)
+(A::SymTridiagonal, B::Tridiagonal) = Tridiagonal(A.ev+B.dl, A.dv+B.d, A.ev+B.du)
-(A::SymTridiagonal, B::Tridiagonal) = Tridiagonal(A.ev-B.dl, A.dv-B.d, A.ev-B.du)


function (+)(A::Diagonal, B::Tridiagonal)
    newdv = A.diag+B.d
    Tridiagonal(B.dl, newdv, B.du)
end

function (-)(A::Diagonal, B::Tridiagonal)
    newdv = A.diag-B.d
    Tridiagonal(-B.dl, newdv, -B.du)
end

function (+)(A::Tridiagonal, B::Diagonal)
    newdv = A.d+B.diag
    Tridiagonal(A.dl, newdv, A.du)
end

function (-)(A::Tridiagonal, B::Diagonal)
    newdv = A.d-B.diag
    Tridiagonal(A.dl, newdv, A.du)
end

function (+)(A::Bidiagonal, B::Tridiagonal)
    newdv = A.dv+B.d
    Tridiagonal((A.uplo == 'U' ? (B.dl, newdv, A.ev+B.du) : (A.ev+B.dl, newdv, B.du))...)
end

function (-)(A::Bidiagonal, B::Tridiagonal)
    newdv = A.dv-B.d
    Tridiagonal((A.uplo == 'U' ? (-B.dl, newdv, A.ev-B.du) : (A.ev-B.dl, newdv, -B.du))...)
end

function (+)(A::Tridiagonal, B::Bidiagonal)
    newdv = A.d+B.dv
    Tridiagonal((B.uplo == 'U' ? (A.dl, newdv, A.du+B.ev) : (A.dl+B.ev, newdv, A.du))...)
end

function (-)(A::Tridiagonal, B::Bidiagonal)
    newdv = A.d-B.dv
    Tridiagonal((B.uplo == 'U' ? (A.dl, newdv, A.du-B.ev) : (A.dl-B.ev, newdv, A.du))...)
end

function (+)(A::Bidiagonal, B::SymTridiagonal)
    newdv = A.dv+B.dv
    Tridiagonal((A.uplo == 'U' ? (B.ev, A.dv+B.dv, A.ev+B.ev) : (A.ev+B.ev, A.dv+B.dv, B.ev))...)
end

function (-)(A::Bidiagonal, B::SymTridiagonal)
    newdv = A.dv-B.dv
    Tridiagonal((A.uplo == 'U' ? (-B.ev, newdv, A.ev-B.ev) : (A.ev-B.ev, newdv, -B.ev))...)
end

function (+)(A::SymTridiagonal, B::Bidiagonal)
    newdv = A.dv+B.dv
    Tridiagonal((B.uplo == 'U' ? (A.ev, newdv, A.ev+B.ev) : (A.ev+B.ev, newdv, A.ev))...)
end

function (-)(A::SymTridiagonal, B::Bidiagonal)
    newdv = A.dv-B.dv
    Tridiagonal((B.uplo == 'U' ? (A.ev, newdv, A.ev-B.ev) : (A.ev-B.ev, newdv, A.ev))...)
end

# fixing uniform scaling problems from #28994
# {<:Number} is required due to the test case from PR #27289 where eltype is a matrix.

function (+)(A::Tridiagonal{<:Number}, B::UniformScaling)
    newd = A.d .+ B.λ
    Tridiagonal(A.dl, newd, A.du)
end

function (+)(A::SymTridiagonal{<:Number}, B::UniformScaling)
    newdv = A.dv .+ B.λ
    SymTridiagonal(newdv, A.ev)
end

function (+)(A::Bidiagonal{<:Number}, B::UniformScaling)
    newdv = A.dv .+ B.λ
    Bidiagonal(newdv, A.ev, A.uplo)
end

(+)(A::UniformScaling, B::Tridiagonal{<:Number}) = Tridiagonal(B.dl, A.λ .+ B.d, B.du)
(+)(A::UniformScaling, B::SymTridiagonal{<:Number}) = SymTridiagonal(A.λ .+ B.dv, B.ev)

function (+)(A::UniformScaling, B::Bidiagonal{<:Number})
    newdv = A.λ .+ B.dv
    Bidiagonal(newdv, B.ev, B.uplo)
end

function (-)(A::UniformScaling, B::Tridiagonal{<:Number})
    newd = A.λ .- B.d
    Tridiagonal(-B.dl, newd, -B.du)
end

function (-)(A::UniformScaling, B::SymTridiagonal{<:Number})
    newdv = A.λ .- B.dv
    SymTridiagonal(newdv, -B.ev)
end

function (-)(A::UniformScaling, B::Bidiagonal{<:Number})
    newdv = A.λ .- B.dv
    Bidiagonal(newdv, -B.ev, B.uplo)
end

# fill[stored]! methods
fillstored!(A::Bidiagonal, x) = (fill!(A.dv, x); fill!(A.ev, x); A)
fillstored!(A::Tridiagonal, x) = (fill!(A.dl, x); fill!(A.d, x); fill!(A.du, x); A)
fillstored!(A::SymTridiagonal, x) = (fill!(A.dv, x); fill!(A.ev, x); A)

_small_enough(A::Bidiagonal) = size(A, 1) <= 1
_small_enough(A::Tridiagonal) = size(A, 1) <= 2
_small_enough(A::SymTridiagonal) = size(A, 1) <= 2

function fill!(A::Union{Bidiagonal,Tridiagonal,SymTridiagonal}, x)
    xT = convert(eltype(A), x)
    (iszero(xT) || _small_enough(A)) && return fillstored!(A, xT)
    throw(ArgumentError("array of type $(typeof(A)) and size $(size(A)) can
    not be filled with $x, since some of its entries are constrained."))
end

one(A::Bidiagonal{T}) where T = Bidiagonal(fill!(similar(A.dv, typeof(one(T))), one(T)), fill!(similar(A.ev, typeof(one(T))), zero(one(T))), A.uplo)
one(A::Tridiagonal{T}) where T = Tridiagonal(fill!(similar(A.du, typeof(one(T))), zero(one(T))), fill!(similar(A.d, typeof(one(T))), one(T)), fill!(similar(A.dl, typeof(one(T))), zero(one(T))))
one(A::SymTridiagonal{T}) where T = SymTridiagonal(fill!(similar(A.dv, typeof(one(T))), one(T)), fill!(similar(A.ev, typeof(one(T))), zero(one(T))))
# equals and approx equals methods for structured matrices
# SymTridiagonal == Tridiagonal is already defined in tridiag.jl

# SymTridiagonal and Bidiagonal have the same field names
==(A::Diagonal, B::Union{SymTridiagonal, Bidiagonal}) = iszero(B.ev) && A.diag == B.dv
==(B::Bidiagonal, A::Diagonal) = A == B

==(A::Diagonal, B::Tridiagonal) = iszero(B.dl) && iszero(B.du) && A.diag == B.d
==(B::Tridiagonal, A::Diagonal) = A == B

function ==(A::Bidiagonal, B::Tridiagonal)
    if A.uplo == 'U'
        return iszero(B.dl) && A.dv == B.d && A.ev == B.du
    else
        return iszero(B.du) && A.dv == B.d && A.ev == B.dl
    end
end
==(B::Tridiagonal, A::Bidiagonal) = A == B

==(A::Bidiagonal, B::SymTridiagonal) = iszero(B.ev) && iszero(A.ev) && A.dv == B.dv
==(B::SymTridiagonal, A::Bidiagonal) = A == B


import LinearAlgebra: TypeFuncs
LinearAlgebra.isstructurepreserving(::Union{typeof(abs),typeof(big)}, ::Union{Tridiagonal,SymTridiagonal,Bidiagonal}) = true
LinearAlgebra.isstructurepreserving(::TypeFuncs, ::Union{Tridiagonal,SymTridiagonal,Bidiagonal}) = true
LinearAlgebra.isstructurepreserving(::TypeFuncs, ::Ref{<:Type}, ::Union{Tridiagonal,SymTridiagonal,Bidiagonal}) = true