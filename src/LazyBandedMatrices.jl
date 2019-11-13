module LazyBandedMatrices
using BandedMatrices, BlockBandedMatrices, LazyArrays, ArrayLayouts, MatrixFactorizations, LinearAlgebra, Base

import MatrixFactorizations: ql, ql!, QLPackedQ, QRPackedQ, reflector!, reflectorApply!

import Base: BroadcastStyle, similar, OneTo, copy
import Base.Broadcast: Broadcasted
import LinearAlgebra: kron, hcat, vcat, AdjOrTrans, AbstractTriangular, BlasFloat, BlasComplex, BlasReal, 
                        lmul!, rmul!

import ArrayLayouts: materialize!, colsupport, rowsupport, MatMulVecAdd, require_one_based_indexing, sublayout
import LazyArrays: LazyArrayStyle, combine_mul_styles, mulapplystyle, PaddedLayout,
                        broadcastlayout, applylayout, arguments, _arguments, 
                        LazyArrayApplyStyle, ApplyArrayBroadcastStyle, ApplyStyle,
                        LazyLayout, ApplyLayout, BroadcastLayout, FlattenMulStyle,
                        _mul_args_rows, _mul_args_cols, paddeddata, factorizestyle,
                        MulMatrix, Mul, CachedMatrix, CachedArray, resizedata!, applybroadcaststyle
import BandedMatrices: bandedcolumns, bandwidths, isbanded, AbstractBandedLayout,
                        prodbandwidths, BandedStyle, BandedColumns, BandedRows,
                        AbstractBandedMatrix, BandedSubBandedMatrix, BandedStyle, _bnds,
                        banded_rowsupport, banded_colsupport, _BandedMatrix, bandeddata,
                        banded_qr_lmul!, banded_qr_rmul!, banded_qr
import BlockBandedMatrices: AbstractBlockBandedLayout, BlockSlice, Block1,
                        isblockbanded, isbandedblockbanded, blockbandwidths, 
                        subblockbandwidths, blocksizes, BlockSizes

BroadcastStyle(::LazyArrayStyle{1}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{1}) = LazyArrayStyle{2}()
BroadcastStyle(::LazyArrayStyle{2}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()

bandedcolumns(::ML) where ML<:LazyLayout = BandedColumns{ML}()
bandedcolumns(::ML) where ML<:ApplyLayout = BandedColumns{LazyLayout}()

for LazyLay in (:(BandedColumns{LazyLayout}), :(BandedRows{LazyLayout}), 
                :(TriangularLayout{UPLO,UNIT,BandedRows{LazyLayout}} where {UPLO,UNIT}),
                :(TriangularLayout{UPLO,UNIT,BandedColumns{LazyLayout}} where {UPLO,UNIT}))
    @eval begin
        combine_mul_styles(::$LazyLay) = LazyArrayApplyStyle()
        mulapplystyle(::QLayout, ::$LazyLay) = LazyArrayApplyStyle()
    end
end

BroadcastStyle(M::ApplyArrayBroadcastStyle{2}, ::BandedStyle) = M
BroadcastStyle(::BandedStyle, M::ApplyArrayBroadcastStyle{2}) = M


bandwidths(M::Mul) = min.(_bnds(M), prodbandwidths(M.args...))

isbanded(K::Kron{<:Any,2}) = all(isbanded, K.args)
function bandwidths(K::Kron{<:Any,2})
    A,B = K.args
    (size(B,1)*bandwidth(A,1) + max(0,size(B,1)-size(B,2))*size(A,1)   + bandwidth(B,1),
        size(B,2)*bandwidth(A,2) + max(0,size(B,2)-size(B,1))*size(A,2) + bandwidth(B,2))
end

const BandedMatrixTypes = (:AbstractBandedMatrix, :(AdjOrTrans{<:Any,<:AbstractBandedMatrix}),
                                    :(AbstractTriangular{<:Any, <:AbstractBandedMatrix}),
                                    :(Symmetric{<:Any, <:AbstractBandedMatrix}))

const OtherBandedMatrixTypes = (:Zeros, :Eye, :Diagonal, :SymTridiagonal)

for T1 in BandedMatrixTypes, T2 in BandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

for T1 in BandedMatrixTypes, T2 in OtherBandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

for T1 in OtherBandedMatrixTypes, T2 in BandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end



###
# Specialised multiplication for arrays padded for zeros
# needed for ∞-dimensional banded linear algebra
###

function similar(M::MulAdd{<:AbstractBandedLayout,<:PaddedLayout}, ::Type{T}, axes) where T
    A,x = M.A,M.B
    xf = paddeddata(x)
    n = max(0,min(length(xf) + bandwidth(A,1),length(M)))
    Vcat(Vector{T}(undef, n), Zeros{T}(size(A,1)-n))
end

function materialize!(M::MatMulVecAdd{<:AbstractBandedLayout,<:PaddedLayout,<:PaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())

    ỹ = paddeddata(y)
    x̃ = paddeddata(x)

    length(ỹ) ≥ min(length(M),length(x̃)+bandwidth(A,1)) ||
        throw(InexactError("Cannot assign non-zero entries to Zero"))

    materialize!(MulAdd(α, view(A, axes(ỹ,1), axes(x̃,1)) , x̃, β, ỹ))
    y
end




###
# MulMatrix
###

bandwidths(M::MulMatrix) = bandwidths(Applied(M))
isbanded(M::Mul) = all(isbanded, M.args)
isbanded(M::MulMatrix) = isbanded(Applied(M))

struct MulBandedLayout <: AbstractBandedLayout end
applylayout(::Type{typeof(*)}, ::AbstractBandedLayout...) = MulBandedLayout()    

applybroadcaststyle(::Type{<:AbstractMatrix}, ::MulBandedLayout) = BandedStyle()
# applybroadcaststyle(::Type{<:AbstractMatrix}, ::MulLayout{<:Tuple{BandedColumns{LazyLayout},Vararg{<:AbstractBandedLayout}}}) = LazyArrayStyle{2}()

@inline colsupport(::MulBandedLayout, A, j) = banded_colsupport(A, j)
@inline rowsupport(::MulBandedLayout, A, j) = banded_rowsupport(A, j)
# @inline colsupport(::MulLayout{<:Tuple{<:AbstractBandedLayout,<:AbstractStridedLayout}}, A, j) = banded_colsupport(A, j)
@inline _arguments(::MulBandedLayout, A) = arguments(A)

###
# BroadcastMatrix
###

bandwidths(M::BroadcastMatrix) = bandwidths(Broadcasted(M))
isbanded(M::BroadcastMatrix) = isbanded(Broadcasted(M))

struct BroadcastBandedLayout{F} <: AbstractBandedLayout end
struct LazyBandedLayout <: AbstractBandedLayout end

broadcastlayout(::Type{F}, ::AbstractBandedLayout) where F = BroadcastBandedLayout{F}()
for op in (:*, :/, :\)
    @eval broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
end
for op in (:*, :/)
    @eval broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::Any) = BroadcastBandedLayout{typeof($op)}()
end
for op in (:*, :\)
    @eval broadcastlayout(::Type{typeof($op)}, ::Any, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
end
broadcastlayout(::Type{typeof(*)}, ::AbstractBandedLayout, ::LazyLayout) = LazyBandedLayout()
broadcastlayout(::Type{typeof(*)}, ::LazyLayout, ::AbstractBandedLayout) = LazyBandedLayout()
broadcastlayout(::Type{typeof(/)}, ::AbstractBandedLayout, ::LazyLayout) = LazyBandedLayout()
broadcastlayout(::Type{typeof(\)}, ::LazyLayout, ::AbstractBandedLayout) = LazyBandedLayout()

# functions that satisfy f(0,0) == 0
for op in (:+, :-)
    @eval broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
end

mulapplystyle(::LazyBandedLayout, ::MulBandedLayout) = FlattenMulStyle()
mulapplystyle(::MulBandedLayout, ::LazyBandedLayout) = FlattenMulStyle()


@inline colsupport(::BroadcastBandedLayout, A, j) = banded_colsupport(A, j)
@inline rowsupport(::BroadcastBandedLayout, A, j) = banded_rowsupport(A, j)


###
# sub materialize
###

function arguments(::MulBandedLayout, V::SubArray)
    P = parent(V)
    kr, jr = parentindices(V)
    as = arguments(P)
    kjr = intersect.(_mul_args_rows(kr, as...), _mul_args_cols(jr, reverse(as)...))
    view.(as, (kr, kjr...), (kjr..., jr))
end

@inline sub_materialize(::MulBandedLayout, V) = BandedMatrix(V)
@inline sub_materialize(::BroadcastBandedLayout, V) = BandedMatrix(V)

_BandedMatrix(::MulBandedLayout, V::AbstractMatrix) = apply(*, map(BandedMatrix,arguments(V))...)
for op in (:+, :-)
    @eval @inline _BandedMatrix(::BroadcastBandedLayout{typeof($op)}, V::AbstractMatrix) = apply($op, map(BandedMatrix,arguments(V))...)
end

function arguments(::BroadcastBandedLayout, V::SubArray)
    A = parent(V)
    kr, jr = parentindices(V)
    view.(arguments(A), Ref(kr), Ref(jr))
end



sublayout(M::MulBandedLayout, ::Type{<:Tuple{Vararg{AbstractUnitRange}}}) = M
sublayout(M::BroadcastBandedLayout, ::Type{<:Tuple{Vararg{AbstractUnitRange}}}) = M



######
# Concat banded matrix
######

bandwidths(M::Hcat) = (bandwidth(M.args[1],1),sum(size.(M.args[1:end-1],2)) + bandwidth(M.args[end],2))
isbanded(M::Hcat) = all(isbanded, M.args)

bandwidths(M::Vcat) = (sum(size.(M.args[1:end-1],1)) + bandwidth(M.args[end],1), bandwidth(M.args[1],2))
isbanded(M::Vcat) = all(isbanded, M.args)


const HcatBandedMatrix{T,N} = Hcat{T,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}
const VcatBandedMatrix{T,N} = Vcat{T,2,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}

BroadcastStyle(::Type{HcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()
BroadcastStyle(::Type{VcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()

Base.replace_in_print_matrix(A::HcatBandedMatrix, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)
Base.replace_in_print_matrix(A::VcatBandedMatrix, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)    

hcat(A::BandedMatrix...) = BandedMatrix(Hcat(A...))    
hcat(A::BandedMatrix, B::AbstractMatrix...) = Matrix(Hcat(A, B...))    

vcat(A::BandedMatrix...) = BandedMatrix(Vcat(A...))    
vcat(A::BandedMatrix, B::AbstractMatrix...) = Matrix(Vcat(A, B...))    



#######
# CachedArray
#######

bandwidths(B::CachedMatrix) = bandwidths(B.data)
isbanded(B::CachedMatrix) = isbanded(B.data)

function resizedata!(B::CachedMatrix{T,BandedMatrix{T,Matrix{T},OneTo{Int}}}, n::Integer, m::Integer) where T<:Number
    @boundscheck checkbounds(Bool, B, n, m) || throw(ArgumentError("Cannot resize beyound size of operator"))

    # increase size of array if necessary
    olddata = B.data
    ν,μ = B.datasize
    n,m = max(ν,n), max(μ,m)

    if (ν,μ) ≠ (n,m)
        l,u = bandwidths(B.array)
        λ,ω = bandwidths(B.data)
        if n ≥ size(B.data,1) || m ≥ size(B.data,2)
            M = 2*max(m,n+u)
            B.data = _BandedMatrix(reshape(resize!(vec(olddata.data), (λ+ω+1)*M), λ+ω+1, M), M+λ, λ,ω)
        end
        if ν > 0 # upper-right
            kr = max(1,μ+1-ω):ν
            jr = μ+1:min(m,ν+ω)
            if !isempty(kr) && !isempty(jr)
                view(B.data, kr, jr) .= B.array[kr, jr]
            end
        end
        view(B.data, ν+1:n, μ+1:m) .= B.array[ν+1:n, μ+1:m]
        if μ > 0
            kr = ν+1:min(n,μ+λ)
            jr = max(1,ν+1-λ):μ
            if !isempty(kr) && !isempty(jr)
                view(B.data, kr, jr) .= B.array[kr, jr]
            end
        end
        B.datasize = (n,m)
    end

    B
end

include("bandedql.jl")

###
# BlockBanded
###

mulapplystyle(::DiagonalLayout, ::AbstractBlockBandedLayout) = MulAddStyle()
mulapplystyle(::AbstractBlockBandedLayout, ::DiagonalLayout) = MulAddStyle()

isblockbanded(K::Kron{<:Any,2}) = isbanded(first(K.args))
isbandedblockbanded(K::Kron{<:Any,2}) = all(isbanded, K.args)
blockbandwidths(K::Kron{<:Any,2}) = bandwidths(first(K.args))
subblockbandwidths(K::Kron{<:Any,2}) = bandwidths(last(K.args))
function blocksizes(K::Kron{<:Any,2})
    A,B = K.args
    BlockSizes(Fill(size(B,1), size(A,1)), Fill(size(B,2), size(A,2)))
end

const SubKron{T,M1,M2,R1,R2} =
    SubArray{T,2,<:Kron{T,2,Tuple{M1,M2}},Tuple{BlockSlice{R1},BlockSlice{R2}}}


BroadcastStyle(::Type{<:SubKron{<:Any,<:Any,B,Block1,Block1}}) where B =
    BroadcastStyle(B)

@inline bandwidths(V::SubKron{<:Any,<:Any,<:Any,Block1,Block1}) =
    subblockbandwidths(parent(V))


end