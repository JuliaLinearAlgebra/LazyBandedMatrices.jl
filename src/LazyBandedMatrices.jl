module LazyBandedMatrices
using BandedMatrices, BlockBandedMatrices, BlockArrays, LazyArrays,
        ArrayLayouts, MatrixFactorizations, LinearAlgebra, Base

import MatrixFactorizations: ql, ql!, QLPackedQ, QRPackedQ, reflector!, reflectorApply!,
            QLPackedQLayout, QRPackedQLayout, AdjQLPackedQLayout, AdjQRPackedQLayout

import Base: BroadcastStyle, similar, OneTo, copy, *, axes, size, getindex
import Base.Broadcast: Broadcasted
import LinearAlgebra: kron, hcat, vcat, AdjOrTrans, AbstractTriangular, BlasFloat, BlasComplex, BlasReal,
                        lmul!, rmul!, checksquare, StructuredMatrixStyle

import ArrayLayouts: materialize!, colsupport, rowsupport, MatMulVecAdd, require_one_based_indexing,
                    sublayout, transposelayout, _copyto!, MemoryLayout
import LazyArrays: LazyArrayStyle, combine_mul_styles, mulapplystyle, PaddedLayout,
                        broadcastlayout, applylayout, arguments, _arguments, call,
                        LazyArrayApplyStyle, ApplyArrayBroadcastStyle, ApplyStyle,
                        LazyLayout, ApplyLayout, BroadcastLayout, FlattenMulStyle, CachedVector,
                        _mul_args_rows, _mul_args_cols, paddeddata, sub_materialize,
                        MulMatrix, Mul, CachedMatrix, CachedArray, cachedlayout, _cache,
                        resizedata!, applybroadcaststyle, 
                        LazyMatrix, LazyVector, LazyArray, MulAddStyle
import BandedMatrices: bandedcolumns, bandwidths, isbanded, AbstractBandedLayout,
                        prodbandwidths, BandedStyle, BandedColumns, BandedRows,
                        AbstractBandedMatrix, BandedSubBandedMatrix, BandedStyle, _bnds,
                        banded_rowsupport, banded_colsupport, _BandedMatrix, bandeddata,
                        banded_qr_lmul!, banded_qr_rmul!, _banded_broadcast!
import BlockBandedMatrices: AbstractBlockBandedLayout, BlockSlice, Block1, AbstractBlockBandedLayout,
                        isblockbanded, isbandedblockbanded, blockbandwidths,
                        bandedblockbandedbroadcaststyle, bandedblockbandedcolumns,
                        BandedBlockBandedColumns, BlockBandedColumns,
                        subblockbandwidths, BandedBlockBandedMatrix, BlockBandedMatrix,
                        AbstractBandedBlockBandedLayout, BandedBlockBandedStyle,
                        blockcolsupport, BlockRange1
import BlockArrays: blockbroadcaststyle, BlockSlice1

export DiagTrav, KronTrav, blockkron

BroadcastStyle(::LazyArrayStyle{1}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{1}) = LazyArrayStyle{2}()
BroadcastStyle(::LazyArrayStyle{2}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()

bandedcolumns(::ML) where ML<:LazyLayout = BandedColumns{ML}()
bandedcolumns(::ML) where ML<:ApplyLayout = BandedColumns{LazyLayout}()

for LazyLay in (:(BandedColumns{LazyLayout}), :(BandedRows{LazyLayout}),
                :(TriangularLayout{UPLO,UNIT,BandedRows{LazyLayout}} where {UPLO,UNIT}),
                :(TriangularLayout{UPLO,UNIT,BandedColumns{LazyLayout}} where {UPLO,UNIT}),
                :(BlockBandedColumns{LazyLayout}), :(BandedBlockBandedColumns{LazyLayout}))
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



function paddeddata(P::PseudoBlockVector)
    C = P.blocks
    data = paddeddata(C)
    ax = axes(P,1)
    N = findblock(ax,length(data))
    n = last(ax[N])
    if n ≠ length(data)
        resizedata!(C,n)
    end
    PseudoBlockVector(data, (ax[Block(1):N],))
end


function similar(Ml::MulAdd{<:AbstractBlockBandedLayout,<:PaddedLayout}, ::Type{T}, _) where T
    A,x = Ml.A,Ml.B
    xf = paddeddata(x)
    ax1,ax2 = axes(A)
    N = findblock(ax2,length(xf))
    M = last(blockcolsupport(A,N))
    m = last(ax1[M]) # number of non-zero entries
    PseudoBlockVector(Vcat(Vector{T}(undef, m), Zeros{T}(length(ax1)-m)), (ax1,))
end

function materialize!(M::MatMulVecAdd{<:AbstractBlockBandedLayout,<:PaddedLayout,<:PaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())

    ỹ = paddeddata(y)
    x̃ = paddeddata(x)

    muladd!(α, view(A, blockaxes(ỹ,1), blockaxes(x̃,1)) , x̃, β, ỹ)
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

@inline colsupport(::MulBandedLayout, A, j) = banded_colsupport(A, j)
@inline rowsupport(::MulBandedLayout, A, j) = banded_rowsupport(A, j)
@inline _arguments(::MulBandedLayout, A) = arguments(A)


struct MulBlockBandedLayout <: AbstractBlockBandedLayout end
applylayout(::Type{typeof(*)}, ::AbstractBlockBandedLayout...) = MulBlockBandedLayout()


prodblockbandwidths(A) = blockbandwidths(A)
prodblockbandwidths() = (0,0)
prodblockbandwidths(A...) = broadcast(+, blockbandwidths.(A)...)

prodsubblockbandwidths(A) = subblockbandwidths(A)
prodsubblockbandwidths() = (0,0)
prodsubblockbandwidths(A...) = broadcast(+, subblockbandwidths.(A)...)

blockbandwidths(M::MulMatrix) = prodblockbandwidths(M.args...)
subblockbandwidths(M::MulMatrix) = prodsubblockbandwidths(M.args...)

###
# BroadcastMatrix
###

bandwidths(M::BroadcastMatrix) = bandwidths(Broadcasted(M))
# TODO: Generalize
for op in (:+, :-)
    @eval begin
        blockbandwidths(M::BroadcastMatrix{<:Any,typeof($op)}) =
            broadcast(max, map(blockbandwidths,arguments(M))...)
        subblockbandwidths(M::BroadcastMatrix{<:Any,typeof($op)}) =
            broadcast(max, map(subblockbandwidths,arguments(M))...)
    end
end

for func in (:blockbandwidths, :subblockbandwidths)
    @eval begin
        $func(M::BroadcastMatrix{<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}}) = $func(M.args[2])
        $func(M::BroadcastMatrix{<:Any,typeof(*),<:Tuple{<:AbstractMatrix,<:Number,}}) = $func(M.args[1])
    end
end

isbanded(M::BroadcastMatrix) = isbanded(Broadcasted(M))

struct BroadcastBandedLayout{F} <: AbstractBandedLayout end
struct BroadcastBlockBandedLayout{F} <: AbstractBlockBandedLayout end
struct BroadcastBandedBlockBandedLayout{F} <: AbstractBandedBlockBandedLayout end
struct LazyBandedLayout <: AbstractBandedLayout end

BroadcastLayout(::BroadcastBandedLayout{F}) where F = BroadcastLayout{F}()

broadcastlayout(::Type{F}, ::AbstractBandedLayout) where F = BroadcastBandedLayout{F}()
# functions that satisfy f(0,0) == 0
for op in (:*, :/, :\, :+, :-)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AbstractBlockBandedLayout, ::AbstractBlockBandedLayout) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedBlockBandedLayout, ::AbstractBandedBlockBandedLayout) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end
for op in (:*, :/)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::Any) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AbstractBlockBandedLayout, ::Any) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedBlockBandedLayout, ::Any) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end
for op in (:*, :\)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::Any, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::Any, ::AbstractBlockBandedLayout) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::Any, ::AbstractBandedBlockBandedLayout) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end
broadcastlayout(::Type{typeof(*)}, ::AbstractBandedLayout, ::LazyLayout) = LazyBandedLayout()
broadcastlayout(::Type{typeof(*)}, ::LazyLayout, ::AbstractBandedLayout) = LazyBandedLayout()
broadcastlayout(::Type{typeof(/)}, ::AbstractBandedLayout, ::LazyLayout) = LazyBandedLayout()
broadcastlayout(::Type{typeof(\)}, ::LazyLayout, ::AbstractBandedLayout) = LazyBandedLayout()


sublayout(LAY::BroadcastBandedBlockBandedLayout, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{BlockRange1}}}) = LAY

combine_mul_styles(::BroadcastBandedLayout, ::BroadcastBandedLayout) = LazyArrayApplyStyle()
combine_mul_styles(::MulBandedLayout, ::MulBandedLayout) = LazyArrayApplyStyle()
combine_mul_styles(::MulBandedLayout, ::BroadcastBandedLayout) = LazyArrayApplyStyle()
combine_mul_styles(::BroadcastBandedLayout, ::MulBandedLayout) = LazyArrayApplyStyle()

mulapplystyle(::LazyBandedLayout, ::LazyBandedLayout) = LazyArrayApplyStyle()
mulapplystyle(::LazyBandedLayout, ::AbstractBandedLayout) = LazyArrayApplyStyle()
mulapplystyle(::AbstractBandedLayout, ::LazyBandedLayout) = LazyArrayApplyStyle()
mulapplystyle(::LazyBandedLayout, ::MulBandedLayout) = FlattenMulStyle()
mulapplystyle(::MulBandedLayout, ::LazyBandedLayout) = FlattenMulStyle()
mulapplystyle(::AbstractBandedLayout, ::PaddedLayout) = MulAddStyle()


@inline colsupport(::BroadcastBandedLayout, A, j) = banded_colsupport(A, j)
@inline rowsupport(::BroadcastBandedLayout, A, j) = banded_rowsupport(A, j)

_broadcasted(bc) = Base.broadcasted(call(bc), arguments(bc)...)

_copyto!(::AbstractBandedLayout, ::BroadcastBandedLayout, dest::AbstractMatrix, bc::AbstractMatrix) =
    copyto!(dest, _broadcasted(bc))

_copyto!(_, ::BroadcastBandedLayout, dest::AbstractMatrix, bc::AbstractMatrix) =
    copyto!(dest, _broadcasted(bc))

_banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractMatrix{T},AbstractMatrix{V}}, ::Tuple{<:Any,MulBandedLayout}) where {T,V} =
    broadcast!(f, dest, BandedMatrix(A), BandedMatrix(B))


function _cache(::AbstractBlockBandedLayout, A::AbstractMatrix{T}) where T
    kr,jr = axes(A)
    CachedArray(BlockBandedMatrix{T}(undef, (kr[Block.(1:0)], jr[Block.(1:0)]), blockbandwidths(A)), A)
end

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

@inline sub_materialize(::MulBandedLayout, V, _) = BandedMatrix(V)
@inline sub_materialize(::BroadcastBandedLayout, V, _) = BandedMatrix(V)
@inline sub_materialize(::BandedColumns{LazyLayout}, V, _) = V
@inline sub_materialize(::BandedColumns{LazyLayout}, V, ::Tuple{<:OneTo,<:OneTo}) = BandedMatrix(V)

###
# copyto!
###

_BandedMatrix(::MulBandedLayout, V::AbstractMatrix) = apply(*, map(BandedMatrix,arguments(V))...)

for op in (:+, :-, :*)
    @eval begin
        @inline _BandedMatrix(::BroadcastBandedLayout{typeof($op)}, V::AbstractMatrix) = broadcast($op, map(BandedMatrix,arguments(V))...)
        _copyto!(::AbstractBandedLayout, ::BroadcastBandedLayout{typeof($op)}, dest::AbstractMatrix, src::AbstractMatrix) =
            broadcast!($op, dest, map(BandedMatrix, arguments(src))...)
        _copyto!(::AbstractBandedBlockBandedLayout, ::BroadcastBandedBlockBandedLayout{typeof($op)}, dest::AbstractMatrix, src::AbstractMatrix) =
            broadcast!($op, dest, map(BandedBlockBandedMatrix, arguments(src))...)
    end
end

_copyto!(::AbstractBandedLayout, ::MulBandedLayout, dest::AbstractMatrix, src::AbstractMatrix) =
    _mulbanded_copyto!(dest, map(BandedMatrix,arguments(src))...)

_mulbanded_copyto!(dest, a) = copyto!(dest, a)
_mulbanded_copyto!(dest::AbstractArray{T}, a, b) where T = muladd!(one(T), a, b, zero(T), dest)
_mulbanded_copyto!(dest::AbstractArray{T}, a, b, c, d...) where T = _mulbanded_copyto!(dest, apply(*,a,b), c, d...)

_broadcast_sub_arguments(::BroadcastBandedLayout{F}, A, V) where F = arguments(BroadcastLayout{F}(), V)
_broadcast_sub_arguments(::BroadcastBandedBlockBandedLayout{F}, A, V) where F = arguments(BroadcastLayout{F}(), V)
_broadcast_sub_arguments(A, V) = _broadcast_sub_arguments(MemoryLayout(A), A, V)
arguments(::BroadcastBandedLayout, V::SubArray) = _broadcast_sub_arguments(parent(V), V)
arguments(::BroadcastBandedBlockBandedLayout, V::SubArray) = _broadcast_sub_arguments(parent(V), V)


call(b::BroadcastBandedLayout, a) = call(BroadcastLayout(b), a)
call(b::BroadcastBandedLayout, a::SubArray) = call(BroadcastLayout(b), a)

sublayout(M::MulBandedLayout, ::Type{<:Tuple{Vararg{AbstractUnitRange}}}) = M
sublayout(M::BroadcastBandedLayout, ::Type{<:Tuple{Vararg{AbstractUnitRange}}}) = M

transposelayout(b::BroadcastBandedLayout) = b
arguments(b::BroadcastBandedLayout, A::AdjOrTrans) where F = arguments(BroadcastLayout(b), A)


######
# Concat banded matrix
######

# cumsum for tuples
_cumsum(a) = a
_cumsum(a, b...) = tuple(a, (a .+ _cumsum(b...))...)

function bandwidths(M::Vcat)
    cs = tuple(0, _cumsum(size.(M.args[1:end-1],1)...)...) # cumsum of sizes
    (maximum(cs .+ bandwidth.(M.args,1)), maximum(bandwidth.(M.args,2) .- cs))
end
isbanded(M::Vcat) = all(isbanded, M.args)

function bandwidths(M::Hcat)
    cs = tuple(0, _cumsum(size.(M.args[1:end-1],2)...)...) # cumsum of sizes
    (maximum(bandwidth.(M.args,1) .- cs), maximum(bandwidth.(M.args,2) .+ cs))
end
isbanded(M::Hcat) = all(isbanded, M.args)


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

cachedlayout(::BandedColumns{DenseColumnMajor}, ::AbstractBandedLayout) = BandedColumns{DenseColumnMajor}()
bandwidths(B::CachedMatrix) = bandwidths(B.data)
isbanded(B::CachedMatrix) = isbanded(B.data)

function bandeddata(A::CachedMatrix)
    resizedata!(A, size(A)...)
    bandeddata(A.data)
end

function bandeddata(B::SubArray{<:Any,2,<:CachedMatrix})
    A = parent(B)
    kr,jr = parentindices(B)
    resizedata!(A, maximum(kr), maximum(jr))
    bandeddata(view(A.data,kr,jr))
end

function resize(A::BandedMatrix, n::Integer, m::Integer)
    l,u = bandwidths(A)
    _BandedMatrix(reshape(resize!(vec(bandeddata(A)), (l+u+1)*m), l+u+1, m), n, l,u)
end
function resize(A::BandedSubBandedMatrix, n::Integer, m::Integer)
    l,u = bandwidths(A)
    _BandedMatrix(reshape(resize!(vec(copy(bandeddata(A))), (l+u+1)*m), l+u+1, m), n, l,u)
end
function resize(A::BlockSkylineMatrix{T}, ax::NTuple{2,AbstractUnitRange{Int}}) where T
    l,u = blockbandwidths(A)
    ret = BlockBandedMatrix{T}(undef, ax, (l,u))
    ret.data[1:length(A.data)] .= A.data
    ret
end

function resizedata!(::BandedColumns{DenseColumnMajor}, _, B::AbstractMatrix{T}, n::Integer, m::Integer) where T<:Number
    (n ≤ 0 || m ≤ 0) && return B
    @boundscheck checkbounds(Bool, B, n, m) || throw(ArgumentError("Cannot resize to ($n,$m) which is beyond size $(size(B))"))

    # increase size of array if necessary
    olddata = B.data
    ν,μ = B.datasize
    n,m = max(ν,n), max(μ,m)

    if (ν,μ) ≠ (n,m)
        l,u = bandwidths(B.array)
        λ,ω = bandwidths(B.data)
        if n ≥ size(B.data,1) || m ≥ size(B.data,2)
            M = 2*max(m,n+u)
            B.data = resize(olddata, M+λ, M)
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

resizedata!(laydat::BlockBandedColumns{<:AbstractColumnMajor}, layarr, B::AbstractMatrix, n::Integer, m::Integer) =
    resizedata!(laydat, layarr, B, findblock.(axes(B), (n,m))...)

function resizedata!(::BlockBandedColumns{<:AbstractColumnMajor}, _, B::AbstractMatrix{T}, N::Block{1}, M::Block{1}) where T<:Number
    (Int(N) ≤ 0 || Int(M) ≤ 0) && return B
    @boundscheck (N in blockaxes(B,1) && M in blockaxes(B,2)) || throw(ArgumentError("Cannot resize to ($N,$M) which is beyond size $(blocksize(B))"))


    N_max, M_max = Block.(blocksize(B))
    # increase size of array if necessary
    olddata = B.data
    ν,μ = B.datasize
    N_old = ν == 0 ? Block(0) : findblock(axes(B)[1], ν)
    M_old = μ == 0 ? Block(0) : findblock(axes(B)[2], μ)
    N,M = max(N_old,N),max(M_old,M)

    n,m = last.(getindex.(axes(B), (N,M)))


    if (ν,μ) ≠ (n,m)
        l,u = blockbandwidths(B.array)
        λ,ω = blockbandwidths(B.data)
        if Int(N) > blocksize(B.data,1) || Int(M) > blocksize(B.data,2)
            M̃ = 2*max(M,N+u)
            B.data = resize(olddata, (axes(B)[1][Block(1):min(M̃+λ,M_max)], axes(B)[2][Block(1):min(M̃,N_max)]))
        end
        if ν > 0 # upper-right
            KR = max(Block(1),M_old+1-ω):N_old
            JR = M_old+1:min(M,N_old+ω)
            if !isempty(KR) && !isempty(JR)
                copyto!(view(B.data, KR, JR), B.array[KR, JR])
            end
        end
        view(B.data, N_old+1:N, M_old+1:M) .= B.array[N_old+1:N, M_old+1:M]
        if μ > 0
            KR = N_old+1:min(N,M_old+λ)
            JR = max(Block(1),N_old+1-λ):M_old
            if !isempty(KR) && !isempty(JR)
                view(B.data, KR, JR) .= B.array[KR, JR]
            end
        end
        B.datasize = (n,m)
    end

    B
end

include("bandedql.jl")
include("blockkron.jl")

###
# ApplyBanded
###

struct ApplyBandedLayout{F} <: AbstractBandedLayout end

arguments(::ApplyBandedLayout{F}, A) where F = arguments(ApplyLayout{F}(), A)
sublayout(::ApplyBandedLayout{F}, A) where F = sublayout(ApplyLayout{F}(), A)

applylayout(::Type{typeof(vcat)}, ::ZerosLayout, ::AbstractBandedLayout) = ApplyBandedLayout{typeof(vcat)}()
sublayout(::ApplyBandedLayout{typeof(vcat)}, ::Type{<:NTuple{2,AbstractUnitRange}}) where J = ApplyBandedLayout{typeof(vcat)}()

applylayout(::Type{typeof(rot180)}, ::BandedColumns{LAY}) where LAY =
    BandedColumns{typeof(sublayout(LAY(), NTuple{2,StepRange{Int,Int}}))}()

applylayout(::Type{typeof(rot180)}, ::AbstractBandedLayout) =
    ApplyBandedLayout{typeof(rot180)}()

call(::MulBandedLayout, A::ApplyMatrix{<:Any,typeof(rot180)}) = *
applylayout(::Type{typeof(rot180)}, ::MulBandedLayout) = MulBandedLayout()
arguments(::MulBandedLayout, A::ApplyMatrix{<:Any,typeof(rot180)}) = ApplyMatrix.(rot180, arguments(A.args...))


bandwidths(R::ApplyMatrix{<:Any,typeof(rot180)}) = bandwidths(Applied(R))
function bandwidths(R::Applied{<:Any,typeof(rot180)})
    m,n = size(R)
    sh = m-n
    l,u = bandwidths(arguments(R)[1])
    u+sh,l-sh
end

bandeddata(R::ApplyMatrix{<:Any,typeof(rot180)}) =
    @view bandeddata(arguments(R)[1])[end:-1:1,end:-1:1]

*(A::LazyMatrix, B::AbstractBandedMatrix) = apply(*, A, B)
*(A::AbstractBandedMatrix, B::LazyMatrix) = apply(*, A, B)
*(A::AbstractBandedMatrix, b::LazyVector) = apply(*, A, b)


end