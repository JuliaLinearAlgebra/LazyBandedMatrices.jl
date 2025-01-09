module LazyBandedMatricesInfiniteArraysExt
using LazyBandedMatrices, InfiniteArrays
using LazyBandedMatrices.BlockArrays
using LazyBandedMatrices.ArrayLayouts

import Base: BroadcastStyle, copy, OneTo, oneto
import LazyBandedMatrices: _krontrav_axes, _block_interlace_axes, _broadcast_sub_arguments, AbstractLazyBandedBlockBandedLayout, KronTravBandedBlockBandedLayout, krontravargs, DiagTravLayout, krontrav_materialize_layout, krontrav
import InfiniteArrays: InfFill, TridiagonalToeplitzLayout, BidiagonalToeplitzLayout, LazyArrayStyle, OneToInf
import LazyBandedMatrices.ArrayLayouts: MemoryLayout, sublayout, RangeCumsum, Mul
import LazyBandedMatrices.BlockArrays: sizes_from_blocks, BlockedOneTo, BlockSlice1, BlockSlice
import LazyBandedMatrices.LazyArrays: BroadcastBandedLayout, AbstractPaddedLayout, simplifiable

const OneToInfCumsum = RangeCumsum{Int,OneToInf{Int}}

MemoryLayout(::Type{<:LazyBandedMatrices.Bidiagonal{<:Any,<:InfFill}}) = BidiagonalToeplitzLayout()
BroadcastStyle(::Type{<:LazyBandedMatrices.Bidiagonal{<:Any,<:InfFill}}) = LazyArrayStyle{2}()

for Typ in (:(LazyBandedMatrices.Tridiagonal{<:Any,<:InfFill,<:InfFill,<:InfFill}),
            :(LazyBandedMatrices.SymTridiagonal{<:Any,<:InfFill,<:InfFill}))
    @eval begin
        MemoryLayout(::Type{<:$Typ}) = TridiagonalToeplitzLayout()
        BroadcastStyle(::Type{<:$Typ}) = LazyArrayStyle{2}()
    end
end

LazyBandedMatrices.unitblocks(a::OneToInf) = blockedrange(Ones{Int}(length(a)))


###
# KronTrav
###

_krontrav_axes(A::OneToInf{Int}, B::OneToInf{Int}) = blockedrange(oneto(length(A)))


struct InfKronTravBandedBlockBandedLayout <: AbstractLazyBandedBlockBandedLayout end
MemoryLayout(::Type{<:KronTrav{<:Any,2,<:Any,NTuple{2,BlockedOneTo{Int,OneToInfCumsum}}}}) = InfKronTravBandedBlockBandedLayout()

sublayout(::InfKronTravBandedBlockBandedLayout, ::Type{<:NTuple{2,BlockSlice1}}) = BroadcastBandedLayout{typeof(*)}()
sublayout(::InfKronTravBandedBlockBandedLayout, ::Type{<:NTuple{2,BlockSlice{BlockRange{1,Tuple{OneTo{Int}}}}}}) = KronTravBandedBlockBandedLayout()

simplifiable(::Mul{InfKronTravBandedBlockBandedLayout, InfKronTravBandedBlockBandedLayout}) = Val(true)
copy(M::Mul{InfKronTravBandedBlockBandedLayout, InfKronTravBandedBlockBandedLayout}) = KronTrav((krontravargs(M.A) .* krontravargs(M.B))...)

_broadcast_sub_arguments(::InfKronTravBandedBlockBandedLayout, M, V) = _broadcast_sub_arguments(KronTravBandedBlockBandedLayout(), M, V)

sizes_from_blocks(A::LazyBandedMatrices.Tridiagonal, ::NTuple{2,OneToInf{Int}}) = size.(A.d, 1), size.(A.d,2)
sizes_from_blocks(A::LazyBandedMatrices.Bidiagonal, ::NTuple{2,OneToInf{Int}}) = size.(A.dv, 1), size.(A.dv,2)


_block_interlace_axes(::Int, ax::Tuple{BlockedOneTo{Int,OneToInf{Int}}}...) = (blockedrange(Fill(length(ax), ∞)),)

_block_interlace_axes(nbc::Int, ax::NTuple{2,BlockedOneTo{Int,OneToInf{Int}}}...) =
    (blockedrange(Fill(length(ax) ÷ nbc, ∞)),blockedrange(Fill(mod1(length(ax),nbc), ∞)))


# KronTrav * DiagTrav

copy(M::Mul{InfKronTravBandedBlockBandedLayout, Lay}) where Lay<:DiagTravLayout{<:AbstractPaddedLayout} = copy(Mul{KronTravBandedBlockBandedLayout, Lay}(M.A, M.B))

krontrav_materialize_layout(::InfKronTravBandedBlockBandedLayout, K) = K


end