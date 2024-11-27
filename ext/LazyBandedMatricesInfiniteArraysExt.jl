module LazyBandedMatricesInfiniteArraysExt

MemoryLayout(::Type{<:Bidiagonal{<:Any,<:InfFill}}) = BidiagonalToeplitzLayout()
BroadcastStyle(::Type{<:Bidiagonal{<:Any,<:InfFill}}) = LazyArrayStyle{2}()

for Typ in (:(LazyBandedMatrices.Tridiagonal{<:Any,<:InfFill,<:InfFill,<:InfFill}),
            :(LazyBandedMatrices.SymTridiagonal{<:Any,<:InfFill,<:InfFill}))
    @eval begin
        MemoryLayout(::Type{<:$Typ}) = TridiagonalToeplitzLayout()
        BroadcastStyle(::Type{<:$Typ}) = LazyArrayStyle{2}()
    end
end


end