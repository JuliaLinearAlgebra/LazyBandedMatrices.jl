struct BlockInterlace{T, N, Arrays} <: AbstractBlockArray{T, N}
    nbc::Int # Number of block rows
    arrays::Arrays
end

BlockInterlace{T,1,Arrays}(arrays::AbstractVector...) where {T,N,Arrays} = 
    BlockInterlace{T,1,Arrays}(1, arrays)
BlockInterlace{T}(arrays::AbstractVector...) where T = 
    BlockInterlace{T,1,typeof(arrays)}(arrays...)
BlockInterlace(arrays::AbstractVector...) = 
    BlockInterlace{mapreduce(eltype, promote_type, arrays)}(arrays...)

BlockInterlace{T}(nbc::Int, arrays::AbstractMatrix...) where T = 
    BlockInterlace{T,2,typeof(arrays)}(nbc, arrays)
BlockInterlace(nbc::Int, arrays::AbstractMatrix...) = 
    BlockInterlace{mapreduce(eltype, promote_type, arrays)}(nbc, arrays...)


function _block_interlace_axes(::Int, ax::Tuple{OneTo{Int}}...)
    n = max(length.(first.(ax))...)
    (blockedrange(Fill(length(ax), n)),)
end

function _block_interlace_axes(nbc::Int, ax::NTuple{2,OneTo{Int}}...)
    n,m = max(length.(first.(ax))...),max(length.(last.(ax))...)
    (blockedrange(Fill(length(ax) ÷ nbc, n)),blockedrange(Fill(mod1(length(ax),nbc), m)))
end

axes(A::BlockInterlace) = _block_interlace_axes(A.nbc, map(axes, A.arrays)...)
size(A::BlockInterlace) = map(length, axes(A))

function getindex(A::BlockInterlace{<:Any,1}, k::Int)
    K = findblockindex(axes(A,1), k)
    A.arrays[blockindex(K)][Int(block(K))]
end

function getindex(A::BlockInterlace{<:Any,2}, k::Int, j::Int)
    K,J = findblockindex.(axes(A), (k,j))
    A.arrays[(blockindex(K)-1)*A.nbc + blockindex(J)][Int(block(K)), Int(block(J))]
end

blockbandwidths(A::BlockInterlace{<:Any,2}) = max.(map(bandwidths,A.arrays)...)
subblockbandwidths(A::BlockInterlace{<:Any,2}) = length(axes(A,1)[Block(1)]),length(axes(A,2)[Block(2)])