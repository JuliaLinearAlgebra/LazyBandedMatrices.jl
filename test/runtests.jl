import Random

Random.seed!(0)

include("test_tridiag.jl")
include("test_bidiag.jl")
include("test_special.jl")
include("test_banded.jl")
include("test_block.jl")
include("test_misc.jl")
include("test_blockkron.jl")
include("test_blockconcat.jl")
