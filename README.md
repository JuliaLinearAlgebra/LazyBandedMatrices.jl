# LazyBandedMatrices.jl
A Julia package for lazy banded matrices

[![Build Status](https://travis-ci.org/JuliaMatrices/LazyBandedMatrices.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/LazyBandedMatrices.jl) 

[![codecov](https://codecov.io/gh/JuliaMatrices/LazyBandedMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMatrices/LazyBandedMatrices.jl)

This package supports lazy banded and block-banded matrices, for example, a lazy multiplication of banded matrices:

```julia
julia> using LazyBandedMatrices, LazyArrays, BandedMatrices

julia> A = brand(10,10,1,1);

julia> ApplyMatrix(*, A, A)
10×10 ApplyArray{Float64,2,typeof(*),Tuple{BandedMatrix{Float64,Array{Float64,2},Base.OneTo{Int64}},BandedMatrix{Float64,Array{Float64,2},Base.OneTo{Int64}}}}:
 0.191109   0.379118   0.318899   ⋅        …   ⋅         ⋅         ⋅      
 0.329746   0.728074   1.12126   0.315324      ⋅         ⋅         ⋅      
 0.0341854  0.138194   0.95911   0.569674      ⋅         ⋅         ⋅      
  ⋅         0.0561613  0.823235  1.19154       ⋅         ⋅         ⋅      
  ⋅          ⋅         0.542728  0.5989        ⋅         ⋅         ⋅      
  ⋅          ⋅          ⋅        0.819362  …  0.113575   ⋅         ⋅      
  ⋅          ⋅          ⋅         ⋅           0.769278  0.623466   ⋅      
  ⋅          ⋅          ⋅         ⋅           0.577351  1.05373   0.590068
  ⋅          ⋅          ⋅         ⋅           0.321916  1.7937    1.39854 
  ⋅          ⋅          ⋅         ⋅           0.201658  1.5645    1.3461  
```