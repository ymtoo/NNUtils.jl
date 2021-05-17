# NNUtils
This package provides neural net utilities:
- standard networks
  - [x] mobilenets
  - [ ] resnet
- neural network blocks
  - [x] DepthwiseSeparableConv
  - [x] BottleneckResidual
  - [x] ResnetResidualv1
  - [x] ResnetResidualv2
- neural network layers
  - [x] Sinc Conv
- saliency maps 
  - [x] Gradient
  - [x] SmoothGradient
  
based on [Flux.jl](https://github.com/FluxML/Flux.jl).


## Installation
```julia
using Pkg; pkg"add https://github.com/ymtoo/NNUtils.jl.git"
```

## Usage
### SincConv
```julia-repl
julia> using Flux, NNUtils

julia> fs = 9600f0
9600.0f0

julia> model = SincConv(fs, (200, 1), 1=>8, identity)
SincConv(9600.0, (200, 1), 1=>8)

julia> params(model) |> length
2

julia> params(model)[1] |> size
(1, 8)

julia> params(model)[2] |> size
(1, 8)

julia> x = randn(Float32, 4800, 1, 1, 16)
4800×1×1×16 Array{Float32, 4}:
[:, :, 1, 1] =
 1.3832613
 0.42255098
 ⋮
 0.3134887

[:, :, 1, 2] =
 2.0712132
 0.13467419
 ⋮

 julia> model(x)
 4601×1×8×16 Array{Float32, 4}:
 [:, :, 1, 1] =
   -3498.0605
  -17983.041
   20055.25
   -1296.112
       ⋮
    1844.9044
   10672.3125
  -12256.136
 
 [:, :, 2, 1] =
    521.0923
   8794.186
   4179.6655
  -4084.1067
      ⋮
```