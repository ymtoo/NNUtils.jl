module NNUtils

using Distributions
using Flux
using LinearAlgebra
using Random
using Statistics
using Zygote

export 

    # utils
    melcutofffrequencies,

    # layers
    SincConv,

    # blocks
    DepthwiseSeparableConv, 
    BottleneckResidual, 
    ResnetResidualv2,

    # mobilenets
    mobilenetv1, 
    mobilenetv1_small,

    # saliencymaps
    Gradient, 
    SmoothGradient, 
    saliencymap,

    # optimiser
    LARS

include("utils.jl")
include("layers/sincconv.jl")
include("networks/blocks.jl")
include("networks/mobilenets.jl")
include("optimiser.jl")
include("saliencymaps.jl")

end # module
