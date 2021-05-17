module NNUtils

using Distributions
using Flux
using Random
using Statistics
using Zygote

export 

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
    saliencymap

include("layers/sincconv.jl")
include("./networks/blocks.jl")
include("./networks/mobilenets.jl")
include("saliencymaps.jl")

end # module
