module NNUtils

using Flux
using Statistics
using Zygote

export 

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

include("./networks/blocks.jl")
include("./networks/mobilenets.jl")
include("saliencymaps.jl")

end # module
