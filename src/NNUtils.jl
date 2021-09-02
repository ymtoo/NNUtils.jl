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
    dftmatrix,

    # layers
    SincConv,

    # blocks
    DepthwiseSeparableConv, 
    BottleneckResidual, 
    ResnetResidualv2,
    SqueezeExcitation,
    MBConv,

    # networks
    dft1Dfunctions,

    # mobilenets
    mobilenetv1, 
    mobilenetv1_small,

    # efficientnets
    efficientnetb0,

    # saliencymaps
    Gradient, 
    SmoothGradient, 
    saliencymap,

    # optimiser
    LARS

include("utils.jl")
include("layers/sincconv.jl")
include("networks/init.jl")
include("networks/blocks.jl")
include("networks/mobilenets.jl")
include("networks/efficientnets.jl")
include("optimiser.jl")
include("saliencymaps.jl")

end # module
