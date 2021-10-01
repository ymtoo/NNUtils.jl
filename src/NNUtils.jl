module NNUtils

using Distributions
using DSP
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
    TDFilterbanks,

    # init
    dft1Dfunctions,
    gaborfilters,

    # mobilenets
    mobilenetv1, 

    # efficientnets
    efficientnetb0,

    # panns
    cnn10,

    # saliencymaps
    Gradient, 
    SmoothGradient, 
    saliencymap,

    # optimiser
    LARS

include("utils.jl")
include("layers/sincconv.jl")
include("init/gabor.jl")
include("init/dft.jl")
include("networks/blocks.jl")
include("networks/mobilenets.jl")
include("networks/efficientnets.jl")
include("networks/panns.jl")
include("optimiser.jl")
include("saliencymaps.jl")

end # module
