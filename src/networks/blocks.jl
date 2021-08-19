"""
Depthwise Seperable Convolutions.

# Reference
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
"""
function DepthwiseSeparableConv(k, ch, σ=identity; stride=1, pad=SamePad())
    Chain(
        Conv(k, first(ch) => first(ch); stride=stride, pad=pad, groups=first(ch)),
        #DepthwiseConv(k, first(ch) => first(ch); stride=stride, pad=pad),
        BatchNorm(first(ch), σ),
        Conv((1, 1), ch),
        BatchNorm(last(ch), σ)
    )
end

"""
Bottleneck residual block.

# Reference
MobileNetV2: Inverted Residuals and Linear Bottlenecks
"""
function BottleneckResidual(k, ch, σ=identity, t=1; stride=1, pad=SamePad())
    tk = t * first(ch)
    block = Chain(
        Conv((1, 1), first(ch) => tk, σ),
        Conv(k, tk => tk, σ; stride=stride, pad=pad, groups=tk),
        #DepthwiseConv(k, tk => tk, σ; stride=stride, pad=pad),
        Conv((1, 1), tk => last(ch)) 
    )
    (first(ch) == last(ch) && (stride == 1)) ? SkipConnection(block, +) : block
end

"""
Resnet Residual Block V1.

# Reference
Deep Residual Learning for Image Recognition.
"""
function ResnetResidualv1(filter, ch, σ=identity; stride=1, pad=SamePad())
    block = Chain(
        Conv(filter, ch; stride=stride, pad=pad),
        BatchNorm(last(ch), σ),
        Conv(filter, last(ch)=>last(ch); stride=stride, pad=pad),
        BatchNorm(last(ch), σ)
    )
    first(ch) == last(ch) && (stride == 1) ? Chain(SkipConnection(block, +), x -> σ.(x)) : Chain(block, x -> σ.(x))
end

"""
Resnet Residual Block V2

# Reference
Identity Mappings in Deep Residual Networks.
"""
function ResnetResidualv2(filter, ch, σ=identity; stride=1, pad=SamePad())
    block = Chain(
        BatchNorm(first(ch), σ),
        Conv(filter, ch; stride=stride, pad=pad),
        BatchNorm(last(ch), σ),
        Conv(filter, last(ch)=>last(ch); stride=stride, pad=pad)
    )
    first(ch) == last(ch) && (stride == 1) ? SkipConnection(block, +) : block
end