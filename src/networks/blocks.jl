"""
Depthwise Seperable Convolutions.

# Reference
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
"""
function DepthwiseSeparableConv(k, ch, σ=identity; stride=1, pad=SamePad(), kwargs...)
    Chain(
        Conv(k, first(ch) => first(ch); stride=stride, pad=pad, groups=first(ch), kwargs...),
        #DepthwiseConv(k, first(ch) => first(ch); stride=stride, pad=pad),
        BatchNorm(first(ch), σ),
        Conv((1, 1), ch, kwargs...),
        BatchNorm(last(ch), σ)
    )
end

"""
Bottleneck residual block.

# Reference
MobileNetV2: Inverted Residuals and Linear Bottlenecks
"""
function BottleneckResidual(k, ch, σ=identity, t=1; stride=1, pad=SamePad(), kwargs...)
    tk = t * first(ch)
    block = Chain(
        Conv((1, 1), first(ch) => tk),
        BatchNorm(tk, σ),
        Conv(k, tk => tk, σ; stride=stride, pad=pad, groups=tk, kwargs...),
        BatchNorm(tk, σ),
        #DepthwiseConv(k, tk => tk, σ; stride=stride, pad=pad),
        Conv((1, 1), tk => last(ch), kwargs...),
        BatchNorm(last(ch))
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

"""
Squeeze and excitation block.

# Reference
Squeeze-and-Excitation Networks
"""
function SqueezeExcitation(ch, ratio)
    block = Chain(
        GlobalMeanPool(),
        flatten,
        Dense(ch, ch÷ratio, relu; bias=false),
        Dense(ch÷ratio, ch, sigmoid; bias=false),
        x -> reshape(x, 1, 1, size(x,1), size(x,2))
    )
    SkipConnection(block, .*)
end

"""
MBConv

# Reference
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
"""
function ConvBnAct(filter, ch, σ; stride=1, pad=SamePad(), groups=1)
    Chain(
        Conv(filter, ch,; stride=stride, pad=pad, groups=groups),
        BatchNorm(last(ch), σ)
    )
end
function MBConv(filter, ch, σ=identity, expansion_factor=1, ratio=4, p=0; stride=1, pad=SamePad())
    expanded = first(ch) * expansion_factor
    block = Chain(
        expansion_factor > 1 ? ConvBnAct((1,1), first(ch)=>expanded, σ) : identity,
        ConvBnAct(filter, expanded=>expanded, σ; stride=stride, pad=pad, groups=expanded),
        SqueezeExcitation(expanded, ratio),
        ConvBnAct(filter, expanded=>last(ch), identity),
        Dropout(p),
    )
    first(ch) == last(ch) && (stride == 1) ? SkipConnection(block, +) : block
end

"""
Time-domain filterbanks.

# Reference
1. Learning filterbanks from raw speech for phone recognition
2. https://github.com/facebookresearch/tdfbanks
"""
function togaborweights(gfilters)
    m, n = size(gfilters)
    weight = zeros(Float32, m, 1, 1, 2 * n)
    i = 1
    for gfilter ∈ eachcol(gfilters)
        weight[:,1,1,i] = real(gfilter)
        weight[:,1,1,i+1] = imag(gfilter)
        i += 2
    end
    weight
end
function towindowweights(window, m, n)
    weight = zeros(Float32, m, 1, 1, n)
    for i ∈ 1:n
        weight[:,1,1,i] = window(m)
    end
    weight
end
function TDFilterbanks(fs, wlen, wstride, ch; stride=1)
    window_size = (fs * wlen) ÷ 1000 + 1
    window_stride = (fs * wstride) ÷ 1000 ÷ stride
    m = last(ch) ÷ 2
    gfilters, _ = gaborfilters(nfilters=m, fs=fs, wlen=wlen)
    gaborweight = togaborweights(gfilters)
    hannweight = towindowweights(hanning, window_size, m)
    Chain(
        Conv((window_size,1), ch; stride=stride, pad=SamePad(), bias=false, weight=gaborweight),
        x -> abs2.(x),
        x -> reshape(x, size(x,1), last(ch), 1, :),
        MeanPool((1,2)),
        x -> reshape(x, size(x,1), 1, m, :),
        Conv((window_size,1), m=>m; stride=window_stride, groups=m, pad=0, bias=false, weight=hannweight),
        x -> log1p.(abs.(x)),
        InstanceNorm(m, momentum=1f0),
        x -> reshape(x, size(x,1), m, 1, :)
    )
end