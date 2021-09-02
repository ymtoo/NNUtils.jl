"""
EfficientNet-B0 baseline network.

# Reference
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
"""
function efficientnetb0(in, out, σ=swish)
    Chain(
        Conv((3,3), in=>32, σ; stride=2, pad=SamePad()),
        MBConv((3,3), 32=>16, σ, 1, 4; stride=1, pad=SamePad()), # layer 1
        MBConv((3,3), 16=>24, σ, 6, 4; stride=2, pad=SamePad()), # layer 2
        MBConv((3,3), 24=>24, σ, 6, 4; stride=1, pad=SamePad()),
        MBConv((5,5), 24=>40, σ, 6, 4; stride=2, pad=SamePad()),  # layer 3
        MBConv((5,5), 40=>40, σ, 6, 4; stride=1, pad=SamePad()),
        MBConv((3,3), 40=>80, σ, 6, 4; stride=2, pad=SamePad()),  # layer 4
        MBConv((3,3), 80=>80, σ, 6, 4; stride=1, pad=SamePad()),
        MBConv((3,3), 80=>80, σ, 6, 4; stride=1, pad=SamePad()),
        MBConv((5,5), 80=>112, σ, 6, 4; stride=1, pad=SamePad()), # layer 5
        MBConv((5,5), 112=>112, σ, 6, 4; stride=1, pad=SamePad()),
        MBConv((5,5), 112=>112, σ, 6, 4; stride=1, pad=SamePad()),
        MBConv((5,5), 112=>192, σ, 6, 4; stride=2, pad=SamePad()), # layer 6
        MBConv((5,5), 192=>192, σ, 6, 4; stride=1, pad=SamePad()),
        MBConv((5,5), 192=>192, σ, 6, 4; stride=1, pad=SamePad()),
        MBConv((5,5), 192=>192, σ, 6, 4; stride=1, pad=SamePad()),
        MBConv((3,3), 192=>320, σ, 6, 4; stride=1, pad=SamePad()), # layer 7
        Conv((1,1), 320=>1280, σ; pad=SamePad()),
        GlobalMeanPool(),
        flatten,
        Dense(1280, out, σ)
    )
end