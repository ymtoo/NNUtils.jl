"""
MobileNetV1

# Reference
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
"""
function mobilenetv1(in, out, σ=relu)
    Chain(
        Conv((3, 3), in=>32, σ; stride=2),
        BatchNorm(32),
        DepthwiseSeparableConv((3, 3), 32=>64, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 64=>128, σ; stride=2),
        DepthwiseSeparableConv((3, 3), 128=>128, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 128=>256, σ; stride=2),
        DepthwiseSeparableConv((3, 3), 256=>256, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 256=>512, σ; stride=2),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 512=>1024, σ; stride=2),
        DepthwiseSeparableConv((3, 3), 1024=>1024, σ; stride=2),
        GlobalMeanPool(),
        x -> reshape(x, 1024, :),
        Dense(1024, out, σ),
        softmax
    )
end
function mobilenetv1_small(in, out, σ=relu)
    Chain(
        Conv((3, 3), in=>32, σ; stride=2),
        BatchNorm(32),
        DepthwiseSeparableConv((3, 3), 32=>64, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 64=>128, σ; stride=2),
        DepthwiseSeparableConv((3, 3), 128=>128, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 128=>256, σ; stride=2),
        DepthwiseSeparableConv((3, 3), 256=>256, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 256=>512, σ; stride=2),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        DepthwiseSeparableConv((3, 3), 512=>512, σ; stride=1),
        GlobalMeanPool(),
        x -> reshape(x, 512, :),
        Dense(512, out, σ),
        softmax
    )
end

"""
MobileNetV2

# Reference
"""
function mobilenetv2(in ,out; σ=reule)
end

"""
MobileNetV3

# Reference
"""
function mobilenetv3(in, out; σ=reule)
end