"""
MobileNetV1

# Reference
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
"""
function mobilenetv1(in, out, σ=relu, out_act=identity)
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
        Dense(1024, out, out_act),
    )
end

"""
MobileNetV2

# Reference
Mobilenetv2: Inverted residuals and linear bottlenecks
"""
function mobilenetv2(in ,out; σ=relu6, out_act=identity)
    filter = (3,3)
    Chain(
        Conv(filter, in=>32, σ; stride=2, pad=SamePad()),
        BottleneckResidual(filter, 32=>16, σ; stride=1),
        BottleneckResidual(filter, 16=>24, σ; stride=2),
        BottleneckResidual(filter, 24=>24, σ; stride=1),
        BottleneckResidual(filter, 24=>32, σ; stride=2),
        BottleneckResidual(filter, 32=>32, σ; stride=1),
        BottleneckResidual(filter, 32=>32, σ; stride=1),
        BottleneckResidual(filter, 32=>64, σ; stride=2),
        BottleneckResidual(filter, 64=>64, σ; stride=1),
        BottleneckResidual(filter, 64=>64, σ; stride=1),
        BottleneckResidual(filter, 64=>64, σ; stride=1),
        BottleneckResidual(filter, 64=>96, σ; stride=1),
        BottleneckResidual(filter, 96=>96, σ; stride=1),
        BottleneckResidual(filter, 96=>96, σ; stride=1),
        BottleneckResidual(filter, 96=>160, σ; stride=2),
        BottleneckResidual(filter, 160=>160, σ; stride=1),
        BottleneckResidual(filter, 160=>320, σ; stride=1),
        Conv((1,1), 320=>1280, σ; stride=1),
        MeanPool((7,7)),
        Conv((1,1), 1280=>out, out_act)
    )
end

"""
MobileNetV3

# Reference
"""
function mobilenetv3(in, out; σ=relu)
end