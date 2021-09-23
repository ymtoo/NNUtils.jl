"""
Pretrained audio neural networks (PANNs).
"""

"""
CNN10.

# Reference
Panns: Large-scale pretrained audio neural networks for audio pattern recognition.
"""
function cnn10(in, out, σ=relu, out_act=identity)
    k = 64
    Chain(
        Conv((3,3), in=>k; pad=SamePad(), bias=false),
        BatchNorm(k, σ),
        Conv((3,3), k=>k; pad=SamePad(), bias=false),
        BatchNorm(k, σ),
        MeanPool((2,2)),
        Conv((3,3), k=>2k; pad=SamePad(), bias=false),
        BatchNorm(2k, σ),
        Conv((3,3), 2k=>2k; pad=SamePad(), bias=false),
        BatchNorm(2k, σ),
        MeanPool((2,2)),
        Conv((3,3), 2k=>4k; pad=SamePad(), bias=false),
        BatchNorm(4k, σ),
        Conv((3,3), 4k=>4k; pad=SamePad(), bias=false),
        BatchNorm(4k, σ),
        MeanPool((2,2)),
        Conv((3,3), 4k=>8k; pad=SamePad(), bias=false),
        BatchNorm(8k, σ),
        Conv((3,3), 8k=>8k; pad=SamePad(), bias=false),
        BatchNorm(8k, σ),
        GlobalMeanPool(),
        flatten,
        Dense(8k, out, out_act)
    )
end