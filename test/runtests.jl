using NNUtils
using FFTW
using Flux
using Flux: CuArray
using Test
using Random

T = Float32
x = randn(T, 288, 1, 12, 1)
y = randn(T, 9, 1)

@testset "networks/blocks" begin

    fs = 9600
    wlen = 25
    wstride = 10
    nfilters = 80
    batchsize = 8
    tdf = TDFilterbanks(fs, wlen, wstride, 1=>nfilters)
    window_size = (fs * wlen) ÷ 1000 + 1
    window_stride = (fs * wstride) ÷ 1000
    for l ∈ [2400, 4800, 9600]
        x_ts = randn(T, l, 1, 1, batchsize)
        @test size(tdf(x_ts)) == (ceil(Int, (l-(2 * (window_size ÷ 2))) / (window_stride)), nfilters ÷ 2, 1, batchsize) 
    end

    model1 = Chain(DepthwiseSeparableConv((11,1), 12=>1, relu; pad=SamePad()),
                  flatten,
                  Dense(288,9,sigmoid))
    model2 = Chain(BottleneckResidual((11,1), 12=>1, relu; pad=SamePad()),
                  flatten,
                  Dense(288,9,sigmoid))
    model3 = Chain(
        MBConv((3,1), 12=>12, relu, 1; pad=SamePad()),
        MBConv((3,1), 12=>1, relu, 6; pad=SamePad()),
        flatten,
        Dense(288,9,sigmoid)
    )
    for model ∈ [model1, model2, model3]
        ps = Flux.params(model)
        gs = gradient(ps) do 
            Flux.Losses.mse(model(x), y)
        end
        modelgpu = gpu(model)
        psgpu = Flux.params(modelgpu)
        gsgpu = gradient(psgpu) do 
            Flux.Losses.mse(modelgpu(gpu(x)), gpu(y))
        end
    end
    
    @test size(SqueezeExcitation(size(x, 3), 2)(x)) == size(x) 
end

@testset "networks/panns" begin
    in = 1
    out = 64
    batchsize = 8
    model = cnn10(in,out,relu,tanh)
    x = randn(Float32,50,80,1,batchsize)
    y = model(x)
    @test size(y) == (out,batchsize)
end

@testset "salencymaps" begin
    model = Chain(Conv((10,1), 12=>1, relu; pad=SamePad()),
                  flatten,
                  Dense(288, 9, sigmoid))
    sm1 = saliencymap(Gradient(), model, x)
    sm2 = saliencymap(SmoothGradient(50, 50), model, x)
    @test size(sm1) == size(sm2) == (288, 1, 12)
end

@testset "layers/sincconv" begin
    @test NNUtils.gett(200) == -99:100
    @test NNUtils.gett(201) == -100:100

    @test NNUtils.gett(200, 1) == reshape(-99:100, 200, 1)
    @test NNUtils.gett(1, 200) == reshape(-99:100, 1, 200)

    @test NNUtils.gett(200, 3) == [99:-1:-100 zeros(Int, 200) -99:100]
    @test NNUtils.gett(3, 200) == [reshape(99:-1:-100, 1, 200); 
                                   zeros(Int, 1, 200); 
                                   reshape(-99:100, 1, 200)]

    dims = (200, 1, 1, 8)
    fs = 9600f0
    f1srand, f2srand = NNUtils.initrandcutofffreqs(dims, fs)
    f1smel, f2smel = NNUtils.initmelcutofffreqs(dims, fs)
    @test length(f1srand) == length(f2srand) == length(f1smel) == length(f2smel) == dims[4]
    weightrand = NNUtils.sincfunctions(f1srand, f2srand, dims, fs)
    weightmel = NNUtils.sincfunctions(f1smel, f2smel, dims, fs)
    @test size(weightrand) == size(weightmel) == dims
    @test eltype(weightrand) == eltype(weightmel) == eltype(f1srand) == eltype(f2srand) == eltype(f1smel) == eltype(f2smel) == typeof(fs)

    x = randn(Float32, 201, 1, 1, 16) 
    model = Chain(SincConv(fs, (201,1), 1=>8)) 
    @test typeof(model(x)) == Array{Float32, 4}

    model = Chain(SincConv(fs, (201,1), 1=>8), flatten) 
    output = randn(Float32, 8, 16)
    ps = Flux.params(model)
    opt = ADAM(0.1)
    loss(x, y) = Flux.mse(model(x), y)
    l1 = loss(x, output)
    for t ∈ 1:3000
        gs = Flux.gradient(ps) do
            loss(x,output)
        end
        Flux.Optimise.update!(opt, ps, gs)
        ps[1] .*= sign.(ps[1]) 
        ps[2] .= ps[1] .+ abs.(ps[2] .- ps[1])
    end
    @test loss(x, output) < l1
end

@testset "Optimise" begin
    # Ensure rng has different state inside and outside the inner @testset
    # so that w and w' are different
    Random.seed!(84)
    w = randn(10, 10)
    @testset for opt in [LARS()]
        Random.seed!(42)
        w′ = randn(10, 10)
        b = Flux.Zeros()
        loss(x) = Flux.Losses.mse(w*x, w′*x .+ b)
        for t = 1: 10^5
          θ = params([w′, b])
          x = rand(10)
          θ̄ = gradient(() -> loss(x), θ)
          Flux.Optimise.update!(opt, θ, θ̄)
        end
        @test loss(rand(10, 10)) < 0.01
    end
end

@testset "utils" begin
    
    nfilters = 12
    fs = 9600
    lowcutoffs, highcutoffs = melcutofffrequencies(nfilters, fs)
    @test length(lowcutoffs) == length(highcutoffs) == nfilters
    @test lowcutoffs ≈ [0.,120.28458738,261.23829184,426.41279371,
                         619.97007672,846.78729962,1112.57968834,1424.04454534,
                         1789.03000492,2216.73278652,2717.92992897,3305.25034496] atol=1e-6
    @test highcutoffs ≈ [261.23829184,426.41279371,619.97007672,846.78729962, 
                          1112.57968834,1424.04454534,1789.03000492,2216.73278652,
                          2717.92992897,3305.25034496,3993.49303795, 4800] atol=1e-6

    N = 257
    x = randn(N)
    W = dftmatrix(N)
    y1 = W * x
    y2 = fft(x)
    @test y1 .* √N ≈ y2 atol=1e-9
end