using NNUtils
using Flux
using Flux: CuArray
using Test
using Random

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
    f1s, f2s = NNUtils.initcutofffreqs(dims, fs)
    @test length(f1s) == length(f2s) == dims[4]
    weight = NNUtils.sincfunctions(f1s, f2s, dims, fs)
    @test size(weight) == dims
    @test eltype(weight) == eltype(f1s) == eltype(f2s) == typeof(fs)

    x = randn(Float32, 4800, 1, 1, 16) |> gpu
    model = Chain(SincConv(fs, (200,1), 1=>8)) |> gpu
    @test typeof(model(x)) == CuArray{Float32, 4}
    
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