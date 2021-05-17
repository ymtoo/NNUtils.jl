using NNUtils
using Flux, Test

import Flux: CuArray

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
    f1s, bws = NNUtils.initcutofffreqs(dims, fs)
    @test length(f1s) == length(bws) == dims[4]
    weight = NNUtils.sincfunctions(f1s, bws, dims, fs)
    @test size(weight) == dims
    @test eltype(weight) == eltype(f1s) == eltype(bws) == typeof(fs)

    x = randn(Float32, 4800, 1, 1, 16) |> gpu
    model = Chain(SincConv(fs, (200,1), 1=>8)) |> gpu
    @test typeof(model(x)) == CuArray{Float32, 4}
    
end