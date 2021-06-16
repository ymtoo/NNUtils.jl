"""
Initialize `Conv` weights based on DFT.
"""
function dft1Dfunctions(dims)
    N = first(dims)
    M = last(dims)
    reshape(transpose(dftmatrix(M, N)), N, 1, 1, M)# |> real
end