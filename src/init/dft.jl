"""
Initialize `Conv` weights based on DFT.
"""
function dft1Dfunctions(nfilters, N)
    transpose(dftmatrix(nfilters, N))
end
