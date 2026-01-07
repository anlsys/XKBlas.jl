using LinearAlgebra, Random, SparseArrays, SparseMatricesCSR, XKLas
const XK = XKLas

function random_csr_arrays(m::Int, n::Int; density::Float64=0.2, rng=Random.default_rng())
    # Step 1: generate a random sparse CSC matrix with Float64 values
    A_csc = sprand(rng, m, n, density)  # default element type = Float64

    # Step 2: convert to CSR format
    A_csr = SparseMatrixCSR(A_csc)

    # Step 3: extract internal arrays (do not modify directly unless you know what you’re doing)
    rowptr = A_csr.rowptr      # Vector{Int} of length m+1
    colind = A_csr.colval      # Vector{Int} of column indices
    values = A_csr.nzval       # Vector{Float64} of nonzero values

    if (n <= 64)
        println(rowptr)
        println(colind)
    end

    return rowptr, colind, values, A_csr
end

# ----------------------------
# Command-line arguments
# ----------------------------
# Usage:
#   julia script.jl [n] [density]
#
n       = length(ARGS) >= 1 ? parse(Int,     ARGS[1]) : 4
density = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0

m  = n
rows, cols, values, A = random_csr_arrays(m, n, density=density)
nnz = length(values)

const T = eltype(values)

X = rand(n)
Y = 0.0 * rand(m)
alpha = T(1.0)
beta  = T(0.0)
transA = XK.BLAS.NO_TRANS
format = XK.BLAS.SPARSE_CSR

# Set tile size parameter
ngpus = XK.get_ngpus()
ts = div(n, ngpus)
XK.set_tile_parameter(ts)

for i in 1:16
    @time begin
        XK.BLAS.spmv_sync(alpha, transA, m, n, nnz, format, rows, cols, values, X, beta, Y)
    end
end
