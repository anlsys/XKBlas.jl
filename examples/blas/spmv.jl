using LinearAlgebra, Random, SparseArrays, SparseMatricesCSR, XK

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

# Example usage
m  = 16 # 16384
n  = m
ts = 2
density=1.0
rows, cols, values, A = random_csr_arrays(m, n, density=density)
nnz = length(values)

const T = eltype(values)

X = rand(n)
Y = 0.0 * rand(m)
alpha = T(1.0)
beta  = T(0.0)
transA = XK.BLAS.NO_TRANS
format = XK.BLAS.SPARSE_CSR

XK.BLAS.set_tile_parameter(ts)

@time begin
    XK.BLAS.spmv(alpha, transA, m, n, nnz, format, rows, cols, values, X, beta, Y)
end

if (n <= 64)
    println("A =")
    display(LinearAlgebra.Matrix(A))  # dense view for clarity
    println("X = ", X)
    println("XK Y = ", Y)
    println(" Julia Y = ", A * X)
end
