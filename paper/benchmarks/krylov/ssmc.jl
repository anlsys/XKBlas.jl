# ----------------------------
# Command-line arguments
# ----------------------------

using LinearAlgebra, SparseArrays, SparseMatricesCSR
using Krylov
using SuiteSparseMatrixCollection, MatrixMarket

function ssmc_get_matrix(name::String)
    # download matrix if not present
    ssmc = ssmc_db()
    matrix = ssmc[ssmc.name .== name, :]
    paths = fetch_ssmc(matrix, format="MM")
    @assert length(paths) == 1
    path = "$(paths[1])/$(name).mtx"
    println(path)

    # read from file
    M = MatrixMarket.mmread(path)

    # convert CSC to CSR
    return SparseMatrixCSR(M)
end

# Solver to use
fname      = length(ARGS) >= 1 ?               ARGS[1]  : "cg"
matrix     = length(ARGS) >= 2 ?               ARGS[2]  : "bcsstk02"
ts         = length(ARGS) >= 3 ? parse(Int,    ARGS[3]) : 8192
use_xkblas = length(ARGS) >= 4 ? parse(Bool,   ARGS[4]) : false
iter       = length(ARGS) >= 5 ? parse(Int,    ARGS[5]) : 5
itermax    = length(ARGS) >= 6 ? parse(Int,    ARGS[6]) : 10000
println("Running fname=$(fname) on matrix $(matrix)")
println("To change parameters, run as `julia script.jl [solver:String] [matrix-name:String] [tiles-size:Int] [use-xkblas:Boolean] [iter:Int]`")

# Get matrix
A = ssmc_get_matrix(matrix)
@assert size(A, 1) == size(A, 2)
n = size(A, 1)
y = rand(n)
println("Matrix loaded")

# Get solver
solver = getproperty(Krylov, Symbol(fname))
tolerance = 1.0e-6

# Set XKBlas backend
if use_xkblas
    using XK
    XK.set_tile_parameter(ts)
    include("./overrides.jl")
    println("Using XKBLAS")
else
    println("Not using XKBLAS")
    using CUDA, CUDA.CUSPARSE
    @assert CUDA.functional()
end

# TODO: memory is not pinned currently

# Run
for i in 1:iter
    @time begin

        if use_xkblas
            # With XKBlas, directly use host memory. Ask for CPU write back explicitly
            (x, stats) = solver(A, y, itmax = itermax)
            XK.BLAS.memory_coherent_sync(x) # TODO: overload krylov solver to automatically do that
        else
            # With CUDA, move memory synchronously first, pass GPU objects to Krylov, and write back
            A_gpu = CuSparseMatrixCSR(A)
            y_gpu = CuVector(y)
            (x_gpu, stats) = solver(A_gpu, y_gpu, itmax = itermax)
            x = Vector{Float64}(undef, n)
            copyto!(x, x_gpu)
        end

        println(stats)

        # Check result
        if false
            r = y - A * x
            if n <= 64
                println("       x is $(x)")
                println("       y is $(y)")
            end
            resid = norm(r) / norm(y)
            println("residual is $(resid)")
            @assert resid ≤ tolerance "Failure"
            println("Success")
        end
    end

    # TODO
#    if use_xkblas
#        XK.memory_invalidate_caches()
#    end

end
