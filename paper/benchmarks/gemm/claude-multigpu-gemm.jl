using CUDA
using CUDA.CUBLAS

"""
    multi_gpu_gemm(A::Matrix{Float32}, B::Matrix{Float32}, C::Matrix{Float32}=zeros(Float32, size(A,1), size(B,2)))

Perform matrix multiplication C = A * B across multiple GPUs using CUBLAS.
Distributes rows of A across available GPUs.

# Arguments
- `A`: MxK matrix
- `B`: KxN matrix
- `C`: MxN result matrix (optional, defaults to zeros)

# Returns
- Result matrix C = A * B
"""
function multi_gpu_gemm(A::Matrix{Float32}, B::Matrix{Float32},
                        C::Matrix{Float32}=zeros(Float32, size(A,1), size(B,2)))
    M, K = size(A)
    K2, N = size(B)

    @assert K == K2 "Inner dimensions must match: A is $(M)x$(K), B is $(K2)x$(N)"
    @assert size(C) == (M, N) "C must be $(M)x$(N)"

    # Get number of available GPUs
    num_gpus = length(devices())
    println("Using $num_gpus GPU(s)")

    if num_gpus == 0
        error("No CUDA devices available")
    end

    # Split A into chunks across GPUs
    rows_per_gpu = ceil(Int, M / num_gpus)

    # Each GPU computes a subset of rows
    tasks = []
    for gpu_id in 0:(num_gpus-1)
        task = Threads.@spawn begin
            # Set device
            device!(gpu_id)

            # Calculate row range for this GPU
            start_row = gpu_id * rows_per_gpu + 1
            end_row = min((gpu_id + 1) * rows_per_gpu, M)

            if start_row > M
                return nothing
            end

            # Extract submatrix for this GPU
            A_sub = A[start_row:end_row, :]

            # Transfer data to GPU
            d_A = CuArray(A_sub)
            d_B = CuArray(B)
            d_C = CUDA.zeros(Float32, size(A_sub, 1), N)

            # Perform GEMM: C = alpha * A * B + beta * C
            alpha = Float32(1.0)
            beta = Float32(0.0)

            CUBLAS.gemm!('N', 'N', alpha, d_A, d_B, beta, d_C)

            # Copy result back to host
            C_sub = Array(d_C)

            # Synchronize device
            CUDA.synchronize()

            return (start_row:end_row, C_sub)
        end
        push!(tasks, task)
    end

    # Collect results from all GPUs
    for task in tasks
        result = fetch(task)
        if result !== nothing
            row_range, C_sub = result
            C[row_range, :] = C_sub
        end
    end

    return C
end

"""
    benchmark_multi_gpu_gemm(M::Int, K::Int, N::Int; num_trials::Int=5)

Benchmark multi-GPU GEMM performance.
"""
function benchmark_multi_gpu_gemm(M::Int, K::Int, N::Int; num_trials::Int=5)
    println("\nBenchmarking Multi-GPU GEMM")
    println("="^50)
    println("Matrix sizes: A=$(M)x$(K), B=$(K)x$(N)")

    # Generate random matrices
    A = rand(Float32, M, K)
    B = rand(Float32, K, N)

    # Warmup
    println("\nWarming up...")
    _ = multi_gpu_gemm(A, B)

    # Benchmark
    println("\nRunning $num_trials trials...")
    times = Float64[]

    for trial in 1:num_trials
        GC.gc()
        CUDA.reclaim()

        t_start = time()
        C = multi_gpu_gemm(A, B)
        t_end = time()

        elapsed = t_end - t_start
        push!(times, elapsed)

        # Calculate GFLOPS (2*M*N*K operations)
        gflops = (2.0 * M * N * K) / elapsed / 1e9
        println("Trial $trial: $(round(elapsed*1000, digits=2)) ms, $(round(gflops, digits=2)) GFLOPS")
    end

    avg_time = sum(times) / length(times)
    avg_gflops = (2.0 * M * N * K) / avg_time / 1e9

    println("\nAverage: $(round(avg_time*1000, digits=2)) ms, $(round(avg_gflops, digits=2)) GFLOPS")
    println("="^50)

    return avg_time, avg_gflops
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("Multi-GPU GEMM using CUBLAS")
    println("Available GPUs: $(length(devices()))")

    # Test with small matrices
    M, K, N = 1024, 1024, 1024
    A = rand(Float32, M, K)
    B = rand(Float32, K, N)

    println("\nComputing C = A * B")
    C_gpu = multi_gpu_gemm(A, B)

    # Verify correctness
    C_cpu = A * B
    max_error = maximum(abs.(C_gpu - C_cpu))
    println("Max error vs CPU: $max_error")

    # Run benchmark
    benchmark_multi_gpu_gemm(4096, 4096, 4096, num_trials=3)
end
