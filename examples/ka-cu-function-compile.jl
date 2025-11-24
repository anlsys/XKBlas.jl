using KernelAbstractions
using CUDA
using GPUCompiler

function get_native_ptr_direct(kernel_func::Function, argtypes::Type{<:Tuple})
    gpu_kernel_name = Symbol(:gpu_, nameof(kernel_func))
    if isdefined(@__MODULE__, gpu_kernel_name)
        gpu_kernel = getfield(@__MODULE__, gpu_kernel_name)
        kernel = CUDA.cufunction(gpu_kernel, argtypes)
        println(kernel.fun)
        return Base.unsafe_convert(Ptr{Cvoid}, kernel.fun)
    else
        error("Could not find generated GPU kernel: $gpu_kernel_name")
    end
end

@kernel unsafe_indices=true function my_kernel(A)
    i = @index(Local, Linear)
    @inbounds A[i] = i * 2.0
end

argtypes = Tuple{CuContext, CuDeviceVector{Float64, 1}}
ptr = get_native_ptr_direct(my_kernel, argtypes)
