module KA

    using CUDA  # or ROCm, oneAPI, etc.
    using CUDA.CUDAKernels

    import ..XKBlas

    function init()
        nothing
    end

    struct KernelTaskFormat
        kernel_function::Function
        access_functions::Tuple{Vararg{Function}}
        fmtid::XKBlas.xkrt_task_format_id_t
        fptr
    end

    function get_cu_function(kernel_function::Function)
        # kernel = @cuda launch=false kernel_function(CUDA.zeros(1), CUDA.zeros(1), CUDA.zeros(1))
        kernel = CUDA.@cuda launch=false kernel_function()
        cu_function::CUDA.CuFunction = kernel.fun
        return Ptr{Cvoid}(cu_function.handle)
    end

    function task_ka_main(
       runtime::Ptr{XKBlas.xkrt_runtime_t},
       device::Ptr{XKBlas.xkrt_device_t},
       task::Ptr{XKBlas.xkrt_task_t}
    )
        XKBlas.Logger.info("Running task")
        fmt_ptr::Ptr{KernelTaskFormat} = XKBlas.xkrt_task_args(task)
        fmt=unsafe_load(fmt_ptr)

        cufunction = get_cu_function(fmt.kernel_function)
        XKBlas.Logger.info("cufunction is $cufunction")
        # XKBlas.xkrt_task_single_kernel_launcher(
        #     runtime,
        #     device,
        #     task,
        #     task_kernel_launcher_fptr
        # )
    end

    function Format(kernel_function::Function, access_functions::Function...)
        fptr = @cfunction(task_ka_main, Cvoid, (Ptr{XKBlas.xkrt_runtime_t},
                                                Ptr{XKBlas.xkrt_device_t},
                                                Ptr{XKBlas.xkrt_task_t}))
        name = nameof(kernel_function)
        fmtid = XKBlas.task_format_put("KA.$name")
        for target in instances(XKBlas.xkrt_task_format_target_t)
            XKBlas.task_format_set(fmtid, target, fptr)
        end
        return KernelTaskFormat(kernel_function, access_functions, fmtid, fptr)
    end

    function device_async(
        device_global_id::XKBlas.xkrt_device_global_id_t,
        fmt::KernelTaskFormat,
        kernel_args...
    )
        if length(fmt.access_functions) !== length(kernel_args)
            throw(ErrorException("Arguments do not match the task format accesses"))
        end
        set_accesses = (accesses) -> begin
            for i in 1:length(kernel_args)
                push!(accesses, fmt.access_functions[i](kernel_args[i]))
            end
        end

        fmt_ref = Ref(fmt)
        args = Base.unsafe_convert(Ptr{Cvoid}, fmt_ref)
        args_size = sizeof(fmt)
        GC.@preserve fmt_ref begin
            XKBlas.device_async(device_global_id, fmt.fmtid, set_accesses=set_accesses, args=args, args_size=args_size)
        end
    end

    function device_async(fmt::KernelTaskFormat, kernel_args...)
        device_global_id = XKBlas.xkrt_device_global_id_t(1)
        return XKBlas.KA.device_async(device_global_id, fmt, kernel_args...)
    end

end
