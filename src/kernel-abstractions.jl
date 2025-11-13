module KA

    using CUDA  # or ROCm, oneAPI, etc.
    using CUDA.CUDAKernels
    using KernelAbstractions

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

    function task_ka_main(
       runtime::Ptr{XKBlas.xkrt_runtime_t},
       device::Ptr{XKBlas.xkrt_device_t},
       task::Ptr{XKBlas.xkrt_task_t}
    )
        XKBlas.Logger.info("Running task")
        fmt_ptr::Ptr{KernelTaskFormat} = XKBlas.xkrt_task_args(task)
        fmt=unsafe_load(fmt_ptr)

        # XKBlas.xkrt_task_single_kernel_launcher(
        #     runtime,
        #     device,
        #     task,
        #     task_kernel_launcher_fptr
        # )
        return
    end

    function get_cu_function(kernel_function::Function)
        kernel = @cuda launch=false kernel_function(CUDA.zeros(1), CUDA.zeros(1), CUDA.zeros(1))
        # kernel = CUDA.@cuda launch=false kernel_function()
        cu_function::CUDA.CuFunction = kernel.fun
        return Ptr{Cvoid}(cu_function.handle)
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

#        ################################################
#        # Execute the kernel once to force compilation #
#        ################################################
#
#        # Create sample arrays
#        n = 1
#        a = CUDA.zeros(Float32, n)
#        b = CUDA.zeros(Float32, n)
#        c = CUDA.zeros(Float32, n)
#
#        # Create the KernelAbstractions kernel
#        backend = CUDABackend()
#        ka_kernel = kernel_function(backend)
#
#        # Execute
#        ndrange = (n,)
#        workgroupsize = (256,)
#        ka_kernel(a, b, c, ndrange=ndrange, workgroupsize=workgroupsize)
#        CUDA.synchronize()
#
#        # The kernel compiled successfully above. Now we need to extract the CUfunction
#        # from the already-compiled kernel. Let's use @cuda to recompile with launch=false
#        # but use StaticSize for workgroupsize to avoid dynamic calls
#
#        iterspace, dynamic = KernelAbstractions.partition(ka_kernel, ndrange, workgroupsize)
#
#        # Create StaticSize version of the iterspace for compilation
#        static_workgroupsize = KernelAbstractions.NDIteration.StaticSize(workgroupsize)
#        static_iterspace, static_dynamic = KernelAbstractions.partition(ka_kernel, ndrange, static_workgroupsize)
#
#        # Build metadata with static workgroup size
#        metadata_type = KernelAbstractions.CompilerMetadata{
#                                                            typeof(static_iterspace),
#                                                            typeof(static_dynamic),
#                                                            Nothing,
#                                                            typeof(ndrange),
#                                                            typeof(static_iterspace)
#                                                           }
#
#        dev_array_type = CUDA.CuDeviceVector{Float32, 1}
#        cu_function = CUDA.cufunction(ka_kernel.f, Tuple{metadata_type, dev_array_type, dev_array_type, dev_array_type})
#        println(typeof(cu_function))
#        XKBlas.Logger.info("cufunction is $cu_function")

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
