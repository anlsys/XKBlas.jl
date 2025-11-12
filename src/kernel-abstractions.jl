module KA

    import ..XKBlas

    function kernel_launcher_func(
        runtime::XKBlas.xkrt_runtime_t,
        device::Ptr{XKBlas.xkrt_device_t},
        task::Ptr{XKBlas.xkrt_task_t}
     )
        XKBlas.Logger.info("Running kernel")
    end

    function init()
        nothing
    end

    struct KernelTaskFormat
        kernel_function::Function
        access_functions::Tuple{Vararg{Function}}
        fmtid::XKBlas.xkrt_task_format_id_t
        fptr
    end

    function Format(kernel_function::Function, access_functions::Function...)
        name = nameof(kernel_function)
        fmtid = XKBlas.task_format_put("KA.$name")
        fptr = @cfunction(
                kernel_launcher_func,
                Cvoid,
                (XKBlas.xkrt_runtime_t,
                 Ptr{XKBlas.xkrt_device_t},
                 Ptr{XKBlas.xkrt_task_t})
               )
        for target in instances(XKBlas.xkrt_task_format_target_t)
            XKBlas.task_format_set(fmtid, target, fptr)
        end
        return KernelTaskFormat(kernel_function, access_functions, fmtid, fptr)
    end

    function async(
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
        XKBlas.async(device_global_id, fmt.fmtid, set_accesses=set_accesses)
    end

    function async(fmt::KernelTaskFormat, kernel_args...)
        device_global_id = XKBlas.xkrt_device_global_id_t(1)
        return XKBlas.KA.async(device_global_id, fmt, kernel_args...)
    end

end
