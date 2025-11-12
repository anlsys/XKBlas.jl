module KA

    import ..XKBlas

    function kernel_launcher_func()
        XKBlas.Logger.info("Running kernel")
    end

    function init()
        nothing
    end

    """
    A struct to hold the registered kernel and its memory accessors.
    This is the object returned by `XKBlas.KA.create`.
    """
    struct KernelTaskFormat
        kernel_function::Function
        access_functions::Tuple{Vararg{Function}}
        fmtid::XKBlas.xkrt_task_format_id_t
    end

    """
    # Arguments
    - `kernel_func::Function`: The kernel function to be executed.
    - `accessors::Function...`: A variable number of functions definining the task accesses
    Each function must correspond to an argument of the kernel_func and define
    its memory access pattern.
    """
    function Format(kernel_function::Function, access_functions::Function...)
        name = nameof(kernel_function)
        fmtid = XKBlas.task_format_put("KA.$name")
        fptr = @cfunction(kernel_launcher_func, Cvoid, (Ptr{Cvoid},))   # TODO: GC ?
        for target in instances(XKBlas.xkrt_task_format_target_t)
            if target !== XKBlas.XKRT_TASK_FORMAT_TARGET_HOST
                XKBlas.task_format_set(fmtid, target, fptr)
            end
        end
        return KernelTaskFormat(kernel_function, access_functions, fmtid)
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
        XKBlas.async(device_global_id, kernel_launcher_func, set_accesses=set_accesses)
        #error("TODO - gotta spawn the device task")
    end

    function device_async(fmt::KernelTaskFormat, kernel_args...)
        device_global_id = XKBlas.xkrt_device_global_id_t(1)
        return XKBlas.KA.device_async(device_global_id, fmt, kernel_args...)
    end

end
