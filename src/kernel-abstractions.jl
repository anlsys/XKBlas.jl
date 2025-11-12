module KA

    import ..XKBlas

    # the task format id
    fmtid = 0

    function kernel_launcher_func()
        println("RUN")
    end

    function init()
        fmtid = XKBlas.task_format_put("KA")
        fptr = @cfunction(kernel_launcher_func, Cvoid, (Ptr{Cvoid},))
        for target in instances(XKBlas.xkrt_task_format_target_t)
            if target !== XKBlas.XKRT_TASK_FORMAT_TARGET_HOST
                XKBlas.task_format_set(fmtid, target, fptr)
            end
        end
    end

    """
    A struct to hold the registered kernel and its memory accessors.
    This is the object returned by `XKBlas.KA.create`.
    """
    struct KernelTaskFormat
        kernel_function::Function
        access_functions::Tuple{Vararg{Function}}
    end

    """
    # Arguments
    - `kernel_func::Function`: The kernel function to be executed.
    - `accessors::Function...`: A variable number of functions definining the task accesses
    Each function must correspond to an argument of the kernel_func and define
    its memory access pattern.
    """
    function Format(kernel_function::Function, access_functions::Function...)
        return KernelTaskFormat(kernel_function, access_functions)
    end

    function device_async(
        device_global_id::XKBlas.xkrt_device_global_id_t,
        fmt::KernelTaskFormat,
        kernel_args...
    )
        if length(fmt.access_functions) !== length(kernel_args)
            throw(ErrorException("Arguments do not match the task format accesses"))
        end
        # TODO
        # XKBlas.device_with_accesses_and_format_async(device_global_id, fmtid, accesses, naccesses)
        println("TODO - gotta spawn the device task")
    end

    function device_async(fmt::KernelTaskFormat, kernel_args...)
        device_global_id = XKBlas.xkrt_device_global_id_t(1)
        return XKBlas.KA.device_async(device_global_id, fmt, kernel_args...)
    end

end
