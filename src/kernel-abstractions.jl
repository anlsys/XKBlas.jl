module KA

    using CUDA  # or ROCm, oneAPI, etc.
    using CUDA.CUDAKernels
    using KernelAbstractions

    import ..XKBlas
    const XK = XKBlas

    # Spawn Julia thread to run Julia code
    # This design is required by Julia not allowing external threads running Julia code
    # that leads to deadlocks in the Julia task scheduler mostly
    module Threading

        using ..KA: XK

        const TASK_CHANNEL = Ref{Channel{Any}}()
        const COND_HANDLE  = Ref{Base.AsyncCondition}()

        # Tells XKRT to run that Julia routine that may enter the Julia runtime.
        # In such case, special treatment is needed to avoid deadlocks, as Julia do not allow
        # entering the Julia runtime from a foreign thread... That's a dirty workd-around, no better solution
        # see https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/#Thread-safety
        function submit_julia_runtime_lambda(func)
            put!(TASK_CHANNEL[], func)
            ccall(:uv_async_send, Cint, (Ptr{Cvoid},), COND_HANDLE[].handle)
        end

        function init()
            TASK_CHANNEL[] = Channel{Any}(Inf)
            COND_HANDLE[]  = Base.AsyncCondition()

            @async begin
                chan = TASK_CHANNEL[]
                cond = COND_HANDLE[]

                try
                    while isopen(cond)
                        wait(cond)

                        # Drain the queue
                        while !isempty(chan)
                            func = take!(chan)
                            try
                                Base.invokelatest(func)
                            catch e
                                @error "Task failed" exception=e
                            end
                        end
                    end
                catch e
                    if !(e isa EOFError)
                        @error "Listener crashed" exception=e
                    end
                end
            end
        end

        function deinit()
            if isassigned(COND_HANDLE) && isopen(COND_HANDLE[])
                close(COND_HANDLE[])
            end
        end
    end

    # Called on XKBlas init once
    function init()
        XK.KA.Threading.init()
    end

    # Called on XKBlas deinit
    function deinit()
        XK.KA.Threading.deinit()
    end

    struct KernelTaskFormat

        # The KA function annotated with @kernel
        kernel_function::Function

        # function to get the grid launch dimensions
        grid_function::Function

        # function to get the amount of shared memory to use
        shared_memory_function::Function

        # functions telling how parameters of the kernels are accessed
        access_functions::Tuple{Vararg{Function}}

        # types used by each access
        access_types::Vector{Type}

        # XKRT task format id
        fmtid::XK.xkrt_task_format_id_t

        # main entry of that task
        task_main::Ptr{Cvoid}

    end

#    #   Get a CuFunction from a KA function annotated with @kernel
#    #
#    #   Example
#    #
#    #   @kernel unsafe_indices=true function my_kernel(A)
#    #      i = @index(Local, Linear)
#    #      @inbounds A[i] = i * 2.0
#    #   end
#    #   argtypes = Tuple{CuContext, CuDeviceVector{Float64, 1}}
#    #   ptr = get_native_cufunction(my_kernel, argtypes)
#    #
#    function get_native_cufunction(kernel_func::Function, argtypes::Type{<:Tuple})
#        GC.enable(false)
#        mod = parentmodule(kernel_func)
#        gpu_kernel_name = Symbol(:gpu_, nameof(kernel_func))
#        if isdefined(mod, gpu_kernel_name)
#            gpu_kernel = getfield(mod, gpu_kernel_name)
#            kernel = CUDA.cufunction(gpu_kernel, argtypes)
#            kernel_fun_ref = Ref(kernel.fun)    # TODO: garbage collection issues maybe
#            return Base.unsafe_convert(Ptr{Cvoid}, kernel_fun_ref)
#
#        else
#            error("Could not find generated GPU kernel: $gpu_kernel_name")
#        end
#        GC.enable(true)
#    end

    # kernel launcher routine - run by a julia task
    function task_ka_launcher(
        runtime::Ptr{XK.xkrt_runtime_t},
        device::Ptr{XK.xkrt_device_t},
        task::Ptr{XK.xkrt_task_t},
        queue::Ptr{XK.xkrt_queue_t},
        command::Ptr{XK.xkrt_command_t},
        index::Ptr{XK.xkrt_queue_command_list_counter_t}
    )
        fmt_ptr::Ptr{KernelTaskFormat} = XK.xkrt_task_args(task)
        fmt=unsafe_load(fmt_ptr)

        XK.Logger.fatal("TODO: run synchronously and set 'event' fulfilld")

        # for the initial ctr (julia tasks running julia runtime)
        XK.xkrt_task_detachable_decr(runtime, task)

#          dim3 T = { (unsigned int) n, (unsigned int) m, 1 }; // How many threads we need
#          dim3 B = { 32, 32, 1 }; // Bloc shape
#          dim3 G = { (T.x + B.x - 1)/B.x,  (T.y + B.y - 1)/B.y, (T.z + B.z - 1)/B.z }; // Grid

#            gx = 1
#            gy = 1
#            gz = 1
#            bx = 1
#            by = 1
#            bz = 1
#            shared_memory_bytes = 0
#            args = C_NULL
#            args_size = 0
#            XK.xkrt_device_kernel_launch(
#                runtime, device,
#                queue, index,
#                cufunction,
#                gx, gy, gz,
#                bx, by, bz,
#                shared_memory_bytes,
#                args, args_size
#            )
    end

    # task routine
    function task_ka_main(
       runtime::Ptr{XK.xkrt_runtime_t},
       device::Ptr{XK.xkrt_device_t},
       task::Ptr{XK.xkrt_task_t}
    )
        # run by an xkrt thread, so dont do shit but submit a julia task
        XK.KA.Threading.submit_julia_runtime_lambda(() -> begin
            fptr = @cfunction(
                task_ka_launcher,
                Cvoid,
                (Ptr{XK.xkrt_runtime_t},
                 Ptr{XK.xkrt_device_t},
                 Ptr{XK.xkrt_task_t},
                 Ptr{XK.xkrt_queue_t},
                 Ptr{XK.xkrt_command_t},
                 Ptr{XK.xkrt_queue_command_list_counter_t})
            )
            GC.@preserve fptr begin
                XK.xkrt_task_detachable_kernel_launch(runtime, device, task, fptr)
            end
        end)
    end

    function Format(
        kernel_function::Function,
        grid_function::Function,
        shared_memory_function::Function,
        access_functions::Function...
    )
        # task main entry point
        task_main = @cfunction(task_ka_main, Cvoid, (Ptr{XK.xkrt_runtime_t},
                                                Ptr{XK.xkrt_device_t},
                                                Ptr{XK.xkrt_task_t}))

        #############################################
        # set task format: the same for all drivers #
        #############################################
        name = nameof(kernel_function)
        fmtid = XK.task_format_put("KA.$name")
        for target in instances(XK.xkrt_task_format_target_t)
            XK.task_format_set(fmtid, target, task_main)
        end

        # retrieve arguments types to compile the function
        access_types=[]
        for access_function in access_functions
            m = first(methods(access_function))
            if length(m.sig.parameters) > 1
                push!(access_types, m.sig.parameters[2])
            end
        end

        return  KernelTaskFormat(
                    kernel_function,
                    grid_function,
                    shared_memory_function,
                    access_functions,
                    access_types,
                    fmtid,
                    task_main
               )
    end

    # Spawn a task to the given device, with the given XK.KA format and kernel arguments
    function device_async(
        device_global_id::XK.xkrt_device_global_id_t,
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
            XK.device_async(device_global_id, fmt.fmtid, set_accesses=set_accesses, args=args, args_size=args_size, may_run_julia_runtime=true)
        end
    end

    function device_async(fmt::KernelTaskFormat, kernel_args...)
        device_global_id = XK.xkrt_device_global_id_t(1)
        return XK.KA.device_async(device_global_id, fmt, kernel_args...)
    end

end
