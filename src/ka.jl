#
#   Kernel abstractions built in XKBlas
#
#   Currently only supports CUDA.
#   TODO: support other and add macros for block dim and shared memory
#
module KA

    using CUDA

    import ..XKBlas
    const XK = XKBlas

    # Called on XKBlas init once
    function init()
    end

    # Called on XKBlas deinit
    function deinit()
    end

    struct FormatStruct

        # 1. The KA function annotated with @kernel
        kernel_function::Function

        # 2. function PTX
        ptx::String

        # 3. function PTX size
        ptx_size::Int

        # 4. function PTX name
        ptx_name::String

        # 5. function to get the grid launch dimensions
        grid_function::Function

        # 6. function to get the amount of shared memory to use
        shared_memory_function::Function

        # 7. functions telling how parameters of the kernels are accessed
        access_functions::Tuple{Vararg{Function}}

        # 8. argument types used by each argument function
        arg_types_list::Vector{Type}

        # 9. return types used by each argument function
        return_types_list::Vector{Type}

        # 10. Number of access for tasks
        num_access::Int

        # 11. XKRT task format id
        fmtid::XK.xkrt_task_format_id_t

        # 12. Cached module
        moodule::XK.xkrt_driver_module_t

        # 13. Cached function
        fn::XK.xkrt_driver_module_fn_t

    end

    # offset in the format struct
    const FORMAT_STRUCT_MOODULE_OFFSET = fieldoffset(XK.KA.FormatStruct, 12)
    const FORMAT_STRUCT_FN_OFFSET      = fieldoffset(XK.KA.FormatStruct, 13)

    # kernel launcher routine - run by a julia task
    function task_ka_launcher(
        runtime::Ptr{XK.xkrt_runtime_t},
        device::Ptr{XK.xkrt_device_t},
        task::Ptr{XK.xkrt_task_t},
        queue::Ptr{XK.xkrt_queue_t},
        command::Ptr{XK.xkrt_command_t},
        index::XK.xkrt_queue_command_list_counter_t
    )
        # TODO: doing a lot of illegal stuff here (we are executing within a foreign thread)
        # But it seems to work, by avoiding any Julia runtime calls, but only XKRT calls

        # retrieve task format
        fmt_ptr::Ptr{FormatStruct} = XK.xkrt_task_args(task)
        fmt=unsafe_load(fmt_ptr)

        # TODO: need to protect that with a mutex in case multiple tasks uses the same format
        # cached cuda function
        if fmt.moodule == C_NULL

            @assert fmt.fn == C_NULL

            # compile the ptx
            driver           = XK.xkrt_device_driver_get(runtime, device)
            device_driver_id = XK.xkrt_device_driver_id_get(runtime, device)
            bin              = fmt.ptx
            binsize          = fmt.ptx_size
            format           = XK.XKRT_DRIVER_MODULE_FORMAT_NATIVE
            moodule          = XK.xkrt_driver_module_load(driver, device_driver_id, bin, binsize, format)

            # get executable function
            fn = XK.xkrt_driver_module_get_fn(driver, moodule, fmt.ptx_name)

            # save module and fn
            unsafe_store!(Ptr{XK.xkrt_driver_module_t   }(fmt_ptr + FORMAT_STRUCT_MOODULE_OFFSET), moodule)
            unsafe_store!(Ptr{XK.xkrt_driver_module_fn_t}(fmt_ptr + FORMAT_STRUCT_FN_OFFSET),      fn)

            fmt=unsafe_load(fmt_ptr)
            @assert fmt.moodule != C_NULL
            @assert fmt.fn != C_NULL

        end

        # dim3 T = { (unsigned int) n, (unsigned int) m, 1 }; // How many threads we need
        # dim3 B = { 32, 32, 1 }; // Bloc shape
        # dim3 G = { (T.x + B.x - 1)/B.x,  (T.y + B.y - 1)/B.y, (T.z + B.z - 1)/B.z }; // Grid

        # TODO: build the arguments
        XK.Logger.fatal("TODO: build arguments")

        # launch the kernel
        gx = 1
        gy = 1
        gz = 1
        bx = 1
        by = 1
        bz = 1
        shared_memory_bytes = 0
        args = C_NULL
        args_size = 0
        XK.xkrt_device_kernel_launch(
            runtime, device,
            queue, index,
            fmt.fn,
            gx, gy, gz,
            bx, by, bz,
            shared_memory_bytes,
            args, args_size
        )

        # unload the module
        # XK.xkrt_driver_module_unload(driver, moodule)

        # task completed
        # XK.xkrt_task_detachable_decr(runtime, task)

        return nothing
    end

    # task routine
    function task_ka_main(
       runtime::Ptr{XK.xkrt_runtime_t},
       device::Ptr{XK.xkrt_device_t},
       task::Ptr{XK.xkrt_task_t}
    )
        # TODO: this is illegal but seems to be working

        # retrieve function pointer
        fptr_task_ka_launcher = @cfunction(
            task_ka_launcher,
            Cvoid,                                  # The C function returns void
            (Ptr{XK.xkrt_runtime_t},                # void * runtime
             Ptr{XK.xkrt_device_t},                 # void * device
             Ptr{XK.xkrt_task_t},                   # void * task
             Ptr{XK.xkrt_queue_t},                  # void * queue
             Ptr{XK.xkrt_command_t},                # void * cmd
             XK.xkrt_queue_command_list_counter_t   # xkrt_queue_command_list_counter_t idx (Non-pointer type)
            )
        )

        # kernel launch is synchronous, XKRT should not expect pending events
        synchronous = Cint(0)

        # launch a synchronous kernel command
        XK.xkrt_task_kernel_launch(runtime, device, task, synchronous, fptr_task_ka_launcher)

        return nothing
    end

    function Format(
        kernel_function::Function,
        grid_function::Function,
        shared_memory_function::Function,
        access_functions::Function...
    )
        # task main entry point
        task_main = @cfunction(task_ka_main, Cvoid, (Ptr{XK.xkrt_runtime_t}, Ptr{XK.xkrt_device_t}, Ptr{XK.xkrt_task_t}))

        #############################################
        # set task format: the same for all drivers #
        #############################################
        name = nameof(kernel_function)
        fmtid = XK.task_format_put("KA.$name")
        for target in instances(XK.xkrt_task_format_target_t)
            XK.task_format_set(fmtid, target, task_main)
        end

        # retrieve arguments types to compile the function,
        # and return type to count the number of actual accesses and values passed by copy
        arg_types_list      = []
        return_types_list   = []
        for access_function in access_functions
            m = first(methods(access_function))
            @assert length(m.sig.parameters) > 1
            args = m.sig.parameters[2]
            push!(arg_types_list, args)

            rts = Base.return_types(access_function, (args,))
            push!(return_types_list, rts[1])
        end

        # count number of task accesses
        num_access = count(==(XK.xkrt_access_t), return_types_list)

        XK.Logger.debug("Number of accesses: $(num_access)")
        XK.Logger.debug("$(arg_types_list)")
        XK.Logger.debug("$(return_types_list)")

        #########################################
        # Compile to ptx to target CUDA devices #
        #########################################

        # Compile to PTX
        kernel_tt = Tuple{
            map(
                T_abstract -> (
                    T_abstract <: AbstractVector ? CUDA.CuDeviceVector{eltype(T_abstract), 1} :
                    T_abstract
                ),
                arg_types_list
            )...
        }
        XK.Logger.debug("Compiling to PTX...")
        XK.Logger.debug("$(kernel_tt)")
        buf = IOBuffer()
        CUDA.code_ptx(buf, kernel_function, kernel_tt; raw=true, kernel=true)
        ptx = String(take!(buf))
        ptx_size = length(ptx)
        XK.Logger.debug("Compiled to")
        XK.Logger.debug(ptx)
        XK.Logger.debug("PTX size = $(ptx_size)")

        # Regex to find function names:
        # 1. `\.func\s+` : Matches the literal ".func" followed by one or more whitespace characters.
        # 2. `([a-zA-Z_0-9]+)`: This is the CAPTURE GROUP. It matches and captures the function name,
        #                       which consists of letters, numbers, and underscores (the standard for assembly/compiler-generated names).
        # 3. `\(`: Matches the literal opening parenthesis that follows the function name.
        regex_func = r"\.entry\s+([a-zA-Z_0-9]+)\("

        # Find all matches and extract the captured name
        function_names = [m.captures[1] for m in eachmatch(regex_func, ptx)]
        XK.Logger.debug("Functions in the PTX: $(function_names)")
        @assert length(function_names) == 1
        ptx_name = function_names[1]

        # return the format
        return  FormatStruct(
                    kernel_function,            # The Julia function
                    ptx,                        # function PTX
                    ptx_size,                   # function PTX size
                    ptx_name,                   # function PTX name
                    grid_function,              # function to get the grid launch dimensions
                    shared_memory_function,     # function to get the amount of shared memory to use
                    access_functions,           # functions telling how parameters of the kernels are accessed
                    arg_types_list,             # argument types used by each argument function
                    return_types_list,          # return types used by each argument function
                    num_access,                 # Number of access for tasks
                    fmtid,                      # XKRT task format id
                    C_NULL,                     # Cached module
                    C_NULL                      # Cached function
               )
    end

    # Spawn a task to the given device, with the given XK.KA format and kernel arguments
    function device_async(
        device_global_id::XK.xkrt_device_global_id_t,
        fmt::FormatStruct,
        kernel_args...
    )
        if length(fmt.access_functions) !== length(kernel_args)
            throw(ErrorException("Arguments do not match the task format accesses"))
        end
        set_accesses = (accesses) -> begin
            @assert length(fmt.access_functions) === length(fmt.return_types_list)
            for i in 1:length(kernel_args)
                if fmt.return_types_list[i] <: XK.xkrt_access_t
                    push!(accesses, fmt.access_functions[i](kernel_args[i]))
                end
            end
        end

        fmt_ref = Ref(fmt)
        args = Base.unsafe_convert(Ptr{Cvoid}, fmt_ref)
        args_size = sizeof(fmt)
        GC.@preserve fmt_ref begin
            XK.device_async(device_global_id, fmt.fmtid, set_accesses=set_accesses, args=args, args_size=args_size, detach_ctr_initial=1)
        end
    end

    function device_async(fmt::FormatStruct, kernel_args...)
        device_global_id = XK.xkrt_device_global_id_t(1)
        return XK.KA.device_async(device_global_id, fmt, kernel_args...)
    end

end
