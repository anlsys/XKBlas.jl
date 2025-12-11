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

    #############################################################
    # A disk cache to avoid recompiling kernels on every launch #
    #############################################################

    module Cache

        import ...XKBlas
        const XK = XKBlas

        using Serialization, SHA

        # Cache directory setup
        const BYTECODE_CACHE_DIR = joinpath(homedir(), ".julia_xkrt_bytecode_cache")
        mkpath(BYTECODE_CACHE_DIR)

        """
            compute_cache_key(kernel_function::Function, kernel_tt::Type)

        Compute a unique hash for the kernel function and its argument types.
        This hash is used as the cache key.
        """
        function compute_cache_key(kernel_function::Function, kernel_tt::Type)
            # Get the method signature
            methods_list = methods(kernel_function, kernel_tt.parameters)

            if isempty(methods_list)
                error("No methods found for kernel function with type signature $kernel_tt")
            end

            method = first(methods_list)

            # Get code info - but hash it directly instead of converting to string
            code_info = code_lowered(kernel_function, kernel_tt.parameters)

            # Hash based on:
            # 1. Function name
            # 2. Method signature
            # 3. Argument types
            # 4. Code info (detects all code changes)
            h = hash(nameof(kernel_function))
            h = hash(method.sig, h)
            h = hash(kernel_tt, h)
            for ci in code_info
                code_str = sprint(show, ci.code)    # that is slow, but makes it constant across invocation for the same code
                h = hash(code_str, h)
            end

            # Convert to hex string for filename compatibility
            return string(h, base=16, pad=16)
        end

        """
            get_cache_path(cache_key::String)

        Get the file path for a given cache key.
        """
        function get_cache_path(cache_key::String)
            return joinpath(BYTECODE_CACHE_DIR, "$(cache_key).jls")
        end

        """
            load_cached_bytecode(cache_key::String)

        Load cached BYTECODE data from disk if it exists.
        Returns `nothing` if cache miss.
        """
        function load_cached_bytecode(cache_key::String)
            path = get_cache_path(cache_key)
            if isfile(path)
                try
                    XK.Logger.debug("BYTECODE cache hit! Loading from disk... $(path)")
                    r = deserialize(path)
                    XK.Logger.debug("Loaded from disk")
                    return r
                catch e
                    # If deserialization fails, remove the corrupted cache file
                    rm(path; force=true)
                end
            end
            XK.Logger.debug("BYTECODE cache miss.")
            return nothing
        end

        """
            save_cached_bytecode(cache_key::String, bytecode::String, bytecode_size::Int, bytecode_name::String)

        Save BYTECODE compilation results to disk cache.
        """
        function save_cached_bytecode(cache_key::String, bytecode::String, bytecode_size::Int, bytecode_name::String)
            path = get_cache_path(cache_key)
            try
                data = (bytecode=bytecode, bytecode_size=bytecode_size, bytecode_name=bytecode_name)
                serialize(path, data)
                XK.Logger.debug("Saved BYTECODE to cache: $path")
            catch e
                @warn "Failed to save BYTECODE cache to $path: $e"
            end
        end

        """
            clear_bytecode_cache!()

        Clear all cached BYTECODE files from disk.
        """
        function clear_bytecode_cache!()
            if isdir(BYTECODE_CACHE_DIR)
                for file in readdir(BYTECODE_CACHE_DIR)
                    filepath = joinpath(BYTECODE_CACHE_DIR, file)
                    rm(filepath; force=true)
                end
                @info "Cleared BYTECODE cache directory: $BYTECODE_CACHE_DIR"
            end
        end

    end

    # Called on XKBlas init once
    function init()
    end

    # Called on XKBlas deinit
    function deinit()
    end

    # Mutable so it is passed by reference
    mutable struct FormatStruct

        # 1. The KA function annotated with @kernel
        kernel_function::Function

        # 2. function BYTECODE
        bytecode::String

        # 3. function BYTECODE size
        bytecode_size::Int

        # 4. function BYTECODE name
        bytecode_name::String

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
        fmt_ptr::Ptr{Ptr{FormatStruct}} = XK.xkrt_task_args(task)
        fmt = unsafe_pointer_to_objref(fmt_ptr)::FormatStruct

        # TODO: need to protect that with a mutex in case multiple tasks uses the same format
        # cached cuda function
        if fmt.moodule == C_NULL

            @assert fmt.fn == C_NULL

            # compile the bytecode
            driver           = XK.xkrt_device_driver_get(runtime, device)
            device_driver_id = XK.xkrt_device_driver_id_get(runtime, device)
            bin              = fmt.bytecode
            binsize          = fmt.bytecode_size
            format           = XK.XKRT_DRIVER_MODULE_FORMAT_NATIVE
            moodule          = XK.xkrt_driver_module_load(driver, device_driver_id, bin, binsize, format)

            # get executable function
            fn = XK.xkrt_driver_module_get_fn(driver, moodule, fmt.bytecode_name)

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
        # Compile to bytecode to target CUDA devices #
        #########################################

        # Compile to BYTECODE
        kernel_tt = Tuple{
            map(
                T_abstract -> (
                    T_abstract <: AbstractVector ? CUDA.CuDeviceVector{eltype(T_abstract), 1} :
                    T_abstract
                ),
                arg_types_list
            )...
        }

        # Try to load from cache
        XK.Logger.debug("Computing cache key...")
        cache_key = XK.KA.Cache.compute_cache_key(kernel_function, kernel_tt)
        XK.Logger.debug("Cache key: $(cache_key)")
        cached_data = XK.KA.Cache.load_cached_bytecode(cache_key)

        if cached_data !== nothing
            bytecode = cached_data.bytecode
            bytecode_size = cached_data.bytecode_size
            bytecode_name = cached_data.bytecode_name
        else
            # Compile to BYTECODE - TODO: do that portably, not only ptx
            buf = IOBuffer()
            XK.Logger.debug("Compiling to BYTECODE")
            CUDA.code_ptx(buf, kernel_function, kernel_tt; raw=true, kernel=true)
            bytecode = String(take!(buf))
            bytecode_size = length(bytecode)

            XK.Logger.debug("Compiled to BYTECODE")
            XK.Logger.debug(bytecode)

            # Regex to find function names
            regex_func = r"\.entry\s+([a-zA-Z_0-9]+)\("

            # Find all matches and extract the captured name
            function_names = [m.captures[1] for m in eachmatch(regex_func, bytecode)]
            XK.Logger.debug("Functions in the BYTECODE: $(function_names)")
            @assert length(function_names) == 1
            bytecode_name = String(function_names[1])

            # Save to cache
            XK.KA.Cache.save_cached_bytecode(cache_key, bytecode, bytecode_size, bytecode_name)
        end
        XK.Logger.debug("BYTECODE of name $(bytecode_name) and size $(bytecode_size)")

        # return the format
        return  FormatStruct(
                    kernel_function,            # The Julia function
                    bytecode,                   # function BYTECODE
                    bytecode_size,              # function BYTECODE size
                    bytecode_name,              # function BYTECODE name
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
        # TODO: check that argument matches the format arguments more precisely
        if length(fmt.access_functions) !== length(kernel_args)
            throw(ErrorException("Arguments do not match the task format accesses"))
        end

        ######################################################
        # Set the args buffer for launching the kernel later #
        ######################################################

        # task arguments = [pointer_to_format | kernel_args...]

        # compute sizes for each kernel argument
        arg_sizes = Vector{Int}(undef, length(kernel_args))
        for i in 1:length(kernel_args)
            if fmt.return_types_list[i] <: XK.xkrt_access_t
                # we will store a raw pointer (machine pointer size)
                arg_sizes[i] = sizeof(Ptr{Cvoid})
            else
                # store the raw bytes of the value
                # sizeof should work for typical scalar isbitstype arguments (Int, Float64, etc.)
                arg_sizes[i] = sizeof(kernel_args[i])
            end
        end

        # allocate contiguous byte buffer
        fmt_size = sizeof(Ptr{Cvoid})
        total_size = fmt_size + sum(arg_sizes)
        args_buf = Vector{UInt8}(undef, total_size)

        # copy format ptr
        fmt_ptr = Base.unsafe_convert(Ptr{Cvoid}, Ref(fmt))
        unsafe_store!(Ptr{Ptr{Cvoid}}(pointer(args_buf)), fmt_ptr)

        # fill rest of buffer with argument representations
        offset = fmt_size
        for i in 1:length(kernel_args)
            arg = kernel_args[i]
            arg_size = arg_sizes[i]
            dest = pointer(args_buf) + offset

            if fmt.return_types_list[i] <: XK.xkrt_access_t
                @assert isa(arg, AbstractArray)
                @assert arg_size == sizeof(Ptr{Cvoid})

                # nothing to do, this space will be filled when the kernel is
                # scheduled with the replicated device memory pointer

            else
                # copy raw bytes of the scalar argument value
                arg_ref = Ref(arg)
                XK.
                GC.@preserve arg_ref begin
                    arg_struct = Base.unsafe_convert(Ptr{typeof(arg)}, arg_ref)
                    arg_u8 = Ptr{UInt8}(arg_struct)
                    unsafe_copyto!(dest, arg_u8, arg_size)
                end
            end

            offset += arg_size
        end

        # set args pointer and size to the buffer
        args = Ptr{Cvoid}(pointer(args_buf))
        args_size = total_size

        ############################################
        # create a lambda to set the task accesses #
        ############################################

        set_accesses = (accesses) -> begin
            @assert length(fmt.access_functions) === length(fmt.return_types_list)
            for i in 1:length(kernel_args)
                if fmt.return_types_list[i] <: XK.xkrt_access_t
                    push!(accesses, fmt.access_functions[i](kernel_args[i]))
                end
            end
        end

        XK.device_async(
            device_global_id,
            fmt.fmtid,
            set_accesses=set_accesses,
            args=args, args_size=args_size,
            detach_ctr_initial=1
        )
    end

    function device_async(fmt::FormatStruct, kernel_args...)
        device_global_id = XK.xkrt_device_global_id_t(1)
        return XK.KA.device_async(device_global_id, fmt, kernel_args...)
    end

end
