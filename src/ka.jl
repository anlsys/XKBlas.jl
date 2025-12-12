#
#   Kernel abstractions built in XKBlas
#   Kernel abstractions built-in XKBlas
#   Main differences with KernelAbstractions.jl and others, is:
#       - passed parameters are raw pointers, not Julia types
#       - there no bounds check: @inbounds is ignored
#
#   Currently only supports CUDA.
#   TODO: support other and add macros for block dim and shared memory
#

module KA

    using CUDA

    import ..XKBlas
    const XK = XKBlas

    # Julia CU context is 16 bytes, ignore them
    const JL_CU_CONTEXT_SIZE = 16

    # GPU block size magic number
    const BLOCK_SIZE = 256

    # Define getindex and setindex for Ptr{T}
    @inline Base.getindex( p::Ptr{T},      i::Integer) where T = unsafe_load(p, i)
    @inline Base.setindex!(p::Ptr{T}, val, i::Integer) where T = unsafe_store!(p, val, i)

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

    # Launcher
    struct LauncherStruct

        # 1. function to get the grid launch dimensions
        threads::Union{Function, Nothing}

        # 2. function to get the amount of shared memory to use
        shared_memory::Union{Function, Nothing}

    end

    # Format launch configuration
    function Launcher(;
        threads::Union{Function, Nothing}=nothing,
        shared_memory::Union{Function, Nothing}=nothing
    )
        return LauncherStruct(threads, shared_memory)
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

        # 5. laucher options
        launcher::LauncherStruct

        # 6. functions telling how parameters of the kernels are accessed
        access_functions::Tuple{Vararg{Function}}

        # 7. argument types used by each argument function
        arg_types_list::Vector{Type}

        # 8. return types used by each argument function
        return_types_list::Vector{Type}

        # 9. Number of access for tasks
        num_access::Int

        # 10. XKRT task format id
        fmtid::XK.xkrt_task_format_id_t

        # 11. Cached module
        moodule::XK.xkrt_driver_module_t

        # 12. Cached function
        fn::XK.xkrt_driver_module_fn_t

    end

    # offset in the format struct
    const FORMAT_STRUCT_MOODULE_OFFSET = fieldoffset(XK.KA.FormatStruct, 11)
    const FORMAT_STRUCT_FN_OFFSET      = fieldoffset(XK.KA.FormatStruct, 12)

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

        # retrieve task arguments
        task_args::Ptr{Int8} = XK.xkrt_task_args(task)

        # retrieve task format
        # 1. Get the raw C pointer (void*) pointing to the location of fmt_ptr
        p_p_fmt::Ptr{Ptr{FormatStruct}} = Ptr{Ptr{FormatStruct}}(task_args)
        fmt_ptr::Ptr{FormatStruct} = unsafe_load(p_p_fmt)
        fmt::FormatStruct = unsafe_load(fmt_ptr)

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

        ###############
        # Grid launch #
        ###############

        threads::Ptr{Int8} = task_args + sizeof(Ptr{Cvoid})
        tx::Int = unsafe_load(Ptr{Int}(threads + 0 * sizeof(Int)))
        ty::Int = unsafe_load(Ptr{Int}(threads + 1 * sizeof(Int)))
        tz::Int = unsafe_load(Ptr{Int}(threads + 2 * sizeof(Int)))
        bx = BLOCK_SIZE
        by = 1
        bz = 1
        gx = ceil(Int, tx / bx)
        gy = ceil(Int, ty / by)
        gz = ceil(Int, tz / bz)

        XK.Logger.debug("threads, blocks, grid: $(tx) $(ty) $(tz) $(bx) $(by) $(bz) $(gx) $(gy) $(gz)")
        #################
        # Shared memory #
        #################

        shared_memory_size_ptr::Ptr{Int8} = task_args + sizeof(Ptr{Cvoid}) + 3 * sizeof(Int)
        shared_memory_size::Int = unsafe_load(Ptr{Int}(shared_memory_size_ptr))

        ###################
        # Build arguments #
        ###################

        # retrieve args buffer, that is right after the format pointer in task arguments
        kernel_args::Ptr{Int8} = task_args + sizeof(Ptr{Cvoid}) + 3*sizeof(Int) + 1*sizeof(Int)

        # parse each accesses, and write replica address
        access_id = 0
        offset = JL_CU_CONTEXT_SIZE
        for return_type in fmt.return_types_list
            if return_type <: XK.xkrt_access_t
                device_ptr::Ptr{Cvoid} = XK.xkrt_task_access_replica(task, access_id)
                kernel_arg::Ptr{Int8}  = kernel_args + offset
                unsafe_store!(Ptr{Ptr{Cvoid}}(kernel_arg), device_ptr)
                access_id += 1
                offset += sizeof(Ptr{Cvoid})
            else
                # nothing to do, the producer thread already copied by value
                offset += sizeof(return_type)
            end
        end

        kernel_args_size = offset

        #####################
        # launch the kernel #
        #####################

        XK.xkrt_device_kernel_launch(
            runtime, device,
            queue, index,
            fmt.fn,
            gx, gy, gz,
            bx, by, bz,
            shared_memory_size,
            Ptr{Cvoid}(kernel_args), kernel_args_size
        )

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

    # Create an XK.KA.FormatStruct
    function Format(
        kernel_function::Function,
        access_functions::Function...;
        launcher::LauncherStruct=LauncherStruct(nothing, nothing)
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

        ##############################################
        # Compile to bytecode to target CUDA devices #
        ##############################################

        # Compile to BYTECODE
        kernel_tt = Tuple{
            map(
                T -> (
                    T <: AbstractVector ? Ptr{eltype(T)} :
                    # T <: AbstractVector ? CUDA.CuDeviceVector{eltype(T), 1} :
                    T
                ),
                arg_types_list
            )...
        }
        XK.Logger.debug("$(kernel_tt)")

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
            CUDA.code_ptx(buf, kernel_function, kernel_tt; raw=false, kernel=true)
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
                    launcher,                   # launch parameters (grid size, shared memory...)
                    access_functions,           # functions telling how parameters of the kernels are accessed
                    arg_types_list,             # argument types used by each argument function
                    return_types_list,          # return types used by each argument function
                    num_access,                 # Number of access for tasks
                    fmtid,                      # XKRT task format id
                    C_NULL,                     # Cached module
                    C_NULL                      # Cached function
               )
    end

    function Format(
        kernel_function::Function,
        launcher::LauncherStruct,
        access_functions::Function...
    )
        return Format(kernel_function, access_functions...; launcher = launcher)
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

        #
        # Task arguments are
        #   [pointer_to_format | tx | ty | tz | shared_memory_size | julia_context | kernel_args...]
        # with
        #   pointer_to_forma t -> a 'void *' to the FormatStruct
        #   tx, ty, tz         -> the number of threads to launch the kernel
        #   shared_memory_size -> amount of shared memory
        #   julia_context      -> opaque structure of `JL_CU_CONTEXT_SIZE` bytes
        #   kernel_args        -> the kernel arguments, prefilled
        #                           - empty spaces of sizeof(void *) bytes per access
        #                           - values, for values passed by copy
        #

        # 1. compute sizes for each kernel argument
        kernel_arg_sizes = Vector{Int}(undef, length(kernel_args))
        for i in 1:length(kernel_args)
            if fmt.return_types_list[i] <: XK.xkrt_access_t
                # we will store a raw pointer (machine pointer size)
                kernel_arg_sizes[i] = sizeof(Ptr{Cvoid})
            else
                # store the raw bytes of the value
                # sizeof should work for typical scalar isbitstype arguments (Int, Float64, etc.)
                kernel_arg_sizes[i] = sizeof(kernel_args[i])
            end
        end

        # 2. allocate contiguous byte buffer
        total_size = sizeof(Ptr{Cvoid}) + 3*sizeof(Int) + 1*sizeof(Int) + JL_CU_CONTEXT_SIZE + sum(kernel_arg_sizes)
        task_args_buf = Vector{UInt8}(undef, total_size)
        task_args_buf_ptr = Ptr{Int8}(pointer(task_args_buf))

        # 3. copy format ptr
        fmt_ptr = Base.unsafe_convert(Ptr{Cvoid}, Ref(fmt))
        unsafe_store!(Ptr{Ptr{Cvoid}}(task_args_buf_ptr), fmt_ptr)

        # 4. set the number of threads
        if fmt.launcher.threads != nothing
            tx, ty, tz = fmt.launcher.threads(kernel_args...)
        else
            tx, ty, tz = BLOCK_SIZE, 1, 1
        end
        threads_ptr = task_args_buf_ptr + sizeof(Ptr{Cvoid})
        unsafe_store!(Ptr{Int}(threads_ptr + 0*sizeof(Int)), tx)
        unsafe_store!(Ptr{Int}(threads_ptr + 1*sizeof(Int)), ty)
        unsafe_store!(Ptr{Int}(threads_ptr + 2*sizeof(Int)), tz)

        # 5. copy shared memory size
        if fmt.launcher.shared_memory != nothing
            shared_memory_size = fmt.launcher.shared_memory(kernel_args...)
        else
            shared_memory_size = 0
        end
        shared_memory_size_ptr = task_args_buf_ptr + sizeof(Ptr{Cvoid})+ 3*sizeof(Int)
        unsafe_store!(Ptr{Int}(shared_memory_size_ptr), shared_memory_size)

        # 6. fill rest of buffer with arguments
        offset = sizeof(Ptr{Cvoid}) + 3*sizeof(Int) + 1*sizeof(Int) + JL_CU_CONTEXT_SIZE
        for i in 1:length(kernel_args)
            kernel_arg = Ptr{Int8}(task_args_buf_ptr + offset)
            kernel_arg_size = kernel_arg_sizes[i]
            kernel_arg_value = kernel_args[i]

            if fmt.return_types_list[i] <: XK.xkrt_access_t
                @assert isa(kernel_arg_value, AbstractArray)
                @assert kernel_arg_size == sizeof(Ptr{Cvoid})
                # nothing to do, this space will be filled when the kernel is
                # scheduled with the replicated device memory pointer
            else
                # copy raw bytes of the scalar argument value
                kernel_arg_value_ref = Ref(kernel_arg_value)
                kernel_arg_value_struct = Base.unsafe_convert(Ptr{typeof(kernel_arg_value)}, kernel_arg_value_ref)
                kernel_arg_value_i8 = Ptr{Int8}(kernel_arg_value_struct)
                unsafe_copyto!(kernel_arg, kernel_arg_value_i8, kernel_arg_size)
            end

            offset += kernel_arg_size
        end

        # set args pointer and size to the buffer
        task_args = Ptr{Cvoid}(task_args_buf_ptr)
        task_args_size = total_size

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
            args=task_args, args_size=task_args_size,
            detach_ctr_initial=0
        )
    end

    # device_async wrapper to automatically target device 1
    function device_async(fmt::FormatStruct, kernel_args...)
        device_global_id = XK.xkrt_device_global_id_t(1)
        return XK.KA.device_async(device_global_id, fmt, kernel_args...)
    end

    """
        @kernel f(args...) = ...

    Wraps a function and ensures it always returns `nothing`.
    """
    macro kernel(ex)
        # Only works on function definitions
        if ex.head != :function
            error("@kernel must be applied to a function definition")
        end

        # Extract the name and body
        fname = ex.args[1]   # e.g., :(vector_add(a,b,c,n))
        fbody = ex.args[2]

        # Wrap the body so the function always returns nothing
        wrapped = quote
            $(Expr(:function, fname, quote
                $fbody
                return nothing
            end))
        end

        # Return an expression that assigns the wrapped function to the original name
        return esc(wrapped)
    end

end
