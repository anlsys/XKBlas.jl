# Spawn Julia thread to run Julia code
# This design is required by Julia not allowing external threads running Julia code
# that leads to deadlocks in the Julia task scheduler mostly
module Threading

    const TASK_CHANNEL = Ref{Channel{Any}}()
    const COND_HANDLE  = Ref{Base.AsyncCondition}()

    # Tells XKRT to run that Julia routine that may enter the Julia runtime.
    # In such case, special treatment is needed to avoid deadlocks, as Julia do not allow
    # entering the Julia runtime from a foreign thread... That's a dirty
    # workd-around, no better solution
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
