using XKBlas

x = 0
XKBlas.host_async(() -> begin global x = 42 end)
XKBlas.sync()
@assert x == 42

# enforced to use a Julia synchronization here, not very manageable on the runtime side
done = Channel{Any}(1)
XKBlas.host_async(() -> begin
    func = () -> begin
        println("Hello world")
        global x = 43
        put!(done, nothing)
    end
    XKBlas.KA.Threading.submit_julia_runtime_lambda(func)
end)

# wait for the task to execute
XKBlas.sync()

# wait for the julia lambda to complete
_ = take!(done)
