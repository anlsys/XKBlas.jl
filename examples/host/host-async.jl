using XKLas

x = 0
XKLas.host_async(() -> begin global x = 42 end)
XKLas.sync()
@assert x == 42

# enforced to use a Julia synchronization here, not very manageable on the runtime side
done = Channel{Any}(1)
XKLas.host_async(() -> begin
    func = () -> begin
        println("Hello world")
        global x = 43
        put!(done, nothing)
    end
    XKLas.KA.Threading.submit_julia_runtime_lambda(func)
end)

# wait for the task to execute
XKLas.sync()

# wait for the julia lambda to complete
_ = take!(done)
