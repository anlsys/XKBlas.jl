using XKBlas

# spawn two tasks
x = 0

# first one sets 'x' to 42
XKBlas.host_async(
    () -> begin
        global x = 42
    end,
    set_accesses = (accesses) -> begin
        push!(accesses, Access(ACCESS_MODE_VW, Segment(0, 64)))
    end
)

# second one sets 'x' to 43 - with an intersecting access to set a dependency
XKBlas.host_async(
    () -> begin
        @assert x == 42
        global x = 43
    end,
    set_accesses = (accesses) -> begin
        push!(accesses, Access(ACCESS_MODE_VR, Segment(16, 48)))
    end
)

# wait for task execution
XKBlas.sync()
println("x is $x")
@assert x == 43
