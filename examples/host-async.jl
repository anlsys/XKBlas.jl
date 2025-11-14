using XKBlas

x = 0
XKBlas.host_async(() -> begin global x = 42 end)
XKBlas.sync()
@assert x == 42

XKBlas.host_async(() -> begin println("DEADLOCK KEK") end)
XKBlas.sync()
