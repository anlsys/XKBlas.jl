using XKBlas

XKBlas.@host_async begin
    body = () -> nothing
end

XKBlas.sync()
