module XKLas

    using Libdl
    using Scratch

    XKLas_pkg = Base.UUID("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88")
    xkblas_dir = get_scratch!(XKLas_pkg, "xkblas")

    @show xkblas_dir
    # --- Load the library handle ---
    libpath = Libdl.find_library(["libxkblas"], [joinpath(xkblas_dir, "lib")])
    @show libpath
    if libpath === nothing
        error("libxkblas not found. Make sure the library is built using `deps/build_local.jl`.")
    end
    const libxkblas = Libdl.LazyLibrary(libpath)

    # --- Include generated bindings ---
    const size_t = Csize_t
    include("bindings.jl")

    # --- High-level wrappers ---
    include("wrappers.jl")
    include("logger.jl")
    include("threading.jl")

    include("KA/KA.jl")

    # Init /Deinit xkblas - call when module is used
    function __init__()
        XKLas.init()
        XKLas.KA.init()
        XKLas.Threading.init()
        function cleanup()
            XKLas.KA.deinit()
            XKLas.Threading.deinit()
            XKLas.deinit()
        end
        atexit(cleanup)
    end

end
