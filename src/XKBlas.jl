module XKBlas

    using Libdl
    using Scratch

    XKBlas_pkg = Base.UUID("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88")
    xkblas_dir = get_scratch!(XKBlas_pkg, "xkblas")

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
    include("KA/KA.jl")
    include("threading.jl")

    # Init /Deinit xkblas - call when module is used
    function __init__()
        XKBlas.init()
        XKBlas.KA.init()
        XKBlas.Threading.init()
        function cleanup()
            XKBlas.KA.deinit()
            XKBlas.Threading.deinit()
            XKBlas.deinit()
        end
        atexit(cleanup)
    end

end
