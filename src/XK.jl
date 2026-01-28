module XK

    using Libdl
    using Scratch

    XK_pkg = Base.UUID("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88")
    xkblas_dir = get_scratch!(XK_pkg, "xkblas")

    @show xkblas_dir
    # --- Load the library handle ---
    # libpath = Libdl.find_library(["libxkblas"], [joinpath(xkblas_dir, "lib")])
    libpath = joinpath(xkblas_dir, "lib/libxkblas.so")
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

    include("BLAS/BLAS.jl")
    include("KA/KA.jl")

    # Init /Deinit xkblas - call when module is used
    function __init__()
        XK.init()
        XK.KA.init()
        XK.Threading.init()
        function cleanup()
            XK.KA.deinit()
            XK.Threading.deinit()
            XK.deinit()
        end
        atexit(cleanup)
    end

end
