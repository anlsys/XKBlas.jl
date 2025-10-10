if length(ARGS) != 1
    println("Usage: test-local-build.jl <example-file-using-XKBlas>")
    exit(1)
end

using Pkg

# Activate the package root as the environment
Pkg.activate("./deps/")
Pkg.instantiate()
Pkg.develop(path=".")

filename = ARGS[1]
println("Processing file: $filename")
include(filename)
