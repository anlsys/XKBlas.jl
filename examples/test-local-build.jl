using Pkg

# Activate the package root as the environment
Pkg.activate("./deps/")
Pkg.instantiate()
Pkg.develop(path=".")

using XKBlas
