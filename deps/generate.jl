using Clang.Generators
using Clang.LibClang.Clang_jll
using Scratch

# XKBlas_pkg = Base.UUID("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88")
# xkblas_dir = get_scratch!(XKBlas_pkg, "xkblas")
# xkblas_include_dir = joinpath(xkblas_dir, "include")
xkblas_include_dir = "/home/rpereira/install/xkrt/debug-a100/include/"

println("Using XKBlas headers in $xkblas_include_dir")

# Load generator options (must include type_map and rename_functions)
options = load_options(joinpath(@__DIR__, "generator.toml"))

# Collect all headers
headers = [
    joinpath(xkblas_include_dir, "xkblas/xkblas.h"),
    joinpath(xkblas_include_dir, "xkblas/kernels.h"),
    # joinpath(xkblas_include_dir, "xkblas/flops.h"), # TODO: almost work in "aggressive" mode, only "_Complex float" are missing
]
@show headers

# Default compiler flags
args = get_default_args()
push!(args, "-I$xkblas_include_dir")

# Create context: only two positional arguments are supported in this version
ctx = create_context(headers, args, options)

# -------------------------------------------------------------------------
# Skip macros that Clang.jl cannot generate
# -------------------------------------------------------------------------
# ctx.options[:macro_filter] = name -> !(startswith(name, "FMULS_") || startswith(name, "FADDS_"))
# -------------------------------------------------------------------------

# Generate bindings
build!(ctx)

println("bindings.jl generated successfully in src/")

# Remove xkblas_ prefix from function names (but keep @ccall names unchanged)
bindings_file = joinpath(@__DIR__, "..", "src", "bindings.jl")
content = read(bindings_file, String)
content = replace(content, r"^function xkblas_(\w+)\("m => s"function \1(")
write(bindings_file, content)

println("Removed xkblas_ prefix from function names")

