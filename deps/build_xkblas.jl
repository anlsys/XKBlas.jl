# build XKBlas with CMake (including XKRT dependency)

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Scratch, Preferences, CMake_jll, Ninja_jll, LibGit2, LLVM_full_jll
using CUDA
# using CUDA_SDK_jll    # TODO: enable me when CUDA_SDK_jll is fixed
using OpenBLAS_jll

XKBlas_pkg = Base.UUID("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88")

# Configuration from install.sh
const XKRT_BRANCH = "master"
const XKBLAS_BRANCH = "v2.0"
const XKRT_URL = "https://gitlab.inria.fr/xkaapi/dev-v2.git"
const XKBLAS_URL = "https://gitlab.inria.fr/xkblas/dev.git"

# get scratch directories for installations
xkrt_install_dir = get_scratch!(XKBlas_pkg, "xkrt")
xkblas_install_dir = get_scratch!(XKBlas_pkg, "xkblas")
rm(xkrt_install_dir; recursive=true, force=true)
rm(xkblas_install_dir; recursive=true, force=true)

# get scratch directories for source code
xkrt_source_dir = get_scratch!(XKBlas_pkg, "xkrt_src")
xkblas_source_dir = get_scratch!(XKBlas_pkg, "xkblas_src")

# get build directories
xkrt_build_dir = if length(ARGS) >= 1
    joinpath(ARGS[1], "xkrt")
else
    mktempdir(prefix="xkrt_build_")
end
mkpath(xkrt_build_dir)

xkblas_build_dir = if length(ARGS) >= 1
    joinpath(ARGS[1], "xkblas")
else
    mktempdir(prefix="xkblas_build_")
end
mkpath(xkblas_build_dir)

#######################
# Clone XKRT source #
#######################
@info "="^60
@info "Cloning/updating XKRT source (branch: $XKRT_BRANCH)..."
@info "="^60

# Check if directory exists and is a valid git repository
if isdir(xkrt_source_dir) && isdir(joinpath(xkrt_source_dir, ".git"))
    @info "XKRT source directory exists, updating..."
    try
        repo = LibGit2.GitRepo(xkrt_source_dir)
        try
            LibGit2.fetch(repo)
            LibGit2.checkout!(repo, XKRT_BRANCH)
            # Pull latest changes
            LibGit2.merge!(repo, fastforward=true)
        finally
            close(repo)
        end
    catch e
        @warn "Failed to update existing repository: $e"
        @info "Removing and re-cloning..."
        rm(xkrt_source_dir; recursive=true, force=true)
        LibGit2.clone(XKRT_URL, xkrt_source_dir; branch=XKRT_BRANCH)
    end
else
    # Remove if directory exists but is not a git repo
    if isdir(xkrt_source_dir)
        @warn "Directory exists but is not a git repository. Removing..."
        rm(xkrt_source_dir; recursive=true, force=true)
    end
    @info "Cloning XKRT from $XKRT_URL"
    LibGit2.clone(XKRT_URL, xkrt_source_dir; branch=XKRT_BRANCH)
end

# Get commit hash for logging
xkrt_hash = let repo = LibGit2.GitRepo(xkrt_source_dir)
    hash = string(LibGit2.head_oid(repo))[1:12]
    close(repo)
    hash
end
@info "Using XKRT commit: $xkrt_hash"

#######################
# Clone XKBlas source #
#######################
@info "="^60
@info "Cloning/updating XKBlas source (branch: $XKBLAS_BRANCH)..."
@info "="^60

# Check if directory exists and is a valid git repository
if isdir(xkblas_source_dir) && isdir(joinpath(xkblas_source_dir, ".git"))
    @info "XKBlas source directory exists, updating..."
    try
        repo = LibGit2.GitRepo(xkblas_source_dir)
        try
            LibGit2.fetch(repo)
            LibGit2.checkout!(repo, XKBLAS_BRANCH)
            # Pull latest changes
            LibGit2.merge!(repo, fastforward=true)
        finally
            close(repo)
        end
    catch e
        @warn "Failed to update existing repository: $e"
        @info "Removing and re-cloning..."
        rm(xkblas_source_dir; recursive=true, force=true)
        LibGit2.clone(XKBLAS_URL, xkblas_source_dir; branch=XKBLAS_BRANCH)
    end
else
    # Remove if directory exists but is not a git repo
    if isdir(xkblas_source_dir)
        @warn "Directory exists but is not a git repository. Removing..."
        rm(xkblas_source_dir; recursive=true, force=true)
    end
    @info "Cloning XKBlas from $XKBLAS_URL"
    LibGit2.clone(XKBLAS_URL, xkblas_source_dir; branch=XKBLAS_BRANCH)
end

# Get commit hash for logging
xkblas_hash = let repo = LibGit2.GitRepo(xkblas_source_dir)
    hash = string(LibGit2.head_oid(repo))[1:12]
    close(repo)
    hash
end
@info "Using XKBlas commit: $xkblas_hash"

# Use LLVM_full_jll compilers (similar to oneAPI's icpx)
clang_path = LLVM_full_jll.clang_path
clangxx_path = clang_path * "++"

@info "Using LLVM_full_jll compilers:"
@info "  CC:  $clang_path"
@info "  CXX: $clangxx_path"

# Check if CUDA is functional
CUDA.set_runtime_version!(local_toolkit=true)
use_cuda = CUDA.functional()
@info "CUDA.functional() = $use_cuda"

# Use CUDA_SDK_jll for CUDA path if available
cmake_prefix_path = ""
if use_cuda
    # append CUDA_PATH to prefix path
    cmake_prefix_path=ENV["CUDA_PATH"]

    # bellow is buggy version using CUDA_SDK_jll
    #    cuda_path = joinpath(CUDA_SDK_jll.artifact_dir, "cuda")
    #    @info "Using CUDA_SDK_jll:"
    #    @info "  CUDA path: $cuda_path"
    #    @info "  nvcc: $(joinpath(cuda_path, "bin", "nvcc"))"
    #    cmake_prefix_path = cuda_path
else
    @warn "CUDA is not functional. Building without CUDA support."
end

#######################
# Build XKRT (xkrt) #
#######################
@info "="^60
@info "Building XKRT (xkrt) dependency..."
@info "="^60

# Use lld linker from LLVM_full_jll instead of system linker
lld_path = joinpath(dirname(clang_path), "ld.lld")
@info "Using lld linker from LLVM_full_jll: $lld_path"

xkrt_cmake_options = String[
    "-DCMAKE_C_COMPILER=$clang_path",
    "-DCMAKE_CXX_COMPILER=$clangxx_path",
    "-DCMAKE_LINKER=$lld_path",
    "-DCMAKE_BUILD_TYPE=Debug",
    "-DCMAKE_INSTALL_PREFIX=$xkrt_install_dir",
    "-DSTRICT=OFF",
    "-DUSE_STATS=ON",
    "-DUSE_CUDA=$(use_cuda ? "on" : "off")",
    "-GNinja",
]

if use_cuda && !isempty(cmake_prefix_path)
    # Help CMake find CUDA from CUDA_SDK_jll
    push!(xkrt_cmake_options, "-DCMAKE_PREFIX_PATH=$cmake_prefix_path")
    push!(xkrt_cmake_options, "-DCUDAToolkit_ROOT=$cmake_prefix_path")
elseif !isempty(cmake_prefix_path)
    push!(xkrt_cmake_options, "-DCMAKE_PREFIX_PATH=$cmake_prefix_path")
end

cmake() do cmake_path
    ninja() do ninja_path
        # Configure XKRT
        cmd = `$cmake_path $xkrt_cmake_options -S $xkrt_source_dir -B $xkrt_build_dir`
        @info "Configuring XKRT..." cmd
        run(cmd)

        # Build and install XKRT
        cmd = `$cmake_path --build $xkrt_build_dir --target install`
        @info "Building and installing XKRT..." cmd
        run(cmd)
    end
end

# Verify XKRT installation
xkrt_lib = joinpath(xkrt_install_dir, "lib", "libxkrt.so")
if !ispath(xkrt_lib)
    xkrt_lib = joinpath(xkrt_install_dir, "lib64", "libxkrt.so")
end
@assert ispath(xkrt_lib) "Could not find libxkrt.so at $xkrt_lib"
@info "XKRT installed successfully at: $xkrt_install_dir"

##################
# Build XKBlas   #
##################
@info "="^60
@info "Building XKBlas..."
@info "="^60

# Update CMAKE_PREFIX_PATH to include XKRT
if !isempty(cmake_prefix_path)
    cmake_prefix_path = "$cmake_prefix_path:$xkrt_install_dir"
else
    cmake_prefix_path = xkrt_install_dir
end

# Get OpenBLAS paths from OpenBLAS_jll
openblas_dir = OpenBLAS_jll.artifact_dir
openblas_include = joinpath(openblas_dir, "include")
openblas_lib = dirname(OpenBLAS_jll.libopenblas_path)

@info "Using OpenBLAS_jll:"
@info "  Include: $openblas_include"
@info "  Library: $openblas_lib"

xkblas_cmake_options = String[
    "-DCMAKE_C_COMPILER=$clang_path",
    "-DCMAKE_CXX_COMPILER=$clangxx_path",
    "-DCMAKE_LINKER=$lld_path",
    "-DCMAKE_C_FLAGS=-Wno-error",
    "-DCMAKE_CXX_FLAGS=-Wno-error",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$xkblas_install_dir",
    "-DCMAKE_PREFIX_PATH=$cmake_prefix_path",
    "-DXKRT_DIR=$(joinpath(xkrt_install_dir, "lib", "cmake", "XKRT"))",
    "-DSTRICT=OFF",
    "-DUSE_CUDA=$(use_cuda ? "on" : "off")",
    "-DUSE_CUBLAS=$(use_cuda ? "on" : "off")",
    "-DUSE_CUSPARSE=$(use_cuda ? "on" : "off")",
    "-DUSE_CBLAS=off",    # OpenBLAS_jll doesn't include lapacke.h needed for tests
    "-DUSE_TESTS=off",    # Disable tests - they require LAPACK headers
    "-DUSE_OPENBLAS=off",
    "-GNinja",
]

# Add CUDAToolkit_ROOT hint if using CUDA
if use_cuda
    push!(xkblas_cmake_options, "-DCUDAToolkit_ROOT=$(split(cmake_prefix_path, ':')[1])")
end

# build and install XKBlas
cmake() do cmake_path
    ninja() do ninja_path
        # Configure
        cmd = `$cmake_path $xkblas_cmake_options -S $xkblas_source_dir -B $xkblas_build_dir`
        @info "Configuring XKBlas..." cmd
        run(cmd)

        # Build and install
        cmd = `$cmake_path --build $xkblas_build_dir --target install`
        @info "Building and installing XKBlas..." cmd
        run(cmd)
    end
end

# Find the installed libraries
# TODO: adapt when we support more platforms
xkblas_lib_path = joinpath(xkblas_install_dir, "lib", "libxkblas.so")
if !ispath(xkblas_lib_path)
    # Try lib64
    xkblas_lib_path = joinpath(xkblas_install_dir, "lib64", "libxkblas.so")
end
@assert ispath(xkblas_lib_path) "Could not find libxkblas.so at $xkblas_lib_path"

xkrt_lib_path = joinpath(xkrt_install_dir, "lib", "libxkrt.so")
if !ispath(xkrt_lib_path)
    xkrt_lib_path = joinpath(xkrt_install_dir, "lib64", "libxkrt.so")
end
@assert ispath(xkrt_lib_path) "Could not find libxkrt.so at $xkrt_lib_path"

@info "="^60
@info "Build complete!"
@info "XKBlas library: $xkblas_lib_path"
@info "XKRT library:   $xkrt_lib_path"
@info "="^60

# tell XKBlas_jll and XKRT_jll to load our libraries instead of the default artifact ones
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "XKBlas_jll",
    "libxkblas_path" => xkblas_lib_path;
    force=true,
)

set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "XKRT_jll",
    "libxkrt_path" => xkrt_lib_path;
    force=true,
)

# copy the preferences to `test/` as well to work around Pkg.jl#2500
cp(joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
   joinpath(dirname(@__DIR__), "test", "LocalPreferences.toml"); force=true)

@info "Library paths saved to LocalPreferences.toml"
