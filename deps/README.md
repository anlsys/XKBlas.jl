# Building XKBlas Locally

This directory contains scripts to build XKBlas and its dependency XKAAPI (xkrt) from source.

## TLDR;
- Run `julia --project=. -e 'using Pkg; Pkg.instantiate(); include("build_local.jl")'`

## Overview

The `build_xkblas.jl` script will:
1. Clone/update XKAAPI from GitLab (branch: `master`)
2. Clone/update XKBlas from GitLab (branch: `v2.0`)
3. Build and install XKAAPI to a scratch directory
4. Build and install XKBlas to a scratch directory
5. Configure `LocalPreferences.toml` to use the locally built libraries

## Requirements

- Julia >= 1.6
- CMake >= 3.17
- Ninja build system
- A C++20 compatible compiler (LLVM/clang >= 20.x recommended)
- CUDA toolkit (optional, for GPU support)
- hwloc library

## Usage

### Basic build

```bash
cd /path/to/XKBlas.jl/deps
julia build_xkblas.jl
```

This will:
- Clone the repositories to scratch directories
- Build both libraries in temporary directories
- Install to scratch directories managed by Julia

### Build with custom build directory

```bash
julia build_xkblas.jl /path/to/build/dir
```

This allows you to inspect the build artifacts and reuse the build directory.

## Configuration

### Compiler Selection

The script automatically uses **LLVM 20** compilers from `LLVM_full_jll`:
- `clang` from LLVM_full_jll for C compilation
- `clang++` from LLVM_full_jll for C++ compilation

This ensures consistent builds across all platforms without requiring system compilers.

### CUDA Support

The script automatically detects CUDA availability using `CUDA.functional()`:
- **If CUDA is functional**: Uses **CUDA_SDK_jll** artifact for CUDA support
  - CUDA toolkit path: `CUDA_SDK_jll.artifact_dir/cuda`
  - `nvcc` compiler location: automatically detected in `bin/` subfolder
  - Enables: `USE_CUDA`, `USE_CUBLAS`, `USE_CUSPARSE`
- **If CUDA is not functional**: Builds without CUDA support (CPU-only)
  - Only CPU BLAS operations will be available
  - `USE_CUDA`, `USE_CUBLAS`, `USE_CUSPARSE` are disabled

No manual CUDA installation or environment variables are required!

### Branch Selection

The branches are configured at the top of `build_xkblas.jl`:
- `XKAAPI_BRANCH`: Currently set to `"master"`
- `XKBLAS_BRANCH`: Currently set to `"v2.0"`

You can modify these constants in the script if you need different branches.

## Build Options

The script enables the following CMake options for XKBlas:
- `USE_CUDA=on` - CUDA support
- `USE_CUBLAS=on` - CUBLAS library support
- `USE_CUSPARSE=on` - CUSPARSE library support
- `USE_CBLAS=on` - CPU BLAS support
- `USE_TESTS=on` - Build tests
- `USE_OPENBLAS=on` - OpenBLAS support

## Scratch Directories

The script uses Julia's `Scratch.jl` to manage installation directories:
- Source code: `~/.julia/scratchspaces/<uuid>/xkaapi_src` and `xkblas_src`
- Installation: `~/.julia/scratchspaces/<uuid>/xkaapi` and `xkblas`

To clean up and rebuild:
```julia
using Scratch
delete_scratch!("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88", "xkaapi")
delete_scratch!("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88", "xkblas")
delete_scratch!("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88", "xkaapi_src")
delete_scratch!("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88", "xkblas_src")
```

## Troubleshooting

### Missing compiler
If you get a compiler error, install clang or gcc:
```bash
# Ubuntu/Debian
sudo apt-get install clang-20 llvm-20

# Or use system clang
sudo apt-get install clang
```

### CUDA not found
Set the `CUDA_PATH` environment variable to your CUDA installation directory.

### Git authentication issues
If cloning from GitLab fails, you may need to set up SSH keys or use HTTPS authentication.

## See Also

- [XKBlas Repository](https://gitlab.inria.fr/xkblas/dev)
- [XKAAPI Repository](https://gitlab.inria.fr/xkaapi/dev-v2)
- Original install script: `../../XKBlas/scripts/install.sh`
