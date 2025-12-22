# Building XKBlas Locally

This directory contains scripts to build XKBlas and its dependency XKRT (xkrt) from source.

## TLDR;
- Run `julia --project=. -e 'using Pkg; Pkg.instantiate(); include("build_local.jl")'`

## Overview

The `build_xkblas.jl` script will:
1. Clone/update XKRT from GitLab (branch: `master`)
2. Clone/update XKBlas from GitLab (branch: `v2.0`)
3. Build and install XKRT to a scratch directory
4. Build and install XKBlas to a scratch directory
5. Configure `LocalPreferences.toml` to use the locally built libraries

## Requirements

- Julia >= 1.11
- CMake >= 3.17
- A C++20 compatible compiler (LLVM/clang >= 20.x recommended)
- hwloc library

## See Also

- [XKBlas Repository](https://gitlab.inria.fr/xkblas/dev)
- [XKRT Repository](https://gitlab.inria.fr/xkaapi/dev-v2)
- Original install script: `../../XKBlas/scripts/install.sh`
