#!/bin/bash

# only do this once
# sudo dnf install gcc-toolset-14 clang lld zlib-devel
TRITON_LLVM_SYSTEM_SUFFIX=almalinux-x64  TRITON_BUILD_WITH_CLANG_LLD=1 DEBUG=1 CMAKE_CXX_COMPILER_LAUNCHER="" CMAKE_C_COMPILER_LAUNCHER="" time pip install -e . --no-build-isolation

