cmake_minimum_required(VERSION 3.20.0)
project(toyc LANGUAGES CXX C)

if(NOT LLVM_LIBRARY_DIR)
  # TODO use a wildcard to avoid put llvm hash on the path
  set(LLVM_LIBRARY_DIR "$ENV{HOME}/.triton/llvm/llvm-4017f04e-ubuntu-x64/lib")
endif()

if (NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(STATUS "MLIR_DIR ${MLIR_DIR}")
  find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})
  message(STATUS "LLVM_CMAKE_DIR ${LLVM_CMAKE_DIR}")
  message(STATUS "MLIR_CMAKE_DIR ${MLIR_CMAKE_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
else()
  message(FATAL_ERROR "this branch not supported yet")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories("include")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)  # tblgen generated headers

add_llvm_executable(toyc toyc.cpp Dialect.cpp)

add_subdirectory(include/toy)
add_dependencies(toyc tblgen_target)

# This has to come after add_llvm_executable so that toyc is defined
target_link_libraries(toyc PUBLIC
  MLIRSupport
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRFunctionInterfaces
  MLIRCastInterfaces
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRFuncAllExtensions
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMemRefDialect
  MLIRSCFToControlFlow
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRControlFlowToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRExecutionEngine

  LLVMOrcJIT
)
