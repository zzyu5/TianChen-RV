cmake_minimum_required(VERSION 3.20)

project(TianChenRVMLIR LANGUAGES C CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

option(TIANCHENRV_ENABLE_TESTS "Enable TianChen-RV lit/FileCheck tests" ON)

set(_TCRV_LLVM_PREFIX_HINTS
  /usr/lib/llvm-20
  /usr/lib/llvm-19
  /usr/lib/llvm-18
  /usr/lib/llvm-17
  /usr/lib/llvm-16
  /usr/lib/llvm-15
  /usr/lib/llvm-14
)

find_package(LLVM CONFIG QUIET HINTS ${_TCRV_LLVM_PREFIX_HINTS})
if(NOT LLVM_FOUND)
  message(FATAL_ERROR
    "TianChen-RV requires an LLVM CMake package. "
    "Install LLVM/MLIR development packages or pass -DLLVM_DIR=/path/to/lib/cmake/llvm.")
endif()

find_package(MLIR CONFIG QUIET HINTS ${_TCRV_LLVM_PREFIX_HINTS})
if(NOT MLIR_FOUND)
  message(FATAL_ERROR
    "TianChen-RV requires an MLIR CMake package. "
    "Install MLIR development packages or pass -DMLIR_DIR=/path/to/lib/cmake/mlir.")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}: ${LLVM_DIR}")
message(STATUS "Found MLIR: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(TianChenRVToolchain)

set(TIANCHENRV_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TIANCHENRV_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR})
file(MAKE_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR})
file(MAKE_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(TIANCHENRV_ENABLE_TESTS)
  tianchenrv_require_testing_tools()
  enable_testing()
  add_subdirectory(test)
endif()
