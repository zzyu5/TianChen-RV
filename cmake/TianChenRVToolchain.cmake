function(tianchenrv_find_tool output_variable)
  cmake_parse_arguments(ARG "" "" "NAMES" ${ARGN})
  if(NOT ARG_NAMES)
    message(FATAL_ERROR "tianchenrv_find_tool requires NAMES")
  endif()

  set(_tool_hints
    "${LLVM_TOOLS_BINARY_DIR}"
    "${LLVM_INSTALL_PREFIX}/bin"
    "/usr/lib/llvm-${LLVM_VERSION_MAJOR}/bin"
    "/usr/lib/llvm-${LLVM_VERSION_MAJOR}/build/utils/lit"
  )

  find_program(${output_variable}
    NAMES ${ARG_NAMES}
    HINTS ${_tool_hints}
  )

  set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
endfunction()

function(tianchenrv_require_tool variable_name human_name)
  if(NOT ${variable_name})
    message(FATAL_ERROR
      "TianChen-RV requires ${human_name}. "
      "Ensure the LLVM/MLIR tool directory is on PATH or configure with LLVM_DIR/MLIR_DIR from a complete installation.")
  endif()
  message(STATUS "TianChen-RV ${human_name}: ${${variable_name}}")
endfunction()

function(tianchenrv_require_testing_tools)
  tianchenrv_find_tool(TIANCHENRV_FILECHECK NAMES FileCheck)
  tianchenrv_require_tool(TIANCHENRV_FILECHECK "FileCheck")

  tianchenrv_find_tool(TIANCHENRV_LLVM_LIT NAMES llvm-lit lit lit.py)
  tianchenrv_require_tool(TIANCHENRV_LLVM_LIT "llvm-lit/lit")

  if(NOT LLVM_EXTERNAL_LIT)
    set(LLVM_EXTERNAL_LIT "${TIANCHENRV_LLVM_LIT}" CACHE STRING "Command used to spawn lit" FORCE)
  endif()
endfunction()
