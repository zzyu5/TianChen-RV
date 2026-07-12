function(weft_find_tool output_variable)
  cmake_parse_arguments(ARG "" "" "NAMES" ${ARGN})
  if(NOT ARG_NAMES)
    message(FATAL_ERROR "weft_find_tool requires NAMES")
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

function(weft_require_tool variable_name human_name)
  if(NOT ${variable_name})
    message(FATAL_ERROR
      "Weft-RV requires ${human_name}. "
      "Ensure the LLVM/MLIR tool directory is on PATH or configure with LLVM_DIR/MLIR_DIR from a complete installation.")
  endif()
  message(STATUS "Weft-RV ${human_name}: ${${variable_name}}")
endfunction()

function(weft_require_testing_tools)
  weft_find_tool(WEFT_FILECHECK NAMES FileCheck)
  weft_require_tool(WEFT_FILECHECK "FileCheck")

  weft_find_tool(WEFT_LLVM_LIT NAMES llvm-lit lit lit.py)
  weft_require_tool(WEFT_LLVM_LIT "llvm-lit/lit")

  if(NOT LLVM_EXTERNAL_LIT)
    set(LLVM_EXTERNAL_LIT "${WEFT_LLVM_LIT}" CACHE STRING "Command used to spawn lit" FORCE)
  endif()
endfunction()
