add_mlir_tool(weft-opt
  weft-opt.cpp
)

target_link_libraries(weft-opt
  PRIVATE
  WeftBuiltinTargetArtifactExporters
  WeftBuiltinPlugins
  WeftInitAll
  WeftExecDialect
  WeftTarget
  WeftTransforms
  WeftConversionRVV
  MLIROptLib
  MLIRIR
  MLIRSupport
)
