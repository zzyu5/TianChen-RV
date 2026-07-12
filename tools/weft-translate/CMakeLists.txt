add_mlir_tool(weft-translate
  weft-translate.cpp
)

target_link_libraries(weft-translate
  PRIVATE
  WeftBuiltinTargetArtifactExporters
  WeftBuiltinPlugins
  WeftInitAll
  WeftRVVTarget
  WeftTarget
  WeftTransforms
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
