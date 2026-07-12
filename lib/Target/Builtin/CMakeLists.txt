add_mlir_library(WeftBuiltinTargetArtifactExporters
  BuiltinTargetArtifactExporters.cpp
  BuiltinTargetTranslateRoutes.cpp

  LINK_LIBS PUBLIC
  WeftBuiltinPlugins
  WeftSupport
  WeftTarget
  MLIRSupport
)
