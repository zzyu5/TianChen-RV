add_mlir_library(TianChenRVBuiltinTargetArtifactExporters
  BuiltinTargetArtifactExporters.cpp
  BuiltinTargetTranslateRoutes.cpp

  LINK_LIBS PUBLIC
  TianChenRVBuiltinPlugins
  TianChenRVSupport
  TianChenRVTarget
  MLIRSupport
)
