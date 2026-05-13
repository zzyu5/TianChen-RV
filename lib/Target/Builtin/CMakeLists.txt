add_mlir_library(TianChenRVBuiltinTargetArtifactExporters
  BuiltinTargetArtifactExporters.cpp
  BuiltinTargetTranslateRoutes.cpp

  LINK_LIBS PUBLIC
  TianChenRVBuiltinPlugins
  TianChenRVOffloadTarget
  TianChenRVRVVTarget
  TianChenRVScalarTarget
  TianChenRVSupport
  TianChenRVTarget
  TianChenRVTemplateTarget
  TianChenRVToyTarget
  MLIRSupport
)
