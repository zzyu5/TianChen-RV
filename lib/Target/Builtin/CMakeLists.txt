add_mlir_library(TianChenRVBuiltinTargetArtifactExporters
  BuiltinTargetArtifactExporters.cpp
  BuiltinTargetTranslateRoutes.cpp

  LINK_LIBS PUBLIC
  TianChenRVBuiltinPlugins
  TianChenRVRVVTarget
  TianChenRVSupport
  TianChenRVTarget
  TianChenRVTemplateTarget
  TianChenRVTensorExtLiteTarget
  TianChenRVToyTarget
  MLIRSupport
)
