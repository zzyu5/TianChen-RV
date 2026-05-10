add_mlir_library(TianChenRVBuiltinTargetArtifactExporters
  BuiltinTargetArtifactExporters.cpp
  BuiltinTargetTranslateRoutes.cpp
  RVVScalarDispatch.cpp

  LINK_LIBS PUBLIC
  TianChenRVBuiltinPlugins
  TianChenRVOffloadTarget
  TianChenRVRVVTarget
  TianChenRVScalarTarget
  TianChenRVSupport
  TianChenRVTarget
  TianChenRVToyTarget
  MLIRSupport
)
