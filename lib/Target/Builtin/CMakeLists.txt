add_mlir_library(TianChenRVBuiltinTargetArtifactExporters
  BuiltinTargetArtifactExporters.cpp
  RVVScalarDispatch.cpp

  LINK_LIBS PUBLIC
  TianChenRVOffloadTarget
  TianChenRVRVVTarget
  TianChenRVScalarTarget
  TianChenRVSupport
  TianChenRVTarget
  MLIRSupport
)
