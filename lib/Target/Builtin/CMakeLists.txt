add_mlir_library(TianChenRVBuiltinTargetArtifactExporters
  BuiltinTargetArtifactExporters.cpp
  RVVScalarDispatch.cpp

  LINK_LIBS PUBLIC
  TianChenRVOffloadTarget
  TianChenRVRVVTarget
  TianChenRVScalarTarget
  TianChenRVTarget
  MLIRSupport
)
