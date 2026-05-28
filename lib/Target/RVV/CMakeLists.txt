add_mlir_library(TianChenRVRVVTarget
  RVVTargetArtifactRouteFamilyValidation.cpp
  RVVTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVRVVEmitCRouteProvider
  TianChenRVRVVDialect
  TianChenRVSupport
  TianChenRVTransforms
  TianChenRVTarget
  MLIRIR
  MLIRPass
  MLIRSupport
)
