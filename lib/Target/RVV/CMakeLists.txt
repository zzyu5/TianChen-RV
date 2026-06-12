add_mlir_library(TianChenRVRVVTarget
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
