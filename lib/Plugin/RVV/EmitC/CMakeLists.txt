add_mlir_library(TianChenRVRVVEmitCRouteProvider
  RVVEmitCRouteProvider.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVRVVDialect
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
