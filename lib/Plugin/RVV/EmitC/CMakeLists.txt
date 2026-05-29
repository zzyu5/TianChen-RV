add_mlir_library(TianChenRVRVVEmitCRouteProvider
  RVVEmitCRoutePlanning.cpp
  RVVEmitCRouteProvider.cpp
  RVVEmitCStatementPlanOwners.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVRVVConstructionProtocol
  TianChenRVRVVDialect
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
