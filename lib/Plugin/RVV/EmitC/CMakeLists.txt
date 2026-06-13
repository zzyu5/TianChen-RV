add_mlir_library(TianChenRVRVVEmitCRouteProvider
  RVVEmitCRoutePlanning.cpp
  RVVEmitCRouteProvider.cpp
  RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp
  RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp
  RVVEmitCContractionRouteFamilyPlanOwners.cpp
  RVVEmitCControlPolicyPlanOwners.cpp
  RVVEmitCElementwiseRouteFamilyPlanOwners.cpp
  RVVLowPrecisionPerformancePolicy.cpp
  RVVEmitCMAccRouteFamilyPlanOwners.cpp
  RVVEmitCResidualStatementPlanOwners.cpp
  RVVEmitCSegment2RouteFamilyPlanOwners.cpp
  RVVEmitCStatementPlanOwners.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVConversionRVV
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVRVVConstructionProtocol
  TianChenRVRVVDialect
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
