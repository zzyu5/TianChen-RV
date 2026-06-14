add_mlir_library(TianChenRVRVVEmitCRouteProvider
  RVVEmitCRoutePlanning.cpp
  RVVEmitCRoutePlanningPrimitives.cpp
  RVVEmitCRouteFamilyDerivation.cpp
  RVVEmitCRouteConfigBinding.cpp
  RVVEmitCRouteAnalysis.cpp
  RVVEmitCRouteVerification.cpp
  RVVEmitCRouteMetadata.cpp
  RVVEmitCRouteProvider.cpp
  RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp
  RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp
  RVVEmitCContractionRouteFamilyPlanOwners.cpp
  RVVEmitCControlPolicyPlanOwners.cpp
  RVVEmitCElementwiseRouteFamilyPlanOwners.cpp
  RVVLowPrecisionPerformancePolicy.cpp
  RVVEmitCMAccRouteFamilyPlanOwners.cpp
  RVVEmitCSegment2RouteFamilyPlanOwners.cpp

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
