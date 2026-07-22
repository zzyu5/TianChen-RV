add_mlir_library(WeftRVVEmitCRouteProvider
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
  RVVEmitCContractionRouteFamilyCommon.cpp
  RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp
  RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp
  RVVEmitCContractionRouteFamilyValidation.cpp
  RVVEmitCControlPolicyPlanOwners.cpp
  RVVEmitCElementwiseRouteFamilyPlanOwners.cpp
  RVVEmitCMAccRouteFamilyPlanOwners.cpp
  RVVEmitCSegment2RouteFamilyPlanOwners.cpp

  DEPENDS
  WeftPassesIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftConversionRVV
  WeftExecDialect
  WeftPlugin
  WeftRVVConstructionProtocol
  WeftRVVDialect
  WeftSupport
  MLIRIR
  MLIRSupport
)
