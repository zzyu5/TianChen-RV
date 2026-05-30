add_mlir_library(TianChenRVRVVEmitCRouteProvider
  RVVEmitCRoutePlanning.cpp
  RVVEmitCRouteProvider.cpp
  RVVEmitCContractionRouteFamilyPlanOwners.cpp
  RVVEmitCControlPolicyPlanOwners.cpp
  RVVEmitCElementwiseRouteFamilyPlanOwners.cpp
  RVVEmitCMAccRouteFamilyPlanOwners.cpp
  RVVEmitCCompareSelectStatementPlanOwners.cpp
  RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp
  RVVEmitCMemoryStatementPlanOwners.cpp
  RVVEmitCReductionAccumulationStatementPlanOwners.cpp
  RVVEmitCResidualStatementPlanOwners.cpp
  RVVEmitCSegment2RouteFamilyPlanOwners.cpp
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
