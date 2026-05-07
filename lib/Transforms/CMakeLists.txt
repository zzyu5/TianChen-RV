add_mlir_library(TianChenRVTransforms
  CheckCapabilityRequires.cpp
  EmissionReadiness.cpp
  ExecutionPlanningPipeline.cpp
  LoweringBoundary.cpp
  VariantDispatchSynthesis.cpp
  VariantMaterialization.cpp
  VariantSelection.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRPass
  MLIRSupport
)
