add_mlir_library(TianChenRVTransforms
  CheckCapabilityRequires.cpp
  DispatchRuntimeGuard.cpp
  EmissionReadiness.cpp
  ExecutionPlanCoherence.cpp
  ExecutionPlanningPipeline.cpp
  HartParallelCapabilities.cpp
  LowerLinalgI32BinaryToExec.cpp
  LoweringBoundary.cpp
  PluginVariantLegality.cpp
  VariantDispatchSynthesis.cpp
  VariantMaterialization.cpp
  VariantSelection.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTarget
  MLIRIR
  MLIRPass
  MLIRSupport
)
