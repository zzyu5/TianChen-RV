add_mlir_library(TianChenRVTransforms
  CheckCapabilityRequires.cpp
  DispatchRuntimeGuard.cpp
  EmitCLowerableMaterialization.cpp
  EmissionReadiness.cpp
  ExecutionPlanCoherence.cpp
  ExecutionPlanningPipeline.cpp
  HartParallelCapabilities.cpp
  LoweringBoundary.cpp
  PluginVariantLegality.cpp
  VariantDispatchSynthesis.cpp
  VariantMaterialization.cpp
  VariantSelection.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVConversionEmitCBuiltin
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTarget
  MLIRIR
  MLIRPass
  MLIRSupport
)
