add_mlir_library(WeftTransforms
  CheckCapabilityRequires.cpp
  DispatchRuntimeGuard.cpp
  EmitCLowerableMaterialization.cpp
  EmissionReadiness.cpp
  ExecutionPlanCoherence.cpp
  ExecutionPlanningPipeline.cpp
  FamilyConstruction.cpp
  HartParallelCapabilities.cpp
  LoweringBoundary.cpp
  PluginVariantLegality.cpp
  VariantDispatchSynthesis.cpp
  VariantMaterialization.cpp
  VariantSelection.cpp

  DEPENDS
  WeftPassesIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftConversionEmitCBuiltin
  WeftExecDialect
  WeftPlugin
  WeftSupport
  WeftTarget
  MLIRIR
  MLIRPass
  MLIRSupport
)
