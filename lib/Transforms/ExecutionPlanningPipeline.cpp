#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace tianchenrv::transforms {
namespace {

constexpr llvm::StringLiteral kExecutionPlanningPipelineName(
    "tcrv-execution-planning-pipeline");
constexpr llvm::StringLiteral kExecutionPlanningPipelineDescription(
    "Compose TianChen-RV plugin variant materialization, capability "
    "checking, generic selection/dispatch planning, selected lowering-boundary "
    "materialization, emission-plan diagnostics, and execution-plan coherence "
    "checking");

} // namespace

void buildExecutionPlanningPipeline(mlir::OpPassManager &pm) {
  static const plugin::ExtensionPluginRegistry emptyRegistry;
  buildExecutionPlanningPipeline(pm, emptyRegistry);
}

void buildExecutionPlanningPipeline(
    mlir::OpPassManager &pm,
    const plugin::ExtensionPluginRegistry &registry) {
  static const target::TargetArtifactExporterRegistry emptyTargetExporters;
  buildExecutionPlanningPipeline(pm, registry, emptyTargetExporters);
}

void buildExecutionPlanningPipeline(
    mlir::OpPassManager &pm, const plugin::ExtensionPluginRegistry &registry,
    const target::TargetArtifactExporterRegistry &targetExporters) {
  pm.addPass(createMaterializePluginVariantsPass(registry));
  pm.addPass(createVerifyPluginVariantLegalityPass(registry));
  pm.addPass(createSelectVariantsPass(registry));
  pm.addPass(createMaterializeDispatchRuntimeGuardsPass());
  pm.addPass(createCheckCapabilityRequiresPass());
  pm.addPass(createMaterializeSelectedLoweringBoundariesPass(registry));
  pm.addPass(createMaterializeEmissionPlansPass(registry));
  pm.addPass(createCheckExecutionPlanCoherencePass(registry, targetExporters));
}

void registerExecutionPlanningPipeline() {
  static const plugin::ExtensionPluginRegistry emptyRegistry;
  registerExecutionPlanningPipeline(emptyRegistry);
}

void registerExecutionPlanningPipeline(
    const plugin::ExtensionPluginRegistry &registry) {
  static const target::TargetArtifactExporterRegistry emptyTargetExporters;
  registerExecutionPlanningPipeline(registry, emptyTargetExporters);
}

void registerExecutionPlanningPipeline(
    const plugin::ExtensionPluginRegistry &registry,
    const target::TargetArtifactExporterRegistry &targetExporters) {
  mlir::PassPipelineRegistration<> registration(
      kExecutionPlanningPipelineName, kExecutionPlanningPipelineDescription,
      [&registry, &targetExporters](mlir::OpPassManager &pm) {
        buildExecutionPlanningPipeline(pm, registry, targetExporters);
      });
  (void)registration;
}

} // namespace tianchenrv::transforms
