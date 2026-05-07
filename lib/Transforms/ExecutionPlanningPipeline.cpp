#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace tianchenrv::transforms {
namespace {

constexpr llvm::StringLiteral kExecutionPlanningPipelineName(
    "tcrv-execution-planning-pipeline");
constexpr llvm::StringLiteral kExecutionPlanningPipelineDescription(
    "Compose TianChen-RV plugin variant materialization, capability "
    "checking, generic selection/dispatch planning, selected lowering-boundary "
    "materialization, and emission-plan diagnostics");

} // namespace

void buildExecutionPlanningPipeline(mlir::OpPassManager &pm) {
  static const plugin::ExtensionPluginRegistry emptyRegistry;
  buildExecutionPlanningPipeline(pm, emptyRegistry);
}

void buildExecutionPlanningPipeline(
    mlir::OpPassManager &pm,
    const plugin::ExtensionPluginRegistry &registry) {
  pm.addPass(createMaterializePluginVariantsPass(registry));
  pm.addPass(createCheckCapabilityRequiresPass());
  pm.addPass(createSelectVariantsPass(registry));
  pm.addPass(createCheckCapabilityRequiresPass());
  pm.addPass(createMaterializeSelectedLoweringBoundariesPass(registry));
  pm.addPass(createMaterializeEmissionPlansPass(registry));
}

void registerExecutionPlanningPipeline() {
  static const plugin::ExtensionPluginRegistry emptyRegistry;
  registerExecutionPlanningPipeline(emptyRegistry);
}

void registerExecutionPlanningPipeline(
    const plugin::ExtensionPluginRegistry &registry) {
  mlir::PassPipelineRegistration<> registration(
      kExecutionPlanningPipelineName, kExecutionPlanningPipelineDescription,
      [&registry](mlir::OpPassManager &pm) {
        buildExecutionPlanningPipeline(pm, registry);
      });
  (void)registration;
}

} // namespace tianchenrv::transforms
