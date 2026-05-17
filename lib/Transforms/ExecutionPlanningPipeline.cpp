#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace tianchenrv::transforms {
namespace {

constexpr llvm::StringLiteral kExecutionPlanningPipelineName(
    "tcrv-execution-planning-pipeline");
constexpr llvm::StringLiteral kExecutionPlanningPipelineDescription(
    "Compose TianChen-RV plugin variant materialization, capability "
    "checking, generic selection/dispatch planning, selected lowering-boundary "
    "materialization, emission-plan diagnostics, and execution-plan coherence "
    "checking");
constexpr llvm::StringLiteral kSourceArtifactFrontDoorPipelineName(
    "tcrv-source-artifact-front-door-pipeline");
constexpr llvm::StringLiteral kSourceArtifactFrontDoorPipelineDescription(
    "Compose enabled plugin source front-door materialization passes with "
    "TianChen-RV "
    "generic legality, capability, emission-plan, and execution-plan coherence "
    "checks so bounded source inputs reach selected emission diagnostics "
    "before any supported target artifact export");

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
  pm.addPass(createCheckHartParallelCapabilitiesPass());
  pm.addPass(createVerifyPluginVariantLegalityPass(registry));
  pm.addPass(createSelectVariantsPass(registry));
  pm.addPass(createMaterializeDispatchRuntimeGuardsPass());
  pm.addPass(createCheckCapabilityRequiresPass());
  pm.addPass(createMaterializeSelectedLoweringBoundariesPass(registry));
  pm.addPass(createMaterializeEmissionPlansPass(registry));
  pm.addPass(createCheckExecutionPlanCoherencePass(registry, targetExporters));
}

void buildSourceArtifactFrontDoorPipeline(
    mlir::OpPassManager &pm,
    llvm::ArrayRef<plugin::SourceFrontDoorPassRegistration> sourceFrontDoorPasses,
    const plugin::ExtensionPluginRegistry &registry,
    const target::TargetArtifactExporterRegistry &targetExporters) {
  for (const plugin::SourceFrontDoorPassRegistration &sourceFrontDoorPass :
       sourceFrontDoorPasses)
    pm.addPass(sourceFrontDoorPass.getFactory()());

  pm.addPass(createCheckHartParallelCapabilitiesPass());
  pm.addPass(createVerifyPluginVariantLegalityPass(registry));
  pm.addPass(createCheckCapabilityRequiresPass());
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

void registerSourceArtifactFrontDoorPipeline(
    llvm::ArrayRef<plugin::SourceFrontDoorPassRegistration>
        sourceFrontDoorPasses,
    const plugin::ExtensionPluginRegistry &registry,
    const target::TargetArtifactExporterRegistry &targetExporters) {
  llvm::SmallVector<plugin::SourceFrontDoorPassRegistration, 4>
      capturedPasses(sourceFrontDoorPasses.begin(), sourceFrontDoorPasses.end());
  mlir::PassPipelineRegistration<> registration(
      kSourceArtifactFrontDoorPipelineName,
      kSourceArtifactFrontDoorPipelineDescription,
      [capturedSourceFrontDoorPasses = std::move(capturedPasses),
       &registry, &targetExporters](mlir::OpPassManager &pm) {
        buildSourceArtifactFrontDoorPipeline(
            pm, capturedSourceFrontDoorPasses, registry, targetExporters);
      });
  (void)registration;
}

} // namespace tianchenrv::transforms
