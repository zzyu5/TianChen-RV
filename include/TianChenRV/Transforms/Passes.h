#ifndef TIANCHENRV_TRANSFORMS_PASSES_H
#define TIANCHENRV_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"

#include <memory>

namespace tianchenrv {

namespace plugin {
class ExtensionPluginRegistry;
class SourceFrontDoorPassRegistration;
} // namespace plugin

namespace target {
class TargetArtifactExporterRegistry;
} // namespace target

namespace transforms {

void buildExecutionPlanningPipeline(::mlir::OpPassManager &pm);
void buildExecutionPlanningPipeline(
    ::mlir::OpPassManager &pm,
    const plugin::ExtensionPluginRegistry &registry);
void buildExecutionPlanningPipeline(
    ::mlir::OpPassManager &pm,
    const plugin::ExtensionPluginRegistry &registry,
    const target::TargetArtifactExporterRegistry &targetExporters);
void buildSourceArtifactFrontDoorPipeline(
    ::mlir::OpPassManager &pm,
    llvm::ArrayRef<plugin::SourceFrontDoorPassRegistration>
        sourceFrontDoorPasses,
    const plugin::ExtensionPluginRegistry &registry,
    const target::TargetArtifactExporterRegistry &targetExporters);
void registerExecutionPlanningPipeline();
void registerExecutionPlanningPipeline(
    const plugin::ExtensionPluginRegistry &registry);
void registerExecutionPlanningPipeline(
    const plugin::ExtensionPluginRegistry &registry,
    const target::TargetArtifactExporterRegistry &targetExporters);
void registerSourceArtifactFrontDoorPipeline(
    llvm::ArrayRef<plugin::SourceFrontDoorPassRegistration>
        sourceFrontDoorPasses,
    const plugin::ExtensionPluginRegistry &registry,
    const target::TargetArtifactExporterRegistry &targetExporters);

std::unique_ptr<::mlir::Pass> createCheckCapabilityRequiresPass();
std::unique_ptr<::mlir::Pass> createCheckHartParallelCapabilitiesPass();
std::unique_ptr<::mlir::Pass> createSynthesizeVariantDispatchPass();
std::unique_ptr<::mlir::Pass> createMaterializePluginVariantsPass();
std::unique_ptr<::mlir::Pass> createMaterializePluginVariantsPass(
    const plugin::ExtensionPluginRegistry &registry);
std::unique_ptr<::mlir::Pass> createVerifyPluginVariantLegalityPass();
std::unique_ptr<::mlir::Pass> createVerifyPluginVariantLegalityPass(
    const plugin::ExtensionPluginRegistry &registry);
std::unique_ptr<::mlir::Pass> createSelectVariantsPass();
std::unique_ptr<::mlir::Pass>
createSelectVariantsPass(const plugin::ExtensionPluginRegistry &registry);
std::unique_ptr<::mlir::Pass> createMaterializeDispatchRuntimeGuardsPass();
std::unique_ptr<::mlir::Pass> createCheckEmissionPathsPass();
std::unique_ptr<::mlir::Pass>
createCheckEmissionPathsPass(const plugin::ExtensionPluginRegistry &registry);
std::unique_ptr<::mlir::Pass> createMaterializeEmissionPlansPass();
std::unique_ptr<::mlir::Pass> createMaterializeEmissionPlansPass(
    const plugin::ExtensionPluginRegistry &registry);
std::unique_ptr<::mlir::Pass> createMaterializeSelectedLoweringBoundariesPass();
std::unique_ptr<::mlir::Pass>
createMaterializeSelectedLoweringBoundariesPass(
    const plugin::ExtensionPluginRegistry &registry);
std::unique_ptr<::mlir::Pass> createMaterializeRVVProbedCapabilityAxesPass();
std::unique_ptr<::mlir::Pass> createMaterializeRVVGearboxSchedulesPass();
std::unique_ptr<::mlir::Pass> createMaterializeEmitCLowerableRoutesPass();
std::unique_ptr<::mlir::Pass> createMaterializeEmitCLowerableRoutesPass(
    const plugin::ExtensionPluginRegistry &registry);
std::unique_ptr<::mlir::Pass> createRVVLowerToEmitCPass();
std::unique_ptr<::mlir::Pass> createCheckExecutionPlanCoherencePass();
std::unique_ptr<::mlir::Pass> createCheckExecutionPlanCoherencePass(
    const plugin::ExtensionPluginRegistry &plugins,
    const target::TargetArtifactExporterRegistry &targetExporters);

#define GEN_PASS_DECL
#include "TianChenRV/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "TianChenRV/Transforms/Passes.h.inc"

} // namespace transforms
} // namespace tianchenrv

#endif // TIANCHENRV_TRANSFORMS_PASSES_H
