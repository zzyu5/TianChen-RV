#ifndef WEFT_TRANSFORMS_EMISSIONREADINESS_H
#define WEFT_TRANSFORMS_EMISSIONREADINESS_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Support/CapabilityModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace weft::transforms {

llvm::Error checkKernelEmissionPaths(
    weft::exec::KernelOp kernel,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error checkKernelEmissionPaths(
    weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error collectKernelEmissionPlans(
    weft::exec::KernelOp kernel,
    llvm::SmallVectorImpl<plugin::VariantEmissionPlan> &out,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error collectKernelEmissionPlans(
    weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<plugin::VariantEmissionPlan> &out,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error materializeKernelEmissionPlanDiagnostics(
    weft::exec::KernelOp kernel,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error materializeKernelEmissionPlanDiagnostics(
    weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    const plugin::ExtensionPluginRegistry &registry);

} // namespace weft::transforms

#endif // WEFT_TRANSFORMS_EMISSIONREADINESS_H
