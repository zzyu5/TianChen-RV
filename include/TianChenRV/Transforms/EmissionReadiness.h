#ifndef TIANCHENRV_TRANSFORMS_EMISSIONREADINESS_H
#define TIANCHENRV_TRANSFORMS_EMISSIONREADINESS_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::transforms {

llvm::Error checkKernelEmissionPaths(
    tcrv::exec::KernelOp kernel,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error checkKernelEmissionPaths(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error collectKernelEmissionPlans(
    tcrv::exec::KernelOp kernel,
    llvm::SmallVectorImpl<plugin::VariantEmissionPlan> &out,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error collectKernelEmissionPlans(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    llvm::SmallVectorImpl<plugin::VariantEmissionPlan> &out,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error materializeKernelEmissionPlanDiagnostics(
    tcrv::exec::KernelOp kernel,
    const plugin::ExtensionPluginRegistry &registry);

llvm::Error materializeKernelEmissionPlanDiagnostics(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    const plugin::ExtensionPluginRegistry &registry);

} // namespace tianchenrv::transforms

#endif // TIANCHENRV_TRANSFORMS_EMISSIONREADINESS_H
