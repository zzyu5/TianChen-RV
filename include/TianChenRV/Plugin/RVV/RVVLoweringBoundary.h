#ifndef TIANCHENRV_PLUGIN_RVV_RVVLOWERINGBOUNDARY_H
#define TIANCHENRV_PLUGIN_RVV_RVVLOWERINGBOUNDARY_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "llvm/Support/Error.h"

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tianchenrv::plugin::rvv {

llvm::Error materializeRVVLoweringBoundaries(
    tcrv::exec::KernelOp kernel, const ExtensionPluginRegistry &registry);

llvm::Error materializeRVVLoweringBoundaries(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry);

std::unique_ptr<::mlir::Pass> createMaterializeRVVLoweringBoundaryPass();
std::unique_ptr<::mlir::Pass> createMaterializeRVVLoweringBoundaryPass(
    const ExtensionPluginRegistry &registry);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVLOWERINGBOUNDARY_H
