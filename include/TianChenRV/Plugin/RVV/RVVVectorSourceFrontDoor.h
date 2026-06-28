#ifndef TIANCHENRV_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tianchenrv::plugin {
class SourceFrontDoorPassRegistration;
class ExtensionPluginRegistry;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::rvv {

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorBinarySourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorCompareSelectSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorRuntimeScalarCompareSelectSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVVectorSourceFrontDoorFamilyPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H
