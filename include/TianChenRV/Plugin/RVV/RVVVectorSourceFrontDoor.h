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
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::rvv {

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorBinarySourceFrontDoorPass();

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorCompareSelectSourceFrontDoorPass();

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorRuntimeScalarCompareSelectSourceFrontDoorPass();

llvm::Error registerRVVVectorSourceFrontDoorFamilyPasses(
    llvm::StringRef ownerPlugin,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H
