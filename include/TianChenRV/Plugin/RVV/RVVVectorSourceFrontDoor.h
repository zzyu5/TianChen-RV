#ifndef TIANCHENRV_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tianchenrv::plugin::rvv {

std::unique_ptr<::mlir::Pass>
createFailClosedRVVLegacyVectorSourceFrontDoorPass();

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorBinarySourceFrontDoorPass();

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorCompareSelectSourceFrontDoorPass();

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H
