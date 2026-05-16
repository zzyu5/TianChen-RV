#ifndef TIANCHENRV_PLUGIN_RVV_RVVSELECTEDBOUNDARYSEED_H
#define TIANCHENRV_PLUGIN_RVV_RVVSELECTEDBOUNDARYSEED_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tianchenrv::plugin::rvv {

std::unique_ptr<::mlir::Pass>
createMaterializeRVVI32M1SelectedBoundarySeedPass();

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVSELECTEDBOUNDARYSEED_H
