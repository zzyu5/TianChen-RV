#ifndef TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITESOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITESOURCEFRONTDOOR_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tianchenrv::plugin::tensorext_lite {

std::unique_ptr<::mlir::Pass>
createMaterializeTensorExtLiteFragmentMmaSourceFrontDoorPass();

} // namespace tianchenrv::plugin::tensorext_lite

#endif // TIANCHENRV_PLUGIN_TENSOREXTLITE_TENSOREXTLITESOURCEFRONTDOOR_H
