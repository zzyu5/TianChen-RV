#ifndef TIANCHENRV_PLUGIN_TOY_TOYSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_TOY_TOYSOURCEFRONTDOOR_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tianchenrv::plugin::toy {

std::unique_ptr<::mlir::Pass> createMaterializeToyTemplateSourceFrontDoorPass();

} // namespace tianchenrv::plugin::toy

#endif // TIANCHENRV_PLUGIN_TOY_TOYSOURCEFRONTDOOR_H
