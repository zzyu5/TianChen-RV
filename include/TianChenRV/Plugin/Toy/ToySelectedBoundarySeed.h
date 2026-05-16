#ifndef TIANCHENRV_PLUGIN_TOY_TOYSELECTEDBOUNDARYSEED_H
#define TIANCHENRV_PLUGIN_TOY_TOYSELECTEDBOUNDARYSEED_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tianchenrv::plugin::toy {

std::unique_ptr<::mlir::Pass>
createMaterializeToyTemplateSelectedBoundarySeedPass();

} // namespace tianchenrv::plugin::toy

#endif // TIANCHENRV_PLUGIN_TOY_TOYSELECTEDBOUNDARYSEED_H
