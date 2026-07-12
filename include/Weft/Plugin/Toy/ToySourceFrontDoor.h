#ifndef WEFT_PLUGIN_TOY_TOYSOURCEFRONTDOOR_H
#define WEFT_PLUGIN_TOY_TOYSOURCEFRONTDOOR_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace weft::plugin::toy {

std::unique_ptr<::mlir::Pass> createMaterializeToyTemplateSourceFrontDoorPass();

} // namespace weft::plugin::toy

#endif // WEFT_PLUGIN_TOY_TOYSOURCEFRONTDOOR_H
