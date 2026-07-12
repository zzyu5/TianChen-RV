#ifndef WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITESOURCEFRONTDOOR_H
#define WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITESOURCEFRONTDOOR_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace weft::plugin::tensorext_lite {

std::unique_ptr<::mlir::Pass>
createMaterializeTensorExtLiteFragmentMmaSourceFrontDoorPass();

} // namespace weft::plugin::tensorext_lite

#endif // WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITESOURCEFRONTDOOR_H
