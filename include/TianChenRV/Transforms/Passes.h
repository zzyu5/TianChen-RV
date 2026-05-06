#ifndef TIANCHENRV_TRANSFORMS_PASSES_H
#define TIANCHENRV_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace tianchenrv {

namespace plugin {
class ExtensionPluginRegistry;
} // namespace plugin

namespace transforms {

std::unique_ptr<::mlir::Pass> createCheckCapabilityRequiresPass();
std::unique_ptr<::mlir::Pass> createSynthesizeVariantDispatchPass();
std::unique_ptr<::mlir::Pass> createSelectVariantsPass();
std::unique_ptr<::mlir::Pass>
createSelectVariantsPass(const plugin::ExtensionPluginRegistry &registry);

#define GEN_PASS_DECL
#include "TianChenRV/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "TianChenRV/Transforms/Passes.h.inc"

} // namespace transforms
} // namespace tianchenrv

#endif // TIANCHENRV_TRANSFORMS_PASSES_H
