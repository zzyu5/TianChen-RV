#ifndef TIANCHENRV_TRANSFORMS_PASSES_H
#define TIANCHENRV_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace tianchenrv::transforms {

std::unique_ptr<::mlir::Pass> createCheckCapabilityRequiresPass();

#define GEN_PASS_DECL
#include "TianChenRV/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "TianChenRV/Transforms/Passes.h.inc"

} // namespace tianchenrv::transforms

#endif // TIANCHENRV_TRANSFORMS_PASSES_H
