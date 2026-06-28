#ifndef TIANCHENRV_PLUGIN_RVV_RVVDEQUANTDOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVDEQUANTDOTSOURCEFRONTDOOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tianchenrv::plugin {
class SourceFrontDoorPassRegistration;
class ExtensionPluginRegistry;
} // namespace tianchenrv::plugin

namespace tianchenrv::plugin::rvv {

// Track B auto-lowering, the DEQUANT rung: the second auto-lowered block, one
// step ABOVE the bare-dot MVP. It matches a GENERIC vector-dialect signed i8
// widening dot-reduce WITH a runtime-f32-scale dequant tail (the MVP tail
// arith.muli/vector.multi_reduction + arith.sitofp + arith.mulf %scale +
// f32 memref.store) and AUTO-CONSTRUCTS the tcrv_rvv
// load/widening_product/standalone_reduce/DEQUANTIZE/store body the unchanged
// EmitC emitter consumes (the existing isLowPrecisionDequantBody sink). The
// integer-core LMUL anchor is the SAME gearbox capability fact the MVP uses
// (enumerateBlockDotShapeCandidates + selectGenericSchedule fed
// deriveMinimumVLEN(march)), so the SAME generic op emits an e8m2/i16m4-form
// body at VLEN128 and an e8m1/i16m2-form body at VLEN256, now with the i32->f32
// dequant fused in. This proves the auto-lowering path scales from bare dot to
// dot+dequant (the q8_0-style integer core + ONE runtime scale), NOT just the
// bare reduce. It is NOT the per-block-fp16-scale q8_0_q8_0 block-dot kernel.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVDequantDotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVDequantDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVDEQUANTDOTSOURCEFRONTDOOR_H
