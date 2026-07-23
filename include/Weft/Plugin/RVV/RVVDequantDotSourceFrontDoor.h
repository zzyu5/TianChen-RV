#ifndef WEFT_PLUGIN_RVV_RVVDEQUANTDOTSOURCEFRONTDOOR_H
#define WEFT_PLUGIN_RVV_RVVDEQUANTDOTSOURCEFRONTDOOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace weft::plugin {
class SourceFrontDoorPassRegistration;
class ExtensionPluginRegistry;
} // namespace weft::plugin

namespace weft::plugin::rvv {

// Track B auto-lowering, the DEQUANT rung: the second auto-lowered block, one
// step ABOVE the bare-dot MVP. It matches a GENERIC vector-dialect signed i8
// widening dot-reduce WITH a runtime-f32-scale dequant tail (the MVP tail
// arith.muli/vector.multi_reduction + arith.sitofp + arith.mulf %scale +
// f32 memref.store) and AUTO-CONSTRUCTS the weft_rvv
// load/widening_product/standalone_reduce/DEQUANTIZE/store body the unchanged
// EmitC emitter consumes (the existing isLowPrecisionDequantBody sink). The
// integer-core LMUL anchor comes from RVVIntegerCoreScheduleFormula over the same
// typed source/capability facts as the MVP, so the SAME generic op emits an e8m2/i16m4-form
// body at VLEN128 and an e8m1/i16m2-form body at VLEN256, now with the i32->f32
// dequant fused in. This proves the auto-lowering path scales from bare dot to
// dot+dequant (the q8_0-style integer core + ONE runtime scale), NOT just the
// bare reduce. It is NOT the per-block-fp16-scale q8_0_q8_0 block-dot kernel.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVDequantDotSourceFrontDoorPass(
    const ::weft::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVDequantDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::weft::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::weft::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVDEQUANTDOTSOURCEFRONTDOOR_H
