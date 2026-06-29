#ifndef TIANCHENRV_PLUGIN_RVV_RVVPACKEDI4DOTSOURCEFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVPACKEDI4DOTSOURCEFRONTDOOR_H

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

// Track B G1 (BOUNDED first step): the COMPILER auto-CONSTRUCTS the q4_0 nibble
// INTEGER-CORE body -- the single-strip generic-op composition
//   load x3 (packed-i4 weight + plain-i8 q8 low/high)
//     -> tcrv_rvv.packed_i4_offset_binary_x_i8_product (the offset-binary nibble
//        decode + asymmetric widening product, ALREADY a first-class generic op)
//     -> tcrv_rvv.standalone_reduce (signed widening reduce)
//     -> tcrv_rvv.store
// from a marked GENERIC source carrying the nibble-core operator identity, instead
// of routing a monolithic op to a per-kernel hand emitter. This proves
// auto-construction REACHES nibble-decode through the generic-op mechanism.
//
// SCOPE (honest): this is the nibble integer CORE ONLY. It does NOT auto-construct
// the full ggml q4_0 x q8_0 KERNEL -- no nb = n / QK outer block loop, no per-block
// dual fp16 scale read, no left-associative fp32 fold. Those three axes (which need
// NEW generic ODS vocabulary) are full G1 and are DEFERRED. The monolithic
// tcrv_rvv.q4_0_q8_0_block_dot op, its KERNEL front door, and the hand emitter all
// STAY (this adds a SEPARATE rung-3 front door, like dequant-vs-reduction).
//
// CAPABILITY framing (no flip claimed). The capability consultation is the SAME
// shared block-dot schedule authority the rung-1/2 front doors use
// (selectIntegerCoreLMUL: enumerateBlockDotShapeCandidates + selectGenericSchedule
// fed deriveMinimumVLEN(march)), run here as the LEGALITY GATE (fail-closed if the
// integer-core path is pruned), NOT as the nibble anchor source. q4_0's nibble
// half-block integer core is pinned at i8mf4-i16mf2-i32m1 at every Zvl128b tier --
// there is NO VLEN128-vs-VLEN256 byte-flip here (the documented q4_0 no-flip form;
// see RVVQ40BlockDotSourceFrontDoor.cpp + the gearbox `signed-i4n2-in-i8mf4` packed-
// i4 candidate). The bounded step pins this no-flip mf4 isolated-core form, byte-
// identical to the existing nibble-core lit, and DEFERS any flip claim.
std::unique_ptr<::mlir::Pass>
createMaterializeRVVPackedI4DotSourceFrontDoorPass(
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVPackedI4DotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVPACKEDI4DOTSOURCEFRONTDOOR_H
