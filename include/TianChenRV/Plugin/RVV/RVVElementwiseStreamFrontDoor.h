//===- RVVElementwiseStreamFrontDoor.h ----------------------------------===//
//
// The CERT-FD forward殿后族 PRE-EMITC front door (the forward sibling of
// RVVDequantizeRowStreamFrontDoor / RVVQuantizeRowStreamFrontDoor). It CONSTRUCTS
// the typed tcrv_rvv.typed_elementwise_loop_body region
//   { <per-model map/reduce/rotate core brick>; typed_elementwise_loop_yield }
// in place of each abstract tcrv_rvv.ggml_forward_elementwise whose model is one of
// the 5 CONSTRUCTED forward operators (scale / silu / rms_norm / soft_max / rope),
// and STOPS -- BEFORE --tcrv-rvv-lower-to-emitc.
//
// WHY it exists: the forward-elementwise operators are REPRESENTED as decomposed
// typed regions (no opaque monolith), but the certification walker needs a REAL
// front-door CONSTRUCTION to walk (abstract source op -> typed region), consistent
// with the dequant/quant families. This pass exposes that construction as an
// explicit pre-emitc step so the walker can walk (and hence machine-CERTIFY) the
// constructed region. The construction is the SHARED byte-exact
// tcrv::rvv::constructTypedElementwiseLoopBody, so the emitted C is byte-identical
// whether the region is built here (pre-emitc) or authored directly. Numerical
// semantics: zero change.
//
//===----------------------------------------------------------------------===//

#ifndef TIANCHENRV_PLUGIN_RVV_RVVELEMENTWISESTREAMFRONTDOOR_H
#define TIANCHENRV_PLUGIN_RVV_RVVELEMENTWISESTREAMFRONTDOOR_H

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

std::unique_ptr<::mlir::Pass>
createMaterializeRVVElementwiseStreamFrontDoorPass();

llvm::Error registerRVVElementwiseStreamFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::tianchenrv::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::tianchenrv::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVELEMENTWISESTREAMFRONTDOOR_H
