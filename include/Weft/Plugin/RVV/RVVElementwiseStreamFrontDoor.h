//===- RVVElementwiseStreamFrontDoor.h ----------------------------------===//
//
// The CERT-FD forward殿后族 PRE-EMITC front door (the forward sibling of
// RVVDequantizeRowStreamFrontDoor / RVVQuantizeRowStreamFrontDoor). It CONSTRUCTS
// the typed weft_rvv.typed_elementwise_loop_body region
//   { <per-model map/reduce/rotate core brick>; typed_elementwise_loop_yield }
// in place of each abstract weft_rvv.ggml_forward_elementwise whose model is one of
// the 5 CONSTRUCTED forward operators (scale / silu / rms_norm / soft_max / rope),
// and STOPS -- BEFORE --weft-rvv-lower-to-emitc.
//
// WHY it exists: the forward-elementwise operators are REPRESENTED as decomposed
// typed regions (no opaque monolith), but the certification walker needs a REAL
// front-door CONSTRUCTION to walk (abstract source op -> typed region), consistent
// with the dequant/quant families. This pass exposes that construction as an
// explicit pre-emitc step so the walker can walk (and hence machine-CERTIFY) the
// constructed region. The construction is the SHARED byte-exact
// weft::rvv::constructTypedElementwiseLoopBody, so the emitted C is byte-identical
// whether the region is built here (pre-emitc) or authored directly. Numerical
// semantics: zero change.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_PLUGIN_RVV_RVVELEMENTWISESTREAMFRONTDOOR_H
#define WEFT_PLUGIN_RVV_RVVELEMENTWISESTREAMFRONTDOOR_H

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

std::unique_ptr<::mlir::Pass>
createMaterializeRVVElementwiseStreamFrontDoorPass();

llvm::Error registerRVVElementwiseStreamFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::weft::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::weft::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVELEMENTWISESTREAMFRONTDOOR_H
