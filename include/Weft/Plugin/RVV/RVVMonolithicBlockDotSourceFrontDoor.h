//===- RVVMonolithicBlockDotSourceFrontDoor.h -------------------*- C++ -*-===//
//
// The ONE table-driven monolithic ggml block-dot source front door. It folds the
// 24 former per-op scaffold-constructor passes (RVV{Q40,Q80,Q4K,IQ4NL,...}
// BlockDotSourceFrontDoor.cpp) into a single generic pass IMPLEMENTATION driven by
// `monolithicBlockDotOpTable()` (RVVMonolithicBlockDotFamily.h). Each table row
// carries the per-op construction DATA (marker attr value, pass argument, dispatch
// policy, variant symbol, scale model, block-format i64 facts, codebook/grid/ksigns
// arrays, ABI purposes). One pass class parameterized by a table-row pointer
// registers ONE CLI front-door argument per row -- so every former per-op pass
// argument (`--weft-rvv-materialize-<op>-block-dot-source-front-door`) stays a valid,
// byte-identical front door, but there is now a single construction mechanism.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_PLUGIN_RVV_RVVMONOLITHICBLOCKDOTSOURCEFRONTDOOR_H
#define WEFT_PLUGIN_RVV_RVVMONOLITHICBLOCKDOTSOURCEFRONTDOOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::plugin {
class SourceFrontDoorPassRegistration;
class ExtensionPluginRegistry;
} // namespace weft::plugin

namespace weft::plugin::rvv {

// Register the whole monolithic ggml block-dot family of source front doors -- one
// SourceFrontDoorPassRegistration per `monolithicBlockDotOpTable()` row, all backed
// by the single generic table-driven pass. Replaces the 24 former per-op
// register...BlockDotSourceFrontDoorPasses calls.
llvm::Error registerRVVMonolithicBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin,
    const ::weft::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::weft::plugin::SourceFrontDoorPassRegistration> &out);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVMONOLITHICBLOCKDOTSOURCEFRONTDOOR_H
