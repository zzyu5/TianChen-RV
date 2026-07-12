#ifndef WEFT_PLUGIN_RVV_RVVCOMPOSITEGATHERMACCSCATTERSELECTEDBODYREALIZATIONOWNER_H
#define WEFT_PLUGIN_RVV_RVVCOMPOSITEGATHERMACCSCATTERSELECTEDBODYREALIZATIONOWNER_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"

#include "llvm/Support/Error.h"

namespace weft::plugin::rvv {

bool hasPreRealizedRVVCompositeGatherMAccScatterOwnerCandidate(
    weft::exec::VariantOp variant);

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVCompositeGatherMAccScatterOwner(
    const VariantLoweringBoundaryRequest &request);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVCOMPOSITEGATHERMACCSCATTERSELECTEDBODYREALIZATIONOWNER_H
