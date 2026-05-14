#ifndef TIANCHENRV_PLUGIN_RVV_RVVBINARYSELECTEDLOWERINGBOUNDARY_H
#define TIANCHENRV_PLUGIN_RVV_RVVBINARYSELECTEDLOWERINGBOUNDARY_H

#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace tianchenrv::plugin::rvv {

using RVVBinaryVariantLegalityVerifier =
    llvm::function_ref<llvm::Error(const VariantLegalityRequest &)>;

llvm::Error materializeRVVBinarySelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out, llvm::StringRef originPlugin,
    RVVBinaryVariantLegalityVerifier verifyLegality);

llvm::Error validateRVVBinarySelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request,
    llvm::StringRef originPlugin,
    RVVBinaryVariantLegalityVerifier verifyLegality);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVBINARYSELECTEDLOWERINGBOUNDARY_H
