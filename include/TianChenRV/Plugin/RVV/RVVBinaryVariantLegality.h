#ifndef TIANCHENRV_PLUGIN_RVV_RVVBINARYVARIANTLEGALITY_H
#define TIANCHENRV_PLUGIN_RVV_RVVBINARYVARIANTLEGALITY_H

#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {

llvm::Error verifyRVVBinaryVariantLegality(
    const VariantLegalityRequest &request, llvm::StringRef originPlugin);

llvm::Error verifyRVVBinarySmokeProbeVariantMetadata(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVBINARYVARIANTLEGALITY_H
