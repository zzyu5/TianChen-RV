#ifndef WEFT_CONVERSION_RVV_RVVREPACKSCHEDULEMATERIALIZATION_H
#define WEFT_CONVERSION_RVV_RVVREPACKSCHEDULEMATERIALIZATION_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVRepackTilingSelection.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace weft::conversion::rvv {

/// The typed, code-affecting part of the selected repack-GEMM schedule stamp.
/// Attribution records remain mirrors and are deliberately absent from this
/// structure.
struct RVVRepackSchedulePlan {
  std::optional<::weft::plugin::rvv::RVVRepackTilingVariant> tiling;
  std::optional<::weft::plugin::rvv::RVVTilingSelectionReason> tilingReason;
  ::weft::plugin::rvv::RVVRepackLoopOrder loopOrder;
  ::weft::plugin::rvv::RVVTilingSelectionReason loopOrderReason;
};

/// Parse and semantically verify one complete selected schedule plan. Both the
/// module preflight and the emitter call this API, so a future direct emission
/// entry cannot consume an unverified or post-preflight-mutated stamp.
llvm::Expected<RVVRepackSchedulePlan>
readAndVerifyRVVRepackSchedulePlan(
    ::weft::rvv::TypedRepackGemmLoopBodyOp op);

/// The single pre-emission verifier for the A4a schedule slice. It rejects
/// missing/partial/wrong-typed/unknown stamps, unrealizable SP4 values, a prior
/// inconsistent with the typed layout facts, and any unrecognized compatibility
/// reason. It never writes or repairs a stamp.
mlir::LogicalResult verifyRVVRepackSchedulePlans(mlir::ModuleOp module);

} // namespace weft::conversion::rvv

#endif // WEFT_CONVERSION_RVV_RVVREPACKSCHEDULEMATERIALIZATION_H
