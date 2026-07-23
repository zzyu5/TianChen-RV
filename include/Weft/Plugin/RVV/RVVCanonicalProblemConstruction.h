#ifndef WEFT_PLUGIN_RVV_RVVCANONICALPROBLEMCONSTRUCTION_H
#define WEFT_PLUGIN_RVV_RVVCANONICALPROBLEMCONSTRUCTION_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/RVV/RVVSelectedTargetCapability.h"
#include "Weft/Support/CapabilityModel.h"

#include "llvm/Support/Error.h"

#include <string>

namespace weft::plugin::rvv {

/// The bounded RVV source-problem surface whose candidates and final typed
/// bodies are constructed by the RVV owner after target binding. This is an
/// owner-local typed dispatch seam, not a common Formula IR or format switch.
bool isRVVCanonicalProblemSupported(mlir::Operation *problem);

/// Derive deterministic candidate identity from exact P. Source operations,
/// artifact routes, emitters, and measurement state are intentionally absent.
llvm::Expected<std::string>
deriveRVVCanonicalProblemVariantName(mlir::Operation *problem);

/// Validate exact P and selected RVV c_o before selection. The body may still
/// be empty here because final construction belongs to the selected owner.
llvm::Error verifyRVVCanonicalProblemCandidate(
    weft::exec::VariantOp variant, mlir::Operation *problem,
    const support::TargetCapabilitySet &capabilities);

/// Forward-construct the selected typed body from exact P and bound C_d.
llvm::Error constructRVVCanonicalProblemBody(
    weft::exec::VariantOp variant, mlir::Operation *problem,
    const support::TargetCapabilitySet &capabilities);

// Semantic-owner entry points implemented beside the existing typed mechanism
// builders. Every signature excludes source adapters and artifact state.
llvm::Error constructRVVVectorProblemBody(
    weft::exec::VariantOp variant, mlir::Operation *problem,
    const RVVSelectedTargetCapabilityFacts &capability);
llvm::Error constructRVVReductionProblemBody(
    weft::exec::VariantOp variant,
    weft::exec::I8WideningDotReduceProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability);
llvm::Error constructRVVDequantDotProblemBody(
    weft::exec::VariantOp variant,
    weft::exec::I8WideningDotReduceProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability);
llvm::Error constructRVVPackedI4Q8DotProblemBody(
    weft::exec::VariantOp variant, weft::exec::PackedI4Q8DotProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability);
llvm::Error constructRVVCodebookI4Q8DotProblemBody(
    weft::exec::VariantOp variant, weft::exec::CodebookI4Q8DotProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability);
llvm::Error constructRVVQuantizedBlockDotProblemBody(
    weft::exec::VariantOp variant,
    weft::exec::QuantizedBlockDotProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVCANONICALPROBLEMCONSTRUCTION_H
