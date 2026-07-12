//===- RVVElementwiseStreamConstruction.h -------------------------------===//
//
// The ONE byte-exact source of truth for the streaming forward-elementwise
// FRONT-DOOR construction: rewrite the abstract weft_rvv.ggml_forward_elementwise
// into the typed weft_rvv.typed_elementwise_loop_body region
//   { <per-model map/reduce/rotate core brick>; typed_elementwise_loop_yield }.
//
// The forward sibling of RVVDequantizeRowConstruction.h. Unlike dequant (a uniform
// (input,output,n) ABI + a per-format decode leaf), the 5 forward operators split
// into TWO region SHAPES driven by `elementwise_model`:
//   MAP    (scale/silu)        -> reduce_map_model "map";    block arg (index);
//                                 yield names NO operand.
//   REDUCE (rms_norm/soft_max) -> reduce_map_model "reduce"; block args
//                                 (index, acc:{f64 | vector<f64,"m1">}); yield the
//                                 updated accumulator.
//   ROTATE (rope)              -> reduce_map_model "rotate"; block args
//                                 (index, theta:f32); yield the stepped theta.
//
// Only WHEN the region is constructed differs (the pre-emitc front door vs a future
// in-emitc fallback); the region SHAPE (op-identity + attrs + operands) and the
// downstream emit arithmetic (emitTypedElementwiseLoopBody) are identical, so both
// paths emit byte-exact C. Numerical semantics: zero change.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_DIALECT_RVV_IR_RVVELEMENTWISESTREAMCONSTRUCTION_H
#define WEFT_DIALECT_RVV_IR_RVVELEMENTWISESTREAMCONSTRUCTION_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace weft::rvv {

// The per-model construction facts (NOT tunable knobs -- the FIXED forward-op
// SHAPE constants) the streaming front door stamps on the constructed
// typed_elementwise_loop_body + <core brick> region:
//   reduceMapModel = the loop shape ("map" | "reduce" | "rotate")
//   arity          = the abi_operands count (n last)
struct ForwardElementwiseFacts {
  llvm::StringRef reduceMapModel;
  unsigned arity;
};

// Look up the SHAPE facts for one of the 5 CONSTRUCTED forward-elementwise
// operators (scale / silu = MAP, rms_norm / soft_max = REDUCE, rope = ROTATE).
// Returns std::nullopt for any unknown model -- the front door leaves such ops
// abstract (fail-open skip, mirroring the dequant tq1_0/tq2_0 skip). This allowlist
// MUST mirror the GgmlForwardElementwiseOp verifier's model gate.
std::optional<ForwardElementwiseFacts>
lookupForwardElementwiseFacts(llvm::StringRef model);

// Construct the typed weft_rvv.typed_elementwise_loop_body region
//   { <elementwise_{scale,silu}_map | elementwise_{rms_norm,soft_max}_reduce_core |
//      elementwise_rope_rotate_core>; typed_elementwise_loop_yield }
// in place of `fwdOp` (driven by `facts`), then erase `fwdOp`. The caller must have
// confirmed `fwdOp.getElementwiseModel()` is one of the 5 constructed models
// (lookupForwardElementwiseFacts != nullopt). Takes mlir::RewriterBase so both a
// plain IRRewriter (the pre-emitc pass) and a ConversionPatternRewriter drive the
// identical construction.
mlir::LogicalResult
constructTypedElementwiseLoopBody(mlir::RewriterBase &rewriter,
                                  GgmlForwardElementwiseOp fwdOp,
                                  const ForwardElementwiseFacts &facts);

} // namespace weft::rvv

#endif // WEFT_DIALECT_RVV_IR_RVVELEMENTWISESTREAMCONSTRUCTION_H
