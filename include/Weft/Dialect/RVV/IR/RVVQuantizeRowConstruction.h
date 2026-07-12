//===- RVVQuantizeRowConstruction.h -------------------------------------===//
//
// The ONE byte-exact source of truth for the streaming quantize_row FRONT-DOOR
// construction: rewrite the abstract per-format weft_rvv.quantize_row_q8_{0,1,K}
// into the typed weft_rvv.typed_quantize_row_loop_body region
//   { quantize_row_encode_core; typed_quantize_row_loop_yield }.
//
// Two callers share this ONE construction (so the constructed region -- and hence
// the emitted C -- is byte-identical regardless of WHEN the region is built):
//
//   1. lib/Plugin/RVV/RVVQuantizeRowStreamFrontDoor.cpp -- the PRE-EMITC front
//      door pass. It constructs the region and STOPS (before
//      --weft-rvv-lower-to-emitc) so the realized typed region is walkable by the
//      certification walker (e5_strong_readout.py). NO emit.
//   2. lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp
//      (constructQuantizeRowRegionAndLower) -- the IN-EMITC fallback (for callers
//      that feed the abstract op straight to --weft-rvv-lower-to-emitc without the
//      front-door pass). It constructs the SAME region then lowers it immediately.
//
// Only WHEN the region is constructed differs between the two; the region SHAPE
// (op-identity + attributes) and the downstream emit arithmetic
// (emitTypedQuantizeRowLoopBody -> emitQuantizeRowQ8*BodyShared) are identical, so
// both paths emit byte-exact C. The MIRROR of RVVDequantizeRowConstruction (the
// f32->QUANT activation-quantizer family rather than the QUANT->f32 decode family).
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_DIALECT_RVV_IR_RVVQUANTIZEROWCONSTRUCTION_H
#define WEFT_DIALECT_RVV_IR_RVVQUANTIZEROWCONSTRUCTION_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace weft::rvv {

// The per-format ggml AoS block-layout facts (NOT tunable knobs -- the FIXED ggml
// ABI shape constants) the streaming quantize_row front door stamps on the
// constructed typed_quantize_row_loop_body + quantize_row_encode_core region:
//   qk                = the QK block element count (32 flat q8_0/q8_1 / 256 q8_K)
//   blockStride       = the AoS block_qX byte stride
//   scaleByteOffset   = the fp16/float scale byte offset within the block
//   quantByteOffset   = the packed-quant (int8 qs) byte offset within the block
// The remaining per-format offsets (the q8_1 fp16 block sum `s`, the q8_K per-16
// int16 bsums) are baked into the per-format encode leaf at emit, so the brick
// carries only the two byte offsets the shared attr surface names.
struct QuantizeRowStreamFacts {
  std::int64_t qk;
  std::int64_t blockStride;
  std::int64_t scaleByteOffset;
  std::int64_t quantByteOffset;
};

// Look up the AoS block-layout facts for one of the 3 CONSTRUCTED streaming
// quantize_row encode_models (the q8_0 block_q8_0 bare fp16-scale int8-narrow
// family-head, the q8_1 SIBLING + integer block sum, the q8_K QK_K=256 min/max-
// symmetric K-quant activation quantizer). Returns std::nullopt for any other
// encode_model -- the front door does NOT construct it. This allowlist MUST mirror
// the verifier's isConstructedQuantizeRowEncodeModel gate (RVVDialectWideningOps.cpp).
std::optional<QuantizeRowStreamFacts>
lookupQuantizeRowStreamFacts(llvm::StringRef encodeModel);

// Construct the typed weft_rvv.typed_quantize_row_loop_body region
//   { quantize_row_encode_core; typed_quantize_row_loop_yield }
// in place of `quantOp` (stamped with `encodeModel` + `facts`), then erase
// `quantOp`. `input`/`output`/`n` are the abstract op's f32 input base pointer,
// block_qX output byte buffer, and runtime element count. The caller must have
// confirmed `encodeModel` is one of the 3 constructed formats
// (lookupQuantizeRowStreamFacts != nullopt). Takes mlir::RewriterBase so BOTH a
// plain IRRewriter (the pre-emitc pass) and a ConversionPatternRewriter (the
// in-emitc fallback) drive the identical construction. Generic over `quantOp`'s
// concrete type (the 3 quantize ops share no common typed base beyond Operation).
mlir::LogicalResult
constructTypedQuantizeRowLoopBody(mlir::RewriterBase &rewriter,
                                  mlir::Operation *quantOp, mlir::Value input,
                                  mlir::Value output, mlir::Value n,
                                  llvm::StringRef encodeModel,
                                  const QuantizeRowStreamFacts &facts);

} // namespace weft::rvv

#endif // WEFT_DIALECT_RVV_IR_RVVQUANTIZEROWCONSTRUCTION_H
