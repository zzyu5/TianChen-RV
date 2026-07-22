//===- RVVQuantizeRowConstruction.h -------------------------------------===//
//
// The ONE byte-exact source of truth for the streaming quantize_row FRONT-DOOR
// construction: rewrite the abstract per-format weft_rvv.quantize_row_q8_{0,1,K}
// into the typed weft_rvv.typed_quantize_row_loop_body region
//   { quantize_row_encode_core; typed_quantize_row_loop_yield }.
//
// This helper only realizes an already-constructed typed formula result.  The
// project-wide pre-emission formula cut and the optional inspection front door both
// call it; the emitter cannot call it and has no abstract-op fallback.  The MIRROR
// of RVVDequantizeRowConstruction (the f32->QUANT activation-quantizer family rather
// than the QUANT->f32 decode family).
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_DIALECT_RVV_IR_RVVQUANTIZEROWCONSTRUCTION_H
#define WEFT_DIALECT_RVV_IR_RVVQUANTIZEROWCONSTRUCTION_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
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
  QuantizeRowLeaf leaf;
  llvm::StringRef encodeModel;
  std::int64_t qk;
  std::int64_t blockStride;
  std::int64_t scaleByteOffset;
  std::int64_t quantByteOffset;
};

// Construct the typed weft_rvv.typed_quantize_row_loop_body region
//   { quantize_row_encode_core; typed_quantize_row_loop_yield }
// in place of `quantOp` (stamped with the formula-produced typed leaf, provenance
// model and layout facts), then erase
// `quantOp`. `input`/`output`/`n` are the abstract op's f32 input base pointer,
// block_qX output byte buffer, and runtime element count. Takes mlir::RewriterBase
// so both the project-wide materializer and the optional inspection pass realize
// the identical typed body. Generic over `quantOp`'s concrete type (the 3 quantize
// ops share no common typed base beyond Operation).
mlir::LogicalResult
constructTypedQuantizeRowLoopBody(mlir::RewriterBase &rewriter,
                                  mlir::Operation *quantOp, mlir::Value input,
                                  mlir::Value output, mlir::Value n,
                                  const QuantizeRowStreamFacts &facts);

} // namespace weft::rvv

#endif // WEFT_DIALECT_RVV_IR_RVVQUANTIZEROWCONSTRUCTION_H
