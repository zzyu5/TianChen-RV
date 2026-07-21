//===- RVVDequantizeRowConstruction.h -----------------------------------===//
//
// The ONE byte-exact source of truth for the streaming dequantize_row FRONT-DOOR
// construction: rewrite the abstract weft_rvv.dequantize_row into the typed
// weft_rvv.typed_dequantize_row_loop_body region
//   { dequantize_row_decode_core; typed_dequantize_row_loop_yield }.
//
// Two callers share this ONE construction (so the constructed region -- and hence
// the emitted C -- is byte-identical regardless of WHEN the region is built):
//
//   1. lib/Plugin/RVV/RVVDequantizeRowStreamFrontDoor.cpp -- the PRE-EMITC front
//      door pass. It constructs the region and STOPS (before
//      --weft-rvv-lower-to-emitc) so the realized typed region is walkable by the
//      certification walker (e5_strong_readout.py). NO emit.
//   2. lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp
//      (constructOrEmitGgmlDequantizeRow) -- the IN-EMITC fallback (for callers
//      that feed the abstract op straight to --weft-rvv-lower-to-emitc without the
//      front-door pass). It constructs the SAME region then lowers it immediately.
//
// Only WHEN the region is constructed differs between the two; the region SHAPE
// (op-identity + attributes) and the downstream emit arithmetic
// (emitTypedDequantizeRowLoopBody -> emitDequantizeRow*BodyShared) are identical,
// so both paths emit byte-exact C.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_DIALECT_RVV_IR_RVVDEQUANTIZEROWCONSTRUCTION_H
#define WEFT_DIALECT_RVV_IR_RVVDEQUANTIZEROWCONSTRUCTION_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Support/CodebookGatherPlan.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace weft::rvv {

// The per-format ggml AoS block-layout facts (NOT tunable knobs -- the FIXED ggml
// ABI shape constants) the streaming dequantize_row front door stamps on the
// constructed typed_dequantize_row_loop_body + dequantize_row_decode_core region:
//   qk                = the QK block element count (32 flat / 256 K-quant + IQ grid)
//   weightBlockStride = the AoS weight block byte stride
//   scaleByteOffset   = the fp16/scale byte offset within the block
//   quantByteOffset   = the packed-quant byte offset within the block
//   codebookEntryLanes= the codebook grid ENTRY byte-width (the g-axis geometry:
//                       grid-of-4 uint32 = 4 for iq3_s, grid-of-8 uint64 = 8 for
//                       iq2_xs/iq1_m); 0 = not a grid-codebook owned-vector body (the
//                       flat / K-quant / non-grid leaves do NOT carry an entry width,
//                       so the descriptor is left unstamped for them).
// The flat nibble-family decode carrier: which ALREADY-SEPARATE decode leaf the flat
// streaming family routes to -- the bare signed-int8 q8_0 leaf vs the shared 4-bit
// nibble body ([K-10] selection between two separate leaves, NOT a plan-internal
// mechanism switch). NotNibbleFamily for every K-quant / IQ / codebook / ternary
// format (they are not part of the flat nibble family and leave the descriptor
// unstamped).
enum class NibbleCarrierKind { NotNibbleFamily, BareInt8, Nibble4 };

struct DequantizeRowStreamFacts {
  std::int64_t qk;
  std::int64_t weightBlockStride;
  std::int64_t scaleByteOffset;
  std::int64_t quantByteOffset;
  std::int64_t codebookEntryLanes;
  /// Canonical mechanism-local scale ABI for production small-codebook leaves;
  /// nullopt for every other dequant mechanism.
  std::optional<::weft::CodebookScaleModel> codebookScaleModel;
  // The flat nibble-family decode-mechanism descriptor (phase-1): the carrier leaf
  // selector + the 4-bit nibble decode facts. carrier == NotNibbleFamily leaves ALL of
  // these unstamped (every non-nibble leaf). For the nibble family:
  //   nibbleBias    = the pre-scale bias `sub` subtracted from the nibble (8 for q4_0,
  //                   16 for q5_0, 0 for the min-fold q4_1/q5_1); present for the
  //                   nibble4 carrier only (the bare-int8 q8_0 has no nibble bias).
  //   minByteOffset = the fp16 min m byte offset; its PRESENCE is the hasMin gate
  //                   (q4_1/q5_1). Absent (== no min) otherwise.
  //   qhByteOffset  = the qh 5th-bit plane byte offset; its PRESENCE is the hasQh gate
  //                   (q5_0/q5_1). Absent (== no 5th bit) otherwise.
  NibbleCarrierKind carrier;
  std::optional<std::int64_t> nibbleBias;
  std::optional<std::int64_t> minByteOffset;
  std::optional<std::int64_t> qhByteOffset;
};

// Look up the AoS block-layout facts for one of the 24 CONSTRUCTED streaming
// dequantize_row formats (the flat q8_0 family-head + the q4_0/q4_1/q5_0/q5_1 4-bit
// nibble leaves + the flat 1-bit binary-sign leaf q1_0, the q2_K/q3_K/q4_K/q5_K/q6_K
// QK_K=256 K-quant super-blocks, the iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s IQ grid-table
// super-blocks, the iq1_s/iq1_m/iq4_nl/iq4_xs/mxfp4/nvfp4 codebook/ternary-grid
// extended leaves, and the tq1_0/tq2_0 ternary super-blocks). Returns std::nullopt
// only for an unrecognized format -- it falls through to the dispatch-wired monolith.
// This allowlist MUST mirror the verifier's isConstructedDequantizeRowDecodeModel gate
// (RVVDialectWideningOps.cpp).
std::optional<DequantizeRowStreamFacts>
lookupDequantizeRowStreamFacts(llvm::StringRef format);

// Construct the typed weft_rvv.typed_dequantize_row_loop_body region
//   { dequantize_row_decode_core; typed_dequantize_row_loop_yield }
// in place of `deqOp` (stamped with `facts`), then erase `deqOp`. The caller must
// have confirmed `deqOp.getFormat()` is one of the 24 constructed formats
// (lookupDequantizeRowStreamFacts != nullopt). Takes mlir::RewriterBase so BOTH a
// plain IRRewriter (the pre-emitc pass) and a ConversionPatternRewriter (the
// in-emitc fallback) drive the identical construction.
mlir::LogicalResult
constructTypedDequantizeRowLoopBody(mlir::RewriterBase &rewriter,
                                    GgmlDequantizeRowOp deqOp,
                                    const DequantizeRowStreamFacts &facts);

} // namespace weft::rvv

#endif // WEFT_DIALECT_RVV_IR_RVVDEQUANTIZEROWCONSTRUCTION_H
