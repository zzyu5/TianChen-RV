//===- RVVDequantizeRowConstruction.cpp ---------------------------------===//
//
// The ONE byte-exact construction of the streaming dequantize_row FRONT-DOOR
// typed region, shared by the pre-emitc RVVDequantizeRowStreamFrontDoor pass and
// the in-emitc constructOrEmitGgmlDequantizeRow fallback. See the header for the
// design split (only WHEN the region is built differs; the region SHAPE and the
// downstream emit arithmetic are identical, so both paths emit byte-exact C).
//
//===----------------------------------------------------------------------===//

#include "Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"

namespace weft::rvv {

std::optional<DequantizeRowStreamFacts>
lookupDequantizeRowStreamFacts(llvm::StringRef format) {
  // The streaming CONSTRUCTED family + its per-format AoS block-layout facts (the
  // ggml ABI shape constants, NOT tunable knobs): qk (32 flat / 256 K-quant + IQ
  // grid), the block stride, the fp16 scale byte offset (scale_byte_offset), and
  // the packed-quant byte offset (quant_byte_offset). The remaining per-format
  // offsets (min/qh for the flat leaves; dmin/scale-plane/qh for K-quant; grid-
  // index/sign planes for the IQ grid-table super-blocks) are baked into the per-
  // format decode leaf at emit, so the brick carries only the two byte offsets the
  // shared q8_0-shaped attr surface names.
  //
  // codebook_entry_lanes (the g-axis grid geometry) is the ONE exception the owned
  // narrow-per-entry grid dequant bodies need at emit but must NOT bake into the
  // mechanism body (律2): it is the codebook grid ENTRY byte-width, stamped ONLY for
  // the three owned grid-codebook decode leaves (iq3_s grid-of-4 uint32 = 4;
  // iq2_xs / iq1_m grid-of-8 uint64 = 8). It stays 0 (unstamped) for every flat /
  // K-quant / non-grid leaf, which never reads an entry width.
  std::int64_t qk = 32, stride = 0, dOff = 0, qsOff = 0, entryLanes = 0;
  // Phase-1 nibble-family decode-mechanism descriptor (unset == NotNibbleFamily /
  // absent for every non-nibble leaf; the flat nibble arms below set them).
  NibbleCarrierKind carrier = NibbleCarrierKind::NotNibbleFamily;
  std::optional<std::int64_t> nibbleBias, minOff, qhOff;
  if (format == "q8_0") {
    stride = 34; qsOff = 2; carrier = NibbleCarrierKind::BareInt8;
  } else if (format == "q4_0") {
    stride = 18; qsOff = 2; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 8;
  } else if (format == "q4_1") {
    stride = 20; qsOff = 4; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 0;
    minOff = 2;
  } else if (format == "q5_0") {
    stride = 22; qsOff = 6; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 16;
    qhOff = 2;
  } else if (format == "q5_1") {
    stride = 24; qsOff = 8; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 0;
    minOff = 2; qhOff = 4;
  } else if (format == "q4_synth") {
    // F1 FALSIFIER (phase-1 C1/C2): a synthetic 4-bit nibble dequant leaf added by THIS
    // SINGLE descriptor row (+ its lit) -- ZERO emitter / verifier mechanism lines. It
    // carries q4_0's EXACT decode tuple (stride 18, quant_byte_offset 2, nibble_bias 8,
    // carrier nibble4, no min / no qh), so its emitted C is BYTE-IDENTICAL to
    // dequantize_row_q4_0 modulo only the provenance token: the constructed typed region
    // dispatches on carrier_kind == "nibble4" (NOT the format name) into the shared
    // nibble body. Directly falsifies "the format name drives dispatch".
    stride = 18; qsOff = 2; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 8;
  } else if (format == "q1_0") {
    // block_q1_0: fp16 d @0, qs[16] @2 (QK1_0=128 packed 1-bit binary {-1,+1}
    // signs, 8 weights/byte). The flat binary-sign leaf: y[j] = bit ? d : -d.
    qk = 128; stride = 18; dOff = 0; qsOff = 2;
  } else if (format == "q2_K") {
    qk = 256; stride = 84; dOff = 80; qsOff = 16;
  } else if (format == "q3_K") {
    qk = 256; stride = 110; dOff = 108; qsOff = 32;
  } else if (format == "q4_K") {
    qk = 256; stride = 144; dOff = 0; qsOff = 16;
  } else if (format == "q5_K") {
    qk = 256; stride = 176; dOff = 0; qsOff = 48;
  } else if (format == "q6_K") {
    qk = 256; stride = 210; dOff = 208; qsOff = 0;
  } else if (format == "iq2_xxs") {
    qk = 256; stride = 66; dOff = 0; qsOff = 2;
  } else if (format == "iq2_xs") {
    // grid-of-8 (int64, 512-entry): each grid entry = 8 contiguous grid bytes.
    qk = 256; stride = 74; dOff = 0; qsOff = 2; entryLanes = 8;
  } else if (format == "iq2_s") {
    qk = 256; stride = 82; dOff = 0; qsOff = 2;
  } else if (format == "iq3_xxs") {
    qk = 256; stride = 98; dOff = 0; qsOff = 2;
  } else if (format == "iq3_s") {
    // grid-of-4 (uint32, EXPLICIT signs): each grid entry = 4 contiguous grid bytes.
    qk = 256; stride = 110; dOff = 0; qsOff = 2; entryLanes = 4;
  } else if (format == "iq1_s") {
    // block_iq1_s: fp16 d @0, qs[32] @2, qh[8] u16 @34 (ternary iq1s_grid + delta).
    qk = 256; stride = 50; dOff = 0; qsOff = 2;
  } else if (format == "iq1_m") {
    // block_iq1_m: NO fp16 d -- qs[32] LEAD the block @0, qh[16] @32, packed scale
    // words @48 (the super-block d is the reconstructed iq1m_scale fp16). grid-of-8
    // (2048-entry iq1s_grid): each grid entry = 8 contiguous ternary grid bytes.
    qk = 256; stride = 56; dOff = 0; qsOff = 0; entryLanes = 8;
  } else if (format == "iq4_nl") {
    // block_iq4_nl: fp16 d @0, qs[16] @2 (flat QK4_NL=32 non-linear codebook).
    qk = 32; stride = 18; dOff = 0; qsOff = 2;
  } else if (format == "iq4_xs") {
    // block_iq4_xs: fp16 d @0, scales_h u16 @2, scales_l[4] @4, qs[128] @8.
    qk = 256; stride = 136; dOff = 0; qsOff = 8;
  } else if (format == "mxfp4") {
    // block_mxfp4: E8M0 exponent byte @0, qs[16] @1 (QK_MXFP4=32 FP4 codebook).
    qk = 32; stride = 17; dOff = 0; qsOff = 1;
  } else if (format == "nvfp4") {
    // block_nvfp4: four UE4M3 sub-block scale bytes @0, qs[32] @4 (QK_NVFP4=64).
    qk = 64; stride = 36; dOff = 0; qsOff = 4;
  } else if (format == "tq1_0") {
    // block_tq1_0: qs[48] @0 (base-3 packed, 5 elems/byte), qh[4] @48, fp16 d @52
    // (the ternary {-1,0,+1} TriLM super-block; scale is at the END, not @0).
    qk = 256; stride = 54; dOff = 52; qsOff = 0;
  } else if (format == "tq2_0") {
    // block_tq2_0: qs[64] @0 (2-bit packed, 4 elems/byte), fp16 d @64 (the 2-bit
    // ternary super-block; scale is at the END, not @0).
    qk = 256; stride = 66; dOff = 64; qsOff = 0;
  } else {
    // Every modeled dequantize_row format is now front-door CONSTRUCTED; an
    // unrecognized format falls through to the dispatch-wired monolith.
    return std::nullopt;
  }
  return DequantizeRowStreamFacts{qk,         stride, dOff,      qsOff,   entryLanes,
                                  carrier, nibbleBias, minOff, qhOff};
}

mlir::LogicalResult
constructTypedDequantizeRowLoopBody(mlir::RewriterBase &rewriter,
                                    GgmlDequantizeRowOp deqOp,
                                    const DequantizeRowStreamFacts &facts) {
  mlir::Location loc = deqOp.getLoc();
  llvm::StringRef format = deqOp.getFormat();
  mlir::Value input = deqOp.getInput();
  mlir::Value output = deqOp.getOutput();
  mlir::Value n = deqOp.getElementCount();
  mlir::Type indexType = rewriter.getIndexType();

  {
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(deqOp);

    mlir::OperationState loopState(
        loc, TypedDequantizeRowLoopBodyOp::getOperationName());
    loopState.addOperands({input, output, n});
    loopState.addAttribute(
        "kind", rewriter.getStringAttr("typed_dequantize_row_loop_body"));
    loopState.addAttribute("qk", rewriter.getI64IntegerAttr(facts.qk));
    loopState.addAttribute("weight_block_stride",
                           rewriter.getI64IntegerAttr(facts.weightBlockStride));
    loopState.addAttribute("decode_model", rewriter.getStringAttr(format));
    loopState.addRegion();
    auto loopBody =
        llvm::cast<TypedDequantizeRowLoopBodyOp>(rewriter.create(loopState));

    mlir::Block *block = rewriter.createBlock(
        &loopBody.getBody(), loopBody.getBody().end(), {indexType}, {loc});
    mlir::Value blockIndex = block->getArgument(0);
    rewriter.setInsertionPointToStart(block);

    mlir::OperationState coreState(
        loc, DequantizeRowDecodeCoreOp::getOperationName());
    coreState.addOperands({input, output, blockIndex});
    coreState.addAttribute("decode_model", rewriter.getStringAttr(format));
    coreState.addAttribute("qk", rewriter.getI64IntegerAttr(facts.qk));
    coreState.addAttribute("weight_block_stride",
                           rewriter.getI64IntegerAttr(facts.weightBlockStride));
    coreState.addAttribute("scale_byte_offset",
                           rewriter.getI64IntegerAttr(facts.scaleByteOffset));
    coreState.addAttribute("quant_byte_offset",
                           rewriter.getI64IntegerAttr(facts.quantByteOffset));
    // The g-axis grid geometry descriptor is stamped ONLY for the three owned
    // grid-codebook decode leaves (codebookEntryLanes != 0); every flat / K-quant /
    // non-grid leaf leaves it unstamped so the OptionalAttr stays absent there.
    if (facts.codebookEntryLanes != 0)
      coreState.addAttribute(
          "codebook_entry_lanes",
          rewriter.getI64IntegerAttr(facts.codebookEntryLanes));
    // Phase-1 nibble-family decode-mechanism descriptor: the carrier leaf selector
    // ([K-10] selects between the ALREADY-SEPARATE bare-int8 q8_0 leaf and the shared
    // nibble body, NOT a plan-internal mechanism switch) + the pre-scale bias, and the
    // OPTIONAL min / qh byte offsets whose PRESENCE is the hasMin / hasQh gate. Stamped
    // ONLY for the flat nibble family (q8_0/q4_0/q4_1/q5_0/q5_1); every K-quant / IQ /
    // codebook / ternary leaf leaves carrier == NotNibbleFamily so ALL stay absent.
    if (facts.carrier == NibbleCarrierKind::BareInt8)
      coreState.addAttribute("carrier_kind",
                             rewriter.getStringAttr("bare_int8"));
    else if (facts.carrier == NibbleCarrierKind::Nibble4)
      coreState.addAttribute("carrier_kind", rewriter.getStringAttr("nibble4"));
    if (facts.nibbleBias)
      coreState.addAttribute("nibble_bias",
                             rewriter.getI64IntegerAttr(*facts.nibbleBias));
    if (facts.minByteOffset)
      coreState.addAttribute("min_byte_offset",
                             rewriter.getI64IntegerAttr(*facts.minByteOffset));
    if (facts.qhByteOffset)
      coreState.addAttribute("qh_byte_offset",
                             rewriter.getI64IntegerAttr(*facts.qhByteOffset));
    rewriter.create(coreState);
    rewriter.create<TypedDequantizeRowLoopYieldOp>(loc);
  }
  rewriter.eraseOp(deqOp);

  return mlir::success();
}

} // namespace weft::rvv
