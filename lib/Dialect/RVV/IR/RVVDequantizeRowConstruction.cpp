//===- RVVDequantizeRowConstruction.cpp ---------------------------------===//
//
// The ONE byte-exact construction of the streaming dequantize_row FRONT-DOOR
// typed region, shared by the explicit RVV source front door and the backend
// pre-conversion dequant plan materializer. Emission accepts only the resulting
// typed region.
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
  // the owned grid-codebook decode leaves (iq3_s grid-of-4 uint32 = 4; iq2_xxs / iq2_xs
  // / iq2_s / iq1_s / iq1_m grid-of-8 uint64 = 8). It stays 0 (unstamped) for every
  // flat / K-quant / non-grid leaf, which never reads an entry width.
  std::int64_t qk = 32, stride = 0, dOff = 0, qsOff = 0, entryLanes = 0;
  // Phase-1 nibble-family decode-mechanism descriptor (unset == NotNibbleFamily /
  // absent for every non-nibble leaf; the flat nibble arms below set them).
  NibbleCarrierKind carrier = NibbleCarrierKind::NotNibbleFamily;
  DequantizeRowMechanismKind mechanism =
      DequantizeRowMechanismKind::Int8Scale;
  std::optional<::weft::CodebookScaleModel> codebookScaleModel;
  std::optional<::weft::KQuantScaleModel> kquantScaleModel;
  std::optional<::weft::GridDecodeLeaf> gridDecodeLeaf;
  std::optional<::weft::TernaryDecodeLeaf> ternaryDecodeLeaf;
  std::optional<std::int64_t> nibbleBias, minOff, qhOff;
  if (format == "q8_0") {
    stride = 34; qsOff = 2; carrier = NibbleCarrierKind::BareInt8;
  } else if (format == "q4_0") {
    mechanism = DequantizeRowMechanismKind::NibbleDecode;
    stride = 18; qsOff = 2; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 8;
  } else if (format == "q4_1") {
    mechanism = DequantizeRowMechanismKind::NibbleDecode;
    stride = 20; qsOff = 4; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 0;
    minOff = 2;
  } else if (format == "q5_0") {
    mechanism = DequantizeRowMechanismKind::NibbleDecode;
    stride = 22; qsOff = 6; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 16;
    qhOff = 2;
  } else if (format == "q5_1") {
    mechanism = DequantizeRowMechanismKind::NibbleDecode;
    stride = 24; qsOff = 8; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 0;
    minOff = 2; qhOff = 4;
  } else if (format == "q4_synth") {
    mechanism = DequantizeRowMechanismKind::NibbleDecode;
    // F1 FALSIFIER (phase-1 C1/C2): a synthetic 4-bit nibble dequant leaf added by THIS
    // SINGLE descriptor row (+ its lit) -- ZERO emitter / verifier mechanism lines. It
    // carries q4_0's EXACT decode tuple (stride 18, quant_byte_offset 2, nibble_bias 8,
    // carrier nibble4, no min / no qh), so its emitted C is BYTE-IDENTICAL to
    // dequantize_row_q4_0 modulo only the provenance token: the constructed typed region
    // dispatches on carrier_kind == "nibble4" (NOT the format name) into the shared
    // nibble body. Directly falsifies "the format name drives dispatch".
    stride = 18; qsOff = 2; carrier = NibbleCarrierKind::Nibble4; nibbleBias = 8;
  } else if (format == "q1_0") {
    mechanism = DequantizeRowMechanismKind::BinarySign;
    // block_q1_0: fp16 d @0, qs[16] @2 (QK1_0=128 packed 1-bit binary {-1,+1}
    // signs, 8 weights/byte). The flat binary-sign leaf: y[j] = bit ? d : -d.
    qk = 128; stride = 18; dOff = 0; qsOff = 2;
  } else if (format == "q2_K") {
    mechanism = DequantizeRowMechanismKind::KQuantScaleMin;
    kquantScaleModel = ::weft::KQuantScaleModel::Q2K;
    qk = 256; stride = 84; dOff = 80; qsOff = 16;
  } else if (format == "q3_K") {
    mechanism = DequantizeRowMechanismKind::KQuantScaleMin;
    kquantScaleModel = ::weft::KQuantScaleModel::Q3K;
    qk = 256; stride = 110; dOff = 108; qsOff = 32;
  } else if (format == "q4_K") {
    mechanism = DequantizeRowMechanismKind::KQuantScaleMin;
    kquantScaleModel = ::weft::KQuantScaleModel::Q4K;
    qk = 256; stride = 144; dOff = 0; qsOff = 16;
  } else if (format == "q5_K") {
    mechanism = DequantizeRowMechanismKind::KQuantScaleMin;
    kquantScaleModel = ::weft::KQuantScaleModel::Q5K;
    qk = 256; stride = 176; dOff = 0; qsOff = 48;
  } else if (format == "q6_K") {
    mechanism = DequantizeRowMechanismKind::KQuantScaleMin;
    kquantScaleModel = ::weft::KQuantScaleModel::Q6K;
    qk = 256; stride = 210; dOff = 208; qsOff = 0;
  } else if (format == "iq2_xxs") {
    mechanism = DequantizeRowMechanismKind::GridLookup;
    gridDecodeLeaf = ::weft::GridDecodeLeaf::Iq2Xxs;
    // grid-of-8 (int64, 256-entry): each grid entry = 8 contiguous grid bytes.
    qk = 256; stride = 66; dOff = 0; qsOff = 2; entryLanes = 8;
  } else if (format == "iq2_xs") {
    mechanism = DequantizeRowMechanismKind::GridLookup;
    gridDecodeLeaf = ::weft::GridDecodeLeaf::Iq2Xs;
    // grid-of-8 (int64, 512-entry): each grid entry = 8 contiguous grid bytes.
    qk = 256; stride = 74; dOff = 0; qsOff = 2; entryLanes = 8;
  } else if (format == "iq2_s") {
    mechanism = DequantizeRowMechanismKind::GridLookup;
    gridDecodeLeaf = ::weft::GridDecodeLeaf::Iq2S;
    // grid-of-8 (int64, 1024-entry): each grid entry = 8 contiguous grid bytes.
    qk = 256; stride = 82; dOff = 0; qsOff = 2; entryLanes = 8;
  } else if (format == "iq3_xxs") {
    mechanism = DequantizeRowMechanismKind::GridLookup;
    gridDecodeLeaf = ::weft::GridDecodeLeaf::Iq3Xxs;
    qk = 256; stride = 98; dOff = 0; qsOff = 2;
  } else if (format == "iq3_s") {
    mechanism = DequantizeRowMechanismKind::GridLookup;
    gridDecodeLeaf = ::weft::GridDecodeLeaf::Iq3S;
    // grid-of-4 (uint32, EXPLICIT signs): each grid entry = 4 contiguous grid bytes.
    qk = 256; stride = 110; dOff = 0; qsOff = 2; entryLanes = 4;
  } else if (format == "iq1_s") {
    mechanism = DequantizeRowMechanismKind::TernaryDecode;
    ternaryDecodeLeaf = ::weft::TernaryDecodeLeaf::Iq1S;
    // block_iq1_s: fp16 d @0, qs[32] @2, qh[8] u16 @34 (ternary iq1s_grid + delta).
    // grid-of-8 (2048-entry iq1s_grid): each grid entry = 8 contiguous ternary bytes.
    qk = 256; stride = 50; dOff = 0; qsOff = 2; entryLanes = 8;
  } else if (format == "iq1_m") {
    mechanism = DequantizeRowMechanismKind::TernaryDecode;
    ternaryDecodeLeaf = ::weft::TernaryDecodeLeaf::Iq1M;
    // block_iq1_m: NO fp16 d -- qs[32] LEAD the block @0, qh[16] @32, packed scale
    // words @48 (the super-block d is the reconstructed iq1m_scale fp16). grid-of-8
    // (2048-entry iq1s_grid): each grid entry = 8 contiguous ternary grid bytes.
    qk = 256; stride = 56; dOff = 0; qsOff = 0; entryLanes = 8;
  } else if (format == "iq4_nl") {
    // Source identity selects only the scale ABI; the shared typed layout row
    // below owns every fixed small-codebook geometry field.
    codebookScaleModel = ::weft::CodebookScaleModel::Fp16Flat;
    mechanism = DequantizeRowMechanismKind::CodebookGather;
  } else if (format == "iq4_xs") {
    codebookScaleModel = ::weft::CodebookScaleModel::Signed6SuperBlock;
    mechanism = DequantizeRowMechanismKind::CodebookGather;
  } else if (format == "mxfp4") {
    codebookScaleModel = ::weft::CodebookScaleModel::E8M0SharedExp;
    mechanism = DequantizeRowMechanismKind::CodebookGather;
  } else if (format == "nvfp4") {
    codebookScaleModel = ::weft::CodebookScaleModel::UE4M3SubBlock;
    mechanism = DequantizeRowMechanismKind::CodebookGather;
  } else if (format == "tq1_0") {
    mechanism = DequantizeRowMechanismKind::TernaryDecode;
    ternaryDecodeLeaf = ::weft::TernaryDecodeLeaf::Tq1_0;
    // block_tq1_0: qs[48] @0 (base-3 packed, 5 elems/byte), qh[4] @48, fp16 d @52
    // (the ternary {-1,0,+1} TriLM super-block; scale is at the END, not @0).
    qk = 256; stride = 54; dOff = 52; qsOff = 0;
  } else if (format == "tq2_0") {
    mechanism = DequantizeRowMechanismKind::TernaryDecode;
    ternaryDecodeLeaf = ::weft::TernaryDecodeLeaf::Tq2_0;
    // block_tq2_0: qs[64] @0 (2-bit packed, 4 elems/byte), fp16 d @64 (the 2-bit
    // ternary super-block; scale is at the END, not @0).
    qk = 256; stride = 66; dOff = 64; qsOff = 0;
  } else {
    // Every modeled dequantize_row format is now front-door CONSTRUCTED; an
    // unrecognized format falls through to the dispatch-wired monolith.
    return std::nullopt;
  }

  if (codebookScaleModel) {
    std::optional<::weft::CodebookGatherLayoutFacts> layout =
        ::weft::lookupCodebookGatherLayoutFacts(*codebookScaleModel);
    if (!layout)
      return std::nullopt;
    qk = layout->qk;
    stride = layout->weightBlockStride;
    dOff = layout->scaleByteOffset;
    qsOff = layout->quantByteOffset;
  }
  DequantizeRowStreamFacts facts{};
  facts.qk = qk;
  facts.weightBlockStride = stride;
  facts.scaleByteOffset = dOff;
  facts.quantByteOffset = qsOff;
  facts.codebookEntryLanes = entryLanes;
  facts.mechanism = mechanism;
  facts.codebookScaleModel = codebookScaleModel;
  facts.kquantScaleModel = kquantScaleModel;
  facts.gridDecodeLeaf = gridDecodeLeaf;
  facts.ternaryDecodeLeaf = ternaryDecodeLeaf;
  facts.carrier = carrier;
  facts.nibbleBias = nibbleBias;
  facts.minByteOffset = minOff;
  facts.qhByteOffset = qhOff;
  return facts;
}

mlir::LogicalResult
constructTypedDequantizeRowLoopBody(mlir::RewriterBase &rewriter,
                                    GgmlDequantizeRowOp deqOp,
                                    const DequantizeRowConstruction &construction) {
  const DequantizeRowStreamFacts &facts = construction.facts;
  const DequantizeRowMechanismPlan &mechanismPlan =
      construction.mechanismPlan;
  bool planMatches = false;
  switch (facts.mechanism) {
  case DequantizeRowMechanismKind::Int8Scale:
  case DequantizeRowMechanismKind::BinarySign:
    planMatches = std::holds_alternative<std::monostate>(mechanismPlan);
    break;
  case DequantizeRowMechanismKind::NibbleDecode:
    planMatches = std::holds_alternative<::weft::NibbleDecodePlan>(mechanismPlan);
    break;
  case DequantizeRowMechanismKind::CodebookGather:
    planMatches =
        std::holds_alternative<::weft::CodebookGatherPlan>(mechanismPlan);
    break;
  case DequantizeRowMechanismKind::KQuantScaleMin:
    planMatches =
        std::holds_alternative<::weft::KQuantScaleMinPlan>(mechanismPlan);
    break;
  case DequantizeRowMechanismKind::GridLookup:
    planMatches = std::holds_alternative<::weft::GridLookupPlan>(mechanismPlan);
    break;
  case DequantizeRowMechanismKind::TernaryDecode:
    planMatches =
        std::holds_alternative<::weft::TernaryDecodePlan>(mechanismPlan);
    break;
  }
  if (!planMatches) {
    deqOp.emitError() << "formula construction returned a mechanism plan that "
                         "does not match typed dequant geometry";
    return mlir::failure();
  }
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
    // Construction owns mechanism and leaf classification for every current
    // dequant path. decode_model remains source provenance only after this point.
    coreState.addAttribute(
        "dequant_mechanism",
        rewriter.getStringAttr(
            stringifyDequantizeRowMechanismKind(facts.mechanism)));
    if (facts.codebookScaleModel) {
      coreState.addAttribute(
          "codebook_scale_model",
          rewriter.getStringAttr(
              ::weft::stringifyCodebookScaleModel(
                  *facts.codebookScaleModel)));
    }
    if (facts.kquantScaleModel)
      coreState.addAttribute(
          "kquant_scale_model",
          rewriter.getStringAttr(::weft::stringifyKQuantScaleModel(
              *facts.kquantScaleModel)));
    if (facts.gridDecodeLeaf)
      coreState.addAttribute(
          "grid_decode_leaf",
          rewriter.getStringAttr(::weft::stringifyGridDecodeLeaf(
              *facts.gridDecodeLeaf)));
    if (facts.ternaryDecodeLeaf)
      coreState.addAttribute(
          "ternary_decode_leaf",
          rewriter.getStringAttr(::weft::stringifyTernaryDecodeLeaf(
              *facts.ternaryDecodeLeaf)));
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

    if (const auto *plan =
            std::get_if<::weft::NibbleDecodePlan>(&mechanismPlan)) {
      coreState.addAttribute("dequant_load_lmul",
                             rewriter.getStringAttr(plan->loadLMUL));
      coreState.addAttribute("dequant_strip_lanes",
                             rewriter.getI64IntegerAttr(plan->stripLanes));
    } else if (const auto *plan =
                   std::get_if<::weft::CodebookGatherPlan>(&mechanismPlan)) {
      coreState.addAttribute("dequant_load_lmul",
                             rewriter.getStringAttr(plan->loadLMUL));
      coreState.addAttribute("dequant_strip_lanes",
                             rewriter.getI64IntegerAttr(plan->stripLanes));
      coreState.addAttribute(
          "codebook_gather_table",
          rewriter.getStringAttr(
              ::weft::stringifyCodebookTable(plan->codebookTable)));
      coreState.addAttribute(
          "codebook_gather_entries",
          rewriter.getI64IntegerAttr(plan->codebookEntries));
    } else if (const auto *plan =
                   std::get_if<::weft::KQuantScaleMinPlan>(&mechanismPlan)) {
      coreState.addAttribute("dequant_load_lmul",
                             rewriter.getStringAttr(plan->loadLMUL));
      coreState.addAttribute("dequant_strip_lanes",
                             rewriter.getI64IntegerAttr(plan->stripLanes));
      coreState.addAttribute("kquant_has_min",
                             rewriter.getBoolAttr(plan->hasMin));
      coreState.addAttribute(
          "kquant_min_byte_offset",
          rewriter.getI64IntegerAttr(plan->minByteOffset));
      coreState.addAttribute(
          "kquant_sub_scale_byte_offset",
          rewriter.getI64IntegerAttr(plan->subScaleByteOffset));
      coreState.addAttribute("kquant_has_high_bit_plane",
                             rewriter.getBoolAttr(plan->hasHighBitPlane));
      coreState.addAttribute(
          "kquant_high_bit_byte_offset",
          rewriter.getI64IntegerAttr(plan->highBitByteOffset));
    }
    rewriter.create(coreState);
    rewriter.create<TypedDequantizeRowLoopYieldOp>(loc);
  }
  rewriter.eraseOp(deqOp);

  return mlir::success();
}

} // namespace weft::rvv
