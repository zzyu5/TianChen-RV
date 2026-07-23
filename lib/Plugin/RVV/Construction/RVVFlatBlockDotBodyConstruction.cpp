//===- RVVFlatBlockDotBodyConstruction.cpp -----------------------------===//
//
// Typed flat block-dot body mechanisms. Exact-P selection remains in
// RVVQuantizedBlockDotProblemConstruction.cpp; this module only constructs the
// selected flat topology.
//
//===----------------------------------------------------------------------===//

#include "RVVBlockDotBodyConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>
#include <string>

namespace weft::plugin::rvv {
namespace {

namespace weftrvv = ::weft::rvv;

// ---------------------------------------------------------------------------
// Typed flat block-dot loop-body construction (M-FLAT step 5b, q8_0 only).
//
// The gated typed path builds the COMPLETE per-block typed chain
// (weft_rvv.typed_flat_block_dot_loop_body region: brick 1 dual-fp16 scale
// product -> two per-block i8 loads -> signed widening product -> standalone
// reduce -> lane0 scalar extract -> brick 2 computed-scale dequant -> brick 3
// cross-block f32 accumulate -> yield) instead of the ONE monolith block-dot op.
// The integer-core helpers are copied near-verbatim from the sibling
// RVVReductionSourceFrontDoor.cpp; the six scalar-domain typed ops have no C++
// builder and are built directly from OperationState + the ODS shape.
// ---------------------------------------------------------------------------

// Loop-capable per-block i8 load: base + block_index*block_stride
// (+ quant_byte_offset). Adapts the sibling single-block createRVVLoad with the
// trailing block_index operand + block_stride/quant_byte_offset facts.
mlir::Value createRVVBlockLoad(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value buffer, mlir::Value vl,
                               mlir::Value blockIndex, std::int64_t blockStride,
                               std::int64_t quantByteOffset,
                               mlir::Type vectorType) {
  mlir::OperationState state(loc, weftrvv::LoadOp::getOperationName());
  state.addOperands({buffer, vl, blockIndex});
  state.addAttribute("block_stride", builder.getI64IntegerAttr(blockStride));
  state.addAttribute("quant_byte_offset",
                     builder.getI64IntegerAttr(quantByteOffset));
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

mlir::Value createWideningProduct(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::Value vl, mlir::Type productType,
                                  llvm::StringRef productRelation) {
  mlir::OperationState state(loc,
                             weftrvv::WideningProductOp::getOperationName());
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr("signed_widening_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

// The q4_0 asymmetric offset-binary packed-i4 x plain-i8 integer core: ONE
// packed-i4 weight operand (each i8 packs two offset-binary nibbles) + TWO plain
// int8 activation operands (the q8 low half paired with the low nibbles, the q8
// high half with the high nibbles) -> ONE widened i16 product. The m1 flat-cohort
// rung (i4m1 weight x i8m1 low/high activation -> i16m2) mirrors the mf4 anchor
// rung; the verifier admits both.
mlir::Value createPackedI4OffsetBinaryProduct(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value activationLow, mlir::Value activationHigh, mlir::Value vl,
    mlir::Type productType, llvm::StringRef productRelation) {
  mlir::OperationState state(
      loc, weftrvv::PackedI4OffsetBinaryXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, vl});
  state.addAttribute(
      "kind", builder.getStringAttr("signed_packed_i4_offset_binary_x_i8_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

// The q4_1 asymmetric UNSIGNED-nibble packed-i4 x plain-i8 integer core: ONE
// packed-i4 weight operand (each u8 packs two UNSIGNED nibbles in [0,15]) + TWO
// plain int8 activation operands (the q8 low half paired with the low nibbles, the
// q8 high half with the high nibbles) -> ONE widened i16 product. The q4_1 `-8`
// shift is folded into the block minimum (a separate min brick), so the decode is
// a plain unsigned split (vand 0x0F / vsrl 0x04) with NO bias -- DISTINCT from the
// q4_0 offset-binary sibling. The m1 rung (i4m1 x i8m1 low/high -> i16m2) is the
// only declared rung.
mlir::Value createUnsignedNibbleXI8Product(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value activationLow, mlir::Value activationHigh, mlir::Value vl,
    mlir::Type productType, llvm::StringRef productRelation) {
  mlir::OperationState state(
      loc, weftrvv::UnsignedNibbleXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("unsigned_nibble_x_i8_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

// The q5_0 asymmetric FIVE-BIT offset-binary packed weight x plain-i8 integer
// core: ONE UNSIGNED packed-i4 weight operand (each u8 packs two nibbles) MERGED
// with a per-element 5th bit read from the scalar-domain qh field (the qh_source
// gate-only token from a preceding block_five_bit_qh_source brick, NOT a byte
// offset baked on THIS op) + TWO plain int8 activation operands (the q8 low half
// paired with the low nibbles, the q8 high half with the high nibbles) -> ONE
// widened i16 product. The `-16` offset-binary bias is applied in the decode. The
// m1 flat-cohort rung (i4m1 + qh x i8m1 low/high -> i16m2) is the only rung.
mlir::Value createFiveBitOffsetBinaryXI8Product(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value qhSource, mlir::Value activationLow, mlir::Value activationHigh,
    mlir::Value vl, mlir::Type productType, llvm::StringRef productRelation) {
  mlir::OperationState state(
      loc, weftrvv::FiveBitOffsetBinaryXI8ProductOp::getOperationName());
  state.addOperands({weight, qhSource, activationLow, activationHigh, vl});
  state.addAttribute(
      "kind", builder.getStringAttr("five_bit_offset_binary_x_i8_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

// iq4_nl codebook table broadcast (the CODEBOOK class's prerequisite structure,
// 2nd primitive class): the 16-entry non-linear int8 kvalues table materialized as
// a structured const + broadcast-loaded ONCE into the i8 vreg the gather indexes.
// Consumes NO SSA operands (the table is a compile-time constant); the codebook
// array + the C symbol are the two structural attrs.
mlir::Value createCodebookTableBroadcast(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         llvm::ArrayRef<std::int8_t> codebook,
                                         llvm::StringRef tableSymbol,
                                         mlir::Type tableType) {
  mlir::OperationState state(
      loc, weftrvv::CodebookTableBroadcastOp::getOperationName());
  state.addAttribute("codebook", builder.getDenseI8ArrayAttr(codebook));
  state.addAttribute("table_symbol", builder.getStringAttr(tableSymbol));
  state.addTypes(tableType);
  return builder.create(state)->getResult(0);
}

// The iq4_nl asymmetric CODEBOOK-GATHER packed-i4 x plain-i8 integer core: ONE
// UNSIGNED packed-i4 weight operand (each u8 packs two 4-bit table INDICES) + TWO
// plain int8 activation operands (the q8 low half paired with the low nibbles, the
// q8 high half with the high nibbles) + the broadcast codebook `table` operand ->
// ONE widened i16 product. Unlike the q4_0 offset-binary / q4_1 unsigned-nibble
// siblings, the 4-bit nibble is an INDEX vrgather-decoded through the 16-entry
// kvalues table (NOT an arithmetic decode); it then feeds the SAME asymmetric
// widening product tail (vwmul low + vwmacc high) -> i16m2 the offset-binary
// sibling uses. The m1 rung (i4m1 idx -> gather -> i8m1 x i8m1 low/high -> i16m2)
// is the only declared rung.
mlir::Value createCodebookGatherXI8Product(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value activationLow, mlir::Value activationHigh, mlir::Value table,
    mlir::Value vl, mlir::Type productType, llvm::StringRef productRelation) {
  mlir::OperationState state(
      loc, weftrvv::CodebookGatherXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, table, vl});
  state.addAttribute(
      "kind", builder.getStringAttr("signed_codebook_gather_x_i8_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

mlir::Value createStandaloneReduce(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value input, mlir::Value accumulatorSeed,
                                   mlir::Value vl, mlir::Type resultType) {
  mlir::OperationState state(loc,
                             weftrvv::StandaloneReduceOp::getOperationName());
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_widening_reduce_add"));
  state.addAttribute(
      "accumulator_layout",
      builder.getStringAttr("scalar-i32-seed-lane0-from-accumulator-input"));
  state.addAttribute(
      "result_layout",
      builder.getStringAttr("store-standalone-reduction-lane0-to-output-scalar"));
  state.addTypes(resultType);
  return builder.create(state)->getResult(0);
}

// brick 1: per-block d_x * d_y fp16 scale product (block_index-sourced form).
mlir::Value createBlockFp16ScaleProduct(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value lhsScaleBase,
                                        mlir::Value rhsScaleBase,
                                        mlir::Value blockIndex,
                                        std::int64_t lhsBlockStride,
                                        std::int64_t rhsBlockStride) {
  mlir::OperationState state(
      loc, weftrvv::BlockFp16ScaleProductOp::getOperationName());
  state.addOperands({lhsScaleBase, rhsScaleBase, blockIndex});
  state.addAttribute("kind",
                     builder.getStringAttr("dual_fp16_per_block_scale_product"));
  state.addAttribute("scale_model",
                     builder.getStringAttr("dual-fp16-per-block-d_x.d_y"));
  state.addAttribute("lhs_block_stride",
                     builder.getI64IntegerAttr(lhsBlockStride));
  state.addAttribute("rhs_block_stride",
                     builder.getI64IntegerAttr(rhsBlockStride));
  state.addTypes(builder.getF32Type());
  return builder.create(state)->getResult(0);
}

// Family-B min brick: per-block dual-fp16 MIN/SUM correction product `m_x * s_y`
// (block_index-sourced form). Distinct op TYPE from brick 1; names the SAME
// weight/activation ABI bases + block_index so the emitter's per-block base memo
// hits. The min/sum headers sit at lhs_min_byte_offset / rhs_sum_byte_offset
// within each block base (2/2 for q4_1).
mlir::Value createBlockFp16MinProduct(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhsMinBase,
    mlir::Value rhsSumBase, mlir::Value blockIndex, std::int64_t lhsBlockStride,
    std::int64_t rhsBlockStride, std::int64_t lhsMinByteOffset,
    std::int64_t rhsSumByteOffset) {
  mlir::OperationState state(
      loc, weftrvv::BlockFp16MinProductOp::getOperationName());
  state.addOperands({lhsMinBase, rhsSumBase, blockIndex});
  state.addAttribute("kind",
                     builder.getStringAttr("dual_fp16_per_block_min_product"));
  state.addAttribute("scale_model",
                     builder.getStringAttr("dual-fp16-per-block-m_x.s_y"));
  state.addAttribute("lhs_min_byte_offset",
                     builder.getI64IntegerAttr(lhsMinByteOffset));
  state.addAttribute("rhs_sum_byte_offset",
                     builder.getI64IntegerAttr(rhsSumByteOffset));
  state.addAttribute("lhs_block_stride",
                     builder.getI64IntegerAttr(lhsBlockStride));
  state.addAttribute("rhs_block_stride",
                     builder.getI64IntegerAttr(rhsBlockStride));
  state.addTypes(builder.getF32Type());
  return builder.create(state)->getResult(0);
}

// q5_0 qh-source brick: the per-block 5th-bit SOURCE (block_index-sourced form).
// Names the SAME weight ABI base + block_index as the nibble weight load so the
// emitter per-block base memo hits; the qh header sits at qh_byte_offset within
// each weight block base. Its i32 result is a GATE-ONLY token the five-bit product
// op names (never materialized standalone) -- the bytes are re-read from this
// brick's own qh_base + qh_byte_offset in the emitter (operand/source-driven, NOT
// a descriptor-baked offset on the product op).
mlir::Value createBlockFiveBitQhSource(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value qhBase,
                                       mlir::Value blockIndex,
                                       std::int64_t blockStride,
                                       std::int64_t qhByteOffset) {
  mlir::OperationState state(
      loc, weftrvv::BlockFiveBitQhSourceOp::getOperationName());
  state.addOperands({qhBase, blockIndex});
  state.addAttribute("kind",
                     builder.getStringAttr("block_five_bit_qh_source"));
  state.addAttribute("qh_byte_offset", builder.getI64IntegerAttr(qhByteOffset));
  state.addAttribute("block_stride", builder.getI64IntegerAttr(blockStride));
  state.addTypes(builder.getI32Type());
  return builder.create(state)->getResult(0);
}

// integer-core -> scalar bridge: i32 m1 vector lane0 -> scalar i32 sumi.
mlir::Value createTypedVectorLane0ToScalarExtract(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value input,
                                                  mlir::Value vl) {
  mlir::OperationState state(
      loc, weftrvv::TypedVectorLane0ToScalarExtractOp::getOperationName());
  state.addOperands({input, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("vector_lane0_to_scalar_i32_extract"));
  state.addAttribute("extract_relation",
                     builder.getStringAttr("i32m1-lane0-to-scalar-i32"));
  state.addTypes(builder.getI32Type());
  return builder.create(state)->getResult(0);
}

// brick 2: (float)sumi * computed scale. computed_scale is brick 1's f32 (NOT an
// imported ABI value -- the verifier structurally requires f32 here).
mlir::Value createBlockComputedScaleDequant(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value sumi,
    mlir::Value computedScale, mlir::Value minTerm = mlir::Value()) {
  mlir::OperationState state(
      loc, weftrvv::BlockComputedScaleDequantOp::getOperationName());
  if (minTerm)
    state.addOperands({sumi, computedScale, minTerm});
  else
    state.addOperands({sumi, computedScale});
  state.addAttribute("kind", builder.getStringAttr("computed_scale_sumi_dequant"));
  state.addAttribute(
      "dequant_relation",
      builder.getStringAttr("scalar-i32-sumi-to-f32-computed-scale-f32"));
  state.addTypes(builder.getF32Type());
  return builder.create(state)->getResult(0);
}

// brick 3: sumf + term (cross-block fp32 fold, strict ascending block order).
mlir::Value createCrossBlockF32Accumulate(mlir::OpBuilder &builder,
                                          mlir::Location loc, mlir::Value acc,
                                          mlir::Value term) {
  mlir::OperationState state(
      loc, weftrvv::CrossBlockF32AccumulateOp::getOperationName());
  state.addOperands({acc, term});
  state.addAttribute("kind",
                     builder.getStringAttr("cross_block_f32_scalar_accumulate"));
  state.addAttribute("accumulate_order",
                     builder.getStringAttr("strict-ascending-block-carried"));
  state.addTypes(builder.getF32Type());
  return builder.create(state)->getResult(0);
}

void createTypedFlatBlockDotLoopYield(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value accNext) {
  mlir::OperationState state(
      loc, weftrvv::TypedFlatBlockDotLoopYieldOp::getOperationName());
  state.addOperands(accNext);
  (void)builder.create(state);
}

// The gated typed q8_0 path: build the whole per-block typed chain inside a new
// weft_rvv.typed_flat_block_dot_loop_body region (result-less, region-carrying),
// replacing the ONE monolith block-dot op. The loop-body op's block strides / qk /
// quant offset come from entry.facts (by-name lookup); multi_block_factor is OMITTED
// (absent = mbf 1). weight/activation/out/n are the shared variant-scope ABI values;
// zeroSeed is the variant-scope reduce seed (dominates the in-region reduce); vl is
// the setvl VL referenced freely by the in-region loads/product/reduce/extract.
// `lmul` is the integer-core LMUL schedule ("m1"|"m2"): it drives the
// integer_core_lmul stamp, the coreLmul/wideLmul region vector types, and (for the
// q8_0 whole-block path) the widening product_relation as ONE consistent source, so
// the four verifier-cross-pinned knobs cannot diverge. It MUST agree with the
// enclosing setvl/with_vl config LMUL (the caller passes the same value to both).
// q8_0 defaults to "m2" (the anchor); the half-block packed-i4 formats always pass
// "m1" (their region product_relation pins i8m1).

} // namespace

void createTypedFlatBlockDotLoopChain(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      const MonolithicBlockDotOpEntry &entry,
                                      mlir::Value weight, mlir::Value activation,
                                      mlir::Value out, mlir::Value n,
                                      mlir::Value vl, mlir::Value zeroSeed,
                                      llvm::StringRef lmul) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed flat block-dot chain: missing block-format fact");
  };
  std::int64_t qk = factByName("qk");
  std::int64_t weightStride = factByName("weight_block_stride");
  std::int64_t activationStride = factByName("activation_block_stride");
  std::int64_t quantByteOffset = factByName("quant_byte_offset");

  // Format branch: q8_0 (plain signed i8xi8 whole-block, m2 anchor, sumi-first
  // fold) vs q4_0 (asymmetric offset-binary packed-i4 x i8 HALF-block, m1 anchor,
  // left-assoc fold + the q8 high-half activation strip). Only these two flat ops
  // take the typed loop path; everything else stays the monolith op. q8_0's chain
  // is byte-unchanged from before the branch.
  const bool isQ40 =
      entry.opName == weftrvv::GgmlBlockDotQ40Q80Op::getOperationName();
  // q4_1 (Family-B): shares q4_0's HALF-block m1 packed-i4 shape (3 loads, m1
  // core), diverging only in {u8 weight load, unsigned-nibble product op, the
  // added MIN brick, the scale_plus_min fold}. Every q4_0-guarded knob below is
  // shared (isQ40 || isQ41) EXCEPT the three-way fold_model.
  const bool isQ41 =
      entry.opName == "weft_rvv.q4_1_q8_1_block_dot";
  // q5_0 (five-bit): shares the HALF-block m1 packed-i4 shape (3 loads, m1 core,
  // u8 weight load like q4_1), diverging in {the qh 5th-bit source brick, the
  // five-bit offset-binary product with the `-16` bias, the ScalesTimesSumi fold,
  // and a DISTINCT activation quant offset (weight qs@6, activation qs@2)}.
  const bool isQ50 =
      entry.opName == "weft_rvv.q5_0_q8_0_block_dot";
  // q5_1 (Family-B five-bit, M-FLAT cohort LAST cell): the UNION of q5_0's
  // five-bit integer core (qh 5th-bit brick + five-bit product) and q4_1's MIN
  // term (min brick + scale_plus_min fold). Every knob is shared with EITHER q5_0
  // (qh brick, u8 weight, five-bit product, divergent quant offsets) OR q4_1 (min
  // brick, scale_plus_min fold) -- no q5_1-only knob. The ONE arithmetic delta vs
  // q5_0 (applyOffsetBias=false) lives entirely in the emit driver.
  const bool isQ51 =
      entry.opName == "weft_rvv.q5_1_q8_1_block_dot";
  // iq4_nl (CODEBOOK class, 2nd primitive class): shares the q4_1 HALF-block m1
  // 3-load shape (a u8 packed-i4 weight + the two plain-i8 q8 halves), but the
  // weight nibble is a codebook INDEX (vrgather through the 16-entry kvalues
  // table) NOT an arithmetic decode, so its integer core is the two codebook
  // bricks (codebook_table_broadcast + codebook_gather_x_i8_product) rather than a
  // packed-i4 product. Its fold is SumiTimesScales (the else-default, node-identical
  // to q8_0). It is NOT in isHalfBlock (that flag gates the offset-binary/unsigned
  // packed-i4 product branch); its high-half activation offset is sourced the SAME
  // way (activation_high_byte_offset).
  const bool isIq4Nl = entry.opName == "weft_rvv.iq4_nl_q8_0_block_dot";
  const bool isHalfBlock = isQ40 || isQ41 || isQ50 || isQ51;
  std::int64_t activationHighOffset =
      (isHalfBlock || isIq4Nl) ? factByName("activation_high_byte_offset") : 0;

  mlir::OperationState loopState(
      loc, weftrvv::TypedFlatBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute("kind",
                         builder.getStringAttr("typed_flat_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  loopState.addAttribute(
      "fold_model",
      builder.getStringAttr(isQ50 ? "scales_times_sumi"
                                   : ((isQ41 || isQ51) ? "scale_plus_min"
                                            : (isQ40 ? "left_assoc"
                                                     : "sumi_times_scales"))));
  loopState.addAttribute("integer_core_lmul", builder.getStringAttr(lmul));
  // strip_elision is a VLEN-legality knob, not free. The elided single-cover emits
  // ONE vsetvl(blockLen); that only covers the whole strip when blockLen <= VLMAX.
  // q8_0's whole-block core has blockLen = qk = 32, so at m2 (VLMAX 32 @VLEN128) the
  // elided cover is whole-block-legal, but at m1 (VLMAX 16 @VLEN128) it would
  // SILENTLY cover half the block -- illegal below VLEN256. The half-block packed-i4
  // formats have blockLen = qk/2 = 16, so their m1 elided cover is whole-strip-legal
  // at VLEN128. So the ONLY form that must fall back to the VLEN-robust re-strip is
  // q8_0 at m1; every currently-live path (q8_0-m2, half-block-m1) keeps "elided"
  // byte-for-byte.
  // iq4_nl's codebook core is a half-block-length strip (qk/2 = 16 elements at
  // e8m1, VLMAX 16 @VLEN128 = whole-strip-legal), so like the packed-i4 half-block
  // formats it keeps "elided"; only q8_0's whole-block (blockLen = qk = 32) m1 core
  // needs the VLEN-robust re-strip.
  loopState.addAttribute(
      "strip_elision",
      builder.getStringAttr((!isHalfBlock && !isIq4Nl && lmul == "m1") ? "robust"
                                                                       : "elided"));
  // Deterministic final schedule fields are constructed explicitly.  The
  // emitter never interprets absence as a code-shape choice.
  loopState.addAttribute("multi_block_factor", builder.getI64IntegerAttr(1));
  loopState.addAttribute("fold_structure", builder.getStringAttr("per-block"));
  loopState.addAttribute("numerics_tier", builder.getStringAttr("strict"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedFlatBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value acc = body.addArgument(builder.getF32Type(), loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  mlir::MLIRContext *ctx = builder.getContext();
  // The integer-core widths derive from the ONE `lmul` schedule (widening steps
  // LMUL up one rung): q8_0 anchors i8m2 -> i16m4 (lmul "m2") but is also
  // constructible at i8m1 -> i16m2 (lmul "m1"); q4_0's half-block packed-i4 core
  // anchors i8m1 -> i16m2 (the caller always passes "m1"). The i32m1 reduce lane
  // is shared.
  llvm::StringRef coreLmul = lmul;
  llvm::StringRef wideLmul = (lmul == "m1") ? "m2" : "m4";
  mlir::Type i8VecType =
      weftrvv::VectorType::get(ctx, builder.getI8Type(), coreLmul);
  // q4_1's packed weight strip is UNSIGNED (the nibble value IS the weight, the
  // `-8` folded into the block minimum), so the region weight LoadOp carries a
  // u8-m1 vector type (cf. the codebook front door). q4_0's weight strip stays
  // signed i8.
  mlir::Type ui8VecType = weftrvv::VectorType::get(
      ctx, builder.getIntegerType(8, /*isSigned=*/false), coreLmul);
  mlir::Type i16VecType =
      weftrvv::VectorType::get(ctx, builder.getI16Type(), wideLmul);
  mlir::Type i32VecType =
      weftrvv::VectorType::get(ctx, builder.getI32Type(), "m1");

  // brick 1: the per-block d_x * d_y fp16 scale product over block_index (shared;
  // the AoS strides differ per format but the scale model is identical).
  mlir::Value dd = createBlockFp16ScaleProduct(
      builder, loc, weight, activation, blockIndex, weightStride,
      activationStride);

  // q4_1 min brick: the per-block m_x * s_y correction product (Family-B). Names
  // the SAME weight/activation ABI bases + block_index as brick 1 so the emitter
  // per-block base memo hits; the min/sum headers sit at weight_min_byte_offset /
  // activation_sum_byte_offset. Null (no brick) for q8_0/q4_0.
  mlir::Value minTerm;
  if (isQ41 || isQ51)
    minTerm = createBlockFp16MinProduct(
        builder, loc, weight, activation, blockIndex, weightStride,
        activationStride, factByName("weight_min_byte_offset"),
        factByName("activation_sum_byte_offset"));

  // q5_0 qh-source brick: the per-block 5th-bit SOURCE token the five-bit product
  // names. Like the q4_1 min brick, names the SAME weight ABI base + block_index
  // (the qh header lives WITHIN the weight block), so the emitter per-block base
  // memo hits. Null (no brick) for q8_0/q4_0/q4_1.
  mlir::Value qhSource;
  if (isQ50 || isQ51)
    qhSource = createBlockFiveBitQhSource(builder, loc, weight, blockIndex,
                                          weightStride,
                                          factByName("weight_qh_byte_offset"));

  // The vector integer core diverges by format. q8_0: two per-block i8 loads ->
  // signed widening product. q4_0/q4_1: ONE packed-i4 weight load + TWO plain-i8
  // activation loads (the q8 low half at quant_off, the q8 high half at
  // quant_off + activation_high_byte_offset) -> the asymmetric packed-i4 x i8
  // product. q4_0 loads the weight SIGNED + offset-binary decode; q4_1 loads it
  // UNSIGNED (u8) + unsigned-nibble decode.
  // Activation quant offset. For q4_0/q4_1 the weight and activation qs offsets
  // COINCIDE, so both loads use quant_byte_offset. q5_0 is the FIRST flat op where
  // they DIVERGE (weight qs@6, activation qs@2): its avLow/avHigh MUST read the
  // separate activation_quant_byte_offset fact (the weight strip still uses
  // quant_byte_offset). Keeping this = quantByteOffset off the q5_0 path leaves the
  // q4_0/q4_1 loads byte-identical.
  std::int64_t activationQuantByteOffset =
      (isQ50 || isQ51) ? factByName("activation_quant_byte_offset")
                       : quantByteOffset;

  mlir::Value prod;
  if (isIq4Nl) {
    // iq4_nl CODEBOOK integer core (2nd primitive class): the 16-entry non-linear
    // int8 kvalues table broadcast ONCE + the u8 packed-i4 weight strip + the two
    // plain-i8 q8 halves -> the asymmetric codebook-gather product. The weight and
    // activation qs offsets COINCIDE (weight qs@2, activation qs@2, like q4_0/q4_1),
    // so both use quantByteOffset; the high half sits at quant_off +
    // activation_high_byte_offset. The nibble is a vrgather INDEX (NOT an arithmetic
    // decode); the gathered i8 weight lanes feed the SAME widening product tail
    // (i8m1 x i8m1x2 -> i16m2). fold_model is SumiTimesScales (the else-default).
    mlir::Value table = createCodebookTableBroadcast(
        builder, loc, entry.codebook, "weft_iq4_nl_kvalues", i8VecType);
    mlir::Value wv =
        createRVVBlockLoad(builder, loc, weight, vl, blockIndex, weightStride,
                           quantByteOffset, ui8VecType);
    mlir::Value avLow =
        createRVVBlockLoad(builder, loc, activation, vl, blockIndex,
                           activationStride, quantByteOffset, i8VecType);
    mlir::Value avHigh = createRVVBlockLoad(
        builder, loc, activation, vl, blockIndex, activationStride,
        quantByteOffset + activationHighOffset, i8VecType);
    prod = createCodebookGatherXI8Product(
        builder, loc, wv, avLow, avHigh, table, vl, i16VecType,
        "codebook-gather-i8-x-i8x2-to-i16");
  } else if (isHalfBlock) {
    // packed-i4 weight strip (base + ib*stride + quant_off). q4_1/q5_0 = u8 (the
    // nibble decode is UNSIGNED), q4_0 = signed i8.
    mlir::Value wv = createRVVBlockLoad(
        builder, loc, weight, vl, blockIndex, weightStride, quantByteOffset,
        (isQ41 || isQ50 || isQ51) ? ui8VecType : i8VecType);
    // q8 low half (activation quant_off) and high half (+ activation_high_offset).
    mlir::Value avLow = createRVVBlockLoad(builder, loc, activation, vl, blockIndex,
                                           activationStride,
                                           activationQuantByteOffset, i8VecType);
    mlir::Value avHigh = createRVVBlockLoad(
        builder, loc, activation, vl, blockIndex, activationStride,
        activationQuantByteOffset + activationHighOffset, i8VecType);
    if (isQ50 || isQ51)
      // asymmetric FIVE-BIT offset-binary packed-i4 (+ qh 5th bit) x i8 product
      // (i4m1 + qh x i8m1x2 -> i16m2). The qh 5th-bit source is the qh brick's
      // gate-only token, NOT a byte offset on this product op. The `-16` bias is a
      // pure EMIT-driver choice (q5_0 on, q5_1 off -- q5_1's bias lives in its
      // per-block MIN scale); the op is structurally identical for both formats.
      prod = createFiveBitOffsetBinaryXI8Product(
          builder, loc, wv, qhSource, avLow, avHigh, vl, i16VecType,
          "five-bit-offset-binary-i4m1-x-i8m1x2-to-i16m2");
    else if (isQ41)
      // asymmetric UNSIGNED-nibble packed-i4 x i8 product (i4m1 x i8m1x2 -> i16m2).
      prod = createUnsignedNibbleXI8Product(
          builder, loc, wv, avLow, avHigh, vl, i16VecType,
          "unsigned-nibble-i4m1-x-i8m1x2-to-i16m2");
    else
      // asymmetric offset-binary packed-i4 x i8 product (i4m1 x i8m1x2 -> i16m2).
      prod = createPackedI4OffsetBinaryProduct(
          builder, loc, wv, avLow, avHigh, vl, i16VecType,
          "offset-binary-i4m1-x-i8m1x2-to-i16m2");
  } else {
    // q8_0: two per-block i8 loads (base + ib*stride + quant_off).
    mlir::Value wv =
        createRVVBlockLoad(builder, loc, weight, vl, blockIndex, weightStride,
                           quantByteOffset, i8VecType);
    mlir::Value av =
        createRVVBlockLoad(builder, loc, activation, vl, blockIndex,
                           activationStride, quantByteOffset, i8VecType);
    // signed widening product: i8m2 x i8m2 -> i16m4 (lmul "m2", the anchor) or
    // i8m1 x i8m1 -> i16m2 (lmul "m1", the byte-anchor dot-reduce rung). Both
    // relations are admitted by the widening_product verifier; the arithmetic is
    // bit-identical (LMUL is a schedule knob, not an arithmetic one).
    prod = createWideningProduct(builder, loc, wv, av, vl, i16VecType,
                                 (lmul == "m1") ? "signed-i8m1xi8m1-to-i16m2"
                                                : "signed-i8m2xi8m2-to-i16m4");
  }
  // reduce i16<wide> -> i32m1 lane0, then extract lane0 -> scalar i32 sumi.
  mlir::Value red =
      createStandaloneReduce(builder, loc, prod, zeroSeed, vl, i32VecType);
  mlir::Value sumi =
      createTypedVectorLane0ToScalarExtract(builder, loc, red, vl);
  // brick 2: (float)sumi * (d_x * d_y) (+ m_x*s_y min_term for q4_1).
  mlir::Value bterm =
      createBlockComputedScaleDequant(builder, loc, sumi, dd, minTerm);
  // brick 3: sumf + term (cross-block fp32 fold).
  mlir::Value accNext = createCrossBlockF32Accumulate(builder, loc, acc, bterm);
  createTypedFlatBlockDotLoopYield(builder, loc, accNext);
}

// ---------------------------------------------------------------------------
// The q1_0 (BINARY {-1,+1}-sign class) sibling of createTypedFlatBlockDotLoopChain
// -- the LAST flat block-dot family member, the genuine structural-special case
// (C_construct 26->27). Unlike q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl (a single per-block
// integer core folded by the shared brick 1 (scale) -> brick 2 (dequant) -> brick 3
// (cross-block accumulate) chain), q1_0's per-super-block contribution is a
// FOUR-sub-block binary sign decode with a DISTINCT TWO-LEVEL fp32 fold
// (`d0 * Σ_k(d1_k * sumi_block_k)`), so no existing flat brick chain expresses it.
// The net-new marginal cost is ONE brick -- the q1_0 BINARY-sign integer core
// (GgmlBlockDotQ10Q80BinarySignCoreOp) -- carrying the whole per-super-block body;
// the two-level fold is emitter-inlined (the super-block scalar-core precedent
// tq1_0/iq1_s applied to the FLAT loop op). q1_0's activation is a FLAT block_q8_0
// stream (four 34-byte q8_0 blocks per q1_0 super-block), so it uses the FLAT loop
// op (typed_flat_block_dot_loop_body, fold_model "flat_binary_two_level") -- NOT the
// super-block one (q8_K). The OUTER with_vl frame stays SEW32/m1 (like the monolith
// / iq4_nl): the e8m2 binary sign decode runs its OWN vsetvl INSIDE the brick, so
// this chain is dispatched OUTSIDE typedFlatLoopPath (which would force SEW8). The
// loop op + brick are left attr-less (default m2 anchor = the byte-exact CORE
// target); q1_0's Win-A gearbox lives on the binary-sign brick (kernel key "q1_0").
void createTypedFlatBlockDotLoopChainQ10(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed flat binary-sign q1_0 chain: missing block-format fact");
  };
  std::int64_t qk = factByName("qk");                            // 128
  std::int64_t weightStride = factByName("weight_block_stride");         //  18
  std::int64_t activationStride = factByName("activation_block_stride"); //  34
  std::int64_t blocksPerWeight = factByName("activation_blocks_per_weight"); // 4
  std::int64_t weightQuantOffset = factByName("weight_quant_byte_offset");   // 2
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");                 //   2

  mlir::Type i32ScalarType = builder.getI32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedFlatBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute("kind",
                         builder.getStringAttr("typed_flat_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "flat_binary_two_level" KEYS the emitter dispatch (the q1_0 branch)
  // + the two-level scalar fold; the emitter disambiguates q1_0 by the in-region
  // binary-sign integer-core brick op TYPE. integer_core_lmul / multi_block_factor /
  // strip_elision are LEFT OFF (attr-less = the default m2 anchor, byte-exact target).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("flat_binary_two_level"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedFlatBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value acc = body.addArgument(builder.getF32Type(), loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the q1_0 BINARY-sign integer core (the FOUR q8_0 sub-blocks, each a
  // vlm_v_b{ratio} packed-bit sign mask + vle8 q8 + i8-domain vneg/vmerge -> ONE
  // vwredsum i8->i16m1, plus the emitter-inlined two-level fp32 fold). The LIVE
  // operands are the weight base (%vx) + activation base (%vy) + n + vl +
  // block_index; it produces ONE scalar i32 SSA result (a placeholder -- the
  // emitter re-emits the whole body including the fold). Per-super-block address
  // vx + ib*18, vy + (ib*4 + k)*34. Left attr-less so the gearbox is free to stamp
  // integer_core_lmul m2/m1 from the VLEN capability fact (kernel key "q1_0").
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotQ10Q80BinarySignCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, blockIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_q1_0_q8_0_binary_sign_core"));
    s.addAttribute("scale_model",
                   builder.getStringAttr("binary-sign-per-bit"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("activation_blocks_per_weight",
                   builder.getI64IntegerAttr(blocksPerWeight));
    s.addAttribute("weight_quant_byte_offset",
                   builder.getI64IntegerAttr(weightQuantOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The loop yield carries the loop-carried acc UNCHANGED (the two-level fold is
  // emitter-inlined by the binary-sign brick lowering, byte-exact = the monolith
  // sumf; mirrors the tq1_0 scalar-core yield contract). The flat verifier requires
  // an f32 acc_next -- acc IS the f32 region arg 1.
  createTypedFlatBlockDotLoopYield(builder, loc, acc);
}

// ---------------------------------------------------------------------------
// The nvfp4 (SECOND FP4-CODEBOOK class, NVIDIA's FP4) sibling of
// createTypedFlatBlockDotLoopChainQ10 -- the LAST dispatch-wired vec_dot to flip
// (C_construct 27->28, closing the ① G1 literal-block-dot zoo). nvfp4 is a
// SUPER-BLOCK codebook quant (block_nvfp4 = {uint8_t d[4]; uint8_t qs[32]}, QK=64,
// four 16-element sub-blocks) whose 64 elements span TWO block_q8_0 activation
// blocks -- a FLAT block_q8_0 stream (like q1_0's four-block stream), so it uses the
// FLAT loop op (typed_flat_block_dot_loop_body, fold_model "flat_nvfp4_codebook") --
// NOT the super-block one (q8_K). Like q1_0 it REUSES mxfp4's 16-entry DOUBLED e2m1
// codebook GATHER (vrgather_vv_i8m1) verbatim; the genuinely-new fact is the
// per-SUB-block UE4M3 fp8 weight scale (ldexpf-based, HALF form). The whole
// per-super-block body (the four UE4M3-scaled codebook sub-blocks + the per-sub-block
// fp32 fold `sumf += (dy*d)*sumi`) is carried by ONE net-new codebook integer-core
// brick (GgmlBlockDotNVFP4Q80CodebookCoreOp) + emitter-inlined through the SAME
// emitNVFP4BlockDotBodyShared the retired monolith emitter called, so the emit is
// byte-identical to the monolith. The OUTER with_vl frame stays SEW32/m1 (like q1_0
// / the monolith): the e8m1 codebook strip runs its OWN vsetvl INSIDE the brick, so
// this chain is dispatched OUTSIDE typedFlatLoopPath (which would force SEW8). The
// loop op + brick are left attr-less on the shape knob (the codebook gather pins m1;
// the emitter's fixed vrgather/i8m1/i16m2 codebook dot matches the untuned monolith
// byte-identically). The brick's per-super-block addressing keys off the loop
// induction variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedFlatBlockDotLoopChainNvfp4(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed flat nvfp4 codebook chain: missing block-format fact");
  };
  std::int64_t qk = factByName("qk");                          // 64 (QK_NVFP4)
  std::int64_t qkSub = factByName("qk_sub");                   // 16 (QK_NVFP4_SUB)
  std::int64_t weightStride = factByName("weight_block_stride");        //  36
  std::int64_t activationStride = factByName("activation_block_stride"); //  34
  std::int64_t weightQuantOffset = factByName("weight_quant_byte_offset");   // 4
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");               //   2
  std::int64_t activationHighOffset =
      factByName("activation_high_byte_offset");                //   8

  mlir::Type i32ScalarType = builder.getI32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedFlatBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute("kind",
                         builder.getStringAttr("typed_flat_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "flat_nvfp4_codebook" KEYS the emitter dispatch (the nvfp4 branch) +
  // the per-sub-block UE4M3-codebook fold; the emitter disambiguates nvfp4 by the
  // in-region codebook integer-core brick op TYPE.  The loop-level scheduling
  // axes are inapplicable to this closed body; the codebook brick still records
  // its fixed m1 compute anchor explicitly.
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("flat_nvfp4_codebook"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedFlatBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value acc = body.addArgument(builder.getF32Type(), loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the nvfp4 FP4-CODEBOOK integer core (the 16-entry DOUBLED e2m1 codebook
  // broadcast + the four UE4M3-scaled 16-element sub-blocks' nibble split + vrgather
  // decode + asymmetric vwmul/vwmacc widening product + seed-0 vwredsum, wrapped in
  // the per-sub-block UE4M3 fp8 weight scale + the two-q8_0-block/half addressing +
  // the per-sub-block float fold into sumf). The LIVE operands are the weight base
  // (%vx) + activation base (%vy) + n + vl + block_index; it produces ONE scalar i32
  // SSA result (an UNUSED per-super-block placeholder -- nvfp4's fold is per-sub-block
  // float, no single scalar state). Per-super-block address vx + ib*36, vy +
  // (2*ib + s/2)*34. The 16-entry codebook (kvalues_mxfp4[16]) is carried as a
  // DenseI8ArrayAttr like the monolith.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotNVFP4Q80CodebookCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, blockIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_nvfp4_q8_0_codebook_core"));
    s.addAttribute("scale_model",
                   builder.getStringAttr("ue4m3-half-per-sub-block"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("qk_sub", builder.getI64IntegerAttr(qkSub));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_quant_byte_offset",
                   builder.getI64IntegerAttr(weightQuantOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addAttribute("activation_high_byte_offset",
                   builder.getI64IntegerAttr(activationHighOffset));
    s.addAttribute("codebook", builder.getDenseI8ArrayAttr(entry.codebook));
    s.addAttribute("integer_core_lmul", builder.getStringAttr("m1"));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The loop yield carries the loop-carried acc UNCHANGED (the per-sub-block fold is
  // emitter-inlined by the codebook brick lowering, byte-exact = the monolith sumf;
  // mirrors the q1_0 core's yield contract). The flat verifier requires an f32
  // acc_next -- acc IS the f32 region arg 1.
  createTypedFlatBlockDotLoopYield(builder, loc, acc);
}

} // namespace weft::plugin::rvv
