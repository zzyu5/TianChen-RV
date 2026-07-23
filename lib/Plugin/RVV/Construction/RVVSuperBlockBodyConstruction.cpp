//===- RVVSuperBlockBodyConstruction.cpp -------------------------------===//
//
// Arithmetic super-block body mechanisms: dual scale/min, symmetric
// scales-times-sumi, and scalar scale/min folds.
//
//===----------------------------------------------------------------------===//

#include "RVVBlockDotBodyConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace weft::plugin::rvv {

namespace weftrvv = ::weft::rvv;

// ---------------------------------------------------------------------------
// Typed SUPER-BLOCK block-dot loop-body construction (M-FLAT q4_K milestone-3).
//
// The super-block sibling of createTypedFlatBlockDotLoopChain: it assembles the
// q4_K/q5_K OUTER nb = n/QK_K loop as ONE region-carrying
// weft_rvv.typed_super_block_block_dot_loop_body op carrying a DUAL accumulator
// (the `sums` 8-lane fp32 vector + the `sumf` scalar fp32), with the 5 in-loop
// q4_K bricks (nibble_unpack -> scale_min_bit_dance -> scaled_dot -> min_term ->
// sums_fold_scale_d) + the dual yield inside the region, replacing the ONE
// monolith weft_rvv.q4_k_q8_k_block_dot op. Every brick's per-super-block
// addressing keys off the loop induction variable (region arg 0), so the emit is
// operand-driven (anti-bypass): a changed brick base operand emits a different
// base, and a dropped block_index fails to legalize.
//
// The bricks' aux8/scales/aux32 SCRATCH operands are function-scoped scratch the
// super-block loop emitter DECLARES itself (int8_t aux8[256] / uint32_t utmp[4] /
// int32_t aux32[8]); it never reads these operand slots in the loop form (it walks
// only the LIVE weight/activation/q8 bases + block_index). So the front door wires
// the vestigial scratch slots to the existing weight ABI base (%vx) rather than
// minting placeholder runtime_abi_values -- keeping the exported ggml vec_dot C
// signature the exact 4-role n/s/vx/vy list (byte-identical to the monolith),
// eliminating milestone-2's 3 dead scratch parameters. The brick verifiers relax
// their aux8/aux32 C-type check in the loop form (block_index present) since the
// scratch is emitter-owned there. The integer-core LMUL is left at the emitter
// default (mf2, no integer_core_lmul stamp on the loop op or the scaled-dot brick),
// so the untuned construction lowers byte-identically to the untuned monolith.
void createTypedSuperBlockBlockDotLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed super-block chain: missing block-format fact");
  };
  // The OPTIONAL q5_K qh 5th-bit plane offset: PRESENT in kQ5KFacts (16), ABSENT
  // in kQ4KFacts. Its presence is the ONLY q5_K-vs-q4_K difference the front door
  // stamps -- it flows onto BRICK 1 (the nibble unpack) as weight_qh_byte_offset;
  // the emitter reads it back to set cx.hasQh. Everything else (the 5 bricks, the
  // dual accumulator, the loop op) is format-shared.
  auto factByNameOpt = [&](llvm::StringRef name) -> std::optional<std::int64_t> {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    return std::nullopt;
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32
  std::int64_t weightStride = factByName("weight_block_stride");   // 144 q4/176 q5
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");  // 16 q4/48 q5
  std::optional<std::int64_t> weightQhOffset =
      factByNameOpt("weight_qh_byte_offset");                  //  16 (q5_K only)
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 //   4
  std::int64_t weightDminOffset = factByName("weight_dmin_byte_offset"); //  2
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4
  std::int64_t bsumsOffset = factByName("activation_bsums_byte_offset"); // 260
  std::int64_t numSubBlocks = qk / subBlock;                   //   8

  mlir::MLIRContext *ctx = builder.getContext();
  mlir::Type i32VecType =
      weftrvv::VectorType::get(ctx, builder.getI32Type(), "m1");
  mlir::Type f32M2VecType =
      weftrvv::VectorType::get(ctx, builder.getF32Type(), "m2");

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  loopState.addAttribute(
      "fold_model", builder.getStringAttr("super_block_two_level_scale_min"));
  // integer_core_lmul is LEFT OFF (emitter default mf2) -- byte-identical to the
  // untuned monolith export.
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sums = body.addArgument(f32M2VecType, loc);
  mlir::Value sumf = body.addArgument(builder.getF32Type(), loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK 1: plain 4-bit nibble unpack -> aux8[256] scratch (Region A). Weight base
  // + block_index (per-super-block address vx + ib*144).
  {
    mlir::OperationState s(loc, weftrvv::Q4KNibbleUnpackOp::getOperationName());
    s.addOperands({weight, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_nibble_unpack"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    // q5_K ONLY: stamp the qh 5th-bit plane offset so the emitter injects the 5th
    // bit (cx.hasQh). Absent for q4_K -> byte-identical q4_K unpack.
    if (weightQhOffset)
      s.addAttribute("weight_qh_byte_offset",
                     builder.getI64IntegerAttr(*weightQhOffset));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK 2: 6-bit scale/min bit-dance -> utmp[4] scratch (Region B).
  {
    mlir::OperationState s(loc,
                           weftrvv::Q4KScaleMinBitDanceOp::getOperationName());
    s.addOperands({weight, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_scale_min_bit_dance"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK 3: per-sub-block uint6-scaled i32 dot + fold-back (Region C). The LIVE
  // operand is the q8 activation base (%vy); the aux8/scales scratch slots are
  // vestigial (emitter-owned), so they are wired to the weight base (%vx). q8 lives
  // at yb + activation_quant_byte_offset (4).
  {
    mlir::OperationState s(loc, weftrvv::Q4KScaledDotOp::getOperationName());
    s.addOperands({weight, weight, activation, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_scaled_dot"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // This closed constructor pins the byte-identical deterministic anchor.
    // A later formula may construct a different legal anchor, but emission never
    // interprets an absent field as mf2.
    s.addAttribute("integer_core_lmul", builder.getStringAttr("mf2"));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK 4: MIN term (sumf -= dmin * sum(mins * bsums)) -- the SCALAR sumf chain.
  // The scales scratch slot is vestigial -> wired to the weight base (%vx).
  {
    mlir::OperationState s(loc, weftrvv::Q4KMinTermOp::getOperationName());
    s.addOperands({weight, weight, activation, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_min_term"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("num_sub_blocks", builder.getI64IntegerAttr(numSubBlocks));
    s.addAttribute("bsums_byte_offset", builder.getI64IntegerAttr(bsumsOffset));
    s.addAttribute("weight_dmin_byte_offset",
                   builder.getI64IntegerAttr(weightDminOffset));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK 6: deferred positive fold (sums += d * (float)aux32) -- the 8-lane fp32
  // sums VECTOR chain. The aux32 scratch slot is vestigial -> wired to %vx.
  {
    mlir::OperationState s(loc,
                           weftrvv::Q4KSumsFoldScaleDOp::getOperationName());
    s.addOperands({weight, weight, activation, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_sums_fold_scale_d"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("num_sub_blocks", builder.getI64IntegerAttr(numSubBlocks));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    // A-line g-axis debake (路 B): stamp the canonical fp32 sums lane count as a
    // FORMAT-DEFINED descriptor fact (8), NOT a subBlock/2 derivation (byte-
    // INEXACT for q6_K). The EmitC fold reads coreOp.getNumLanes() fail-closed.
    s.addAttribute("num_lanes", builder.getI64IntegerAttr(8));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // The DUAL carried-out accumulators (sums vector + sumf scalar).
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sums, sumf});
    (void)builder.create(s);
  }
}

// ---------------------------------------------------------------------------
// Typed SUPER-BLOCK SINGLE-accumulator block-dot loop-body construction
// (M-FLAT q6_K milestone-2).
//
// The q6_K sibling of createTypedSuperBlockBlockDotLoopChain. q6_K has NO
// per-block min, so it assembles the OUTER nb = n/QK_K loop as ONE region-carrying
// weft_rvv.typed_super_block_block_dot_loop_body op carrying a SINGLE accumulator
// (the `sums` 8-lane fp32 vector -- no `sumf` scalar), with just TWO in-loop
// bricks inside the region: the q6_K INTEGER CORE
// (weft_rvv.q6_k_q8_k_aux32_partial: the 2-bit qh + 8-bit signed scale unpack into
// the per-super-block aux32[8]) followed by the REUSED no-min positive fold
// (weft_rvv.q4_k_sums_fold_scale_d: sums += fp16(x.d) * y.d * (float)aux32, the
// SAME positive fold q4_K/q5_K use, differing only in the d byte offset 208), then
// a SINGLE yield naming the `sums` vector. It replaces the ONE monolith
// weft_rvv.q6_k_q8_k_block_dot op. fold_model "scales_times_sumi" KEYS the
// single-accumulator arity (the loop op verifier rejects a sumf-carrying yield
// here). Each brick's per-super-block addressing keys off the loop induction
// variable (region arg 0), so the emit is operand-driven (anti-bypass).
//
// The bricks' aux32/output SCRATCH operands are function-scoped scratch the loop
// emitter DECLARES itself (int8_t aux8[256] / the aux32 accumulator vector); it
// never reads these operand slots in the loop form (it walks only the LIVE
// weight/activation bases + block_index). So the front door wires the vestigial
// scratch slots to the existing weight ABI base (%vx) rather than minting
// placeholder runtime_abi_values -- keeping the exported ggml vec_dot C signature
// the exact 4-role n/s/vx/vy list (byte-identical to the monolith). The aux32 op
// and fold brick verifiers relax their output/aux32 C-type check in the loop form
// (block_index present) since the scratch is emitter-owned there. The
// integer-core LMUL is left at the emitter default (mf2, no integer_core_lmul
// stamp), so the untuned construction lowers byte-identically to the untuned
// monolith.
void createTypedSuperBlockScalesTimesSumiLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl,
    RVVBlockDotBodyMechanism mechanism) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed super-block single-accum chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  16 (q6_K/q3_K)
  std::int64_t weightStride = factByName("weight_block_stride");        // 210/110
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 // 192/96
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      // 208/108
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4
  // q6_K and q3_K share this SINGLE-accumulator no-min chain (fold_model
  // "scales_times_sumi"); the ONLY difference is the integer-core BRICK and its
  // weight sub-plane offsets. q6_K reads the qh 5th/6th-bit plane @128; q3_K reads
  // the hmask high-bit plane @0 + the 2-bit qs plane @32. The reused positive fold
  // + single `sums` yield are IDENTICAL.
  const bool isQ3K =
      mechanism == RVVBlockDotBodyMechanism::SuperBlockScalesTimesSumiQ3;
  std::int64_t weightQhOffset = isQ3K ? 0 : factByName("weight_qh_byte_offset");
  std::int64_t weightHmaskOffset =
      isQ3K ? factByName("weight_hmask_byte_offset") : 0;      //   0
  std::int64_t weightQsOffset =
      isQ3K ? factByName("weight_qs_byte_offset") : 0;         //  32

  mlir::MLIRContext *ctx = builder.getContext();
  mlir::Type i32VecType =
      weftrvv::VectorType::get(ctx, builder.getI32Type(), "m1");
  mlir::Type f32M2VecType =
      weftrvv::VectorType::get(ctx, builder.getF32Type(), "m2");

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scales_times_sumi" KEYS the SINGLE-accumulator arity (no min).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scales_times_sumi"));
  // integer_core_lmul is LEFT OFF (emitter default mf2) -- byte-identical to the
  // untuned monolith export.
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sums = body.addArgument(f32M2VecType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the aux32 INTEGER CORE (the per-super-block aux32[8] integer state).
  // The LIVE operands are the weight base (%vx) + activation base (%vy) +
  // block_index; the output int32_t* scratch slot is emitter-owned (vestigial) ->
  // wired to %vx. Per-super-block address vx + ib*stride, vy + ib*292.
  // q6_K: the 2-bit qh + 8-bit signed scale unpack. q3_K: the 2-bit +
  // SUBTRACTIVE-hmask unpack + the SIGNED 6-bit scale dance (its OWN brick op).
  {
    mlir::OperationState s(
        loc, isQ3K ? weftrvv::GgmlBlockDotQ3KQ8KAux32Op::getOperationName()
                   : weftrvv::GgmlBlockDotQ6KQ8KAux32Op::getOperationName());
    s.addOperands({weight, activation, weight, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr(isQ3K ? "ggml_q3_k_q8_k_aux32_partial"
                                               : "ggml_q6_k_q8_k_aux32_partial"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(isQ3K
                                  ? "per-sub-block-int6-signed-scale-i32-domain"
                                  : "per-sub-block-int8-scale-i32-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    if (isQ3K) {
      s.addAttribute("weight_hmask_byte_offset",
                     builder.getI64IntegerAttr(weightHmaskOffset));
      s.addAttribute("weight_qs_byte_offset",
                     builder.getI64IntegerAttr(weightQsOffset));
    } else {
      s.addAttribute("weight_qh_byte_offset",
                     builder.getI64IntegerAttr(weightQhOffset));
    }
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK: the REUSED no-min positive fold (sums += fp16(x.d) * y.d *
  // (float)aux32) -- the 8-lane fp32 `sums` VECTOR chain, NO min term. The LIVE
  // operands are the weight base (%vx, the fp16 d) + activation base (%vy, the
  // fp32 y.d @0) + block_index; the aux32 scratch slot is emitter-owned
  // (vestigial) -> wired to %vx. weight_d_byte_offset flows from entry.facts
  // (block_q6_K d @208 / block_q3_K d @108); sub_block/num_sub_blocks carry the
  // fold's fixed 8-lane facts (32/8), which the 8-lane fp fold does not read.
  {
    mlir::OperationState s(loc,
                           weftrvv::Q4KSumsFoldScaleDOp::getOperationName());
    s.addOperands({weight, weight, activation, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_sums_fold_scale_d"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(32));
    s.addAttribute("num_sub_blocks", builder.getI64IntegerAttr(8));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    // A-line g-axis debake (路 B): stamp the canonical fp32 sums lane count as a
    // FORMAT-DEFINED descriptor fact (8). The EmitC fold reads
    // coreOp.getNumLanes() fail-closed (no baked default).
    s.addAttribute("num_lanes", builder.getI64IntegerAttr(8));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // The SINGLE carried-out accumulator (the `sums` vector ONLY -- no sumf).
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sums});
    (void)builder.create(s);
  }
}

// ---------------------------------------------------------------------------
// Typed SUPER-BLOCK SCALAR-accumulator block-dot loop-body construction
// (M-FLAT q2_K milestone-2).
//
// The q2_K sibling of createTypedSuperBlockScalesTimesSumiLoopChain. q2_K HAS a
// per-block min (like q4_K/q5_K) but its whole fold is a SINGLE per-super-block
// SCALAR `sumf += dall*isum - dmin*summs` (isum/summs being the two SCALAR
// integer states of the q2_K integer core), NOT an 8-lane deferred vector. So it
// assembles the OUTER nb = n/QK_K loop as ONE region-carrying
// weft_rvv.typed_super_block_block_dot_loop_body op carrying a SINGLE `sumf`
// SCALAR accumulator (region args (index, sumf:f32)), with just ONE in-loop brick
// inside the region: the q2_K INTEGER CORE
// (weft_rvv.q2_k_q8_k_integer_core: the 2-bit unpack + PLAIN uint4-nibble scale/
// min + per-sub-block scalar i32 dot producing the two scalar states isum +
// summs), then a SINGLE yield naming the `sumf` scalar. The scalar fold itself
// (dall*isum - dmin*summs, fp16 d@80/dmin@82) has NO separate fold brick -- its
// offsets are FIXED block_q2_K constants, so the SCALAR-accumulator lowering
// emitter inlines it. It replaces the ONE monolith weft_rvv.q2_k_q8_k_block_dot
// op. fold_model "scalar_scale_min" KEYS the scalar-accumulator arity (the loop
// op verifier rejects an 8-lane `sums` vector region here). The brick's
// per-super-block addressing keys off the loop induction variable (region arg 0),
// so the emit is operand-driven (anti-bypass).
//
// The integer-core LMUL is NOT stamped (q2_K carries no shape knob -- its dot is
// FIXED at e8m1/i16m2/i32m1 in the emitter, matching the untuned monolith), so
// the untuned construction lowers byte-identically to the untuned monolith.
void createTypedSuperBlockScalarScaleMinLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed super-block scalar-accum chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  16 (q2_K)
  std::int64_t weightStride = factByName("weight_block_stride");        //  84
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //  16
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4
  std::int64_t activationBsumsOffset =
      factByName("activation_bsums_byte_offset");              // 260

  mlir::MLIRContext *ctx = builder.getContext();
  (void)ctx;
  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_scale_min" KEYS the SCALAR-accumulator arity (q2_K).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_scale_min"));
  // integer_core_lmul is LEFT OFF (q2_K carries no shape knob; the emitter's
  // fixed e8m1/i16m2/i32m1 dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the q2_K INTEGER CORE (the 2-bit unpack + PLAIN uint4-nibble scale/min
  // + per-sub-block scalar i32 dot). The LIVE operands are the weight base (%vx) +
  // activation base (%vy) + n + vl + block_index; it produces the two SCALAR i32
  // states isum + summs (NO output pointer -- q2_K's states are scalar registers,
  // unlike q6_K's aux32[8] memory state). Per-super-block address vx + ib*84,
  // vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotQ2KQ8KIntegerCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_q2_k_q8_k_integer_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-sub-block-uint4-scale-i32-domain-min"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addAttribute("activation_bsums_byte_offset",
                   builder.getI64IntegerAttr(activationBsumsOffset));
    s.addTypes({i32ScalarType, i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector, no second min-term operand).
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

} // namespace weft::plugin::rvv
