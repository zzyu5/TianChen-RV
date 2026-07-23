//===- RVVGridCodebookBlockDotBodyConstruction.cpp ---------------------===//
//
// Grid, codebook, and ternary super-block integer-core body mechanisms sharing
// the scalar-delta fold topology.
//
//===----------------------------------------------------------------------===//

#include "RVVBlockDotBodyConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace weft::plugin::rvv {

namespace weftrvv = ::weft::rvv;

// The iq1_s sibling of createTypedSuperBlockScalarScaleMinLoopChain. iq1_s is a
// super-block quant whose per-sub-block decode is a codebook GRID GATHER
// (decode_model=lookup -- the 2048-entry TERNARY iq1s_grid + a vluxei16 gather),
// NOT an arithmetic bit-unpack, but its whole fold is a SINGLE per-super-block
// SCALAR `sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)` -- the SAME scalar-
// accumulator arity as q2_K, with sumi (the qh-scaled POSITIVE ternary-grid dot) and
// sumi1 (the iq1_s DELTA-bsum integer sum) being the two SCALAR i32 states of the
// iq1_s grid core. So it assembles the OUTER nb = n/QK_K loop as ONE region-carrying
// weft_rvv.typed_super_block_block_dot_loop_body op carrying a SINGLE `sumf` SCALAR
// accumulator (region args (index, sumf:f32)), with just ONE in-loop brick inside
// the region: the iq1_s TERNARY-grid INTEGER CORE
// (weft_rvv.iq1_s_q8_k_grid_core: the 11-bit grid index build from qs+qh + the
// vluxei16 ternary-grid gather + the signed widening grid dot + the qh-encoded
// per-sub-block scale + the delta-bsum sum, producing the two scalar states sumi +
// sumi1), then a SINGLE yield naming the `sumf` scalar. The scalar delta fold itself
// (d*((float)sumi + 0.125f*(float)sumi1), fp16 x.d @0 / fp32 y.d @0) has NO separate
// fold brick -- its offsets are FIXED block_iq1_s / block_q8_K constants, so the
// SCALAR-accumulator GRID lowering emitter inlines it. It replaces the ONE monolith
// weft_rvv.iq1_s_q8_k_block_dot op (RETIRED the SAME action as the flip). fold_model
// "scalar_delta_grid" KEYS the scalar-accumulator arity AND the iq1_s grid emitter
// (the loop op verifier rejects an 8-lane `sums` vector region here; the emitter
// dispatch keys the grid gather off this fold_model). The brick's per-super-block
// addressing keys off the loop induction variable (region arg 0), so the emit is
// operand-driven (anti-bypass).
//
// The integer-core LMUL is NOT stamped (iq1_s carries no shape knob -- its grid dot
// is FIXED at the emitter's vluxei16/i8m2/i16m4/i32m1 anchor, matching the untuned
// monolith), so the untuned construction lowers byte-identically to the untuned
// monolith.
void createTypedSuperBlockScalarDeltaGridLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed super-block scalar-delta-grid chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq1_s)
  std::int64_t weightStride = factByName("weight_block_stride");        //  50
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  34
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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the iq1_s
  // GRID emitter (vs q2_K's arithmetic "scalar_scale_min").
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq1_s carries no shape knob; the emitter's fixed
  // vluxei16/i8m2 grid dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq1_s TERNARY-grid INTEGER CORE (the 11-bit grid index build from
  // qs+qh + the vluxei16 ternary-grid gather + the signed widening grid dot + the
  // qh-encoded per-sub-block scale + the delta-bsum sum). The LIVE operands are the
  // weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces the
  // two SCALAR i32 states sumi + sumi1 (NO output pointer -- iq1_s's states are scalar
  // registers, like q2_K's). Per-super-block address vx + ib*50, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ1SQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq1_s_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(
            "per-sub-block-qh-scale-ternary-grid-codebook-delta-bsum-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addAttribute("activation_bsums_byte_offset",
                   builder.getI64IntegerAttr(activationBsumsOffset));
    // A-line g-axis debake (路 B): stamp the iq1_s grid group count as a
    // FORMAT-DEFINED descriptor fact (4 grid groups per sub-block), so the EmitC
    // grid loop reads coreOp.getGroupsPerSub() instead of a baked literal.
    s.addAttribute("groups_per_sub", builder.getI64IntegerAttr(4));
    s.addTypes({i32ScalarType, i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector, no second delta-term operand; the scalar delta fold is
  // emitter-inlined).
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq1_m sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the C2
// marginal-cost payoff. iq1_m REUSES the WHOLE iq1_s super-block SCALAR-accumulator
// GRID scaffold (the SAME loop op weft_rvv.typed_super_block_block_dot_loop_body with
// fold_model "scalar_delta_grid", the SAME single `sumf` scalar accumulator + region
// (index, sumf:f32) contract, the SAME selector SuperBlockScalarDeltaGrid, the SAME
// emitter dispatch, the SAME single-yield): the ONLY marginal cost is a DISTINCT
// in-loop brick -- the iq1_m TERNARY-grid integer core
// (weft_rvv.iq1_m_q8_k_grid_core, producing the two scalar states sumi1 (the qh-index
// half-scaled grid dot) + sumi2 (the per-group four-sign delta sum)) -- because
// iq1_m's integer decode is structurally different (a packed-scale fp16 reconstruct,
// TWO per-sub-block half scales ls1/ls2, a half-split per-half grid dot, a per-group
// FRESH Σq8 delta with FOUR independent signs, and NO bsums). The scalar delta fold
// itself (d*((float)sumi1 + IQ1M_DELTA*(float)sumi2), IQ1M_DELTA=0.125f, the packed
// iq1m_scale fp16 reconstruct + fp32 y.d @0) has NO separate fold brick -- the
// SCALAR-accumulator GRID lowering emitter inlines it, keyed off the iq1_m brick
// identity. It resolves to iq1_m's OWN export entry by fold_model + weight_block_stride
// 56 (vs iq1_s 50). The brick's per-super-block addressing keys off the loop induction
// variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq1M(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq1_m chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq1_m)
  std::int64_t weightStride = factByName("weight_block_stride");        //  56
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   0
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  32
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 //  48
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid
  // emitter; weight_block_stride 56 disambiguates iq1_m from iq1_s (50).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq1_m carries no shape knob; the emitter's fixed
  // vluxei16/i8m1/i16m2 grid dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq1_m TERNARY-grid INTEGER CORE (the packed iq1m_scale fp16 reconstruct
  // + the per-half vluxei16 grid dot with two half scales ls1/ls2 + the per-group
  // four-sign delta via a fresh Σq8). The LIVE operands are the weight base (%vx) +
  // activation base (%vy) + n + vl + block_index; it produces the two SCALAR i32
  // states sumi1 + sumi2 (NO output pointer -- iq1_m's states are scalar registers,
  // like iq1_s's). Per-super-block address vx + ib*56, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ1MQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq1_m_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(
            "packed-iq1m-scale-per-half-scale-ternary-grid-codebook-per-group-"
            "delta-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq1_m grid group count as a
    // FORMAT-DEFINED descriptor fact (4 grid groups per sub-block).
    s.addAttribute("groups_per_sub", builder.getI64IntegerAttr(4));
    s.addTypes({i32ScalarType, i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector, no second delta-term operand; the scalar delta fold is
  // emitter-inlined). IDENTICAL to iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq3_xxs sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the L3
// coverage payoff. iq3_xxs is a super-block GRID/codebook quant whose whole fold is
// the SAME SINGLE per-super-block SCALAR accumulator arity as iq1_s (fold_model
// "scalar_delta_grid", single `sumf` scalar, emitter-inlined fold), REUSING the whole
// iq1_s super-block SCALAR-accumulator GRID scaffold (the SAME loop op
// weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield contract, the
// SAME selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY
// marginal cost is a DISTINCT in-loop brick -- the iq3_xxs GRID-of-4 integer core
// (weft_rvv.iq3_xxs_q8_k_grid_core, producing the ONE scalar state bsum) -- because
// iq3_xxs's decode is structurally different from iq1: the 256-entry uint32
// iq3xxs_grid GRID-of-4 gathered via vluxei16_v_i32m1 (vs iq1_s's uint64 grid-of-8),
// the per-sign-group ksigns_iq2xs SIGN plane (per-group 7-bit selectors), the
// per-sub-block aux32 4-bit scale ls, and the trailing 0.25f factor. The scalar fold
// itself (`sumf += d*(float)bsum` then `*s = 0.25f*sumf`, fp16 x.d @0 / fp32 y.d @0)
// has NO separate fold brick -- the SCALAR-accumulator GRID lowering emitter inlines
// it, keyed off the iq3_xxs brick identity. It resolves to iq3_xxs's OWN export entry
// by fold_model + weight_block_stride 98 (vs iq1_s 50, iq1_m 56). The brick's
// per-super-block addressing keys off the loop induction variable (region arg 0), so
// the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq3_xxs chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq3_xxs)
  std::int64_t weightStride = factByName("weight_block_stride");        //  98
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightGasOffset = factByName("weight_gas_byte_offset");  //  66
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid
  // emitter; weight_block_stride 98 disambiguates iq3_xxs from iq1_s (50)/iq1_m (56).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq3_xxs carries no shape knob; the emitter's fixed
  // vluxei16/i8m1/i16m2 grid-of-4 dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq3_xxs GRID-of-4 INTEGER CORE (the aux32 4-bit-scale + 4-sign-group
  // ksigns decode + the two-index-per-group vluxei16_v_i32m1 grid gather + the signed
  // widening grid dot + the per-sub-block scale fold into bsum). The LIVE operands are
  // the weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces
  // the ONE SCALAR i32 state bsum (NO output pointer -- iq3_xxs's bsum is a scalar
  // register, like iq1_s's states). Per-super-block address vx + ib*98, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ3XXSQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq3_xxs_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-sub-block-aux32-scale-grid-of-4-codebook-"
                              "ksigns-sign-plane-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_gas_byte_offset",
                   builder.getI64IntegerAttr(weightGasOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq3_xxs grid sub-structure counts as
    // FORMAT-DEFINED descriptor facts (4 sign groups per sub-block, 8 grid index
    // bytes per sub-block, 8 grid lanes per sign group).
    s.addAttribute("num_groups", builder.getI64IntegerAttr(4));
    s.addAttribute("indices_per_sub_block", builder.getI64IntegerAttr(8));
    s.addAttribute("group_lanes", builder.getI64IntegerAttr(8));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector; the scalar fold + trailing 0.25f are emitter-inlined). IDENTICAL to
  // iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq3_s sibling of createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs -- the
// C_construct 22->23 payoff (EXPLICIT-SIGNS variant). iq3_s is the iq3_xxs GRID-of-4
// sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
// per-super-block SCALAR accumulator arity as iq1_s/iq3_xxs (fold_model
// "scalar_delta_grid", single `sumf` scalar, emitter-inlined fold), REUSING the whole
// iq1_s super-block SCALAR-accumulator GRID scaffold (the SAME loop op
// weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield contract, the SAME
// selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal cost
// is a DISTINCT in-loop brick -- the iq3_s GRID-of-4 explicit-signs integer core
// (weft_rvv.iq3_s_q8_k_grid_core, producing the ONE scalar state bsum) -- because iq3_s's
// decode swaps THREE mechanisms from iq3_xxs to iq2_s: the 512-entry uint32 iq3s_grid
// GRID-of-4 gathered via vluxei16_v_i32m1 (a LARGER 9-bit-index table than iq3_xxs's
// 256-entry 8-bit one), the qh 9th-bit inject (mask 256, the two passes taking shifts
// 8-2l and 7-2l), the per-lane sign read from an EXPLICIT per-sub-block signs region (at
// weight offset 74, NO ksigns plane), and the per-sub-block scale from the EXPLICIT
// two-nibble scales[] (at weight offset 106). The scalar fold itself (`sumf +=
// d*(float)bsum` then `*s = sumf` -- NO trailing factor -- fp16 x.d @0 / fp32 y.d @0) has
// NO separate fold brick -- the SCALAR-accumulator GRID lowering emitter inlines it, keyed
// off the iq3_s brick identity. It resolves to iq3_s's OWN export entry by fold_model +
// weight_block_stride 110 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74, iq2_s
// 82). The brick's per-super-block addressing keys off the loop induction variable (region
// arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq3s(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq3_s chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq3_s)
  std::int64_t weightStride = factByName("weight_block_stride");        // 110
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  66
  std::int64_t weightSignsOffset =
      factByName("weight_signs_byte_offset");                  //  74
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 // 106
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid emitter;
  // weight_block_stride 110 disambiguates iq3_s from iq1_s (50)/iq1_m (56)/iq3_xxs (98)/
  // iq2_xxs (66)/iq2_xs (74)/iq2_s (82).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq3_s carries no shape knob; the emitter's fixed
  // vluxei16/i8m1/i16m2 grid-of-4 dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq3_s GRID-of-4 EXPLICIT-SIGNS INTEGER CORE (the explicit two-nibble scale
  // + qh 9th-bit inject + explicit per-sub-block signs decode + the two-index-per-group
  // vluxei16_v_i32m1 grid gather + the signed widening grid dot + the per-sub-block scale
  // fold into bsum). The LIVE operands are the weight base (%vx) + activation base (%vy) +
  // n + vl + block_index; it produces the ONE SCALAR i32 state bsum (NO output pointer --
  // iq3_s's bsum is a scalar register, like iq3_xxs's states). Per-super-block address vx +
  // ib*110, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ3SQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq3_s_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-sub-block-explicit-scale-grid-of-4-codebook-"
                              "qh-plane-explicit-signs-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("weight_signs_byte_offset",
                   builder.getI64IntegerAttr(weightSignsOffset));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq3_s grid sub-structure counts as
    // FORMAT-DEFINED descriptor facts (4 sign groups per sub-block, 8 grid index
    // bytes per sub-block, 4 explicit sign bytes per sub-block, 8 grid lanes per
    // sign group).
    s.addAttribute("num_groups", builder.getI64IntegerAttr(4));
    s.addAttribute("indices_per_sub_block", builder.getI64IntegerAttr(8));
    s.addAttribute("signs_per_sub_block", builder.getI64IntegerAttr(4));
    s.addAttribute("group_lanes", builder.getI64IntegerAttr(8));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector; the scalar fold is emitter-inlined, and iq3_s applies NO trailing
  // factor). IDENTICAL to iq1_s/iq3_xxs's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq4_xs sibling of the scalar-delta-grid chain builders -- the C_construct 23->24
// payoff (the FIRST super-block CODEBOOK member vs the grid siblings). iq4_xs is the
// SUPER-BLOCK rung of the flat iq4_nl codebook -- a super-block CODEBOOK quant whose whole
// fold is the SAME SINGLE per-super-block SCALAR accumulator arity as iq1_s/iq3_s
// (fold_model "scalar_delta_grid", single `sumf` scalar, emitter-inlined fold), REUSING the
// whole iq1_s super-block SCALAR-accumulator scaffold (the SAME loop op
// weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield contract, the SAME
// selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal cost is
// a DISTINCT in-loop brick -- the iq4_xs CODEBOOK integer core
// (weft_rvv.iq4_xs_q8_k_codebook_core) -- because iq4_xs's decode swaps the grid gather for
// iq4_nl's 16-entry non-linear int8 CODEBOOK gather (the SAME kvalues_iq4nl[16] vrgather
// lookup iq4_nl uses, carried as a DenseI8ArrayAttr:$codebook) wrapped in the q4_K-style
// super-block SIGNED 6-bit scale bit-dance (per-sub-block ls = ((scales_l>>...)&0xf) |
// (((scales_h>>...)&0x3)<<4) biased -32). UNLIKE the grid siblings iq4_xs's fold runs
// PER-SUB-BLOCK in float (`sumf += (d4d8*(ls-32))*sumi`, 8 fp folds per super-block, NO
// trailing factor), but the single-scalar accumulator arity is identical, so the whole
// per-super-block body is emitter-inlined keyed off the codebook-core brick identity. It
// resolves to iq4_xs's OWN export entry by fold_model + weight_block_stride 136 (vs iq1_s
// 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74, iq2_s 82, iq3_s 110). The brick carries
// NO gearbox (the codebook gather pins m1). The brick's per-super-block addressing keys off
// the loop induction variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq4xs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq4_xs chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32
  std::int64_t weightStride = factByName("weight_block_stride");        // 136
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightScalesHOffset =
      factByName("weight_scales_h_byte_offset");                //   2
  std::int64_t weightScalesLOffset =
      factByName("weight_scales_l_byte_offset");                //   4
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   8
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                   //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");               //   4

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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid/codebook
  // emitter dispatch; weight_block_stride 136 disambiguates iq4_xs from iq1_s (50)/iq1_m
  // (56)/iq3_xxs (98)/iq2_xxs (66)/iq2_xs (74)/iq2_s (82)/iq3_s (110).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq4_xs carries no shape knob; the codebook gather pins
  // m1, and the emitter's fixed vrgather/i8m1/i16m2 codebook dot matches the untuned
  // monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq4_xs CODEBOOK INTEGER CORE (the 16-entry non-linear int8 codebook
  // broadcast + the per-sub-block nibble split + vrgather decode + asymmetric vwmul/vwmacc
  // widening product + seed-0 vwredsum, wrapped in the q4_K-style super-block signed 6-bit
  // scale bit-dance + the per-sub-block float fold into sumf). The LIVE operands are the
  // weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces ONE scalar
  // i32 SSA result (an UNUSED per-super-block placeholder -- iq4_xs's fold is per-sub-block
  // float, no single scalar state). Per-super-block address vx + ib*136, vy + ib*292. The
  // 16-entry codebook (kvalues_iq4nl[16]) is carried as a DenseI8ArrayAttr like the monolith.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ4XSQ8KCodebookCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq4_xs_q8_k_codebook_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(
            "per-sub-block-signed-6bit-scale-codebook-gather-float-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_scales_h_byte_offset",
                   builder.getI64IntegerAttr(weightScalesHOffset));
    s.addAttribute("weight_scales_l_byte_offset",
                   builder.getI64IntegerAttr(weightScalesLOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addAttribute("codebook", builder.getDenseI8ArrayAttr(entry.codebook));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane `sums`
  // vector; the per-sub-block float fold is emitter-inlined, and iq4_xs applies NO trailing
  // factor). IDENTICAL to iq1_s/iq3_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The tq2_0 sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the FIRST TQ-family
// member (C_construct 24->25). tq2_0 is the 2-bit TERNARY ({-1,0,+1}) TriLM K-quant whose
// whole fold is the SAME SINGLE per-super-block SCALAR accumulator arity as iq1_s/iq3_s
// (fold_model "scalar_delta_grid", single `sumf` scalar, emitter-inlined fold), REUSING the
// whole iq1_s super-block SCALAR-accumulator scaffold (the SAME loop op
// weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield contract, the SAME
// selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal cost is a
// DISTINCT in-loop brick -- the tq2_0 FUSED 2-bit TERNARY integer core
// (weft_rvv.tq2_0_q8_k_ternary_core) -- because tq2_0's decode is ARITHMETIC (q2_K's 2-bit
// `(qs>>shift)&3` unpack + the per-element `-1` ternary bias, NO grid/codebook gather). Its
// whole fold is a single per-super-block scalar `sumf += (float)sumi * d` with `d = fp16(x.d
// @64) * y.d @0` and NO trailing factor, so the whole per-super-block body is emitter-inlined
// keyed off the ternary-core brick identity. It shares weight_block_stride 66 with iq2_xxs but
// dispatches by its OWN DISTINCT brick op type. UNLIKE the iq4_xs codebook sibling the brick
// PRESERVES tq2_0's Win-A integer_core_lmul m2/m1 gearbox (kernel key "tq2_0", left attr-less
// at construction = the default m2 anchor). The brick's per-super-block addressing keys off
// the loop induction variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainTq20(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid tq2_0 chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t weightStride = factByName("weight_block_stride");        //  66
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   0
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //  64
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                   //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");               //   4

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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the emitter dispatch;
  // the emitter disambiguates tq2_0 from the iq* siblings (and from iq2_xxs which shares
  // stride 66) by the in-region ternary-core brick op TYPE.
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF here (attr-less construction = the default m2 anchor, the
  // byte-exact CORE target); tq2_0's Win-A gearbox lives on the ternary-core brick below and
  // is refined m2->m1 at VLEN>=256 by the separate schedule pass.
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the tq2_0 FUSED 2-bit TERNARY INTEGER CORE (the 32-byte qs chunk load + the 4
  // 2-bit planes each unpacked to 32 ternary lanes via vand/vsrl + the `-1` bias vsub and
  // vwmacc'd DIRECTLY against the matching 32 q8 lanes into a wide i16 accumulator + ONE
  // vwredsum per chunk into the per-super-block scalar sumi). The LIVE operands are the
  // weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces ONE scalar
  // i32 SSA result (the per-super-block sumi placeholder -- the emitter re-emits the whole
  // body including the fold). Per-super-block address vx + ib*66, vy + ib*292. Left attr-less
  // so the gearbox is free to stamp integer_core_lmul m2/m1 from the VLEN capability fact.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotTQ20Q8KTernaryCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_tq2_0_q8_k_ternary_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(
            "ternary-2bit-fused-plane-single-fp16-scale-i32-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane `sums`
  // vector; the scalar fold `sumf += (float)sumi * d` is emitter-inlined, and tq2_0 applies
  // NO trailing factor). IDENTICAL to iq1_s/iq3_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The tq1_0 sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the SECOND TQ-family
// member (C_construct 25->26), the base-3-packed sibling of tq2_0. tq1_0 is the BASE-3
// TERNARY ({-1,0,+1}) TriLM K-quant whose whole fold is the SAME SINGLE per-super-block
// SCALAR accumulator arity as tq2_0/iq1_s (fold_model "scalar_delta_grid", single `sumf`
// scalar, emitter-inlined fold), REUSING the WHOLE tq2_0 ternary scaffold at C2 marginal cost
// (the SAME loop op weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield
// contract, the SAME selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY
// marginal cost is a DISTINCT in-loop brick -- the tq1_0 BASE-3 TERNARY integer core
// (weft_rvv.tq1_0_q8_k_ternary_core) -- because tq1_0's decode is base-3 (the `q =
// (uint8_t)(byte*pow3[l]); xi = ((uint16_t)q*3)>>8; xi-1` power-of-three trit unpack over the
// qs+qh weight arrays, NOT tq2_0's `(qs>>shift)&3` 2-bit field). Its whole fold is a single
// per-super-block scalar `sumf += (float)sumi * d` with `d = fp16(x.d @52) * y.d @0` and NO
// trailing factor, so the whole per-super-block body is emitter-inlined keyed off the
// ternary-core brick identity. It resolves to its OWN export entry by fold_model +
// weight_block_stride 54 (UNIQUE among the scalar_delta_grid bricks -- tq2_0/iq2_xxs are 66),
// so NO stride tie-breaker is needed. Unlike tq2_0, the realized tq1_0 brick is
// VLEN-universal and has no inert LMUL schedule field. The brick's per-super-block addressing keys off the loop induction
// variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainTq10(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid tq1_0 chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t weightStride = factByName("weight_block_stride");        //  54
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   0
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  48
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //  52
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                   //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");               //   4

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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the emitter dispatch;
  // the emitter disambiguates tq1_0 from the iq* / tq2_0 siblings by the in-region base-3
  // ternary-core brick op TYPE (and tq1_0's stride 54 is UNIQUE among these bricks).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // The tq1_0 ternary-core brick below has one fixed realized vector body.
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the tq1_0 BASE-3 TERNARY INTEGER CORE (the three base-3 unpack regions -- qs main
  // + qs tail + qh -- each `q=(uint8_t)(byte*pow3[l]); xi=((uint16_t)q*3)>>8; xi-1` decoded
  // into an element-ordered aux8[256], then the flat-256 widened i8*i8 dot into the
  // per-super-block scalar sumi). The LIVE operands are the weight base (%vx) + activation
  // base (%vy) + n + vl + block_index; it produces ONE scalar i32 SSA result (the
  // per-super-block sumi placeholder -- the emitter re-emits the whole body including the
  // fold). Per-super-block address vx + ib*54, vy + ib*292. Left attr-less so the gearbox is
  // free to stamp integer_core_lmul m2/m1 from the VLEN capability fact.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotTQ10Q8KTernaryCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_tq1_0_q8_k_ternary_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("ternary-base3-single-fp16-scale-i32-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane `sums`
  // vector; the scalar fold `sumf += (float)sumi * d` is emitter-inlined, and tq1_0 applies
  // NO trailing factor). IDENTICAL to tq2_0/iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq2_xxs sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the L3 coverage
// payoff (SIGN-PLANE signs64 variant). iq2_xxs is another iq1_s grid sibling -- a
// super-block GRID/codebook quant whose whole fold is the SAME SINGLE per-super-block
// SCALAR accumulator arity as iq1_s (fold_model "scalar_delta_grid", single `sumf`
// scalar, emitter-inlined fold), REUSING the whole iq1_s super-block SCALAR-accumulator
// GRID scaffold (the SAME loop op, the SAME single-yield contract, the SAME selector
// SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal cost is a
// DISTINCT in-loop brick -- the iq2_xxs GRID-of-8 integer core
// (weft_rvv.iq2_xxs_q8_k_grid_core, producing the ONE scalar state bsum) -- because
// iq2_xxs's decode is structurally different: the 256-entry uint64 iq2xxs_grid GRID-of-8
// gathered via vluxei16_v_i64<core> (vs iq3_xxs's uint32 grid-of-4), the SIGN read via a
// SECOND vluxei16 gather over the DERIVED keven_signs_q2xs signs64 sign plane (vs
// iq3_xxs's per-group scalar ksigns fold), the per-sub-block aux1 4-bit scale ls, and the
// trailing 0.125f factor. The scalar fold itself (`sumf += d*(float)bsum` then `*s =
// 0.125f*sumf`, fp16 x.d @0 / fp32 y.d @0) has NO separate fold brick -- the
// SCALAR-accumulator GRID lowering emitter inlines it, keyed off the iq2_xxs brick
// identity. It resolves to iq2_xxs's OWN export entry by fold_model + weight_block_stride
// 66 (vs iq1_s 50, iq1_m 56, iq3_xxs 98). The brick carries the SAME Win-A
// integer_core_lmul gearbox (kernel key "iq2_xxs") so the m2->m1 VLEN capability
// selection is preserved on the constructed op. The brick's per-super-block addressing
// keys off the loop induction variable (region arg 0), so the emit is operand-driven
// (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq2xxs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq2_xxs chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq2_xxs)
  std::int64_t weightStride = factByName("weight_block_stride");        //  66
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid emitter;
  // weight_block_stride 66 disambiguates iq2_xxs from iq1_s (50)/iq1_m (56)/iq3_xxs (98).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF here (the constructed brick lowers at the emitter's m2
  // default = the retired monolith's byte-exact default; the unified autotuner REFINES
  // m2->m1 at VLEN256 by stamping the brick -- the Win-A gearbox, preserved).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq2_xxs GRID-of-8 INTEGER CORE (the aux1 4-bit-scale + 4-sign-group decode
  // + the 4-index vluxei16_v_i64<core> grid gather + the SECOND signs64 vluxei16 gather +
  // the vmul-onto-grid sign fold + the signed widening grid dot + the per-sub-block scale
  // fold into bsum). The LIVE operands are the weight base (%vx) + activation base (%vy) +
  // n + vl + block_index; it produces the ONE SCALAR i32 state bsum (NO output pointer --
  // iq2_xxs's bsum is a scalar register). Per-super-block address vx + ib*66, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ2XXSQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq2_xxs_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-sub-block-aux1-scale-grid-of-8-codebook-"
                              "signs64-sign-plane-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq2_xxs grid sign-group count as a
    // FORMAT-DEFINED descriptor fact (4 sign groups per sub-block).
    s.addAttribute("num_groups", builder.getI64IntegerAttr(4));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane `sums`
  // vector; the scalar fold + trailing 0.125f are emitter-inlined). IDENTICAL to iq1_s's
  // yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq2_xs sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the L3 coverage
// payoff (SIGN-PLANE signs64 variant, PER-HALF explicit scale). iq2_xs is the iq2_xxs grid
// sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
// per-super-block SCALAR accumulator arity as iq1_s (fold_model "scalar_delta_grid", single
// `sumf` scalar, emitter-inlined fold), REUSING the whole iq1_s super-block
// SCALAR-accumulator GRID scaffold (the SAME loop op, the SAME single-yield contract, the
// SAME selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal
// cost is a DISTINCT in-loop brick -- the iq2_xs per-half-explicit-scale GRID integer core
// (weft_rvv.iq2_xs_q8_k_grid_core, producing the ONE scalar state bsum) -- because iq2_xs's
// decode is structurally different from iq2_xxs: the 512-entry uint64 iq2xs_grid indexed by
// the 9-bit `w & 511` of each uint16 qs word (vs iq2_xxs's 256-entry aux1-interleaved
// grid), the SIGN read via a SECOND vluxei16 gather over the DERIVED keven_signs_q2xs
// signs64 sign plane keyed by the 7-bit `w >> 9` selector, and -- the load-bearing delta --
// the EXPLICIT per-sub-block 4-bit scales[8] byte stream splitting each sub-block into TWO
// 16-lane HALVES with distinct scales ls1/ls2 (vs iq2_xxs's single aux1 scale). The scalar
// fold itself (`sumf += d*(float)bsum` then `*s = 0.125f*sumf`, fp16 x.d @0 / fp32 y.d @0)
// has NO separate fold brick -- the SCALAR-accumulator GRID lowering emitter inlines it,
// keyed off the iq2_xs brick identity. It resolves to iq2_xs's OWN export entry by
// fold_model + weight_block_stride 74 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66). The
// brick's per-super-block addressing keys off the loop induction variable (region arg 0),
// so the emit is operand-driven (anti-bypass). UNLIKE iq2_xxs the brick carries NO
// integer_core_lmul gearbox (fixed 16-lane per-half shape, not in any autotuner).
void createTypedSuperBlockScalarDeltaGridLoopChainIq2xs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq2_xs chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq2_xs)
  std::int64_t weightStride = factByName("weight_block_stride");        //  74
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                  //  66
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid emitter;
  // weight_block_stride 74 disambiguates iq2_xs from iq1_s (50)/iq1_m (56)/iq3_xxs (98)/
  // iq2_xxs (66).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq2_xs per-half-explicit-scale GRID INTEGER CORE (the per-sub-block explicit
  // 4-bit scale byte -> ls1/ls2 + the two-half 16-lane decode + the 2-index vluxei16_v_i64m1
  // grid gather + the SECOND signs64 vluxei16 gather + the vmul-onto-grid sign fold + the
  // signed widening grid dot + the per-half scale fold into bsum). The LIVE operands are the
  // weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces the ONE
  // SCALAR i32 state bsum (NO output pointer -- iq2_xs's bsum is a scalar register).
  // Per-super-block address vx + ib*74, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ2XSQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq2_xs_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-half-int4-explicit-scales-grid-codebook-"
                              "signs64-sign-plane-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq2_xs per-16-lane-half group count
    // as a FORMAT-DEFINED descriptor fact (2 grid/sign groups per 16-lane half).
    s.addAttribute("num_groups_per_half", builder.getI64IntegerAttr(2));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY). IDENTICAL to
  // iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq2_s sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the L3 coverage
// payoff (SIGN-PLANE explicit-signs variant, PER-HALF explicit scale). iq2_s is the iq2_xs
// grid sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
// per-super-block SCALAR accumulator arity as iq1_s (fold_model "scalar_delta_grid", single
// `sumf` scalar, emitter-inlined fold), REUSING the whole iq1_s super-block
// SCALAR-accumulator GRID scaffold (the SAME loop op, the SAME single-yield contract, the
// SAME selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal
// cost is a DISTINCT in-loop brick -- the iq2_s per-half-explicit-scale GRID integer core
// (weft_rvv.iq2_s_q8_k_grid_core, producing the ONE scalar state bsum) -- because iq2_s's
// decode is structurally different from iq2_xs: the 1024-entry uint64 iq2s_grid indexed by
// the 10-bit `qs[l] | ((qh<<(8-2l))&0x300)` (a single index byte + 2 qh-plane bits, vs
// iq2_xs's 9-bit `w & 511` of a uint16 qs word), the SIGN read via a SECOND vluxei16 gather
// over the UNIVERSAL signs256 explicit-sign-byte plane keyed by the RAW sign byte read
// DIRECTLY from the sign region at qs+32 (vs iq2_xs's DERIVED keven_signs_q2xs plane keyed
// by `w >> 9`), and -- shared with iq2_xs -- the EXPLICIT per-sub-block 4-bit scales[8]
// byte stream splitting each sub-block into TWO 16-lane HALVES with distinct scales ls1/ls2.
// The scalar fold itself (`sumf += d*(float)bsum` then `*s = 0.125f*sumf`, fp16 x.d @0 /
// fp32 y.d @0) has NO separate fold brick -- the SCALAR-accumulator GRID lowering emitter
// inlines it, keyed off the iq2_s brick identity. It resolves to iq2_s's OWN export entry by
// fold_model + weight_block_stride 82 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs
// 74). The brick's per-super-block addressing keys off the loop induction variable (region
// arg 0), so the emit is operand-driven (anti-bypass). Like iq2_xs the brick carries NO
// integer_core_lmul gearbox (fixed 16-lane per-half shape, not in any autotuner).
void createTypedSuperBlockScalarDeltaGridLoopChainIq2s(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq2_s chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq2_s)
  std::int64_t weightStride = factByName("weight_block_stride");        //  82
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightSignsOffset =
      factByName("weight_signs_byte_offset");                  //  34
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  66
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                  //  74
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

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
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid emitter;
  // weight_block_stride 82 disambiguates iq2_s from iq1_s (50)/iq1_m (56)/iq3_xxs (98)/
  // iq2_xxs (66)/iq2_xs (74).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq2_s per-half-explicit-scale GRID INTEGER CORE (the per-sub-block explicit
  // 4-bit scale byte -> ls1/ls2 + the qh-plane byte + the two-half 16-lane decode + the
  // 2-index vluxei16_v_i64m1 grid gather + the SECOND signs256 vluxei16 gather + the
  // vmul-onto-grid sign fold + the signed widening grid dot + the per-half scale fold into
  // bsum). The LIVE operands are the weight base (%vx) + activation base (%vy) + n + vl +
  // block_index; it produces the ONE SCALAR i32 state bsum (NO output pointer -- iq2_s's
  // bsum is a scalar register). Per-super-block address vx + ib*82, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ2SQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq2_s_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-half-int4-explicit-scales-grid-codebook-qh-"
                              "plane-explicit-signs-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_signs_byte_offset",
                   builder.getI64IntegerAttr(weightSignsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq2_s grid group counts as
    // FORMAT-DEFINED descriptor facts (4 grid groups per sub-block, 2 grid/sign
    // groups per 16-lane half).
    s.addAttribute("groups_per_sub", builder.getI64IntegerAttr(4));
    s.addAttribute("num_groups_per_half", builder.getI64IntegerAttr(2));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY). IDENTICAL to
  // iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

} // namespace weft::plugin::rvv
