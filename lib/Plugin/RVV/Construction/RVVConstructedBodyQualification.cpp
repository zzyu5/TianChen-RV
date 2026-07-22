#include "Weft/Plugin/RVV/RVVFormulaConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace weft::plugin::rvv {

namespace {

// M-FLAT loop-scaffold step 4/6: the loop-aware allowlist RECURSIVE validator --
// the strong-form gate ([L-8], core-invariants I5/I7). It RECURSIVELY walks the
// weft_rvv.typed_flat_block_dot_loop_body region tree and asserts every op is a
// member of the forward-compatible typed-primitive allowlist (the bricks + the
// vector-core primitives + the scalar-extract bridge + the loop op/yield + the
// setvl/with_vl/load/store structural ops). It FAIL-CLOSES on any op outside the
// allowlist -- in particular an opaque emitc.call_opaque helper or a hand-written
// *_block_dot monolith leaking into the region at ANY nesting depth. This is the
// machine-checkable provenance basis a future q8_0 constructed(strong) flip
// depends on: it certifies the loop body is built entirely from pattern-library
// primitives with no sanctioned-opaque piece.
//
// It is deliberately the DUAL of the single-block rejectMixedPreRealizedContraction
// body BLOCKLIST (RVVEmitCContractionRouteFamilyInternal.h): that template rejects
// a variant that MIXES already-realized ops with the flat pre-realized op and does
// NOT recurse into any region; feeding the bricks into that blocklist would wrongly
// reject the brick BODY (the anti-write trap). This new allowlist walk is additive
// and never touches rejectMixed, so the 3 single-block strong paths are structurally
// unchanged.
// The shared forward-compatible typed-primitive allowlist membership test (the
// bricks + the vector-core primitives + the scalar-extract bridge + the loop
// ops/yields + the setvl/with_vl/load/store structural ops). Both the flat
// single-accumulator loop body and the q4_K/q5_K DUAL-accumulator super-block
// loop body share ONE union allowlist (the flat validator already unions all
// four flat folds' primitives regardless of which fold a given body uses; adding
// the q4_K super-block bricks keeps that union design -- it only ever makes the
// allowlist MORE permissive, never rejecting an op it used to accept, so the flat
// paths are structurally unchanged).
bool isTypedBlockDotLoopBodyAllowlistOp(mlir::Operation *op) {
  return llvm::isa<
      // brick 1 (per-block fp16 scale product / per-block-source) +
      // the Family-B (q4_1) per-block MIN/SUM correction product
      weft::rvv::BlockFp16ScaleProductOp, weft::rvv::BlockFp16MinProductOp,
      // vector integer core primitives (q8_0 plain signed widening
      // product + q4_0 asymmetric offset-binary packed-i4 x i8 product
      // + q4_1 asymmetric unsigned-nibble packed-i4 x i8 product)
      weft::rvv::WideningProductOp, weft::rvv::PackedI4OffsetBinaryXI8ProductOp,
      weft::rvv::UnsignedNibbleXI8ProductOp,
      // q5_0 five-bit offset-binary packed-i4 (+ qh 5th bit) x i8 product
      // + its per-block qh 32-bit-field source brick
      weft::rvv::FiveBitOffsetBinaryXI8ProductOp,
      weft::rvv::BlockFiveBitQhSourceOp, weft::rvv::StandaloneReduceOp,
      // iq4_nl / FP4 codebook class (2nd primitive class): the 16-entry non-linear
      // int8 lookup-table broadcast brick + the asymmetric codebook-gather packed-i4
      // x plain-i8 widening product brick (vrgather decode, NOT offset-binary)
      weft::rvv::CodebookTableBroadcastOp,
      weft::rvv::CodebookGatherXI8ProductOp,
      // step 2 scalar-lane extract bridge (integer core -> scalar fold)
      weft::rvv::TypedVectorLane0ToScalarExtractOp,
      // brick 2 (per-block computed-scale dequant term)
      weft::rvv::BlockComputedScaleDequantOp,
      // brick 3 (cross-block f32 loop-carried accumulate)
      weft::rvv::CrossBlockF32AccumulateOp,
      // W4: the q4_K/q5_K super-block bricks (nibble unpack, 6-bit scale/min
      // bit-dance, per-sub-block scaled i32 dot, MIN term, deferred positive
      // fold, post-loop horizontal fold)
      weft::rvv::Q4KNibbleUnpackOp, weft::rvv::Q4KScaleMinBitDanceOp,
      weft::rvv::Q4KScaledDotOp, weft::rvv::Q4KMinTermOp,
      weft::rvv::Q4KSumsFoldScaleDOp, weft::rvv::Q4KHorizontalFoldOp,
      // W-C (q6_K milestone-1): the q6_K no-min super-block INTEGER core (the
      // 2-bit qh + 8-bit signed scale unpack + per-sub-block i32 aux32 dot); it
      // is the single-accumulator body's integer brick (paired with the reused
      // q4_K Q4KSumsFoldScaleDOp for the no-min positive fold)
      weft::rvv::GgmlBlockDotQ6KQ8KAux32Op,
      // q3_K first flip: the q3_K no-min super-block INTEGER core (the 2-bit +
      // SUBTRACTIVE-hmask unpack + SIGNED 6-bit scale dance + per-sub-block i32
      // aux32 dot); the q3_K sibling of the q6_K aux32 brick, driving the SAME
      // single-accumulator no-min body (paired with the reused positive fold)
      weft::rvv::GgmlBlockDotQ3KQ8KAux32Op,
      // W-C' (q2_K milestone-1): the q2_K scalar super-block INTEGER core (the
      // 2-bit unpack + plain uint4-nibble scale/min + per-sub-block scalar i32
      // dot producing the two SCALAR states isum + summs); it is the
      // scalar-accumulator body's integer brick (the byte-exact scalar fold
      // sumf += dall*isum - dmin*summs is a later step)
      weft::rvv::GgmlBlockDotQ2KQ8KIntegerCoreOp,
      // iq1_s milestone-1 (first super-block GRID/codebook body): the iq1_s
      // scalar super-block TERNARY-grid INTEGER core (decode_model=lookup -- the
      // 11-bit grid index build from qs+qh, the vluxei16 ternary-grid gather, the
      // signed widening grid dot, the qh-encoded per-sub-block scale, the
      // delta-bsum sum, producing the two SCALAR states sumi + sumi1); it is the
      // scalar-accumulator body's grid brick under fold_model "scalar_delta_grid"
      // (the byte-exact scalar fold sumf += d*(sumi + IQ1S_DELTA*sumi1) is a
      // later step, milestone-2)
      weft::rvv::GgmlBlockDotIQ1SQ8KGridCoreOp,
      // iq1_m (iq1_s sibling): the iq1_m scalar super-block TERNARY-grid INTEGER
      // core (decode_model=lookup -- the SAME 2048-entry iq1s_grid, the packed
      // iq1m_scale fp16 reconstruct, the half-split per-half vluxei16 grid dot with
      // two half scales ls1/ls2, the per-group four-sign delta via a fresh Σq8,
      // producing the two SCALAR states sumi1 + sumi2); it is the scalar-accumulator
      // body's grid brick under fold_model "scalar_delta_grid" (disambiguated from
      // iq1_s by weight_block_stride 56 vs 50)
      weft::rvv::GgmlBlockDotIQ1MQ8KGridCoreOp,
      // iq3_xxs (iq1_s grid sibling): the iq3_xxs scalar super-block GRID-of-4
      // INTEGER core (decode_model=lookup -- the 256-entry uint32 iq3xxs_grid
      // gathered via vluxei16_v_i32m1, the aux32 4-bit-scale + 4-sign-group
      // ksigns decode, producing the ONE SCALAR state bsum); it is the
      // scalar-accumulator body's grid brick under fold_model "scalar_delta_grid"
      // (disambiguated from iq1_s/iq1_m by weight_block_stride 98 and the i32 grid)
      weft::rvv::GgmlBlockDotIQ3XXSQ8KGridCoreOp,
      // iq2_xxs (iq1_s grid sibling, SIGN-PLANE signs64 variant): the iq2_xxs scalar
      // super-block GRID-of-8 INTEGER core (decode_model=lookup -- the 256-entry uint64
      // iq2xxs_grid gathered via vluxei16_v_i64<core>, the SECOND vluxei16 gather over
      // the DERIVED keven_signs_q2xs signs64 sign plane, the aux1 4-bit-scale +
      // 4-sign-group decode, producing the ONE SCALAR state bsum); it is the
      // scalar-accumulator body's grid brick under fold_model "scalar_delta_grid"
      // (disambiguated from iq1_s/iq1_m/iq3_xxs by weight_block_stride 66; carries the
      // Win-A integer_core_lmul m2/m1 gearbox)
      weft::rvv::GgmlBlockDotIQ2XXSQ8KGridCoreOp,
      // iq2_xs (iq2_xxs grid sibling, SIGN-PLANE signs64 variant, PER-HALF explicit scale):
      // the iq2_xs scalar super-block per-half-scale GRID INTEGER core (decode_model=lookup
      // -- the 512-entry uint64 iq2xs_grid gathered via vluxei16_v_i64m1 indexed by
      // `w & 511`, the SECOND vluxei16 gather over the DERIVED keven_signs_q2xs signs64 sign
      // plane keyed by `w >> 9`, the EXPLICIT per-sub-block 4-bit scales[8] two-half split
      // ls1/ls2, producing the ONE SCALAR state bsum); it is the scalar-accumulator body's
      // grid brick under fold_model "scalar_delta_grid" (disambiguated from
      // iq1_s/iq1_m/iq3_xxs/iq2_xxs by weight_block_stride 74; NO gearbox -- fixed 16-lane
      // per-half shape)
      weft::rvv::GgmlBlockDotIQ2XSQ8KGridCoreOp,
      // iq2_s (iq2_xs grid sibling, SIGN-PLANE explicit-signs variant, PER-HALF explicit
      // scale): the iq2_s scalar super-block per-half-scale GRID INTEGER core
      // (decode_model=lookup -- the 1024-entry uint64 iq2s_grid gathered via
      // vluxei16_v_i64m1 indexed by `qs[l] | ((qh<<(8-2l))&0x300)`, the SECOND vluxei16
      // gather over the UNIVERSAL signs256 explicit-sign-byte plane keyed by the raw sign
      // byte, the EXPLICIT per-sub-block 4-bit scales[8] two-half split ls1/ls2, producing
      // the ONE SCALAR state bsum); it is the scalar-accumulator body's grid brick under
      // fold_model "scalar_delta_grid" (disambiguated from
      // iq1_s/iq1_m/iq3_xxs/iq2_xxs/iq2_xs by weight_block_stride 82; NO gearbox -- fixed
      // 16-lane per-half shape)
      weft::rvv::GgmlBlockDotIQ2SQ8KGridCoreOp,
      // iq3_s (iq3_xxs grid sibling, EXPLICIT-SIGNS variant, qh 9th-bit inject, explicit
      // two-nibble scale): the iq3_s scalar super-block GRID-of-4 INTEGER core
      // (decode_model=lookup -- the 512-entry uint32 iq3s_grid gathered via
      // vluxei16_v_i32m1 indexed by the qh-9th-bit-injected index `qs[l] |
      // ((qh<<(8-2l))&256)`, the EXPLICIT per-sub-block sign bytes read from the signs
      // region at offset 74 folded via the inline kmask {1<<j}, the EXPLICIT per-sub-block
      // two-nibble scales[4], producing the ONE SCALAR state bsum); it is the
      // scalar-accumulator body's grid brick under fold_model "scalar_delta_grid"
      // (disambiguated from iq1_s/iq1_m/iq3_xxs/iq2_xxs/iq2_xs/iq2_s by weight_block_stride
      // 110; NO gearbox -- fixed grid-of-4 shape, NO ksigns plane)
      weft::rvv::GgmlBlockDotIQ3SQ8KGridCoreOp,
      // iq4_xs (flat iq4_nl CODEBOOK sibling, SUPER-BLOCK rung): the iq4_xs scalar
      // super-block CODEBOOK INTEGER core (decode_model=lookup -- REUSES iq4_nl's
      // 16-entry non-linear int8 codebook gathered via vrgather_vv_i8m1, the per-sub-block
      // SIGNED 6-bit scale from scales_l[4]+scales_h biased -32 applied in the FLOAT
      // domain, producing the 8 per-sub-block sumi that fold into the scalar sumf); it is
      // the scalar-accumulator body's codebook brick under fold_model "scalar_delta_grid"
      // (disambiguated from the grid siblings by weight_block_stride 136; NO gearbox --
      // the codebook gather pins m1). UNLIKE the grid siblings the fold runs per-sub-block
      // in float, but the single-scalar accumulator arity is identical.
      weft::rvv::GgmlBlockDotIQ4XSQ8KCodebookCoreOp,
      // tq2_0 (the FIRST TQ-family member, ARITHMETIC 2-bit ternary decode): the tq2_0
      // FUSED 2-bit TERNARY super-block INTEGER core (decode_model=arithmetic -- q2_K's
      // 2-bit `(qs>>shift)&3` unpack over the 4 shifts {0,2,4,6}, the per-element `-1`
      // ternary bias vsub, the fused-plane vwmacc against q8 into a wide i16 accumulator,
      // ONE vwredsum per 32-byte chunk, producing the ONE SCALAR state sumi -- NO
      // grid/codebook gather); it is the scalar-accumulator body's ternary brick under
      // fold_model "scalar_delta_grid" (it SHARES weight_block_stride 66 with iq2_xxs but
      // the DISTINCT brick op type disambiguates; carries the Win-A integer_core_lmul
      // m2/m1 gearbox, kernel key "tq2_0")
      weft::rvv::GgmlBlockDotTQ20Q8KTernaryCoreOp,
      // tq1_0 (the SECOND TQ-family member, ARITHMETIC BASE-3 ternary decode): the tq1_0
      // BASE-3 TERNARY super-block INTEGER core (decode_model=arithmetic -- the qs main/tail
      // + qh base-3 trit unpack `q=(uint8_t)(byte*pow3[l]); xi=((uint16_t)q*3)>>8; xi-1` into
      // an element-ordered aux8[256], then the flat-256 widened i8*i8 dot -- vle8 i8 x q8 i8
      // -> vwmul i16 -> vwredsum i32 -- producing the ONE SCALAR state sumi; NO grid/codebook
      // gather); it is the scalar-accumulator body's ternary brick under fold_model
      // "scalar_delta_grid" (its weight_block_stride 54 is UNIQUE, so it dispatches by stride;
      // the DISTINCT base-3 brick op type also disambiguates; carries the Win-A
      // fixed VLEN-universal body with no fake LMUL schedule axis). REUSES the tq2_0 ternary
      // scaffold at C2 marginal cost.
      weft::rvv::GgmlBlockDotTQ10Q8KTernaryCoreOp,
      // q1_0 (the BINARY {-1,+1}-sign class, the LAST flat block-dot family
      // member, C_construct 26->27): the q1_0 BINARY-sign FLAT integer core
      // (decode_model=arithmetic -- the four q8_0 sub-blocks' vlm_v_b{ratio}
      // packed-bit sign mask loaded DIRECTLY as the i8 sign mask, i8-domain
      // vneg/vmerge -> ONE vwredsum i8->i16m1 per sub-block, plus the
      // emitter-inlined TWO-LEVEL fp32 fold `d0 * Σ_k(d1_k * sumi_block_k)`; NO
      // grid/codebook gather, NO nibble unpack); it is the flat loop body's
      // binary-sign brick under fold_model "flat_binary_two_level" (the FLAT
      // scaffold sibling of the super-block scalar cores -- q1_0's activation is a
      // FLAT block_q8_0 stream). Carries the Win-A integer_core_lmul m2/m1
      // gearbox, kernel key "q1_0".
      weft::rvv::GgmlBlockDotQ10Q80BinarySignCoreOp,
      // nvfp4 (the SECOND FP4-codebook class, NVIDIA's FP4): the nvfp4 FP4-CODEBOOK
      // FLAT integer core (decode_model=lookup -- REUSES mxfp4's 16-entry DOUBLED
      // e2m1 codebook gathered via vrgather_vv_i8m1, wrapped in the per-sub-block
      // UE4M3 fp8 weight scale + the two-q8_0-block/half addressing, producing the 4
      // per-sub-block sumi that fold into the scalar sumf); it is the flat loop
      // body's codebook brick under fold_model "flat_nvfp4_codebook" (the FLAT
      // scaffold sibling of q1_0 -- nvfp4's activation is a block_q8_0 stream). NO
      // gearbox -- the codebook gather pins m1.
      weft::rvv::GgmlBlockDotNVFP4Q80CodebookCoreOp,
      // structural VL / memory ops
      weft::rvv::SetVLOp, weft::rvv::WithVLOp, weft::rvv::LoadOp,
      weft::rvv::StoreOp,
      // M-FLAT REPACK (q4_0 16x1 GEVM): the per-block lane-wise nibble-dot
      // integer-core brick (seed i16 lo/hi + nibble-step vwmacc + lo/hi vwadd
      // combine) carried inside the repack GEVM loop body region
      weft::rvv::RepackLaneWiseQ4Q8DotOp,
      // M-FLAT REPACK (q4_0 16x1 GEMM, finale): the ONE-strip N-column integer
      // core brick + the per-column dual-fp16 scale FOLD brick carried inside the
      // repack GEMM loop body region (the runtime-strip x interleaved-column
      // transpose of the GEVM bricks)
      weft::rvv::RepackGemmLaneWiseQ4Q8DotOp,
      weft::rvv::RepackGemmDualFp16ScaleFoldOp,
      // the loop ops themselves + their terminators (forward-compatible): the
      // flat single-accumulator loop op, the q4_K DUAL-accumulator super-block
      // loop op, the q4_0 16x1-repacked GEVM per-strip vector-accumulator loop, and
      // the q4_0 16x1-repacked GEMM per-column vector-accumulator loop
      weft::rvv::TypedFlatBlockDotLoopBodyOp,
      weft::rvv::TypedFlatBlockDotLoopYieldOp,
      weft::rvv::TypedSuperBlockBlockDotLoopBodyOp,
      weft::rvv::TypedSuperBlockBlockDotLoopYieldOp,
      weft::rvv::TypedRepackGemvLoopBodyOp,
      weft::rvv::TypedRepackGemvLoopYieldOp,
      weft::rvv::TypedRepackGemmLoopBodyOp,
      weft::rvv::TypedRepackGemmLoopYieldOp,
      // M-FLAT forward-elementwise scaffold (line C, ① 之后): the typed
      // elementwise strip-loop op + its map core brick + terminator. They join
      // the SAME union allowlist (making it strictly MORE permissive, never
      // rejecting an op it used to accept, so every block-dot path is
      // structurally unchanged); the dedicated elementwise validator walk below
      // fires only on the elementwise loop op_kind.
      weft::rvv::TypedElementwiseLoopBodyOp,
      weft::rvv::TypedElementwiseLoopYieldOp,
      weft::rvv::ElementwiseScaleMapOp,
      // The SECOND forward-elementwise map core brick (silu), reusing the SAME
      // loop op + terminator + validator above; only the per-strip map primitive
      // is new. The union stays strictly MORE permissive (zero block-dot
      // regression).
      weft::rvv::ElementwiseSiluMapOp,
      // The FIRST forward-elementwise REDUCE core brick (rms_norm), reusing the
      // SAME loop op + terminator + validator, now under reduce_map_model
      // "reduce" (a loop-carried f64 accumulator). Its Σx² fold + rsqrt +
      // normalize are re-emitted by the loop op's reduce branch. The union stays
      // strictly MORE permissive (zero block-dot / map regression).
      weft::rvv::ElementwiseRmsNormReduceCoreOp,
      // The fused rms_norm->mul EPILOGUE consumer brick, carried inside the
      // rms_norm reduce core's optional $epilogue region. The reduce-body emitter
      // splices its per-lane vfmul_vv into the normalize strip (the register-kept
      // vy flows straight in; the intermediate normalized row never touches
      // memory). The union stays strictly MORE permissive (zero block-dot / map /
      // reduce regression).
      weft::rvv::ElementwiseMulMapOp,
      // The [FMT-PROP] fused-activation-quantize EPILOGUE consumer brick, carried
      // inside the mul_map's optional $quant_epilogue region. The reduce-body
      // emitter runs the per-block q8_0 amax/scale/narrow body on the register-kept
      // weighted vz (the f32 z[] intermediate never touches memory; the downstream
      // independent quantize_row pass is elided). The union stays strictly MORE
      // permissive (zero block-dot / map / reduce regression).
      weft::rvv::ElementwiseQuantizeQ80MapOp,
      // The SECOND forward-elementwise REDUCE core brick (soft_max), reusing the
      // SAME loop op + terminator + validator + reduce_map_model "reduce", EXCEPT
      // the loop-carried accumulator is the f64m1 WIDENING vector (vfwredusum Σe^x
      // fold) not a scalar double. Its fused exp-store-widening-reduce strip is
      // re-emitted by the soft_max return-carrying branch. The union stays
      // strictly MORE permissive (zero block-dot / map / rms_norm regression).
      weft::rvv::ElementwiseSoftMaxReduceCoreOp,
      // The FIRST forward-elementwise ROTATE core brick (rope), reusing the SAME
      // loop op + terminator + validator, now under reduce_map_model "rotate" (a
      // per-pair scalar loop with a loop-carried f32 theta recurrence). Its
      // position-dependent 2x2 rotation + scalar cos/sin angle seam + theta
      // recurrence are re-emitted by the loop op's rotate branch. The union stays
      // strictly MORE permissive (zero block-dot / map / reduce regression).
      weft::rvv::ElementwiseRopeRotateCoreOp,
      // The THREE forward SUPPORT-op MAP core bricks (the add/mul BINARY two-input
      // map, the cpy pass-through COPY map, and the gelu scalar tanh map), reusing
      // the SAME loop op + terminator + validator + reduce_map_model "map". Their
      // per-strip/per-element bodies are re-emitted by the loop op's map branch
      // via the SHARED byte-exact strip/loop helpers. The union stays strictly
      // MORE permissive (zero block-dot / map / reduce / rotate regression).
      weft::rvv::ElementwiseBinaryMapOp,
      weft::rvv::ElementwiseCopyMapOp,
      weft::rvv::ElementwiseGeluMapOp>(op);
}

// Shared recursive allowlist walk over a loop-body region: fail-close on any op
// outside the union allowlist (the strong-form [L-8] gate). `surfaceName` names
// the loop surface in the diagnostic.
llvm::LogicalResult validateLoopBodyAllowlist(mlir::Region &body,
                                              llvm::StringRef surfaceName) {
  mlir::WalkResult walk =
      body.walk([&](mlir::Operation *op) -> mlir::WalkResult {
        if (isTypedBlockDotLoopBodyAllowlistOp(op))
          return mlir::WalkResult::advance();
        op->emitOpError()
            << "is not in the M-FLAT typed " << surfaceName
            << " block-dot loop-body allowlist; the loop-aware allowlist "
               "validator (the strong-form [L-8] gate) recursively certifies "
               "every op in the loop region is a typed pattern-library primitive "
               "and fail-closes on any non-allowlist op (an opaque "
               "emitc.call_opaque or a hand-written *_block_dot helper must never "
               "leak into the constructed-strong body)";
        return mlir::WalkResult::interrupt();
      });
  return walk.wasInterrupted() ? mlir::failure() : mlir::success();
}

llvm::LogicalResult validateTypedFlatBlockDotLoopBodyAllowlist(
    weft::rvv::TypedFlatBlockDotLoopBodyOp loopBody) {
  return validateLoopBodyAllowlist(loopBody.getBody(), "flat");
}

// W4: the q4_K/q5_K DUAL-accumulator super-block loop body's [L-8] allowlist gate
// -- the SAME recursive strong-form certifier, wired to the super-block loop op.
llvm::LogicalResult validateTypedSuperBlockBlockDotLoopBodyAllowlist(
    weft::rvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) {
  return validateLoopBodyAllowlist(loopBody.getBody(), "super-block");
}

// M-FLAT forward-elementwise scaffold: the typed elementwise strip-loop body's
// [L-8] allowlist gate -- the SAME recursive strong-form certifier, wired to the
// forward-pass (non-dot) elementwise loop op. It fail-closes on any op outside
// the union allowlist (an opaque emitc.call_opaque or a hand-written monolith
// forward helper leaking into the constructed body), the machine-checkable
// provenance basis for the scale constructed(strong) flip.
llvm::LogicalResult validateTypedElementwiseLoopBodyAllowlist(
    weft::rvv::TypedElementwiseLoopBodyOp loopBody) {
  return validateLoopBodyAllowlist(loopBody.getBody(), "elementwise");
}

} // namespace

mlir::LogicalResult
validateRVVConstructedTypedBodies(mlir::ModuleOp module) {
  bool rejected = false;
  module.walk([&](weft::rvv::TypedFlatBlockDotLoopBodyOp loopBody) {
    if (mlir::failed(validateTypedFlatBlockDotLoopBodyAllowlist(loopBody)))
      rejected = true;
  });
  module.walk([&](weft::rvv::TypedSuperBlockBlockDotLoopBodyOp loopBody) {
    if (mlir::failed(
            validateTypedSuperBlockBlockDotLoopBodyAllowlist(loopBody)))
      rejected = true;
  });
  module.walk([&](weft::rvv::TypedElementwiseLoopBodyOp loopBody) {
    if (mlir::failed(validateTypedElementwiseLoopBodyAllowlist(loopBody)))
      rejected = true;
  });
  return rejected ? mlir::failure() : mlir::success();
}

} // namespace weft::plugin::rvv
