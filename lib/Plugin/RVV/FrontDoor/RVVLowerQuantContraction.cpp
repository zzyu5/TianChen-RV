//===- RVVLowerQuantContraction.cpp ---------------------------------------===//
//
// The option-2 front-of-pipeline pass that lowers the abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction op to a CONCRETE contraction
// op, running BEFORE the EmitC lowering.
//
// STAGE B (this file): the pass makes "the COMPILER itself selects
// repack-vs-block-dot from capability facts" actually TRUE. Per walked
// quant_contraction request it derives the target VLEN from the pass's -march
// (deriveMinimumVLEN -- the SAME capability authority every other capability-gated
// pass uses; the op's advisory min_vlen attr is NOT the source), READS the op's
// STRUCTURED OPPONENT FACTS (opponent_vlen_native_floor / block_dot_compute_heavy)
// and its committed m_regime, and calls the pure, branch-free,
// capability-fact-driven selectContractionAlgorithm. Routing reads the structured
// facts, NEVER the (now optional) quant format LABEL: deleting `quant` and keeping
// the facts selects the IDENTICAL algorithm and constructs the IDENTICAL region.
// The decision is stamped on the lowered op as three INERT audit attrs
// (weft_rvv.*) so it is provable in-IR and lit-CHECKable.
//
// STAGE C1 (this file, the in-IR BRIDGE): the pass now LOWERS a repack-SELECTED
// request to the REAL weft_rvv.repack_gemv_q4_0_q8_0 op and DECLARES the kernel's
// weight-layout requirement as an OUTPUT CONTRACT (weft_rvv.weight_layout_contract
// = "x16"). The compiler is the layout's CONSUMER + the contract's DECLARER; the
// plain->x16 weight MATERIALIZATION lives OUTSIDE the IR (the load-time / JIT
// producer, stages C3-C4). This is the layout-as-input / declared-contract model:
// the per-tensor limit does NOT dissolve, it RELOCATES to the system layer that
// owns the bytes.
//
// THE CRUX (stage B was Option (i): byte-identical block-dot on every path; C1
// flips the repack-SELECTED cell): the concrete repack target requires
// pre-interleaved block_q4_0x16 weights (stride 288, interleave 16) the abstract
// op's PLAIN stride-18 weights cannot supply. The bridge emits the repack op
// carrying the x16 facts + the contract, and REALIZES it ONLY where the target
// capability affords a valid e16m1 strip width (deriveRepackHalfLanes(minVLEN) in
// {8, 16} -- minVLEN >= 128). A repack-SELECTED request with no capability strip
// width (the q4_0-prefill cell at VLEN0: half_lanes 0) stays the deferred
// block-dot stub. The block-dot-SELECTED branch (q4_0@K1, q8_0, q4_K) is
// UNCHANGED + byte-identical. The repack-SELECTED cell DELIBERATELY changes (it
// emits the repack kernel) -- that is the POINT of C1; lit-verified, NOT run.
//
// SAFETY (NOT a latent miscompile): the emitted repack kernel reads x16 weights
// but the abstract op carries PLAIN weights, so the emit is correct ONLY when the
// contract is honored (x16 provided = stages C3-C4). The abstract
// GgmlQuantContractionOp has NO real producer (it is authored ONLY in lit
// fixtures; there is no rewriter.create of it in any real pass), so this repack
// op is reachable ONLY via lit, NEVER in the real llama.cpp pipeline. The bridge
// ASSERTS the layout; the system must make it true. NO e2e/perf claim is made.
//
// The block-dot identity branch DROPS the abstract op's column_count (nc)
// operand: the block-dot kernel (ggml_vec_dot_q4_0_q8_0) writes ONE fp32 and
// delegates the M/N loops to ggml's mul_mat caller (a bare 4-operand vec_dot). NO
// schedule attrs are stamped -- those remain MaterializeRVVQ40Schedule's job
// downstream. On any module with no quant_contraction op the pass is a no-op.
//
//===----------------------------------------------------------------------===//

#include "Weft/Transforms/Passes.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Plugin/RVV/RVVContractionPathSelection.h"
#include "Weft/Plugin/RVV/RVVFormulaDecision.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Plugin/RVV/RVVRepackScheduleFormula.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Support/DeclaredInstanceHash.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace weftrvv = ::weft::rvv;
namespace pluginrvv = ::weft::plugin::rvv;

namespace weft::transforms {

#define GEN_PASS_DEF_RVVLOWERQUANTCONTRACTION
#include "Weft/Transforms/Passes.h.inc"

namespace {

// The inert audit-attr names the stage-B pass stamps on the lowered concrete op.
// They are pure provenance (no SEW/LMUL/policy/dataflow config) -- the EmitC
// emitter ignores them, exactly like the MaterializeRVVQ40Schedule pass's
// additive "weft_rvv.q4_0_schedule.*" trail -- so the emitted C is byte-identical
// with them present. The block-dot verifier's attribute allow-list is widened to
// accept this bounded namespace (RVVDialectWideningOps.cpp isAllowedBlockDotAttr).
constexpr llvm::StringLiteral kAlgorithmAttr = "weft_rvv.contraction_algorithm";
constexpr llvm::StringLiteral kReasonAttr = "weft_rvv.path_selection_reason";
constexpr llvm::StringLiteral kMaterializationAttr =
    "weft_rvv.path_materialization";

// The option-2 stage-C1 OUTPUT CONTRACT carrier (carrier A, the in-IR op attr).
// When the bridge realizes a repack-SELECTED request as the real repack-GEMV op
// it stamps weft_rvv.weight_layout_contract = "x16": the DECLARED requirement
// that the weight bytes the emitted kernel reads are in the block_q4_0x16 layout
// (the op's weight_block_stride = 288 contract). The compiler ASSERTS the layout;
// some later layer (the load-time / JIT producer, stages C3-C4) must make it
// true. The repack-GEMV verifier's attr allow-list accepts this bounded name
// (RVVDialectWideningOps.cpp GgmlRepackGemvQ40Q80Op::verify isAllowedAttr).
constexpr llvm::StringLiteral kWeightLayoutContractAttr =
    "weft_rvv.weight_layout_contract";

// [档 C#9 / full-LMUL[B]] the repack accumulator-LMUL (m1 whole-LMUL vs mf2 fractional
// core) selection PROVENANCE the front-door stamps on every constructed repack loop
// body op, parallel to the path/tiling/loop-order reason attrs. Pure INERT discardable
// provenance (the EmitC emitter derives the widening chain from integer_core_lmul /
// half_lanes -- it NEVER reads this reason), so stamping it is byte-exact: it records
// WHY m1/mf2 was chosen ("correctness-rvv0p7" / "measured" / "capability-default-mf2")
// so every cell answers the "why this accumulator LMUL" question. Previously the
// A2 centralizes the reason/key mirror in one stamp helper for all repack builders.
constexpr llvm::StringLiteral kAccumulatorLmulReasonAttr =
    "weft_rvv.repack_accumulator_lmul_selection_reason";
constexpr llvm::StringLiteral kAccumulatorLmulMeasurementKeyAttr =
    "weft_rvv.repack_accumulator_lmul_measurement_key";

// The RVV vector register file is 32 architectural vector registers as a HARD ISA
// fact (rvv1.0 v0..v31), independent of VLEN -- the register-budget capability fact
// the SP4 legality filter reasons over (alongside the derived minimum VLEN). Read
// from the ONE plugin-local authority (getRVVArchitecturalVectorRegisterCount, the
// schema `vreg_count` hardware-fact) so the budget is a named capability fact, not
// a magic literal scattered across selectors.
const std::int64_t kRVVArchVectorRegisterCount =
    pluginrvv::getRVVArchitecturalVectorRegisterCount();

// The repacked weight 16-way interleave (block_q4_0x16: 16 weight rows per group
// occupy 16 distinct vector lanes). MIRRORS the op verifier's weight_interleave
// == 16 pin and deriveRepackHalfLanes's clamp input.
constexpr std::int64_t kWeightInterleave = 16;

// The repacked GEMM (prefill) activation 4-way interleave (block_q8_0x4: 4
// activation columns per group). MIRRORS the repack-GEMM op verifier's
// activation_interleave == 4 pin. The GEVM (decode) reads a single plain q8_0
// column and carries no interleave.
constexpr std::int64_t kActivationInterleave = 4;

// The abstract request's committed decode-FAMILY discriminator: the ternary
// tq2_0 scale_model WHAT. Routing (repack-vs-block-dot) stays fact-driven off the
// opponent facts; the decode FAMILY (which core brick + block facts the lowering
// CONSTRUCTS) is keyed off this committed scale_model (a required WHAT attr, NOT
// the optional `quant` format label). A request carrying it is a BitNet-class
// 2-bit trit super-block contraction whose repack-SELECTED lowering CONSTRUCTS
// the ternary typed_repack_gem{v,m}_loop_body region (fold_model
// "ternary_single_fp16_scale", decode_model "tq2_0"), the sibling of the q4_0
// nibble region. Deleting the optional `quant` label keeps this WHAT intact.
constexpr llvm::StringLiteral kTernaryTQ20ScaleModel =
    "superblock-d.fp16-single-scale-2bit-ternary-nomin";

// The repacked block_tq2_0x16 / block_q8_K byte facts the ternary lowering
// RECONSTRUCTS (the stage-C x16 materialization the DECLARED weight_layout_contract
// asserts): the 16-inline-fp16-d weight super-block stride (1056), the weight trit
// quant byte offset (32, after the 16 fp16 d strip), the PLAIN q8_K GEVM activation
// stride (292) + its quant offset (4, after the fp32 d), and the INTERLEAVED
// block_q8_Kx4 GEMM activation stride (1168) + its quant offset (16). These MIRROR
// the retired monolithic tq2_0 repack op verifiers' pins.
constexpr std::int64_t kTernaryTQ20WeightBlockStride = 1056;
constexpr std::int64_t kTernaryTQ20WeightQuantByteOffset = 32;
constexpr std::int64_t kTernaryTQ20GevmActivationBlockStride = 292;
constexpr std::int64_t kTernaryTQ20GevmActivationQuantByteOffset = 4;
constexpr std::int64_t kTernaryTQ20GemmActivationBlockStride = 1168;
constexpr std::int64_t kTernaryTQ20GemmActivationQuantByteOffset = 16;

// The GEMM (prefill) ternary loop-body scale_model. The abstract op commits to the
// GEVM/base ternary scale_model (kTernaryTQ20ScaleModel); the GEMM lowering sets
// the 4-column-amortized variant on the constructed loop body (a pure I4 mirror --
// the loop-body verifier does NOT pin scale_model for the ternary fold; the emitter
// routes on fold_model + the core brick's decode_model).
constexpr llvm::StringLiteral kTernaryTQ20GemmScaleModel =
    "superblock-d.fp16-single-scale-2bit-ternary-4col-nomin";

// The ternary tq1_0 decode-FAMILY discriminators + repacked byte facts, the base-3
// sibling of the tq2_0 set. tq1_0 shares the ENTIRE ternary loop-body region + core
// brick scaffold (single fp16 super-block scale, LINEAR no-min lane-wise fold); the
// DELTA vs tq2_0 is the WEIGHT DECODE -- a base-3 5-trit/qs-byte + 4-trit/qh-byte
// two-plane unpack instead of the 2-bit field peel. The repacked block_tq1_0x16
// weight super-block stride is 864 (16 inline fp16 d + 768 qs base-3 bytes + 64 qh
// base-3 bytes), the qs plane at +32 (after the 16 fp16 d), the qh plane at +800
// (after the 16 d + 768 qs -- the SECOND weight plane tq2_0 lacks). The plain
// block_q8_K activation facts (292/4 GEVM, 1168/16 GEMM) are IDENTICAL to tq2_0.
constexpr llvm::StringLiteral kTernaryTQ10ScaleModel =
    "superblock-d.fp16-single-scale-base3-ternary-nomin";
constexpr llvm::StringLiteral kTernaryTQ10GemmScaleModel =
    "superblock-d.fp16-single-scale-base3-ternary-4col-nomin";
constexpr std::int64_t kTernaryTQ10WeightBlockStride = 864;
constexpr std::int64_t kTernaryTQ10WeightQuantByteOffset = 32;
constexpr std::int64_t kTernaryTQ10WeightQhByteOffset = 800;
constexpr std::int64_t kTernaryTQ10GevmActivationBlockStride = 292;
constexpr std::int64_t kTernaryTQ10GevmActivationQuantByteOffset = 4;
constexpr std::int64_t kTernaryTQ10GemmActivationBlockStride = 1168;
constexpr std::int64_t kTernaryTQ10GemmActivationQuantByteOffset = 16;

// The per-family ternary decode facts the shared lowerToRepackGem{v,m}Ternary read to
// CONSTRUCT the typed_repack region + the repack_gem{v,m}_ternary_core brick. The
// decode FAMILY (which base facts + core-brick decode_model + optional qh SECOND
// plane) is keyed off the committed abstract scale_model WHAT; routing (repack-vs-
// block-dot) stays fact-driven upstream. weightQhByteOffset == 0 is the "single-plane
// 2-bit ternary (tq2_0), NO qh plane" sentinel; a positive value is the tq1_0 base-3
// qh SECOND-plane offset the lowering stamps on the loop body op.
struct TernaryDecodeFacts {
  llvm::StringRef decodeModel;             // "tq2_0" | "tq1_0" (core brick)
  llvm::StringRef gemmScaleModel;          // the 4-col GEMM loop-op scale_model
  std::int64_t weightBlockStride;          // 1056 | 864
  std::int64_t weightQuantByteOffset;      // 32 (qs plane)
  std::int64_t weightQhByteOffset;         // 0 (tq2_0) | 800 (tq1_0 qh plane)
  std::int64_t gevmActivationBlockStride;  // 292
  std::int64_t gevmActivationQuantByteOffset;   // 4
  std::int64_t gemmActivationBlockStride;  // 1168
  std::int64_t gemmActivationQuantByteOffset;   // 16
};

constexpr TernaryDecodeFacts kTernaryTQ20DecodeFacts = {
    /*decodeModel=*/"tq2_0",
    /*gemmScaleModel=*/kTernaryTQ20GemmScaleModel,
    /*weightBlockStride=*/kTernaryTQ20WeightBlockStride,
    /*weightQuantByteOffset=*/kTernaryTQ20WeightQuantByteOffset,
    /*weightQhByteOffset=*/0,
    /*gevmActivationBlockStride=*/kTernaryTQ20GevmActivationBlockStride,
    /*gevmActivationQuantByteOffset=*/kTernaryTQ20GevmActivationQuantByteOffset,
    /*gemmActivationBlockStride=*/kTernaryTQ20GemmActivationBlockStride,
    /*gemmActivationQuantByteOffset=*/kTernaryTQ20GemmActivationQuantByteOffset,
};

constexpr TernaryDecodeFacts kTernaryTQ10DecodeFacts = {
    /*decodeModel=*/"tq1_0",
    /*gemmScaleModel=*/kTernaryTQ10GemmScaleModel,
    /*weightBlockStride=*/kTernaryTQ10WeightBlockStride,
    /*weightQuantByteOffset=*/kTernaryTQ10WeightQuantByteOffset,
    /*weightQhByteOffset=*/kTernaryTQ10WeightQhByteOffset,
    /*gevmActivationBlockStride=*/kTernaryTQ10GevmActivationBlockStride,
    /*gevmActivationQuantByteOffset=*/kTernaryTQ10GevmActivationQuantByteOffset,
    /*gemmActivationBlockStride=*/kTernaryTQ10GemmActivationBlockStride,
    /*gemmActivationQuantByteOffset=*/kTernaryTQ10GemmActivationQuantByteOffset,
};

// The K-quant q4_K decode-FAMILY discriminator (the abstract request's committed
// scale_model WHAT): the q4_K super-block dual d/dmin fp16 scale + 6-bit
// per-sub-block scale/min + activation bsums-min, 8 sub-blocks of 32. Routing
// (repack-vs-block-dot) stays fact-driven off the opponent facts; the decode FAMILY
// (which core brick + block facts the lowering CONSTRUCTS) is keyed off this
// committed scale_model. A request carrying it is a K-quant super-block contraction
// whose repack-SELECTED lowering CONSTRUCTS the K-quant typed_repack_gem{v,m}_loop_body
// region (fold_model "kquant_dmin_bsums_min", decode_model "q4_K") -- the K-quant
// sibling of the q4_0 nibble region and the ternary trit region. The GEMM lowering
// sets the 4-column-amortized variant on the constructed loop body (a pure I4 mirror;
// the loop-body verifier does NOT pin scale_model for the K-quant fold).
constexpr llvm::StringLiteral kKQuantQ4KScaleModel =
    "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks";
constexpr llvm::StringLiteral kKQuantQ4KGemmScaleModel =
    "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col";

// The K-quant q6_K decode-FAMILY discriminator (G3 主线A T3): the q6_K super-block
// SINGLE fp16 d scale + 16 SIGNED int8 per-16-element scales + 6-bit two-plane
// (ql|qh) offset-binary weight (-32 bias LANE-WISE), 16 sub-blocks of 16, and NO
// dmin / NO per-sub-block min / NO activation bsums (the single-accumulator no-min
// fold, the honest structural delta over q4_K's dual d/dmin + bsums-min). A request
// carrying it is a K-quant super-block contraction whose repack-SELECTED lowering
// CONSTRUCTS the K-quant typed_repack_gem{v,m}_loop_body region (fold_model
// "kquant_single_scale_no_min", decode_model "q6_K") -- the no-min sibling of the
// q4_K region. The framework (KQuantDecodeFacts + lowerToRepackGem{v,m}KQuant) is
// GENERALIZED across the two folds by the per-family facts.hasMin / facts.foldModel
// (the K-quant re-pay is ~0; only the q6_K decode leaf + no-min gating is net-new).
constexpr llvm::StringLiteral kKQuantQ6KScaleModel =
    "superblock-d.fp16-signed8-scale-16-subblocks-6bit-nomin";
constexpr llvm::StringLiteral kKQuantQ6KGemmScaleModel =
    "superblock-d.fp16-signed8-scale-16-subblocks-6bit-4col-nomin";

// The K-quant q2_K decode-FAMILY discriminator (G3 主线A T3 format2): the q2_K
// super-block dual d/dmin fp16 scale + per-sub-block 4-bit-packed scale/min +
// activation bsums-min, 16 sub-blocks of 16, with a 2-BIT UNSIGNED weight (4 lanes
// per byte, NO offset-binary bias -- the bias lives entirely in the 4-bit MIN). It
// is the LOWEST-bit K-quant and the min-term REGRESSION: it shares the SAME dual
// d/dmin + bsums-min FOLD structure as q4_K (fold_model "kquant_dmin_bsums_min",
// hasMin=true), so it REUSES the existing q4_K KQuantDecodeFacts fold WITHOUT any
// framework generalization -- the ONLY q2_K-specific work is the decode leaf (the
// 2-bit weight peel + the 4-bit packed scale/min unpack + the 16-sub-block single
// bsum) and the q2_K block offsets. A request carrying it is a K-quant super-block
// contraction whose repack-SELECTED lowering CONSTRUCTS the K-quant
// typed_repack_gem{v,m}_loop_body region (fold_model "kquant_dmin_bsums_min",
// decode_model "q2_K") -- the min-fold sibling of the q4_K region.
constexpr llvm::StringLiteral kKQuantQ2KScaleModel =
    "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit";
constexpr llvm::StringLiteral kKQuantQ2KGemmScaleModel =
    "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit-4col";

// The K-quant q3_K decode-FAMILY discriminator (G3 主线A T3 format3): the q3_K
// super-block SINGLE fp16 d scale + 16 SIGNED 6-bit per-16-element scales (pre-unpacked
// + -32-biased at repack, stored as signed int8) + 3-BIT SUBTRACTIVE weight assembled
// from a 2-bit `qs` low plane + a 1-bit `hmask` high plane (`((qs>>shift)&3 | hbit<<2)
// - 4`, a SIGNED value in [-4,3]; the -4 SUBTRACTIVE bias lives LANE-WISE inside each
// weight), 16 sub-blocks of 16, and NO dmin / NO per-sub-block min / NO activation
// bsums (the single-accumulator no-min fold). It is q6_K's NO-MIN structural cousin:
// it SHARES the SAME "kquant_single_scale_no_min" fold (hasMin=false), so it REUSES the
// no-min KQuantDecodeFacts fold WITHOUT any framework generalization -- the ONLY
// q3_K-specific work is the decode leaf (the 3-bit subtractive qs|hmask weight peel)
// and the q3_K block offsets. Its repack-SELECTED lowering CONSTRUCTS the K-quant
// typed_repack_gem{v,m}_loop_body region (fold_model "kquant_single_scale_no_min",
// decode_model "q3_K") -- the no-min sibling of the q6_K region, keying the hmask
// SECOND weight plane off the SHARED weightQhByteOffset (the q6_K qh slot).
constexpr llvm::StringLiteral kKQuantQ3KScaleModel =
    "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin";
constexpr llvm::StringLiteral kKQuantQ3KGemmScaleModel =
    "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin";

// The K-quant q5_K decode-FAMILY discriminator (G3 主线A T3 format4, the LAST K-quant):
// the q5_K super-block dual d/dmin fp16 scale + 6-bit per-sub-block scale/min
// (get_scale_min_k4, SAME as q4_K) + activation bsums-min, 8 sub-blocks of 32, with a
// 5-BIT weight = the q4_K 4-bit nibble PLUS a qh 5th (high) bit injected LANE-WISE
// (A = nibble | ((qh_bit & 1) << 4), a value in [0,31]; the offset-binary bias lives
// entirely in the 6-bit MIN, exactly q4_K). It is q4_K's 5-BIT structural cousin: it
// SHARES the SAME dual d/dmin + bsums-min FOLD (fold_model "kquant_dmin_bsums_min",
// hasMin=true), so it REUSES the existing q4_K KQuantDecodeFacts fold WHOLE WITHOUT any
// framework generalization -- the ONLY q5_K-specific work is the decode leaf (the qh
// 5th-bit inject onto the q4_K nibble) and the q5_K block offsets (which ADD a qh
// SECOND weight plane the min-fold q4_K/q2_K lack, keyed off the SHARED
// weightQhByteOffset slot the q6_K/q3_K no-min fold uses). A request carrying it is a
// K-quant super-block contraction whose repack-SELECTED lowering CONSTRUCTS the K-quant
// typed_repack_gem{v,m}_loop_body region (fold_model "kquant_dmin_bsums_min",
// decode_model "q5_K") -- the min-fold sibling of the q4_K region WITH the qh plane.
constexpr llvm::StringLiteral kKQuantQ5KScaleModel =
    "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5";
constexpr llvm::StringLiteral kKQuantQ5KGemmScaleModel =
    "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5";

// The repacked block_q{4,6}_Kx16 / block_q8_K{,x4} byte facts the K-quant lowering
// RECONSTRUCTS (the stage-C x16 materialization the DECLARED weight_layout_contract
// asserts). For q4_K (hasMin): the 16-inline-fp16-d + 16-inline-fp16-dmin +
// 192-scales + 2048-nibble weight super-block stride (2304), the weight nibble quant
// byte offset (256), the per-column dmin strip (32), the custom 6-bit scales/mins
// region (64), the PLAIN block_q8_K GEVM activation stride (292) + its quant offset
// (4) + its int16 bsums (260), the INTERLEAVED block_q8_Kx4 GEMM activation stride
// (1168) + its quant offset (16) + its bsums (1040), and 8 sub-blocks. For q6_K
// (no-min): the 16-inline-fp16-d + 256-signed-int8-scales + 1024-qh-high-2-bit +
// 2048-ql-low-4-bit super-block stride (3360), the weight ql quant offset (1312),
// the qh high-2-bit plane offset (288), the signed scales offset (32), 16 sub-blocks,
// and the SAME q8_K activation ABI (292/4 GEVM, 1168/16 GEMM) but with NO bsums read.
// These MIRROR the retired monolithic q{4,6}_K repack op verifiers' pins.
struct KQuantDecodeFacts {
  llvm::StringRef decodeModel;              // "q4_K" / "q6_K" (core brick)
  llvm::StringRef gemmScaleModel;           // the 4-col GEMM loop-op scale_model
  llvm::StringRef foldModel;                // loop-body fold_model
  bool hasMin;                              // q4_K dual d/dmin + bsums-min; q6_K no-min
  std::int64_t weightBlockStride;           // 2304 (q4_K) / 3360 (q6_K)
  std::int64_t weightQuantByteOffset;       // 256 nibble (q4_K) / 1312 ql (q6_K)
  std::int64_t weightDminByteOffset;        // 32 dmin strip (q4_K only; 0 for q6_K)
  std::int64_t weightScalesByteOffset;      // 64 6-bit (q4_K) / 32 signed int8 (q6_K)
  std::int64_t weightQhByteOffset;          // q6_K high-2-bit plane 288 (0 for q4_K)
  std::int64_t gevmActivationBlockStride;   // 292 (plain block_q8_K)
  std::int64_t gevmActivationQuantByteOffset;    // 4
  std::int64_t gevmActivationBsumsByteOffset;    // 260 (q4_K only; 0 for q6_K)
  std::int64_t gemmActivationBlockStride;   // 1168 (interleaved block_q8_Kx4)
  std::int64_t gemmActivationQuantByteOffset;    // 16
  std::int64_t gemmActivationBsumsByteOffset;    // 1040 (q4_K only; 0 for q6_K)
  std::int64_t nSubblocks;                  // 8 (q4_K) / 16 (q6_K)
};

constexpr KQuantDecodeFacts kQ4KDecodeFacts = {
    /*decodeModel=*/"q4_K",
    /*gemmScaleModel=*/kKQuantQ4KGemmScaleModel,
    /*foldModel=*/"kquant_dmin_bsums_min",
    /*hasMin=*/true,
    /*weightBlockStride=*/2304,
    /*weightQuantByteOffset=*/256,
    /*weightDminByteOffset=*/32,
    /*weightScalesByteOffset=*/64,
    /*weightQhByteOffset=*/0,
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/260,
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/1040,
    /*nSubblocks=*/8,
};

// q6_K: the SINGLE-accumulator no-min sibling. No dmin (single super-block d), no
// per-sub-block min, no activation bsums -- the -32 offset-binary bias lives INSIDE
// each 6-bit weight lane. The qh high-2-bit plane (288) is the q6_K-specific second
// weight plane the ql low-4-bit plane (1312) is fused with.
constexpr KQuantDecodeFacts kQ6KDecodeFacts = {
    /*decodeModel=*/"q6_K",
    /*gemmScaleModel=*/kKQuantQ6KGemmScaleModel,
    /*foldModel=*/"kquant_single_scale_no_min",
    /*hasMin=*/false,
    /*weightBlockStride=*/3360,
    /*weightQuantByteOffset=*/1312,
    /*weightDminByteOffset=*/0,
    /*weightScalesByteOffset=*/32,
    /*weightQhByteOffset=*/288,
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/0,
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/0,
    /*nSubblocks=*/16,
};

// q2_K: the MIN-fold sibling of q4_K. It REUSES the q4_K dual d/dmin + bsums-min
// fold (hasMin=true, foldModel "kquant_dmin_bsums_min"), so the framework re-pay is
// ZERO -- only the block offsets change. block_q2_Kx16 stride 1344 (16 fp16 d + 16
// fp16 dmin + 256 packed 4-bit scale/min + 1024 2-bit quant bytes): dmin strip @32,
// packed scale/min region @64, 2-bit weights @320. 16 sub-blocks of 16 (one bsum
// each). The q8_K activation ABI is byte-identical to q4_K (292/4/260 GEVM,
// 1168/16/1040 GEMM). NO qh plane (weightQhByteOffset == 0).
constexpr KQuantDecodeFacts kQ2KDecodeFacts = {
    /*decodeModel=*/"q2_K",
    /*gemmScaleModel=*/kKQuantQ2KGemmScaleModel,
    /*foldModel=*/"kquant_dmin_bsums_min",
    /*hasMin=*/true,
    /*weightBlockStride=*/1344,
    /*weightQuantByteOffset=*/320,
    /*weightDminByteOffset=*/32,
    /*weightScalesByteOffset=*/64,
    /*weightQhByteOffset=*/0,
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/260,
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/1040,
    /*nSubblocks=*/16,
};

// q3_K: the SINGLE-accumulator no-min sibling of q6_K (SHARES the no-min fold, K-quant
// re-pay ZERO -- only the block offsets + the decode leaf change). No dmin (single
// super-block d), no per-sub-block min, no activation bsums -- the -4 SUBTRACTIVE bias
// lives INSIDE each 3-bit weight lane. block_q3_Kx16 stride 1824 (16 fp16 d + 256
// signed int8 scales + 512 hmask high-bit + 1024 qs low-2-bit): signed scales @32, the
// hmask high-bit SECOND weight plane @288 (the q6_K qh slot -- weightQhByteOffset), the
// qs low-2-bit plane @800. 16 sub-blocks of 16. The q8_K activation ABI is byte-identical
// to q6_K (292/4 GEVM, 1168/16 GEMM, NO bsums read).
constexpr KQuantDecodeFacts kQ3KDecodeFacts = {
    /*decodeModel=*/"q3_K",
    /*gemmScaleModel=*/kKQuantQ3KGemmScaleModel,
    /*foldModel=*/"kquant_single_scale_no_min",
    /*hasMin=*/false,
    /*weightBlockStride=*/1824,
    /*weightQuantByteOffset=*/800,
    /*weightDminByteOffset=*/0,
    /*weightScalesByteOffset=*/32,
    /*weightQhByteOffset=*/288,
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/0,
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/0,
    /*nSubblocks=*/16,
};

// q5_K: the 5-BIT min-fold sibling of q4_K (= q4_K nibble + a qh 5th-bit plane). It
// REUSES the q4_K dual d/dmin + bsums-min fold WHOLE (fold_model "kquant_dmin_bsums_min",
// hasMin=true), so the framework re-pay is ZERO -- only the block offsets change (they
// ADD a qh SECOND weight plane) and the decode leaf adds the qh 5th-bit inject.
// block_q5_Kx16 stride 2816 (16 fp16 d + 16 fp16 dmin + 192 6-bit scales/mins + 512 qh
// high-bit + 2048 nibble bytes): dmin strip @32, 6-bit scales/mins region @64, the qh
// high-bit SECOND weight plane @256 (the SHARED weightQhByteOffset slot the q6_K/q3_K
// no-min fold uses -- here on a MIN fold), the nibbles @768. 8 sub-blocks of 32. The
// q8_K activation ABI is byte-identical to q4_K (292/4/260 GEVM, 1168/16/1040 GEMM).
constexpr KQuantDecodeFacts kQ5KDecodeFacts = {
    /*decodeModel=*/"q5_K",
    /*gemmScaleModel=*/kKQuantQ5KGemmScaleModel,
    /*foldModel=*/"kquant_dmin_bsums_min",
    /*hasMin=*/true,
    /*weightBlockStride=*/2816,
    /*weightQuantByteOffset=*/768,
    /*weightDminByteOffset=*/32,
    /*weightScalesByteOffset=*/64,
    /*weightQhByteOffset=*/256,
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/260,
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/1040,
    /*nSubblocks=*/8,
};

// The iq4_nl codebook decode-FAMILY discriminator (G3 M2, the FIRST codebook decode
// family): a FLAT 32-element block whose 4-bit weight nibble is an INDEX into a 16-entry
// NON-LINEAR int8 codebook (the weight is `codebook[nibble]`), ONE fp16 super-block scale,
// i32 IN-BLOCK accumulator, NO min / NO sub-block. It is the codebook analogue of the q4_0
// flat linear repack: NO dmin, NO 6-bit scales, NO bsums, NO qh SECOND plane -- the ONLY
// codebook-specific decode fact is the 16-entry kvalues table the MEMORY vluxei16 gather
// indexes. A request carrying it is a flat codebook contraction whose repack-SELECTED
// lowering CONSTRUCTS the codebook typed_repack_gem{v,m}_loop_body region (fold_model
// "codebook_flat_single_scale", decode_model "iq4_nl") carrying the
// repack_gem{v,m}_codebook_core brick. The GEMM lowering sets the 4-column-amortized
// variant on the constructed loop body (a pure I4 mirror). The next codebook sibling
// (iq4_xs) will ride the SAME core brick with a K-quant 6-bit signed sub-block scale.
constexpr llvm::StringLiteral kCodebookIq4NlScaleModel =
    "flat.fp16-single-scale-codebook-nomin";
constexpr llvm::StringLiteral kCodebookIq4NlGemmScaleModel =
    "flat.fp16-single-scale-codebook-4col-nomin";

// The iq4_xs codebook decode-family discriminator (the SECOND codebook sibling, COMPLETING
// the iq4 codebook pair): the SUPER-BLOCK (QK_K=256) analogue of iq4_nl. It rides the SAME
// codebook core brick (RepackGem{v,m}CodebookCoreOp) + the SAME 16-entry non-linear int8
// codebook, but the flat single-fp16-scale fold is replaced by a K-quant-style 6-bit SIGNED
// per-sub-block scale: each of the 8 sub-blocks carries a 6-bit scale assembled LANE-WISE
// from a scales_l LOW pair nibble + a scales_h HIGH 2-bit field (the q4_K vand/vsrl/vsll/vor
// bit-dance) then biased -32 (NO min / NO bsums / NO qh SECOND plane), sign-extended to i32
// and folded via vmacc onto the i32 sub-block dot. A request carrying it is a super-block
// codebook contraction whose repack-SELECTED lowering CONSTRUCTS the codebook
// typed_repack_gem{v,m}_loop_body region (fold_model "codebook_superblock_signed6_no_min",
// decode_model "iq4_xs") carrying the SHARED codebook core brick.
constexpr llvm::StringLiteral kCodebookIq4XsScaleModel =
    "superblock.fp16-signed6-scale-codebook-nomin";
constexpr llvm::StringLiteral kCodebookIq4XsGemmScaleModel =
    "superblock.fp16-signed6-scale-codebook-4col-nomin";

// The mxfp4 codebook decode-family discriminator (retirement_batch 4, the THIRD codebook
// sibling): the FLAT 32-element block analogue of iq4_nl WITH an E8M0 (micro-exponent)
// shared-exponent per-column scale instead of the fp16 single scale. It rides the SAME
// codebook core brick (RepackGem{v,m}CodebookCoreOp) + a 16-entry DOUBLED-E2M1 int8 fp4
// codebook, but the flat single-fp16-scale fold is replaced by the E8M0 bit-construction
// scale: each per-column weight scale is ONE E8M0 exponent BYTE reconstructed to
// 2^(e-128) by ggml's EXACT u32-domain bit dance (vzext_vf4 -> vsll/vmerge -> vreinterpret
// to f32, NO min, NO sub-block). A request carrying it is a flat codebook contraction whose
// repack-SELECTED lowering CONSTRUCTS the codebook typed_repack_gem{v,m}_loop_body region
// (fold_model "codebook_flat_e8m0_scale", decode_model "mxfp4") carrying the SHARED codebook
// core brick. The GEMM lowering sets the 4-column-amortized variant on the constructed loop
// body (a pure I4 mirror).
constexpr llvm::StringLiteral kCodebookMxfp4ScaleModel =
    "flat.e8m0-single-scale-codebook-nomin";
constexpr llvm::StringLiteral kCodebookMxfp4GemmScaleModel =
    "flat.e8m0-single-scale-codebook-4col-nomin";

// The q4_1 decode-FAMILY discriminator (the abstract request's committed
// scale_model WHAT): the asymmetric q4_1 flat nibble format = the q4_0 dual-fp16
// per-block scale (d_x * d_y) PLUS a single per-block MIN term (m_x * s_y). Routing
// (repack-vs-block-dot) stays fact-driven off the opponent facts; the decode FAMILY
// is keyed off this committed scale_model. A request carrying it is a q4_1 flat
// contraction whose repack-SELECTED lowering CONSTRUCTS the SAME typed_repack
// gem{v,m}_loop_body region as q4_0 (fold_model "lane_wise_vector_scale_min") --
// the SHARED q4_0 core + fold bricks, the core stamping weight_nibble_unsigned (the
// RAW-nibble [0,15] decode, NO offset-binary -8) and the fold stamping the single
// MIN-fold offset pair (m_x @ +32 weight, s_y @ +2 activation). The q4_1 min is the
// SIMPLIFIED single-scalar cousin of the q4_K 8-submin fold (no register-cliff): a
// plain per-block AlreadyLean lane-wise fold. The GEMM lowering sets the
// 4-column-amortized scale_model variant on the constructed loop body (a pure I4
// mirror).
constexpr llvm::StringLiteral kNibbleQ41ScaleModel =
    "dual-fp16-per-block-d_x.d_y-plus-min";
constexpr llvm::StringLiteral kNibbleQ41GemmScaleModel =
    "dual-fp16-per-block-d_x.d_y-plus-min-4col";

// The q5_0 decode-FAMILY discriminator (the abstract request's committed
// scale_model WHAT): the flat FIVE-bit nibble+qh format = the q4_0 dual-fp16
// per-block scale (d_x * d_y) with NO min, but each weight is
// `((nibble) | (qh_bit << 4)) - 16` -- a RAW unsigned nibble peel OR the
// transposed qh 5th bit, centered by -16. A request carrying it is a q5_0 flat
// contraction whose repack-SELECTED lowering CONSTRUCTS the SAME typed_repack
// gem{v,m}_loop_body region as q4_0 (fold_model "lane_wise_vector_scale", the
// d-only fold WHOLE) -- the SHARED q4_0 core + fold bricks, the core stamping
// weight_nibble_unsigned + weight_qh_byte_offset (@288) + weight_offset_bias (16)
// (the 5th-bit decode leaf). The constructed loop body carries the q4_0 fold
// scale_model (the ...-five-bit string is the routing discriminator only). NO
// min-term (unlike q4_1/q5_1).
constexpr llvm::StringLiteral kNibbleQ50ScaleModel =
    "dual-fp16-per-block-d_x.d_y-five-bit";

// The q5_1 decode-FAMILY discriminator (the abstract request's committed scale_model
// WHAT): the flat FIVE-bit nibble+qh format WITH a min -- the UNION of q5_0's 5th-bit
// qh decode and q4_1's single per-block MIN fold. Each weight is
// `(nibble) | (qh_bit << 4)` UNSIGNED in [0,31] (NO offset-binary -16, unlike q5_0 --
// the asymmetric bias lives in the separate per-block MIN scale); the fold is q4_1's
// dual-fp16 d_x*d_y PLUS the m_x*s_y min term over a block_q8_1 activation. A request
// carrying it CONSTRUCTS the SAME typed_repack gem{v,m}_loop_body region as q4_1
// (fold_model "lane_wise_vector_scale_min") -- the SHARED q4_0 core + fold bricks, the
// core stamping weight_nibble_unsigned + weight_qh_byte_offset (@320) with NO
// weight_offset_bias (the unsigned 5th-bit decode leaf), the fold stamping the single
// MIN offset pair. The constructed loop body carries the q4_1 fold scale_model (the
// ...-plus-min-five-bit string is the routing discriminator only).
constexpr llvm::StringLiteral kNibbleQ51ScaleModel =
    "dual-fp16-per-block-d_x.d_y-plus-min-five-bit";

// The q8_0 decode-FAMILY discriminator (the abstract request's committed scale_model
// WHAT): the flat FULL-int8 format = the q4_0 dual-fp16 per-block scale (d_x * d_y)
// with NO min, but each weight is a FULL signed int8 (NO nibble unpack, NO qh, NO
// offset -- qk=32 positions per block, one int8 weight byte each). It is the SIMPLEST
// flat variant. A request carrying it CONSTRUCTS the SAME typed_repack gem{v,m}_loop_body
// region as q4_0 (fold_model "lane_wise_vector_scale", the d-only fold WHOLE) -- the
// SHARED dual-fp16 fold brick + the q8_0 FULL-int8 CORE brick variant (the core stamping
// weight_full_i8: vle8 i8 + per-position vwmul/vwadd_wv i32 in-block accumulation, NO
// nibble decode). The constructed loop body carries the q4_0 fold scale_model.
constexpr llvm::StringLiteral kNibbleQ80ScaleModel =
    "dual-fp16-per-block-d_x.d_y-full-i8";

// The q4_0 base decode-family discriminator (the abstract request's committed scale_model
// WHAT for the plain q4_0 nibble): the dual-fp16 per-block d_x*d_y scale, NO min, offset-
// binary -8 nibble. It is the DEFAULT flat construction (no kNibbleQ*/kCodebook*/kKQuant*
// row matches). Named here as a registry key for the VLEN256-decode measured table below;
// it MIRRORS the "dual-fp16-per-block-d_x.d_y" the pass stamps on the constructed q4_0 loop
// body (RVVLowerQuantContraction ... builder.getStringAttr("dual-fp16-per-block-d_x.d_y")).
constexpr llvm::StringLiteral kNibbleQ40ScaleModel =
    "dual-fp16-per-block-d_x.d_y";

// [G8 六.3] The per-(decode-family, VLEN256, DECODE) BOARD-MEASURED repack-vs-block-dot
// registry. It is the SEED that FALSIFIES the selector's old blanket fact-3 rule ("VLEN256
// decode always declines repack", RVVContractionPathSelection.cpp), which mislabeled a
// q4_0-only 0.74x LOSS as a NON-per-format capability rule. The k1 GEVM sweep (casefile
// experiments/active/g8-stage3-attack/k1-gevm-sweep/evidence.md, commit ac5ea76f) MEASURED
// the VLEN256 decode repack-GEVM leaf per format on real k1/X60 silicon (cold median,
// K=2048, N>=10, 32MiB flush, byte-exact vs stock native-vec AND an independent oracle) and
// found the disposition SPLITS by format:
//   * q5_0 / q5_1  -> BENEFICIAL (1.190x / 1.306x): the contiguous block_q5x16 x16 weight
//                     stream + wide SIMD reduction (1 fused GEVM) out-VLENs the DIVERGENT
//                     per-column 5-bit qh block-dot (512 scattered vec_dot calls) at VLEN256.
//   * q4_0 / iq4_nl-> NEGATIVE (0.74x / 0.248x): q4_0's linear block-dot is already near-
//                     optimal (repack overhead nets a loss); iq4_nl's repack leaf is
//                     codebook-gather-bound (vsetvl 260 / 64 gather).
// This is registration-as-data -- the SAME status/metric/hook mechanism as the IME
// wide-vmadot kIMEWideFormatMeasurements registry, NOT a per-format C++ switch: a row is a
// BOARD FACT keyed on the committed decode-family scale_model WHAT (NOT the optional `quant`
// label; the pass already dispatches CONSTRUCTION on scale_model, and "delete quant, keep
// the facts" still routes identically). A scale_model with NO row is UNMEASURED -> the
// selector's conservative VLEN256-decode DECLINE (add a format = add a data row, 换键不改
// 条目). The selector consumes ONLY the resulting fact (vlen256DecodeRepackBeneficial) and
// stays BLIND to the format label.
enum class Vlen256DecodeRepackDisposition { Beneficial, Negative };
struct RepackVlen256DecodeMeasurement {
  llvm::StringRef scaleModel; ///< committed decode-family WHAT (registry key)
  Vlen256DecodeRepackDisposition disposition; ///< board-measured repack disposition
  llvm::StringRef metric;      ///< measured cold repack-vs-block-dot ratio
  llvm::StringRef metricsHook; ///< the board evidence provenance
};
constexpr RepackVlen256DecodeMeasurement kRepackVlen256DecodeMeasurements[] = {
    {kNibbleQ50ScaleModel, Vlen256DecodeRepackDisposition::Beneficial, "1.190x",
     "experiments/active/g8-stage3-attack/k1-gevm-sweep/evidence.md "
     "(q5_0@k1 VLEN256 decode repack-GEVM leaf 1.190x vs stock native-vec, "
     "byte-exact; k1 GEVM sweep ac5ea76f)"},
    {kNibbleQ51ScaleModel, Vlen256DecodeRepackDisposition::Beneficial, "1.306x",
     "experiments/active/g8-stage3-attack/k1-gevm-sweep/evidence.md "
     "(q5_1@k1 VLEN256 decode repack-GEVM leaf 1.306x vs stock native-vec, "
     "byte-exact; k1 GEVM sweep ac5ea76f)"},
    {kNibbleQ40ScaleModel, Vlen256DecodeRepackDisposition::Negative, "0.74x",
     "experiments/active/g8-stage3-attack/k1-gevm-sweep/evidence.md "
     "(q4_0@k1 VLEN256 decode repack 0.74x LOSS -- near-optimal linear block-dot; "
     "the original fact-3 seed, now scoped per-format; k1 GEVM sweep ac5ea76f)"},
    {kCodebookIq4NlScaleModel, Vlen256DecodeRepackDisposition::Negative, "0.248x",
     "experiments/active/g8-stage3-attack/k1-gevm-sweep/evidence.md "
     "(iq4_nl@k1 VLEN256 decode repack-GEVM leaf 0.248x LOSS -- codebook-gather-bound, "
     "vsetvl 260 / 64 gather; k1 GEVM sweep ac5ea76f)"},
};

// Pure DATA lookup: the board-measured VLEN256-decode repack disposition for the committed
// decode-family scale_model, or nullopt when the format has NO measured row (=> the
// selector's conservative VLEN256-decode decline). Mirrors lookupWideFormatMeasuredNegative.
std::optional<Vlen256DecodeRepackDisposition>
lookupRepackVlen256Decode(llvm::StringRef scaleModel) {
  for (const RepackVlen256DecodeMeasurement &m : kRepackVlen256DecodeMeasurements)
    if (m.scaleModel == scaleModel)
      return m.disposition;
  return std::nullopt;
}

// The iq2_xxs GRID CODEBOOK + SIGN-PLANE decode-FAMILY discriminator (the abstract
// request's committed scale_model WHAT): the FIRST grid decode sibling (retirement_batch
// 3). A QK_K=256 super-block whose per-(sub-block, group, column) 8-bit GRID INDEX selects
// an 8-byte ENTRY from the FIXED 256-entry iq2xxs_grid (each entry = 8 packed int8 grid
// bytes), whose 7-bit SIGN SELECTOR gathers a +-1 byte from the DERIVED signs64 plane
// (folded onto the grid), with a per-sub-block int8 ls scale ([1,31], NO -32 bias, NO min)
// and a trailing 0.125 (1/8) factor. NO dmin, NO 6-bit scales, NO bsums, NO qh second
// plane, NO codebook array -- the ONLY grid-specific decode facts (grid-index / ls-scale /
// sign-selector byte offsets + n_subblocks) ride on the NEW grid core brick, and the FIXED
// grid + signs64 planes stay DERIVED canonical static const tables (NEVER op attrs -- the
// signs64 op-attr blocker never recurs). A request carrying it is a grid contraction whose
// repack-SELECTED lowering CONSTRUCTS the grid typed_repack_gem{v,m}_loop_body region
// (fold_model "grid_sign_single_scale_eighth", decode_model "iq2_xxs") carrying the
// repack_gem{v,m}_grid_core brick. The GEMM lowering sets the 4-column-amortized scale_model
// variant on the constructed loop body (a pure I4 mirror). The next grid siblings (iq2_xs /
// iq2_s) will ride the SAME core brick with a DUAL per-group ls scale.
constexpr llvm::StringLiteral kGridIq2XxsScaleModel =
    "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth";
constexpr llvm::StringLiteral kGridIq2XxsGemmScaleModel =
    "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth-4col";

// The iq2_xs / iq2_s DUAL-scale GRID CODEBOOK + SIGN-PLANE decode-FAMILY discriminators
// (the abstract request's committed scale_model WHAT): the DUAL-ls grid decode siblings of
// iq2_xxs (retirement_batch 3). Same QK_K=256 super-block / 8-byte grid ENTRY gather / sign
// fold / NO min / trailing 0.125 as iq2_xxs, with TWO structural deltas: (a) the grid INDEX
// is a u16 strip (9-bit iq2_xs / 10-bit iq2_s, cannot live in a byte, so vle16 direct, NO
// vzext); (b) each 32-lane sub-block splits into TWO ls-weighted group-halves (ls1 groups
// 0-1, ls2 groups 2-3). iq2_xs signs are DERIVED signs64 (7-bit selector, like iq2_xxs);
// iq2_s signs are DIRECT signs256 (explicit 8-bit sign byte). A request carrying either
// scale_model is a grid contraction whose repack-SELECTED lowering CONSTRUCTS the grid
// typed_repack_gem{v,m}_loop_body region (fold_model "grid_sign_dualscale_eighth",
// decode_model "iq2_xs" / "iq2_s") carrying the SAME repack_gem{v,m}_grid_core brick. The
// FIXED grid + signs planes stay DERIVED canonical static const tables (NEVER op attrs).
constexpr llvm::StringLiteral kGridIq2XsScaleModel =
    "superblock-d.fp16-grid-sign-dualscale-nomin-eighth";
constexpr llvm::StringLiteral kGridIq2XsGemmScaleModel =
    "superblock-d.fp16-grid-sign-dualscale-4col-nomin-eighth";
constexpr llvm::StringLiteral kGridIq2SScaleModel =
    "superblock-d.fp16-grid-explicitsign-dualscale-nomin-eighth";
constexpr llvm::StringLiteral kGridIq2SGemmScaleModel =
    "superblock-d.fp16-grid-explicitsign-dualscale-4col-nomin-eighth";

// The iq1_s TERNARY-DELTA GRID decode-FAMILY discriminator (C4a-2): the FOURTH grid
// sibling and the first NON-iq2 one. Same QK_K=256 super-block / 8-byte grid ENTRY
// gather as the iq2 rows (which is why it was the lowest-risk 4th row), with FOUR
// structural deltas: (a) the grid is the 2048-entry TERNARY iq1s_grid and its bytes are
// ALREADY signed {-1,0,+1}, so there is NO sign plane and NO sign fold -- the gathered
// grid byte IS the weight; (b) the 11-bit grid index is ASSEMBLED at repack time from
// qs[l] | (((qh[ib] >> 3l) & 7) << 8) and lands as a u16 strip; (c) the SINGLE ls per
// sub-block is DERIVED at repack time as 2*((qh[ib] >> 12) & 7) + 1 (8 values, [1,15]);
// (d) a per-sub-block +-1 DELTA (1 - 2*((qh[ib] >> 15) & 1)) rides the sign-plane byte
// offset slot as a DELTA strip and feeds a SECOND integer accumulator sumi1 = sum_ib
// ls*delta*(bsums[2ib]+bsums[2ib+1]) over the ACTIVATION bsums -- the ONLY grid row that
// reads them. The fold is therefore NOT the iq2 store-side 0.125: it is the per-block
// dual-accumulator sumf += d*(sumi + 0.125f*sumi1) (fold_model
// "grid_ternary_delta_eighth"), byte-exact to ggml_vec_dot_iq1_s_q8_K. The FIXED
// 2048-entry grid stays a DERIVED emit-period static const table (NEVER an op attr).
constexpr llvm::StringLiteral kGridIq1SScaleModel =
    "superblock-d.fp16-grid-ternary-delta-singlescale-nomin-eighth";
constexpr llvm::StringLiteral kGridIq1SGemmScaleModel =
    "superblock-d.fp16-grid-ternary-delta-singlescale-4col-nomin-eighth";

// The iq1_m decode-FAMILY discriminator (the abstract request's committed scale_model
// WHAT): iq1_s's TERNARY-DELTA sibling, and the C4a-3 row. It shares iq1_s's grid
// (the SAME 2048 ternary literals -- ggml's ggml_vec_dot_iq1_m_q8_K indexes `iq1s_grid`
// itself), its 8-byte entry, its "no sign plane, the gathered byte IS the weight" decode
// and the 0.125 dual-accumulator fold expression; and it shares iq2_xs/iq2_s's DUAL ls
// arity. Its THREE structural deltas vs iq1_s, all checkable against ggml:
//   (a) NO inline d. block_iq1_m is 56 B of qs[32] + qh[16] + scales[8] with NO ggml_half
//       member at all (ggml-common.h). The fp16 super-block d is ASSEMBLED from four
//       nibbles scattered across the scales words: scale.u16 = (sc[0] >> 12) |
//       ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000). The repack
//       does this ONCE, so the kernel sees an ordinary inline fp16 d strip.
//   (b) DUAL ls. ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 7)+1 over groups 0-1, ls2 =
//       2*((sc[ib/2] >> (6*(ib%2)+3)) & 7)+1 over groups 2-3 -- the SAME shape as
//       iq2_xs/iq2_s, so it REUSES the dual-ls strip layout (foldModel is dual, arity Dual).
//   (c) PER-GROUP delta + NO bsums. iq1_s has ONE +-1 per sub-block and its delta term is
//       exactly ls*delta*(bsums[2ib]+bsums[2ib+1]). iq1_m has FOUR INDEPENDENT +-1 per
//       sub-block (delta[l] = qh[l/2] & (l%2 ? 0x80 : 0x08) ? -1 : 1) and its delta term
//       needs the per-GROUP-of-8 activation sum. block_q8_K's bsums are sums over groups
//       of SIXTEEN, and the two 8-groups inside one bsums entry carry INDEPENDENT delta
//       signs, so a bsums entry CANNOT express it. iq1_m therefore stamps NO
//       activation_bsums_byte_offset and accumulates the group sum IN-KERNEL from the same
//       8 activation scalars the grid dot already reads. This is what makes the fold
//       "grid_ternary_delta_groupsum_eighth" a leaf of its own rather than an ls-arity
//       variant of iq1_s's.
// The grid index is qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700), assembled at
// repack time into a u16 strip. Byte-exact to ggml_vec_dot_iq1_m_q8_K. The FIXED 2048-entry
// grid stays a DERIVED emit-period static const table (NEVER an op attr).
constexpr llvm::StringLiteral kGridIq1MScaleModel =
    "superblock-d.fp16-grid-ternary-delta-groupsum-dualscale-nomin-eighth";
constexpr llvm::StringLiteral kGridIq1MGemmScaleModel =
    "superblock-d.fp16-grid-ternary-delta-groupsum-dualscale-4col-nomin-eighth";

// The iq3_xxs DUAL-ENTRY GRID CODEBOOK + SIGN-PLANE decode-FAMILY discriminator (C4a-4):
// the SIXTH grid sibling, and the one that is iq2_xxs in every respect but the one that
// matters. Checkable against ggml_vec_dot_iq3_xxs_q8_K, what it SHARES with iq2_xxs is
// the whole decode skeleton: `ls = 2*(aux32 >> 28) + 1` (SINGLE per sub-block), the
// `ksigns_iq2xs[(aux32 >> 7*l) & 127]` 7-bit sign selector tested by `kmask_iq2xs[j]`
// (the SAME sign plane -- ggml reads ksigns_iq2xs by name from iq3_xxs too), the single
// i32 `bsum += sumi*ls` / `sumf += d*bsum` chain, and a 256-entry grid.
//
// Its TWO structural deltas:
//   (a) DUAL GRID ENTRY per group. iq3xxs_grid is `uint32_t[256]`, not uint64_t, so one
//       entry covers only 4 of a group's 8 lanes and the decode reads TWO
//       (`grid1 = iq3xxs_grid + q3[2*l+0]`, `grid2 = iq3xxs_grid + q3[2*l+1]`, then
//       `for j<4: grid1[j]*q8[j+0]; grid2[j]*q8[j+4]`). This is what forces a new EMITTER
//       LEAF rather than a parameter: every other grid leaf hoists ONE gather base per
//       group and walks j = 0..7 against it. The repacked grid-index strip doubles to
//       match (8 u8 indices per sub-block, vs iq2_xxs's 4).
//   (b) The store constant is 0.25f (`*s = 0.25f * sumf`), not the iq2 rows' 0.125f --
//       the SAME fold shape with a different constant, which rides the plan's
//       storeScaleLiteral as data.
// The plain block_iq3_xxs is 98 B (fp16 d + qs[3*QK_K/8] = 96), whose qs region SPLITS at
// QK_K/4 = 64 (`gas = x[i].qs + QK_K/4`) into 64 raw grid-index bytes + 8 uint32 aux
// words; the repack pulls the aux words' ls + sign selectors into flat strips ONCE so the
// kernel never sees one. Byte-exact to ggml_vec_dot_iq3_xxs_q8_K. The FIXED 256-entry
// uint32 grid + the DERIVED signs64 plane stay emit-period static const tables (NEVER op
// attrs).
constexpr llvm::StringLiteral kGridIq3XxsScaleModel =
    "superblock-d.fp16-grid-sign-dual-entry-4bit-scale-nomin-quarter";
constexpr llvm::StringLiteral kGridIq3XxsGemmScaleModel =
    "superblock-d.fp16-grid-sign-dual-entry-4bit-scale-4col-nomin-quarter";

// The iq3_s DUAL-ENTRY + EXPLICIT-SIGN grid decode-FAMILY discriminator (C4a-5): the
// SEVENTH and LAST grid sibling. Its interest is that it resembles iq3_xxs closely enough
// to reuse that row's whole emitter leaf, yet on the two axes below it sides with a
// DIFFERENT sibling -- so the family resemblance is not a decode fact and the WHAT has to
// say which.
//
// Read off ggml_vec_dot_iq3_s_q8_K:
//   * DUAL GRID ENTRY per group, like iq3_xxs: iq3s_grid is `uint32_t[512]`, so one entry
//     covers 4 of a group's 8 lanes and the decode reads TWO
//     (`grid1 = iq3s_grid + (qs[2*l+0] | ((qh[ib32] << (8-2*l)) & 256))`, likewise grid2
//     with shift 7-2*l, then `for j<4: grid1[j]*q8[j+0]; grid2[j]*q8[j+4]`). Same nest,
//     same leaf.
//   * 9-BIT INDEX (0..511), unlike iq3_xxs's raw byte: the low 8 bits come from qs and bit
//     8 from a qh bit (`(qh[ib32] >> e) & 1` for index slot e). Assembled ONCE at repack
//     into a u16 strip, exactly as iq2_xs's 9-bit and iq1_s's 11-bit indices are.
//   * EXPLICIT SIGNS, like iq2_s and NOT like iq3_xxs: block_iq3_s carries a real
//     `uint8_t signs[QK_K/8]` plane (one byte per 8-lane group) and the decode tests it
//     directly (`signs[l] & kmask_iq2xs[j]`). There is NO aux word and NO ksigns_iq2xs
//     selector indirection -- hence "explicitsign" in the WHAT below, the same token
//     iq2_s's WHAT carries, and hence the DERIVED signs256 plane rather than signs64.
//   * SINGLE ls per 32-element sub-block, NOT dual: ggml names two scales ls1/ls2 per
//     iteration but its loop steps `ib32 += 2` and spends one on EACH sub-block
//     (`ls1 = 2*(scales[ib32/2] & 0xf) + 1` for sub-block ib32, the high nibble for
//     ib32+1). Two sub-blocks share a scales byte; a sub-block has one scale.
//   * NO STORE CONSTANT: ggml ends `*s = sumf`, not 0.25f * sumf (iq3_xxs) and not
//     0.125f * sumf (every iq2 row). Carried as the plan's storeScaleLiteral "1.0f" --
//     see weft::GridDecodePlan::storeScaleLiteral for why it is written that way.
// The plain block_iq3_s is 110 B (fp16 d + qs[QK_K/4] + qh[QK_K/32] + signs[QK_K/8] +
// scales[QK_K/64] = 2+64+8+32+4), and unlike iq3_xxs there is no double-duty region: each
// plane is its own field. Byte-exact to ggml_vec_dot_iq3_s_q8_K. The FIXED 512-entry
// uint32 grid + the DERIVED signs256 plane stay emit-period static const tables (NEVER op
// attrs).
constexpr llvm::StringLiteral kGridIq3SScaleModel =
    "superblock-d.fp16-grid-explicitsign-dual-entry-4bit-scale-nomin-unit";
constexpr llvm::StringLiteral kGridIq3SGemmScaleModel =
    "superblock-d.fp16-grid-explicitsign-dual-entry-4bit-scale-4col-nomin-unit";

// The iq4_nl NON-LINEAR int8 codebook (kvalues_iq4nl) the codebook decode indexes. The
// abstract quant_contraction request carries NO codebook; the compiler RECONSTRUCTS this
// table (the load-bearing WHAT the memory gather reads, stamped onto the core brick).
constexpr int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                      1,    13,   25,  38,  53,  69,  89,  113};

// The mxfp4 DOUBLED-E2M1 int8 fp4 codebook (kvalues_mxfp4) the codebook decode indexes.
// The abstract quant_contraction request carries NO codebook; the compiler RECONSTRUCTS
// this table (the load-bearing WHAT the memory gather reads, stamped onto the core brick).
constexpr int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                      0, -1, -2, -3, -4, -6, -8, -12};

// The repacked block_iq4_nlx16 / block_q8_0{,x4} byte facts the codebook lowering
// RECONSTRUCTS (the stage-C x16 materialization the DECLARED weight_layout_contract
// asserts): the 16-inline-fp16-d + 256-nibble weight super-block stride (288), the weight
// nibble quant byte offset (32), the PLAIN block_q8_0 GEVM activation stride (34) + its
// quant offset (2), the INTERLEAVED block_q8_0x4 GEMM activation stride (136) + its quant
// offset (8), and the 16-entry non-linear int8 codebook. These MIRROR the retired
// monolithic iq4_nl repack op verifiers' pins.
struct CodebookDecodeFacts {
  llvm::StringRef decodeModel;              // "iq4_nl" / "iq4_xs" (core brick)
  llvm::StringRef gemmScaleModel;           // the 4-col GEMM loop-op scale_model
  llvm::StringRef foldModel;                // loop-body fold_model
  std::int64_t weightBlockStride;           // 288 (iq4_nl) / 2176 (iq4_xs)
  std::int64_t weightQuantByteOffset;       // 32 (iq4_nl) / 128 (iq4_xs) nibble plane
  std::int64_t gevmActivationBlockStride;   // 34 (plain block_q8_0) / 292 (block_q8_K)
  std::int64_t gevmActivationQuantByteOffset;    // 2 (iq4_nl) / 4 (iq4_xs)
  std::int64_t gemmActivationBlockStride;   // 136 (block_q8_0x4) / 1168 (block_q8_Kx4)
  std::int64_t gemmActivationQuantByteOffset;    // 8 (iq4_nl) / 16 (iq4_xs)
  // The iq4_xs SUPER-BLOCK signed-6 scale facts (0 for the iq4_nl flat fold): the scales_l
  // LOW pair region byte offset (+64), the scales_h HIGH 2-bit region byte offset (+32),
  // and the sub-block count (8). The lowering stamps them onto the loop body op's OPTIONAL
  // super-block attrs ONLY under the codebook super-block fold.
  std::int64_t weightScalesLowByteOffset;   // 64 (iq4_xs) / 0 (iq4_nl)
  std::int64_t weightScalesHighByteOffset;  // 32 (iq4_xs) / 0 (iq4_nl)
  std::int64_t nSubblocks;                  // 8 (iq4_xs) / 0 (iq4_nl)
  llvm::ArrayRef<int8_t> codebook;          // the 16-entry non-linear kvalues (SHARED)
};

constexpr CodebookDecodeFacts kIq4NlDecodeFacts = {
    /*decodeModel=*/"iq4_nl",
    /*gemmScaleModel=*/kCodebookIq4NlGemmScaleModel,
    /*foldModel=*/"codebook_flat_single_scale",
    /*weightBlockStride=*/288,
    /*weightQuantByteOffset=*/32,
    /*gevmActivationBlockStride=*/34,
    /*gevmActivationQuantByteOffset=*/2,
    /*gemmActivationBlockStride=*/136,
    /*gemmActivationQuantByteOffset=*/8,
    /*weightScalesLowByteOffset=*/0,
    /*weightScalesHighByteOffset=*/0,
    /*nSubblocks=*/0,
    /*codebook=*/llvm::ArrayRef<int8_t>(kvalues_iq4nl),
};

// iq4_xs: the SUPER-BLOCK codebook sibling. It RECONSTRUCTS the block_iq4_xsx16 x16 weight
// facts (stride 2176, nibbles @128, scales_l LOW pair @64, scales_h HIGH 2-bit @32) + the
// block_q8_K activation facts (292/4 GEVM, interleaved block_q8_Kx4 1168/16 GEMM) + the SAME
// 16-entry non-linear int8 codebook iq4_nl uses (the abstract request carries none). The
// 6-bit SIGNED per-sub-block scale (biased -32, NO min) rides the codebook super-block fold.
constexpr CodebookDecodeFacts kIq4XsDecodeFacts = {
    /*decodeModel=*/"iq4_xs",
    /*gemmScaleModel=*/kCodebookIq4XsGemmScaleModel,
    /*foldModel=*/"codebook_superblock_signed6_no_min",
    /*weightBlockStride=*/2176,
    /*weightQuantByteOffset=*/128,
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*weightScalesLowByteOffset=*/64,
    /*weightScalesHighByteOffset=*/32,
    /*nSubblocks=*/8,
    /*codebook=*/llvm::ArrayRef<int8_t>(kvalues_iq4nl),
};

// mxfp4: the FLAT codebook sibling WITH the E8M0 shared-exponent scale (retirement_batch 4).
// It RECONSTRUCTS the block_mxfp4x16 x16 weight facts (stride 272, E8M0 exponent strip @0,
// nibbles @16) + the plain block_q8_0 activation facts (34/2 GEVM, interleaved block_q8_0x4
// 136/8 GEMM) + the 16-entry DOUBLED-E2M1 int8 codebook (the abstract request carries none).
// It is the FLAT sibling of iq4_nl (nSubblocks == 0, no super-block scales), so the E8M0 fold
// rides the codebook flat fold with NO scales_l/scales_h/n_subblocks attrs.
constexpr CodebookDecodeFacts kMxfp4DecodeFacts = {
    /*decodeModel=*/"mxfp4",
    /*gemmScaleModel=*/kCodebookMxfp4GemmScaleModel,
    /*foldModel=*/"codebook_flat_e8m0_scale",
    /*weightBlockStride=*/272,
    /*weightQuantByteOffset=*/16,
    /*gevmActivationBlockStride=*/34,
    /*gevmActivationQuantByteOffset=*/2,
    /*gemmActivationBlockStride=*/136,
    /*gemmActivationQuantByteOffset=*/8,
    /*weightScalesLowByteOffset=*/0,
    /*weightScalesHighByteOffset=*/0,
    /*nSubblocks=*/0,
    /*codebook=*/llvm::ArrayRef<int8_t>(kvalues_mxfp4),
};

// The block_iq2_xxsx16 / block_q8_K{,x4} byte facts the GRID lowering RECONSTRUCTS (the
// stage-C x16 materialization the DECLARED weight_layout_contract asserts): the
// 16-inline-fp16-d + 8 per-sub-block int8 ls strips (+32) + 512 grid-index bytes (+160) +
// 512 sign-selector bytes (+672) weight super-block (stride 1184); the plain block_q8_K
// GEVM activation (stride 292, fp32 d @0, int8 quants @4) + the INTERLEAVED block_q8_Kx4
// GEMM activation (stride 1168, quants @16). Unlike the codebook facts there is NO codebook
// array: the FIXED 256-entry grid + the DERIVED signs64 +-1 plane are emit-period static
// const tables (NEVER op attrs). The grid-index / ls-scale / sign-selector byte offsets +
// n_subblocks ride on the grid core brick (NOT the loop body op). These MIRROR the retired
// monolithic iq2_xxs repack op verifiers' pins.
struct Iq2GridDecodeFacts {
  llvm::StringRef decodeModel;              // "iq2_xxs" (core brick)
  llvm::StringRef gemmScaleModel;           // the 4-col GEMM loop-op scale_model
  llvm::StringRef foldModel;                // loop-body fold_model
  std::int64_t weightBlockStride;           // 1184
  std::int64_t weightGridIdxByteOffset;     // 160 (grid-index plane)
  std::int64_t weightLsByteOffset;          // 32 (per-sub-block int8 ls plane)
  // The sign-plane byte offset. For the iq2 rows this addresses the sign SELECTOR
  // strip (672 / 1312). For the iq1_s TernaryDelta row there is no sign plane, so
  // this SAME slot addresses the per-sub-block +-1 DELTA strip (160) instead.
  std::int64_t weightSignByteOffset;
  std::int64_t gevmActivationBlockStride;   // 292 (plain block_q8_K)
  std::int64_t gevmActivationQuantByteOffset;    // 4
  // The activation int16 bsums byte offsets. NON-ZERO ONLY for the iq1_s delta fold
  // (260 plain block_q8_K / 1040 interleaved block_q8_Kx4, group16-major /
  // column-minor); 0 for the iq2 rows, which never read bsums and therefore must
  // NOT stamp the attr (the loop-body verifier rejects it outside the two folds
  // that read it).
  std::int64_t gevmActivationBsumsByteOffset;
  std::int64_t gemmActivationBlockStride;   // 1168 (block_q8_Kx4)
  std::int64_t gemmActivationQuantByteOffset;    // 16
  std::int64_t gemmActivationBsumsByteOffset;
  std::int64_t nSubblocks;                  // 8
};

constexpr Iq2GridDecodeFacts kIq2XxsDecodeFacts = {
    /*decodeModel=*/"iq2_xxs",
    /*gemmScaleModel=*/kGridIq2XxsGemmScaleModel,
    /*foldModel=*/"grid_sign_single_scale_eighth",
    /*weightBlockStride=*/1184,
    /*weightGridIdxByteOffset=*/160,
    /*weightLsByteOffset=*/32,
    /*weightSignByteOffset=*/672,
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/0, // iq2_xxs reads no bsums.
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/0,
    /*nSubblocks=*/8,
};

// The block_iq2_xsx16 / block_iq2_sx16 DUAL-scale grid facts the GRID lowering
// RECONSTRUCTS (the stage-C x16 materialization). Both share the SAME repacked strip layout
// (stride 1824: 16 fp16 d + 256 dual ls @32 + 1024 u16 grid-index @288 + 512 sign-selector
// @1312) and the SAME plain block_q8_K GEVM (292/4) / interleaved block_q8_Kx4 GEMM
// (1168/16) activation facts -- iq2_xs and iq2_s differ ONLY in the DERIVED grid table
// (512-entry iq2xs_grid + signs64 vs 1024-entry iq2s_grid + explicit signs256), selected in
// the emitter by the decode_model (the FIXED planes stay DERIVED static const, NEVER op
// attrs). The dual foldModel "grid_sign_dualscale_eighth" splits each sub-block into ls1
// (groups 0-1) / ls2 (groups 2-3), sharing the SAME weight_ls_byte_offset base.
constexpr Iq2GridDecodeFacts kIq2XsDecodeFacts = {
    /*decodeModel=*/"iq2_xs",
    /*gemmScaleModel=*/kGridIq2XsGemmScaleModel,
    /*foldModel=*/"grid_sign_dualscale_eighth",
    /*weightBlockStride=*/1824,
    /*weightGridIdxByteOffset=*/288,
    /*weightLsByteOffset=*/32,
    /*weightSignByteOffset=*/1312,
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/0, // iq2_xs reads no bsums.
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/0,
    /*nSubblocks=*/8,
};

constexpr Iq2GridDecodeFacts kIq2SDecodeFacts = {
    /*decodeModel=*/"iq2_s",
    /*gemmScaleModel=*/kGridIq2SGemmScaleModel,
    /*foldModel=*/"grid_sign_dualscale_eighth",
    /*weightBlockStride=*/1824,
    /*weightGridIdxByteOffset=*/288,
    /*weightLsByteOffset=*/32,
    /*weightSignByteOffset=*/1312,
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/0, // iq2_s reads no bsums.
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/0,
    /*nSubblocks=*/8,
};

// iq1_s (C4a-2): the TERNARY-DELTA grid sibling. It RECONSTRUCTS the block_iq1_sx16 x16
// weight facts -- a repack layout this line DESIGNED (ggml's block_iq1_s is 50 B of
// PACKED qs/qh; the repack DECODES the qh word once, at repack time, into three flat
// per-column strips so the kernel never re-derives it):
//
//   d[16]         fp16 @ +0     (32 B)
//   ls[8][16]     int8 @ +32    (128 B)  2*((qh>>12)&7)+1, in [1,15]
//   delta[8][16]  int8 @ +160   (128 B)  1-2*((qh>>15)&1), in {-1,+1}
//   gidx[8][4][16] u16 @ +288   (1024 B) qs[l] | (((qh>>3l)&7)<<8), in [0,2047]
//   ------------------------------------ stride 1312
//
// plus the plain block_q8_K activation (292, quants @4, bsums @260) and the INTERLEAVED
// block_q8_Kx4 GEMM activation (1168, quants @16 as pos*4+c, bsums @1040 as g16*4+c).
// The delta strip rides the weightSignByteOffset slot; iq1_s is the ONLY grid row with
// non-zero bsums offsets.
constexpr Iq2GridDecodeFacts kIq1SDecodeFacts = {
    /*decodeModel=*/"iq1_s",
    /*gemmScaleModel=*/kGridIq1SGemmScaleModel,
    /*foldModel=*/"grid_ternary_delta_eighth",
    /*weightBlockStride=*/1312,
    /*weightGridIdxByteOffset=*/288,
    /*weightLsByteOffset=*/32,
    /*weightSignByteOffset=*/160, // the +-1 DELTA strip (no sign plane exists).
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/260,
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/1040,
    /*nSubblocks=*/8,
};

// iq1_m (C4a-3): iq1_s's DUAL-ls, PER-GROUP-delta ternary sibling. It RECONSTRUCTS the
// block_iq1_mx16 x16 weight facts -- again a repack layout this line DESIGNED (ggml's
// block_iq1_m is 56 B of PACKED qs/qh/scales with NO inline d; the repack ASSEMBLES the
// scattered fp16 d nibbles and DECODES the qh/scales bits once, at repack time, into four
// flat per-column strips so the kernel never re-derives them):
//
//   d[16]           fp16 @ +0     (32 B)   the ASSEMBLED iq1m_scale_t fp16 (see (a) above)
//   ls[8][2][16]    int8 @ +32    (256 B)  ls1/ls2 per sub-block, in [1,15] -- the SAME
//                                          (ib*2 + gh)*16 dual-ls strip shape iq2_xs uses
//   delta[8][4][16] int8 @ +288   (512 B)  PER-GROUP 1-2*bit, in {-1,+1} (FOUR per
//                                          sub-block, vs iq1_s's one)
//   gidx[8][4][16]  u16  @ +800   (1024 B) qs[l] | ((qh[l/2] << (8-4*(l%2))) & 0x700),
//                                          in [0,2047]
//   ---------------------------------------- stride 1824
//
// plus the plain block_q8_K activation (292, quants @4) and the INTERLEAVED block_q8_Kx4
// GEMM activation (1168, quants @16 as pos*4+c). The delta strip rides the
// weightSignByteOffset slot (as it does for iq1_s -- a TernaryDelta row has no sign plane
// to put there).
//
// NOTE the bsums offsets are ZERO, and that is a FACT, not an omission: iq1_m's delta term
// needs per-GROUP-of-8 activation sums and block_q8_K's bsums are per-SIXTEEN, so the bsums
// plane is INEXPRESSIVE for it (see (c) above). The loop-body verifier rejects
// activation_bsums_byte_offset outside the two folds that READ it, so stamping one here
// would be rejected -- correctly.
constexpr Iq2GridDecodeFacts kIq1MDecodeFacts = {
    /*decodeModel=*/"iq1_m",
    /*gemmScaleModel=*/kGridIq1MGemmScaleModel,
    /*foldModel=*/"grid_ternary_delta_groupsum_eighth",
    /*weightBlockStride=*/1824,
    /*weightGridIdxByteOffset=*/800,
    /*weightLsByteOffset=*/32,
    /*weightSignByteOffset=*/288, // the PER-GROUP +-1 DELTA strip (no sign plane exists).
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/0, // per-16 bsums CANNOT express a per-8 group sum.
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/0,
    /*nSubblocks=*/8,
};

// iq3_xxs (C4a-4): the DUAL-ENTRY grid sibling. It RECONSTRUCTS the block_iq3_xxsx16 x16
// weight facts -- a repack layout this line DESIGNED. ggml's block_iq3_xxs is 98 B (fp16 d
// + 96 B qs) in which the qs region does double duty: `gas = x[i].qs + QK_K/4` splits it
// into 64 raw u8 grid indices and 8 uint32 aux words, each aux word packing one
// sub-block's ls (bits 28-31) and its four 7-bit sign selectors (bits 0-27). The repack
// decodes those aux words ONCE into flat per-column strips, so the kernel never sees one:
//
//   d[16]           fp16 @ +0     (32 B)
//   ls[8][16]       int8 @ +32    (128 B)  2*(aux32 >> 28) + 1, in [1,31]
//   gidx[8][8][16]  u8   @ +160   (1024 B) the raw q3 index bytes -- EIGHT per sub-block
//                                          (TWO per 8-lane group), the strip that
//                                          DOUBLES vs iq2_xxs's four
//   ssel[8][4][16]  u8   @ +1184  (512 B)  (aux32 >> 7*l) & 127 -- FOUR per sub-block
//                                          (ONE per group, as iq2_xxs has)
//   ---------------------------------------- stride 1696
//
// plus the plain block_q8_K activation (292, quants @4) and the INTERLEAVED block_q8_Kx4
// GEMM activation (1168, quants @16 as pos*4+c). The 1696 stride is iq2_xxs's 1184 plus
// the extra 512 B of grid index; every other strip is identical in shape and offset.
//
// NOTE the bsums offsets are ZERO, and that is a FACT: iq3_xxs's fold is the
// single-accumulator SignScaleStore shape (`bsum += sumi*ls`, `sumf += d*bsum`,
// `*s = 0.25f*sumf`) with no delta term at all, so there is no second accumulator to feed
// and nothing reads a bsums plane. The loop-body verifier rejects
// activation_bsums_byte_offset outside the two folds that READ it, so stamping one here
// would be rejected -- correctly.
constexpr Iq2GridDecodeFacts kIq3XxsDecodeFacts = {
    /*decodeModel=*/"iq3_xxs",
    /*gemmScaleModel=*/kGridIq3XxsGemmScaleModel,
    /*foldModel=*/"grid_sign_dual_entry_single_scale_quarter",
    /*weightBlockStride=*/1696,
    /*weightGridIdxByteOffset=*/160,
    /*weightLsByteOffset=*/32,
    /*weightSignByteOffset=*/1184, // a REAL sign-selector strip (not a delta slot).
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/0, // iq3_xxs reads no bsums (no delta term).
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/0,
    /*nSubblocks=*/8,
};

// iq3_s (C4a-5): the LAST grid sibling. It RECONSTRUCTS the block_iq3_sx16 x16 weight
// facts -- a repack layout this line DESIGNED, and it is iq3_xxs's layout with ONE strip
// widened. ggml's block_iq3_s is 110 B with FOUR separate weight planes (no double-duty
// qs region, unlike iq3_xxs's aux-word split):
//   d       fp16 @ +0    (2 B)
//   qs[64]       @ +2            the low 8 bits of 64 grid indices (8 per sub-block)
//   qh[8]        @ +66           bit 8 of each: index slot e of sub-block ib takes
//                                (qh[ib] >> e) & 1
//   signs[32]    @ +74           ONE explicit sign byte per 8-lane group
//   scales[4]    @ +106          TWO sub-blocks' ls nibbles per byte
// The repack assembles the 9-bit index and splits the scales byte ONCE into flat
// per-column strips, so the kernel never sees a qh bit or a nibble:
//
//   d[16]            fp16 @ +0     (32 B)
//   ls[8][16]        int8 @ +32    (128 B)  2*nibble + 1, in [1,31] -- SINGLE per
//                                           sub-block (see the WHAT above; the byte
//                                           holding two of them is packing, not arity)
//   gidx[8][8][16]   u16  @ +160   (2048 B) the ASSEMBLED 9-bit index, EIGHT per
//                                           sub-block (TWO per group). u16, not u8:
//                                           this is the ONE strip that differs from
//                                           iq3_xxs's layout, and it differs because
//                                           511 does not fit a byte
//   signs[8][4][16]  u8   @ +2208  (512 B)  the EXPLICIT sign byte -- FOUR per
//                                           sub-block (ONE per group)
//   ---------------------------------------- stride 2720
//
// plus the plain block_q8_K activation (292, quants @4) and the INTERLEAVED block_q8_Kx4
// GEMM activation (1168, quants @16 as pos*4+c). The 2720 stride is iq3_xxs's 1696 plus
// the extra 1024 B the index strip gains going u8 -> u16; every other strip is identical
// in shape and offset.
//
// NOTE the bsums offsets are ZERO, and that is a FACT for the same reason iq3_xxs's are:
// the fold is the single-accumulator SignScaleStore shape (`bsum += sumi*ls`,
// `sumf += d*bsum`, `*s = sumf`) with no delta term, so there is no second accumulator to
// feed and nothing reads a bsums plane. The loop-body verifier rejects
// activation_bsums_byte_offset outside the two folds that READ it, so stamping one here
// would be rejected -- correctly.
constexpr Iq2GridDecodeFacts kIq3SDecodeFacts = {
    /*decodeModel=*/"iq3_s",
    /*gemmScaleModel=*/kGridIq3SGemmScaleModel,
    /*foldModel=*/"grid_sign_dual_entry_single_scale_unit",
    /*weightBlockStride=*/2720,
    /*weightGridIdxByteOffset=*/160,
    /*weightLsByteOffset=*/32,
    /*weightSignByteOffset=*/2208, // a REAL explicit sign strip (not a delta slot).
    /*gevmActivationBlockStride=*/292,
    /*gevmActivationQuantByteOffset=*/4,
    /*gevmActivationBsumsByteOffset=*/0, // iq3_s reads no bsums (no delta term).
    /*gemmActivationBlockStride=*/1168,
    /*gemmActivationQuantByteOffset=*/16,
    /*gemmActivationBsumsByteOffset=*/0,
    /*nSubblocks=*/8,
};

// Derives the resource-aware e16m1 strip width (half_lanes) from the guaranteed
// minimum VLEN via the NAMED CLOSED FORM getRVVRepackStripHalfLanes(minVLEN,
// interleave) in the gearbox header ([GAP-P1] family/width selector): half_lanes =
// min(vlen/16, 16), so 128 -> 8, 256 -> 16. Returns 0 when the evidence guarantees
// no >= 128 minimum (an empty -march, or a constrained tier) -- the bridge then has
// NO capability-derived strip width and CANNOT form a well-formed x16 repack op, so
// it leaves the request as the deferred block-dot stub (the honest no-capability
// behavior, e.g. the q4_0-prefill-at-VLEN0 cell). `vlenBits` is PULLED off the in-IR
// capability provider op (resolveRVVMinimumVLEN) at the call site, never a -march
// re-parse -- so a capability-file VLEN drives the width, provably (byte-exact: the
// closed form returns the same value the old inline helper did).
std::int64_t deriveRepackHalfLanes(std::int64_t vlenBits) {
  return pluginrvv::getRVVRepackStripHalfLanes(vlenBits, kWeightInterleave)
      .halfLanes;
}

//===----------------------------------------------------------------------===//
// full-LMUL[B] repack accumulator-LMUL MEASURED-GATE (keying-audit report §一C /
// §二 P1: the last parametric class-C under-keyed lever).
//
// The repack GEVM/GEMM integer core realizes EITHER the mf2 FRACTIONAL chain
// (i8mf2 -> i16m1 -> i32m2: two 8-lane strips @VLEN128, half_lanes 8, no
// integer_core_lmul anchor) OR the m1 WHOLE-LMUL chain (i8m1 -> i16m2 -> i32m4:
// one 16-lane strip, half_lanes 16, integer_core_lmul "m1"). The emitter
// (RVVToEmitCBlockQuantLinear.cpp:2854-2857) is version-BLIND and fully
// attribute-driven -- it reads integer_core_lmul (default mf2) + half_lanes off
// the loop-body op and derives the l8/l16/l32 widening chain from THAT alone --
// so the m1 arm ALREADY auto-realizes on ANY ISA once those attrs are stamped.
// The ONLY reason m1 never shipped on RVV1.0 is that the front-door STAMP was
// gated on `isM1 = isRVV0p7` -- a CORRECTNESS fork (RVV0.7.1 xtheadvector has no
// fractional LMUL, so the whole-LMUL chain is mandatory), NEVER a perf predicate.
// q4 wants m1 (whole 16-lane strip, one pass) while q8 wants mf2 (its lean core
// spills at m1) -- a PER-FORMAT x PER-BOARD crossover that this ISA-generation
// fork cannot express. This selector REPLACES that correctness-only fork:
//
//   * RVV0.7.1: m1 is a CORRECTNESS constraint (no fractional LMUL) -> kept.
//   * RVV1.0: BOTH chains are constructible once the capability affords a strip
//     width (halfLanes != 0 => minVLEN >= 128). The m1 chain's register
//     footprint (i16m2 product + i32m4 accumulator, computed via the SAME gate4
//     footprint helper getRVVLMULRegisterFootprint) trivially fits the 32-vreg
//     budget, so both are budget-legal.
//
// [GAP-P1] IRON RULE -- widen-to-m1 was FALSIFIED TWICE (micro win washes at e2e
// / regfile spill). We do NOT blind-select the wider m1 (that is exactly what the
// gate4 widest-legal selector does for a DIFFERENT kernel path; reusing it here
// verbatim would re-commit [GAP-P1]). The DEPLOYED default stays mf2; ONLY a
// per-format BOARD MEASUREMENT recording m1-faster for this (format, board) flips
// it (reason "measured"). The measured table is EMPTY today (STAGE THREE
// populates the per-format x board crossover, board-MEASURED and NEVER projected
// -- [GAP-P1]), so every RVV1.0 format resolves to mf2 => BYTE-EXACT with the
// pre-selector emit; every existing fixture stays green unchanged.
// STAGE THREE fill (user-adjudicated 2026-07-19: [GAP-P1] LOOSENED to permit a
// board-measured spill-free format to take the wide m1 chain -- the IRON RULE's
// OWN escape hatch: "ONLY a per-format BOARD MEASUREMENT recording m1-faster
// flips it (reason measured)"). This per-format board-MEASURED m1-vs-mf2
// crossover registry is the ONLY thing that flips the mf2 default. It is
// registration-as-DATA -- the SAME status/metric/hook mechanism as
// kRepackVlen256DecodeMeasurements / the IME kIMEWideFormatMeasurements registry,
// NOT a per-format C++ switch and NEVER a VLEN/footprint PROJECTION. A row is a
// BOARD FACT keyed on the committed decode-family scale_model WHAT; a scale_model
// with NO row is UNMEASURED => nullopt => the mf2 default holds (byte-exact).
// Adding a board-measured format = adding a data row (换键不改条目); the selector
// consumes ONLY the resulting bool and stays blind to the format label.
//
// The a-priori PROJECTION above guessed "q8 wants mf2 (its lean core spills at
// m1), q4 wants m1". The BOARD REFUTED it -- exactly why [GAP-P1] forbids
// projection. (1) The register-pressure inequality (rvvRegisterPressureLegal)
// proves the q8 m1 whole-LMUL chain is SPILL-FREE at unroll=1 (i16m2 + i32m4 =
// 6 vregs <= 32). (2) The rvv board (VLEN128, clang 17.0.6) measured the DEPLOYED
// (CORE==PROD) q8_0 full-int8 m1 chain FASTER than the mf2 default in BOTH
// regimes, byte-exact (3-arm mism=0), 2-seed cold: decode-GEVM 1.6-2.4x
// (footprint-robust THROUGH 35.6 MB DRAM-bound -- the "micro washes at e2e"
// concern does NOT wash out within the deployed GEVM) AND prefill-GEMM ~1.40x
// (the wide 16-lane strip beats the mf2 4-column columnsPerPass amortization).
//
// r51g family board-sweep (experiments/active/r51g-b1-repack-family-sweep) extends
// the sweep to the DEPLOYED flat nibble family (resolving W2's scalar-nibble-unpack
// confound: the deployed emit VECTORIZES the unpack). All 3-arm byte-exact
// (mism=0), spill-free-verified by objdump, 2-seed cold reps=15/7:
//   * q4_0 (kNibbleQ40ScaleModel): m1 FASTER in BOTH regimes -> FLIP. decode-GEVM
//     0.39-0.44 (~2.3-2.5x), prefill-GEMM 0.78-0.82 (~1.24x), both spill=0.
//   * q4_1 (kNibbleQ41ScaleModel): m1 FASTER decode, PARITY prefill -> FLIP (no
//     regression). decode-GEVM 0.40-0.45 (~2.3x), prefill-GEMM 0.99-1.02 (parity),
//     both spill=0.
//   * q5_0/q5_1: m1 wins decode (GEVM 0.55-0.65) but the prefill GEMM m1 emit
//     SPILLS (objdump 1-2 whole-register reloads) and is 2.3-2.4x SLOWER -- the
//     [GAP-P1] regfile-spill concern is VINDICATED for the qh+min prefill core at
//     m1. Because one measured row flips BOTH regimes, q5_0/q5_1 stay mf2 (a fill
//     would deploy a 2.3x prefill regression). NOT flipped -- honest null.
// Only board-MEASURED spill-free double-safe formats are asserted here.
struct RepackMeasuredM1FasterMeasurement {
  llvm::StringRef scaleModel;  ///< committed decode-family WHAT (registry key)
  bool m1Faster;               ///< board-measured: m1 chain faster than mf2 default
  llvm::StringRef metric;      ///< measured cold m1/mf2 ratio (<1 => m1 faster)
  llvm::StringRef metricsHook; ///< board evidence provenance
};
constexpr RepackMeasuredM1FasterMeasurement
    kRepackMeasuredM1FasterMeasurements[] = {
        {kNibbleQ80ScaleModel, /*m1Faster=*/true,
         "decode-GEVM 0.42-0.62 / prefill-GEMM 0.69-0.72 cold m1/mf2 (1.4-2.4x faster)",
         "experiments/active/r51f-gapp1-q8-deployed-board/FINDING.md + "
         "experiments/active/w2-widen-to-m1-board/FINDING.md "
         "(q8_0 full-i8 DEPLOYED repack GEVM+GEMM m1 whole-LMUL chain faster than "
         "the mf2 default @rvv VLEN128 clang-17.0.6, spill-free by "
         "rvvRegisterPressureLegal, byte-exact 3-arm 2-seed cold, GEVM "
         "footprint-robust THROUGH 35.6 MB DRAM-bound; CORE==PROD deployed emit)"},
        {kNibbleQ40ScaleModel, /*m1Faster=*/true,
         "decode-GEVM 0.39-0.44 / prefill-GEMM 0.78-0.82 cold m1/mf2 (2.3-2.5x / 1.24x faster)",
         "experiments/active/r51g-b1-repack-family-sweep/FINDING.md "
         "(q4_0 signed-nibble DEPLOYED repack GEVM+GEMM m1 whole-LMUL chain faster "
         "than the mf2 default @rvv VLEN128 clang-17.0.6, spill-free by objdump, "
         "byte-exact 3-arm 2-seed cold; resolves W2 scalar-unpack confound -- the "
         "deployed emit vectorizes the nibble unpack, which mf2 does TWICE per "
         "16-col group and m1 collapses to ONE 16-lane pass)"},
        {kNibbleQ41ScaleModel, /*m1Faster=*/true,
         "decode-GEVM 0.40-0.45 / prefill-GEMM 0.99-1.02 cold m1/mf2 (2.3x faster / parity)",
         "experiments/active/r51g-b1-repack-family-sweep/FINDING.md "
         "(q4_1 unsigned-nibble+min DEPLOYED repack GEVM+GEMM: m1 2.3x faster in "
         "decode, PARITY in prefill (spill-free, no regression) @rvv VLEN128 "
         "clang-17.0.6, byte-exact 3-arm 2-seed cold; the decode-dominant flat "
         "family win, prefill neutral)"},
};

// Pure DATA lookup: the board-measured m1-vs-mf2 disposition for the committed
// decode-family scale_model, or nullopt when the format has NO measured row (=>
// the mf2 default). Mirrors lookupRepackVlen256Decode / lookupWideFormatMeasured.
inline std::optional<bool>
lookupRepackMeasuredM1Faster(llvm::StringRef scaleModel) {
  for (const RepackMeasuredM1FasterMeasurement &m :
       kRepackMeasuredM1FasterMeasurements)
    if (m.scaleModel == scaleModel)
      return m.m1Faster;
  return std::nullopt;
}

/// Project the front-door's typed owners into the plugin-local decision contract.
/// This is the sole production constructor: scale_model is consumed only as the
/// current measurement key, while geometry and target capability remain separate.
inline pluginrvv::RepackAccumulatorLMULDecision
buildRepackAccumulatorLMULDecision(weftrvv::GgmlQuantContractionOp op,
                                   std::optional<bool> hasFractionalLMUL,
                                   std::int64_t capabilityHalfLanes) {
  pluginrvv::RepackAccumulatorLMULFormulaResult formula =
      pluginrvv::constructRepackAccumulatorLMULFormula(
          /*g=*/{/*weightInterleave=*/kWeightInterleave},
          /*c=*/{/*hasFractionalLMUL=*/hasFractionalLMUL,
                 /*halfLanes=*/capabilityHalfLanes,
                 /*vectorRegisterBudget=*/
                     pluginrvv::resolveRVVVectorRegisterBudget(
                         op->getParentOfType<mlir::ModuleOp>())},
          pluginrvv::RepackAccumulatorLMULNoStaticContext{});

  // Winner memory is consulted only after the formula has constructed the
  // complete legal set and analytic prior.  It cannot alter either one.
  pluginrvv::RepackAccumulatorLMULSelectionInput selectionInput;
  if (std::optional<bool> measuredM1Faster =
          lookupRepackMeasuredM1Faster(op.getScaleModel())) {
    selectionInput.measurement =
        pluginrvv::RepackAccumulatorLMULQualifiedMeasurement{
        /*key=*/{op.getScaleModel()},
        /*winner=*/*measuredM1Faster
            ? pluginrvv::RepackAccumulatorLMULCandidate::M1
            : pluginrvv::RepackAccumulatorLMULCandidate::MF2};
  }
  return pluginrvv::selectRepackAccumulatorLMUL(formula, selectionInput);
}

template <typename LoopOp>
void stampRepackAccumulatorLMULDecision(
    mlir::OpBuilder &builder, LoopOp loop,
    const pluginrvv::RepackAccumulatorLMULDecision &decision) {
  loop->setAttr(kAccumulatorLmulReasonAttr,
                builder.getStringAttr(
                    pluginrvv::stringifyRepackAccumulatorLMULReason(
                        decision.reason)));
  if (decision.measurementKey)
    loop->setAttr(kAccumulatorLmulMeasurementKeyAttr,
                  builder.getStringAttr(
                      decision.measurementKey->scaleModel));
}

class RVVLowerQuantContractionPass final
    : public impl::RVVLowerQuantContractionBase<RVVLowerQuantContractionPass> {
public:
  using impl::RVVLowerQuantContractionBase<
      RVVLowerQuantContractionPass>::RVVLowerQuantContractionBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::WalkResult result =
        module.walk([&](weftrvv::GgmlQuantContractionOp op) -> mlir::WalkResult {
          if (mlir::failed(lowerOne(op)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  // Read the abstract op's STRUCTURED OPPONENT FACTS (the IR declaration layer)
  // into the selector's pure fact struct. This is the C1 relocation: routing
  // reads opponent_vlen_native_floor / block_dot_compute_heavy /
  // block_dot_memory_bound from the IR, NEVER the quant format LABEL, so a request
  // that deleted its `quant` label but kept the facts selects the IDENTICAL
  // algorithm. Absent facts default to the conservative "no repack advantage" (no
  // VLEN-native opponent floor / neither compute-heavy nor memory-bandwidth-bound),
  // which routes to the safe block-dot path.
  static pluginrvv::ContractionOpponentFacts
  readOpponentFacts(weftrvv::GgmlQuantContractionOp op) {
    pluginrvv::ContractionOpponentFacts facts;
    if (mlir::IntegerAttr floor = op.getOpponentVlenNativeFloorAttr())
      facts.ggmlVlenNativeKernelFloor = floor.getInt();
    facts.blockDotComputeHeavy = op.getBlockDotComputeHeavy().value_or(false);
    facts.blockDotMemoryBound = op.getBlockDotMemoryBound().value_or(false);
    return facts;
  }

  static mlir::FailureOr<pluginrvv::MRegime>
  liftMRegime(weftrvv::GgmlQuantContractionOp op) {
    if (op.getMRegime() == "decode")
      return pluginrvv::MRegime::Decode;
    if (op.getMRegime() == "prefill")
      return pluginrvv::MRegime::Prefill;
    return mlir::FailureOr<pluginrvv::MRegime>(
        op.emitError() << "stage-B contraction-path selection does not "
                          "recognize m_regime \""
                       << op.getMRegime() << "\"");
  }

  /// Construct the complete repack schedule before the typed loop operation is
  /// created.  OperationState receives only final code-affecting fields; candidate
  /// sets, priors, winner lookup and diagnostics remain transient C++ values.
  mlir::LogicalResult addRepackScheduleFormula(
      mlir::OpBuilder &builder, mlir::OperationState &loopState,
      llvm::StringRef kernel, pluginrvv::RVVRepackScheduleRegime regime,
      weftrvv::GgmlQuantContractionOp op) {
    auto getI64 = [&](llvm::StringRef name) -> std::optional<std::int64_t> {
      auto attr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
          loopState.attributes.get(name));
      if (!attr)
        return std::nullopt;
      return attr.getInt();
    };
    auto getString = [&](llvm::StringRef name) -> llvm::StringRef {
      auto attr = llvm::dyn_cast_or_null<mlir::StringAttr>(
          loopState.attributes.get(name));
      return attr ? attr.getValue() : llvm::StringRef();
    };

    const std::optional<std::int64_t> weightStride =
        getI64("weight_block_stride");
    const std::optional<std::int64_t> activationStride =
        getI64("activation_block_stride");
    const std::optional<std::int64_t> qk = getI64("qk");
    const std::optional<std::int64_t> weightInterleave =
        getI64("weight_interleave");
    std::optional<std::int64_t> activationInterleave =
        getI64("activation_interleave");
    if (!activationInterleave &&
        regime == pluginrvv::RVVRepackScheduleRegime::Gemv)
      activationInterleave = 1;
    const std::optional<std::int64_t> halfLanes = getI64("half_lanes");
    if (!weightStride || !activationStride || !qk || !weightInterleave ||
        !activationInterleave || !halfLanes)
      return op.emitOpError(
          "cannot construct repack schedule without complete typed geometry");

    std::string declaredInstanceHash;
    if (auto kernelOp = op->getParentOfType<weft::exec::KernelOp>()) {
      if (llvm::Expected<support::TargetCapabilitySet> capabilities =
              support::TargetCapabilitySet::buildFromKernelChecked(kernelOp))
        declaredInstanceHash =
            support::computeDeclaredInstanceHash(*capabilities);
      else
        llvm::consumeError(capabilities.takeError());
    }

    const std::int64_t minimumVLEN = pluginrvv::resolveRVVMinimumVLEN(
        op->getParentOfType<mlir::ModuleOp>(), march, isaVectorHints);

    pluginrvv::RVVRepackScheduleGeometryFacts g{
        *weightStride, *activationStride, *qk, *weightInterleave,
        *activationInterleave, *halfLanes, getString("integer_core_lmul"),
        getString("fold_model")};
    pluginrvv::RVVRepackScheduleCapabilityFacts c{
        minimumVLEN, kRVVArchVectorRegisterCount};
    pluginrvv::RVVRepackScheduleContext omega{regime};
    std::optional<pluginrvv::RVVRepackScheduleFormulaResult> formula =
        pluginrvv::constructRVVRepackScheduleFormula(g, c, omega);
    if (!formula)
      return op.emitOpError(
          "cannot construct a legal repack schedule from typed g/c/omega");

    pluginrvv::RVVRepackScheduleSelectionInput selection;
    selection.qualifiedLoopOrderWinner =
        pluginrvv::lookupQualifiedRepackLoopOrderWinner(declaredInstanceHash,
                                                        kernel);
    selection.qualifiedMainTermWinner =
        pluginrvv::lookupQualifiedRepackMainTermWinner(
            formula->unrolledMainTermVwmacc, g.integerCoreLMUL);
    pluginrvv::RVVRepackFinalSchedule finalSchedule =
        pluginrvv::selectRVVRepackSchedule(*formula, selection);
    if (regime == pluginrvv::RVVRepackScheduleRegime::GemmPrefill &&
        !finalSchedule.loopOrder)
      return op.emitOpError("repack GEMM formula produced no final loop order");
    if (pluginrvv::isRepackKQuantMainTermFold(g.foldModel) &&
        !finalSchedule.mainTermForm)
      return op.emitOpError(
          "K-quant repack formula produced no final main-term form");

    if (finalSchedule.loopOrder)
      loopState.addAttribute(
          "loop_order",
          builder.getStringAttr(pluginrvv::stringifyRVVRepackLoopOrder(
              *finalSchedule.loopOrder)));
    if (finalSchedule.mainTermForm)
      loopState.addAttribute(
          "main_term_form",
          builder.getStringAttr(pluginrvv::stringifyRVVRepackMainTermForm(
              *finalSchedule.mainTermForm)));
    return mlir::success();
  }

  // The IN-COMPILER selection: resolve the target VLEN off the in-IR RVV capability
  // provider op (the pulled pipe -- resolveRVVMinimumVLEN; -march is the un-probed
  // fallback, NOT the op's advisory min_vlen attr), lift the committed WHAT axes,
  // and ask the pure fact-driven selector which algorithm to commit to. Both
  // branches emit the byte-identical block-dot body (Option (i)), differentiated
  // only by the inert audit attrs -- so the emitted C is unchanged on every cell
  // and the repack EFFECT is honestly deferred to stage C.
  mlir::LogicalResult lowerOne(weftrvv::GgmlQuantContractionOp op) {
    // Read the per-format OPPONENT FACTS from the op's structured attrs -- routing
    // is fact-driven, NOT keyed on the (now optional) quant format label.
    pluginrvv::ContractionOpponentFacts facts = readOpponentFacts(op);
    mlir::FailureOr<pluginrvv::MRegime> mRegime = liftMRegime(op);
    if (mlir::failed(mRegime))
      return mlir::failure();

    // The DERIVED capability fact: the guaranteed minimum VLEN of the configured
    // target, PULLED off the in-IR RVV capability provider op (resolveRVVMinimumVLEN;
    // -march is the un-probed fallback -- default -march "" => 0 => no capability
    // => block-dot, the honest no-capability behavior).
    std::int64_t minVLEN = pluginrvv::resolveRVVMinimumVLEN(
        op->getParentOfType<mlir::ModuleOp>(), march, isaVectorHints);

    pluginrvv::ContractionAlgorithmFormulaResult formula =
        pluginrvv::constructContractionAlgorithmFormula(
            facts, {/*minimumVLEN=*/minVLEN},
            {/*mRegime=*/*mRegime});

    // The board-qualified residual correction is read only after g/c/omega have
    // produced the complete legal set and analytic prior.  The selector can
    // choose one of those candidates; it cannot create a contraction path.
    pluginrvv::ContractionSelectionInput selectionInput;
    if (std::optional<Vlen256DecodeRepackDisposition> disposition =
            lookupRepackVlen256Decode(op.getScaleModel())) {
      selectionInput.measurement = pluginrvv::ContractionQualifiedMeasurement{
          /*key=*/{op.getScaleModel()},
          /*winner=*/
              *disposition == Vlen256DecodeRepackDisposition::Beneficial
                  ? pluginrvv::ContractionAlgorithm::Repack
                  : pluginrvv::ContractionAlgorithm::BlockDot};
    }
    pluginrvv::ContractionSelection selection =
        pluginrvv::selectContractionAlgorithm(formula, selectionInput);

    // STAGE C1 (the in-IR BRIDGE): when the selection is Repack AND the target
    // capability supplies a valid e16m1 strip width (minVLEN >= 128 => half_lanes
    // in {8, 16}), REALIZE the request as the real weft_rvv.repack_gemv_q4_0_q8_0
    // op carrying the block_q4_0x16 facts + the DECLARED weight_layout_contract =
    // "x16" (the OUTPUT CONTRACT). The block-dot-SELECTED branch (q4_0@K1, q8_0,
    // q4_K) is UNCHANGED -- it still emits the byte-identical block-dot body with
    // weight_layout_contract IMPLICITLY plain (the deferred-stub provenance). A
    // Repack-SELECTED request with NO capability strip width (the prefill cell at
    // VLEN0: half_lanes 0) CANNOT form a well-formed x16 repack op, so it stays
    // the deferred block-dot stub -- the bridge realizes x16 ONLY where the
    // capability fact actually affords the strip width.
    bool isRepack =
        selection.algorithm == pluginrvv::ContractionAlgorithm::Repack;
    std::int64_t halfLanes = deriveRepackHalfLanes(minVLEN);
    // The RVV ISA generation, PULLED off the in-IR RVV capability provider op
    // (resolveRVVVersion reads the materialized rvv_version fact; -march is only the
    // un-probed fallback) -- so a capability file rvv_version=1.0 WINS over a
    // conflicting -march xtheadvector (0.7): hasFractionalLMUL follows the
    // capability object, not a format label or an emitter-side march reparse.
    pluginrvv::RVVVersion rvvVersion = pluginrvv::resolveRVVVersion(
        op->getParentOfType<mlir::ModuleOp>(), march, isaVectorHints);
    std::optional<bool> hasFractionalLMUL;
    if (rvvVersion == pluginrvv::RVVVersion::RVV1p0)
      hasFractionalLMUL = true;
    else if (rvvVersion == pluginrvv::RVVVersion::RVV0p7)
      hasFractionalLMUL = false;
    if (isRepack && halfLanes != 0) {
      // A2 authority cutover: construct the accumulator-LMUL decision exactly
      // once for this declared repack slice.  Every mutually exclusive builder
      // below consumes this selected typed result; none may look up measurement,
      // re-read capability, or select/stamp independently.
      pluginrvv::RepackAccumulatorLMULDecision accLmulDecision =
          buildRepackAccumulatorLMULDecision(op, hasFractionalLMUL,
                                              halfLanes);
      if (!accLmulDecision.isLegal())
        return op.emitError()
               << "repack accumulator-LMUL decision failed closed: "
               << pluginrvv::stringifyRepackAccumulatorLMULReason(
                      accLmulDecision.reason);
      // The m_regime committed WHAT axis chooses the repacked GRANULARITY: the
      // PREFILL (M-amortized) regime realizes the repack as the typed
      // weft_rvv.typed_repack_gemm_loop_body REGION (the block-as-lane GEMM that
      // internalizes BOTH the M-row and N-column loops over the interleaved
      // block_q8_0x4 activation), the DECODE regime as the typed
      // weft_rvv.typed_repack_gemv_loop_body REGION (the single-activation-row
      // GEVM that internalizes only the N-column loop over a plain q8_0 stream).
      // Both share the SAME capability gate (isRepack + a valid e16m1 strip
      // width); only the granularity differs. The decode FAMILY (q4_0 nibble vs
      // ternary tq2_0 2-bit vs ternary tq1_0 base-3 trit) is keyed off the
      // committed scale_model WHAT: the ternary families CONSTRUCT the ternary
      // typed_repack region + the repack_gem{v,m}_ternary_core brick via the SHARED
      // lowerToRepackGem{v,m}Ternary, parameterized by the per-family
      // TernaryDecodeFacts (base facts + core-brick decode_model + optional qh
      // SECOND plane); the q4_0 family builds the nibble core + dual-fp16 fold.
      const TernaryDecodeFacts *ternary =
          op.getScaleModel() == kTernaryTQ20ScaleModel ? &kTernaryTQ20DecodeFacts
          : op.getScaleModel() == kTernaryTQ10ScaleModel
              ? &kTernaryTQ10DecodeFacts
              : nullptr;
      // The K-quant family (q4_K dual d/dmin + bsums-min, q6_K single-scale no-min,
      // OR q2_K 2-bit-weight dual d/dmin + bsums-min -- the q4_K min-fold sibling)
      // builds the K-quant typed_repack region + the repack_gem{v,m}_kquant_core
      // brick via lowerToRepackGem{v,m}KQuant, parameterized by the per-family
      // KQuantDecodeFacts (facts.hasMin / facts.foldModel select the fold arity).
      const KQuantDecodeFacts *kquant =
          op.getScaleModel() == kKQuantQ4KScaleModel   ? &kQ4KDecodeFacts
          : op.getScaleModel() == kKQuantQ6KScaleModel ? &kQ6KDecodeFacts
          : op.getScaleModel() == kKQuantQ2KScaleModel ? &kQ2KDecodeFacts
          : op.getScaleModel() == kKQuantQ3KScaleModel ? &kQ3KDecodeFacts
          : op.getScaleModel() == kKQuantQ5KScaleModel ? &kQ5KDecodeFacts
                                                       : nullptr;
      // The codebook family (iq4_nl flat single-scale non-linear codebook, OR iq4_xs the
      // SUPER-BLOCK sibling with a K-quant-style 6-bit SIGNED per-sub-block scale) builds
      // the codebook typed_repack region + the SHARED repack_gem{v,m}_codebook_core brick
      // via lowerToRepackGem{v,m}Codebook, parameterized by the per-family CodebookDecodeFacts
      // (facts.foldModel selects flat vs super-block-signed6; both RECONSTRUCT the SAME
      // 16-entry kvalues table + iq4_xs its signed-6 scale offsets).
      const CodebookDecodeFacts *codebook =
          op.getScaleModel() == kCodebookIq4NlScaleModel   ? &kIq4NlDecodeFacts
          : op.getScaleModel() == kCodebookIq4XsScaleModel ? &kIq4XsDecodeFacts
          : op.getScaleModel() == kCodebookMxfp4ScaleModel ? &kMxfp4DecodeFacts
                                                           : nullptr;
      // The GRID family (iq2_xxs GRID CODEBOOK + SIGN-PLANE single ls-scale, the FIRST
      // grid sibling) builds the grid typed_repack region + the NEW
      // repack_gem{v,m}_grid_core brick via lowerToRepackGem{v,m}Grid, parameterized by
      // the per-family Iq2GridDecodeFacts. Unlike the codebook family there is NO codebook
      // array: the FIXED grid + DERIVED signs64 planes are emit-period static const tables,
      // and the grid/ls/sign byte offsets + n_subblocks ride on the grid core brick.
      const Iq2GridDecodeFacts *grid =
          op.getScaleModel() == kGridIq2XxsScaleModel ? &kIq2XxsDecodeFacts
          : op.getScaleModel() == kGridIq2XsScaleModel ? &kIq2XsDecodeFacts
          : op.getScaleModel() == kGridIq2SScaleModel  ? &kIq2SDecodeFacts
          : op.getScaleModel() == kGridIq1SScaleModel  ? &kIq1SDecodeFacts
          : op.getScaleModel() == kGridIq1MScaleModel  ? &kIq1MDecodeFacts
          : op.getScaleModel() == kGridIq3XxsScaleModel ? &kIq3XxsDecodeFacts
          : op.getScaleModel() == kGridIq3SScaleModel  ? &kIq3SDecodeFacts
                                                       : nullptr;
      // The q4_1 family (unsigned nibble + single MIN fold) builds the SAME typed
      // q4_0 repack region via lowerToRepackGem{v,m}Q41 (the SHARED q4_0 core + fold
      // bricks, the core stamping weight_nibble_unsigned and the fold stamping the
      // MIN-fold offset pair). Keyed off the committed q4_1 scale_model WHAT.
      bool isQ41 = op.getScaleModel() == kNibbleQ41ScaleModel;
      // The q5_0 family (unsigned nibble + qh 5th-bit + -16 offset, NO min) builds
      // the SAME typed q4_0 repack region via lowerToRepackGem{v,m}Q50 (the SHARED
      // q4_0 core + fold bricks, the core stamping weight_nibble_unsigned +
      // weight_qh_byte_offset + weight_offset_bias). Keyed off the committed q5_0
      // scale_model WHAT.
      bool isQ50 = op.getScaleModel() == kNibbleQ50ScaleModel;
      // The q5_1 family (unsigned 5-bit nibble+qh + single MIN fold) builds the SAME
      // typed q4_0 repack region via lowerToRepackGem{v,m}Q51 (the SHARED q4_0 core +
      // fold bricks, the core stamping weight_nibble_unsigned + weight_qh_byte_offset
      // with NO weight_offset_bias -- the UNSIGNED 5th-bit decode -- and the fold
      // stamping the MIN-fold offset pair). = q5_0's qh gather (unsigned/no -16) +
      // q4_1's min fold. Keyed off the committed q5_1 scale_model WHAT.
      bool isQ51 = op.getScaleModel() == kNibbleQ51ScaleModel;
      // The q8_0 family (FULL int8, NO nibble/qh/offset/min) builds the SAME typed
      // q4_0 repack region via lowerToRepackGem{v,m}Q80 (the SHARED dual-fp16 d-only
      // fold brick + the q8_0 full-i8 CORE brick variant, the core stamping
      // weight_full_i8). Keyed off the committed q8_0 scale_model WHAT.
      bool isQ80 = op.getScaleModel() == kNibbleQ80ScaleModel;
      if (*mRegime == pluginrvv::MRegime::Prefill)
        return grid
                   ? lowerToRepackGemmGrid(op, selection, accLmulDecision,
                                            *grid)
               : codebook
                   ? lowerToRepackGemmCodebook(op, selection, accLmulDecision,
                                                *codebook)
               : kquant
                   ? lowerToRepackGemmKQuant(op, selection, accLmulDecision,
                                              *kquant)
               : ternary
                   ? lowerToRepackGemmTernary(op, selection, accLmulDecision,
                                              *ternary)
               : isQ41 ? lowerToRepackGemmQ41(op, selection, accLmulDecision)
               : isQ50 ? lowerToRepackGemmQ50(op, selection, accLmulDecision)
               : isQ51 ? lowerToRepackGemmQ51(op, selection, accLmulDecision)
               : isQ80 ? lowerToRepackGemmQ80(op, selection, accLmulDecision)
                       : lowerToRepackGemm(op, selection, accLmulDecision);
      return grid
                 ? lowerToRepackGemvGrid(op, selection, accLmulDecision, *grid)
             : codebook
                 ? lowerToRepackGemvCodebook(op, selection, accLmulDecision,
                                              *codebook)
             : kquant
                 ? lowerToRepackGemvKQuant(op, selection, accLmulDecision,
                                            *kquant)
             : ternary
                 ? lowerToRepackGemvTernary(op, selection, accLmulDecision,
                                            *ternary)
             : isQ41 ? lowerToRepackGemvQ41(op, selection, accLmulDecision)
             : isQ50 ? lowerToRepackGemvQ50(op, selection, accLmulDecision)
             : isQ51 ? lowerToRepackGemvQ51(op, selection, accLmulDecision)
             : isQ80 ? lowerToRepackGemvQ80(op, selection, accLmulDecision)
                     : lowerToRepackGemv(op, selection, accLmulDecision);
    }

    // Fail-closed (I7): a ternary (tq2_0 2-bit / tq1_0 base-3) OR K-quant (q4_K
    // dual-scale+min / q6_K 6-bit no-min) request has NO block-dot decline path --
    // the block-dot identity lowering reconstructs a q4_0 nibble body, which would
    // MISCOMPILE ternary trit / q4_K / q6_K super-block weights. Such a request that
    // does not afford a repack strip width (minVLEN < 128) is rejected, never
    // silently mis-lowered into a q4_0 block-dot.
    if (op.getScaleModel() == kTernaryTQ20ScaleModel ||
        op.getScaleModel() == kTernaryTQ10ScaleModel ||
        op.getScaleModel() == kKQuantQ4KScaleModel ||
        op.getScaleModel() == kKQuantQ6KScaleModel ||
        op.getScaleModel() == kKQuantQ2KScaleModel ||
        op.getScaleModel() == kKQuantQ3KScaleModel ||
        op.getScaleModel() == kKQuantQ5KScaleModel ||
        op.getScaleModel() == kCodebookIq4NlScaleModel ||
        op.getScaleModel() == kCodebookIq4XsScaleModel ||
        op.getScaleModel() == kCodebookMxfp4ScaleModel ||
        op.getScaleModel() == kGridIq2XxsScaleModel ||
        op.getScaleModel() == kGridIq2XsScaleModel ||
        op.getScaleModel() == kGridIq2SScaleModel ||
        op.getScaleModel() == kGridIq1SScaleModel ||
        op.getScaleModel() == kGridIq1MScaleModel ||
        op.getScaleModel() == kGridIq3XxsScaleModel ||
        op.getScaleModel() == kGridIq3SScaleModel ||
        op.getScaleModel() == kNibbleQ50ScaleModel ||
        op.getScaleModel() == kNibbleQ51ScaleModel ||
        op.getScaleModel() == kNibbleQ80ScaleModel)
      return op.emitError()
             << "ternary / K-quant / codebook / grid / q5_0 / q5_1 / q8_0 "
                "quant_contraction requires a repack-affording capability (a valid "
                "e16m1 strip width, minVLEN >= 128); there is no ternary / K-quant / "
                "codebook / grid / q5_0 / q5_1 / q8_0 block-dot decline path (the "
                "block-dot identity lowering is q4_0-nibble-only, which would "
                "MISCOMPILE the iq2_xxs grid/sign weights as nibbles)";

    return lowerToBlockDot(op, selection);
  }

  // STAGE C1 bridge (M-FLAT REPACK, region form): realize a repack-SELECTED,
  // capability-afforded request as the typed weft_rvv.typed_repack_gemv_loop_body
  // REGION -- the same construction the flat/super-block front doors do, now for
  // the q4_0 16x1-repacked GEVM. The region is the SOLE representation of the
  // repacked GEVM (the monolithic weft_rvv.repack_gemv_q4_0_q8_0 op is retired):
  // the compiler CONSTRUCTS the inner contraction-block loop out of the two
  // decomposed typed bricks -- the per-block lane-wise integer CORE
  // (weft_rvv.repack_lane_wise_q4_x_i8_dot, producing numHalves per-strip sumi) +
  // the per-strip dual-fp16 scale FOLD (weft_rvv.repack_dual_fp16_scale_fold, one
  // per strip) -- around a per-strip LANE-WISE f32 VECTOR loop-carried
  // accumulator, exactly like the hand-authored region tests, but reachable now
  // from a REAL quant_contraction request (not test-authored).
  //
  // It reconstructs the block_q4_0x16 x16 facts (stride 288, interleave 16,
  // weight quant offset 32, activation stride 34 / offset 2) the verifier pins,
  // derives the resource-aware half_lanes from the capability VLEN, and stamps
  // the DECLARED OUTPUT CONTRACT weft_rvv.weight_layout_contract = "x16". The op
  // carries the SAME SSA weight pointer the abstract op carried (the IR cannot
  // tell a plain base from an x16 base; both are const uint8_t *) -- the contract
  // is the bridge's ASSERTION that some later layer (C3-C4) hands it x16 bytes.
  // On RVV0.7.1 the whole-LMUL core anchor (integer_core_lmul = "m1") with its
  // mandatory ONE 16-lane strip (half_lanes = 16, numHalves 1, f32m4 accumulator)
  // is pinned; on RVV1.0 integer_core_lmul is stamped EXPLICITLY as "mf2" (the
  // fractional chain: half_lanes 8 -> two 8-lane strips at VLEN128, f32m2
  // accumulators). ISSUE-033: the front door no longer uses attr-ABSENCE to mean
  // "mf2" -- the width decision is spoken outright so the attr can become required
  // downstream (absence is no longer a silent signal).
  //
  // SAFETY (NOT a latent miscompile): the emitted kernel reads x16 weights but
  // the abstract op carries PLAIN weights, so this emit is correct ONLY when the
  // contract is honored. The abstract GgmlQuantContractionOp has NO real producer
  // (it is authored ONLY in lit fixtures; there is no rewriter.create of it in
  // any real pass), so this region is reachable ONLY via lit, NEVER in the real
  // llama.cpp pipeline. The bridge ASSERTS the layout; the system (C3-C4) must
  // make it true. This is NOT yet e2e-correct on plain weights.
  mlir::LogicalResult
  lowerToRepackGemv(weftrvv::GgmlQuantContractionOp op,
                    const pluginrvv::ContractionSelection &selection,
                    const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    // On RVV0.7.1 the repack core is the WHOLE-LMUL chain (i8m1 -> i16m2 ->
    // i32m4 -> f32m4): no fractional LMUL, so the 16-block-as-lane group is ONE
    // 16-lane strip (half_lanes 16, integer_core_lmul "m1", f32m4 accumulator).
    // RVV1.0 stamps integer_core_lmul EXPLICITLY as "mf2" (ISSUE-033: no silent
    // absence), the fractional default, with the
    // capability-derived strip width (8 @VLEN128 -> two strips, 16 @VLEN256 ->
    // one). numHalves == weight_interleave / half_lanes is the disjoint-strip
    // count, and the region carries ONE per-strip vector accumulator per strip.
    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    // The per-strip accumulator + per-strip integer sumi share the ONE LMUL rung:
    // f32m4/i32m4 for the m1 whole-LMUL chain, f32m2/i32m2 for the mf2 fractional
    // chain (the emitter derives l32 the same way).
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);

    std::int64_t weightQuantByteOffset = 32;
    std::int64_t activationQuantByteOffset =
        static_cast<std::int64_t>(op.getQuantByteOffset());

    // The region-carrying loop op: FIVE ABI operands (weight base, activation
    // base, output, element count n, column count nc), NO vl operand and NO
    // result (the per-strip lane-wise vector store is the sink; the region bricks
    // reference the enclosing setvl VL freely). The repack-GEVM INTERNALIZES the N
    // loop, so it consumes the runtime column count nc the abstract op ALWAYS
    // carries (column_count) -- the exact reason stage A always carries nc.
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemvLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(),
                           op.getColumnCount()});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemv_loop_body"));
    loopState.addAttribute("scale_model",
                           builder.getStringAttr(op.getScaleModel()));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    // The block_q4_0x16 x16 ABI facts the verifier pins (NOT the abstract op's
    // plain stride-18 facts -- this region reads the REPACKED layout the contract
    // declares).
    loopState.addAttribute("weight_block_stride", builder.getI64IntegerAttr(288));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(op.getActivationBlockStride()));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    // The in-compiler decision audit (the same INERT provenance triple the
    // block-dot branch stamps) PLUS the stage-C1 DECLARED OUTPUT CONTRACT. These
    // are discardable, emitter-inert, dialect-namespaced provenance attrs.
    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    // Region entry args: block_index (index) FOLLOWED by numHalves loop-carried
    // per-strip f32 VECTOR accumulators.
    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t h = 0; h < numHalves; ++h)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // Integer CORE brick: ONE weft_rvv.repack_lane_wise_q4_x_i8_dot producing the
    // numHalves per-strip i32 sumi (a variadic result group). block_index-tied
    // (anti-bypass) and named off the loop-body's OWN weight/activation ABI bases.
    mlir::OperationState coreState(
        loc, weftrvv::RepackLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t h = 0; h < numHalves; ++h)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    // numHalves dual-fp16 scale FOLD bricks (one per strip): each consumes the
    // integer brick's strip-h sumi + the strip-h carried accumulator and produces
    // the strip-h folded-out accumulator (the yield's acc_next). Every strip
    // shares the ONE within-block fp16 scale byte offset (weight @0, activation
    // @0 -- the shared fold leaf applies the per-strip stride*half*2).
    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t h = 0; h < numHalves; ++h) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(h), accArgs[h], vl, blockIndex});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    // Terminate the region: the loop yield names the numHalves carried-out
    // per-strip f32 vector accumulators.
    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemvLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    // The region op is result-less; the abstract op's result must be dead (the
    // repacked lane-wise GEVM writes through the output pointer, not an SSA vector
    // -- fail-closed if some producer ever wired the result live).
    if (!op.getResult().use_empty())
      return op.emitError()
             << "repack-GEVM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEVM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (M-FLAT REPACK GEMM finale, region form): realize a
  // repack-SELECTED, capability-afforded PREFILL request as the typed
  // weft_rvv.typed_repack_gemm_loop_body REGION -- the block-as-lane GEMM sibling
  // of lowerToRepackGemv. The compiler CONSTRUCTS the inner contraction-block loop
  // out of the two decomposed typed GEMM bricks -- the ONE-strip N-column integer
  // CORE (weft_rvv.repack_gemm_lane_wise_q4_x_i8_dot, producing columnsPerPass
  // per-column sumi) + the per-column dual-fp16 scale FOLD
  // (weft_rvv.repack_gemm_dual_fp16_scale_fold, one per column) -- around the
  // columnsPerPass per-column LANE-WISE f32 VECTOR loop-carried accumulators, with
  // the block_index + runtime strip_row_offset region entry args the emitter's
  // outer row-group / column-group / runtime-strip / column-pass nest supplies.
  //
  // It reconstructs the block_q4_0x16 weight facts (stride 288, interleave 16,
  // weight quant offset 32) AND the block_q8_0x4 INTERLEAVED activation facts
  // (stride 136, interleave 4, activation quant offset 8) the GEMM verifier pins --
  // NOT the abstract op's PLAIN q8_0 facts (stride 34 / offset 2 the GEVM keeps):
  // the repacked GEMM reads BOTH sides in the repacked layout the DECLARED OUTPUT
  // CONTRACT weft_rvv.weight_layout_contract = "x16" asserts. The GEMM internalizes
  // the M-row loop, so it needs the runtime row count (nr) and the fp32 output row
  // stride (bs) the abstract op does NOT carry (the abstract op delegates M/N to
  // the mul_mat caller and carries only column_count); the bridge MATERIALIZES
  // those two runtime ABI values -- the honest "the compiler materializes the GEMM
  // ABI the internalized nest requires" story, exactly as it materializes the x16
  // weight layout. On RVV0.7.1 the whole-LMUL core anchor (integer_core_lmul = "m1",
  // half_lanes = 16, numHalves 1, f32m4, columnsPerPass 1) is pinned; on RVV1.0
  // integer_core_lmul is unset (the fractional mf2 default: half_lanes 8 -> two
  // 8-lane strips at VLEN128, f32m2, columnsPerPass 4).
  //
  // SAFETY (NOT a latent miscompile): the emitted kernel reads x16 weights /
  // q8_0x4 activations but the abstract op carries PLAIN weights / q8_0, so this
  // emit is correct ONLY when the contract is honored. The abstract
  // GgmlQuantContractionOp has NO real producer (it is authored ONLY in lit
  // fixtures; there is no rewriter.create of it in any real pass), so this region
  // is reachable ONLY via lit, NEVER in the real llama.cpp pipeline. NO e2e/perf
  // claim is made.
  mlir::LogicalResult
  lowerToRepackGemm(weftrvv::GgmlQuantContractionOp op,
                    const pluginrvv::ContractionSelection &selection,
                    const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    // On RVV0.7.1 the repack core is the WHOLE-LMUL chain (no fractional LMUL), so
    // the 16-block-as-lane group is ONE 16-lane strip (half_lanes 16,
    // integer_core_lmul "m1", f32m4 accumulator, columnsPerPass 1). RVV1.0 stamps
    // integer_core_lmul EXPLICITLY as "mf2" (ISSUE-033: no silent absence) with the
    // capability-derived strip width and folds all activation_interleave columns in
    // ONE pass (columnsPerPass 4). numHalves == weight_interleave / half_lanes.
    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);
    std::int64_t columnsPerPass = accLmulDecision.usesM1() ? 1 : kActivationInterleave;

    // The repacked GEMM ABI byte facts the verifier pins (the x16 weight + x4
    // interleaved activation layouts the OUTPUT CONTRACT declares, NOT the abstract
    // op's plain stride-18 / stride-34 facts).
    std::int64_t weightBlockStride = 288;
    std::int64_t weightQuantByteOffset = 32;
    std::int64_t activationBlockStride = 136;
    std::int64_t activationQuantByteOffset = 8;

    // Materialize the two runtime ABI values the internalized M-tiling GEMM nest
    // needs but the abstract op does not carry (row count nr, output row stride
    // bs). They are declared at the variant scope (as siblings of the runtime ABI
    // values the abstract op already reads) so the EmitC param collection renders
    // them as function parameters. This is the compiler MATERIALIZING the GEMM ABI,
    // the same declared-contract move as the x16 weight layout.
    auto variant = op->getParentOfType<weft::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a weft.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, weftrvv::RuntimeABIValueOp::getOperationName());
        st.addAttribute("role", builder.getStringAttr(role));
        st.addAttribute("c_name", builder.getStringAttr(cName));
        st.addAttribute("c_type", builder.getStringAttr("size_t"));
        st.addAttribute("ownership",
                        builder.getStringAttr("target-export-abi-owned"));
        st.addAttribute("purpose", builder.getStringAttr(purpose));
        st.addTypes(builder.getIndexType());
        return builder.create(st)->getResult(0);
      };
      // The role spellings the supported runtime-ABI role set accepts. nr binds
      // source-byte-stride, bs binds output-stride: the object-export ABI arity gate
      // (RVVTargetSupportBundle.cpp) expects exactly this ordered role set for the
      // RepackGemm route family (monolithicRepackGemmABI7).
      rowCount = makeAbi("nr", "source-byte-stride", "nr");
      outputRowStride = makeAbi("bs", "output-stride", "bs");
    }

    // The region-carrying loop op: SEVEN ABI operands (weight base, activation
    // base, output, element count n, row count nr, column count nc, output row
    // stride bs), NO vl operand and NO result (the per-column lane-wise vector
    // store is the sink; the region bricks reference the enclosing setvl VL).
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemmLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(), rowCount,
                           op.getColumnCount(), outputRowStride});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemm_loop_body"));
    loopState.addAttribute("scale_model",
                           builder.getStringAttr(op.getScaleModel()));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute("weight_block_stride",
                           builder.getI64IntegerAttr(weightBlockStride));
    loopState.addAttribute("activation_block_stride",
                           builder.getI64IntegerAttr(activationBlockStride));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("activation_interleave",
                           builder.getI64IntegerAttr(kActivationInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, "q4_0",
            pluginrvv::RVVRepackScheduleRegime::GemmPrefill, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    // The in-compiler decision audit (the same INERT provenance triple the GEVM
    // branch stamps) PLUS the stage-C1 DECLARED OUTPUT CONTRACT.
    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    // Region entry args: block_index (index), strip_row_offset (index), FOLLOWED
    // by columnsPerPass loop-carried per-column f32 VECTOR accumulators.
    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value stripOffset = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // Integer CORE brick: ONE weft_rvv.repack_gemm_lane_wise_q4_x_i8_dot producing
    // the columnsPerPass per-column i32 sumi (a variadic result group). block_index
    // + strip_row_offset tied (anti-bypass) and named off the loop-body's OWN
    // weight/activation ABI bases.
    mlir::OperationState coreState(
        loc, weftrvv::RepackGemmLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex, stripOffset});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_gemm_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    // columnsPerPass dual-fp16 scale FOLD bricks (one per column): each consumes
    // the integer brick's column-c sumi + the column-c carried accumulator and
    // produces the column-c folded-out accumulator (the yield's acc_next). Every
    // column shares the ONE within-block fp16 scale byte offset (weight @0,
    // activation @0 -- the scale d leads each block).
    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t c = 0; c < columnsPerPass; ++c) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackGemmDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(c), accArgs[c], vl, blockIndex,
                             stripOffset});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_gemm_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    // Terminate the region: the loop yield names the columnsPerPass carried-out
    // per-column f32 vector accumulators.
    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemmLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    // The region op is result-less; the abstract op's result must be dead (the
    // repacked lane-wise GEMM writes through the output pointer, not an SSA vector).
    if (!op.getResult().use_empty())
      return op.emitError()
             << "repack-GEMM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEMM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (q4_1 REPACK, GEVM region form): the asymmetric q4_1 sibling of
  // lowerToRepackGemv. Realize a repack-SELECTED, capability-afforded q4_1 DECODE
  // request as the SAME typed weft_rvv.typed_repack_gemv_loop_body REGION as q4_0 --
  // the SHARED q4_0 decomposed bricks (the per-block lane-wise integer CORE
  // weft_rvv.repack_lane_wise_q4_x_i8_dot + the per-strip dual-fp16 scale FOLD
  // weft_rvv.repack_dual_fp16_scale_fold) -- with TWO decode-leaf deltas the q4_1
  // format needs: (a) the CORE brick stamps weight_nibble_unsigned (the q4_1 RAW
  // nibble [0,15] decode -- NO offset-binary -8, the bias lives in the separate MIN
  // scale); (b) the FOLD brick stamps the single MIN-fold offset pair
  // (weight_min_byte_offset @ +32 = the per-row fp16 m_x strip,
  // activation_sum_byte_offset @ +2 = the block_q8_1 scaled-sum s_y), so the fold
  // ADDS the lane-wise `acc += m_x*s_y` correction AFTER the shared `acc +=
  // (d_x*d_y)*sumi` scale fold. It reconstructs the block_q4_1x16 x16 facts (stride
  // 320 = 16 d + 16 m + 256 nibbles, weight quant offset 64) + the PLAIN block_q8_1
  // activation facts (stride 36 / quant offset 4 the op carries), derives the
  // resource-aware half_lanes from the capability VLEN, and stamps the DECLARED
  // OUTPUT CONTRACT weight_layout_contract = "x16". SAFETY: identical to
  // lowerToRepackGemv (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemvQ41(weftrvv::GgmlQuantContractionOp op,
                       const pluginrvv::ContractionSelection &selection,
                       const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);

    // The block_q4_1x16 x16 weight facts (stride 320, weight nibble quant offset 64,
    // per-row fp16 MIN strip @32) + the single-block s_y activation scaled-sum @2.
    std::int64_t weightQuantByteOffset = 64;
    std::int64_t weightMinByteOffset = 32;
    std::int64_t activationSumByteOffset = 2;
    std::int64_t activationQuantByteOffset =
        static_cast<std::int64_t>(op.getQuantByteOffset());

    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemvLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(),
                           op.getColumnCount()});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemv_loop_body"));
    loopState.addAttribute("scale_model",
                           builder.getStringAttr(op.getScaleModel()));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute("weight_block_stride", builder.getI64IntegerAttr(320));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(op.getActivationBlockStride()));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale_min"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t h = 0; h < numHalves; ++h)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // Integer CORE brick (SHARED q4_0 nibble dot) with the q4_1 UNSIGNED-nibble
    // decode selector.
    mlir::OperationState coreState(
        loc, weftrvv::RepackLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    coreState.addAttribute("weight_nibble_unsigned", builder.getUnitAttr());
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t h = 0; h < numHalves; ++h)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    // numHalves dual-fp16 scale FOLD bricks (SHARED q4_0 fold) with the q4_1 single
    // MIN-fold offset pair.
    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t h = 0; h < numHalves; ++h) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(h), accArgs[h], vl, blockIndex});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("weight_min_byte_offset",
                             builder.getI64IntegerAttr(weightMinByteOffset));
      foldState.addAttribute("activation_sum_byte_offset",
                             builder.getI64IntegerAttr(activationSumByteOffset));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemvLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "q4_1 repack-GEVM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEVM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (q4_1 REPACK GEMM finale, region form): the asymmetric q4_1
  // sibling of lowerToRepackGemm. Realize a repack-SELECTED, capability-afforded q4_1
  // PREFILL request as the SAME typed weft_rvv.typed_repack_gemm_loop_body REGION as
  // q4_0 -- the SHARED q4_0 GEMM bricks (integer CORE
  // weft_rvv.repack_gemm_lane_wise_q4_x_i8_dot + per-column dual-fp16 scale FOLD
  // weft_rvv.repack_gemm_dual_fp16_scale_fold) -- with the CORE stamping
  // weight_nibble_unsigned and the FOLD stamping the per-column MIN-fold offset pair
  // (weight_min @ +32, per-column s_y @ +8). It reconstructs the block_q4_1x16 weight
  // facts (stride 320, weight quant offset 64) AND the INTERLEAVED block_q8_1x4
  // activation facts (stride 144, quant offset 16, per-column sum offset 8),
  // MATERIALIZES the two GEMM ABI values (nr, bs), and stamps weight_layout_contract
  // = "x16". SAFETY: identical to lowerToRepackGemm (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemmQ41(weftrvv::GgmlQuantContractionOp op,
                       const pluginrvv::ContractionSelection &selection,
                       const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);
    std::int64_t columnsPerPass = accLmulDecision.usesM1() ? 1 : kActivationInterleave;

    // The block_q4_1x16 weight facts + block_q8_1x4 interleaved activation facts.
    std::int64_t weightBlockStride = 320;
    std::int64_t weightQuantByteOffset = 64;
    std::int64_t weightMinByteOffset = 32;
    std::int64_t activationBlockStride = 144;
    std::int64_t activationQuantByteOffset = 16;
    std::int64_t activationSumByteOffset = 8;

    auto variant = op->getParentOfType<weft::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "q4_1 repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a weft.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, weftrvv::RuntimeABIValueOp::getOperationName());
        st.addAttribute("role", builder.getStringAttr(role));
        st.addAttribute("c_name", builder.getStringAttr(cName));
        st.addAttribute("c_type", builder.getStringAttr("size_t"));
        st.addAttribute("ownership",
                        builder.getStringAttr("target-export-abi-owned"));
        st.addAttribute("purpose", builder.getStringAttr(purpose));
        st.addTypes(builder.getIndexType());
        return builder.create(st)->getResult(0);
      };
      rowCount = makeAbi("nr", "source-byte-stride", "nr");
      outputRowStride = makeAbi("bs", "output-stride", "bs");
    }

    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemmLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(), rowCount,
                           op.getColumnCount(), outputRowStride});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemm_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr(kNibbleQ41GemmScaleModel));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute("weight_block_stride",
                           builder.getI64IntegerAttr(weightBlockStride));
    loopState.addAttribute("activation_block_stride",
                           builder.getI64IntegerAttr(activationBlockStride));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("activation_interleave",
                           builder.getI64IntegerAttr(kActivationInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale_min"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, "q4_1",
            pluginrvv::RVVRepackScheduleRegime::GemmPrefill, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value stripOffset = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    mlir::OperationState coreState(
        loc, weftrvv::RepackGemmLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex, stripOffset});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_gemm_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    coreState.addAttribute("weight_nibble_unsigned", builder.getUnitAttr());
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t c = 0; c < columnsPerPass; ++c) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackGemmDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(c), accArgs[c], vl, blockIndex,
                             stripOffset});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_gemm_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("weight_min_byte_offset",
                             builder.getI64IntegerAttr(weightMinByteOffset));
      foldState.addAttribute("activation_sum_byte_offset",
                             builder.getI64IntegerAttr(activationSumByteOffset));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemmLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "q4_1 repack-GEMM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEMM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (q5_0 REPACK, GEVM region form): the five-bit sibling of
  // lowerToRepackGemv. Realize a repack-SELECTED, capability-afforded q5_0 DECODE
  // request as the SAME typed weft_rvv.typed_repack_gemv_loop_body REGION as q4_0 --
  // the SHARED q4_0 decomposed bricks (the per-block lane-wise integer CORE
  // weft_rvv.repack_lane_wise_q4_x_i8_dot + the per-strip dual-fp16 scale FOLD
  // weft_rvv.repack_dual_fp16_scale_fold, d-ONLY, NO min) -- with ONE decode-leaf
  // delta the q5_0 format needs: the CORE brick stamps weight_nibble_unsigned +
  // weight_qh_byte_offset (@288 = the transposed bit-packed qh 5th-bit plane after
  // the 256 nibbles) + weight_offset_bias (16), so the 5-bit weight assembles
  // `((nibble) | (qh_bit << 4)) - 16` off the RAW unsigned nibble peel. The fold is
  // q4_0's WHOLE (no min). It reconstructs the block_q5_0x16 x16 facts (stride 352 =
  // 16 d + 256 nibbles + 64 transposed qh bytes, weight quant offset 32) + the PLAIN
  // block_q8_0 activation facts (stride 34 / quant offset 2 the op carries), derives
  // the resource-aware half_lanes from the capability VLEN, and stamps the DECLARED
  // OUTPUT CONTRACT weight_layout_contract = "x16". The constructed loop body carries
  // the q4_0 fold scale_model (the abstract ...-five-bit string is the routing
  // discriminator only). SAFETY: identical to lowerToRepackGemv (lit-only, NO
  // e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemvQ50(weftrvv::GgmlQuantContractionOp op,
                       const pluginrvv::ContractionSelection &selection,
                       const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);

    // The block_q5_0x16 x16 weight facts (stride 352, weight nibble quant offset 32,
    // transposed qh 5th-bit plane @288) + the plain block_q8_0 activation quant @2.
    std::int64_t weightQuantByteOffset = 32;
    std::int64_t weightQhByteOffset = 288;
    std::int64_t weightOffsetBias = 16;
    std::int64_t activationQuantByteOffset =
        static_cast<std::int64_t>(op.getQuantByteOffset());

    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemvLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(),
                           op.getColumnCount()});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemv_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr("dual-fp16-per-block-d_x.d_y"));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute("weight_block_stride", builder.getI64IntegerAttr(352));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(op.getActivationBlockStride()));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t h = 0; h < numHalves; ++h)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // Integer CORE brick (SHARED q4_0 nibble dot) with the q5_0 UNSIGNED-nibble +
    // qh 5th-bit + -16 offset decode selector.
    mlir::OperationState coreState(
        loc, weftrvv::RepackLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    coreState.addAttribute("weight_nibble_unsigned", builder.getUnitAttr());
    coreState.addAttribute("weight_qh_byte_offset",
                           builder.getI64IntegerAttr(weightQhByteOffset));
    coreState.addAttribute("weight_offset_bias",
                           builder.getI64IntegerAttr(weightOffsetBias));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t h = 0; h < numHalves; ++h)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    // numHalves dual-fp16 scale FOLD bricks (SHARED q4_0 d-only fold, NO min).
    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t h = 0; h < numHalves; ++h) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(h), accArgs[h], vl, blockIndex});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemvLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "q5_0 repack-GEVM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEVM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (q5_0 REPACK GEMM finale, region form): the five-bit sibling of
  // lowerToRepackGemm. Realize a repack-SELECTED, capability-afforded q5_0 PREFILL
  // request as the SAME typed weft_rvv.typed_repack_gemm_loop_body REGION as q4_0 --
  // the SHARED q4_0 GEMM bricks (integer CORE
  // weft_rvv.repack_gemm_lane_wise_q4_x_i8_dot + per-column dual-fp16 scale FOLD
  // weft_rvv.repack_gemm_dual_fp16_scale_fold, d-ONLY, NO min) -- with the CORE
  // stamping weight_nibble_unsigned + weight_qh_byte_offset (@288) +
  // weight_offset_bias (16); the per-strip qh bit is selected by mask >>
  // strip_row_offset (the RUNTIME strip lane shift). It reconstructs the
  // block_q5_0x16 weight facts (stride 352, weight quant offset 32, qh @288) AND the
  // INTERLEAVED block_q8_0x4 activation facts (stride 136, quant offset 8),
  // MATERIALIZES the two GEMM ABI values (nr, bs), and stamps weight_layout_contract
  // = "x16". SAFETY: identical to lowerToRepackGemm (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemmQ50(weftrvv::GgmlQuantContractionOp op,
                       const pluginrvv::ContractionSelection &selection,
                       const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);
    std::int64_t columnsPerPass = accLmulDecision.usesM1() ? 1 : kActivationInterleave;

    // The block_q5_0x16 weight facts + block_q8_0x4 interleaved activation facts.
    std::int64_t weightBlockStride = 352;
    std::int64_t weightQuantByteOffset = 32;
    std::int64_t weightQhByteOffset = 288;
    std::int64_t weightOffsetBias = 16;
    std::int64_t activationBlockStride = 136;
    std::int64_t activationQuantByteOffset = 8;

    auto variant = op->getParentOfType<weft::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "q5_0 repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a weft.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, weftrvv::RuntimeABIValueOp::getOperationName());
        st.addAttribute("role", builder.getStringAttr(role));
        st.addAttribute("c_name", builder.getStringAttr(cName));
        st.addAttribute("c_type", builder.getStringAttr("size_t"));
        st.addAttribute("ownership",
                        builder.getStringAttr("target-export-abi-owned"));
        st.addAttribute("purpose", builder.getStringAttr(purpose));
        st.addTypes(builder.getIndexType());
        return builder.create(st)->getResult(0);
      };
      rowCount = makeAbi("nr", "source-byte-stride", "nr");
      outputRowStride = makeAbi("bs", "output-stride", "bs");
    }

    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemmLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(), rowCount,
                           op.getColumnCount(), outputRowStride});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemm_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr("dual-fp16-per-block-d_x.d_y"));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute("weight_block_stride",
                           builder.getI64IntegerAttr(weightBlockStride));
    loopState.addAttribute("activation_block_stride",
                           builder.getI64IntegerAttr(activationBlockStride));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("activation_interleave",
                           builder.getI64IntegerAttr(kActivationInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, "q5_0",
            pluginrvv::RVVRepackScheduleRegime::GemmPrefill, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value stripOffset = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    mlir::OperationState coreState(
        loc, weftrvv::RepackGemmLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex, stripOffset});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_gemm_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    coreState.addAttribute("weight_nibble_unsigned", builder.getUnitAttr());
    coreState.addAttribute("weight_qh_byte_offset",
                           builder.getI64IntegerAttr(weightQhByteOffset));
    coreState.addAttribute("weight_offset_bias",
                           builder.getI64IntegerAttr(weightOffsetBias));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t c = 0; c < columnsPerPass; ++c) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackGemmDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(c), accArgs[c], vl, blockIndex,
                             stripOffset});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_gemm_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemmLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "q5_0 repack-GEMM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEMM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (q5_1 REPACK, GEVM region form): the five-bit-with-min sibling
  // of lowerToRepackGemv -- the UNION of the q5_0 5th-bit qh decode and the q4_1
  // single MIN fold. Realize a repack-SELECTED, capability-afforded q5_1 DECODE
  // request as the SAME typed weft_rvv.typed_repack_gemv_loop_body REGION as q4_1 --
  // the SHARED q4_0 decomposed bricks (the per-block lane-wise integer CORE
  // weft_rvv.repack_lane_wise_q4_x_i8_dot + the per-strip dual-fp16 scale FOLD
  // weft_rvv.repack_dual_fp16_scale_fold with the q4_1 MIN pair) -- with the CORE
  // brick stamping weight_nibble_unsigned + weight_qh_byte_offset (@320 = the
  // transposed bit-packed qh 5th-bit plane after the 256 nibbles) but NO
  // weight_offset_bias, so the 5-bit weight assembles UNSIGNED `(nibble) |
  // (qh_bit << 4)` in [0,31] off the RAW nibble peel (the asymmetric bias lives in
  // the q4_1 min fold, unlike q5_0's -16 centering), and the FOLD stamping the
  // single MIN offset pair (weight_min @+32, per-block s_y @+2). It reconstructs the
  // block_q5_1x16 x16 facts (stride 384 = 16 d + 16 m + 256 nibbles + 64 transposed
  // qh bytes, weight quant offset 64, m strip @32, qh @320) + the PLAIN block_q8_1
  // activation facts (stride 36 / quant offset 4 / scaled-sum @2 the op carries),
  // derives the resource-aware half_lanes from the capability VLEN, and stamps the
  // DECLARED OUTPUT CONTRACT weight_layout_contract = "x16". The constructed loop
  // body carries the q4_1 fold scale_model (the abstract ...-plus-min-five-bit
  // string is the routing discriminator only). SAFETY: identical to
  // lowerToRepackGemv (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemvQ51(weftrvv::GgmlQuantContractionOp op,
                       const pluginrvv::ContractionSelection &selection,
                       const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);

    // The block_q5_1x16 x16 weight facts (stride 384, weight nibble quant offset 64,
    // per-row fp16 MIN strip @32, transposed qh 5th-bit plane @320) + the plain
    // block_q8_1 activation scaled-sum @2.
    std::int64_t weightQuantByteOffset = 64;
    std::int64_t weightMinByteOffset = 32;
    std::int64_t weightQhByteOffset = 320;
    std::int64_t activationSumByteOffset = 2;
    std::int64_t activationQuantByteOffset =
        static_cast<std::int64_t>(op.getQuantByteOffset());

    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemvLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(),
                           op.getColumnCount()});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemv_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr(kNibbleQ41ScaleModel));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute("weight_block_stride", builder.getI64IntegerAttr(384));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(op.getActivationBlockStride()));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale_min"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t h = 0; h < numHalves; ++h)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // Integer CORE brick (SHARED q4_0 nibble dot) with the q5_1 UNSIGNED-nibble + qh
    // 5th-bit decode selector -- NO weight_offset_bias (unsigned [0,31]).
    mlir::OperationState coreState(
        loc, weftrvv::RepackLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    coreState.addAttribute("weight_nibble_unsigned", builder.getUnitAttr());
    coreState.addAttribute("weight_qh_byte_offset",
                           builder.getI64IntegerAttr(weightQhByteOffset));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t h = 0; h < numHalves; ++h)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    // numHalves dual-fp16 scale FOLD bricks (SHARED q4_0 fold) with the q4_1 single
    // MIN-fold offset pair (the q5_1 min term).
    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t h = 0; h < numHalves; ++h) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(h), accArgs[h], vl, blockIndex});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("weight_min_byte_offset",
                             builder.getI64IntegerAttr(weightMinByteOffset));
      foldState.addAttribute("activation_sum_byte_offset",
                             builder.getI64IntegerAttr(activationSumByteOffset));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemvLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "q5_1 repack-GEVM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEVM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (q5_1 REPACK GEMM finale, region form): the five-bit-with-min
  // sibling of lowerToRepackGemm -- the UNION of the q5_0 5th-bit qh decode and the
  // q4_1 per-column MIN fold. Realize a repack-SELECTED, capability-afforded q5_1
  // PREFILL request as the SAME typed weft_rvv.typed_repack_gemm_loop_body REGION as
  // q4_1 -- the SHARED q4_0 GEMM bricks (integer CORE
  // weft_rvv.repack_gemm_lane_wise_q4_x_i8_dot + per-column dual-fp16 scale FOLD
  // weft_rvv.repack_gemm_dual_fp16_scale_fold) -- with the CORE stamping
  // weight_nibble_unsigned + weight_qh_byte_offset (@320) but NO weight_offset_bias
  // (the per-strip qh bit selected by mask >> strip_row_offset, the RUNTIME strip
  // lane shift), and the FOLD stamping the per-column MIN-fold offset pair
  // (weight_min @+32, per-column s_y @+8). It reconstructs the block_q5_1x16 weight
  // facts (stride 384, weight quant offset 64, m @32, qh @320) AND the INTERLEAVED
  // block_q8_1x4 activation facts (stride 144, quant offset 16, per-column sum offset
  // 8), MATERIALIZES the two GEMM ABI values (nr, bs), and stamps
  // weight_layout_contract = "x16". SAFETY: identical to lowerToRepackGemm (lit-only,
  // NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemmQ51(weftrvv::GgmlQuantContractionOp op,
                       const pluginrvv::ContractionSelection &selection,
                       const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);
    std::int64_t columnsPerPass = accLmulDecision.usesM1() ? 1 : kActivationInterleave;

    // The block_q5_1x16 weight facts + block_q8_1x4 interleaved activation facts.
    std::int64_t weightBlockStride = 384;
    std::int64_t weightQuantByteOffset = 64;
    std::int64_t weightMinByteOffset = 32;
    std::int64_t weightQhByteOffset = 320;
    std::int64_t activationBlockStride = 144;
    std::int64_t activationQuantByteOffset = 16;
    std::int64_t activationSumByteOffset = 8;

    auto variant = op->getParentOfType<weft::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "q5_1 repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a weft.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, weftrvv::RuntimeABIValueOp::getOperationName());
        st.addAttribute("role", builder.getStringAttr(role));
        st.addAttribute("c_name", builder.getStringAttr(cName));
        st.addAttribute("c_type", builder.getStringAttr("size_t"));
        st.addAttribute("ownership",
                        builder.getStringAttr("target-export-abi-owned"));
        st.addAttribute("purpose", builder.getStringAttr(purpose));
        st.addTypes(builder.getIndexType());
        return builder.create(st)->getResult(0);
      };
      rowCount = makeAbi("nr", "source-byte-stride", "nr");
      outputRowStride = makeAbi("bs", "output-stride", "bs");
    }

    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemmLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(), rowCount,
                           op.getColumnCount(), outputRowStride});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemm_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr(kNibbleQ41GemmScaleModel));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute("weight_block_stride",
                           builder.getI64IntegerAttr(weightBlockStride));
    loopState.addAttribute("activation_block_stride",
                           builder.getI64IntegerAttr(activationBlockStride));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("activation_interleave",
                           builder.getI64IntegerAttr(kActivationInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale_min"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, "q5_1",
            pluginrvv::RVVRepackScheduleRegime::GemmPrefill, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value stripOffset = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    mlir::OperationState coreState(
        loc, weftrvv::RepackGemmLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex, stripOffset});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_gemm_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    coreState.addAttribute("weight_nibble_unsigned", builder.getUnitAttr());
    coreState.addAttribute("weight_qh_byte_offset",
                           builder.getI64IntegerAttr(weightQhByteOffset));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t c = 0; c < columnsPerPass; ++c) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackGemmDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(c), accArgs[c], vl, blockIndex,
                             stripOffset});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_gemm_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("weight_min_byte_offset",
                             builder.getI64IntegerAttr(weightMinByteOffset));
      foldState.addAttribute("activation_sum_byte_offset",
                             builder.getI64IntegerAttr(activationSumByteOffset));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemmLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "q5_1 repack-GEMM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEMM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (q8_0 REPACK, GEVM region form, FLAT-4 收官格): the FULL-int8
  // sibling of lowerToRepackGemv -- the SIMPLEST flat variant (NO nibble unpack, NO
  // qh, NO offset, NO min). Realize a repack-SELECTED, capability-afforded q8_0
  // DECODE request as the SAME typed weft_rvv.typed_repack_gemv_loop_body REGION as
  // q4_0 -- the SHARED per-strip dual-fp16 scale FOLD brick
  // (weft_rvv.repack_dual_fp16_scale_fold, d-ONLY, NO min) -- with the q8_0 CORE
  // brick (weft_rvv.repack_lane_wise_q4_x_i8_dot) stamping weight_full_i8: each
  // weight is a FULL signed int8 (qk=32 positions per block, one vle8 i8 strip load
  // per position, per-position vwmul i8xi8 -> i16 folded into an i32 IN-BLOCK
  // accumulator via vwadd_wv -- NO nibble decode, NO lo/hi split). It reconstructs
  // the block_q8_0x16 x16 facts (stride 544 = 16 d + 512 int8 quants, weight quant
  // offset 32) + the PLAIN block_q8_0 activation facts (stride 34 / quant offset 2
  // the op carries), derives the resource-aware half_lanes from the capability VLEN,
  // and stamps the DECLARED OUTPUT CONTRACT weight_layout_contract = "x16". The
  // constructed loop body carries the q4_0 fold scale_model (the ...-full-i8 string
  // is the routing discriminator only). SAFETY: identical to lowerToRepackGemv
  // (lit-only, NO e2e/perf). This is the FRONT-DOOR construction of the RETIRED
  // emitRepackGemvQ8_0Q8_0 direct emitter.
  mlir::LogicalResult
  lowerToRepackGemvQ80(weftrvv::GgmlQuantContractionOp op,
                       const pluginrvv::ContractionSelection &selection,
                       const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);

    // The block_q8_0x16 x16 weight facts (stride 544 = 16 fp16 d + 512 int8 quants,
    // FULL int8 quants @32) + the plain block_q8_0 activation quant @2.
    std::int64_t weightQuantByteOffset = 32;
    std::int64_t activationQuantByteOffset =
        static_cast<std::int64_t>(op.getQuantByteOffset());

    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemvLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(),
                           op.getColumnCount()});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemv_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr("dual-fp16-per-block-d_x.d_y"));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute("weight_block_stride", builder.getI64IntegerAttr(544));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(op.getActivationBlockStride()));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t h = 0; h < numHalves; ++h)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // Integer CORE brick (SHARED lane-wise dot op) with the q8_0 FULL-int8 decode
    // selector (weight_full_i8): NO nibble/qh/offset flags.
    mlir::OperationState coreState(
        loc, weftrvv::RepackLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    coreState.addAttribute("weight_full_i8", builder.getUnitAttr());
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t h = 0; h < numHalves; ++h)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    // numHalves dual-fp16 scale FOLD bricks (SHARED q4_0 d-only fold, NO min).
    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t h = 0; h < numHalves; ++h) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(h), accArgs[h], vl, blockIndex});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemvLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "q8_0 repack-GEVM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEVM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (q8_0 REPACK GEMM finale, region form, FLAT-4): the FULL-int8
  // sibling of lowerToRepackGemm. NET-NEW construction (no q8_0 GEMM direct emitter
  // ever existed). Realize a repack-SELECTED, capability-afforded q8_0 PREFILL
  // request as the SAME typed weft_rvv.typed_repack_gemm_loop_body REGION as q4_0 --
  // the SHARED per-column dual-fp16 scale FOLD brick
  // (weft_rvv.repack_gemm_dual_fp16_scale_fold, d-ONLY, NO min) -- with the CORE
  // brick (weft_rvv.repack_gemm_lane_wise_q4_x_i8_dot) stamping weight_full_i8. It
  // reconstructs the block_q8_0x16 weight facts (stride 544, weight quant offset 32)
  // AND the INTERLEAVED block_q8_0x4 activation facts (stride 136, quant offset 8),
  // MATERIALIZES the two GEMM ABI values (nr, bs), and stamps weight_layout_contract
  // = "x16". SAFETY: identical to lowerToRepackGemm (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemmQ80(weftrvv::GgmlQuantContractionOp op,
                       const pluginrvv::ContractionSelection &selection,
                       const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);
    std::int64_t columnsPerPass = accLmulDecision.usesM1() ? 1 : kActivationInterleave;

    // The block_q8_0x16 weight facts + block_q8_0x4 interleaved activation facts.
    std::int64_t weightBlockStride = 544;
    std::int64_t weightQuantByteOffset = 32;
    std::int64_t activationBlockStride = 136;
    std::int64_t activationQuantByteOffset = 8;

    auto variant = op->getParentOfType<weft::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "q8_0 repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a weft.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, weftrvv::RuntimeABIValueOp::getOperationName());
        st.addAttribute("role", builder.getStringAttr(role));
        st.addAttribute("c_name", builder.getStringAttr(cName));
        st.addAttribute("c_type", builder.getStringAttr("size_t"));
        st.addAttribute("ownership",
                        builder.getStringAttr("target-export-abi-owned"));
        st.addAttribute("purpose", builder.getStringAttr(purpose));
        st.addTypes(builder.getIndexType());
        return builder.create(st)->getResult(0);
      };
      rowCount = makeAbi("nr", "source-byte-stride", "nr");
      outputRowStride = makeAbi("bs", "output-stride", "bs");
    }

    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemmLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(), rowCount,
                           op.getColumnCount(), outputRowStride});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemm_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr("dual-fp16-per-block-d_x.d_y"));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute("weight_block_stride",
                           builder.getI64IntegerAttr(weightBlockStride));
    loopState.addAttribute("activation_block_stride",
                           builder.getI64IntegerAttr(activationBlockStride));
    loopState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    loopState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("activation_interleave",
                           builder.getI64IntegerAttr(kActivationInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("lane_wise_vector_scale"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, "q8_0",
            pluginrvv::RVVRepackScheduleRegime::GemmPrefill, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value stripOffset = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    mlir::OperationState coreState(
        loc, weftrvv::RepackGemmLaneWiseQ4Q8DotOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex, stripOffset});
    coreState.addAttribute(
        "kind", builder.getStringAttr("repack_gemm_lane_wise_q4_x_i8_dot"));
    coreState.addAttribute("weight_quant_byte_offset",
                           builder.getI64IntegerAttr(weightQuantByteOffset));
    coreState.addAttribute("activation_quant_byte_offset",
                           builder.getI64IntegerAttr(activationQuantByteOffset));
    coreState.addAttribute("weight_full_i8", builder.getUnitAttr());
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      coreState.addTypes(i32ResType);
    mlir::Operation *core = builder.create(coreState);

    llvm::SmallVector<mlir::Value> accNext;
    for (std::int64_t c = 0; c < columnsPerPass; ++c) {
      mlir::OperationState foldState(
          loc, weftrvv::RepackGemmDualFp16ScaleFoldOp::getOperationName());
      foldState.addOperands({op.getWeightBase(), op.getActivationBase(),
                             core->getResult(c), accArgs[c], vl, blockIndex,
                             stripOffset});
      foldState.addAttribute(
          "kind", builder.getStringAttr("repack_gemm_dual_fp16_scale_fold"));
      foldState.addAttribute("weight_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      foldState.addAttribute("activation_scale_byte_offset",
                             builder.getI64IntegerAttr(0));
      if (integerCoreLmul)
        foldState.addAttribute("integer_core_lmul", integerCoreLmul);
      foldState.addTypes(f32AccType);
      accNext.push_back(builder.create(foldState)->getResult(0));
    }

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemmLoopYieldOp::getOperationName());
    yieldState.addOperands(accNext);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "q8_0 repack-GEMM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "GEMM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (TERNARY REPACK, GEVM region form): the ternary tq2_0 sibling
  // of lowerToRepackGemv. Realize a repack-SELECTED, capability-afforded ternary
  // tq2_0 DECODE request as the typed weft_rvv.typed_repack_gemv_loop_body REGION
  // carrying the SINGLE decomposed ternary integer CORE brick
  // (weft_rvv.repack_gemv_ternary_core, decode_model "tq2_0") -- the front-door
  // CONSTRUCTION of the RETIRED monolithic emitRepackGemvTQ20Q8K direct emitter.
  // Where q4_0 decomposes into a nibble core + numHalves dual-fp16 FOLD bricks,
  // the LINEAR ternary fold (single fp16 super-block scale, NO per-sub-block
  // scale, NO dmin, NO min) is RE-EMITTED whole by the typed emitter keyed off the
  // core brick's identity, so the region carries ONLY the core brick + the
  // per-strip accumulator pass-through yield (the SAME construction discipline as
  // the flat ternary vec_dot core brick). It reconstructs the block_tq2_0x16 x16
  // weight facts (stride 1056, interleave 16, weight quant offset 32) + the PLAIN
  // block_q8_K activation facts (stride 292, quant offset 4) the ternary GEVM
  // reads, derives the resource-aware half_lanes from the capability VLEN, and
  // stamps the DECLARED OUTPUT CONTRACT weight_layout_contract = "x16".
  //
  // SAFETY (NOT a latent miscompile): the emitted kernel reads x16 ternary weights
  // but the abstract op carries PLAIN weights, so the emit is correct ONLY when the
  // contract is honored. The abstract GgmlQuantContractionOp has NO real producer
  // (authored ONLY in lit fixtures), so this region is reachable ONLY via lit,
  // NEVER in the real llama.cpp pipeline. NO e2e/perf claim.
  mlir::LogicalResult
  lowerToRepackGemvTernary(weftrvv::GgmlQuantContractionOp op,
                           const pluginrvv::ContractionSelection &selection,
                           const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision,
                           const TernaryDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);

    // The region-carrying loop op: FIVE ABI operands (weight base, activation
    // base, output, element count n, column count nc), NO vl operand and NO result.
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemvLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(),
                           op.getColumnCount()});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemv_loop_body"));
    // The ternary GEVM scale_model WHAT (the abstract op's committed base ternary
    // scale_model -- tq2_0 2-bit or tq1_0 base-3 -- passed through, a pure I4
    // mirror on the loop op).
    loopState.addAttribute("scale_model",
                           builder.getStringAttr(op.getScaleModel()));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    // The repacked block_tq{2,1}_0x16 x16 weight + PLAIN block_q8_K activation ABI
    // facts the verifier / emitter pin (NOT the abstract op's plain stride-66/54 /
    // stride-292 facts -- this region reads the REPACKED weight layout the contract
    // declares), keyed off the per-family TernaryDecodeFacts.
    loopState.addAttribute(
        "weight_block_stride",
        builder.getI64IntegerAttr(facts.weightBlockStride));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(facts.gevmActivationBlockStride));
    loopState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    loopState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gevmActivationQuantByteOffset));
    // The base-3 tq1_0 fold reads a SECOND weight plane (qh); stamp its repacked
    // byte offset ONLY for that family (weightQhByteOffset == 0 == tq2_0 no-qh).
    if (facts.weightQhByteOffset != 0)
      loopState.addAttribute(
          "weight_qh_byte_offset",
          builder.getI64IntegerAttr(facts.weightQhByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("ternary_single_fp16_scale"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    // Region entry args: block_index (index) FOLLOWED by numHalves loop-carried
    // per-strip f32 VECTOR accumulators.
    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t h = 0; h < numHalves; ++h)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // The SOLE in-region brick: ONE weft_rvv.repack_gemv_ternary_core producing the
    // numHalves per-strip i32 sumi (a variadic result group). block_index-tied
    // (anti-bypass) + named off the loop body's OWN weight/activation ABI bases.
    // The typed emitter RE-EMITS the whole byte-exact ternary GEVM body from this
    // brick's identity; the yield passes the carried-in per-strip accumulators
    // through (the ternary fold has no separate FOLD brick, unlike q4_0).
    mlir::OperationState coreState(
        loc, weftrvv::RepackGemvTernaryCoreOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex});
    coreState.addAttribute("kind",
                           builder.getStringAttr("repack_gemv_ternary_core"));
    coreState.addAttribute("decode_model",
                           builder.getStringAttr(facts.decodeModel));
    coreState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    coreState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gevmActivationQuantByteOffset));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t h = 0; h < numHalves; ++h)
      coreState.addTypes(i32ResType);
    (void)builder.create(coreState);

    // Terminate the region: the loop yield names the numHalves carried-out
    // per-strip f32 vector accumulators (the ternary fold is RE-EMITTED whole, so
    // the accumulators pass THROUGH unchanged).
    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemvLoopYieldOp::getOperationName());
    yieldState.addOperands(accArgs);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "ternary repack-GEVM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "ternary GEVM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (TERNARY REPACK GEMM finale, region form): the ternary tq2_0
  // sibling of lowerToRepackGemm. Realize a repack-SELECTED, capability-afforded
  // ternary tq2_0 PREFILL request as the typed weft_rvv.typed_repack_gemm_loop_body
  // REGION carrying the SINGLE decomposed ternary GEMM integer CORE brick
  // (weft_rvv.repack_gemm_ternary_core, decode_model "tq2_0") -- the front-door
  // CONSTRUCTION of the RETIRED monolithic emitRepackGemmTQ20Q8K direct emitter. It
  // reconstructs the block_tq2_0x16 weight facts (stride 1056, interleave 16, weight
  // quant offset 32) AND the INTERLEAVED block_q8_Kx4 activation facts (stride 1168,
  // interleave 4, activation quant offset 16), MATERIALIZES the two GEMM ABI values
  // (row count nr, output row stride bs) the abstract op does not carry, and stamps
  // the DECLARED OUTPUT CONTRACT weight_layout_contract = "x16". The LINEAR ternary
  // fold is RE-EMITTED whole by the typed emitter keyed off the core brick, so the
  // region carries ONLY the core brick + the per-column accumulator pass-through
  // yield. SAFETY: identical to lowerToRepackGemvTernary (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemmTernary(weftrvv::GgmlQuantContractionOp op,
                           const pluginrvv::ContractionSelection &selection,
                           const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision,
                           const TernaryDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);
    std::int64_t columnsPerPass = accLmulDecision.usesM1() ? 1 : kActivationInterleave;

    // Materialize the two runtime ABI values the internalized M-tiling GEMM nest
    // needs but the abstract op does not carry (row count nr, output row stride
    // bs) -- the SAME declared-contract move as the q4_0 repack GEMM.
    auto variant = op->getParentOfType<weft::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "ternary repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a weft.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, weftrvv::RuntimeABIValueOp::getOperationName());
        st.addAttribute("role", builder.getStringAttr(role));
        st.addAttribute("c_name", builder.getStringAttr(cName));
        st.addAttribute("c_type", builder.getStringAttr("size_t"));
        st.addAttribute("ownership",
                        builder.getStringAttr("target-export-abi-owned"));
        st.addAttribute("purpose", builder.getStringAttr(purpose));
        st.addTypes(builder.getIndexType());
        return builder.create(st)->getResult(0);
      };
      rowCount = makeAbi("nr", "source-byte-stride", "nr");
      outputRowStride = makeAbi("bs", "output-stride", "bs");
    }

    // The region-carrying loop op: SEVEN ABI operands (weight base, activation
    // base, output, element count n, row count nr, column count nc, output row
    // stride bs), NO vl operand and NO result.
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemmLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(), rowCount,
                           op.getColumnCount(), outputRowStride});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemm_loop_body"));
    // The 4-column-amortized ternary GEMM scale_model WHAT (the loop op derives it
    // from the per-family TernaryDecodeFacts + the prefill regime; a pure I4 mirror
    // -- the loop-body verifier does NOT pin scale_model for the ternary fold).
    loopState.addAttribute(
        "scale_model", builder.getStringAttr(facts.gemmScaleModel));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute(
        "weight_block_stride",
        builder.getI64IntegerAttr(facts.weightBlockStride));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(facts.gemmActivationBlockStride));
    loopState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    loopState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gemmActivationQuantByteOffset));
    // The base-3 tq1_0 fold reads a SECOND weight plane (qh); stamp its repacked
    // byte offset ONLY for that family (weightQhByteOffset == 0 == tq2_0 no-qh).
    if (facts.weightQhByteOffset != 0)
      loopState.addAttribute(
          "weight_qh_byte_offset",
          builder.getI64IntegerAttr(facts.weightQhByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("activation_interleave",
                           builder.getI64IntegerAttr(kActivationInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr("ternary_single_fp16_scale"));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, facts.decodeModel,
            pluginrvv::RVVRepackScheduleRegime::GemmPrefill, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    // Region entry args: block_index (index), strip_row_offset (index), FOLLOWED
    // by columnsPerPass loop-carried per-column f32 VECTOR accumulators.
    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value stripOffset = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // The SOLE in-region brick: ONE weft_rvv.repack_gemm_ternary_core producing the
    // columnsPerPass per-column i32 sumi (a variadic result group). block_index +
    // strip_row_offset tied (anti-bypass) + named off the loop body's OWN
    // weight/activation ABI bases. The typed emitter RE-EMITS the whole byte-exact
    // ternary GEMM body from this brick's identity; the accumulators pass through.
    mlir::OperationState coreState(
        loc, weftrvv::RepackGemmTernaryCoreOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex, stripOffset});
    coreState.addAttribute("kind",
                           builder.getStringAttr("repack_gemm_ternary_core"));
    coreState.addAttribute("decode_model",
                           builder.getStringAttr(facts.decodeModel));
    coreState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    coreState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gemmActivationQuantByteOffset));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      coreState.addTypes(i32ResType);
    (void)builder.create(coreState);

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemmLoopYieldOp::getOperationName());
    yieldState.addOperands(accArgs);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "ternary repack-GEMM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "ternary GEMM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (K-QUANT REPACK, GEVM region form): the K-quant q4_K sibling of
  // lowerToRepackGemvTernary. Realize a repack-SELECTED, capability-afforded q4_K
  // DECODE request as the typed weft_rvv.typed_repack_gemv_loop_body REGION carrying
  // the SINGLE decomposed K-quant integer CORE brick
  // (weft_rvv.repack_gemv_kquant_core, decode_model "q4_K") -- the front-door
  // CONSTRUCTION of the RETIRED monolithic emitRepackGemvQ4KQ8K direct emitter. Like
  // the ternary core (and UNLIKE the q4_0 nibble core + SEPARATE dual-fp16 FOLD
  // brick), the K-quant dual d/dmin + 6-bit per-sub-block scale/min + bsums-min fold
  // is a NEW fold structure that is RE-EMITTED WHOLE by the typed emitter keyed off
  // the core brick's identity (the K-quant fold is NOT decomposed into a separate
  // fold brick -- it is emitter-inlined, honest scope), so the region carries ONLY
  // the core brick + the per-strip accumulator pass-through yield. It reconstructs
  // the block_q4_Kx16 x16 weight facts (stride 2304, nibbles @256, dmin @32, 6-bit
  // scales @64) + the PLAIN block_q8_K activation facts (stride 292, quant @4, bsums
  // @260) the K-quant GEVM reads (carried on the loop body op's OPTIONAL K-quant
  // attrs), derives the resource-aware half_lanes from the capability VLEN, and
  // stamps the DECLARED OUTPUT CONTRACT weight_layout_contract = "x16".
  //
  // SAFETY (NOT a latent miscompile): the emitted kernel reads x16 q4_K weights but
  // the abstract op carries PLAIN weights, so the emit is correct ONLY when the
  // contract is honored. The abstract GgmlQuantContractionOp has NO real producer
  // (authored ONLY in lit fixtures), so this region is reachable ONLY via lit, NEVER
  // in the real llama.cpp pipeline. NO e2e/perf claim.
  mlir::LogicalResult
  lowerToRepackGemvKQuant(weftrvv::GgmlQuantContractionOp op,
                          const pluginrvv::ContractionSelection &selection,
                          const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision,
                          const KQuantDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);

    // The region-carrying loop op: FIVE ABI operands (weight base, activation base,
    // output, element count n, column count nc), NO vl operand and NO result.
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemvLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(),
                           op.getColumnCount()});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemv_loop_body"));
    loopState.addAttribute("scale_model",
                           builder.getStringAttr(op.getScaleModel()));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute(
        "weight_block_stride",
        builder.getI64IntegerAttr(facts.weightBlockStride));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(facts.gevmActivationBlockStride));
    loopState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    loopState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gevmActivationQuantByteOffset));
    // The K-quant super-block decode facts on the loop body op's OPTIONAL K-quant
    // attrs; the emitter reads them only under the K-quant fold. The scales region
    // + sub-block count are shared by BOTH folds; the q4_K min structure (dmin strip
    // + activation int16 bsums) is set ONLY for hasMin, and the qh SECOND weight
    // plane is set whenever the family HAS one (weightQhByteOffset != 0) -- the q6_K
    // high-2-bit / q3_K hmask high-bit no-min plane, OR the q5_K 5th-bit plane on the
    // MIN fold (q5_K = q4_K min fold + qh, so it stamps BOTH the min structure AND qh).
    if (facts.hasMin) {
      loopState.addAttribute(
          "weight_dmin_byte_offset",
          builder.getI64IntegerAttr(facts.weightDminByteOffset));
      loopState.addAttribute(
          "activation_bsums_byte_offset",
          builder.getI64IntegerAttr(facts.gevmActivationBsumsByteOffset));
    }
    if (facts.weightQhByteOffset != 0) {
      loopState.addAttribute(
          "weight_qh_byte_offset",
          builder.getI64IntegerAttr(facts.weightQhByteOffset));
    }
    loopState.addAttribute(
        "weight_scales_byte_offset",
        builder.getI64IntegerAttr(facts.weightScalesByteOffset));
    loopState.addAttribute("n_subblocks",
                           builder.getI64IntegerAttr(facts.nSubblocks));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model", builder.getStringAttr(facts.foldModel));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, facts.decodeModel,
            pluginrvv::RVVRepackScheduleRegime::Gemv, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t h = 0; h < numHalves; ++h)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // The SOLE in-region brick: ONE weft_rvv.repack_gemv_kquant_core producing the
    // numHalves per-strip i32 sumi. block_index-tied (anti-bypass) + named off the
    // loop body's OWN weight/activation ABI bases. The typed emitter RE-EMITS the
    // whole byte-exact q4_K GEVM body from this brick's identity; the yield passes
    // the carried-in per-strip accumulators through (the K-quant fold has no separate
    // FOLD brick -- it is re-emitted whole, unlike q4_0).
    mlir::OperationState coreState(
        loc, weftrvv::RepackGemvKQuantCoreOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex});
    coreState.addAttribute("kind",
                           builder.getStringAttr("repack_gemv_kquant_core"));
    coreState.addAttribute("decode_model",
                           builder.getStringAttr(facts.decodeModel));
    coreState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    coreState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gevmActivationQuantByteOffset));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t h = 0; h < numHalves; ++h)
      coreState.addTypes(i32ResType);
    (void)builder.create(coreState);

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemvLoopYieldOp::getOperationName());
    yieldState.addOperands(accArgs);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "K-quant repack-GEVM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "K-quant GEVM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (K-QUANT REPACK GEMM finale, region form): the K-quant q4_K
  // sibling of lowerToRepackGemmTernary. Realize a repack-SELECTED,
  // capability-afforded q4_K PREFILL request as the typed
  // weft_rvv.typed_repack_gemm_loop_body REGION carrying the SINGLE decomposed
  // K-quant GEMM integer CORE brick (weft_rvv.repack_gemm_kquant_core, decode_model
  // "q4_K") -- the front-door CONSTRUCTION of the RETIRED monolithic
  // emitRepackGemmQ4KQ8K direct emitter. It reconstructs the block_q4_Kx16 weight
  // facts (stride 2304, nibbles @256, dmin @32, 6-bit scales @64) AND the INTERLEAVED
  // block_q8_Kx4 activation facts (stride 1168, quant @16, bsums @1040), MATERIALIZES
  // the two GEMM ABI values (row count nr, output row stride bs) the abstract op does
  // not carry, and stamps the DECLARED OUTPUT CONTRACT weight_layout_contract = "x16".
  // The K-quant dual d/dmin + bsums-min fold is RE-EMITTED whole by the typed emitter
  // keyed off the core brick, so the region carries ONLY the core brick + the
  // per-column accumulator pass-through yield. SAFETY: identical to
  // lowerToRepackGemvKQuant (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemmKQuant(weftrvv::GgmlQuantContractionOp op,
                          const pluginrvv::ContractionSelection &selection,
                          const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision,
                          const KQuantDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);
    std::int64_t columnsPerPass = accLmulDecision.usesM1() ? 1 : kActivationInterleave;

    // Materialize the two runtime ABI values the internalized M-tiling GEMM nest
    // needs but the abstract op does not carry (row count nr, output row stride bs)
    // -- the SAME declared-contract move as the q4_0 / ternary repack GEMM.
    auto variant = op->getParentOfType<weft::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "K-quant repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a weft.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, weftrvv::RuntimeABIValueOp::getOperationName());
        st.addAttribute("role", builder.getStringAttr(role));
        st.addAttribute("c_name", builder.getStringAttr(cName));
        st.addAttribute("c_type", builder.getStringAttr("size_t"));
        st.addAttribute("ownership",
                        builder.getStringAttr("target-export-abi-owned"));
        st.addAttribute("purpose", builder.getStringAttr(purpose));
        st.addTypes(builder.getIndexType());
        return builder.create(st)->getResult(0);
      };
      rowCount = makeAbi("nr", "source-byte-stride", "nr");
      outputRowStride = makeAbi("bs", "output-stride", "bs");
    }

    // The region-carrying loop op: SEVEN ABI operands (weight base, activation base,
    // output, element count n, row count nr, column count nc, output row stride bs),
    // NO vl operand and NO result.
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemmLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(), rowCount,
                           op.getColumnCount(), outputRowStride});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemm_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr(facts.gemmScaleModel));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute(
        "weight_block_stride",
        builder.getI64IntegerAttr(facts.weightBlockStride));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(facts.gemmActivationBlockStride));
    loopState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    loopState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gemmActivationQuantByteOffset));
    // K-quant super-block decode facts (OPTIONAL loop-body attrs). scales + n_subblocks
    // are shared; the q4_K min structure (dmin + bsums) is set ONLY for hasMin, and the
    // qh SECOND weight plane is set whenever the family HAS one (weightQhByteOffset != 0):
    // the q6_K/q3_K no-min plane OR the q5_K 5th-bit plane on the MIN fold (mirrors the
    // GEVM).
    if (facts.hasMin) {
      loopState.addAttribute(
          "weight_dmin_byte_offset",
          builder.getI64IntegerAttr(facts.weightDminByteOffset));
      loopState.addAttribute(
          "activation_bsums_byte_offset",
          builder.getI64IntegerAttr(facts.gemmActivationBsumsByteOffset));
    }
    if (facts.weightQhByteOffset != 0) {
      loopState.addAttribute(
          "weight_qh_byte_offset",
          builder.getI64IntegerAttr(facts.weightQhByteOffset));
    }
    loopState.addAttribute(
        "weight_scales_byte_offset",
        builder.getI64IntegerAttr(facts.weightScalesByteOffset));
    loopState.addAttribute("n_subblocks",
                           builder.getI64IntegerAttr(facts.nSubblocks));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("activation_interleave",
                           builder.getI64IntegerAttr(kActivationInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model", builder.getStringAttr(facts.foldModel));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, facts.decodeModel,
            pluginrvv::RVVRepackScheduleRegime::GemmPrefill, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value stripOffset = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // The SOLE in-region brick: ONE weft_rvv.repack_gemm_kquant_core producing the
    // columnsPerPass per-column i32 sumi. block_index + strip_row_offset tied
    // (anti-bypass) + named off the loop body's OWN weight/activation ABI bases. The
    // typed emitter RE-EMITS the whole byte-exact q4_K GEMM body from this brick's
    // identity; the accumulators pass through.
    mlir::OperationState coreState(
        loc, weftrvv::RepackGemmKQuantCoreOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex, stripOffset});
    coreState.addAttribute("kind",
                           builder.getStringAttr("repack_gemm_kquant_core"));
    coreState.addAttribute("decode_model",
                           builder.getStringAttr(facts.decodeModel));
    coreState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    coreState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gemmActivationQuantByteOffset));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      coreState.addTypes(i32ResType);
    (void)builder.create(coreState);

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemmLoopYieldOp::getOperationName());
    yieldState.addOperands(accArgs);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "K-quant repack-GEMM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "K-quant GEMM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (CODEBOOK REPACK, GEVM region form): the codebook iq4_nl sibling of
  // lowerToRepackGemvKQuant. Realize a repack-SELECTED, capability-afforded iq4_nl DECODE
  // request as the typed weft_rvv.typed_repack_gemv_loop_body REGION carrying the SINGLE
  // decomposed codebook integer CORE brick (weft_rvv.repack_gemv_codebook_core, decode_model
  // "iq4_nl") -- the front-door CONSTRUCTION of the RETIRED monolithic emitRepackGemvIq4NlQ80
  // direct emitter. Like the K-quant core (and UNLIKE the q4_0 nibble core + SEPARATE
  // dual-fp16 FOLD brick), the codebook flat single-fp16-scale fold is RE-EMITTED WHOLE by
  // the typed emitter keyed off the core brick's identity, so the region carries ONLY the
  // core brick + the per-strip accumulator pass-through yield. It reconstructs the
  // block_iq4_nlx16 x16 weight facts (stride 288, nibbles @32) + the PLAIN block_q8_0
  // activation facts (stride 34, quant @2) the codebook GEVM reads (carried on the loop
  // body op's block facts), derives the resource-aware half_lanes from the capability VLEN,
  // stamps the DECLARED OUTPUT CONTRACT weight_layout_contract = "x16", and RECONSTRUCTS the
  // 16-entry non-linear codebook (kvalues_iq4nl) onto the core brick (the abstract op
  // carries no codebook). SAFETY: identical to lowerToRepackGemvKQuant (lit-only, NO
  // e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemvCodebook(weftrvv::GgmlQuantContractionOp op,
                            const pluginrvv::ContractionSelection &selection,
                            const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision,
                            const CodebookDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);

    // The region-carrying loop op: FIVE ABI operands (weight base, activation base,
    // output, element count n, column count nc), NO vl operand and NO result. The
    // codebook flat fold carries NO K-quant super-block attrs (no dmin/scales/bsums/qh/
    // n_subblocks) -- it is the flat single-scale sibling of the q4_0 nibble fold.
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemvLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(),
                           op.getColumnCount()});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemv_loop_body"));
    loopState.addAttribute("scale_model",
                           builder.getStringAttr(op.getScaleModel()));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute(
        "weight_block_stride",
        builder.getI64IntegerAttr(facts.weightBlockStride));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(facts.gevmActivationBlockStride));
    loopState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    loopState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gevmActivationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    // The iq4_xs SUPER-BLOCK signed-6 scale facts on the loop body op's OPTIONAL attrs
    // (the emitter reads them only under the codebook super-block fold): the scales_l LOW
    // pair region + the scales_h HIGH 2-bit region + the sub-block count. ABSENT for the
    // iq4_nl flat single-scale fold (facts.nSubblocks == 0).
    if (facts.nSubblocks != 0) {
      loopState.addAttribute(
          "weight_scales_byte_offset",
          builder.getI64IntegerAttr(facts.weightScalesLowByteOffset));
      loopState.addAttribute(
          "weight_scales_high_byte_offset",
          builder.getI64IntegerAttr(facts.weightScalesHighByteOffset));
      loopState.addAttribute("n_subblocks",
                             builder.getI64IntegerAttr(facts.nSubblocks));
    }
    loopState.addAttribute("fold_model",
                           builder.getStringAttr(facts.foldModel));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t h = 0; h < numHalves; ++h)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // The SOLE in-region brick: ONE weft_rvv.repack_gemv_codebook_core producing the
    // numHalves per-strip i32 sumi. block_index-tied (anti-bypass) + named off the loop
    // body's OWN weight/activation ABI bases + the RECONSTRUCTED 16-entry codebook. The
    // typed emitter RE-EMITS the whole byte-exact iq4_nl GEVM body from this brick's
    // identity; the yield passes the carried-in per-strip accumulators through.
    mlir::OperationState coreState(
        loc, weftrvv::RepackGemvCodebookCoreOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex});
    coreState.addAttribute("kind",
                           builder.getStringAttr("repack_gemv_codebook_core"));
    coreState.addAttribute("decode_model",
                           builder.getStringAttr(facts.decodeModel));
    coreState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    coreState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gevmActivationQuantByteOffset));
    coreState.addAttribute("codebook",
                           builder.getDenseI8ArrayAttr(facts.codebook));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t h = 0; h < numHalves; ++h)
      coreState.addTypes(i32ResType);
    (void)builder.create(coreState);

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemvLoopYieldOp::getOperationName());
    yieldState.addOperands(accArgs);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "codebook repack-GEVM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "codebook GEVM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (CODEBOOK REPACK GEMM, region form): the codebook iq4_nl sibling of
  // lowerToRepackGemmKQuant. Realize a repack-SELECTED, capability-afforded iq4_nl PREFILL
  // request as the typed weft_rvv.typed_repack_gemm_loop_body REGION carrying the SINGLE
  // decomposed codebook GEMM integer CORE brick (weft_rvv.repack_gemm_codebook_core,
  // decode_model "iq4_nl") -- the front-door CONSTRUCTION of the RETIRED monolithic
  // emitRepackGemmIq4NlQ80 direct emitter. It reconstructs the block_iq4_nlx16 weight facts
  // (stride 288, nibbles @32) AND the INTERLEAVED block_q8_0x4 activation facts (stride 136,
  // quant @8), MATERIALIZES the two GEMM ABI values (row count nr, output row stride bs) the
  // abstract op does not carry, stamps weight_layout_contract = "x16", and RECONSTRUCTS the
  // 16-entry codebook onto the core brick. The codebook GEMM ships PLAIN (untiled): iq4_nl
  // already sits at the <=32-vreg cliff, so S6 output tiling is a structural no-op. SAFETY:
  // identical to lowerToRepackGemvCodebook (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemmCodebook(weftrvv::GgmlQuantContractionOp op,
                            const pluginrvv::ContractionSelection &selection,
                            const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision,
                            const CodebookDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);
    std::int64_t columnsPerPass = accLmulDecision.usesM1() ? 1 : kActivationInterleave;

    // Materialize the two runtime ABI values the internalized M-tiling GEMM nest needs but
    // the abstract op does not carry (row count nr, output row stride bs) -- the SAME
    // declared-contract move as the q4_0 / ternary / K-quant repack GEMM.
    auto variant = op->getParentOfType<weft::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "codebook repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a weft.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, weftrvv::RuntimeABIValueOp::getOperationName());
        st.addAttribute("role", builder.getStringAttr(role));
        st.addAttribute("c_name", builder.getStringAttr(cName));
        st.addAttribute("c_type", builder.getStringAttr("size_t"));
        st.addAttribute("ownership",
                        builder.getStringAttr("target-export-abi-owned"));
        st.addAttribute("purpose", builder.getStringAttr(purpose));
        st.addTypes(builder.getIndexType());
        return builder.create(st)->getResult(0);
      };
      rowCount = makeAbi("nr", "source-byte-stride", "nr");
      outputRowStride = makeAbi("bs", "output-stride", "bs");
    }

    // The region-carrying loop op: SEVEN ABI operands (weight base, activation base,
    // output, element count n, row count nr, column count nc, output row stride bs), NO vl
    // operand and NO result. The codebook flat fold carries NO K-quant super-block attrs.
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemmLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(), rowCount,
                           op.getColumnCount(), outputRowStride});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemm_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr(facts.gemmScaleModel));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute(
        "weight_block_stride",
        builder.getI64IntegerAttr(facts.weightBlockStride));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(facts.gemmActivationBlockStride));
    loopState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    loopState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gemmActivationQuantByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("activation_interleave",
                           builder.getI64IntegerAttr(kActivationInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    // The iq4_xs SUPER-BLOCK signed-6 scale facts on the loop body op's OPTIONAL attrs
    // (the emitter reads them only under the codebook super-block fold): the scales_l LOW
    // pair region + the scales_h HIGH 2-bit region + the sub-block count. ABSENT for the
    // iq4_nl flat single-scale fold (facts.nSubblocks == 0).
    if (facts.nSubblocks != 0) {
      loopState.addAttribute(
          "weight_scales_byte_offset",
          builder.getI64IntegerAttr(facts.weightScalesLowByteOffset));
      loopState.addAttribute(
          "weight_scales_high_byte_offset",
          builder.getI64IntegerAttr(facts.weightScalesHighByteOffset));
      loopState.addAttribute("n_subblocks",
                             builder.getI64IntegerAttr(facts.nSubblocks));
    }
    loopState.addAttribute("fold_model",
                           builder.getStringAttr(facts.foldModel));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, facts.decodeModel,
            pluginrvv::RVVRepackScheduleRegime::GemmPrefill, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value stripOffset = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // The SOLE in-region brick: ONE weft_rvv.repack_gemm_codebook_core producing the
    // columnsPerPass per-column i32 sumi. block_index + strip_row_offset tied (anti-bypass)
    // + named off the loop body's OWN weight/activation ABI bases + the RECONSTRUCTED
    // codebook. The typed emitter RE-EMITS the whole byte-exact iq4_nl GEMM body (PLAIN
    // untiled) from this brick's identity; the accumulators pass through.
    mlir::OperationState coreState(
        loc, weftrvv::RepackGemmCodebookCoreOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex, stripOffset});
    coreState.addAttribute("kind",
                           builder.getStringAttr("repack_gemm_codebook_core"));
    coreState.addAttribute("decode_model",
                           builder.getStringAttr(facts.decodeModel));
    coreState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightQuantByteOffset));
    coreState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gemmActivationQuantByteOffset));
    coreState.addAttribute("codebook",
                           builder.getDenseI8ArrayAttr(facts.codebook));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      coreState.addTypes(i32ResType);
    (void)builder.create(coreState);

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemmLoopYieldOp::getOperationName());
    yieldState.addOperands(accArgs);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "codebook repack-GEMM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "codebook GEMM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (GRID REPACK GEVM, region form): the iq2_xxs GRID CODEBOOK +
  // SIGN-PLANE sibling of lowerToRepackGemvCodebook. Realize a repack-SELECTED,
  // capability-afforded iq2_xxs DECODE request as the typed
  // weft_rvv.typed_repack_gemv_loop_body REGION carrying the SINGLE decomposed grid GEVM
  // integer CORE brick (weft_rvv.repack_gemv_grid_core, decode_model "iq2_xxs") -- the
  // front-door CONSTRUCTION of the RETIRED monolithic emitRepackGemvIq2XxsQ8K direct
  // emitter. It reconstructs the block_iq2_xxsx16 weight facts (stride 1184, grid-index
  // @160, ls @32, sign @672) + the plain block_q8_K activation facts (292/4), stamps
  // weight_layout_contract = "x16", and emplaces the grid decode facts (grid/ls/sign byte
  // offsets + n_subblocks) onto the grid core brick. The FIXED grid + DERIVED signs64
  // planes are NOT reconstructed here (the emitter spells them as static const decls).
  // SAFETY: identical to lowerToRepackGemvCodebook (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemvGrid(weftrvv::GgmlQuantContractionOp op,
                        const pluginrvv::ContractionSelection &selection,
                        const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision,
                        const Iq2GridDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);

    // The region-carrying loop op: FIVE ABI operands (weight base, activation base,
    // output, element count n, column count nc), NO vl operand and NO result. The grid
    // fold carries NO super-block attrs on the loop body -- the grid/ls/sign offsets +
    // n_subblocks ride on the grid core brick.
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemvLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(),
                           op.getColumnCount()});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemv_loop_body"));
    loopState.addAttribute("scale_model",
                           builder.getStringAttr(op.getScaleModel()));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute(
        "weight_block_stride",
        builder.getI64IntegerAttr(facts.weightBlockStride));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(facts.gevmActivationBlockStride));
    loopState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightGridIdxByteOffset));
    loopState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gevmActivationQuantByteOffset));
    // The activation int16 bsums offset: stamped ONLY by the iq1_s delta fold, the
    // only grid row that reads them (the iq2 rows carry 0 and MUST NOT stamp it --
    // the loop-body verifier rejects the attr outside the two folds that read it).
    if (facts.gevmActivationBsumsByteOffset != 0)
      loopState.addAttribute(
          "activation_bsums_byte_offset",
          builder.getI64IntegerAttr(facts.gevmActivationBsumsByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr(facts.foldModel));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t h = 0; h < numHalves; ++h)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // The SOLE in-region brick: ONE weft_rvv.repack_gemv_grid_core producing the numHalves
    // per-strip i32 sumi. block_index-tied (anti-bypass) + named off the loop body's OWN
    // weight/activation ABI bases + the grid/ls/sign decode facts. The typed emitter
    // RE-EMITS the whole byte-exact iq2_xxs GEVM body from this brick's identity; the yield
    // passes the carried-in per-strip accumulators through.
    mlir::OperationState coreState(
        loc, weftrvv::RepackGemvGridCoreOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex});
    coreState.addAttribute("kind",
                           builder.getStringAttr("repack_gemv_grid_core"));
    coreState.addAttribute("decode_model",
                           builder.getStringAttr(facts.decodeModel));
    coreState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightGridIdxByteOffset));
    coreState.addAttribute(
        "weight_ls_byte_offset",
        builder.getI64IntegerAttr(facts.weightLsByteOffset));
    coreState.addAttribute(
        "weight_sign_byte_offset",
        builder.getI64IntegerAttr(facts.weightSignByteOffset));
    coreState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gevmActivationQuantByteOffset));
    coreState.addAttribute("n_subblocks",
                           builder.getI64IntegerAttr(facts.nSubblocks));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t h = 0; h < numHalves; ++h)
      coreState.addTypes(i32ResType);
    (void)builder.create(coreState);

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemvLoopYieldOp::getOperationName());
    yieldState.addOperands(accArgs);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "grid repack-GEVM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "grid GEVM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // STAGE C1 bridge (GRID REPACK GEMM, region form): the iq2_xxs sibling of
  // lowerToRepackGemmCodebook. Realize a repack-SELECTED, capability-afforded iq2_xxs
  // PREFILL request as the typed weft_rvv.typed_repack_gemm_loop_body REGION carrying the
  // SINGLE decomposed grid GEMM integer CORE brick (weft_rvv.repack_gemm_grid_core,
  // decode_model "iq2_xxs") -- the front-door CONSTRUCTION of the RETIRED monolithic
  // emitRepackGemmIq2XxsQ8K direct emitter. It reconstructs the block_iq2_xxsx16 weight
  // facts (stride 1184, grid-index @160, ls @32, sign @672) AND the INTERLEAVED
  // block_q8_Kx4 activation facts (stride 1168, quant @16), MATERIALIZES the two GEMM ABI
  // values (row count nr, output row stride bs) the abstract op does not carry, stamps
  // weight_layout_contract = "x16", and emplaces the grid decode facts onto the grid core
  // brick. The grid GEMM ships PLAIN (untiled): iq2_xxs already sits at the <=32-vreg cliff
  // (a memory-gather grid decode, NO min-fold weight-panel spill), so S6 output tiling is a
  // structural no-op. SAFETY: identical to lowerToRepackGemvGrid (lit-only, NO e2e/perf).
  mlir::LogicalResult
  lowerToRepackGemmGrid(weftrvv::GgmlQuantContractionOp op,
                        const pluginrvv::ContractionSelection &selection,
                        const pluginrvv::RepackAccumulatorLMULDecision &accLmulDecision,
                        const Iq2GridDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = accLmulDecision.selectedHalfLanes;
    llvm::StringRef accLmul = accLmulDecision.accumulatorLMUL;
    mlir::Type f32AccType =
        weftrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        weftrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        builder.getStringAttr(accLmulDecision.integerCoreLMUL);
    std::int64_t columnsPerPass = accLmulDecision.usesM1() ? 1 : kActivationInterleave;

    // Materialize the two runtime ABI values the internalized M-tiling GEMM nest needs but
    // the abstract op does not carry (row count nr, output row stride bs) -- the SAME
    // declared-contract move as the q4_0 / ternary / K-quant / codebook repack GEMM.
    auto variant = op->getParentOfType<weft::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "grid repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a weft.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, weftrvv::RuntimeABIValueOp::getOperationName());
        st.addAttribute("role", builder.getStringAttr(role));
        st.addAttribute("c_name", builder.getStringAttr(cName));
        st.addAttribute("c_type", builder.getStringAttr("size_t"));
        st.addAttribute("ownership",
                        builder.getStringAttr("target-export-abi-owned"));
        st.addAttribute("purpose", builder.getStringAttr(purpose));
        st.addTypes(builder.getIndexType());
        return builder.create(st)->getResult(0);
      };
      rowCount = makeAbi("nr", "source-byte-stride", "nr");
      outputRowStride = makeAbi("bs", "output-stride", "bs");
    }

    // The region-carrying loop op: SEVEN ABI operands (weight base, activation base,
    // output, element count n, row count nr, column count nc, output row stride bs), NO vl
    // operand and NO result. The grid fold carries NO super-block attrs on the loop body.
    mlir::OperationState loopState(
        loc, weftrvv::TypedRepackGemmLoopBodyOp::getOperationName());
    loopState.addOperands({op.getWeightBase(), op.getActivationBase(),
                           op.getOutput(), op.getElementCount(), rowCount,
                           op.getColumnCount(), outputRowStride});
    loopState.addAttribute(
        "kind", builder.getStringAttr("typed_repack_gemm_loop_body"));
    loopState.addAttribute(
        "scale_model", builder.getStringAttr(facts.gemmScaleModel));
    loopState.addAttribute("qk", builder.getI64IntegerAttr(op.getQk()));
    loopState.addAttribute(
        "weight_block_stride",
        builder.getI64IntegerAttr(facts.weightBlockStride));
    loopState.addAttribute(
        "activation_block_stride",
        builder.getI64IntegerAttr(facts.gemmActivationBlockStride));
    loopState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightGridIdxByteOffset));
    loopState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gemmActivationQuantByteOffset));
    // The INTERLEAVED block_q8_Kx4 bsums offset (group16-major / column-minor):
    // stamped ONLY by the iq1_s delta fold. See the GEVM sibling.
    if (facts.gemmActivationBsumsByteOffset != 0)
      loopState.addAttribute(
          "activation_bsums_byte_offset",
          builder.getI64IntegerAttr(facts.gemmActivationBsumsByteOffset));
    loopState.addAttribute("weight_interleave",
                           builder.getI64IntegerAttr(kWeightInterleave));
    loopState.addAttribute("activation_interleave",
                           builder.getI64IntegerAttr(kActivationInterleave));
    loopState.addAttribute("half_lanes",
                           builder.getI64IntegerAttr(emittedHalfLanes));
    loopState.addAttribute("fold_model",
                           builder.getStringAttr(facts.foldModel));
    if (integerCoreLmul)
      loopState.addAttribute("integer_core_lmul", integerCoreLmul);
    if (mlir::failed(addRepackScheduleFormula(
            builder, loopState, facts.decodeModel,
            pluginrvv::RVVRepackScheduleRegime::GemmPrefill, op)))
      return mlir::failure();
    loopState.addRegion();
    auto loop = llvm::cast<weftrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    stampRepackAccumulatorLMULDecision(builder, loop, accLmulDecision);
    loop->setAttr(kMaterializationAttr, builder.getStringAttr("realized"));
    loop->setAttr(kWeightLayoutContractAttr, builder.getStringAttr("x16"));

    mlir::Block &body = loop.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value stripOffset = body.addArgument(builder.getIndexType(), loc);
    llvm::SmallVector<mlir::Value> accArgs;
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      accArgs.push_back(body.addArgument(f32AccType, loc));

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::Value vl = op.getVl();

    // The SOLE in-region brick: ONE weft_rvv.repack_gemm_grid_core producing the
    // columnsPerPass per-column i32 sumi. block_index + strip_row_offset tied (anti-bypass)
    // + named off the loop body's OWN weight/activation ABI bases + the grid decode facts.
    // The typed emitter RE-EMITS the whole byte-exact iq2_xxs GEMM body (PLAIN untiled) from
    // this brick's identity; the accumulators pass through.
    mlir::OperationState coreState(
        loc, weftrvv::RepackGemmGridCoreOp::getOperationName());
    coreState.addOperands(
        {op.getWeightBase(), op.getActivationBase(), vl, blockIndex, stripOffset});
    coreState.addAttribute("kind",
                           builder.getStringAttr("repack_gemm_grid_core"));
    coreState.addAttribute("decode_model",
                           builder.getStringAttr(facts.decodeModel));
    coreState.addAttribute(
        "weight_quant_byte_offset",
        builder.getI64IntegerAttr(facts.weightGridIdxByteOffset));
    coreState.addAttribute(
        "weight_ls_byte_offset",
        builder.getI64IntegerAttr(facts.weightLsByteOffset));
    coreState.addAttribute(
        "weight_sign_byte_offset",
        builder.getI64IntegerAttr(facts.weightSignByteOffset));
    coreState.addAttribute(
        "activation_quant_byte_offset",
        builder.getI64IntegerAttr(facts.gemmActivationQuantByteOffset));
    coreState.addAttribute("n_subblocks",
                           builder.getI64IntegerAttr(facts.nSubblocks));
    if (integerCoreLmul)
      coreState.addAttribute("integer_core_lmul", integerCoreLmul);
    for (std::int64_t c = 0; c < columnsPerPass; ++c)
      coreState.addTypes(i32ResType);
    (void)builder.create(coreState);

    mlir::OperationState yieldState(
        loc, weftrvv::TypedRepackGemmLoopYieldOp::getOperationName());
    yieldState.addOperands(accArgs);
    (void)builder.create(yieldState);

    if (!op.getResult().use_empty())
      return op.emitError()
             << "grid repack-GEMM region lowering requires the abstract "
                "quant_contraction result to be unused (the repacked lane-wise "
                "grid GEMM sinks through the output pointer, not an SSA vector)";
    op.erase();
    return mlir::success();
  }

  // Option (i): emit the byte-identical weft_rvv.q4_0_q8_0_block_dot body for BOTH
  // the BlockDot-selected (realized) and the Repack-selected (deferred-stage-c)
  // cases, reconstructing today's hand-authored attrs verbatim and DROPPING
  // column_count (nc). The ONLY per-cell difference is the three inert audit attrs
  // recording the in-compiler decision. The emitted C is byte-identical to today
  // on every path; the repack op is materialized by stage C, never here.
  mlir::LogicalResult
  lowerToBlockDot(weftrvv::GgmlQuantContractionOp op,
                  const pluginrvv::ContractionSelection &selection) {
    mlir::OpBuilder builder(op);

    // Operands: DROP column_count (nc) -- the block-dot vec_dot delegates M/N to
    // ggml's mul_mat caller and is a bare 4-operand-plus-vl op.
    auto blockDot = builder.create<weftrvv::GgmlBlockDotQ40Q80Op>(
        op.getLoc(), op.getResult().getType(),
        /*weight_base=*/op.getWeightBase(),
        /*activation_base=*/op.getActivationBase(),
        /*output=*/op.getOutput(),
        /*element_count=*/op.getElementCount(),
        /*vl=*/op.getVl(),
        // Attrs reconstructed verbatim from the abstract request's pinned facts.
        /*kind=*/llvm::StringRef("ggml_q4_0_q8_0_block_dot"),
        /*scale_model=*/op.getScaleModel(),
        /*qk=*/static_cast<uint64_t>(op.getQk()),
        /*weight_block_stride=*/
        static_cast<uint64_t>(op.getWeightBlockStride()),
        /*activation_block_stride=*/
        static_cast<uint64_t>(op.getActivationBlockStride()),
        /*quant_byte_offset=*/static_cast<uint64_t>(op.getQuantByteOffset()),
        /*activation_high_byte_offset=*/
        static_cast<uint64_t>(op.getActivationHighByteOffset()),
        // NO schedule knobs -- MaterializeRVVQ40Schedule stamps them downstream,
        // exactly as today.
        /*integer_core_lmul=*/::mlir::StringAttr(),
        /*multi_block_factor=*/::mlir::IntegerAttr(),
        /*strip_elision=*/::mlir::StringAttr());

    // Stamp the in-compiler decision as INERT audit attrs (emitter-ignored
    // provenance, like weft_rvv.q4_0_schedule.*). Repack-selected records that
    // the repack DECISION is real but its weight materialization is deferred to
    // stage C; BlockDot-selected records the choice as fully realized.
    bool isRepack =
        selection.algorithm == pluginrvv::ContractionAlgorithm::Repack;
    blockDot->setAttr(kAlgorithmAttr,
                      builder.getStringAttr(isRepack ? "repack" : "block-dot"));
    blockDot->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
    blockDot->setAttr(kMaterializationAttr,
                      builder.getStringAttr(isRepack ? "deferred-stage-c"
                                                     : "realized"));

    op.getResult().replaceAllUsesWith(blockDot.getResult());
    op.erase();
    return mlir::success();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createRVVLowerQuantContractionPass() {
  return std::make_unique<RVVLowerQuantContractionPass>();
}

} // namespace weft::transforms
