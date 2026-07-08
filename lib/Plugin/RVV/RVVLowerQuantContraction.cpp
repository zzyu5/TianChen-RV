//===- RVVLowerQuantContraction.cpp ---------------------------------------===//
//
// The option-2 front-of-pipeline pass that lowers the abstract,
// algorithm-UNCOMMITTED tcrv_rvv.quant_contraction op to a CONCRETE contraction
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
// (tcrv_rvv.*) so it is provable in-IR and lit-CHECKable.
//
// STAGE C1 (this file, the in-IR BRIDGE): the pass now LOWERS a repack-SELECTED
// request to the REAL tcrv_rvv.repack_gemv_q4_0_q8_0 op and DECLARES the kernel's
// weight-layout requirement as an OUTPUT CONTRACT (tcrv_rvv.weight_layout_contract
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

#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVContractionPathSelection.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;
namespace pluginrvv = ::tianchenrv::plugin::rvv;

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_RVVLOWERQUANTCONTRACTION
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

// The inert audit-attr names the stage-B pass stamps on the lowered concrete op.
// They are pure provenance (no SEW/LMUL/policy/dataflow config) -- the EmitC
// emitter ignores them, exactly like the MaterializeRVVQ40Schedule pass's
// additive "tcrv_rvv.q4_0_schedule.*" trail -- so the emitted C is byte-identical
// with them present. The block-dot verifier's attribute allow-list is widened to
// accept this bounded namespace (RVVDialectWideningOps.cpp isAllowedBlockDotAttr).
constexpr llvm::StringLiteral kAlgorithmAttr = "tcrv_rvv.contraction_algorithm";
constexpr llvm::StringLiteral kReasonAttr = "tcrv_rvv.path_selection_reason";
constexpr llvm::StringLiteral kMaterializationAttr =
    "tcrv_rvv.path_materialization";

// The option-2 stage-C1 OUTPUT CONTRACT carrier (carrier A, the in-IR op attr).
// When the bridge realizes a repack-SELECTED request as the real repack-GEMV op
// it stamps tcrv_rvv.weight_layout_contract = "x16": the DECLARED requirement
// that the weight bytes the emitted kernel reads are in the block_q4_0x16 layout
// (the op's weight_block_stride = 288 contract). The compiler ASSERTS the layout;
// some later layer (the load-time / JIT producer, stages C3-C4) must make it
// true. The repack-GEMV verifier's attr allow-list accepts this bounded name
// (RVVDialectWideningOps.cpp GgmlRepackGemvQ40Q80Op::verify isAllowedAttr).
constexpr llvm::StringLiteral kWeightLayoutContractAttr =
    "tcrv_rvv.weight_layout_contract";

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

// Derives the resource-aware e16m1 strip width (half_lanes) from the guaranteed
// minimum VLEN, the SAME pure rule MaterializeRVVRepackStripWidth uses
// (RVVRepackStripWidthMaterialization.cpp:78): half_lanes = min(vlen/16, 16),
// so 128 -> 8, 256 -> 16. Returns 0 when the evidence guarantees no >= 128
// minimum (an empty -march, or a constrained tier) -- the bridge then has NO
// capability-derived strip width and CANNOT form a well-formed x16 repack op, so
// it leaves the request as the deferred block-dot stub (the honest no-capability
// behavior, e.g. the q4_0-prefill-at-VLEN0 cell).
std::int64_t deriveRepackHalfLanes(std::int64_t vlenBits) {
  if (vlenBits < 128)
    return 0;
  return std::min<std::int64_t>(vlenBits / 16, kWeightInterleave);
}

class RVVLowerQuantContractionPass final
    : public impl::RVVLowerQuantContractionBase<RVVLowerQuantContractionPass> {
public:
  using impl::RVVLowerQuantContractionBase<
      RVVLowerQuantContractionPass>::RVVLowerQuantContractionBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::WalkResult result =
        module.walk([&](tcrvrvv::GgmlQuantContractionOp op) -> mlir::WalkResult {
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
  // reads opponent_vlen_native_floor / block_dot_compute_heavy from the IR, NEVER
  // the quant format LABEL, so a request that deleted its `quant` label but kept
  // the facts selects the IDENTICAL algorithm. Absent facts default to the
  // conservative "no repack advantage" (no VLEN-native opponent floor / not
  // compute-heavy), which routes to the safe block-dot path.
  static pluginrvv::ContractionOpponentFacts
  readOpponentFacts(tcrvrvv::GgmlQuantContractionOp op) {
    pluginrvv::ContractionOpponentFacts facts;
    if (mlir::IntegerAttr floor = op.getOpponentVlenNativeFloorAttr())
      facts.ggmlVlenNativeKernelFloor = floor.getInt();
    facts.blockDotComputeHeavy = op.getBlockDotComputeHeavy().value_or(false);
    return facts;
  }

  static mlir::FailureOr<pluginrvv::MRegime>
  liftMRegime(tcrvrvv::GgmlQuantContractionOp op) {
    if (op.getMRegime() == "decode")
      return pluginrvv::MRegime::Decode;
    if (op.getMRegime() == "prefill")
      return pluginrvv::MRegime::Prefill;
    return mlir::FailureOr<pluginrvv::MRegime>(
        op.emitError() << "stage-B contraction-path selection does not "
                          "recognize m_regime \""
                       << op.getMRegime() << "\"");
  }

  // The IN-COMPILER selection: derive the target VLEN from the pass's -march (the
  // capability authority -- NOT the op's advisory min_vlen attr), lift the
  // committed WHAT axes, and ask the pure fact-driven selector which algorithm to
  // commit to. Both branches emit the byte-identical block-dot body (Option (i)),
  // differentiated only by the inert audit attrs -- so the emitted C is unchanged
  // on every cell and the repack EFFECT is honestly deferred to stage C.
  mlir::LogicalResult lowerOne(tcrvrvv::GgmlQuantContractionOp op) {
    // Read the per-format OPPONENT FACTS from the op's structured attrs -- routing
    // is fact-driven, NOT keyed on the (now optional) quant format label.
    pluginrvv::ContractionOpponentFacts facts = readOpponentFacts(op);
    mlir::FailureOr<pluginrvv::MRegime> mRegime = liftMRegime(op);
    if (mlir::failed(mRegime))
      return mlir::failure();

    // The DERIVED capability fact: the guaranteed minimum VLEN of the configured
    // target, from the SAME plugin-local authority deriveHasZvl128b /
    // MaterializeRVVQ40Schedule consume (default -march "" => 0 => no capability
    // => block-dot, the honest no-capability behavior).
    std::int64_t minVLEN = pluginrvv::deriveMinimumVLEN(march, isaVectorHints);

    pluginrvv::ContractionSelection selection =
        pluginrvv::selectContractionAlgorithm(facts, *mRegime, minVLEN);

    // STAGE C1 (the in-IR BRIDGE): when the selection is Repack AND the target
    // capability supplies a valid e16m1 strip width (minVLEN >= 128 => half_lanes
    // in {8, 16}), REALIZE the request as the real tcrv_rvv.repack_gemv_q4_0_q8_0
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
    bool isRVV0p7 = pluginrvv::deriveRVVVersion(march, isaVectorHints) ==
                    pluginrvv::RVVVersion::RVV0p7;
    if (isRepack && halfLanes != 0) {
      // The m_regime committed WHAT axis chooses the repacked GRANULARITY: the
      // PREFILL (M-amortized) regime realizes the repack as the typed
      // tcrv_rvv.typed_repack_gemm_loop_body REGION (the block-as-lane GEMM that
      // internalizes BOTH the M-row and N-column loops over the interleaved
      // block_q8_0x4 activation), the DECODE regime as the typed
      // tcrv_rvv.typed_repack_gemv_loop_body REGION (the single-activation-row
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
                                                       : nullptr;
      if (*mRegime == pluginrvv::MRegime::Prefill)
        return kquant ? lowerToRepackGemmKQuant(op, selection, halfLanes,
                                                isRVV0p7, *kquant)
               : ternary ? lowerToRepackGemmTernary(op, selection, halfLanes,
                                                    isRVV0p7, *ternary)
                         : lowerToRepackGemm(op, selection, halfLanes, isRVV0p7);
      return kquant ? lowerToRepackGemvKQuant(op, selection, halfLanes, isRVV0p7,
                                              *kquant)
             : ternary ? lowerToRepackGemvTernary(op, selection, halfLanes,
                                                  isRVV0p7, *ternary)
                       : lowerToRepackGemv(op, selection, halfLanes, isRVV0p7);
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
        op.getScaleModel() == kKQuantQ2KScaleModel)
      return op.emitError()
             << "ternary / K-quant quant_contraction requires a repack-affording "
                "capability (a valid e16m1 strip width, minVLEN >= 128); there "
                "is no ternary / K-quant block-dot decline path (the block-dot "
                "identity lowering is q4_0-nibble-only)";

    return lowerToBlockDot(op, selection);
  }

  // STAGE C1 bridge (M-FLAT REPACK, region form): realize a repack-SELECTED,
  // capability-afforded request as the typed tcrv_rvv.typed_repack_gemv_loop_body
  // REGION -- the same construction the flat/super-block front doors do, now for
  // the q4_0 16x1-repacked GEVM. The region is the SOLE representation of the
  // repacked GEVM (the monolithic tcrv_rvv.repack_gemv_q4_0_q8_0 op is retired):
  // the compiler CONSTRUCTS the inner contraction-block loop out of the two
  // decomposed typed bricks -- the per-block lane-wise integer CORE
  // (tcrv_rvv.repack_lane_wise_q4_x_i8_dot, producing numHalves per-strip sumi) +
  // the per-strip dual-fp16 scale FOLD (tcrv_rvv.repack_dual_fp16_scale_fold, one
  // per strip) -- around a per-strip LANE-WISE f32 VECTOR loop-carried
  // accumulator, exactly like the hand-authored region tests, but reachable now
  // from a REAL quant_contraction request (not test-authored).
  //
  // It reconstructs the block_q4_0x16 x16 facts (stride 288, interleave 16,
  // weight quant offset 32, activation stride 34 / offset 2) the verifier pins,
  // derives the resource-aware half_lanes from the capability VLEN, and stamps
  // the DECLARED OUTPUT CONTRACT tcrv_rvv.weight_layout_contract = "x16". The op
  // carries the SAME SSA weight pointer the abstract op carried (the IR cannot
  // tell a plain base from an x16 base; both are const uint8_t *) -- the contract
  // is the bridge's ASSERTION that some later layer (C3-C4) hands it x16 bytes.
  // On RVV0.7.1 the whole-LMUL core anchor (integer_core_lmul = "m1") with its
  // mandatory ONE 16-lane strip (half_lanes = 16, numHalves 1, f32m4 accumulator)
  // is pinned; on RVV1.0 integer_core_lmul is left unset (the fractional mf2
  // default: half_lanes 8 -> two 8-lane strips at VLEN128, f32m2 accumulators).
  //
  // SAFETY (NOT a latent miscompile): the emitted kernel reads x16 weights but
  // the abstract op carries PLAIN weights, so this emit is correct ONLY when the
  // contract is honored. The abstract GgmlQuantContractionOp has NO real producer
  // (it is authored ONLY in lit fixtures; there is no rewriter.create of it in
  // any real pass), so this region is reachable ONLY via lit, NEVER in the real
  // llama.cpp pipeline. The bridge ASSERTS the layout; the system (C3-C4) must
  // make it true. This is NOT yet e2e-correct on plain weights.
  mlir::LogicalResult
  lowerToRepackGemv(tcrvrvv::GgmlQuantContractionOp op,
                    const pluginrvv::ContractionSelection &selection,
                    std::int64_t halfLanes, bool isRVV0p7) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    // On RVV0.7.1 the repack core is the WHOLE-LMUL chain (i8m1 -> i16m2 ->
    // i32m4 -> f32m4): no fractional LMUL, so the 16-block-as-lane group is ONE
    // 16-lane strip (half_lanes 16, integer_core_lmul "m1", f32m4 accumulator).
    // RVV1.0 leaves integer_core_lmul unset (the fractional mf2 default), with the
    // capability-derived strip width (8 @VLEN128 -> two strips, 16 @VLEN256 ->
    // one). numHalves == weight_interleave / half_lanes is the disjoint-strip
    // count, and the region carries ONE per-strip vector accumulator per strip.
    std::int64_t emittedHalfLanes = isRVV0p7 ? 16 : halfLanes;
    bool isM1 = isRVV0p7;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    // The per-strip accumulator + per-strip integer sumi share the ONE LMUL rung:
    // f32m4/i32m4 for the m1 whole-LMUL chain, f32m2/i32m2 for the mf2 fractional
    // chain (the emitter derives l32 the same way).
    llvm::StringRef accLmul = isM1 ? "m4" : "m2";
    mlir::Type f32AccType =
        tcrvrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        tcrvrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        isM1 ? builder.getStringAttr("m1") : mlir::StringAttr();

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
        loc, tcrvrvv::TypedRepackGemvLoopBodyOp::getOperationName());
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
    auto loop = llvm::cast<tcrvrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    // The in-compiler decision audit (the same INERT provenance triple the
    // block-dot branch stamps) PLUS the stage-C1 DECLARED OUTPUT CONTRACT. These
    // are discardable, emitter-inert, dialect-namespaced provenance attrs.
    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
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

    // Integer CORE brick: ONE tcrv_rvv.repack_lane_wise_q4_x_i8_dot producing the
    // numHalves per-strip i32 sumi (a variadic result group). block_index-tied
    // (anti-bypass) and named off the loop-body's OWN weight/activation ABI bases.
    mlir::OperationState coreState(
        loc, tcrvrvv::RepackLaneWiseQ4Q8DotOp::getOperationName());
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
          loc, tcrvrvv::RepackDualFp16ScaleFoldOp::getOperationName());
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
        loc, tcrvrvv::TypedRepackGemvLoopYieldOp::getOperationName());
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
  // tcrv_rvv.typed_repack_gemm_loop_body REGION -- the block-as-lane GEMM sibling
  // of lowerToRepackGemv. The compiler CONSTRUCTS the inner contraction-block loop
  // out of the two decomposed typed GEMM bricks -- the ONE-strip N-column integer
  // CORE (tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot, producing columnsPerPass
  // per-column sumi) + the per-column dual-fp16 scale FOLD
  // (tcrv_rvv.repack_gemm_dual_fp16_scale_fold, one per column) -- around the
  // columnsPerPass per-column LANE-WISE f32 VECTOR loop-carried accumulators, with
  // the block_index + runtime strip_row_offset region entry args the emitter's
  // outer row-group / column-group / runtime-strip / column-pass nest supplies.
  //
  // It reconstructs the block_q4_0x16 weight facts (stride 288, interleave 16,
  // weight quant offset 32) AND the block_q8_0x4 INTERLEAVED activation facts
  // (stride 136, interleave 4, activation quant offset 8) the GEMM verifier pins --
  // NOT the abstract op's PLAIN q8_0 facts (stride 34 / offset 2 the GEVM keeps):
  // the repacked GEMM reads BOTH sides in the repacked layout the DECLARED OUTPUT
  // CONTRACT tcrv_rvv.weight_layout_contract = "x16" asserts. The GEMM internalizes
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
  lowerToRepackGemm(tcrvrvv::GgmlQuantContractionOp op,
                    const pluginrvv::ContractionSelection &selection,
                    std::int64_t halfLanes, bool isRVV0p7) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    // On RVV0.7.1 the repack core is the WHOLE-LMUL chain (no fractional LMUL), so
    // the 16-block-as-lane group is ONE 16-lane strip (half_lanes 16,
    // integer_core_lmul "m1", f32m4 accumulator, columnsPerPass 1). RVV1.0 leaves
    // integer_core_lmul unset (the fractional mf2 default) with the
    // capability-derived strip width and folds all activation_interleave columns in
    // ONE pass (columnsPerPass 4). numHalves == weight_interleave / half_lanes.
    std::int64_t emittedHalfLanes = isRVV0p7 ? 16 : halfLanes;
    bool isM1 = isRVV0p7;
    llvm::StringRef accLmul = isM1 ? "m4" : "m2";
    mlir::Type f32AccType =
        tcrvrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        tcrvrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        isM1 ? builder.getStringAttr("m1") : mlir::StringAttr();
    std::int64_t columnsPerPass = isM1 ? 1 : kActivationInterleave;

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
    auto variant = op->getParentOfType<tcrv::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a tcrv.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, tcrvrvv::RuntimeABIValueOp::getOperationName());
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
        loc, tcrvrvv::TypedRepackGemmLoopBodyOp::getOperationName());
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
    loopState.addRegion();
    auto loop = llvm::cast<tcrvrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    // The in-compiler decision audit (the same INERT provenance triple the GEVM
    // branch stamps) PLUS the stage-C1 DECLARED OUTPUT CONTRACT.
    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
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

    // Integer CORE brick: ONE tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot producing
    // the columnsPerPass per-column i32 sumi (a variadic result group). block_index
    // + strip_row_offset tied (anti-bypass) and named off the loop-body's OWN
    // weight/activation ABI bases.
    mlir::OperationState coreState(
        loc, tcrvrvv::RepackGemmLaneWiseQ4Q8DotOp::getOperationName());
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
          loc, tcrvrvv::RepackGemmDualFp16ScaleFoldOp::getOperationName());
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
        loc, tcrvrvv::TypedRepackGemmLoopYieldOp::getOperationName());
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

  // STAGE C1 bridge (TERNARY REPACK, GEVM region form): the ternary tq2_0 sibling
  // of lowerToRepackGemv. Realize a repack-SELECTED, capability-afforded ternary
  // tq2_0 DECODE request as the typed tcrv_rvv.typed_repack_gemv_loop_body REGION
  // carrying the SINGLE decomposed ternary integer CORE brick
  // (tcrv_rvv.repack_gemv_ternary_core, decode_model "tq2_0") -- the front-door
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
  lowerToRepackGemvTernary(tcrvrvv::GgmlQuantContractionOp op,
                           const pluginrvv::ContractionSelection &selection,
                           std::int64_t halfLanes, bool isRVV0p7,
                           const TernaryDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = isRVV0p7 ? 16 : halfLanes;
    bool isM1 = isRVV0p7;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = isM1 ? "m4" : "m2";
    mlir::Type f32AccType =
        tcrvrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        tcrvrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        isM1 ? builder.getStringAttr("m1") : mlir::StringAttr();

    // The region-carrying loop op: FIVE ABI operands (weight base, activation
    // base, output, element count n, column count nc), NO vl operand and NO result.
    mlir::OperationState loopState(
        loc, tcrvrvv::TypedRepackGemvLoopBodyOp::getOperationName());
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
    auto loop = llvm::cast<tcrvrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
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

    // The SOLE in-region brick: ONE tcrv_rvv.repack_gemv_ternary_core producing the
    // numHalves per-strip i32 sumi (a variadic result group). block_index-tied
    // (anti-bypass) + named off the loop body's OWN weight/activation ABI bases.
    // The typed emitter RE-EMITS the whole byte-exact ternary GEVM body from this
    // brick's identity; the yield passes the carried-in per-strip accumulators
    // through (the ternary fold has no separate FOLD brick, unlike q4_0).
    mlir::OperationState coreState(
        loc, tcrvrvv::RepackGemvTernaryCoreOp::getOperationName());
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
        loc, tcrvrvv::TypedRepackGemvLoopYieldOp::getOperationName());
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
  // ternary tq2_0 PREFILL request as the typed tcrv_rvv.typed_repack_gemm_loop_body
  // REGION carrying the SINGLE decomposed ternary GEMM integer CORE brick
  // (tcrv_rvv.repack_gemm_ternary_core, decode_model "tq2_0") -- the front-door
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
  lowerToRepackGemmTernary(tcrvrvv::GgmlQuantContractionOp op,
                           const pluginrvv::ContractionSelection &selection,
                           std::int64_t halfLanes, bool isRVV0p7,
                           const TernaryDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = isRVV0p7 ? 16 : halfLanes;
    bool isM1 = isRVV0p7;
    llvm::StringRef accLmul = isM1 ? "m4" : "m2";
    mlir::Type f32AccType =
        tcrvrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        tcrvrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        isM1 ? builder.getStringAttr("m1") : mlir::StringAttr();
    std::int64_t columnsPerPass = isM1 ? 1 : kActivationInterleave;

    // Materialize the two runtime ABI values the internalized M-tiling GEMM nest
    // needs but the abstract op does not carry (row count nr, output row stride
    // bs) -- the SAME declared-contract move as the q4_0 repack GEMM.
    auto variant = op->getParentOfType<tcrv::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "ternary repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a tcrv.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, tcrvrvv::RuntimeABIValueOp::getOperationName());
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
        loc, tcrvrvv::TypedRepackGemmLoopBodyOp::getOperationName());
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
    loopState.addRegion();
    auto loop = llvm::cast<tcrvrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
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

    // The SOLE in-region brick: ONE tcrv_rvv.repack_gemm_ternary_core producing the
    // columnsPerPass per-column i32 sumi (a variadic result group). block_index +
    // strip_row_offset tied (anti-bypass) + named off the loop body's OWN
    // weight/activation ABI bases. The typed emitter RE-EMITS the whole byte-exact
    // ternary GEMM body from this brick's identity; the accumulators pass through.
    mlir::OperationState coreState(
        loc, tcrvrvv::RepackGemmTernaryCoreOp::getOperationName());
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
        loc, tcrvrvv::TypedRepackGemmLoopYieldOp::getOperationName());
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
  // DECODE request as the typed tcrv_rvv.typed_repack_gemv_loop_body REGION carrying
  // the SINGLE decomposed K-quant integer CORE brick
  // (tcrv_rvv.repack_gemv_kquant_core, decode_model "q4_K") -- the front-door
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
  lowerToRepackGemvKQuant(tcrvrvv::GgmlQuantContractionOp op,
                          const pluginrvv::ContractionSelection &selection,
                          std::int64_t halfLanes, bool isRVV0p7,
                          const KQuantDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = isRVV0p7 ? 16 : halfLanes;
    bool isM1 = isRVV0p7;
    std::int64_t numHalves = kWeightInterleave / emittedHalfLanes;
    llvm::StringRef accLmul = isM1 ? "m4" : "m2";
    mlir::Type f32AccType =
        tcrvrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        tcrvrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        isM1 ? builder.getStringAttr("m1") : mlir::StringAttr();

    // The region-carrying loop op: FIVE ABI operands (weight base, activation base,
    // output, element count n, column count nc), NO vl operand and NO result.
    mlir::OperationState loopState(
        loc, tcrvrvv::TypedRepackGemvLoopBodyOp::getOperationName());
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
    // + activation int16 bsums) is set ONLY for hasMin, and the q6_K qh high-2-bit
    // SECOND weight plane is set ONLY for the no-min q6_K fold.
    if (facts.hasMin) {
      loopState.addAttribute(
          "weight_dmin_byte_offset",
          builder.getI64IntegerAttr(facts.weightDminByteOffset));
      loopState.addAttribute(
          "activation_bsums_byte_offset",
          builder.getI64IntegerAttr(facts.gevmActivationBsumsByteOffset));
    } else {
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
    loopState.addRegion();
    auto loop = llvm::cast<tcrvrvv::TypedRepackGemvLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
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

    // The SOLE in-region brick: ONE tcrv_rvv.repack_gemv_kquant_core producing the
    // numHalves per-strip i32 sumi. block_index-tied (anti-bypass) + named off the
    // loop body's OWN weight/activation ABI bases. The typed emitter RE-EMITS the
    // whole byte-exact q4_K GEVM body from this brick's identity; the yield passes
    // the carried-in per-strip accumulators through (the K-quant fold has no separate
    // FOLD brick -- it is re-emitted whole, unlike q4_0).
    mlir::OperationState coreState(
        loc, tcrvrvv::RepackGemvKQuantCoreOp::getOperationName());
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
        loc, tcrvrvv::TypedRepackGemvLoopYieldOp::getOperationName());
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
  // tcrv_rvv.typed_repack_gemm_loop_body REGION carrying the SINGLE decomposed
  // K-quant GEMM integer CORE brick (tcrv_rvv.repack_gemm_kquant_core, decode_model
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
  lowerToRepackGemmKQuant(tcrvrvv::GgmlQuantContractionOp op,
                          const pluginrvv::ContractionSelection &selection,
                          std::int64_t halfLanes, bool isRVV0p7,
                          const KQuantDecodeFacts &facts) {
    mlir::OpBuilder builder(op);
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Location loc = op.getLoc();

    std::int64_t emittedHalfLanes = isRVV0p7 ? 16 : halfLanes;
    bool isM1 = isRVV0p7;
    llvm::StringRef accLmul = isM1 ? "m4" : "m2";
    mlir::Type f32AccType =
        tcrvrvv::VectorType::get(ctx, builder.getF32Type(), accLmul);
    mlir::Type i32ResType =
        tcrvrvv::VectorType::get(ctx, builder.getI32Type(), accLmul);
    mlir::StringAttr integerCoreLmul =
        isM1 ? builder.getStringAttr("m1") : mlir::StringAttr();
    std::int64_t columnsPerPass = isM1 ? 1 : kActivationInterleave;

    // Materialize the two runtime ABI values the internalized M-tiling GEMM nest
    // needs but the abstract op does not carry (row count nr, output row stride bs)
    // -- the SAME declared-contract move as the q4_0 / ternary repack GEMM.
    auto variant = op->getParentOfType<tcrv::exec::VariantOp>();
    if (!variant)
      return op.emitError() << "K-quant repack-GEMM region lowering requires the "
                               "quant_contraction to sit inside a tcrv.exec.variant";
    mlir::Value rowCount, outputRowStride;
    {
      mlir::OpBuilder::InsertionGuard abiGuard(builder);
      builder.setInsertionPointToStart(&variant.getBody().front());
      auto makeAbi = [&](llvm::StringRef cName, llvm::StringRef role,
                         llvm::StringRef purpose) -> mlir::Value {
        mlir::OperationState st(
            loc, tcrvrvv::RuntimeABIValueOp::getOperationName());
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
        loc, tcrvrvv::TypedRepackGemmLoopBodyOp::getOperationName());
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
    // q6_K qh SECOND weight plane ONLY for the no-min q6_K fold (mirrors the GEVM).
    if (facts.hasMin) {
      loopState.addAttribute(
          "weight_dmin_byte_offset",
          builder.getI64IntegerAttr(facts.weightDminByteOffset));
      loopState.addAttribute(
          "activation_bsums_byte_offset",
          builder.getI64IntegerAttr(facts.gemmActivationBsumsByteOffset));
    } else {
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
    loopState.addRegion();
    auto loop = llvm::cast<tcrvrvv::TypedRepackGemmLoopBodyOp>(
        builder.create(loopState));

    loop->setAttr(kAlgorithmAttr, builder.getStringAttr("repack"));
    loop->setAttr(kReasonAttr, builder.getStringAttr(selection.reason));
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

    // The SOLE in-region brick: ONE tcrv_rvv.repack_gemm_kquant_core producing the
    // columnsPerPass per-column i32 sumi. block_index + strip_row_offset tied
    // (anti-bypass) + named off the loop body's OWN weight/activation ABI bases. The
    // typed emitter RE-EMITS the whole byte-exact q4_K GEMM body from this brick's
    // identity; the accumulators pass through.
    mlir::OperationState coreState(
        loc, tcrvrvv::RepackGemmKQuantCoreOp::getOperationName());
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
        loc, tcrvrvv::TypedRepackGemmLoopYieldOp::getOperationName());
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

  // Option (i): emit the byte-identical tcrv_rvv.q4_0_q8_0_block_dot body for BOTH
  // the BlockDot-selected (realized) and the Repack-selected (deferred-stage-c)
  // cases, reconstructing today's hand-authored attrs verbatim and DROPPING
  // column_count (nc). The ONLY per-cell difference is the three inert audit attrs
  // recording the in-compiler decision. The emitted C is byte-identical to today
  // on every path; the repack op is materialized by stage C, never here.
  mlir::LogicalResult
  lowerToBlockDot(tcrvrvv::GgmlQuantContractionOp op,
                  const pluginrvv::ContractionSelection &selection) {
    mlir::OpBuilder builder(op);

    // Operands: DROP column_count (nc) -- the block-dot vec_dot delegates M/N to
    // ggml's mul_mat caller and is a bare 4-operand-plus-vl op.
    auto blockDot = builder.create<tcrvrvv::GgmlBlockDotQ40Q80Op>(
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
    // provenance, like tcrv_rvv.q4_0_schedule.*). Repack-selected records that
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

} // namespace tianchenrv::transforms
