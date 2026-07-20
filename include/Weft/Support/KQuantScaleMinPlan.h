//===- KQuantScaleMinPlan.h - K-quant super-block dequant decode plan -----===//
//
// The DequantMechanismPlan abstraction's THIRD landed mechanism (phase-4, family
// #3): the QK_K=256 K-quant super-block (q2_K / q3_K / q4_K / q5_K / q6_K) scale/min
// DECODE plan.
//
// A KQuantScaleMinPlan is the transient C++ compile-period object the RVV plugin's
// dequant FormulaProvider (kquantScaleMinPlanFromFacts, RVVGearboxSchedule.h -- the
// formula-layer home) produces from the stamped decode_core descriptor facts, and the
// K-quant EMITTERS (emitDequantizeRowQ45KVectorBody / ...Q2K / ...Q3K / ...Q6K) read
// plan.* INSTEAD of scatter-reading the per-format super-block geometry off the format
// name. Format names lose ALL dispatch power: the plan's `mechanism` tag routes the
// K-quant family (the dispatch tests plan.mechanism == KQuantScaleMin, NOT the format
// string), the `scaleModel` field selects the per-super-block decode leaf, and
// `provenanceFormat` is diagnostic ONLY.
//
// [K-10] (STRUCTURAL vs PARAMETRIC). The five dequant decode mechanisms
// (NibbleDecode / KQuantScaleMin / CodebookGather / GridLookup / TernaryDecode) differ
// in data-consumption contract and iteration topology, so each is a STRUCTURALLY
// distinct mechanism and gets its OWN plan struct -- they are NOT collapsed into one
// plan the emitter switches on by a `mechanism` discriminant (that is the "structural
// axis as a knob" the GEMM/GEMV rulings forbid). KQuantScaleMinPlan is the
// KQuantScaleMin mechanism ONLY; its `mechanism` field is a STRUCTURAL TAG that is
// always KQuantScaleMin (asserted, never switched-on to reach a different mechanism's
// body). The FIVE K-quant formats all realize the SAME mechanism -- a QK_K=256
// super-block whose per-sub-block scale (and, for the min-fold formats, min) is
// reconstructed from a packed 6-bit / int8 scale plane and folded onto a bit-unpacked
// integer quant plane. They differ only in the PARAMETRIC leaf shape -- the quant
// bit-width (2 / 3 / 4 / 5 / 6), whether a min is folded, and whether a high-bit plane
// merges -- so `scaleModel` (Q2K / Q3K / Q4K / Q5K / Q6K) is a PARAMETRIC leaf-shape
// field WITHIN the one mechanism, exactly as NibbleDecode's per-format bias/min/qh
// fields and CodebookGather's scaleModel/table parametrize their one shared body. It is
// NOT a sub-mechanism switch. Because the five K-quant bit-unpack topologies are
// genuinely distinct (2-bit vs 3-bit+hmask vs 4/5-bit get_scale_min_k4 vs 6-bit ql+qh)
// they were ALREADY four separate emitter leaves (q4_K/q5_K share one, keyed by a 5th-bit
// flag); scaleModel selects WHICH leaf -- a leaf SELECTION between already-separate
// bodies, sanctioned by [K-10] exactly as NibbleDecode's carrier selects between the
// bare-int8 and nibble4 bodies. `loadLMUL` / `stripLanes` are the further PARAMETRIC
// (capability) axes; phase-4 pins them to the FIXED ggml-ABI super-block geometry
// (reproduce-current, NOT c-driven -- phase-3-kquant makes them f(VLEN)).
//
// This header lives in Support because WeftRVVDialect (the verifier) and
// WeftConversionRVV (the emitter) both link it, and Conversion -> Dialect is the only
// legal direction between those two -- the SAME siting rule GridDecodePlan.h /
// NibbleDecodePlan.h / CodebookGatherPlan.h record. The plan is a transient
// plugin-internal C++ object; it MUST NOT be lifted into any cross-ABI ExtensionPlugin
// boundary struct ([P-1] quasi-ABI). The closed DequantMechanism taxonomy is shared
// from NibbleDecodePlan.h (the first landed mechanism's home); this header REUSES it, it
// does not redefine it.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_SUPPORT_KQUANTSCALEMINPLAN_H
#define WEFT_SUPPORT_KQUANTSCALEMINPLAN_H

#include "Weft/Support/NibbleDecodePlan.h" // the shared closed DequantMechanism taxonomy

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace weft {

/// The per-format K-quant super-block decode leaf the plan realizes. PARAMETRIC
/// leaf-shape within the ONE KQuantScaleMin mechanism ([K-10]) -- it selects which of
/// the (already-separate) bit-unpack + scale/min-fold bodies runs, NOT a distinct
/// mechanism. Each value is 1:1 with a ggml block type but is a STRUCTURAL fact (the
/// block's quant bit-width + scale/min ABI), NOT the format name used as an execution
/// key ([F-1]).
enum class KQuantScaleModel {
  Q2K, ///< block_q2_K: 2-bit quant, packed-6bit scale+min fold (scales@0, d@80, dmin@82).
  Q3K, ///< block_q3_K: 3-bit quant (2-bit + hmask high bit), aux-shuffled 6-bit signed
       ///< scale, SINGLE mul / no min (scales@96, d@108, hmask@0).
  Q4K, ///< block_q4_K: 4-bit quant, get_scale_min_k4 6-bit scale+min fold (scales@4,
       ///< d@0, dmin@2).
  Q5K, ///< block_q5_K: 5-bit quant (4-bit + qh high bit), get_scale_min_k4 scale+min
       ///< fold (scales@4, d@0, dmin@2, qh@16).
  Q6K, ///< block_q6_K: 6-bit quant (4-bit ql + 2-bit qh), signed-int8 per-sub scale,
       ///< SINGLE mul / no min (scales@192, d@208, qh@128).
};

/// The legality gate (fail-closed, the GridDecodePlan / NibbleDecodePlan /
/// CodebookGatherPlan discipline): a K-quant plan is realizable ONLY when a legal load
/// anchor exists for the target. Phase-4 reproduce-current pins the VLEN>=128 anchor
/// (m1 for the 16-lane q2_K/q3_K/q6_K sub-groups, m2 for the 32-lane q4_K/q5_K
/// super-subs), so isLegal is true; a degenerate VLEN/SEW with no covering rung yields
/// an EMPTY loadLMUL and isLegal == false, which the emitter fail-CLOSES on rather than
/// silently emitting a truncated pipeline.
struct KQuantScaleMinLegality {
  bool isLegal; ///< true iff a covering load anchor was selected (loadLMUL non-empty).
};

/// The KQuantScaleMin MechanismPlan: the transient re-packaging of the stamped K-quant
/// decode facts plus the reproduce-current load geometry and provenance. Pure DATA --
/// the FormulaProvider builds it, the emitter reads it, nothing owns route/dtype
/// authority.
struct KQuantScaleMinPlan {
  /// STRUCTURAL TAG ([K-10]): always KQuantScaleMin for this plan type. Names the
  /// mechanism family; it is asserted, never switched-on to reach another mechanism.
  DequantMechanism mechanism;

  /// The per-super-block decode leaf (PARAMETRIC within KQuantScaleMin).
  KQuantScaleModel scaleModel;

  //--- The re-packaged super-block ABI-shape facts (FIXED ggml AoS layout constants, NOT
  //--- tunable knobs; the emitter used to scatter-read these from the format name). The
  //--- primary offsets (qk / stride / scale-block / quant) ride the stamped decode_core
  //--- descriptor; the per-sub scale-plane / min / high-bit offsets are the FormulaProvider's
  //--- per-leaf derivation (the block-type's structural scale ABI, retired from the body). ---

  /// The super-block element count qk (256 for every K-quant format): the block-count
  /// divisor nb = n / qk and the output block stride.
  std::int64_t superBlockElements;
  /// The AoS weight block byte stride (84 q2_K, 110 q3_K, 144 q4_K, 176 q5_K, 210 q6_K).
  std::int64_t weightBlockStride;
  /// The packed-quant base byte offset (q2_K qs@16, q3_K qs@32, q4_K qs@16, q5_K qs@48,
  /// q6_K ql@0); the per-sub loops add their stride to this base.
  std::int64_t quantByteOffset;
  /// The fp16 super-block scale d byte offset (q2_K d@80, q3_K d@108, q4_K/q5_K d@0,
  /// q6_K d@208 -- == the stamped scale_byte_offset). Read at the block base (+ this
  /// offset when non-zero).
  std::int64_t scaleBlockByteOffset;

  /// Whether this leaf folds a per-sub block min (q2_K / q4_K / q5_K carry a dmin +
  /// fused mul-sub; q3_K / q6_K fold with a SINGLE mul and carry no min).
  bool hasMin;
  /// The fp16 super-block min dmin byte offset (== scaleBlockByteOffset + 2; the dmin
  /// sits two bytes after d in every min-fold K-quant). Meaningful iff hasMin.
  std::int64_t minByteOffset;
  /// The packed per-sub scale plane byte offset the scale reconstruction reads (q2_K
  /// scales@0, q3_K aux-6bit@96, q4_K/q5_K get_scale_min_k4@4, q6_K signed-int8@192).
  std::int64_t subScaleByteOffset;

  /// Whether this leaf merges a high-bit plane onto the low quant bits (q3_K hmask 3rd
  /// bit, q5_K qh 5th bit, q6_K qh high 2 bits; q2_K / q4_K have none).
  bool hasHighBitPlane;
  /// The high-bit plane byte offset (q3_K hmask@0, q5_K qh@16, q6_K qh@128). Meaningful
  /// iff hasHighBitPlane.
  std::int64_t highBitByteOffset;

  //--- Reproduce-current load geometry (PARAMETRIC axes; phase-4 pins them to the FIXED
  //--- ggml-ABI super-block geometry, NOT c-driven -- phase-3-kquant derives them from
  //--- VLEN). ---

  /// The i8 quant-plane load LMUL anchor. Phase-4 pins the fixed VLEN>=128 anchor: "m1"
  /// for the 16-lane q2_K/q3_K/q6_K sub-groups, "m2" for the 32-lane q4_K/q5_K
  /// super-subs. The emitter fail-CLOSES on an illegal (empty) value; the widened chain
  /// is DERIVED from it in the emitter (deriveWideningChain), so this field names the
  /// realized anchor. Carried as the phase-3-kquant c-driving seam (phase-3 selects it
  /// f(VLEN)); this cut's leaves pin their fixed shape.
  llvm::StringRef loadLMUL;
  /// The per-strip lane count (== the vector length the vle8 quant load and the unpack
  /// pipeline run at): 16 for the q2_K/q3_K/q6_K 16-element sub-groups, 32 for the
  /// q4_K/q5_K 32-element super-subs. Read for the per-strip vl, so a plan change to
  /// stripLanes CHANGES the emitted per-strip vl (a load-bearing witness that the emit
  /// consumes the PLAN).
  std::int64_t stripLanes;

  /// The load-anchor legality gate (fail-closed).
  KQuantScaleMinLegality legality;

  //--- Provenance (mirror, NOT authority -- I4). Carried for the paper's reason trace
  //--- and diagnostics; NEVER emitted into the executable C (that would break
  //--- byte-exactness) and NEVER a dispatch/address key. ---

  /// A static reason-trace string naming the selected mechanism + scale model.
  llvm::StringRef reason;
  /// The decode_model / ggml format name -- [F-1] declarative: name AS DATA for
  /// diagnostics, never an execution key.
  llvm::StringRef provenanceFormat;
};

} // namespace weft

#endif // WEFT_SUPPORT_KQUANTSCALEMINPLAN_H
