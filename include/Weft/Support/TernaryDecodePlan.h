//===- TernaryDecodePlan.h - Ternary dequant decode plan -----------------===//
//
// The DequantMechanismPlan abstraction's FIFTH (and last) landed mechanism (phase-4,
// family #5): the base-3 / delta ternary family (iq1_s / iq1_m / tq1_0 / tq2_0)
// dequant-row DECODE plan.
//
// A TernaryDecodePlan is the transient C++ compile-period object the RVV plugin's dequant
// FormulaProvider (ternaryDecodePlanFromFacts, RVVGearboxSchedule.h -- the formula-layer
// home) produces from the stamped decode_core descriptor facts, and the dequant-row ternary
// EMITTER (emitDequantizeRowTernaryDecodeBodyShared) reads plan.* INSTEAD of dispatching on
// the decode_model string. Format names lose ALL dispatch power: the plan's `mechanism` tag
// routes the ternary family (the dispatch tests plan.mechanism == TernaryDecode, NOT the
// format string), the `leaf` field selects the per-format owned real-vector body,
// `entryLanes` carries the grid ENTRY byte-width g-axis geometry for the iq1 grid leaves
// (律2), and `provenanceFormat` is diagnostic ONLY.
//
// SPLITTING TERNARY OUT OF THE DOUBLE-HEADED GridDecodePlan (audit finding, [K-10]
// restoration). The GridDecodePlan REGISTRY (GridDecodePlan.cpp) LUMPS the ternary iq1_s /
// iq1_m rows in WITH the grid rows via GridFoldArith::TernaryDelta / DeltaGrid /
// DeltaGridGroupSum -- a [K-10] violation (grid and ternary are structurally distinct
// mechanisms). This plan gives ternary its OWN mechanism tag (TernaryDecode) and its OWN
// struct, distinct from GridLookupPlan, and -- unlike GridLookupPlan -- it does NOT consult
// lookupGridDecodePlan at all: ternary is a SEPARATE mechanism, and tq1_0/tq2_0 are not even
// grid-registry rows (they are pure-arithmetic base-3 / 2-bit ternary super-blocks). So for
// the dequant-row head, grid != ternary is RESTORED: grid consults the grid registry,
// ternary does not, and the two ride two separate plans. The GridDecodePlan registry's own
// lumping (iq1_s/iq1_m's TernaryDelta/DeltaGrid rows) serves the block-dot VERIFIER + repack
// emitters and is left UNTOUCHED (that is the vec_dot/block-dot path, ISSUE-122 deferred).
//
// [K-10] (STRUCTURAL vs PARAMETRIC). The five dequant decode mechanisms
// (NibbleDecode / KQuantScaleMin / CodebookGather / GridLookup / TernaryDecode) differ in
// data-consumption contract and iteration topology, so each is a STRUCTURALLY distinct
// mechanism and gets its OWN plan struct -- they are NOT collapsed into one plan the emitter
// switches on by a `mechanism` discriminant (that is the "structural axis as a knob" the
// GEMM/GEMV rulings forbid). TernaryDecodePlan is the TernaryDecode mechanism ONLY; its
// `mechanism` field is a STRUCTURAL TAG that is always TernaryDecode (asserted, never
// switched-on to reach a different mechanism's body). The FOUR ternary formats realize the
// SAME mechanism -- a {-1, 0, 1} ternary weight decoded (by a signed grid gather for the
// iq1 leaves, by pure base-3 / 2-bit arithmetic for the tq leaves) and scaled -- so `leaf`
// selects among the FOUR ALREADY-SEPARATE owned real-vector bodies (the shared tq1_0/tq2_0
// arithmetic body keyed by which base, the iq1_m per-group-delta body, the iq1_s
// per-sub-block-delta body), a leaf SELECTION sanctioned by [K-10]. `entryLanes` is the
// PARAMETRIC (g-axis) geometry the phase-1 律2 migration retired into the descriptor for the
// iq1 grid leaves; the tq leaves carry no grid entry.
//
// This header lives in Support because WeftRVVDialect (the verifier) and WeftConversionRVV
// (the emitter) both link it, and Conversion -> Dialect is the only legal direction between
// those two -- the SAME siting rule GridDecodePlan.h / NibbleDecodePlan.h /
// CodebookGatherPlan.h / KQuantScaleMinPlan.h / GridLookupPlan.h record. The plan is a
// transient plugin-internal C++ object; it MUST NOT be lifted into any cross-ABI
// ExtensionPlugin boundary struct ([P-1] quasi-ABI). The closed DequantMechanism taxonomy is
// shared from NibbleDecodePlan.h (the first landed mechanism's home); this header REUSES it,
// it does not redefine it.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_SUPPORT_TERNARYDECODEPLAN_H
#define WEFT_SUPPORT_TERNARYDECODEPLAN_H

#include "Weft/Support/NibbleDecodePlan.h" // the shared closed DequantMechanism taxonomy

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace weft {

/// Which of the FOUR ALREADY-SEPARATE ternary owned real-vector bodies the plan realizes.
/// PARAMETRIC leaf SELECTION within the ONE TernaryDecode mechanism ([K-10]) -- each value
/// picks a distinct emitter body, NOT a sub-mechanism switch. Each value is 1:1 with a ggml
/// block type but is a STRUCTURAL fact (the ternary decode topology the body realizes), NOT
/// the format name used as an execution key ([F-1]).
enum class TernaryDecodeLeaf {
  Tq1_0, ///< tq1_0: base-3 packed ternary super-block (pure arithmetic, no grid, no gather).
  Tq2_0, ///< tq2_0: 2-bit ternary super-block (pure arithmetic, no grid, no gather).
  Iq1M,  ///< iq1_m: signed iq1s_grid gather + per-8-group delta, reconstructed packed scale.
  Iq1S,  ///< iq1_s: signed iq1s_grid gather + per-sub-block delta, fp16 d.
};

/// The legality gate (fail-closed, the shared MechanismPlan discipline). Ternary is its own
/// mechanism and is NOT gated on the grid registry (the [K-10] split point): the four
/// ternary formats are all realizable, so isLegal is always true here. The per-leaf entry-
/// lane requirement for the iq1 grid leaves stays a fail-closed gate in the emitter (it is a
/// g-axis descriptor presence check, not a registry membership check).
struct TernaryDecodeLegality {
  bool isLegal; ///< true for every ternary format (not a grid-registry membership gate).
};

/// The TernaryDecode MechanismPlan: the transient re-packaging of the stamped ternary decode
/// facts (the entry-lane g-axis descriptor for the iq1 leaves) plus the per-format leaf
/// selection and provenance. Pure DATA -- the FormulaProvider builds it, the emitter reads
/// it, nothing owns route/dtype authority.
struct TernaryDecodePlan {
  /// STRUCTURAL TAG ([K-10]): always TernaryDecode for this plan type. Names the mechanism
  /// family; it is asserted, never switched-on to reach another mechanism.
  DequantMechanism mechanism;

  /// The per-format owned-body leaf (PARAMETRIC leaf SELECTION within TernaryDecode).
  TernaryDecodeLeaf leaf;

  //--- The re-packaged grid ENTRY byte-width g-axis geometry (律2 descriptor) for the iq1
  //--- grid leaves; the tq leaves carry no grid entry (hasEntryLanes == false). ---

  /// Whether the codebook_entry_lanes descriptor was stamped. The iq1_m / iq1_s owned grid
  /// bodies REQUIRE it (and fail-CLOSE if absent -- no value_or self-supply to the g-axis);
  /// the tq1_0 / tq2_0 pure-arithmetic bodies do not read it.
  bool hasEntryLanes;
  /// The grid ENTRY lane count (8 for the iq1s int64 grid). Meaningful iff hasEntryLanes;
  /// drives the grid-entry pointer stride + the narrow vle8 vl -- the load-bearing witness
  /// that the iq1 emit consumes the PLAN (mutating the descriptor changes the emitted vl).
  std::int64_t entryLanes;

  /// The legality gate (fail-closed; trivially legal for every ternary format).
  TernaryDecodeLegality legality;

  //--- Provenance (mirror, NOT authority -- I4). Carried for the paper's reason trace and
  //--- diagnostics; NEVER emitted into the executable C (that would break byte-exactness)
  //--- and NEVER a dispatch/address key. ---

  /// A static reason-trace string naming the selected mechanism + ternary leaf.
  llvm::StringRef reason;
  /// The decode_model / ggml format name -- [F-1] declarative: name AS DATA for
  /// diagnostics, never an execution key.
  llvm::StringRef provenanceFormat;
};

} // namespace weft

#endif // WEFT_SUPPORT_TERNARYDECODEPLAN_H
