//===- GridLookupPlan.h - IQ grid-table dequant decode plan --------------===//
//
// The DequantMechanismPlan abstraction's FOURTH landed mechanism (phase-4, family
// #4): the IQ grid-table gather family (iq2_xxs / iq2_xs / iq2_s / iq3_xxs / iq3_s)
// dequant-row DECODE plan.
//
// A GridLookupPlan is the transient C++ compile-period object the RVV plugin's dequant
// family-local construction formula (`constructGridLookupPlan`) produces from the
// stamped decode_core descriptor facts, and the dequant-row grid
// EMITTER (emitDequantizeRowIQGridBodyShared) reads plan.* INSTEAD of dispatching on the
// decode_model string. Format names lose ALL dispatch power: the plan's `mechanism` tag
// routes the grid family (the dispatch tests plan.mechanism == GridLookup, NOT the format
// string), the `leaf` field selects the per-format owned real-vector body, `entryLanes`
// carries the grid ENTRY byte-width g-axis geometry (律2), and `provenanceFormat` is
// diagnostic ONLY.
//
// FOLDING THE DOUBLE-HEADED GridDecodePlan REGISTRY (audit finding, dequant-row head).
// GridDecodePlan.cpp is a SEPARATE fail-closed Support registry (lookupGridDecodePlan)
// that the grid core VERIFIER (RVVDialectWideningOps.cpp) and the repack GEMM/GEVM
// BLOCK-DOT emitters (RVVToEmitCBlockQuantLinear.cpp) query DIRECTLY -- it is the closed
// authority for the legal grid decode_model set. The dequant-row grid path did NOT go
// through it at all (it dispatched on the decode_model string and read the entry-lane
// descriptor). This plan single-heads the dequant-row consumer: the typed grid leaf and
// entry geometry determine legality before emission, then the dequant-row emitter reads
// plan.* -- it no longer keys on the format string. The registry's other head (the
// block-dot VERIFIER + repack emitters) still queries lookupGridDecodePlan directly and is
// left UNTOUCHED (that is the vec_dot/block-dot path, ISSUE-122 deferred). So the fold is
// COMPLETE for the dequant-row head and the registry stays the single closed-set authority
// both heads consult.
//
// [K-10] (STRUCTURAL vs PARAMETRIC). The five dequant decode mechanisms
// (NibbleDecode / KQuantScaleMin / CodebookGather / GridLookup / TernaryDecode) differ in
// data-consumption contract and iteration topology, so each is a STRUCTURALLY distinct
// mechanism and gets its OWN plan struct -- they are NOT collapsed into one plan the
// emitter switches on by a `mechanism` discriminant (that is the "structural axis as a
// knob" the GEMM/GEMV rulings forbid). GridLookupPlan is the GridLookup mechanism ONLY;
// its `mechanism` field is a STRUCTURAL TAG that is always GridLookup (asserted, never
// switched-on to reach a different mechanism's body). CRUCIALLY, the ternary siblings
// (iq1_s / iq1_m / tq1_0 / tq2_0) are a DIFFERENT mechanism (TernaryDecode, its own plan)
// and are NOT folded here: the GridDecodePlan REGISTRY lumps iq1_s/iq1_m in with grid via
// GridFoldArith::TernaryDelta/DeltaGrid for the block-dot head, but the dequant-row head
// splits them into TernaryDecodePlan -- restoring [K-10] grid != ternary for this head.
// The FIVE grid formats all realize the SAME mechanism -- a grid-table gather + sign fold
// over an AoS super-block loop -- so `leaf` selects among the FIVE ALREADY-SEPARATE owned
// real-vector bodies (each format fanned out to its own gather-free narrow-per-entry body,
// W4/W5), a leaf SELECTION sanctioned by [K-10] exactly as NibbleDecode's carrier selects
// between the bare-int8 and nibble4 bodies. `entryLanes` is the PARAMETRIC (g-axis)
// geometry the phase-1 律2 migration retired into the descriptor; the emitter reads it for
// the grid-entry pointer stride + the narrow-pipeline vl (reproduce-current).
//
// This header lives in Support because WeftRVVDialect (the verifier) and
// WeftConversionRVV (the emitter) both link it, and Conversion -> Dialect is the only legal
// direction between those two -- the SAME siting rule GridDecodePlan.h / NibbleDecodePlan.h
// / CodebookGatherPlan.h / KQuantScaleMinPlan.h record. The plan is a transient
// plugin-internal C++ object; it MUST NOT be lifted into any cross-ABI ExtensionPlugin
// boundary struct ([P-1] quasi-ABI). The closed DequantMechanism taxonomy is shared from
// NibbleDecodePlan.h (the first landed mechanism's home); this header REUSES it, it does
// not redefine it.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_SUPPORT_GRIDLOOKUPPLAN_H
#define WEFT_SUPPORT_GRIDLOOKUPPLAN_H

#include "Weft/Support/NibbleDecodePlan.h" // the shared closed DequantMechanism taxonomy

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <cstdint>
#include <optional>

namespace weft {

/// Which of the FIVE ALREADY-SEPARATE IQ grid owned real-vector bodies the plan realizes.
/// PARAMETRIC leaf SELECTION within the ONE GridLookup mechanism ([K-10]) -- each value
/// picks a distinct gather-free narrow-per-entry emitter body (W4/W5 de-lottery), NOT a
/// sub-mechanism switch. Each value is 1:1 with a ggml block type but is a STRUCTURAL fact
/// (the grid geometry the body realizes), NOT the format name used as an execution key
/// ([F-1]).
enum class GridDecodeLeaf {
  Iq2Xxs, ///< iq2_xxs: grid-of-8 (int64, 256-entry) + aux-packed ksigns selectors.
  Iq2Xs,  ///< iq2_xs: grid-of-8 (int64, 512-entry) + ksigns selectors, dual ls.
  Iq2S,   ///< iq2_s: grid-of-8 (int64, 1024-entry) + EXPLICIT signs256 bytes.
  Iq3Xxs, ///< iq3_xxs: grid-of-4 (uint32, 256-entry) + ksigns, single-mul store scale.
  Iq3S,   ///< iq3_s: grid-of-4 (uint32, 512-entry) + EXPLICIT signs256 bytes.
};

inline llvm::StringRef stringifyGridDecodeLeaf(GridDecodeLeaf leaf) {
  switch (leaf) {
  case GridDecodeLeaf::Iq2Xxs:
    return "iq2-xxs";
  case GridDecodeLeaf::Iq2Xs:
    return "iq2-xs";
  case GridDecodeLeaf::Iq2S:
    return "iq2-s";
  case GridDecodeLeaf::Iq3Xxs:
    return "iq3-xxs";
  case GridDecodeLeaf::Iq3S:
    return "iq3-s";
  }
  return "";
}

inline std::optional<GridDecodeLeaf> parseGridDecodeLeaf(llvm::StringRef value) {
  return llvm::StringSwitch<std::optional<GridDecodeLeaf>>(value)
      .Case("iq2-xxs", GridDecodeLeaf::Iq2Xxs)
      .Case("iq2-xs", GridDecodeLeaf::Iq2Xs)
      .Case("iq2-s", GridDecodeLeaf::Iq2S)
      .Case("iq3-xxs", GridDecodeLeaf::Iq3Xxs)
      .Case("iq3-s", GridDecodeLeaf::Iq3S)
      .Default(std::nullopt);
}

/// The legality gate (fail-closed, the GridDecodePlan / NibbleDecodePlan /
/// CodebookGatherPlan / KQuantScaleMinPlan discipline). isLegal is set by the
/// FormulaProvider from lookupGridDecodePlan: a grid decode_model that is NOT in the closed
/// GridDecodePlan registry yields isLegal == false, which the emitter fail-CLOSES on --
/// preserving the registry's [D-1] "unknown => reject" for the dequant-row head, now shared
/// with the block-dot head instead of a second closed-set string chain that could drift.
struct GridLookupLegality {
  bool isLegal; ///< true iff the decode_model is a registered GridDecodePlan row.
};

/// The GridLookup MechanismPlan: the transient re-packaging of the stamped grid decode
/// facts (the entry-lane g-axis descriptor) plus the per-format leaf selection and
/// provenance. Pure DATA -- the FormulaProvider builds it (consulting the GridDecodePlan
/// registry for legality), the emitter reads it, nothing owns route/dtype authority.
struct GridLookupPlan {
  /// STRUCTURAL TAG ([K-10]): always GridLookup for this plan type. Names the mechanism
  /// family; it is asserted, never switched-on to reach another mechanism.
  DequantMechanism mechanism;

  /// The per-format owned-body leaf (PARAMETRIC leaf SELECTION within GridLookup).
  GridDecodeLeaf leaf;

  //--- The re-packaged grid ENTRY byte-width g-axis geometry (FIXED per-format ggml grid
  //--- packing fact, retired into the codebook_entry_lanes descriptor by 律2, NOT a knob;
  //--- the owned narrow-per-entry bodies read it for the grid-entry pointer stride + the
  //--- narrow-pipeline vl instead of baking a `const int64_t entryLanes = N`). ---

  /// Whether the codebook_entry_lanes descriptor was stamped by the dequant-stream front
  /// door. iq3_xxs's owned body derives its geometry internally and does NOT require it;
  /// the other four leaves DO -- and fail-CLOSE if it is absent (no value_or self-supply to
  /// the g-axis). Meaningful as a genuine gate, not decoration.
  bool hasEntryLanes;
  /// The grid ENTRY lane count (grid bytes per entry: 4 for iq3_xxs/iq3_s, 8 for the iq2
  /// int64 grids). Meaningful iff hasEntryLanes; drives the grid-entry pointer stride
  /// (idx * entryLanes) AND the narrow unit-stride vle8 vl -- the load-bearing witness that
  /// the emit consumes the PLAN (mutating the descriptor changes the emitted vl literal).
  std::int64_t entryLanes;

  /// The registry legality gate (fail-closed; folds GridDecodePlan through the provider).
  GridLookupLegality legality;

  //--- Provenance (mirror, NOT authority -- I4). Carried for the paper's reason trace and
  //--- diagnostics; NEVER emitted into the executable C (that would break byte-exactness)
  //--- and NEVER a dispatch/address key. ---

  /// A static reason-trace string naming the selected mechanism + grid leaf.
  llvm::StringRef reason;
  /// The decode_model / ggml format name -- [F-1] declarative: name AS DATA for
  /// diagnostics, never an execution key.
  llvm::StringRef provenanceFormat;
};

} // namespace weft

#endif // WEFT_SUPPORT_GRIDLOOKUPPLAN_H
