//===- NibbleDecodePlan.h - Flat nibble-family dequant decode plan --------===//
//
// The DequantMechanismPlan abstraction's FIRST landed mechanism: the flat nibble
// family (q8_0 / q4_0 / q4_1 / q5_0 / q5_1) DECODE plan.
//
// A NibbleDecodePlan is the transient C++ compile-period object the RVV plugin's
// typed decision provider (decideNibbleDecode, RVVFormulaDecision.h) produces from
// stamped decode_core mechanism facts. The nibble emitter reads plan.* instead of
// scatter-reading the descriptor tuple.
// Format names lose ALL dispatch power: the plan's `carrier` selects the leaf and the
// `mechanism` tag names the family; `provenanceFormat` is diagnostic ONLY.
//
// [K-10] (STRUCTURAL vs PARAMETRIC). The five dequant decode mechanisms
// (NibbleDecode / KQuantScaleMin / CodebookGather / GridLookup / TernaryDecode) differ
// in data-consumption contract and iteration topology, so each is a STRUCTURALLY
// distinct mechanism and gets its OWN plan struct -- they are NOT collapsed into one
// plan the emitter switches on by a `mechanism` discriminant (that is the "structural
// axis as a knob" the GEMM/GEMV rulings forbid). NibbleDecodePlan is the NibbleDecode
// mechanism ONLY; its `mechanism` field is a STRUCTURAL TAG that is always
// NibbleDecode (asserted, never switched-on to reach a different mechanism's body).
// The DequantMechanism enum below names all five so the taxonomy is closed; the other
// four mechanisms now live in their own plan structs. The ONE in-plan
// choice, `carrier`, selects between two ALREADY-SEPARATE leaves (the bare-int8 q8_0
// leaf vs the shared 4-bit nibble body) -- a leaf SELECTION, sanctioned by [K-10] as
// distinct from a plan-internal mechanism switch. `stripLanes=qk/2` is analytic g
// consumption; `loadLMUL=m1` is the sole realizable structural anchor. The c and
// omega axes are explicitly honest-null; a future c-driven split must add a real
// legal candidate, not restore an ignored VLEN seam.
//
// This header lives in Support because WeftRVVDialect (the verifier) and
// WeftConversionRVV (the emitter) both link it, and Conversion -> Dialect is the only
// legal direction between those two -- the SAME siting rule GridDecodePlan.h records.
// The plan is a transient plugin-internal C++ object; it MUST NOT be lifted into any
// cross-ABI ExtensionPlugin boundary struct ([P-1] quasi-ABI).
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_SUPPORT_NIBBLEDECODEPLAN_H
#define WEFT_SUPPORT_NIBBLEDECODEPLAN_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace weft {

/// The closed dequant-decode mechanism taxonomy ([K-10]). Each value denotes a
/// STRUCTURALLY distinct decode mechanism with its OWN plan struct; this is NOT a
/// discriminant an emitter switches on to reach a different mechanism's body. Only
/// Each mechanism has its own plan struct; this enum closes the shared taxonomy.
enum class DequantMechanism {
  NibbleDecode,   ///< flat nibble family: q8_0 / q4_0 / q4_1 / q5_0 / q5_1.
  KQuantScaleMin, ///< QK_K=256 K-quant super-blocks: q2_K / q3_K / q4_K / q5_K / q6_K.
  CodebookGather, ///< nibble/FP4 codebook gather: iq4_nl / iq4_xs / mxfp4 / nvfp4.
  GridLookup,     ///< IQ grid-table gather: iq2_xxs / iq2_xs / iq2_s / iq3_xxs / iq3_s.
  TernaryDecode,  ///< base-3 / delta ternary: iq1_s / iq1_m / tq1_0 / tq2_0 / q1_0.
};

/// Which of the two ALREADY-SEPARATE nibble-family leaves the plan realizes. [K-10]:
/// a leaf SELECTION between two separate emitter bodies, NOT a plan-internal mechanism
/// switch (the q8_0 leaf and the shared 4-bit nibble body have always been distinct).
enum class NibbleCarrier {
  BareInt8, ///< q8_0: a bare signed-int8 scale (no nibble split / bias / min / qh).
  Nibble4,  ///< q4_0 / q4_1 / q5_0 / q5_1: the shared 4-bit nibble decode body.
};

/// The NibbleDecode MechanismPlan: the transient re-packaging of phase-1's stamped
/// decode 8-tuple plus the reproduce-current geometry and provenance. Pure DATA -- the
/// FormulaProvider builds it, the emitter reads it, nothing owns route/dtype authority.
struct NibbleDecodePlan {
  /// STRUCTURAL TAG ([K-10]): always NibbleDecode for this plan type. Names the
  /// mechanism family; it is asserted, never switched-on to reach another mechanism.
  DequantMechanism mechanism;

  /// The leaf selector (bare-int8 q8_0 vs the shared nibble4 body).
  NibbleCarrier carrier;

  //--- The re-packaged phase-1 decode 8-tuple (FIXED ggml AoS ABI-shape facts, NOT
  //--- tunable knobs; the emitter used to scatter-read these off the decode_core). ---

  std::int64_t weightBlockStride; ///< AoS weight block byte stride.
  std::int64_t scaleByteOffset;   ///< fp16 block scale d byte offset (dOff).
  std::int64_t quantByteOffset;   ///< packed-nibble qs byte offset (qsOff).
  /// The pre-scale bias `sub` subtracted from the nibble (8 for q4_0, 16 for q5_0);
  /// 0 for the min-fold q4_1/q5_1 and the bare-int8 q8_0.
  std::int64_t nibbleBias;
  bool hasMin;                  ///< q4_1/q5_1 carry a block min m (== min_byte_offset present).
  std::int64_t minByteOffset;   ///< fp16 min m byte offset (mOff); meaningful iff hasMin.
  bool hasQh;                   ///< q5_0/q5_1 carry a 5th-bit qh plane (== qh_byte_offset present).
  std::int64_t qhByteOffset;    ///< qh 5th-bit plane byte offset (qhOff); meaningful iff hasQh.

  //--- Current analytic/structural geometry. ---

  /// The base i8 nibble-plane load LMUL anchor. The current legal set contains only
  /// m1; the emitter fails closed on any other value, so this is a real gate.
  llvm::StringRef loadLMUL;
  /// The half-block nibble strip lane count, analytically derived as qk/2 (16 for
  /// QK=32). The emitter reads it for the per-strip vl. This is the load-bearing
  /// witness that the emit consumes the PLAN (a raw descriptor read has no qk/2 field).
  std::int64_t stripLanes;

  //--- Provenance (mirror, NOT authority -- I4). Carried for the paper's reason trace
  //--- and diagnostics; NEVER emitted into the executable C (that would break
  //--- byte-exactness) and NEVER a dispatch/address key. ---

  /// A static reason-trace string naming the selected mechanism + fold shape.
  llvm::StringRef reason;
  /// The decode_model / ggml format name -- [F-1] declarative: name AS DATA for
  /// diagnostics, never an execution key.
  llvm::StringRef provenanceFormat;
};

/// The mechanism family name (for reason traces / diagnostics).
inline llvm::StringRef dequantMechanismName(DequantMechanism m) {
  switch (m) {
  case DequantMechanism::NibbleDecode:
    return "NibbleDecode";
  case DequantMechanism::KQuantScaleMin:
    return "KQuantScaleMin";
  case DequantMechanism::CodebookGather:
    return "CodebookGather";
  case DequantMechanism::GridLookup:
    return "GridLookup";
  case DequantMechanism::TernaryDecode:
    return "TernaryDecode";
  }
  return "";
}

} // namespace weft

#endif // WEFT_SUPPORT_NIBBLEDECODEPLAN_H
