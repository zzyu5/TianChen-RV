//===- CodebookGatherPlan.h - Small-codebook dequant decode plan ---------===//
//
// The DequantMechanismPlan abstraction's small 16-entry codebook-gather family
// (iq4_nl / iq4_xs / mxfp4 / nvfp4) DECODE plan.
//
// A CodebookGatherPlan is the transient C++ compile-period object the RVV plugin's
// mechanism-specific typed decision `decideCodebookGather(g,c,omega)` produces from
// stamped geometry/capability facts.  A pre-emission materialization owner stamps the
// selected fields onto the typed decode core; the codebook emitter mechanically reads
// that stamp into this transient plan.  It neither resolves capabilities nor reruns
// selection. After source construction, format names lose ALL dispatch power:
// `mechanism` routes the codebook body, `scaleModel` / `codebookTable` carry the
// structural leaf facts, and `provenanceFormat` is diagnostic ONLY. Source identity
// may select the canonical construction row once, before this boundary.
//
// [K-10] (STRUCTURAL vs PARAMETRIC). The five dequant decode mechanisms
// (NibbleDecode / KQuantScaleMin / CodebookGather / GridLookup / TernaryDecode) differ
// in data-consumption contract and iteration topology, so each is a STRUCTURALLY
// distinct mechanism and gets its OWN plan struct -- they are NOT collapsed into one
// plan the emitter switches on by a `mechanism` discriminant (that is the "structural
// axis as a knob" the GEMM/GEMV rulings forbid). CodebookGatherPlan is the
// CodebookGather mechanism ONLY; its `mechanism` field is a STRUCTURAL TAG that is
// always CodebookGather (asserted, never switched-on to reach a different mechanism's
// body). The FOUR codebook formats all realize the SAME mechanism -- a 16-entry
// register-resident vrgather codebook decode over a block loop -- so `scaleModel`
// (E8M0 / fp16 / UE4M3 / signed-6) and `codebookTable` (FP4 e2m1 / non-linear) are
// PARAMETRIC leaf-shape fields WITHIN the one mechanism, exactly as NibbleDecode's
// per-format bias/min/qh fields parametrize its one shared nibble body. They are NOT a
// sub-mechanism switch. `loadLMUL` / `stripLanes` are the further PARAMETRIC
// (capability) axes. A3 selects the narrowest legal anchor inside the bounded
// declared {mf2,m1,m2} realization set: VLEN64 selects m2, VLEN128 selects m1,
// while VLEN256 with fractional-LMUL support selects mf2. This does not claim global narrowness over
// undeployed VLEN512/1024 profiles; new fractional candidates require owned
// realization + integration evidence. On the wide side, m4 is excluded because
// direct `vsext_vf4` would require an unrepresentable i32m16 destination.
//
// This header lives in Support because WeftRVVDialect (the verifier) and
// WeftConversionRVV (the emitter) both link it, and Conversion -> Dialect is the only
// legal direction between those two -- the SAME siting rule GridDecodePlan.h /
// NibbleDecodePlan.h record. The plan is a transient plugin-internal C++ object; it
// MUST NOT be lifted into any cross-ABI ExtensionPlugin boundary struct ([P-1]
// quasi-ABI). The closed DequantMechanism taxonomy is shared from NibbleDecodePlan.h
// (the first landed mechanism's home); this header REUSES it, it does not redefine it.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_SUPPORT_CODEBOOKGATHERPLAN_H
#define WEFT_SUPPORT_CODEBOOKGATHERPLAN_H

#include "Weft/Support/NibbleDecodePlan.h" // the shared closed DequantMechanism taxonomy

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace weft {

/// The per-format scale-decode + sub-block topology leaf the codebook plan realizes.
/// PARAMETRIC leaf-shape within the ONE CodebookGather mechanism ([K-10]) -- it selects
/// which scale reconstruction + block-loop shape the shared vrgather codebook body runs,
/// NOT a distinct mechanism. Each value is 1:1 with a ggml block type but is a STRUCTURAL
/// fact (the block's scale ABI), NOT the format name used as an execution key ([F-1]).
enum class CodebookScaleModel {
  E8M0SharedExp,     ///< mxfp4: one E8M0 shared-exponent block scale (qk=32, 1 group).
  Fp16Flat,          ///< iq4_nl: one flat fp16 d block scale (qk=32, 1 group).
  UE4M3SubBlock,     ///< nvfp4: four UE4M3 fp8 sub-block scales (qk=64, 4 sub-blocks).
  Signed6SuperBlock, ///< iq4_xs: fp16 d + 8 signed-6 sub scales (qk=256, 8 sub-blocks).
};

inline llvm::StringRef stringifyCodebookScaleModel(
    CodebookScaleModel value) {
  switch (value) {
  case CodebookScaleModel::E8M0SharedExp:
    return "e8m0-shared-exp";
  case CodebookScaleModel::Fp16Flat:
    return "fp16-flat";
  case CodebookScaleModel::UE4M3SubBlock:
    return "ue4m3-sub-block";
  case CodebookScaleModel::Signed6SuperBlock:
    return "signed6-super-block";
  }
  return "";
}

inline std::optional<CodebookScaleModel>
parseCodebookScaleModel(llvm::StringRef value) {
  if (value == "e8m0-shared-exp")
    return CodebookScaleModel::E8M0SharedExp;
  if (value == "fp16-flat")
    return CodebookScaleModel::Fp16Flat;
  if (value == "ue4m3-sub-block")
    return CodebookScaleModel::UE4M3SubBlock;
  if (value == "signed6-super-block")
    return CodebookScaleModel::Signed6SuperBlock;
  return std::nullopt;
}

/// Which 16-entry int8 codebook the gather broadcasts. PARAMETRIC ([K-10]); a
/// STRUCTURAL fact of the block type ([F-1]: table AS DATA, not a dispatch key).
enum class CodebookTable {
  Fp4E2M1,   ///< mxfp4 / nvfp4 share the FP4 e2m1 kvalues (weft_dequant_mxfp4_kvalues).
  NonLinear, ///< iq4_nl / iq4_xs share the 16-entry non-linear kvalues
             ///< (weft_dequant_iq4nl_kvalues).
};

inline llvm::StringRef stringifyCodebookTable(CodebookTable value) {
  switch (value) {
  case CodebookTable::Fp4E2M1:
    return "fp4-e2m1";
  case CodebookTable::NonLinear:
    return "non-linear";
  }
  return "";
}

inline std::optional<CodebookTable>
parseCodebookTable(llvm::StringRef value) {
  if (value == "fp4-e2m1")
    return CodebookTable::Fp4E2M1;
  if (value == "non-linear")
    return CodebookTable::NonLinear;
  return std::nullopt;
}

/// Canonical fixed ggml AoS geometry for one small-codebook scale ABI. This is
/// the single production row consumed by both abstract-format construction and
/// the codebook Formula decision: construction maps source identity to only the
/// scale model, then copies this row; Formula validates typed g and constructs
/// its plan against the same row. These are external layout facts, never knobs.
struct CodebookGatherLayoutFacts {
  CodebookScaleModel scaleModel;
  std::int64_t qk;
  std::int64_t weightBlockStride;
  std::int64_t scaleByteOffset;
  std::int64_t quantByteOffset;
  CodebookTable codebookTable;
  std::int64_t stripLanes;
  std::int64_t codebookEntries;
};

inline std::optional<CodebookGatherLayoutFacts>
lookupCodebookGatherLayoutFacts(CodebookScaleModel scaleModel) {
  switch (scaleModel) {
  case CodebookScaleModel::E8M0SharedExp:
    return CodebookGatherLayoutFacts{scaleModel, 32, 17, 0, 1,
                                     CodebookTable::Fp4E2M1, 16, 16};
  case CodebookScaleModel::Fp16Flat:
    return CodebookGatherLayoutFacts{scaleModel, 32, 18, 0, 2,
                                     CodebookTable::NonLinear, 16, 16};
  case CodebookScaleModel::UE4M3SubBlock:
    return CodebookGatherLayoutFacts{scaleModel, 64, 36, 0, 4,
                                     CodebookTable::Fp4E2M1, 8, 16};
  case CodebookScaleModel::Signed6SuperBlock:
    return CodebookGatherLayoutFacts{scaleModel, 256, 136, 0, 8,
                                     CodebookTable::NonLinear, 16, 16};
  }
  return std::nullopt;
}

/// The legality gate (fail-closed, the GridDecodePlan / NibbleDecodePlan discipline):
/// a codebook plan is realizable ONLY when a legal gather anchor exists for the target
/// (the i8 gather VLMAX must reach codebookEntries and the widening chain must remain
/// representable). The pre-emission decision selects one legal bounded rung; an empty
/// legal set has no plan/stamp and fails closed before emission.
struct CodebookGatherLegality {
  bool isLegal; ///< true iff a covering gather anchor was selected (loadLMUL non-empty).
};

/// The CodebookGather MechanismPlan: the transient re-packaging of the stamped codebook
/// decode facts plus the selected capability-legal gather geometry and provenance.
/// Pure DATA: the pre-emission materializer stamps it, the emitter reads it, and
/// neither route nor dtype authority lives here.
struct CodebookGatherPlan {
  /// STRUCTURAL TAG ([K-10]): always CodebookGather for this plan type. Names the
  /// mechanism family; it is asserted, never switched-on to reach another mechanism.
  DequantMechanism mechanism;

  /// The scale-decode + sub-block topology leaf (PARAMETRIC within CodebookGather).
  CodebookScaleModel scaleModel;
  /// Which 16-entry table the gather broadcasts (PARAMETRIC within CodebookGather).
  CodebookTable codebookTable;

  //--- The re-packaged codebook ABI-shape facts (FIXED ggml AoS layout constants, NOT
  //--- tunable knobs; the emitter used to re-derive these from the format name). ---

  /// The broadcast lookup-table entry count (16 for every small-codebook format); it
  /// drives BOTH the vle8 table load length AND the gather anchor's required VLMAX.
  std::int64_t codebookEntries;
  /// The base packed-nibble qs byte offset (mxfp4 @+1, iq4_nl @+2, nvfp4 @+4,
  /// iq4_xs @+8); the sub-block loops add their per-sub stride to this base.
  std::int64_t codebookByteOffset;
  /// The super-block element count qk (32 mxfp4/iq4_nl, 64 nvfp4, 256 iq4_xs): the
  /// block-count divisor nb = n / qk and the output block stride.
  std::int64_t superBlockElements;
  /// The AoS weight block byte stride (17 mxfp4, 18 iq4_nl, 36 nvfp4, 136 iq4_xs).
  std::int64_t weightBlockStride;

  //--- Selected gather geometry (PARAMETRIC c-driven axes). ---

  /// The selected i8 codebook-gather anchor LMUL in {mf2,m1,m2}. The widened i32/f32 chain is
  /// DERIVED from it in the emitter (deriveWideningChain, the SAME single-source-of-truth
  /// the vec_dot codebook body uses). The emitter fail-CLOSES on any non-legal value, so
  /// this field is a genuine gate, not decoration.
  llvm::StringRef loadLMUL;
  /// The per-strip nibble lane count (== the half-block width the vle8 nibble load and
  /// the vrgather run at): 16 for the mxfp4/iq4_nl single group and iq4_xs 32-lane
  /// sub-block, 8 for the nvfp4 16-lane sub-block. It comes from the canonical typed
  /// layout row selected by scale ABI, so it is a load-bearing witness that the emitter
  /// consumes the plan rather than rebuilding a per-format constant.
  std::int64_t stripLanes;

  /// The gather-anchor legality gate (fail-closed).
  CodebookGatherLegality legality;

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

#endif // WEFT_SUPPORT_CODEBOOKGATHERPLAN_H
