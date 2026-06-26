//===- RVVScheduleDescriptorRegistry.cpp ----------------------------------===//
//
// The plugin-local SINGLE SOURCE OF TRUTH for the per-kernel schedule
// descriptors, plus the interface-driven materialize runner. A tunable op
// returns its kernel key from TunableScheduleOpInterface::getScheduleKernelKey;
// lookupRVVScheduleDescriptor maps that key to the SAME descriptor data the six
// per-kernel providers used to build inline (makeBlockDotScheduleDescriptor for
// the five block-dot kernels; the GEMM-descriptor data for the M-block). The
// runner walks EVERY op implementing the interface -- via
// dyn_cast<TunableScheduleOpInterface>, NOT a hardcoded op-type list -- and runs
// the ONE shared select+stamp loop (reusing selectGenericSchedule /
// stampRVVSchedule), so the stamped IR is byte-identical to the old typed
// template path.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVScheduleMaterialization.h"

#include "TianChenRV/Conversion/EmitC/TunableScheduleOpInterface.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

// q1_0's candidate space is a SINGLE knob (integer_core_lmul) over the two
// whole-LMUL anchors {m2, m1}. There is no multi_block_factor / strip_elision
// (the binary sign decode is ALWAYS one 32-lane sub-block body), so q1_0 cannot
// reuse the shared makeBlockDotScheduleDescriptor enumeration (it would stamp the
// factor/elision knobs the q1_0 verifier rejects fail-closed). Each anchor is
// legal iff its i8 strip VLMAX spans the 32-element sub-block at the derived
// minimum VLEN -- the SAME getRVVStripVLMAXElements truth source the verifier
// recomputes legality from. At VLEN128 only m2 spans it (e8m1 VLMAX 16 < 32); at
// VLEN256 m1 also reaches 32, and on that exact cost tie the lighter m1 (1 vreg)
// wins over m2 (2 vregs) -- the SAME peak-live resource discriminator q8_0 uses,
// so the pick is attributable to a resource fact, not enumeration order.
static llvm::SmallVector<GenericScheduleCandidate>
enumerateRVVQ10ShapeCandidates(std::int64_t minimumVLEN) {
  constexpr std::int64_t kQ10SubBlockLen = 32; // the 32-element q8 sub-block.
  llvm::SmallVector<GenericScheduleCandidate> candidates;
  // Ordered widest-anchor-first (m2, then m1); the argmin tiebreak (lighter
  // tieBreakVregCost) makes m1 win where both are legal regardless of order.
  for (auto anchor : {llvm::StringRef("m2"), llvm::StringRef("m1")}) {
    std::int64_t stripVLMAX = getRVVStripVLMAXElements(
        getRVVBlockDotStripLMUL(anchor), getRVVBlockDotStripSEW(anchor),
        minimumVLEN);
    GenericScheduleCandidate candidate;
    // Capability-blind structural cost: identical across anchors (one 32-lane
    // vle/vmerge/vwredsum per sub-block either way), so legality + the resource
    // tiebreak fully determine the pick.
    candidate.cost = 0;
    candidate.isLegal = stripVLMAX >= kQ10SubBlockLen;
    candidate.tieBreakVregCost = (anchor == "m1") ? 1 : 2;
    candidate.knobs.push_back(
        {"lmul", "integer_core_lmul", anchor.str(), false});
    candidates.push_back(candidate);
  }
  return candidates;
}

// tq2_0's candidate space mirrors q1_0 EXACTLY: a SINGLE knob (integer_core_lmul)
// over the two whole-LMUL anchors {m2, m1}, no multi_block_factor / strip_elision
// (the fused ternary dot is ALWAYS one 32-lane plane body). Each anchor is legal
// iff its i8 strip VLMAX spans the 32-element 2-bit plane at the derived minimum
// VLEN -- the SAME getRVVStripVLMAXElements truth source the verifier recomputes
// legality from. At VLEN128 only m2 spans it (e8m1 VLMAX 16 < 32); at VLEN256 m1
// also reaches 32 and the lighter footprint breaks the cost tie.
static llvm::SmallVector<GenericScheduleCandidate>
enumerateRVVTQ20ShapeCandidates(std::int64_t minimumVLEN) {
  constexpr std::int64_t kTQ20PlaneLen = 32; // the 32-element 2-bit plane.
  llvm::SmallVector<GenericScheduleCandidate> candidates;
  for (auto anchor : {llvm::StringRef("m2"), llvm::StringRef("m1")}) {
    std::int64_t stripVLMAX = getRVVStripVLMAXElements(
        getRVVBlockDotStripLMUL(anchor), getRVVBlockDotStripSEW(anchor),
        minimumVLEN);
    GenericScheduleCandidate candidate;
    candidate.cost = 0;
    candidate.isLegal = stripVLMAX >= kTQ20PlaneLen;
    candidate.tieBreakVregCost = (anchor == "m1") ? 1 : 2;
    candidate.knobs.push_back(
        {"lmul", "integer_core_lmul", anchor.str(), false});
    candidates.push_back(candidate);
  }
  return candidates;
}

// tq1_0's candidate space mirrors tq2_0/q1_0: a SINGLE knob (integer_core_lmul)
// over {m2, m1}. The knob tunes ONLY the integer dot (section B; the base-3 unpack
// is unchanged), which widens to 32-lane strips over the flat aux8[256]. Each
// anchor is legal iff its i8 strip VLMAX spans the 32-lane strip at the derived
// minimum VLEN -- the SAME getRVVStripVLMAXElements truth source the verifier uses.
static llvm::SmallVector<GenericScheduleCandidate>
enumerateRVVTQ10ShapeCandidates(std::int64_t minimumVLEN) {
  constexpr std::int64_t kTQ10StripLen = 32; // the 32-lane dot strip.
  llvm::SmallVector<GenericScheduleCandidate> candidates;
  for (auto anchor : {llvm::StringRef("m2"), llvm::StringRef("m1")}) {
    std::int64_t stripVLMAX = getRVVStripVLMAXElements(
        getRVVBlockDotStripLMUL(anchor), getRVVBlockDotStripSEW(anchor),
        minimumVLEN);
    GenericScheduleCandidate candidate;
    candidate.cost = 0;
    candidate.isLegal = stripVLMAX >= kTQ10StripLen;
    candidate.tieBreakVregCost = (anchor == "m1") ? 1 : 2;
    candidate.knobs.push_back(
        {"lmul", "integer_core_lmul", anchor.str(), false});
    candidates.push_back(candidate);
  }
  return candidates;
}

std::optional<RVVScheduleMaterializationDescriptor>
lookupRVVScheduleDescriptor(llvm::StringRef kernelKey) {
  // The five block-dot kernels share makeBlockDotScheduleDescriptor; each line
  // is the SAME DATA its provider passed (kernelKey, attr prefix, producer, the
  // two reason literals, the budget, the enumerate fn). q8_0 is the odd one:
  // enumerate12=nullptr, enumerate18=...Q80....
  if (kernelKey == "q4_0")
    return makeBlockDotScheduleDescriptor(
        /*kernelKey=*/"q4_0",
        /*attrPrefix=*/"tcrv_rvv.q4_0_schedule",
        /*producerName=*/"rvv-q4-0-autotuner",
        /*measuredReason=*/
        "measured-fastest legal Q4_0 shape (on-board best-of-N; "
        "tuning-record-backed)",
        /*staticReason=*/
        "min-cost legal Q4_0 shape (capability-blind structural cost; "
        "strip-elision gated on Zvl128b)",
        /*vectorRegisterBudget=*/kRVVQ40ShapeVectorRegisterBudget,
        /*enumerate12=*/enumerateRVVQ40Q80ShapeCandidates);

  if (kernelKey == "q8_0") {
    RVVScheduleMaterializationDescriptor descriptor =
        makeBlockDotScheduleDescriptor(
            /*kernelKey=*/"q8_0",
            /*attrPrefix=*/"tcrv_rvv.q8_0_schedule",
            /*producerName=*/"rvv-q8-0-autotuner",
            /*measuredReason=*/
            "measured-fastest legal Q8_0 shape (on-board best-of-N; "
            "tuning-record-backed)",
            /*staticReason=*/
            "min-cost legal Q8_0 shape (capability-blind structural cost shared "
            "with Q4_0; strip-elision admitted at the narrowest anchor whose "
            "VLMAX spans the block at the derived minimum VLEN)",
            /*vectorRegisterBudget=*/kRVVQ80ShapeVectorRegisterBudget,
            /*enumerate12=*/nullptr,
            /*enumerate18=*/enumerateRVVQ80Q80ShapeCandidates);
    // q8_0 is the ONE block-dot kernel whose elided-correct anchor moves with VLEN
    // (its 32-element block straddles m1's VLMAX boundary between 128/256). Stamp
    // the SEMANTIC minimum_vlen attr its verifier recomputes legality from.
    descriptor.minimumVLENAttrName = "minimum_vlen";
    return descriptor;
  }

  if (kernelKey == "q4_1")
    return makeBlockDotScheduleDescriptor(
        /*kernelKey=*/"q4_1",
        /*attrPrefix=*/"tcrv_rvv.q4_1_schedule",
        /*producerName=*/"rvv-q4-1-autotuner",
        /*measuredReason=*/
        "measured-fastest legal Q4_1 shape (on-board best-of-N; "
        "tuning-record-backed; FIXES the static mis-pick)",
        /*staticReason=*/
        "min-cost legal Q4_1 shape (capability-blind structural cost; "
        "strip-elision gated on Zvl128b)",
        /*vectorRegisterBudget=*/kRVVQ41ShapeVectorRegisterBudget,
        /*enumerate12=*/enumerateRVVQ41Q81ShapeCandidates);

  if (kernelKey == "q5_0")
    return makeBlockDotScheduleDescriptor(
        /*kernelKey=*/"q5_0",
        /*attrPrefix=*/"tcrv_rvv.q5_0_schedule",
        /*producerName=*/"rvv-q5-0-autotuner",
        /*measuredReason=*/
        "measured-fastest legal Q5_0 shape (on-board best-of-N; "
        "tuning-record-backed)",
        /*staticReason=*/
        "min-cost legal Q5_0 shape (capability-blind structural cost; "
        "strip-elision gated on Zvl128b)",
        /*vectorRegisterBudget=*/kRVVQ50ShapeVectorRegisterBudget,
        /*enumerate12=*/enumerateRVVQ50Q80ShapeCandidates);

  if (kernelKey == "q5_1")
    return makeBlockDotScheduleDescriptor(
        /*kernelKey=*/"q5_1",
        /*attrPrefix=*/"tcrv_rvv.q5_1_schedule",
        /*producerName=*/"rvv-q5-1-autotuner",
        /*measuredReason=*/
        "measured-fastest legal Q5_1 shape (on-board best-of-N; "
        "tuning-record-backed)",
        /*staticReason=*/
        "min-cost legal Q5_1 shape (capability-blind structural cost; "
        "strip-elision gated on Zvl128b)",
        /*vectorRegisterBudget=*/kRVVQ51ShapeVectorRegisterBudget,
        /*enumerate12=*/enumerateRVVQ51Q81ShapeCandidates);

  // q1_0 (the BINARY-sign class): a CUSTOM single-knob descriptor (not the shared
  // block-dot factory, whose factor/elision stamps the q1_0 verifier rejects). Its
  // elided-correct anchor MOVES with VLEN (the 32-element sub-block straddles m1's
  // i8 VLMAX boundary between 128/256) exactly like q8_0, so it stamps the SEMANTIC
  // minimum_vlen attr the verifier recomputes the anchor legality from.
  if (kernelKey == "q1_0") {
    RVVScheduleMaterializationDescriptor descriptor;
    descriptor.kernelKey = "q1_0";
    descriptor.attrPrefix = "tcrv_rvv.q1_0_schedule";
    descriptor.producerName = "rvv-q1-0-autotuner";
    descriptor.measuredReason =
        "measured-fastest legal Q1_0 shape (on-board best-of-N; "
        "tuning-record-backed)";
    descriptor.staticReason =
        "min-cost legal Q1_0 anchor (capability-blind structural cost; the "
        "binary sign decode's single 32-lane sub-block body is admitted only at "
        "the whole-LMUL anchor whose i8 VLMAX spans the 32-element sub-block at "
        "the derived minimum VLEN -- m2 at VLEN128, the lighter m1 at VLEN256)";
    descriptor.budgetAttrName = "vector_register_budget";
    descriptor.budgetValue = kRVVQ80ShapeVectorRegisterBudget;
    descriptor.stampPeakLiveVregs = false; // single-knob: no peak-live footprint.
    descriptor.stampHasZvl128b = true;
    descriptor.minimumVLENAttrName = "minimum_vlen";
    descriptor.requiredKnobKeys = {"lmul"};
    descriptor.enumerate = [](std::int64_t minimumVLEN,
                              std::int64_t /*budget*/) {
      return enumerateRVVQ10ShapeCandidates(minimumVLEN);
    };
    return descriptor;
  }

  // tq2_0 (the 2-bit TERNARY class): a CUSTOM single-knob descriptor mirroring
  // q1_0 (not the shared block-dot factory, whose factor/elision stamps the
  // tq2_0 verifier rejects). Its span-correct anchor MOVES with VLEN (the
  // 32-element 2-bit plane straddles m1's i8 VLMAX boundary between 128/256)
  // exactly like q1_0, so it stamps the SEMANTIC minimum_vlen attr the verifier
  // recomputes the anchor legality from.
  if (kernelKey == "tq2_0") {
    RVVScheduleMaterializationDescriptor descriptor;
    descriptor.kernelKey = "tq2_0";
    descriptor.attrPrefix = "tcrv_rvv.tq2_0_schedule";
    descriptor.producerName = "rvv-tq2-0-autotuner";
    descriptor.measuredReason =
        "measured-fastest legal TQ2_0 shape (on-board best-of-N; "
        "tuning-record-backed)";
    descriptor.staticReason =
        "min-cost legal TQ2_0 anchor (capability-blind structural cost; the "
        "fused ternary dot's single 32-lane plane body is admitted only at the "
        "whole-LMUL anchor whose i8 VLMAX spans the 32-element 2-bit plane at "
        "the derived minimum VLEN -- m2 at VLEN128, the lighter m1 at VLEN256)";
    descriptor.budgetAttrName = "vector_register_budget";
    descriptor.budgetValue = kRVVQ80ShapeVectorRegisterBudget;
    descriptor.stampPeakLiveVregs = false; // single-knob: no peak-live footprint.
    descriptor.stampHasZvl128b = true;
    descriptor.minimumVLENAttrName = "minimum_vlen";
    descriptor.requiredKnobKeys = {"lmul"};
    descriptor.enumerate = [](std::int64_t minimumVLEN,
                              std::int64_t /*budget*/) {
      return enumerateRVVTQ20ShapeCandidates(minimumVLEN);
    };
    return descriptor;
  }

  // tq1_0 (the BASE-3 TERNARY class): a CUSTOM single-knob descriptor mirroring
  // tq2_0/q1_0. The knob tunes ONLY the integer dot (section B); the base-3 unpack
  // (section A) + the aux8 scratch are unchanged. The dot widens to 32-lane strips
  // whose span-correct anchor MOVES with VLEN (the 32-lane strip straddles m1's i8
  // VLMAX boundary between 128/256), so it stamps the SEMANTIC minimum_vlen attr
  // the verifier recomputes the anchor legality from.
  if (kernelKey == "tq1_0") {
    RVVScheduleMaterializationDescriptor descriptor;
    descriptor.kernelKey = "tq1_0";
    descriptor.attrPrefix = "tcrv_rvv.tq1_0_schedule";
    descriptor.producerName = "rvv-tq1-0-autotuner";
    descriptor.measuredReason =
        "measured-fastest legal TQ1_0 shape (on-board best-of-N; "
        "tuning-record-backed)";
    descriptor.staticReason =
        "min-cost legal TQ1_0 dot anchor (capability-blind structural cost; the "
        "widened integer dot's 32-lane strip is admitted only at the whole-LMUL "
        "anchor whose i8 VLMAX spans 32 at the derived minimum VLEN -- m2 at "
        "VLEN128, the lighter m1 at VLEN256; tunes the dot only, the base-3 "
        "unpack is fixed)";
    descriptor.budgetAttrName = "vector_register_budget";
    descriptor.budgetValue = kRVVQ80ShapeVectorRegisterBudget;
    descriptor.stampPeakLiveVregs = false; // single-knob: no peak-live footprint.
    descriptor.stampHasZvl128b = true;
    descriptor.minimumVLENAttrName = "minimum_vlen";
    descriptor.requiredKnobKeys = {"lmul"};
    descriptor.enumerate = [](std::int64_t minimumVLEN,
                              std::int64_t /*budget*/) {
      return enumerateRVVTQ10ShapeCandidates(minimumVLEN);
    };
    return descriptor;
  }

  // The CODEBOOK class (FP4 family): iq4_nl + mxfp4. They share the codebook
  // enumeration verbatim (the {m1, mf2} anchor set + the gather-VLMAX>=16 prune);
  // they differ ONLY in block strides + codebook values (DATA the gearbox never
  // reasons over). Like q8_0 their elided-correct anchor MOVES with VLEN -- the mf2
  // anchor is pruned at VLEN128 (gather VLMAX 8 < 16) but admitted at VLEN256 (the
  // ggml `_vl256` shape), so the SEMANTIC minimum_vlen attr the verifier recomputes
  // the gather legality from is stamped (single source of truth).
  if (kernelKey == "iq4_nl") {
    RVVScheduleMaterializationDescriptor descriptor =
        makeBlockDotScheduleDescriptor(
            /*kernelKey=*/"iq4_nl",
            /*attrPrefix=*/"tcrv_rvv.iq4_nl_schedule",
            /*producerName=*/"rvv-iq4-nl-autotuner",
            /*measuredReason=*/
            "measured-fastest legal IQ4_NL shape (on-board best-of-N; "
            "tuning-record-backed)",
            /*staticReason=*/
            "min-cost legal IQ4_NL codebook shape (capability-blind structural "
            "cost shared with Q4_0; the codebook i8 gather anchor admitted only "
            "where its VLMAX spans the 16-entry table at the derived minimum VLEN)",
            /*vectorRegisterBudget=*/kRVVCodebookShapeVectorRegisterBudget,
            /*enumerate12=*/enumerateRVVCodebookShapeCandidates);
    descriptor.minimumVLENAttrName = "minimum_vlen";
    return descriptor;
  }

  if (kernelKey == "mxfp4") {
    RVVScheduleMaterializationDescriptor descriptor =
        makeBlockDotScheduleDescriptor(
            /*kernelKey=*/"mxfp4",
            /*attrPrefix=*/"tcrv_rvv.mxfp4_schedule",
            /*producerName=*/"rvv-mxfp4-autotuner",
            /*measuredReason=*/
            "measured-fastest legal MXFP4 shape (on-board best-of-N; "
            "tuning-record-backed)",
            /*staticReason=*/
            "min-cost legal MXFP4 codebook shape (capability-blind structural "
            "cost shared with Q4_0; the codebook i8 gather anchor admitted only "
            "where its VLMAX spans the 16-entry table at the derived minimum VLEN)",
            /*vectorRegisterBudget=*/kRVVCodebookShapeVectorRegisterBudget,
            /*enumerate12=*/enumerateRVVCodebookShapeCandidates);
    descriptor.minimumVLENAttrName = "minimum_vlen";
    return descriptor;
  }

  if (kernelKey == "q4_0_q8_0_gemm") {
    // The GEMM descriptor: a SINGLE M knob (activation_cols), the vreg-ceiling
    // resource bound, and the GEMM-specific audit flavor. stampHasZvl128b is left
    // at the struct default (true); only stampPeakLiveVregs is overridden.
    RVVScheduleMaterializationDescriptor descriptor;
    descriptor.kernelKey = "q4_0_q8_0_gemm";
    descriptor.attrPrefix = "tcrv_rvv.q4_0_gemm_schedule";
    descriptor.producerName = "rvv-gemm-m-autotuner";
    descriptor.measuredReason =
        "measured-fastest GEMM M-block (on-board best-of-N; tuning-record-backed; "
        "the noisy cache/vreg M optimum is selected by measurement, overturning "
        "the naive static default)";
    descriptor.staticReason =
        "static default GEMM M-block (the safe cache-friendly tile; no tuning "
        "record -> the measurement-tuned M is not available)";
    descriptor.budgetAttrName = "vreg_ceiling";
    descriptor.budgetValue = kRVVGemmMaxActivationCols;
    descriptor.stampPeakLiveVregs = false; // GEMM has no peak-live-vreg stamp.
    descriptor.requiredKnobKeys = {"activation_cols"};
    descriptor.enumerate = [](std::int64_t /*minimumVLEN*/,
                              std::int64_t vregCeiling) {
      llvm::SmallVector<GenericScheduleCandidate> generic;
      for (const RVVGemmMCandidate &candidate :
           enumerateRVVGemmMCandidates(vregCeiling))
        generic.push_back(toGenericGemmCandidate(candidate));
      return generic;
    };
    return descriptor;
  }

  return std::nullopt;
}

void runRVVScheduleMaterializationViaInterface(
    mlir::ModuleOp module, llvm::StringRef march, llvm::StringRef isaVectorHints,
    llvm::StringRef tuneRecord, bool dumpCandidates,
    std::optional<mlir::TypeID> onlyOpType) {
  mlir::MLIRContext *ctx = module.getContext();

  // (1) Derive the capability VLEN facts ONCE. The block-dot enumerations now
  // reason over the REAL minimum VLEN bits (deriveMinimumVLEN: 0/128/256/512...),
  // which drives the per-anchor reduction count AND the elided-cover legality via
  // VLMAX -- replacing the old 1-bit Zvl128b boolean as the selection input. The
  // boolean is still derived (== minimumVLEN >= 128) for the audit stamp's
  // `has_zvl128b` attr (the cost FORMULA itself remains capability-blind). For
  // GEMM both are audit-only.
  std::int64_t minimumVLEN = deriveMinimumVLEN(march, isaVectorHints);
  bool hasZvl128b = deriveHasZvl128b(march, isaVectorHints);

  // (4) Load the optional measurement record ONCE (absent/unreadable => static
  // fallback; the record is an advisory cache, never a correctness authority).
  std::optional<std::string> recordText =
      loadRVVBlockDotTuningRecord(tuneRecord);

  // Collect the tunable targets first (the walk mutates the IR by stamping).
  // Auto-discovery: an op is a target iff it implements TunableScheduleOpInterface
  // -- dispatched via dyn_cast, NOT a hardcoded op-type list. When onlyOpType is
  // set the walk is filtered to that op type (how each preserved per-kernel pass
  // restricts itself to its own op).
  llvm::SmallVector<conversion::emitc::TunableScheduleOpInterface, 4> targets;
  module.walk([&](mlir::Operation *op) {
    if (onlyOpType && op->getName().getTypeID() != *onlyOpType)
      return;
    if (auto iface = llvm::dyn_cast<conversion::emitc::TunableScheduleOpInterface>(op))
      targets.push_back(iface);
  });

  // (3) dump-candidates: emit the legal set (single source of truth for the
  // offline tune driver) and exit without stamping. The legacy per-pass dump
  // emitted ONCE per kernel from the statically-built descriptor; here we dedup
  // by kernel key and emit once per distinct discovered kernel in
  // first-encounter order, then exit (no stamping).
  if (dumpCandidates) {
    llvm::DenseSet<llvm::StringRef> dumped;
    for (conversion::emitc::TunableScheduleOpInterface iface : targets) {
      llvm::StringRef kernelKey = iface.getScheduleKernelKey();
      if (!dumped.insert(kernelKey).second)
        continue;
      std::optional<RVVScheduleMaterializationDescriptor> descriptor =
          lookupRVVScheduleDescriptor(kernelKey);
      if (!descriptor)
        continue;
      llvm::SmallVector<GenericScheduleCandidate> candidates =
          descriptor->enumerate(minimumVLEN, descriptor->budgetValue);
      dumpGenericLegalCandidates(llvm::outs(), descriptor->kernelKey, march,
                                 candidates);
    }
    return;
  }

  for (conversion::emitc::TunableScheduleOpInterface iface : targets) {
    // (5) No-clobber guard: a hand-authored shape pins the op; leave untouched.
    if (iface.isSchedulePinned())
      continue;

    // The kernel key drives the descriptor lookup; an unregistered key is
    // skipped fail-closed.
    llvm::StringRef kernelKey = iface.getScheduleKernelKey();
    std::optional<RVVScheduleMaterializationDescriptor> descriptor =
        lookupRVVScheduleDescriptor(kernelKey);
    if (!descriptor)
      continue;

    // (2) Enumerate + prune the generic candidate space (VLEN-derived for the
    // block-dot kernels: reduction count + elided legality from VLMAX).
    llvm::SmallVector<GenericScheduleCandidate> candidates =
        descriptor->enumerate(minimumVLEN, descriptor->budgetValue);

    // (6) Select: measured-best (still-legal record) else the static argmin.
    // Fail-closed (I7): nullopt only when every candidate was pruned.
    std::optional<GenericScheduleSelection> selected = selectGenericSchedule(
        candidates, recordText, descriptor->kernelKey, march,
        descriptor->requiredKnobKeys);
    if (!selected)
      continue;

    // (7) Stamp the chosen knobs + the provenance audit trail + (q8_0) the
    // semantic minimum_vlen legality input.
    stampRVVSchedule(ctx, iface.getOperation(), *descriptor, hasZvl128b,
                     minimumVLEN, static_cast<std::int64_t>(candidates.size()),
                     *selected);
  }
}

} // namespace tianchenrv::plugin::rvv
