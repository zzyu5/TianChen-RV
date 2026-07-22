//===- RVVScheduleDescriptorRegistry.cpp ----------------------------------===//
//
// Plugin-local executable schedule formulas and their interface-driven
// construction runner.  A descriptor enumerates the legal candidates for one
// kernel family.  Selection is bounded by that set, and construction writes only
// the final typed schedule parameters consumed by the body.  Candidate counts,
// costs, provenance mirrors and post-hoc formula replay are deliberately absent.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVScheduleFormula.h"

#include "Weft/Conversion/EmitC/TunableScheduleOpInterface.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace weft::plugin::rvv {

// q1_0's candidate space is a SINGLE knob (integer_core_lmul) over the two
// whole-LMUL anchors {m2, m1}. There is no multi_block_factor / strip_elision
// (the binary sign decode is ALWAYS one 32-lane sub-block body), so q1_0 cannot
// reuse the shared makeBlockDotScheduleFormula enumeration (it would construct the
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

// iq2_xxs's candidate space mirrors tq2_0/q1_0 EXACTLY: a SINGLE knob
// (integer_core_lmul) over the two whole-LMUL anchors {m2, m1}, no
// multi_block_factor / strip_elision (the grid+sign vluxei16 gather + dot is ALWAYS
// one 32-lane sub-block body). Each anchor is legal iff its i8 strip VLMAX spans
// the 32-element grid-codebook sub-block at the derived minimum VLEN -- the SAME
// getRVVStripVLMAXElements truth source the verifier recomputes legality from. At
// VLEN128 only m2 spans it (e8m1 VLMAX 16 < 32); at VLEN256 m1 also reaches 32
// (and i64m1 VLMAX reaches the 4 grid entries) and the lighter footprint breaks the
// cost tie -- the m1-32-lane shape ggml's shipped _vl256 kernel uses.
static llvm::SmallVector<GenericScheduleCandidate>
enumerateRVVIQ2XXSShapeCandidates(std::int64_t minimumVLEN) {
  constexpr std::int64_t kIQ2XXSSubBlockLen = 32; // the 32-element sub-block.
  llvm::SmallVector<GenericScheduleCandidate> candidates;
  for (auto anchor : {llvm::StringRef("m2"), llvm::StringRef("m1")}) {
    std::int64_t stripVLMAX = getRVVStripVLMAXElements(
        getRVVBlockDotStripLMUL(anchor), getRVVBlockDotStripSEW(anchor),
        minimumVLEN);
    GenericScheduleCandidate candidate;
    candidate.cost = 0;
    candidate.isLegal = stripVLMAX >= kIQ2XXSSubBlockLen;
    candidate.tieBreakVregCost = (anchor == "m1") ? 1 : 2;
    candidate.knobs.push_back(
        {"lmul", "integer_core_lmul", anchor.str(), false});
    candidates.push_back(candidate);
  }
  return candidates;
}

std::optional<RVVScheduleFormulaDescriptor>
lookupRVVScheduleFormula(llvm::StringRef kernelKey) {
  // These families share one candidate schema.  Only the kernel key, resource
  // budget and executable enumeration differ.
  if (kernelKey == "q4_0") {
    RVVScheduleFormulaDescriptor descriptor = makeBlockDotScheduleFormula(
        /*kernelKey=*/"q4_0",
        /*vectorRegisterBudget=*/kRVVQ40ShapeVectorRegisterBudget,
        /*enumerate12=*/enumerateRVVQ40Q80ShapeCandidates);
    descriptor.minimumVLENAttrName = "minimum_vlen";
    return descriptor;
  }

  // q1_0 (the BINARY-sign class): a CUSTOM single-knob descriptor (not the shared
  // block-dot factory, whose factor/elision fields q1_0 does not implement). Its
  // elided-correct anchor MOVES with VLEN (the 32-element sub-block straddles m1's
  // i8 VLMAX boundary between 128/256) exactly like q8_0, so it carries the semantic
  // minimum_vlen attr the verifier recomputes the anchor legality from.
  if (kernelKey == "q1_0") {
    RVVScheduleFormulaDescriptor descriptor;
    descriptor.kernelKey = "q1_0";
    descriptor.resourceBudget = kRVVQ80ShapeVectorRegisterBudget;
    descriptor.minimumVLENAttrName = "minimum_vlen";
    descriptor.requiredKnobKeys = {"lmul"};
    descriptor.enumerate = [](std::int64_t minimumVLEN,
                              std::int64_t /*budget*/) {
      return enumerateRVVQ10ShapeCandidates(minimumVLEN);
    };
    return descriptor;
  }

  // tq2_0 (the 2-bit TERNARY class): a CUSTOM single-knob descriptor mirroring
  // q1_0 (not the shared block-dot factory, whose factor/elision fields the
  // tq2_0 verifier rejects). Its span-correct anchor MOVES with VLEN (the
  // 32-element 2-bit plane straddles m1's i8 VLMAX boundary between 128/256)
  // exactly like q1_0, so it carries the semantic minimum_vlen attr the verifier
  // recomputes the anchor legality from.
  if (kernelKey == "tq2_0") {
    RVVScheduleFormulaDescriptor descriptor;
    descriptor.kernelKey = "tq2_0";
    descriptor.resourceBudget = kRVVQ80ShapeVectorRegisterBudget;
    descriptor.minimumVLENAttrName = "minimum_vlen";
    descriptor.requiredKnobKeys = {"lmul"};
    descriptor.enumerate = [](std::int64_t minimumVLEN,
                              std::int64_t /*budget*/) {
      return enumerateRVVTQ20ShapeCandidates(minimumVLEN);
    };
    return descriptor;
  }

  // iq2_xxs (the GRID-codebook class): a CUSTOM single-knob descriptor mirroring
  // tq2_0 / q1_0 (not the shared block-dot factory, whose factor/elision fields the
  // iq2_xxs verifier rejects). Its span-correct anchor MOVES with VLEN (the 32-lane
  // grid-codebook sub-block = 4 i64 grid entries straddles m1's i8 VLMAX boundary
  // between 128/256: at VLEN256 i64m1 reaches the 4 entries and e8m1 reaches 32),
  // exactly like tq2_0, so it carries the semantic minimum_vlen attr the verifier
  // recomputes the anchor legality from. The m1 refinement at VLEN256 matches ggml's
  // shipped _vl256 m1-32-lane shape (right-sizing the gather ggml leaves at m2).
  if (kernelKey == "iq2_xxs") {
    RVVScheduleFormulaDescriptor descriptor;
    descriptor.kernelKey = "iq2_xxs";
    descriptor.resourceBudget = kRVVQ80ShapeVectorRegisterBudget;
    descriptor.minimumVLENAttrName = "minimum_vlen";
    descriptor.requiredKnobKeys = {"lmul"};
    descriptor.enumerate = [](std::int64_t minimumVLEN,
                              std::int64_t /*budget*/) {
      return enumerateRVVIQ2XXSShapeCandidates(minimumVLEN);
    };
    return descriptor;
  }

  // The live tunable CODEBOOK class uses the {m1, mf2} anchor set and the
  // gather-VLMAX>=16 prune. The mf2
  // anchor is pruned at VLEN128 (gather VLMAX 8 < 16) but admitted at VLEN256 (the
  // ggml `_vl256` shape), so the SEMANTIC minimum_vlen attr the verifier recomputes
  // the gather legality from accompanies the final plan.
  if (kernelKey == "mxfp4") {
    RVVScheduleFormulaDescriptor descriptor =
        makeBlockDotScheduleFormula(
            /*kernelKey=*/"mxfp4",
            /*vectorRegisterBudget=*/kRVVCodebookShapeVectorRegisterBudget,
            /*enumerate12=*/enumerateRVVCodebookShapeCandidates);
    descriptor.minimumVLENAttrName = "minimum_vlen";
    return descriptor;
  }

  if (kernelKey == "q4_0_q8_0_gemm") {
    // The GEMM formula has one final M knob and a bounded resource domain.
    RVVScheduleFormulaDescriptor descriptor;
    descriptor.kernelKey = "q4_0_q8_0_gemm";
    descriptor.resourceBudget = kRVVGemmMaxActivationCols;
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

RVVScheduleFormulaResult evaluateRVVScheduleFormula(
    const RVVScheduleFormulaDescriptor &descriptor,
    const RVVScheduleGeometryFacts &geometry,
    const RVVScheduleCapabilityFacts &capability,
    RVVScheduleNoStaticContext) {
  RVVScheduleFormulaResult result;
  if (geometry.kernelKey != descriptor.kernelKey)
    return result;

  result.candidates =
      descriptor.enumerate(capability.minimumVLEN, capability.resourceBudget);
  result.analyticPrior = selectGenericMinCostCandidate(result.candidates);
  return result;
}

std::optional<GenericScheduleCandidate> selectRVVSchedule(
    const RVVScheduleFormulaDescriptor &descriptor,
    const RVVScheduleFormulaResult &formula,
    const RVVScheduleSelectionInput &selectionInput) {
  if (selectionInput.qualifiedWinnerMemory) {
    std::optional<GenericTuningRecordEntry> entry = lookupGenericTuningRecord(
        *selectionInput.qualifiedWinnerMemory, descriptor.kernelKey,
        selectionInput.targetKey, descriptor.requiredKnobKeys);
    if (entry) {
      std::optional<GenericScheduleCandidate> winner =
          revalidateGenericTuningRecord(formula.candidates, *entry);
      if (winner)
        return winner;
    }
  }
  return formula.analyticPrior;
}

void constructRVVFinalSchedule(
    mlir::Operation *op, const RVVScheduleFormulaDescriptor &descriptor,
    std::int64_t minimumVLEN, const GenericScheduleCandidate &candidate) {
  mlir::Builder builder(op->getContext());
  for (const NamedKnob &knob : candidate.knobs) {
    if (knob.isInteger) {
      std::int64_t value = 0;
      bool invalid = llvm::StringRef(knob.value).getAsInteger(10, value);
      (void)invalid;
      assert(!invalid && "integer schedule candidate must carry an i64 value");
      op->setAttr(knob.attrName, builder.getI64IntegerAttr(value));
      continue;
    }
    op->setAttr(knob.attrName, builder.getStringAttr(knob.value));
  }
  if (!descriptor.minimumVLENAttrName.empty())
    op->setAttr(descriptor.minimumVLENAttrName,
                builder.getI64IntegerAttr(minimumVLEN));
}

static bool matchesFinalSchedule(mlir::Operation *op,
                                 const GenericScheduleCandidate &candidate) {
  for (const NamedKnob &knob : candidate.knobs) {
    mlir::Attribute attr = op->getAttr(knob.attrName);
    if (knob.isInteger) {
      auto integer = llvm::dyn_cast_or_null<mlir::IntegerAttr>(attr);
      std::int64_t expected = 0;
      if (!integer || llvm::StringRef(knob.value).getAsInteger(10, expected) ||
          integer.getInt() != expected)
        return false;
      continue;
    }
    auto string = llvm::dyn_cast_or_null<mlir::StringAttr>(attr);
    if (!string || string.getValue() != knob.value)
      return false;
  }
  return true;
}

static mlir::LogicalResult validateFormulaSchema(
    mlir::Operation *op, llvm::ArrayRef<GenericScheduleCandidate> candidates) {
  if (candidates.empty())
    return op->emitError("schedule formula produced no candidates");
  llvm::ArrayRef<NamedKnob> schema = candidates.front().knobs;
  for (const GenericScheduleCandidate &candidate : candidates) {
    if (candidate.knobs.size() != schema.size())
      return op->emitError("schedule formula produced inconsistent final fields");
    for (auto [actual, expected] : llvm::zip(candidate.knobs, schema)) {
      if (actual.recordKey != expected.recordKey ||
          actual.attrName != expected.attrName ||
          actual.isInteger != expected.isInteger)
        return op->emitError(
            "schedule formula produced inconsistent final field schemas");
    }
  }
  return mlir::success();
}

static std::optional<std::int64_t> resolveScheduleMinimumVLEN(
    mlir::Operation *op, const RVVScheduleFormulaDescriptor &descriptor,
    std::int64_t moduleMinimumVLEN) {
  if (descriptor.minimumVLENAttrName.empty())
    return moduleMinimumVLEN;

  mlir::Attribute attr = op->getAttr(descriptor.minimumVLENAttrName);
  if (!attr)
    return moduleMinimumVLEN;

  auto integer = llvm::dyn_cast<mlir::IntegerAttr>(attr);
  if (!integer || integer.getInt() < 0) {
    op->emitError() << "carries an invalid " << descriptor.minimumVLENAttrName
                    << " capability input; expected a non-negative i64";
    return std::nullopt;
  }

  const std::int64_t bodyMinimumVLEN = integer.getInt();
  if (moduleMinimumVLEN > 0 && bodyMinimumVLEN != moduleMinimumVLEN) {
    op->emitError()
        << "carries a minimum_vlen value inconsistent with the canonical "
           "capability input used to establish schedule legality";
    return std::nullopt;
  }

  // Direct typed-body inputs already carry the capability fact that made their
  // final schedule legal.  When no module/provider or command-line capability
  // exists, use that same fact to validate the formula result instead of
  // manufacturing an unknown-VLEN (0) second world.  A real canonical module
  // capability, when present, remains authoritative and must agree above.
  return bodyMinimumVLEN;
}

mlir::LogicalResult constructRVVSchedulesViaInterface(
    mlir::ModuleOp module, llvm::StringRef march, llvm::StringRef isaVectorHints,
    llvm::StringRef tuneRecord, bool dumpCandidates,
    std::optional<mlir::TypeID> onlyOpType) {
  // Resolve the capability input once.  Candidate generators consume the actual
  // minimum VLEN; the command-line strings are only the existing unprobed
  // fallback used by resolveRVVMinimumVLEN.
  std::int64_t minimumVLEN = resolveRVVMinimumVLEN(module, march, isaVectorHints);
  std::int64_t vectorRegisterBudget =
      resolveRVVVectorRegisterBudget(module);
  std::optional<std::string> recordText =
      loadRVVBlockDotTuningRecord(tuneRecord);

  // Auto-discovery is structural: adding a tunable op does not require another
  // family-name branch in this runner.
  llvm::SmallVector<conversion::emitc::TunableScheduleOpInterface, 4> targets;
  module.walk([&](mlir::Operation *op) {
    if (onlyOpType && op->getName().getTypeID() != *onlyOpType)
      return;
    if (auto iface = llvm::dyn_cast<conversion::emitc::TunableScheduleOpInterface>(op))
      targets.push_back(iface);
  });

  // Candidate dumping is a read-only view of the same construction formula.
  if (dumpCandidates) {
    llvm::DenseSet<llvm::StringRef> dumped;
    for (conversion::emitc::TunableScheduleOpInterface iface : targets) {
      llvm::StringRef kernelKey = iface.getScheduleKernelKey();
      if (!dumped.insert(kernelKey).second)
        continue;
      std::optional<RVVScheduleFormulaDescriptor> descriptor =
          lookupRVVScheduleFormula(kernelKey);
      if (!descriptor) {
        iface->emitError() << "has no registered schedule formula for kernel '"
                           << kernelKey << "'";
        return mlir::failure();
      }
      std::optional<std::int64_t> scheduleMinimumVLEN =
          resolveScheduleMinimumVLEN(iface.getOperation(), *descriptor,
                                     minimumVLEN);
      if (!scheduleMinimumVLEN)
        return mlir::failure();
      RVVScheduleFormulaResult formula = evaluateRVVScheduleFormula(
          *descriptor, RVVScheduleGeometryFacts{kernelKey},
          RVVScheduleCapabilityFacts{
              *scheduleMinimumVLEN,
              std::min(descriptor->resourceBudget, vectorRegisterBudget)},
          RVVScheduleNoStaticContext{});
      if (mlir::failed(
              validateFormulaSchema(iface.getOperation(), formula.candidates)))
        return mlir::failure();
      dumpGenericLegalCandidates(llvm::outs(), descriptor->kernelKey, march,
                                 formula.candidates);
    }
    return mlir::success();
  }

  for (conversion::emitc::TunableScheduleOpInterface iface : targets) {
    llvm::StringRef kernelKey = iface.getScheduleKernelKey();
    std::optional<RVVScheduleFormulaDescriptor> descriptor =
        lookupRVVScheduleFormula(kernelKey);
    if (!descriptor) {
      iface->emitError() << "has no registered schedule formula for kernel '"
                         << kernelKey << "'";
      return mlir::failure();
    }

    std::optional<std::int64_t> scheduleMinimumVLEN =
        resolveScheduleMinimumVLEN(iface.getOperation(), *descriptor,
                                   minimumVLEN);
    if (!scheduleMinimumVLEN)
      return mlir::failure();
    RVVScheduleFormulaResult formula = evaluateRVVScheduleFormula(
        *descriptor, RVVScheduleGeometryFacts{kernelKey},
        RVVScheduleCapabilityFacts{
            *scheduleMinimumVLEN,
            std::min(descriptor->resourceBudget, vectorRegisterBudget)},
        RVVScheduleNoStaticContext{});
    if (mlir::failed(
            validateFormulaSchema(iface.getOperation(), formula.candidates)))
      return mlir::failure();

    llvm::ArrayRef<NamedKnob> finalFields = formula.candidates.front().knobs;
    std::size_t presentFields = 0;
    for (const NamedKnob &knob : finalFields)
      presentFields += iface->hasAttr(knob.attrName);

    if (presentFields != 0 && presentFields != finalFields.size()) {
      iface->emitError()
          << "carries a partial final schedule; either provide every formula "
             "field or leave all fields for construction";
      return mlir::failure();
    }

    if (presentFields == finalFields.size()) {
      bool matchesLegalCandidate = false;
      for (const GenericScheduleCandidate &candidate : formula.candidates) {
        if (candidate.isLegal &&
            matchesFinalSchedule(iface.getOperation(), candidate)) {
          matchesLegalCandidate = true;
          break;
        }
      }
      if (!matchesLegalCandidate) {
        iface->emitError()
            << "carries a complete final schedule that is not a currently "
               "legal formula candidate (minimum_vlen="
            << *scheduleMinimumVLEN << ", resource_budget="
            << std::min(descriptor->resourceBudget, vectorRegisterBudget)
            << ")";
        return mlir::failure();
      }

      if (!descriptor->minimumVLENAttrName.empty()) {
        mlir::Attribute attr = iface->getAttr(descriptor->minimumVLENAttrName);
        if (!attr) {
          mlir::Builder builder(module.getContext());
          iface->setAttr(descriptor->minimumVLENAttrName,
                         builder.getI64IntegerAttr(*scheduleMinimumVLEN));
        }
      }
      continue;
    }

    // The measurement record may choose only an already-generated legal
    // candidate.  Otherwise the analytic/static prior selects within that same
    // set.  Neither path can create a body or bypass legality.
    std::optional<GenericScheduleCandidate> selected = selectRVVSchedule(
        *descriptor, formula, RVVScheduleSelectionInput{march, recordText});
    if (!selected) {
      iface->emitError() << "schedule formula produced no legal candidate";
      return mlir::failure();
    }
    constructRVVFinalSchedule(iface.getOperation(), *descriptor,
                              *scheduleMinimumVLEN, *selected);
  }

  return mlir::success();
}

} // namespace weft::plugin::rvv
