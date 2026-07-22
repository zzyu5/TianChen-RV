#ifndef WEFT_PLUGIN_RVV_RVVDEQUANTFORMULA_H
#define WEFT_PLUGIN_RVV_RVVDEQUANTFORMULA_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"
#include "Weft/Plugin/RVV/RVVFormulaDecision.h"
#include "Weft/Plugin/RVV/RVVSelectedTargetCapability.h"
#include "Weft/Support/GridLookupPlan.h"
#include "Weft/Support/KQuantScaleMinPlan.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Support/TernaryDecodePlan.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace weft::plugin::rvv {

struct DequantNoCapabilityInput {};
struct DequantNoStaticContext {};

struct FlatScaleGeometryFacts {
  std::int64_t qk = 0;
  std::int64_t weightBlockStride = 0;
  std::int64_t scaleByteOffset = 0;
  std::int64_t quantByteOffset = 0;
};

struct FlatScalePlan {
  std::int64_t qk = 0;
  std::int64_t weightBlockStride = 0;
  std::int64_t scaleByteOffset = 0;
  std::int64_t quantByteOffset = 0;
  bool isLegal = false;
  llvm::StringRef reason;
};

inline FlatScalePlan constructInt8ScalePlan(
    const FlatScaleGeometryFacts &g, DequantNoCapabilityInput,
    DequantNoStaticContext) {
  FlatScalePlan plan{g.qk, g.weightBlockStride, g.scaleByteOffset,
                     g.quantByteOffset, false, "Int8Scale/bare-int8"};
  plan.isLegal = g.qk == 32 && g.weightBlockStride > 0 &&
                 g.scaleByteOffset >= 0 && g.quantByteOffset >= 0;
  return plan;
}

inline FlatScalePlan constructBinarySignPlan(
    const FlatScaleGeometryFacts &g, DequantNoCapabilityInput,
    DequantNoStaticContext) {
  FlatScalePlan plan{g.qk, g.weightBlockStride, g.scaleByteOffset,
                     g.quantByteOffset, false,
                     "BinarySign/packed-one-bit"};
  plan.isLegal = g.qk > 0 && g.qk % 8 == 0 &&
                 g.weightBlockStride > 0 && g.scaleByteOffset >= 0 &&
                 g.quantByteOffset >= 0;
  return plan;
}

struct KQuantScaleMinGeometryFacts {
  weft::KQuantScaleModel scaleModel = weft::KQuantScaleModel::Q2K;
  std::int64_t qk = 0;
  std::int64_t weightBlockStride = 0;
  std::int64_t scaleByteOffset = 0;
  std::int64_t quantByteOffset = 0;
};

inline weft::KQuantScaleMinPlan constructKQuantScaleMinPlan(
    const KQuantScaleMinGeometryFacts &g, DequantNoCapabilityInput,
    DequantNoStaticContext) {
  weft::KQuantScaleMinPlan plan{};
  plan.mechanism = weft::DequantMechanism::KQuantScaleMin;
  plan.scaleModel = g.scaleModel;
  plan.superBlockElements = g.qk;
  plan.weightBlockStride = g.weightBlockStride;
  plan.quantByteOffset = g.quantByteOffset;
  plan.scaleBlockByteOffset = g.scaleByteOffset;
  switch (g.scaleModel) {
  case weft::KQuantScaleModel::Q2K:
    plan.hasMin = true;
    plan.subScaleByteOffset = 0;
    plan.hasHighBitPlane = false;
    plan.highBitByteOffset = 0;
    plan.loadLMUL = "m1";
    plan.stripLanes = 16;
    plan.reason = "KQuantScaleMin/q2_k/2bit_min_fold";
    break;
  case weft::KQuantScaleModel::Q3K:
    plan.hasMin = false;
    plan.subScaleByteOffset = 96;
    plan.hasHighBitPlane = true;
    plan.highBitByteOffset = 0;
    plan.loadLMUL = "m1";
    plan.stripLanes = 16;
    plan.reason = "KQuantScaleMin/q3_k/3bit_hmask_single_mul";
    break;
  case weft::KQuantScaleModel::Q4K:
    plan.hasMin = true;
    plan.subScaleByteOffset = 4;
    plan.hasHighBitPlane = false;
    plan.highBitByteOffset = 0;
    plan.loadLMUL = "m2";
    plan.stripLanes = 32;
    plan.reason = "KQuantScaleMin/q4_k/4bit_scale_min_k4";
    break;
  case weft::KQuantScaleModel::Q5K:
    plan.hasMin = true;
    plan.subScaleByteOffset = 4;
    plan.hasHighBitPlane = true;
    plan.highBitByteOffset = 16;
    plan.loadLMUL = "m2";
    plan.stripLanes = 32;
    plan.reason = "KQuantScaleMin/q5_k/5bit_qh_scale_min_k4";
    break;
  case weft::KQuantScaleModel::Q6K:
    plan.hasMin = false;
    plan.subScaleByteOffset = 192;
    plan.hasHighBitPlane = true;
    plan.highBitByteOffset = 128;
    plan.loadLMUL = "m1";
    plan.stripLanes = 16;
    plan.reason = "KQuantScaleMin/q6_k/6bit_ql_qh_single_mul";
    break;
  }
  plan.minByteOffset = plan.scaleBlockByteOffset + 2;
  plan.legality.isLegal = g.qk == 256 && g.weightBlockStride > 0 &&
                          g.scaleByteOffset >= 0 &&
                          g.quantByteOffset >= 0 && !plan.loadLMUL.empty();
  return plan;
}

struct GridLookupGeometryFacts {
  weft::GridDecodeLeaf leaf = weft::GridDecodeLeaf::Iq2Xxs;
  std::optional<std::int64_t> entryLanes;
};

inline weft::GridLookupPlan constructGridLookupPlan(
    const GridLookupGeometryFacts &g, DequantNoCapabilityInput,
    DequantNoStaticContext) {
  weft::GridLookupPlan plan{};
  plan.mechanism = weft::DequantMechanism::GridLookup;
  plan.leaf = g.leaf;
  plan.hasEntryLanes = g.entryLanes.has_value();
  plan.entryLanes = g.entryLanes.value_or(0);
  const bool needsEntryLanes = g.leaf != weft::GridDecodeLeaf::Iq3Xxs;
  plan.legality.isLegal =
      !needsEntryLanes || (g.entryLanes && *g.entryLanes > 0);
  switch (g.leaf) {
  case weft::GridDecodeLeaf::Iq2Xxs:
    plan.reason = "GridLookup/iq2_xxs/grid8_ksigns";
    break;
  case weft::GridDecodeLeaf::Iq2Xs:
    plan.reason = "GridLookup/iq2_xs/grid8_ksigns_dual_ls";
    break;
  case weft::GridDecodeLeaf::Iq2S:
    plan.reason = "GridLookup/iq2_s/grid8_signs256";
    break;
  case weft::GridDecodeLeaf::Iq3Xxs:
    plan.reason = "GridLookup/iq3_xxs/grid4_ksigns";
    break;
  case weft::GridDecodeLeaf::Iq3S:
    plan.reason = "GridLookup/iq3_s/grid4_signs256";
    break;
  }
  return plan;
}

struct TernaryDecodeGeometryFacts {
  weft::TernaryDecodeLeaf leaf = weft::TernaryDecodeLeaf::Tq1_0;
  std::optional<std::int64_t> entryLanes;
};

inline weft::TernaryDecodePlan constructTernaryDecodePlan(
    const TernaryDecodeGeometryFacts &g, DequantNoCapabilityInput,
    DequantNoStaticContext) {
  weft::TernaryDecodePlan plan{};
  plan.mechanism = weft::DequantMechanism::TernaryDecode;
  plan.leaf = g.leaf;
  plan.hasEntryLanes = g.entryLanes.has_value();
  plan.entryLanes = g.entryLanes.value_or(0);
  const bool needsEntryLanes = g.leaf == weft::TernaryDecodeLeaf::Iq1M ||
                               g.leaf == weft::TernaryDecodeLeaf::Iq1S;
  plan.legality.isLegal =
      !needsEntryLanes || (g.entryLanes && *g.entryLanes > 0);
  switch (g.leaf) {
  case weft::TernaryDecodeLeaf::Tq1_0:
    plan.reason = "TernaryDecode/tq1_0/base3_arith";
    break;
  case weft::TernaryDecodeLeaf::Tq2_0:
    plan.reason = "TernaryDecode/tq2_0/2bit_arith";
    break;
  case weft::TernaryDecodeLeaf::Iq1M:
    plan.reason = "TernaryDecode/iq1_m/grid_delta_per_group";
    break;
  case weft::TernaryDecodeLeaf::Iq1S:
    plan.reason = "TernaryDecode/iq1_s/grid_delta_per_sub";
    break;
  }
  return plan;
}

inline llvm::Error makeDequantizeRowFormulaError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("RVV dequantize-row formula rejected construction: ") +
          message,
      llvm::errc::invalid_argument);
}

/// Canonical projection from the selected RVV capability into the only dequant
/// mechanism whose current formula has a decisive c-axis.
inline llvm::Expected<CodebookGatherCapabilityFacts>
projectCodebookGatherCapability(
    const RVVSelectedTargetCapabilityFacts &selected) {
  if (!selected.minimumVLEN)
    return makeDequantizeRowFormulaError(
        llvm::Twine("selected RVV provider @") +
        selected.selectedProviderSymbol + " is missing typed minimum_vlen");

  CodebookGatherCapabilityFacts c;
  c.minimumVLEN = *selected.minimumVLEN;
  if (selected.supportedSEW.empty()) {
    c.supportsSEW8 = true;
    c.supportsSEW32 = true;
  } else {
    c.supportsSEW8 =
        rvvCapabilityPropertyListContains(selected.supportedSEW, "8");
    c.supportsSEW32 =
        rvvCapabilityPropertyListContains(selected.supportedSEW, "32");
  }
  if (selected.supportedLMUL.empty()) {
    c.supportsM1 = true;
    c.supportsM2 = true;
    c.supportsM4 = true;
    c.supportsM8 = true;
    c.supportsMF2 = selected.rvvVersion == "1.0";
  } else {
    c.supportsMF2 =
        rvvCapabilityPropertyListContains(selected.supportedLMUL, "mf2");
    c.supportsM1 =
        rvvCapabilityPropertyListContains(selected.supportedLMUL, "m1");
    c.supportsM2 =
        rvvCapabilityPropertyListContains(selected.supportedLMUL, "m2");
    c.supportsM4 =
        rvvCapabilityPropertyListContains(selected.supportedLMUL, "m4");
    c.supportsM8 =
        rvvCapabilityPropertyListContains(selected.supportedLMUL, "m8");
  }
  return c;
}

/// Resolve c at the formula boundary.  Non-codebook mechanisms return an
/// honest-null optional without reading target state.
inline llvm::Expected<std::optional<CodebookGatherCapabilityFacts>>
projectDequantizeRowCapability(
    mlir::Operation *anchor,
    const ::weft::rvv::DequantizeRowStreamFacts &facts,
    llvm::StringRef context) {
  if (facts.mechanism !=
      ::weft::rvv::DequantizeRowMechanismKind::CodebookGather)
    return std::optional<CodebookGatherCapabilityFacts>();

  weft::exec::VariantOp variant = anchor->getParentOfType<weft::exec::VariantOp>();
  weft::exec::KernelOp kernel =
      variant ? variant->getParentOfType<weft::exec::KernelOp>()
              : weft::exec::KernelOp();
  if (!variant || !kernel)
    return makeDequantizeRowFormulaError(
        "codebook construction requires a selected variant nested in a kernel");
  llvm::Expected<::weft::support::TargetCapabilitySet> capabilities =
      ::weft::support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  llvm::Expected<RVVSelectedTargetCapabilityFacts> selected =
      collectRVVSelectedTargetCapabilityFacts(variant, *capabilities, context);
  if (!selected)
    return selected.takeError();
  llvm::Expected<CodebookGatherCapabilityFacts> projected =
      projectCodebookGatherCapability(*selected);
  if (!projected)
    return projected.takeError();
  return std::optional<CodebookGatherCapabilityFacts>(*projected);
}

/// Construct one complete dequant mechanism plan from typed g and the real c
/// projection.  The result is handed directly to typed IR construction; no
/// optional stamp, formula replay, provider callback or emission-side default is
/// involved.
inline llvm::Expected<::weft::rvv::DequantizeRowConstruction>
constructDequantizeRowFormula(
    const ::weft::rvv::DequantizeRowStreamFacts &facts,
    const std::optional<CodebookGatherCapabilityFacts> &codebookCapability) {
  using Construction = ::weft::rvv::DequantizeRowConstruction;
  using Mechanism = ::weft::rvv::DequantizeRowMechanismKind;
  Construction result{facts, std::monostate{}};

  switch (facts.mechanism) {
  case Mechanism::Int8Scale: {
    FlatScalePlan plan = constructInt8ScalePlan(
        {facts.qk, facts.weightBlockStride, facts.scaleByteOffset,
         facts.quantByteOffset},
        DequantNoCapabilityInput{}, DequantNoStaticContext{});
    if (!plan.isLegal)
      return makeDequantizeRowFormulaError("invalid int8-scale geometry");
    return result;
  }
  case Mechanism::BinarySign: {
    FlatScalePlan plan = constructBinarySignPlan(
        {facts.qk, facts.weightBlockStride, facts.scaleByteOffset,
         facts.quantByteOffset},
        DequantNoCapabilityInput{}, DequantNoStaticContext{});
    if (!plan.isLegal)
      return makeDequantizeRowFormulaError("invalid binary-sign geometry");
    return result;
  }
  case Mechanism::NibbleDecode: {
    if (facts.carrier != ::weft::rvv::NibbleCarrierKind::Nibble4 ||
        !facts.nibbleBias)
      return makeDequantizeRowFormulaError(
          "nibble mechanism lacks typed carrier/bias geometry");
    NibbleDecodeDecision decision = decideNibbleDecode(
        {weft::NibbleCarrier::Nibble4, facts.qk, facts.weightBlockStride,
         facts.scaleByteOffset, facts.quantByteOffset, *facts.nibbleBias,
         facts.minByteOffset, facts.qhByteOffset},
        NibbleDecodeNoCapabilityInput{}, NibbleDecodeNoStaticContext{});
    if (!decision.isLegal || !decision.selectedPlan)
      return makeDequantizeRowFormulaError("invalid nibble geometry");
    result.mechanismPlan = *decision.selectedPlan;
    return result;
  }
  case Mechanism::CodebookGather: {
    if (!facts.codebookScaleModel || !codebookCapability)
      return makeDequantizeRowFormulaError(
          "codebook mechanism lacks typed scale model or capability");
    CodebookGatherDecision decision = decideCodebookGather(
        {*facts.codebookScaleModel, facts.qk, facts.weightBlockStride,
         facts.scaleByteOffset, facts.quantByteOffset},
        *codebookCapability, CodebookGatherNoStaticContext{});
    if (!decision.isLegal() || !decision.selectedPlan)
      return makeDequantizeRowFormulaError(
          llvm::Twine("codebook legal set is empty: ") +
          stringifyCodebookGatherReason(decision.reason));
    result.mechanismPlan = *decision.selectedPlan;
    return result;
  }
  case Mechanism::KQuantScaleMin: {
    if (!facts.kquantScaleModel)
      return makeDequantizeRowFormulaError(
          "K-quant mechanism lacks typed scale model");
    weft::KQuantScaleMinPlan plan = constructKQuantScaleMinPlan(
        {*facts.kquantScaleModel, facts.qk, facts.weightBlockStride,
         facts.scaleByteOffset, facts.quantByteOffset},
        DequantNoCapabilityInput{}, DequantNoStaticContext{});
    if (!plan.legality.isLegal)
      return makeDequantizeRowFormulaError("invalid K-quant geometry");
    result.mechanismPlan = plan;
    return result;
  }
  case Mechanism::GridLookup: {
    if (!facts.gridDecodeLeaf)
      return makeDequantizeRowFormulaError(
          "grid mechanism lacks typed decode leaf");
    std::optional<std::int64_t> entryLanes =
        facts.codebookEntryLanes > 0
            ? std::optional<std::int64_t>(facts.codebookEntryLanes)
            : std::nullopt;
    weft::GridLookupPlan plan = constructGridLookupPlan(
        {*facts.gridDecodeLeaf, entryLanes}, DequantNoCapabilityInput{},
        DequantNoStaticContext{});
    if (!plan.legality.isLegal)
      return makeDequantizeRowFormulaError("invalid grid geometry");
    result.mechanismPlan = plan;
    return result;
  }
  case Mechanism::TernaryDecode: {
    if (!facts.ternaryDecodeLeaf)
      return makeDequantizeRowFormulaError(
          "ternary mechanism lacks typed decode leaf");
    std::optional<std::int64_t> entryLanes =
        facts.codebookEntryLanes > 0
            ? std::optional<std::int64_t>(facts.codebookEntryLanes)
            : std::nullopt;
    weft::TernaryDecodePlan plan = constructTernaryDecodePlan(
        {*facts.ternaryDecodeLeaf, entryLanes}, DequantNoCapabilityInput{},
        DequantNoStaticContext{});
    if (!plan.legality.isLegal)
      return makeDequantizeRowFormulaError("invalid ternary geometry");
    result.mechanismPlan = plan;
    return result;
  }
  }
  return makeDequantizeRowFormulaError("unknown typed mechanism");
}

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVDEQUANTFORMULA_H
