#include "Weft/Conversion/RVV/RVVCodebookGatherPlanMaterialization.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVFormulaDecision.h"
#include "Weft/Plugin/RVV/RVVSelectedTargetCapability.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Support/CodebookGatherPlan.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace weft::conversion::rvv {
namespace {

namespace pluginrvv = ::weft::plugin::rvv;
namespace weftrvv = ::weft::rvv;

bool isCodebookFacts(const weftrvv::DequantizeRowStreamFacts &facts) {
  return facts.codebookScaleModel.has_value();
}

llvm::Expected<pluginrvv::CodebookGatherCapabilityFacts>
projectCodebookGatherCapability(
    const pluginrvv::RVVSelectedTargetCapabilityFacts &selected) {
  if (!selected.minimumVLEN)
    return pluginrvv::makeRVVSelectedTargetCapabilityError(
        llvm::Twine("codebook dequant formula materialization selected RVV ") +
        "provider @" + selected.selectedProviderSymbol +
        " is missing typed minimum_vlen");

  pluginrvv::CodebookGatherCapabilityFacts c;
  c.minimumVLEN = *selected.minimumVLEN;

  // supported_* are optional restriction lists on an already-selected base-V
  // provider, not independent capability ids. Absence means the base RVV1.0
  // whole-rung/SEW contract; an explicit list is authoritative. Fractional mf2
  // still requires positive RVV1.0 evidence (or an explicit mf2 token).
  if (selected.supportedSEW.empty()) {
    c.supportsSEW8 = true;
    c.supportsSEW32 = true;
  } else {
    c.supportsSEW8 = pluginrvv::rvvCapabilityPropertyListContains(
        selected.supportedSEW, "8");
    c.supportsSEW32 = pluginrvv::rvvCapabilityPropertyListContains(
        selected.supportedSEW, "32");
  }
  if (selected.supportedLMUL.empty()) {
    c.supportsM1 = true;
    c.supportsM2 = true;
    c.supportsM4 = true;
    c.supportsM8 = true;
    c.supportsMF2 = selected.rvvVersion == "1.0";
  } else {
    c.supportsMF2 = pluginrvv::rvvCapabilityPropertyListContains(
        selected.supportedLMUL, "mf2");
    c.supportsM1 = pluginrvv::rvvCapabilityPropertyListContains(
        selected.supportedLMUL, "m1");
    c.supportsM2 = pluginrvv::rvvCapabilityPropertyListContains(
        selected.supportedLMUL, "m2");
    c.supportsM4 = pluginrvv::rvvCapabilityPropertyListContains(
        selected.supportedLMUL, "m4");
    c.supportsM8 = pluginrvv::rvvCapabilityPropertyListContains(
        selected.supportedLMUL, "m8");
  }
  return c;
}

mlir::LogicalResult verifyOrWriteStringStamp(weftrvv::DequantizeRowDecodeCoreOp core,
                                             llvm::StringRef name,
                                             llvm::StringRef expected,
                                             bool hasAnyStamp) {
  auto existing = core->getAttrOfType<mlir::StringAttr>(name);
  if (hasAnyStamp) {
    if (!existing || existing.getValue() != expected)
      return core.emitOpError()
             << "pre-emission codebook plan stamp '" << name
             << "' must equal recomputed selected value '" << expected
             << "'; got " << (existing ? existing.getValue() : "<missing>");
    return mlir::success();
  }
  core->setAttr(name, mlir::StringAttr::get(core.getContext(), expected));
  return mlir::success();
}

mlir::LogicalResult verifyOrWriteI64Stamp(weftrvv::DequantizeRowDecodeCoreOp core,
                                          llvm::StringRef name,
                                          std::int64_t expected,
                                          bool hasAnyStamp) {
  auto existing = core->getAttrOfType<mlir::IntegerAttr>(name);
  if (hasAnyStamp) {
    if (!existing || !existing.getType().isSignlessInteger(64) ||
        existing.getInt() != expected)
      return core.emitOpError()
             << "pre-emission codebook plan stamp '" << name
             << "' must equal recomputed selected i64 value " << expected
             << "; got " << (existing ? llvm::Twine(existing.getInt()).str()
                                         : std::string("<missing>"));
    return mlir::success();
  }
  core->setAttr(name,
                mlir::IntegerAttr::get(mlir::IntegerType::get(core.getContext(), 64),
                                       expected));
  return mlir::success();
}

mlir::LogicalResult stampSelectedPlan(
    weftrvv::DequantizeRowDecodeCoreOp core,
    const pluginrvv::RVVSelectedTargetCapabilityFacts &selected,
    const pluginrvv::CodebookGatherDecision &decision) {
  if (!decision.selectedPlan)
    return core.emitOpError()
           << "cannot stamp a rejected codebook gather decision";
  const ::weft::CodebookGatherPlan &plan = *decision.selectedPlan;

  static constexpr llvm::StringLiteral kStampNames[] = {
      "codebook_gather_table",
      "codebook_gather_entries",
      "codebook_gather_strip_lanes",
      "codebook_gather_load_lmul",
      "codebook_gather_minimum_vlen",
      "codebook_gather_provider",
      "codebook_gather_selection_reason",
  };
  bool hasAnyStamp = false;
  for (llvm::StringRef name : kStampNames)
    hasAnyStamp |= core->hasAttr(name);

  if (mlir::failed(verifyOrWriteStringStamp(
          core, "codebook_gather_table",
          ::weft::stringifyCodebookTable(plan.codebookTable), hasAnyStamp)) ||
      mlir::failed(verifyOrWriteI64Stamp(core, "codebook_gather_entries",
                                        plan.codebookEntries, hasAnyStamp)) ||
      mlir::failed(verifyOrWriteI64Stamp(core, "codebook_gather_strip_lanes",
                                        plan.stripLanes, hasAnyStamp)) ||
      mlir::failed(verifyOrWriteStringStamp(core, "codebook_gather_load_lmul",
                                            plan.loadLMUL, hasAnyStamp)) ||
      mlir::failed(verifyOrWriteI64Stamp(core, "codebook_gather_minimum_vlen",
                                        *selected.minimumVLEN, hasAnyStamp)) ||
      mlir::failed(verifyOrWriteStringStamp(core, "codebook_gather_provider",
                                            selected.selectedProviderSymbol,
                                            hasAnyStamp)) ||
      mlir::failed(verifyOrWriteStringStamp(
          core, "codebook_gather_selection_reason",
          pluginrvv::stringifyCodebookGatherReason(decision.reason),
          hasAnyStamp)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult materializeOne(weftrvv::DequantizeRowDecodeCoreOp core) {
  auto mechanism = core.getDequantMechanismAttr();
  if (!mechanism || mechanism.getValue() != "codebook-gather")
    return mlir::success();

  auto scaleModelAttr = core.getCodebookScaleModelAttr();
  if (!scaleModelAttr)
    return core.emitOpError()
           << "codebook plan materialization requires construction-owned typed g "
              "{dequant_mechanism = \"codebook-gather\", recognized "
              "codebook_scale_model}; decode_model is not a post-construction "
              "formula classifier or input";

  std::optional<::weft::CodebookScaleModel> scaleModel =
      ::weft::parseCodebookScaleModel(scaleModelAttr.getValue());
  if (!scaleModel)
    return core.emitOpError()
           << "has unsupported typed codebook_scale_model '"
           << scaleModelAttr.getValue() << "'";

  weft::exec::VariantOp variant =
      core->getParentOfType<weft::exec::VariantOp>();
  weft::exec::KernelOp kernel =
      variant ? variant->getParentOfType<weft::exec::KernelOp>()
              : weft::exec::KernelOp();
  if (!variant || !kernel)
    return core.emitOpError()
           << "codebook plan materialization requires a selected variant nested "
              "in a kernel";

  llvm::Expected<::weft::support::TargetCapabilitySet> capabilities =
      ::weft::support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return core.emitOpError()
           << "could not build canonical target capability set: "
           << llvm::toString(capabilities.takeError());
  llvm::Expected<pluginrvv::RVVSelectedTargetCapabilityFacts> selected =
      pluginrvv::collectRVVSelectedTargetCapabilityFacts(
          variant, *capabilities, "codebook dequant formula materialization");
  if (!selected)
    return core.emitOpError()
           << llvm::toString(selected.takeError());
  llvm::Expected<pluginrvv::CodebookGatherCapabilityFacts> c =
      projectCodebookGatherCapability(*selected);
  if (!c)
    return core.emitOpError() << llvm::toString(c.takeError());

  pluginrvv::CodebookGatherGeometryFacts g;
  g.scaleModel = *scaleModel;
  g.qk = core.getQkAttr().getInt();
  g.weightBlockStride = core.getWeightBlockStrideAttr().getInt();
  g.scaleByteOffset = core.getScaleByteOffsetAttr().getInt();
  g.quantByteOffset = core.getQuantByteOffsetAttr().getInt();
  pluginrvv::CodebookGatherDecision decision =
      pluginrvv::decideCodebookGather(
          g, *c, pluginrvv::CodebookGatherNoStaticContext{});
  if (!decision.isLegal() || !decision.selectedPlan)
    return core.emitOpError()
           << "codebook gather decision rejected domain '" << decision.domain
           << "': " << pluginrvv::stringifyCodebookGatherReason(decision.reason);

  return stampSelectedPlan(core, *selected, decision);
}

} // namespace

mlir::LogicalResult
materializeRVVCodebookGatherPlans(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());

  // Direct lower-to-EmitC callers may still carry the abstract op. Construct
  // only the A3 codebook slice here, before any conversion/emission attempt.
  llvm::SmallVector<weftrvv::GgmlDequantizeRowOp, 4> abstractCodebookRows;
  module.walk([&](weftrvv::GgmlDequantizeRowOp op) {
    std::optional<weftrvv::DequantizeRowStreamFacts> facts =
        weftrvv::lookupDequantizeRowStreamFacts(op.getFormat());
    if (facts && isCodebookFacts(*facts))
      abstractCodebookRows.push_back(op);
  });
  for (weftrvv::GgmlDequantizeRowOp op : abstractCodebookRows) {
    std::optional<weftrvv::DequantizeRowStreamFacts> facts =
        weftrvv::lookupDequantizeRowStreamFacts(op.getFormat());
    if (!facts || mlir::failed(weftrvv::constructTypedDequantizeRowLoopBody(
                      rewriter, op, *facts)))
      return mlir::failure();
  }

  llvm::SmallVector<weftrvv::DequantizeRowDecodeCoreOp, 4> cores;
  module.walk([&](weftrvv::DequantizeRowDecodeCoreOp core) {
    auto mechanism = core.getDequantMechanismAttr();
    if (mechanism && mechanism.getValue() == "codebook-gather")
      cores.push_back(core);
  });
  for (weftrvv::DequantizeRowDecodeCoreOp core : cores)
    if (mlir::failed(materializeOne(core)))
      return mlir::failure();
  return mlir::success();
}

} // namespace weft::conversion::rvv
