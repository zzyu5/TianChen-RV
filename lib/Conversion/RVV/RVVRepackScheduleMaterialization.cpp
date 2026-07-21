#include "Weft/Conversion/RVV/RVVRepackScheduleMaterialization.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"

namespace weft::conversion::rvv {
namespace {

namespace pluginrvv = ::weft::plugin::rvv;
namespace weftrvv = ::weft::rvv;

llvm::Expected<llvm::StringRef>
readRequiredString(weftrvv::TypedRepackGemmLoopBodyOp op,
                   llvm::StringRef name) {
  mlir::Attribute raw = op->getAttr(name);
  if (!raw)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "missing required selected repack schedule stamp '%s'",
        name.str().c_str());
  auto value = llvm::dyn_cast<mlir::StringAttr>(raw);
  if (!value)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "selected repack schedule stamp '%s' must be a typed string attribute",
        name.str().c_str());
  if (value.getValue().empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "selected repack schedule stamp '%s' must not be empty",
        name.str().c_str());
  return value.getValue();
}

llvm::Error invalidToken(llvm::StringRef name, llvm::StringRef token,
                         llvm::StringRef expected) {
  return llvm::createStringError(
      llvm::errc::invalid_argument,
      "selected repack schedule stamp '%s' has unknown value '%s'; expected %s",
      name.str().c_str(), token.str().c_str(), expected.str().c_str());
}

mlir::LogicalResult verifyOne(weftrvv::TypedRepackGemmLoopBodyOp op) {
  llvm::Expected<RVVRepackSchedulePlan> plan =
      readAndVerifyRVVRepackSchedulePlan(op);
  if (!plan)
    return op.emitOpError()
           << "failed pre-emission selected schedule verification: "
           << llvm::toString(plan.takeError());
  return mlir::success();
}

} // namespace

static llvm::Expected<RVVRepackSchedulePlan>
readRVVRepackSchedulePlanFields(weftrvv::TypedRepackGemmLoopBodyOp op) {
  RVVRepackSchedulePlan plan;

  llvm::Expected<llvm::StringRef> loopOrder =
      readRequiredString(op, "weft_rvv.loop_order");
  if (!loopOrder)
    return loopOrder.takeError();
  std::optional<pluginrvv::RVVRepackLoopOrder> parsedLoopOrder =
      pluginrvv::parseRVVRepackLoopOrder(*loopOrder);
  if (!parsedLoopOrder)
    return invalidToken("weft_rvv.loop_order", *loopOrder,
                        "'row_outer' or 'col_outer'");
  plan.loopOrder = *parsedLoopOrder;

  llvm::Expected<llvm::StringRef> loopReason =
      readRequiredString(op, "weft_rvv.loop_order_selection_reason");
  if (!loopReason)
    return loopReason.takeError();
  std::optional<pluginrvv::RVVTilingSelectionReason> parsedLoopReason =
      pluginrvv::parseRVVTilingSelectionReason(*loopReason);
  if (!parsedLoopReason)
    return invalidToken("weft_rvv.loop_order_selection_reason", *loopReason,
                        "'measured' or 'prior'");
  plan.loopOrderReason = *parsedLoopReason;

  std::optional<pluginrvv::RVVTilingBottleneckShape> shape =
      pluginrvv::classifyTilingBottleneckShape(op.getFoldModel());
  bool hasVariant = op->hasAttr("weft_rvv.tiling_variant");
  bool hasReason = op->hasAttr("weft_rvv.tiling_selection_reason");
  if (hasVariant != hasReason)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "incomplete selected SP4 schedule stamp: tiling_variant and "
        "tiling_selection_reason must be present together");
  if (shape && !hasVariant)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "missing required selected SP4 schedule stamp for a fold_model with a "
        "realized output-tiling body");
  if (!shape && hasVariant)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "selected SP4 schedule stamp is invalid for a fold_model with no SP4 "
        "output-tiling axis");

  if (hasVariant) {
    llvm::Expected<llvm::StringRef> tiling =
        readRequiredString(op, "weft_rvv.tiling_variant");
    if (!tiling)
      return tiling.takeError();
    plan.tiling = pluginrvv::parseRVVRepackTilingVariant(*tiling);
    if (!plan.tiling)
      return invalidToken("weft_rvv.tiling_variant", *tiling,
                          "'plain' or 's6_tiled'");

    llvm::Expected<llvm::StringRef> tilingReason =
        readRequiredString(op, "weft_rvv.tiling_selection_reason");
    if (!tilingReason)
      return tilingReason.takeError();
    plan.tilingReason =
        pluginrvv::parseRVVTilingSelectionReason(*tilingReason);
    if (!plan.tilingReason)
      return invalidToken("weft_rvv.tiling_selection_reason", *tilingReason,
                          "'only_feasible'");
  }

  return plan;
}

llvm::Expected<RVVRepackSchedulePlan>
readAndVerifyRVVRepackSchedulePlan(
    weftrvv::TypedRepackGemmLoopBodyOp op) {
  llvm::Expected<RVVRepackSchedulePlan> plan =
      readRVVRepackSchedulePlanFields(op);
  if (!plan)
    return plan.takeError();

  std::optional<pluginrvv::RVVTilingBottleneckShape> shape =
      pluginrvv::classifyTilingBottleneckShape(op.getFoldModel());
  if (shape) {
    // The typed body carries the selected vector strip as a construction fact:
    // eight e16 lanes require VLEN>=128, sixteen require VLEN>=256. RVV has a
    // fixed 32 architectural vector registers. Reuse the selector's legality
    // function rather than recreating a second SP4 domain in Conversion.
    const std::int64_t bodyMinimumVLEN =
        static_cast<std::int64_t>(op.getHalfLanes()) * 16;
    constexpr std::int64_t kRVVArchitecturalVectorRegisters = 32;
    llvm::SmallVector<pluginrvv::RVVRepackTilingVariant, 2> feasible =
        pluginrvv::tilingVariantFeasibleSet(
            *shape, bodyMinimumVLEN, kRVVArchitecturalVectorRegisters);
    if (feasible.size() != 1 || !plan->tiling ||
        *plan->tiling != feasible.front())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "stale, forged, or unrealizable SP4 selected stamp for fold_model "
          "'%s'; the current typed resource facts admit only '%s'",
          op.getFoldModel().str().c_str(),
          feasible.empty()
              ? "<none>"
              : pluginrvv::stringifyRVVRepackTilingVariant(feasible.front())
                    .str()
                    .c_str());
    if (!plan->tilingReason ||
        *plan->tilingReason !=
            pluginrvv::RVVTilingSelectionReason::OnlyFeasible)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "SP4 selection_reason must be 'only_feasible' because the current "
          "legal set contains exactly one realized body");
  }

  pluginrvv::RVVRepackLoopOrder layoutPrior =
      pluginrvv::repackColGroupOuterForLayout(
          op.getWeightBlockStrideAttr().getInt(),
          op.getActivationBlockStrideAttr().getInt())
          ? pluginrvv::RVVRepackLoopOrder::ColOuter
          : pluginrvv::RVVRepackLoopOrder::RowOuter;
  switch (plan->loopOrderReason) {
  case pluginrvv::RVVTilingSelectionReason::Measured:
    // A qualified measurement may rank either already-realized loop body. A4a
    // checks structure and mechanical consumption; A5 owns evidence qualification.
    break;
  case pluginrvv::RVVTilingSelectionReason::Prior:
    if (plan->loopOrder != layoutPrior)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "stale or forged loop-order selected stamp: reason 'prior' requires "
          "the current layout-formula value '%s'",
          pluginrvv::stringifyRVVRepackLoopOrder(layoutPrior).str().c_str());
    break;
  case pluginrvv::RVVTilingSelectionReason::OnlyFeasible:
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "loop-order reason 'only_feasible' is invalid because both row_outer "
        "and col_outer have realized bodies");
  }

  return std::move(*plan);
}

mlir::LogicalResult verifyRVVRepackSchedulePlans(mlir::ModuleOp module) {
  llvm::SmallVector<weftrvv::TypedRepackGemmLoopBodyOp, 8> loops;
  module.walk([&](weftrvv::TypedRepackGemmLoopBodyOp op) {
    loops.push_back(op);
  });
  for (weftrvv::TypedRepackGemmLoopBodyOp op : loops)
    if (mlir::failed(verifyOne(op)))
      return mlir::failure();
  return mlir::success();
}

} // namespace weft::conversion::rvv
