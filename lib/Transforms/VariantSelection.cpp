#include "TianChenRV/Transforms/VariantSelection.h"

#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_SELECTVARIANTS
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kConditionAttrName("condition");
constexpr llvm::StringLiteral kGuardAttrName("guard");
constexpr llvm::StringLiteral kMessageAttrName("message");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kReasonAttrName("reason");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSelectionKindAttrName("selection_kind");
constexpr llvm::StringLiteral kSeverityAttrName("severity");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kTargetAttrName("target");
constexpr llvm::StringLiteral kSelectedReasonValue("variant-selected");
constexpr llvm::StringLiteral kMissingFallbackReasonValue(
    "fallback-coverage-missing");
constexpr llvm::StringLiteral kMissingFallbackSelectionKindValue(
    "missing-conservative-fallback");
constexpr llvm::StringLiteral kStaticSelectionKindValue("static-variant");
constexpr llvm::StringLiteral kFallbackOnlySelectionKindValue("fallback-only");
constexpr llvm::StringLiteral kPreferenceAvailableAttrName(
    "preference_available");
constexpr llvm::StringLiteral kPreferenceScoreAttrName("preference_score");
constexpr llvm::StringLiteral kPreferenceRankAttrName("preference_rank");
constexpr llvm::StringLiteral kPreferencePolicyAttrName("preference_policy");
constexpr llvm::StringLiteral kPreferenceExplanationAttrName(
    "preference_explanation");
constexpr llvm::StringLiteral kPreferenceTieBreakAttrName(
    "preference_tie_break");
constexpr llvm::StringLiteral kFallbackRoleAttrName("fallback_role");
constexpr llvm::StringLiteral kRuntimeGuardPolicy("capability_dispatch_guard");

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::VariantCostEstimate;
using tianchenrv::plugin::VariantCostRankingEntry;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

struct RequirementLegality {
  bool available = false;
  bool conflictFree = false;
  bool requiresRuntimeCapabilityGuard = false;
};

llvm::Error makeSelectionError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV variant selection failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeSelectionError(KernelOp kernel, VariantOp variant,
                               llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV variant selection failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";

  if (variant)
    stream << ", variant @" << variant.getSymName();
  else
    stream << ", variant <missing>";

  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

bool hasKernelBody(KernelOp kernel) {
  return kernel && !kernel.getBody().empty();
}

bool hasDirectParent(VariantOp variant, KernelOp kernel) {
  return variant && kernel && variant->getParentOp() == kernel.getOperation();
}

bool hasDirectDispatch(KernelOp kernel) {
  if (!hasKernelBody(kernel))
    return false;

  for (mlir::Operation &operation : kernel.getBody().front()) {
    if (llvm::isa<DispatchOp>(operation))
      return true;
  }
  return false;
}

bool hasNonEmptyStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return attr && !attr.getValue().trim().empty();
}

bool hasGenericDecisionMetadata(VariantOp variant) {
  return hasNonEmptyStringAttr(variant.getOperation(), kConditionAttrName) ||
         hasNonEmptyStringAttr(variant.getOperation(), kGuardAttrName) ||
         hasNonEmptyStringAttr(variant.getOperation(), kPolicyAttrName);
}

bool hasConservativeFallbackRoleAttr(VariantOp variant) {
  auto roleAttr = variant->getAttrOfType<mlir::StringAttr>(
      plugin::kVariantFallbackRoleAttrName);
  return roleAttr &&
         roleAttr.getValue() == plugin::kConservativeFallbackRoleValue;
}

bool isConservativeFallbackCandidate(VariantOp variant,
                                     const VariantCostEstimate &estimate) {
  return estimate.getFallbackRole() == plugin::VariantFallbackRole::
                                           ConservativeFallback ||
         hasConservativeFallbackRoleAttr(variant);
}

void copyStringAttrIfPresent(mlir::OperationState &state, VariantOp variant,
                             llvm::StringRef attrName) {
  auto attr = variant->getAttrOfType<mlir::StringAttr>(attrName);
  if (attr && !attr.getValue().trim().empty())
    state.addAttribute(attrName, attr);
}

llvm::Expected<RequirementLegality> analyzeRequirementLegality(
    KernelOp kernel, VariantOp variant,
    const TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeSelectionError(
        kernel, variant,
        "requires structured array attribute 'requires' containing capability "
        "symbol references");

  RequirementLegality legality;
  legality.available = true;
  legality.conflictFree = true;

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeSelectionError(
          kernel, variant,
          "requires metadata must contain only capability symbol references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      return makeSelectionError(
          kernel, variant,
          llvm::Twine("requires capability @") + symbolRef.getValue() +
              " is not present in the supplied TargetCapabilitySet");

    if (!capability->isAvailable()) {
      legality.available = false;
      legality.conflictFree = false;
      legality.requiresRuntimeCapabilityGuard = true;
      continue;
    }

    llvm::SmallVector<support::CapabilityConflict, 4> conflicts;
    capabilities.collectAvailableConflictsForCapability(*capability,
                                                        conflicts);
    if (!conflicts.empty()) {
      legality.conflictFree = false;
      legality.requiresRuntimeCapabilityGuard = true;
    }
  }

  return legality;
}

llvm::Error validateOriginOwnedCost(KernelOp kernel,
                                    const VariantCostRankingEntry &entry) {
  VariantOp variant = entry.variant;
  if (!variant)
    return makeSelectionError(kernel, variant,
                              "cost ranking entry has no variant");

  if (!hasDirectParent(variant, kernel))
    return makeSelectionError(
        kernel, variant,
        "ranked variant is not a direct child of the request kernel");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue().trim().empty())
    return makeSelectionError(
        kernel, variant,
        "requires non-empty string attribute 'origin' for origin-owned cost");

  if (!entry.estimate.hasScore())
    return makeSelectionError(
        kernel, variant,
        "requires origin-owned cost estimate with a present score");

  if (entry.estimate.getOriginPlugin() != originAttr.getValue())
    return makeSelectionError(
        kernel, variant,
        llvm::Twine("cost estimate origin '") +
            entry.estimate.getOriginPlugin() +
            "' does not match variant origin '" + originAttr.getValue() + "'");

  if (entry.estimate.getVariantSymbol() != variant.getSymName())
    return makeSelectionError(
        kernel, variant,
        llvm::Twine("cost estimate variant @") +
            entry.estimate.getVariantSymbol() +
            " does not match ranked variant @" + variant.getSymName());

  return llvm::Error::success();
}

llvm::Error validatePlanVariant(KernelOp kernel, VariantOp variant,
                                llvm::StringRef role) {
  if (!variant)
    return makeSelectionError(kernel, variant,
                              llvm::Twine("missing ") + role + " variant");

  if (!hasDirectParent(variant, kernel))
    return makeSelectionError(
        kernel, variant,
        llvm::Twine(role) + " variant is not a direct child of the plan kernel");

  return llvm::Error::success();
}

llvm::Error validatePlanCase(KernelOp kernel,
                             const VariantSelectionCase &selectionCase,
                             llvm::StringRef role) {
  if (llvm::Error error =
          validatePlanVariant(kernel, selectionCase.variant, role))
    return error;

  return validateOriginOwnedCost(
      kernel, VariantCostRankingEntry{selectionCase.variant, selectionCase.cost,
                                      selectionCase.originalIndex});
}

std::optional<std::size_t>
findRankedCaseIndex(const VariantSelectionPlan &plan, VariantOp variant) {
  for (std::size_t index = 0; index < plan.rankedVariants.size(); ++index) {
    if (plan.rankedVariants[index].variant == variant)
      return index;
  }
  return std::nullopt;
}

const VariantSelectionCase *
findRankedCase(const VariantSelectionPlan &plan, VariantOp variant) {
  std::optional<std::size_t> index = findRankedCaseIndex(plan, variant);
  if (!index)
    return nullptr;
  return &plan.rankedVariants[*index];
}

std::string buildPreferenceTieBreakReason(const VariantSelectionPlan &plan,
                                          std::size_t rank) {
  if (rank >= plan.rankedVariants.size())
    return "target-neutral preference metadata is unavailable";

  const VariantSelectionCase &selectionCase = plan.rankedVariants[rank];
  llvm::StringRef preferenceAvailability =
      selectionCase.cost.hasExplicitPreference()
          ? "explicit plugin preference"
          : "no explicit plugin preference";

  if (rank > 0 &&
      plan.rankedVariants[rank - 1].cost.hasExplicitPreference() ==
          selectionCase.cost.hasExplicitPreference() &&
      plan.rankedVariants[rank - 1].cost.getScore() ==
          selectionCase.cost.getScore()) {
    return (llvm::Twine("equal ") + preferenceAvailability +
            " score; fallback role, original IR order, then symbol name "
            "decide the stable tie-break")
        .str();
  }

  if (rank == 0)
    return (llvm::Twine("best ") + preferenceAvailability +
            " score; equal scores use fallback role, original IR order, then "
            "symbol name")
        .str();

  return (llvm::Twine("ranked by ") + preferenceAvailability +
          " score; fallback role, original IR order, then symbol name remain "
          "the stable tie-breaks")
      .str();
}

void addPreferenceMetadata(mlir::OpBuilder &builder,
                           mlir::OperationState &state,
                           const VariantSelectionPlan &plan,
                           const VariantSelectionCase &selectionCase) {
  std::optional<std::size_t> rank =
      findRankedCaseIndex(plan, selectionCase.variant);
  std::size_t rankValue = rank.value_or(selectionCase.originalIndex);

  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(selectionCase.cost.getOriginPlugin()));
  state.addAttribute(
      kPreferenceAvailableAttrName,
      builder.getBoolAttr(selectionCase.cost.hasExplicitPreference()));
  state.addAttribute(kPreferenceScoreAttrName,
                     builder.getF64FloatAttr(selectionCase.cost.getScore()));
  state.addAttribute(kPreferenceRankAttrName,
                     builder.getI64IntegerAttr(rankValue));
  if (selectionCase.cost.hasPolicy())
    state.addAttribute(kPreferencePolicyAttrName,
                       builder.getStringAttr(selectionCase.cost.getPolicy()));
  if (selectionCase.cost.hasExplanation())
    state.addAttribute(
        kPreferenceExplanationAttrName,
        builder.getStringAttr(selectionCase.cost.getExplanation()));
  state.addAttribute(
      kPreferenceTieBreakAttrName,
      builder.getStringAttr(buildPreferenceTieBreakReason(plan, rankValue)));
  if (isConservativeFallbackCandidate(selectionCase.variant,
                                      selectionCase.cost))
    state.addAttribute(kFallbackRoleAttrName,
                       builder.getStringAttr(
                           plugin::kConservativeFallbackRoleValue));
}

DispatchCaseOp createDispatchCase(mlir::OpBuilder &builder,
                                  const VariantSelectionPlan &plan,
                                  VariantSelectionCase selectionCase) {
  VariantOp variant = selectionCase.variant;
  mlir::OperationState state(variant.getLoc(),
                             DispatchCaseOp::getOperationName());
  state.addAttribute(kTargetAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  copyStringAttrIfPresent(state, variant, kConditionAttrName);
  copyStringAttrIfPresent(state, variant, kGuardAttrName);
  copyStringAttrIfPresent(state, variant, kPolicyAttrName);
  if (selectionCase.requiresRuntimeCapabilityGuard &&
      !selectionCase.hasGenericDecisionMetadata)
    state.addAttribute(kPolicyAttrName,
                       builder.getStringAttr(kRuntimeGuardPolicy));
  addPreferenceMetadata(builder, state, plan, selectionCase);
  return llvm::cast<DispatchCaseOp>(builder.create(state));
}

FallbackOp createFallback(mlir::OpBuilder &builder,
                          const VariantSelectionPlan &plan,
                          VariantSelectionCase selectionCase) {
  VariantOp variant = selectionCase.variant;
  mlir::OperationState state(variant.getLoc(), FallbackOp::getOperationName());
  state.addAttribute(kTargetAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  addPreferenceMetadata(builder, state, plan, selectionCase);
  return llvm::cast<FallbackOp>(builder.create(state));
}

llvm::StringRef stringifySelectedMarkerKind(VariantSelectionKind kind) {
  switch (kind) {
  case VariantSelectionKind::StaticVariant:
    return kStaticSelectionKindValue;
  case VariantSelectionKind::FallbackOnly:
    return kFallbackOnlySelectionKindValue;
  case VariantSelectionKind::RuntimeDispatch:
    return "runtime-dispatch";
  case VariantSelectionKind::NoViableVariant:
    return "no-viable-variant";
  }
  return "unknown";
}

llvm::StringRef getSelectedMarkerMessage(VariantSelectionKind kind) {
  switch (kind) {
  case VariantSelectionKind::StaticVariant:
    return "static variant selected by generic cost and capability planning";
  case VariantSelectionKind::FallbackOnly:
    return "fallback-only variant selected by generic cost and capability planning";
  case VariantSelectionKind::RuntimeDispatch:
  case VariantSelectionKind::NoViableVariant:
    break;
  }
  return "variant selection marker";
}

bool isSelectedPathMarker(DiagnosticOp diagnostic) {
  if (!diagnostic)
    return false;

  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(kReasonAttrName);
  return reason && reason.getValue() == kSelectedReasonValue &&
         diagnostic->hasAttr(kTargetAttrName) &&
         diagnostic->hasAttr(kSelectionKindAttrName);
}

bool isMissingFallbackCoverageDiagnostic(DiagnosticOp diagnostic) {
  if (!diagnostic)
    return false;

  auto reason =
      diagnostic->getAttrOfType<mlir::StringAttr>(kReasonAttrName);
  return reason && reason.getValue() == kMissingFallbackReasonValue &&
         diagnostic->hasAttr(kSelectionKindAttrName);
}

void collectDirectSelectedPathMarkers(
    KernelOp kernel, llvm::SmallVectorImpl<DiagnosticOp> &markers) {
  if (!hasKernelBody(kernel))
    return;

  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(operation);
    if (diagnostic && isSelectedPathMarker(diagnostic))
      markers.push_back(diagnostic);
  }
}

void collectDirectMissingFallbackDiagnostics(
    KernelOp kernel, llvm::SmallVectorImpl<DiagnosticOp> &diagnostics) {
  if (!hasKernelBody(kernel))
    return;

  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(operation);
    if (diagnostic && isMissingFallbackCoverageDiagnostic(diagnostic))
      diagnostics.push_back(diagnostic);
  }
}

bool markerMatchesSelectionPlan(DiagnosticOp marker,
                                VariantSelectionKind selectionKind,
                                VariantOp selectedVariant) {
  auto target =
      marker->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  auto kind =
      marker->getAttrOfType<mlir::StringAttr>(kSelectionKindAttrName);
  return target && kind && selectedVariant &&
         target.getValue() == selectedVariant.getSymName() &&
         kind.getValue() == stringifySelectedMarkerKind(selectionKind);
}

bool diagnosticTargetsVariant(DiagnosticOp diagnostic, VariantOp variant) {
  auto target =
      diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  return target && variant && target.getValue() == variant.getSymName();
}

llvm::Error materializeMissingFallbackCoverageDiagnostic(
    mlir::OpBuilder &builder, const VariantSelectionPlan &plan,
    const VariantSelectionCase &selectedCase) {
  KernelOp kernel = plan.kernel;
  VariantOp selectedVariant = selectedCase.variant;
  if (!selectedVariant)
    return makeSelectionError(
        kernel,
        "missing-fallback diagnostic requires a selected variant target");

  llvm::SmallVector<DiagnosticOp, 2> existingDiagnostics;
  collectDirectMissingFallbackDiagnostics(kernel, existingDiagnostics);
  if (existingDiagnostics.size() > 1)
    return makeSelectionError(
        kernel,
        "kernel already contains multiple direct missing-fallback diagnostics");

  if (!existingDiagnostics.empty()) {
    DiagnosticOp existing = existingDiagnostics.front();
    if (diagnosticTargetsVariant(existing, selectedVariant))
      return llvm::Error::success();
    return makeSelectionError(
        kernel,
        "kernel already contains a direct missing-fallback diagnostic for a "
        "different selected target");
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&kernel.getBody().front());
  mlir::OperationState state(selectedVariant.getLoc(),
                             DiagnosticOp::getOperationName());
  state.addAttribute(kReasonAttrName,
                     builder.getStringAttr(kMissingFallbackReasonValue));
  state.addAttribute(
      kMessageAttrName,
      builder.getStringAttr(
          "no plugin-provided conservative fallback candidate is available; "
          "tcrv.exec.dispatch fallback is not invented"));
  state.addAttribute(kSeverityAttrName, builder.getStringAttr("warning"));
  state.addAttribute(kStatusAttrName, builder.getStringAttr("missing"));
  state.addAttribute(kTargetAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  selectedVariant.getSymName()));
  state.addAttribute(kSelectionKindAttrName,
                     builder.getStringAttr(kMissingFallbackSelectionKindValue));
  addPreferenceMetadata(builder, state, plan, selectedCase);
  builder.create(state);
  return llvm::Error::success();
}

class SelectVariantsPass final
    : public impl::SelectVariantsBase<SelectVariantsPass> {
public:
  SelectVariantsPass() : registry(&ownedRegistry) {}

  explicit SelectVariantsPass(const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  SelectVariantsPass(const SelectVariantsPass &other)
      : impl::SelectVariantsBase<SelectVariantsPass>(other),
        registry(other.registry == &other.ownedRegistry ? &ownedRegistry
                                                        : other.registry) {}

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());
    mlir::WalkResult walkResult =
        getOperation()->walk([&](KernelOp kernel) -> mlir::WalkResult {
          if (mlir::failed(runSelection(builder, kernel)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });

    if (walkResult.wasInterrupted())
      signalPassFailure();
  }

private:
  mlir::LogicalResult runSelection(mlir::OpBuilder &builder, KernelOp kernel) {
    TargetCapabilitySet capabilities =
        TargetCapabilitySet::buildFromKernel(kernel);
    if (llvm::Error error =
            registry->verifyKernelVariantLegality(kernel, capabilities))
      return emitSelectionError(kernel, std::move(error));

    llvm::Expected<VariantSelectionPlan> planOrError =
        planKernelVariantSelection(kernel, capabilities, *registry);
    if (!planOrError)
      return emitSelectionError(kernel, planOrError.takeError());

    VariantSelectionPlan plan = std::move(*planOrError);
    switch (plan.kind) {
    case VariantSelectionKind::RuntimeDispatch:
      if (llvm::Error error = materializeRuntimeDispatchPlan(builder, plan))
        return emitSelectionError(kernel, std::move(error));
      return mlir::success();
    case VariantSelectionKind::StaticVariant:
    case VariantSelectionKind::FallbackOnly:
      if (llvm::Error error = materializeSelectedVariantMarker(builder, plan))
        return emitSelectionError(kernel, std::move(error));
      return mlir::success();
    case VariantSelectionKind::NoViableVariant:
      return mlir::success();
    }

    return emitSelectionError(
        kernel, makeSelectionError(kernel, "unknown variant selection kind"));
  }

  mlir::LogicalResult emitSelectionError(KernelOp kernel, llvm::Error error) {
    std::string message = llvm::toString(std::move(error));
    if (kernel)
      kernel.emitError() << message;
    else
      getOperation()->emitError() << message;
    return mlir::failure();
  }

  ExtensionPluginRegistry ownedRegistry;
  const ExtensionPluginRegistry *registry = nullptr;
};

} // namespace

llvm::Expected<VariantSelectionPlan> planKernelVariantSelection(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry) {
  if (!kernel)
    return makeSelectionError(kernel, "requires a tcrv.exec.kernel");

  if (!hasKernelBody(kernel))
    return makeSelectionError(kernel,
                              "requires kernel to have a materialized body "
                              "block");

  llvm::SmallVector<VariantCostRankingEntry, 4> rankedCosts;
  if (llvm::Error error =
          registry.rankKernelVariantsByCost(kernel, capabilities, rankedCosts))
    return std::move(error);

  VariantSelectionPlan plan;
  plan.kernel = kernel;
  if (rankedCosts.empty()) {
    plan.kind = VariantSelectionKind::NoViableVariant;
    return plan;
  }

  std::optional<std::size_t> selectedIndex;
  std::optional<std::size_t> fallbackIndex;
  for (const VariantCostRankingEntry &entry : rankedCosts) {
    if (llvm::Error error = validateOriginOwnedCost(kernel, entry))
      return std::move(error);

    llvm::Expected<RequirementLegality> legality =
        analyzeRequirementLegality(kernel, entry.variant, capabilities);
    if (!legality)
      return legality.takeError();

    bool hasDecisionMetadata = hasGenericDecisionMetadata(entry.variant);
    if (!legality->available && !hasDecisionMetadata)
      return makeSelectionError(
          kernel, entry.variant,
          "unavailable variant lacks generic condition/guard/policy metadata "
          "and cannot be selected or retained by runtime dispatch");

    bool conflictFreeAvailable = legality->available && legality->conflictFree;
    bool conservativeFallback =
        isConservativeFallbackCandidate(entry.variant, entry.estimate);
    if (conflictFreeAvailable && !selectedIndex)
      selectedIndex = plan.rankedVariants.size();
    if (conflictFreeAvailable && conservativeFallback && !fallbackIndex)
      fallbackIndex = plan.rankedVariants.size();

    plan.rankedVariants.push_back(VariantSelectionCase{
        entry.variant,
        entry.estimate,
        entry.originalIndex,
        legality->available,
        legality->conflictFree,
        hasDecisionMetadata,
        legality->requiresRuntimeCapabilityGuard});
  }

  bool hasDispatchGuardedCandidate = llvm::any_of(
      plan.rankedVariants, [](const VariantSelectionCase &candidate) {
        return candidate.requiresRuntimeCapabilityGuard ||
               candidate.hasGenericDecisionMetadata;
      });

  if (!selectedIndex && hasDispatchGuardedCandidate && !fallbackIndex)
    return makeSelectionError(
        kernel,
        "no plugin-provided conflict-free conservative fallback candidate is "
        "available; cannot materialize tcrv.exec.dispatch without inventing an "
        "implicit fallback");

  if (!selectedIndex)
    return makeSelectionError(
        kernel,
        "no direct variant is conflict-free and generically available for selection "
        "under the supplied TargetCapabilitySet");

  const VariantSelectionCase &selectedCase =
      plan.rankedVariants[*selectedIndex];
  plan.selectedVariant = selectedCase.variant;
  if (fallbackIndex) {
    const VariantSelectionCase &fallbackCase =
        plan.rankedVariants[*fallbackIndex];
    plan.fallback = fallbackCase.variant;
  } else {
    plan.missingFallbackCoverage = true;
  }

  bool dispatchWouldNeedFallback = false;
  for (std::size_t index = 0; index < plan.rankedVariants.size(); ++index) {
    const VariantSelectionCase &candidate = plan.rankedVariants[index];
    if (index == *selectedIndex)
      continue;
    if (candidate.requiresRuntimeCapabilityGuard ||
        (candidate.hasGenericDecisionMetadata &&
         (index < *selectedIndex || !candidate.genericallyAvailable))) {
      dispatchWouldNeedFallback = true;
      break;
    }
  }

  if (dispatchWouldNeedFallback && !fallbackIndex)
    return makeSelectionError(
        kernel,
        "no plugin-provided conflict-free conservative fallback candidate is "
        "available; cannot materialize tcrv.exec.dispatch without inventing an "
        "implicit fallback");

  if (fallbackIndex && *selectedIndex != *fallbackIndex) {
    if (!selectedCase.hasGenericDecisionMetadata &&
        !selectedCase.requiresRuntimeCapabilityGuard)
      return makeSelectionError(
          kernel, selectedCase.variant,
          "selected dispatch case requires non-empty generic condition, guard, "
          "or policy metadata when a distinct fallback is present");
    plan.dispatchCases.push_back(selectedCase);
  }

  for (std::size_t index = 0; index < plan.rankedVariants.size(); ++index) {
    if ((fallbackIndex && index == *fallbackIndex) || index == *selectedIndex)
      continue;
    const VariantSelectionCase &candidate = plan.rankedVariants[index];
    if (!candidate.requiresRuntimeCapabilityGuard &&
        !candidate.hasGenericDecisionMetadata)
      continue;

    if (candidate.requiresRuntimeCapabilityGuard ||
        (fallbackIndex && index < *fallbackIndex) || index < *selectedIndex ||
        !candidate.genericallyAvailable)
      plan.dispatchCases.push_back(candidate);
  }

  if (fallbackIndex && !plan.dispatchCases.empty()) {
    plan.kind = VariantSelectionKind::RuntimeDispatch;
    return plan;
  }

  if (fallbackIndex && plan.rankedVariants.size() == 1) {
    plan.kind = VariantSelectionKind::FallbackOnly;
    return plan;
  }

  plan.kind = VariantSelectionKind::StaticVariant;
  return plan;
}

llvm::Expected<VariantSelectionPlan> planKernelVariantSelection(
    KernelOp kernel, const ExtensionPluginRegistry &registry) {
  if (!kernel)
    return makeSelectionError(kernel, "requires a tcrv.exec.kernel");

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  return planKernelVariantSelection(kernel, capabilities, registry);
}

llvm::Error materializeRuntimeDispatchPlan(mlir::OpBuilder &builder,
                                           const VariantSelectionPlan &plan,
                                           DispatchOp *createdDispatch) {
  if (createdDispatch)
    *createdDispatch = DispatchOp();

  KernelOp kernel = plan.kernel;
  if (!kernel)
    return makeSelectionError(kernel,
                              "requires a tcrv.exec.kernel for dispatch "
                              "materialization");

  if (!hasKernelBody(kernel))
    return makeSelectionError(kernel,
                              "requires kernel to have a materialized body "
                              "block for dispatch materialization");

  if (plan.kind != VariantSelectionKind::RuntimeDispatch)
    return makeSelectionError(
        kernel,
        "can only materialize a VariantSelectionKind::RuntimeDispatch plan");

  if (hasDirectDispatch(kernel))
    return makeSelectionError(
        kernel,
        "kernel already contains a direct tcrv.exec.dispatch; selection "
        "materialization refuses to create a competing dispatch");

  const VariantSelectionCase *fallbackCase =
      findRankedCase(plan, plan.fallback);
  if (!fallbackCase)
    return makeSelectionError(
        kernel, plan.fallback,
        "fallback variant is missing from ranked preference metadata");
  if (llvm::Error error = validatePlanCase(kernel, *fallbackCase, "fallback"))
    return error;
  if (!fallbackCase->genericallyAvailable || !fallbackCase->conflictFree ||
      fallbackCase->requiresRuntimeCapabilityGuard)
    return makeSelectionError(
        kernel, plan.fallback,
        "runtime dispatch fallback must be conflict-free and generically "
        "available under the plan TargetCapabilitySet");

  if (plan.dispatchCases.empty())
    return makeSelectionError(
        kernel, "runtime dispatch plan requires at least one dispatch case");

  llvm::StringSet<> seenCaseTargets;
  for (const VariantSelectionCase &selectionCase : plan.dispatchCases) {
    if (llvm::Error error =
            validatePlanCase(kernel, selectionCase, "dispatch case"))
      return error;

    if (selectionCase.variant == plan.fallback)
      return makeSelectionError(
          kernel, selectionCase.variant,
          "dispatch case must not duplicate the fallback target");

    if (!selectionCase.hasGenericDecisionMetadata &&
        !selectionCase.requiresRuntimeCapabilityGuard)
      return makeSelectionError(
          kernel, selectionCase.variant,
          "dispatch case requires non-empty generic condition, guard, or "
          "policy metadata");

    VariantOp caseVariant = selectionCase.variant;
    if (!seenCaseTargets.insert(caseVariant.getSymName()).second)
      return makeSelectionError(kernel, selectionCase.variant,
                                "duplicate dispatch case target in selection "
                                "plan");
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  DispatchOp dispatch = builder.create<DispatchOp>(kernel.getLoc());
  if (dispatch.getBody().empty())
    dispatch.getBody().emplaceBlock();

  builder.setInsertionPointToEnd(&dispatch.getBody().front());
  for (VariantSelectionCase selectionCase : plan.dispatchCases)
    createDispatchCase(builder, plan, selectionCase);
  createFallback(builder, plan, *fallbackCase);

  if (createdDispatch)
    *createdDispatch = dispatch;

  return llvm::Error::success();
}

llvm::Error materializeSelectedVariantMarker(
    mlir::OpBuilder &builder, const VariantSelectionPlan &plan,
    DiagnosticOp *createdMarker) {
  if (createdMarker)
    *createdMarker = DiagnosticOp();

  KernelOp kernel = plan.kernel;
  if (!kernel)
    return makeSelectionError(kernel,
                              "requires a tcrv.exec.kernel for selected-path "
                              "marker materialization");

  if (!hasKernelBody(kernel))
    return makeSelectionError(kernel,
                              "requires kernel to have a materialized body "
                              "block for selected-path marker materialization");

  if (plan.kind != VariantSelectionKind::StaticVariant &&
      plan.kind != VariantSelectionKind::FallbackOnly)
    return makeSelectionError(
        kernel,
        "can only materialize selected-path marker for StaticVariant or "
        "FallbackOnly plans");

  if (hasDirectDispatch(kernel))
    return makeSelectionError(
        kernel,
        "kernel already contains a direct tcrv.exec.dispatch; selected-path "
        "marker materialization refuses to create a competing selected surface");

  VariantOp selectedVariant = plan.selectedVariant ? plan.selectedVariant
                                                   : plan.fallback;
  const VariantSelectionCase *selectedCase =
      findRankedCase(plan, selectedVariant);
  if (!selectedCase)
    return makeSelectionError(
        kernel, selectedVariant,
        "selected variant is missing from ranked preference metadata");
  if (llvm::Error error = validatePlanCase(kernel, *selectedCase, "selected"))
    return error;

  llvm::SmallVector<DiagnosticOp, 2> existingMarkers;
  collectDirectSelectedPathMarkers(kernel, existingMarkers);
  if (existingMarkers.size() > 1)
    return makeSelectionError(
        kernel,
        "kernel already contains multiple direct selected-path diagnostic "
        "markers");

  if (!existingMarkers.empty()) {
    if (markerMatchesSelectionPlan(existingMarkers.front(), plan.kind,
                                   selectedVariant)) {
      if (createdMarker)
        *createdMarker = existingMarkers.front();
      if (plan.missingFallbackCoverage)
        return materializeMissingFallbackCoverageDiagnostic(
            builder, plan, *selectedCase);
      return llvm::Error::success();
    }

    return makeSelectionError(
        kernel,
        "kernel already contains a direct selected-path diagnostic marker for "
        "a different target or selection kind");
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  mlir::OperationState state(selectedVariant.getLoc(),
                             DiagnosticOp::getOperationName());
  state.addAttribute(kReasonAttrName,
                     builder.getStringAttr(kSelectedReasonValue));
  state.addAttribute(kMessageAttrName,
                     builder.getStringAttr(getSelectedMarkerMessage(plan.kind)));
  state.addAttribute(kSeverityAttrName, builder.getStringAttr("note"));
  state.addAttribute(kStatusAttrName, builder.getStringAttr("selected"));
  state.addAttribute(kTargetAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  selectedVariant.getSymName()));
  state.addAttribute(kSelectionKindAttrName,
                     builder.getStringAttr(stringifySelectedMarkerKind(plan.kind)));
  addPreferenceMetadata(builder, state, plan, *selectedCase);
  DiagnosticOp marker = llvm::cast<DiagnosticOp>(builder.create(state));

  if (createdMarker)
    *createdMarker = marker;

  if (plan.missingFallbackCoverage)
    return materializeMissingFallbackCoverageDiagnostic(builder, plan,
                                                       *selectedCase);

  return llvm::Error::success();
}

std::unique_ptr<::mlir::Pass> createSelectVariantsPass() {
  return std::make_unique<SelectVariantsPass>();
}

std::unique_ptr<::mlir::Pass>
createSelectVariantsPass(const ExtensionPluginRegistry &registry) {
  return std::make_unique<SelectVariantsPass>(registry);
}

} // namespace tianchenrv::transforms
