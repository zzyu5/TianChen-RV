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
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kTargetAttrName("target");

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::VariantCostRankingEntry;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

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

void copyStringAttrIfPresent(mlir::OperationState &state, VariantOp variant,
                             llvm::StringRef attrName) {
  auto attr = variant->getAttrOfType<mlir::StringAttr>(attrName);
  if (attr && !attr.getValue().trim().empty())
    state.addAttribute(attrName, attr);
}

llvm::Expected<bool> areRequiredCapabilitiesGenericallyAvailable(
    KernelOp kernel, VariantOp variant,
    const TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeSelectionError(
        kernel, variant,
        "requires structured array attribute 'requires' containing capability "
        "symbol references");

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

    if (!capability->isAvailable())
      return false;
  }

  return true;
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

DispatchCaseOp createDispatchCase(mlir::OpBuilder &builder,
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
  return llvm::cast<DispatchCaseOp>(builder.create(state));
}

FallbackOp createFallback(mlir::OpBuilder &builder, VariantOp variant) {
  mlir::OperationState state(variant.getLoc(), FallbackOp::getOperationName());
  state.addAttribute(kTargetAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  return llvm::cast<FallbackOp>(builder.create(state));
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
    llvm::Expected<VariantSelectionPlan> planOrError =
        planKernelVariantSelection(kernel, *registry);
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

  std::optional<std::size_t> fallbackIndex;
  for (const VariantCostRankingEntry &entry : rankedCosts) {
    if (llvm::Error error = validateOriginOwnedCost(kernel, entry))
      return std::move(error);

    llvm::Expected<bool> availability =
        areRequiredCapabilitiesGenericallyAvailable(kernel, entry.variant,
                                                    capabilities);
    if (!availability)
      return availability.takeError();

    bool hasDecisionMetadata = hasGenericDecisionMetadata(entry.variant);
    if (!*availability && !hasDecisionMetadata)
      return makeSelectionError(
          kernel, entry.variant,
          "unavailable variant lacks generic condition/guard/policy metadata "
          "and cannot be selected or retained by runtime dispatch");

    if (*availability && !fallbackIndex)
      fallbackIndex = plan.rankedVariants.size();

    plan.rankedVariants.push_back(VariantSelectionCase{
        entry.variant, entry.estimate, entry.originalIndex, *availability,
        hasDecisionMetadata});
  }

  if (!fallbackIndex)
    return makeSelectionError(
        kernel,
        "no direct variant is generically available as selection fallback "
        "under the supplied TargetCapabilitySet");

  const VariantSelectionCase &fallbackCase =
      plan.rankedVariants[*fallbackIndex];
  plan.selectedVariant = fallbackCase.variant;
  plan.fallback = fallbackCase.variant;

  for (std::size_t index = 0; index < plan.rankedVariants.size(); ++index) {
    if (index == *fallbackIndex)
      continue;

    const VariantSelectionCase &candidate = plan.rankedVariants[index];
    if (!candidate.hasGenericDecisionMetadata)
      continue;

    if (index < *fallbackIndex || !candidate.genericallyAvailable)
      plan.dispatchCases.push_back(candidate);
  }

  if (!plan.dispatchCases.empty()) {
    plan.kind = VariantSelectionKind::RuntimeDispatch;
    return plan;
  }

  if (plan.rankedVariants.size() == 1) {
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

  if (llvm::Error error = validatePlanVariant(kernel, plan.fallback, "fallback"))
    return error;

  if (plan.dispatchCases.empty())
    return makeSelectionError(
        kernel, "runtime dispatch plan requires at least one dispatch case");

  llvm::StringSet<> seenCaseTargets;
  for (const VariantSelectionCase &selectionCase : plan.dispatchCases) {
    if (llvm::Error error =
            validatePlanVariant(kernel, selectionCase.variant, "dispatch case"))
      return error;

    if (selectionCase.variant == plan.fallback)
      return makeSelectionError(
          kernel, selectionCase.variant,
          "dispatch case must not duplicate the fallback target");

    if (!hasGenericDecisionMetadata(selectionCase.variant))
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
    createDispatchCase(builder, selectionCase);
  createFallback(builder, plan.fallback);

  if (createdDispatch)
    *createdDispatch = dispatch;

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
