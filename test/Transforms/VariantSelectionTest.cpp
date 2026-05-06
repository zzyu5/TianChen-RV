#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/Passes.h"
#include "TianChenRV/Transforms/VariantSelection.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>

using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantCostEstimate;
using tianchenrv::plugin::VariantCostRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::transforms::VariantSelectionKind;
using tianchenrv::transforms::VariantSelectionPlan;

namespace {

enum class CostBehavior {
  Valid,
  Failure,
  MissingScore,
};

class SelectionCostPlugin final : public ExtensionPlugin {
public:
  SelectionCostPlugin(llvm::StringRef name, double score, bool enabled = true,
                      CostBehavior behavior = CostBehavior::Valid)
      : name(name.str()), score(score), enabled(enabled), behavior(behavior) {}

  llvm::StringRef getName() const override { return name; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return enabled; }

  llvm::Error estimateVariantCost(const VariantCostRequest &request,
                                  VariantCostEstimate &out) const override {
    if (behavior == CostBehavior::Failure)
      return llvm::make_error<llvm::StringError>(
          "plugin-local selection cost failed", llvm::errc::invalid_argument);

    out = VariantCostEstimate();
    out.setOriginPlugin(name);
    out.setVariantSymbol(request.getVariant().getSymName());
    if (behavior == CostBehavior::Valid)
      out.setScore(score);
    return llvm::Error::success();
  }

private:
  std::string name;
  double score;
  bool enabled;
  CostBehavior behavior;
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expect(bool condition, llvm::Twine message) {
  if (condition)
    return 0;
  return fail(message);
}

int expectSuccess(llvm::Error error, llvm::Twine context) {
  if (!error)
    return 0;

  std::string message = llvm::toString(std::move(error));
  return fail(context + ": " + message);
}

int expectErrorContains(llvm::Error error,
                        std::initializer_list<llvm::StringRef> fragments) {
  if (!error)
    return fail("expected variant selection error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("variant selection error text missing '") +
                  fragment + "': " + message);
  }
  return 0;
}

int expectPlanErrorContains(
    llvm::Expected<VariantSelectionPlan> expected,
    std::initializer_list<llvm::StringRef> fragments) {
  if (expected)
    return fail("expected variant selection plan failure");
  return expectErrorContains(expected.takeError(), fragments);
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

KernelOp findKernel(mlir::ModuleOp module, llvm::StringRef name) {
  KernelOp kernel;
  module->walk([&](KernelOp candidate) {
    if (candidate.getSymName() == name)
      kernel = candidate;
  });
  return kernel;
}

VariantOp findDirectVariant(KernelOp kernel, llvm::StringRef name) {
  if (!kernel || kernel.getBody().empty())
    return VariantOp();

  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<VariantOp>(operation);
    if (variant && variant.getSymName() == name)
      return variant;
  }
  return VariantOp();
}

unsigned countDirectVariants(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return 0;

  unsigned count = 0;
  for (mlir::Operation &operation : kernel.getBody().front()) {
    if (llvm::isa<VariantOp>(operation))
      ++count;
  }
  return count;
}

llvm::StringRef getTarget(mlir::Operation *operation) {
  auto target = operation->getAttrOfType<mlir::FlatSymbolRefAttr>("target");
  if (!target)
    return {};
  return target.getValue();
}

llvm::StringRef getStringAttr(mlir::Operation *operation,
                              llvm::StringRef attrName) {
  auto attr = operation->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return {};
  return attr.getValue();
}

int runStaticAndTieSelectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @static_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @expensive_first attributes {
      origin = "expensive",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @cheap_second attributes {
      origin = "cheap",
      requires = [@generic_base]
    } {
    }
  }

  tcrv.exec.kernel @tie_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @tie_a attributes {
      origin = "tie-a",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @tie_b attributes {
      origin = "tie-b",
      requires = [@generic_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse static selection module");

  SelectionCostPlugin expensive("expensive", 9.0);
  SelectionCostPlugin cheap("cheap", 1.0);
  SelectionCostPlugin tieA("tie-a", 2.0);
  SelectionCostPlugin tieB("tie-b", 2.0);
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(expensive), "register expensive"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(cheap), "register cheap"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(tieB), "register tie-b"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(tieA), "register tie-a"))
    return result;

  KernelOp staticKernel = findKernel(*module, "static_anchor");
  TargetCapabilitySet staticCapabilities =
      TargetCapabilitySet::buildFromKernel(staticKernel);
  auto staticPlanOrError = tianchenrv::transforms::planKernelVariantSelection(
      staticKernel, staticCapabilities, registry);
  if (!staticPlanOrError)
    return fail("static selection failed: " +
                llvm::toString(staticPlanOrError.takeError()));
  VariantSelectionPlan staticPlan = std::move(*staticPlanOrError);

  if (int result = expect(staticPlan.kind == VariantSelectionKind::StaticVariant,
                          "lowest-cost available plan is static"))
    return result;
  if (int result =
          expect(staticPlan.selectedVariant.getSymName() == "cheap_second",
                 "static selection chooses lowest-cost available variant, not first IR variant"))
    return result;
  if (int result = expect(staticPlan.dispatchCases.empty(),
                          "static selection does not retain dispatch cases"))
    return result;

  KernelOp tieKernel = findKernel(*module, "tie_anchor");
  TargetCapabilitySet tieCapabilities =
      TargetCapabilitySet::buildFromKernel(tieKernel);
  auto tiePlanOrError = tianchenrv::transforms::planKernelVariantSelection(
      tieKernel, tieCapabilities, registry);
  if (!tiePlanOrError)
    return fail("tie selection failed: " +
                llvm::toString(tiePlanOrError.takeError()));
  VariantSelectionPlan tiePlan = std::move(*tiePlanOrError);

  if (int result = expect(tiePlan.selectedVariant.getSymName() == "tie_a",
                          "equal-cost static selection preserves IR order"))
    return result;
  if (int result = expect(tiePlan.rankedVariants.size() == 2 &&
                              tiePlan.rankedVariants[0].variant.getSymName() ==
                                  "tie_a" &&
                              tiePlan.rankedVariants[1].variant.getSymName() ==
                                  "tie_b",
                          "equal-cost ranking remains visible in plan order"))
    return result;

  return 0;
}

int runRuntimeDispatchPlanningAndMaterializationTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @runtime_anchor attributes {} {
    tcrv.exec.capability @generic_probe {
      id = "generic.probe",
      kind = "runtime",
      status = "missing"
    }
    tcrv.exec.capability @generic_baseline_a {
      id = "generic.baseline.a",
      kind = "toolchain"
    }
    tcrv.exec.capability @generic_baseline_b {
      id = "generic.baseline.b",
      kind = "toolchain"
    }
    tcrv.exec.variant @expensive_ir_first_fallback attributes {
      origin = "expensive-fallback",
      requires = [@generic_baseline_a]
    } {
    }
    tcrv.exec.variant @guarded_fast attributes {
      condition = "opaque(condition:fast)",
      guard = "opaque_guard_fast",
      origin = "guarded-fast",
      policy = "opaque_policy_fast",
      requires = [@generic_probe]
    } {
    }
    tcrv.exec.variant @guarded_tie_a attributes {
      guard = "opaque_guard_tie_a",
      origin = "guarded-tie-a",
      requires = [@generic_probe]
    } {
    }
    tcrv.exec.variant @guarded_tie_b attributes {
      policy = "opaque_policy_tie_b",
      origin = "guarded-tie-b",
      requires = [@generic_probe]
    } {
    }
    tcrv.exec.variant @cheap_ranked_fallback attributes {
      origin = "cheap-fallback",
      requires = [@generic_baseline_b]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse runtime dispatch selection module");

  SelectionCostPlugin expensiveFallback("expensive-fallback", 9.0);
  SelectionCostPlugin guardedFast("guarded-fast", 0.5);
  SelectionCostPlugin guardedTieA("guarded-tie-a", 1.0);
  SelectionCostPlugin guardedTieB("guarded-tie-b", 1.0);
  SelectionCostPlugin cheapFallback("cheap-fallback", 2.0);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(expensiveFallback),
                                 "register expensive fallback"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(guardedFast),
                                 "register guarded fast"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(guardedTieB),
                                 "register guarded tie b"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(guardedTieA),
                                 "register guarded tie a"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(cheapFallback),
                                 "register cheap fallback"))
    return result;

  KernelOp kernel = findKernel(*module, "runtime_anchor");
  unsigned variantsBefore = countDirectVariants(kernel);
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  auto planOrError = tianchenrv::transforms::planKernelVariantSelection(
      kernel, capabilities, registry);
  if (!planOrError)
    return fail("runtime dispatch planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan plan = std::move(*planOrError);

  if (int result = expect(plan.kind == VariantSelectionKind::RuntimeDispatch,
                          "guarded lower-cost variants produce runtime dispatch"))
    return result;
  if (int result =
          expect(plan.fallback.getSymName() == "cheap_ranked_fallback",
                 "runtime dispatch fallback is chosen by cost ranking, not IR order"))
    return result;
  if (int result = expect(plan.dispatchCases.size() == 3,
                          "runtime dispatch retains lower-cost guarded cases"))
    return result;
  if (int result =
          expect(plan.dispatchCases[0].variant.getSymName() == "guarded_fast" &&
                     plan.dispatchCases[1].variant.getSymName() ==
                         "guarded_tie_a" &&
                     plan.dispatchCases[2].variant.getSymName() ==
                         "guarded_tie_b",
                 "dispatch case plan follows cost ranking with stable ties"))
    return result;

  mlir::OpBuilder builder(&context);
  DispatchOp createdDispatch;
  if (int result =
          expectSuccess(tianchenrv::transforms::materializeRuntimeDispatchPlan(
                            builder, plan, &createdDispatch),
                        "materialize runtime dispatch plan"))
    return result;
  if (int result = expect(static_cast<bool>(createdDispatch),
                          "materialization returns typed dispatch op"))
    return result;
  if (int result = expect(countDirectVariants(kernel) == variantsBefore,
                          "dispatch materialization does not erase variants"))
    return result;

  llvm::SmallVector<DispatchCaseOp, 4> cases;
  FallbackOp fallback;
  for (mlir::Operation &operation : createdDispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(operation)) {
      cases.push_back(dispatchCase);
      continue;
    }
    if (auto fallbackCandidate = llvm::dyn_cast<FallbackOp>(operation))
      fallback = fallbackCandidate;
  }

  if (int result = expect(cases.size() == 3,
                          "materialized dispatch has typed case ops"))
    return result;
  if (int result = expect(static_cast<bool>(fallback),
                          "materialized dispatch has typed fallback op"))
    return result;
  if (int result = expect(getTarget(cases[0].getOperation()) == "guarded_fast" &&
                              getTarget(cases[1].getOperation()) ==
                                  "guarded_tie_a" &&
                              getTarget(cases[2].getOperation()) ==
                                  "guarded_tie_b",
                          "materialized dispatch preserves planned case order"))
    return result;
  if (int result = expect(getTarget(fallback.getOperation()) ==
                              "cheap_ranked_fallback",
                          "materialized fallback uses planned fallback target"))
    return result;
  if (int result = expect(getStringAttr(cases[0].getOperation(), "condition") ==
                              "opaque(condition:fast)",
                          "case copies opaque condition metadata verbatim"))
    return result;
  if (int result = expect(getStringAttr(cases[0].getOperation(), "guard") ==
                              "opaque_guard_fast",
                          "case copies opaque guard metadata verbatim"))
    return result;
  if (int result = expect(getStringAttr(cases[0].getOperation(), "policy") ==
                              "opaque_policy_fast",
                          "case copies opaque policy metadata verbatim"))
    return result;
  if (int result = expect(getStringAttr(cases[1].getOperation(), "guard") ==
                              "opaque_guard_tie_a",
                          "case with guard-only metadata remains guarded"))
    return result;
  if (int result = expect(getStringAttr(cases[2].getOperation(), "policy") ==
                              "opaque_policy_tie_b",
                          "case with policy-only metadata remains guarded"))
    return result;

  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "materialized dispatch verifies"))
    return result;

  mlir::PassManager passManager(&context);
  passManager.addPass(
      tianchenrv::transforms::createCheckCapabilityRequiresPass());
  if (int result =
          expect(mlir::succeeded(passManager.run(*module)),
                 "check-capability-requires accepts guarded dispatch and available fallback"))
    return result;

  return 0;
}

int runNoViableAndUnavailableNegativeTests(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @empty_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
  }

  tcrv.exec.kernel @no_fallback_anchor attributes {} {
    tcrv.exec.capability @generic_probe {
      id = "generic.probe",
      kind = "runtime",
      status = "missing"
    }
    tcrv.exec.variant @guarded_only attributes {
      condition = "opaque_runtime_probe",
      origin = "guarded-only",
      requires = [@generic_probe]
    } {
    }
  }

  tcrv.exec.kernel @unguarded_unavailable_anchor attributes {} {
    tcrv.exec.capability @generic_missing {
      id = "generic.missing",
      kind = "runtime",
      status = "missing"
    }
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @unguarded_unavailable attributes {
      origin = "unguarded-unavailable",
      requires = [@generic_missing]
    } {
    }
    tcrv.exec.variant @fallback_path attributes {
      origin = "fallback",
      requires = [@generic_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse negative selection module");

  SelectionCostPlugin guardedOnly("guarded-only", 1.0);
  SelectionCostPlugin unguardedUnavailable("unguarded-unavailable", 0.5);
  SelectionCostPlugin fallback("fallback", 1.0);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(guardedOnly),
                                 "register guarded-only"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(unguardedUnavailable),
                                 "register unguarded-unavailable"))
    return result;
  if (int result =
          expectSuccess(registry.registerPlugin(fallback), "register fallback"))
    return result;

  KernelOp emptyKernel = findKernel(*module, "empty_anchor");
  auto emptyPlanOrError = tianchenrv::transforms::planKernelVariantSelection(
      emptyKernel, TargetCapabilitySet::buildFromKernel(emptyKernel), registry);
  if (!emptyPlanOrError)
    return fail("empty kernel selection failed: " +
                llvm::toString(emptyPlanOrError.takeError()));
  if (int result =
          expect(emptyPlanOrError->kind == VariantSelectionKind::NoViableVariant,
                 "kernel with no direct variants returns NoViableVariant"))
    return result;

  KernelOp noFallbackKernel = findKernel(*module, "no_fallback_anchor");
  if (int result = expectPlanErrorContains(
          tianchenrv::transforms::planKernelVariantSelection(
              noFallbackKernel,
              TargetCapabilitySet::buildFromKernel(noFallbackKernel), registry),
          {"no direct variant is generically available", "no_fallback_anchor"}))
    return result;

  KernelOp unguardedKernel =
      findKernel(*module, "unguarded_unavailable_anchor");
  if (int result = expectPlanErrorContains(
          tianchenrv::transforms::planKernelVariantSelection(
              unguardedKernel,
              TargetCapabilitySet::buildFromKernel(unguardedKernel), registry),
          {"unavailable variant lacks generic condition/guard/policy metadata",
           "unguarded_unavailable", "unguarded_unavailable_anchor"}))
    return result;

  return 0;
}

int runMaterializationNegativeTests(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @existing_dispatch_anchor attributes {} {
    tcrv.exec.capability @generic_probe {
      id = "generic.probe",
      kind = "runtime",
      status = "missing"
    }
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @guarded_path attributes {
      policy = "opaque_runtime_policy",
      origin = "guarded",
      requires = [@generic_probe]
    } {
    }
    tcrv.exec.variant @fallback_path attributes {
      origin = "fallback",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.dispatch attributes {} {
      tcrv.exec.case @guarded_path {policy = "preexisting_policy"}
      tcrv.exec.fallback @fallback_path
    }
  }

  tcrv.exec.kernel @other_anchor attributes {} {
    tcrv.exec.capability @generic_other {
      id = "generic.other",
      kind = "toolchain"
    }
    tcrv.exec.variant @other_path attributes {
      policy = "opaque_other_policy",
      origin = "other",
      requires = [@generic_other]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse materialization negative module");

  SelectionCostPlugin guarded("guarded", 0.5);
  SelectionCostPlugin fallback("fallback", 1.0);
  SelectionCostPlugin other("other", 0.25);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(guarded),
                                 "register guarded"))
    return result;
  if (int result =
          expectSuccess(registry.registerPlugin(fallback), "register fallback"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(other), "register other"))
    return result;

  KernelOp kernel = findKernel(*module, "existing_dispatch_anchor");
  auto planOrError = tianchenrv::transforms::planKernelVariantSelection(
      kernel, TargetCapabilitySet::buildFromKernel(kernel), registry);
  if (!planOrError)
    return fail("existing dispatch planning failed: " +
                llvm::toString(planOrError.takeError()));

  mlir::OpBuilder builder(&context);
  if (int result = expectErrorContains(
          tianchenrv::transforms::materializeRuntimeDispatchPlan(
              builder, *planOrError),
          {"already contains a direct tcrv.exec.dispatch",
           "existing_dispatch_anchor"}))
    return result;

  VariantSelectionPlan crossKernelPlan;
  KernelOp otherKernel = findKernel(*module, "other_anchor");
  crossKernelPlan.kind = VariantSelectionKind::RuntimeDispatch;
  crossKernelPlan.kernel = otherKernel;
  crossKernelPlan.fallback = findDirectVariant(otherKernel, "other_path");
  crossKernelPlan.dispatchCases.push_back(
      tianchenrv::transforms::VariantSelectionCase{
          findDirectVariant(kernel, "guarded_path"),
          VariantCostEstimate(), 0, true, true});

  if (int result = expectErrorContains(
          tianchenrv::transforms::materializeRuntimeDispatchPlan(
              builder, crossKernelPlan),
          {"dispatch case variant is not a direct child",
           "other_anchor", "guarded_path"}))
    return result;

  VariantSelectionPlan missingKernelPlan;
  missingKernelPlan.kind = VariantSelectionKind::RuntimeDispatch;
  if (int result = expectErrorContains(
          tianchenrv::transforms::materializeRuntimeDispatchPlan(
              builder, missingKernelPlan),
          {"requires a tcrv.exec.kernel", "<missing>"}))
    return result;

  return 0;
}

int runCostFailurePropagationTests(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @cost_failure_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @unknown_origin attributes {
      origin = "missing-plugin",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @disabled_origin attributes {
      origin = "disabled",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @failing_origin attributes {
      origin = "failing",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @invalid_origin attributes {
      origin = "invalid",
      requires = [@generic_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse cost failure propagation module");

  KernelOp kernel = findKernel(*module, "cost_failure_anchor");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  {
    ExtensionPluginRegistry registry;
    if (int result = expectPlanErrorContains(
            tianchenrv::transforms::planKernelVariantSelection(
                kernel, capabilities, registry),
            {"unknown origin plugin 'missing-plugin'", "unknown_origin",
             "cost_failure_anchor"}))
      return result;
  }

  {
    mlir::Attribute oldOrigin =
        findDirectVariant(kernel, "unknown_origin")->getAttr("origin");
    findDirectVariant(kernel, "unknown_origin")
        ->setAttr("origin", mlir::StringAttr::get(&context, "disabled"));
    SelectionCostPlugin disabled("disabled", 1.0, false);
    ExtensionPluginRegistry registry;
    if (int result =
            expectSuccess(registry.registerPlugin(disabled), "register disabled"))
      return result;
    if (int result = expectPlanErrorContains(
            tianchenrv::transforms::planKernelVariantSelection(
                kernel, capabilities, registry),
            {"origin plugin 'disabled' is disabled", "unknown_origin",
             "cost_failure_anchor"}))
      return result;
    findDirectVariant(kernel, "unknown_origin")->setAttr("origin", oldOrigin);
  }

  {
    mlir::Attribute oldOrigin =
        findDirectVariant(kernel, "unknown_origin")->getAttr("origin");
    findDirectVariant(kernel, "unknown_origin")
        ->setAttr("origin", mlir::StringAttr::get(&context, "failing"));
    SelectionCostPlugin failing("failing", 1.0, true, CostBehavior::Failure);
    ExtensionPluginRegistry registry;
    if (int result =
            expectSuccess(registry.registerPlugin(failing), "register failing"))
      return result;
    if (int result = expectPlanErrorContains(
            tianchenrv::transforms::planKernelVariantSelection(
                kernel, capabilities, registry),
            {"origin plugin 'failing' failed cost estimate", "unknown_origin",
             "cost_failure_anchor", "plugin-local selection cost failed"}))
      return result;
    findDirectVariant(kernel, "unknown_origin")->setAttr("origin", oldOrigin);
  }

  {
    mlir::Attribute oldOrigin =
        findDirectVariant(kernel, "unknown_origin")->getAttr("origin");
    findDirectVariant(kernel, "unknown_origin")
        ->setAttr("origin", mlir::StringAttr::get(&context, "invalid"));
    SelectionCostPlugin invalid("invalid", 1.0, true,
                                CostBehavior::MissingScore);
    ExtensionPluginRegistry registry;
    if (int result =
            expectSuccess(registry.registerPlugin(invalid), "register invalid"))
      return result;
    if (int result = expectPlanErrorContains(
            tianchenrv::transforms::planKernelVariantSelection(
                kernel, capabilities, registry),
            {"produced invalid cost estimate", "score is missing",
             "unknown_origin", "cost_failure_anchor"}))
      return result;
    findDirectVariant(kernel, "unknown_origin")->setAttr("origin", oldOrigin);
  }

  {
    SelectionCostPlugin disabled("disabled", 1.0, false);
    ExtensionPluginRegistry registry;
    if (int result =
            expectSuccess(registry.registerPlugin(disabled), "register disabled"))
      return result;
    if (int result = expectPlanErrorContains(
            tianchenrv::transforms::planKernelVariantSelection(
                KernelOp(), registry),
            {"requires a tcrv.exec.kernel", "<missing>"}))
      return result;
  }

  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runStaticAndTieSelectionTest(context))
    return result;
  if (int result = runRuntimeDispatchPlanningAndMaterializationTest(context))
    return result;
  if (int result = runNoViableAndUnavailableNegativeTests(context))
    return result;
  if (int result = runMaterializationNegativeTests(context))
    return result;
  if (int result = runCostFailurePropagationTests(context))
    return result;

  llvm::outs() << "variant selection planning smoke test passed\n";
  return 0;
}
