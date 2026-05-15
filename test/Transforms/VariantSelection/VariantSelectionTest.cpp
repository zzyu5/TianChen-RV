#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/Passes.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"
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

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>

using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantCostEstimate;
using tianchenrv::plugin::VariantCostRequest;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
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
    if (behavior == CostBehavior::Valid) {
      out.setScore(score);
      out.setExplicitPreference(true);
    }
    return llvm::Error::success();
  }

private:
  std::string name;
  double score;
  bool enabled;
  CostBehavior behavior;
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

class NoPreferencePlugin final : public ExtensionPlugin {
public:
  explicit NoPreferencePlugin(llvm::StringRef name) : name(name.str()) {}

  llvm::StringRef getName() const override { return name; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

private:
  std::string name;
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

unsigned countDirectDispatches(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return 0;

  unsigned count = 0;
  for (mlir::Operation &operation : kernel.getBody().front()) {
    if (llvm::isa<DispatchOp>(operation))
      ++count;
  }
  return count;
}

bool isSelectedPathMarker(DiagnosticOp diagnostic) {
  auto reason = diagnostic->getAttrOfType<mlir::StringAttr>("reason");
  return reason && reason.getValue() == "variant-selected" &&
         diagnostic->hasAttr("target") && diagnostic->hasAttr("selection_kind");
}

unsigned countDirectSelectedPathMarkers(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return 0;

  unsigned count = 0;
  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(operation);
    if (diagnostic && isSelectedPathMarker(diagnostic))
      ++count;
  }
  return count;
}

DiagnosticOp findDirectSelectedPathMarker(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return DiagnosticOp();

  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(operation);
    if (diagnostic && isSelectedPathMarker(diagnostic))
      return diagnostic;
  }
  return DiagnosticOp();
}

bool isMissingFallbackCoverageDiagnostic(DiagnosticOp diagnostic) {
  auto reason = diagnostic->getAttrOfType<mlir::StringAttr>("reason");
  return reason && reason.getValue() == "fallback-coverage-missing" &&
         diagnostic->hasAttr("selection_kind");
}

DiagnosticOp findDirectMissingFallbackCoverageDiagnostic(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return DiagnosticOp();

  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(operation);
    if (diagnostic && isMissingFallbackCoverageDiagnostic(diagnostic))
      return diagnostic;
  }
  return DiagnosticOp();
}

DispatchOp findDirectDispatch(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return DispatchOp();

  for (mlir::Operation &operation : kernel.getBody().front()) {
    if (auto dispatch = llvm::dyn_cast<DispatchOp>(operation))
      return dispatch;
  }
  return DispatchOp();
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

bool getBoolAttr(mlir::Operation *operation, llvm::StringRef attrName) {
  auto attr = operation->getAttrOfType<mlir::BoolAttr>(attrName);
  return attr && attr.getValue();
}

std::optional<std::int64_t> getIntegerAttr(mlir::Operation *operation,
                                           llvm::StringRef attrName) {
  auto attr = operation->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

std::optional<double> getFloatAttr(mlir::Operation *operation,
                                   llvm::StringRef attrName) {
  auto attr = operation->getAttrOfType<mlir::FloatAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getValueAsDouble();
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

  tcrv.exec.kernel @fallback_only_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
	    tcrv.exec.variant @only_path attributes {
	      fallback_role = "conservative",
	      origin = "only",
	      requires = [@generic_base]
    } {
    }
  }

  tcrv.exec.kernel @no_preference_tie_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @no_pref_a attributes {
      origin = "no-pref-a",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @no_pref_b attributes {
      origin = "no-pref-b",
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
  SelectionCostPlugin only("only", 3.0);
  NoPreferencePlugin noPrefA("no-pref-a");
  NoPreferencePlugin noPrefB("no-pref-b");
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
  if (int result = expectSuccess(registry.registerPlugin(only), "register only"))
    return result;
  if (int result =
          expectSuccess(registry.registerPlugin(noPrefA), "register no-pref-a"))
    return result;
  if (int result =
          expectSuccess(registry.registerPlugin(noPrefB), "register no-pref-b"))
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

  KernelOp fallbackOnlyKernel = findKernel(*module, "fallback_only_anchor");
  TargetCapabilitySet fallbackOnlyCapabilities =
      TargetCapabilitySet::buildFromKernel(fallbackOnlyKernel);
  auto fallbackOnlyPlanOrError =
      tianchenrv::transforms::planKernelVariantSelection(
          fallbackOnlyKernel, fallbackOnlyCapabilities, registry);
  if (!fallbackOnlyPlanOrError)
    return fail("fallback-only selection failed: " +
                llvm::toString(fallbackOnlyPlanOrError.takeError()));
  VariantSelectionPlan fallbackOnlyPlan = std::move(*fallbackOnlyPlanOrError);

  if (int result =
          expect(fallbackOnlyPlan.kind == VariantSelectionKind::FallbackOnly,
                 "single available path produces fallback-only plan"))
    return result;
  if (int result =
          expect(fallbackOnlyPlan.selectedVariant.getSymName() == "only_path",
                 "fallback-only plan records selected variant"))
    return result;

  KernelOp noPreferenceKernel = findKernel(*module, "no_preference_tie_anchor");
  auto noPreferencePlanOrError =
      tianchenrv::transforms::planKernelVariantSelection(
          noPreferenceKernel,
          TargetCapabilitySet::buildFromKernel(noPreferenceKernel), registry);
  if (!noPreferencePlanOrError)
    return fail("no-preference tie selection failed: " +
                llvm::toString(noPreferencePlanOrError.takeError()));
  VariantSelectionPlan noPreferencePlan = std::move(*noPreferencePlanOrError);
  if (int result =
          expect(noPreferencePlan.selectedVariant.getSymName() == "no_pref_a" &&
                     noPreferencePlan.rankedVariants.size() == 2 &&
                     !noPreferencePlan.rankedVariants[0]
                          .cost.hasExplicitPreference() &&
                     !noPreferencePlan.rankedVariants[1]
                          .cost.hasExplicitPreference(),
                 "missing preference hooks use target-neutral IR-order tie-break"))
    return result;

  mlir::PassManager passManager(&context);
  passManager.addPass(tianchenrv::transforms::createSelectVariantsPass(registry));
  if (int result =
          expect(mlir::succeeded(passManager.run(*module)),
                 "selection pass materializes selected-path markers"))
    return result;

  staticKernel = findKernel(*module, "static_anchor");
  DiagnosticOp staticMarker = findDirectSelectedPathMarker(staticKernel);
  if (int result = expect(countDirectDispatches(staticKernel) == 0,
                          "static marker pass does not create dispatch"))
    return result;
  if (int result =
          expect(countDirectSelectedPathMarkers(staticKernel) == 1,
                 "static marker pass creates one selected-path marker"))
    return result;
  if (int result =
          expect(getTarget(staticMarker.getOperation()) == "cheap_second",
                 "static marker targets cost-ranked selected variant"))
    return result;
  if (int result =
          expect(getStringAttr(staticMarker.getOperation(), "selection_kind") ==
                     "static-variant",
                 "static marker records generic selection kind"))
    return result;
  if (int result =
          expect(getStringAttr(staticMarker.getOperation(), "origin") ==
                         "cheap" &&
                     getBoolAttr(staticMarker.getOperation(),
                                 "preference_available") &&
                     getIntegerAttr(staticMarker.getOperation(),
                                    "preference_rank") == 0 &&
                     getFloatAttr(staticMarker.getOperation(),
                                  "preference_score") == 1.0 &&
                     getStringAttr(staticMarker.getOperation(),
                                   "preference_tie_break")
                         .contains("explicit plugin preference"),
                 "static marker records generic plugin preference metadata"))
    return result;

  tieKernel = findKernel(*module, "tie_anchor");
  DiagnosticOp tieMarker = findDirectSelectedPathMarker(tieKernel);
  if (int result = expect(getTarget(tieMarker.getOperation()) == "tie_a",
                          "tie marker preserves stable selected target"))
    return result;

  fallbackOnlyKernel = findKernel(*module, "fallback_only_anchor");
  DiagnosticOp fallbackOnlyMarker =
      findDirectSelectedPathMarker(fallbackOnlyKernel);
  if (int result =
          expect(getTarget(fallbackOnlyMarker.getOperation()) == "only_path",
                 "fallback-only marker targets the only selected path"))
    return result;
  if (int result = expect(getStringAttr(fallbackOnlyMarker.getOperation(),
                                        "selection_kind") == "fallback-only",
                          "fallback-only marker records generic selection kind"))
    return result;
  if (int result =
          expect(getStringAttr(fallbackOnlyMarker.getOperation(),
                               "fallback_role") == "conservative",
                 "fallback-only marker records generic fallback preference role"))
    return result;

  mlir::PassManager rerunPassManager(&context);
  rerunPassManager.addPass(
      tianchenrv::transforms::createSelectVariantsPass(registry));
  if (int result =
          expect(mlir::succeeded(rerunPassManager.run(*module)),
                 "selection pass selected-path markers are idempotent"))
    return result;
  staticKernel = findKernel(*module, "static_anchor");
  if (int result =
          expect(countDirectSelectedPathMarkers(staticKernel) == 1,
                 "selection rerun does not duplicate static selected marker"))
    return result;

  return 0;
}

int runFallbackRoleTieBreakTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @fallback_tie_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @fallback_ir_first attributes {
      fallback_role = "conservative",
      origin = "fallback-tie",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @non_fallback_ir_second attributes {
      origin = "non-fallback-tie",
      policy = "opaque_non_fallback_tie",
      requires = [@generic_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse fallback role tie-break module");

  SelectionCostPlugin fallbackTie("fallback-tie", 4.0);
  SelectionCostPlugin nonFallbackTie("non-fallback-tie", 4.0);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(fallbackTie),
                                 "register fallback-tie"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(nonFallbackTie),
                                 "register non-fallback-tie"))
    return result;

  KernelOp kernel = findKernel(*module, "fallback_tie_anchor");
  auto planOrError = tianchenrv::transforms::planKernelVariantSelection(
      kernel, TargetCapabilitySet::buildFromKernel(kernel), registry);
  if (!planOrError)
    return fail("fallback tie selection failed: " +
                llvm::toString(planOrError.takeError()));

  VariantSelectionPlan plan = std::move(*planOrError);
  if (int result = expect(plan.selectedVariant.getSymName() ==
                              "non_fallback_ir_second",
                          "equal preference tie ranks conservative fallback after non-fallback"))
    return result;
  if (int result =
          expect(plan.kind == VariantSelectionKind::StaticVariant &&
                     plan.fallback.getSymName() == "fallback_ir_first" &&
                     plan.dispatchCases.empty(),
                 "printable policy metadata preserves fallback ranking without "
                 "creating runtime dispatch"))
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
	      fallback_role = "conservative",
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
	      fallback_role = "conservative",
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
  if (int result =
          expect(getBoolAttr(cases[0].getOperation(),
                             "runtime_guard_required") &&
                     getBoolAttr(cases[1].getOperation(),
                                 "runtime_guard_required") &&
                     getBoolAttr(cases[2].getOperation(),
                                 "runtime_guard_required"),
                 "runtime dispatch cases carry typed guard requirement metadata"))
    return result;
  if (int result =
          expect(getStringAttr(cases[0].getOperation(), "origin") ==
                         "guarded-fast" &&
                     getBoolAttr(cases[0].getOperation(),
                                 "preference_available") &&
                     getIntegerAttr(cases[0].getOperation(),
                                    "preference_rank") == 0 &&
                     getFloatAttr(cases[0].getOperation(),
                                  "preference_score") == 0.5,
                 "dispatch case records generic plugin preference metadata"))
    return result;
  if (int result =
          expect(getStringAttr(fallback.getOperation(), "origin") ==
                         "cheap-fallback" &&
                     getStringAttr(fallback.getOperation(), "fallback_role") ==
                         "conservative" &&
                     getIntegerAttr(fallback.getOperation(),
                                    "preference_rank") == 3,
                 "dispatch fallback records generic fallback preference metadata"))
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

int runInjectedRegistrySelectionPassTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @pass_injected_registry attributes {} {
    tcrv.exec.capability @runtime_probe {
      id = "runtime.probe",
      kind = "runtime-offload",
      status = "missing"
    }
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @guarded_fast attributes {
      condition = "runtime_probe_available",
      origin = "guarded-fast",
      requires = [@runtime_probe]
    } {
    }
	    tcrv.exec.variant @portable_fallback attributes {
	      fallback_role = "conservative",
	      origin = "portable-fallback",
	      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @available_slow attributes {
      origin = "available-slow",
      requires = [@generic_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse injected selection pass module");

  SelectionCostPlugin guardedFast("guarded-fast", 0.5);
  SelectionCostPlugin portableFallback("portable-fallback", 2.0);
  SelectionCostPlugin availableSlow("available-slow", 5.0);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(guardedFast),
                                 "register pass guarded fast"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(portableFallback),
                                 "register pass portable fallback"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(availableSlow),
                                 "register pass available slow"))
    return result;

  KernelOp kernel = findKernel(*module, "pass_injected_registry");
  unsigned variantsBefore = countDirectVariants(kernel);
  if (int result = expect(countDirectDispatches(kernel) == 0,
                          "injected selection pass starts without dispatch"))
    return result;

  mlir::PassManager passManager(&context);
  passManager.addPass(tianchenrv::transforms::createSelectVariantsPass(registry));
  passManager.addPass(tianchenrv::transforms::createCheckCapabilityRequiresPass());
  if (int result = expect(mlir::succeeded(passManager.run(*module)),
                          "injected registry selection pass succeeds"))
    return result;

  kernel = findKernel(*module, "pass_injected_registry");
  if (int result =
          expect(countDirectVariants(kernel) == variantsBefore,
                 "injected selection pass does not erase materialized variants"))
    return result;
  if (int result = expect(countDirectDispatches(kernel) == 1,
                          "injected selection pass materializes one dispatch"))
    return result;

  DispatchOp dispatch = findDirectDispatch(kernel);
  if (int result = expect(static_cast<bool>(dispatch),
                          "injected selection pass produced typed dispatch"))
    return result;

  llvm::SmallVector<DispatchCaseOp, 2> cases;
  FallbackOp fallback;
  for (mlir::Operation &operation : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(operation)) {
      cases.push_back(dispatchCase);
      continue;
    }
    if (auto fallbackCandidate = llvm::dyn_cast<FallbackOp>(operation))
      fallback = fallbackCandidate;
  }

  if (int result = expect(cases.size() == 1,
                          "injected selection pass keeps lower-cost guarded case"))
    return result;
  if (int result =
          expect(getTarget(cases[0].getOperation()) == "guarded_fast",
                 "injected selection pass dispatch case uses planned target"))
    return result;
  if (int result = expect(getStringAttr(cases[0].getOperation(), "condition") ==
                              "runtime_probe_available",
                          "injected selection pass copies generic condition"))
    return result;
  if (int result = expect(static_cast<bool>(fallback),
                          "injected selection pass emits typed fallback"))
    return result;
  if (int result =
          expect(getTarget(fallback.getOperation()) == "portable_fallback",
                 "injected selection pass chooses cost-ranked available fallback"))
    return result;

  return 0;
}

int runConflictAwareRuntimeDispatchSelectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @conflict_dispatch_anchor attributes {} {
    tcrv.exec.capability @fast_runtime {
      id = "generic.fast.runtime",
      kind = "runtime",
      conflicts = ["build.policy.disable_fast_runtime"],
      status = "available"
    }
    tcrv.exec.capability @disable_fast_profile {
      id = "generic.build.profile",
      kind = "build-policy",
      provides = ["build.policy.disable_fast_runtime"],
      status = "available"
    }
    tcrv.exec.capability @baseline_capability {
      id = "generic.baseline",
      kind = "toolchain",
      status = "available"
    }
    tcrv.exec.variant @conflicting_fast attributes {
      origin = "conflicting-fast",
      requires = [@fast_runtime]
    } {
    }
    tcrv.exec.variant @portable_fallback attributes {
      fallback_role = "conservative",
      origin = "portable-fallback",
      requires = [@baseline_capability]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse conflict-aware selection module");

  SelectionCostPlugin conflictingFast("conflicting-fast", 0.25);
  SelectionCostPlugin portableFallback("portable-fallback", 10.0);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(conflictingFast),
                                 "register conflicting fast"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(portableFallback),
                                 "register portable fallback"))
    return result;

  KernelOp kernel = findKernel(*module, "conflict_dispatch_anchor");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  auto planOrError = tianchenrv::transforms::planKernelVariantSelection(
      kernel, capabilities, registry);
  if (!planOrError)
    return fail("conflict-aware runtime dispatch planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan plan = std::move(*planOrError);

  if (int result = expect(plan.kind == VariantSelectionKind::RuntimeDispatch,
                          "conflicting available variant requires runtime dispatch"))
    return result;
  if (int result =
          expect(plan.fallback.getSymName() == "portable_fallback",
                 "conflict-aware dispatch uses conflict-free fallback"))
    return result;
  if (int result =
          expect(plan.dispatchCases.size() == 1 &&
                     plan.dispatchCases.front().variant.getSymName() ==
                         "conflicting_fast" &&
                     plan.dispatchCases.front().genericallyAvailable &&
                     !plan.dispatchCases.front().conflictFree &&
                     plan.dispatchCases.front().requiresRuntimeCapabilityGuard,
                 "conflicting available variant is retained as guarded case"))
    return result;

  mlir::OpBuilder builder(&context);
  DispatchOp createdDispatch;
  if (int result =
          expectSuccess(tianchenrv::transforms::materializeRuntimeDispatchPlan(
                            builder, plan, &createdDispatch),
                        "materialize conflict-aware runtime dispatch"))
    return result;

  llvm::SmallVector<DispatchCaseOp, 1> cases;
  FallbackOp fallback;
  for (mlir::Operation &operation : createdDispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(operation)) {
      cases.push_back(dispatchCase);
      continue;
    }
    if (auto fallbackCandidate = llvm::dyn_cast<FallbackOp>(operation))
      fallback = fallbackCandidate;
  }

  if (int result =
          expect(cases.size() == 1 &&
                     getTarget(cases.front().getOperation()) ==
                         "conflicting_fast",
                 "materialized dispatch case targets conflicting variant"))
    return result;
  if (int result =
          expect(getStringAttr(cases.front().getOperation(), "policy") ==
                     "capability_dispatch_guard",
                 "conflicting case receives synthesized generic policy guard"))
    return result;
  if (int result =
          expect(getBoolAttr(cases.front().getOperation(),
                             "runtime_guard_required"),
                 "conflicting case carries typed runtime guard requirement"))
    return result;
  if (int result =
          expect(fallback &&
                     getTarget(fallback.getOperation()) == "portable_fallback",
                 "materialized fallback remains conflict-free"))
    return result;

  mlir::PassManager passManager(&context);
  passManager.addPass(
      tianchenrv::transforms::createCheckCapabilityRequiresPass());
  if (int result =
          expect(mlir::succeeded(passManager.run(*module)),
                 "check-capability-requires accepts conflict-aware dispatch"))
    return result;

  return 0;
}

int runBuiltinRVVScalarFallbackSelectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_plus_scalar attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_typed_body attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %runtime_n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = tcrv_rvv.setvl %runtime_n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
      } : !tcrv_rvv.vl
    }
  }

  tcrv.exec.kernel @scalar_only attributes {} {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }

  tcrv.exec.kernel @rvv_only attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.variant @rvv_typed_body attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %runtime_n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = tcrv_rvv.setvl %runtime_n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse built-in RVV/scalar selection module");

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerBuiltinExtensionPlugins(
                            registry),
                        "register built-in RVV and scalar fallback plugins"))
    return result;
  ExtensionPluginRegistry scalarMaterializationRegistry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerScalarExtensionPlugin(
              scalarMaterializationRegistry),
          "register scalar fallback plugin for explicit typed RVV fixture"))
    return result;

  mlir::OpBuilder builder(&context);
  KernelOp rvvScalarKernel = findKernel(*module, "rvv_plus_scalar");
  TargetCapabilitySet rvvScalarCapabilities =
      TargetCapabilitySet::buildFromKernel(rvvScalarKernel);
  llvm::SmallVector<VariantOp, 2> materializedVariants;
  VariantProposalRequest rvvScalarRequest(rvvScalarKernel.getOperation(),
                                          rvvScalarKernel,
                                          rvvScalarCapabilities);
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, scalarMaterializationRegistry, rvvScalarRequest,
              &materializedVariants),
          "materialize scalar fallback beside explicit typed RVV body"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "built-in registry materializes scalar fallback while typed "
                 "RVV body authority is already explicit"))
    return result;

  VariantOp rvvVariant = findDirectVariant(rvvScalarKernel, "rvv_typed_body");
  VariantOp scalarVariant =
      findDirectVariant(rvvScalarKernel,
                        tianchenrv::plugin::scalar::
                            getScalarFallbackFirstSliceVariantName());
  if (int result = expect(rvvVariant && scalarVariant,
                          "materialized RVV and scalar variants are present"))
    return result;
  if (int result =
          expect(getStringAttr(rvvVariant.getOperation(), "origin") ==
                         tianchenrv::plugin::rvv::getRVVExtensionPluginName() &&
                     getStringAttr(scalarVariant.getOperation(), "origin") ==
                         tianchenrv::plugin::scalar::
                             getScalarExtensionPluginName(),
                 "materialized variants preserve plugin origins"))
    return result;
  if (int result =
          expect(getStringAttr(scalarVariant.getOperation(),
                               tianchenrv::plugin::kVariantFallbackRoleAttrName) ==
                     tianchenrv::plugin::kConservativeFallbackRoleValue,
                 "scalar fallback proposal materializes generic fallback role"))
    return result;
  if (int result = expectSuccess(
          registry.verifyKernelVariantLegality(rvvScalarKernel,
                                               rvvScalarCapabilities),
          "built-in materialized variants pass plugin legality"))
    return result;

  auto rvvScalarPlanOrError =
      tianchenrv::transforms::planKernelVariantSelection(
          rvvScalarKernel, rvvScalarCapabilities, registry);
  if (!rvvScalarPlanOrError)
    return fail("RVV plus scalar selection failed: " +
                llvm::toString(rvvScalarPlanOrError.takeError()));
  VariantSelectionPlan rvvScalarPlan = std::move(*rvvScalarPlanOrError);
  if (int result =
          expect(rvvScalarPlan.kind == VariantSelectionKind::StaticVariant &&
                     rvvScalarPlan.selectedVariant == rvvVariant &&
                     rvvScalarPlan.fallback == scalarVariant &&
                     rvvScalarPlan.dispatchCases.empty() &&
                     !rvvScalarPlan.missingFallbackCoverage,
                 "available RVV is selected statically while scalar remains "
                 "explicit fallback"))
    return result;
  if (int result =
          expect(rvvScalarPlan.rankedVariants.size() == 2 &&
                     rvvScalarPlan.rankedVariants[0].variant == rvvVariant &&
                     rvvScalarPlan.rankedVariants[0]
                         .cost.hasExplicitPreference() &&
                     rvvScalarPlan.rankedVariants[0].cost.getScore() == 1.0 &&
                     rvvScalarPlan.rankedVariants[1].variant == scalarVariant &&
                     rvvScalarPlan.rankedVariants[1]
                         .cost.hasExplicitPreference() &&
                     rvvScalarPlan.rankedVariants[1].cost.getScore() == 1000.0,
                 "RVV plugin preference outranks scalar fallback preference"))
    return result;
  DiagnosticOp rvvScalarMarker;
  if (int result =
          expectSuccess(tianchenrv::transforms::materializeSelectedVariantMarker(
                            builder, rvvScalarPlan, &rvvScalarMarker),
                        "materialize RVV/scalar selected marker"))
    return result;
  if (int result =
          expect(countDirectDispatches(rvvScalarKernel) == 0 &&
                     rvvScalarMarker &&
                     getTarget(rvvScalarMarker.getOperation()) ==
                         rvvVariant.getSymName() &&
                     getStringAttr(rvvScalarMarker.getOperation(),
                                   "selection_kind") == "static-variant" &&
                     !rvvScalarMarker->hasAttr("runtime_guard_required") &&
                     !rvvScalarMarker->hasAttr("runtime_guard"),
                 "available RVV printable annotations do not create runtime "
                 "dispatch or typed guard metadata"))
    return result;

  KernelOp scalarOnlyKernel = findKernel(*module, "scalar_only");
  TargetCapabilitySet scalarOnlyCapabilities =
      TargetCapabilitySet::buildFromKernel(scalarOnlyKernel);
  llvm::SmallVector<VariantOp, 1> scalarOnlyVariants;
  VariantProposalRequest scalarOnlyRequest(scalarOnlyKernel.getOperation(),
                                           scalarOnlyKernel,
                                           scalarOnlyCapabilities);
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, scalarMaterializationRegistry, scalarOnlyRequest,
              &scalarOnlyVariants),
          "materialize scalar-only fallback proposal"))
    return result;
  if (int result =
          expect(scalarOnlyVariants.size() == 1,
                 "scalar-only capability materializes one fallback variant"))
    return result;
  auto scalarOnlyPlanOrError =
      tianchenrv::transforms::planKernelVariantSelection(
          scalarOnlyKernel, scalarOnlyCapabilities, registry);
  if (!scalarOnlyPlanOrError)
    return fail("scalar-only selection failed: " +
                llvm::toString(scalarOnlyPlanOrError.takeError()));
  VariantSelectionPlan scalarOnlyPlan = std::move(*scalarOnlyPlanOrError);
  if (int result =
          expect(scalarOnlyPlan.kind == VariantSelectionKind::FallbackOnly &&
                     scalarOnlyPlan.selectedVariant == scalarOnlyVariants.front() &&
                     scalarOnlyPlan.fallback == scalarOnlyVariants.front() &&
                     !scalarOnlyPlan.missingFallbackCoverage,
                 "scalar-only capability produces fallback-only selected path"))
    return result;

  KernelOp rvvOnlyKernel = findKernel(*module, "rvv_only");
  TargetCapabilitySet rvvOnlyCapabilities =
      TargetCapabilitySet::buildFromKernel(rvvOnlyKernel);
  VariantOp rvvOnlyVariant = findDirectVariant(rvvOnlyKernel, "rvv_typed_body");
  if (int result =
          expect(rvvOnlyVariant,
                 "RVV-only typed body keeps one explicit RVV variant without "
                 "no-body synthesis"))
    return result;
  auto rvvOnlyPlanOrError = tianchenrv::transforms::planKernelVariantSelection(
      rvvOnlyKernel, rvvOnlyCapabilities, registry);
  if (!rvvOnlyPlanOrError)
    return fail("RVV-only selection failed: " +
                llvm::toString(rvvOnlyPlanOrError.takeError()));
  VariantSelectionPlan rvvOnlyPlan = std::move(*rvvOnlyPlanOrError);
  if (int result =
          expect(rvvOnlyPlan.kind == VariantSelectionKind::StaticVariant &&
                     rvvOnlyPlan.selectedVariant == rvvOnlyVariant &&
                     !rvvOnlyPlan.fallback &&
                     rvvOnlyPlan.missingFallbackCoverage,
                 "RVV-only plan selects RVV without inventing fallback"))
    return result;
  DiagnosticOp rvvOnlyMarker;
  if (int result =
          expectSuccess(tianchenrv::transforms::materializeSelectedVariantMarker(
                            builder, rvvOnlyPlan, &rvvOnlyMarker),
                        "materialize RVV-only selected marker"))
    return result;
  if (int result =
          expect(countDirectDispatches(rvvOnlyKernel) == 0,
                 "RVV-only selection does not materialize fallback-less dispatch"))
    return result;
  DiagnosticOp missingFallback =
      findDirectMissingFallbackCoverageDiagnostic(rvvOnlyKernel);
  if (int result =
          expect(missingFallback &&
                     getStringAttr(missingFallback.getOperation(),
                                   "selection_kind") ==
                         "missing-conservative-fallback",
                 "RVV-only selection records missing fallback diagnostic"))
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
	  }

  tcrv.exec.kernel @conflict_without_fallback_anchor attributes {} {
    tcrv.exec.capability @fast_runtime {
      id = "generic.fast.runtime",
      kind = "runtime",
      conflicts = ["build.policy.disable_fast_runtime"],
      status = "available"
    }
    tcrv.exec.capability @disable_fast_profile {
      id = "generic.build.profile",
      kind = "build-policy",
      provides = ["build.policy.disable_fast_runtime"],
      status = "available"
    }
    tcrv.exec.variant @conflicting_only attributes {
      origin = "conflicting-only",
      requires = [@fast_runtime]
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
  SelectionCostPlugin conflictingOnly("conflicting-only", 0.25);
  SelectionCostPlugin fallback("fallback", 1.0);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(guardedOnly),
                                 "register guarded-only"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(unguardedUnavailable),
                                 "register unguarded-unavailable"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(conflictingOnly),
                                 "register conflicting-only"))
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
          {"no plugin-provided conflict-free conservative fallback candidate",
           "no_fallback_anchor"}))
    return result;

  KernelOp unguardedKernel =
      findKernel(*module, "unguarded_unavailable_anchor");
  if (int result = expectPlanErrorContains(
          tianchenrv::transforms::planKernelVariantSelection(
              unguardedKernel,
              TargetCapabilitySet::buildFromKernel(unguardedKernel), registry),
          {"no plugin-provided conflict-free conservative fallback candidate",
           "unguarded_unavailable_anchor"}))
    return result;

  KernelOp conflictWithoutFallbackKernel =
      findKernel(*module, "conflict_without_fallback_anchor");
  if (int result = expectPlanErrorContains(
          tianchenrv::transforms::planKernelVariantSelection(
              conflictWithoutFallbackKernel,
              TargetCapabilitySet::buildFromKernel(conflictWithoutFallbackKernel),
              registry),
          {"no plugin-provided conflict-free conservative fallback candidate",
           "conflict_without_fallback_anchor"}))
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
	      fallback_role = "conservative",
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
  VariantCostEstimate otherEstimate;
  otherEstimate.setScore(0.25);
  otherEstimate.setExplicitPreference(true);
  otherEstimate.setOriginPlugin("other");
  otherEstimate.setVariantSymbol("other_path");
  crossKernelPlan.rankedVariants.push_back(
      tianchenrv::transforms::VariantSelectionCase{
          crossKernelPlan.fallback, otherEstimate, 0, true, true, true, false});
  crossKernelPlan.dispatchCases.push_back(
      tianchenrv::transforms::VariantSelectionCase{
          findDirectVariant(kernel, "guarded_path"),
          VariantCostEstimate(), 0, false, false, true, true});

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
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerBuiltinExtensionPlugins(
                            dialectPlugins),
                        "register built-in plugin dialects"))
    return result;
  tianchenrv::registerPluginDialects(dialectPlugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runStaticAndTieSelectionTest(context))
    return result;
  if (int result = runFallbackRoleTieBreakTest(context))
    return result;
  if (int result = runRuntimeDispatchPlanningAndMaterializationTest(context))
    return result;
  if (int result = runInjectedRegistrySelectionPassTest(context))
    return result;
  if (int result = runConflictAwareRuntimeDispatchSelectionTest(context))
    return result;
  if (int result = runBuiltinRVVScalarFallbackSelectionTest(context))
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
