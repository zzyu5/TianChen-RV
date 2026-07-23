#include "Weft/InitWeftDialects.h"
#include "Weft/Plugin/BuiltinExtensionPlugins.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"
#include "Weft/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Transforms/Passes.h"
#include "Weft/Transforms/VariantMaterialization.h"
#include "Weft/Transforms/VariantSelection.h"

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
#include <utility>

using weft::plugin::ExtensionPlugin;
using weft::plugin::ExtensionBundleRegistry;
using weft::plugin::ExtensionPluginRegistry;
using weft::plugin::PluginCapability;
using weft::plugin::VariantCostEstimate;
using weft::plugin::VariantCostRequest;
using weft::plugin::VariantProposalRequest;
using weft::support::TargetCapabilitySet;
using weft::exec::DiagnosticOp;
using weft::exec::DispatchCaseOp;
using weft::exec::DispatchOp;
using weft::exec::FallbackOp;
using weft::exec::KernelOp;
using weft::exec::VariantOp;
using weft::transforms::VariantSelectionKind;
using weft::transforms::VariantSelectionPlan;

namespace {

enum class CostBehavior {
  Valid,
  Failure,
  MissingScore,
};

class SelectionCostPlugin final : public ExtensionPlugin {
public:
  SelectionCostPlugin(llvm::StringRef name, double score, bool enabled = true,
                      CostBehavior behavior = CostBehavior::Valid,
                      llvm::StringRef constructionDomain = "test-domain")
      : name(name.str()), score(score), enabled(enabled), behavior(behavior),
        constructionDomain(constructionDomain.str()) {}

  llvm::StringRef getName() const override { return name; }
  llvm::StringRef getConstructionDomain() const override {
    return constructionDomain;
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return enabled; }

  void collectFormulaDescriptors(
      llvm::SmallVectorImpl<weft::plugin::FormulaDescriptor> &out)
      const override {
    weft::plugin::FormulaDescriptor descriptor(
        getFormulaID(), getName(), "test/selection-cost",
        weft::plugin::FormulaResultKind::AnalyticPrior,
        weft::plugin::FormulaConstructionStrength::ConstructedWeak);
    descriptor.getGeometryAxis().set(
        weft::plugin::FormulaAxisUse::Decisive,
        "SelectionCostVariantFacts");
    descriptor.getGeometryAxis().addConsumedField("variant-symbol");
    descriptor.getCapabilityAxis().set(
        weft::plugin::FormulaAxisUse::HonestNull,
        "SelectionCostNoCapabilityProjection");
    descriptor.getStaticContextAxis().set(
        weft::plugin::FormulaAxisUse::HonestNull,
        "SelectionCostNoStaticContext");
    descriptor.addSemanticCase("ranked");
    descriptor.addSemanticCase("invalid-test-fixture");
    descriptor.addProductionEntry("plugin:analytic-cost");
    out.push_back(std::move(descriptor));
  }

  llvm::Error estimateVariantCost(const VariantCostRequest &request,
                                  VariantCostEstimate &out) const override {
    if (behavior == CostBehavior::Failure)
      return llvm::make_error<llvm::StringError>(
          "plugin-local selection cost failed", llvm::errc::invalid_argument);

    out = VariantCostEstimate();
    out.setOriginPlugin(name);
    out.setFormulaID(getFormulaID());
    out.setVariantSymbol(request.getVariant().getSymName());
    if (behavior == CostBehavior::Valid) {
      out.setScore(score);
      out.setExplicitPreference(true);
    }
    return llvm::Error::success();
  }

private:
  std::string getFormulaID() const {
    return name + ".selection-cost.analytic-prior";
  }
  std::string name;
  double score;
  bool enabled;
  CostBehavior behavior;
  std::string constructionDomain;
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

class NoPreferencePlugin final : public ExtensionPlugin {
public:
  explicit NoPreferencePlugin(llvm::StringRef name) : name(name.str()) {}

  llvm::StringRef getName() const override { return name; }
  llvm::StringRef getConstructionDomain() const override {
    return "test-domain";
  }

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
  weft.exec.kernel @static_anchor attributes {} {
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    weft.exec.variant @expensive_first attributes {
      origin = "expensive",
      requires = [@generic_base]
    } {
    }
    weft.exec.variant @cheap_second attributes {
      origin = "cheap",
      requires = [@generic_base]
    } {
    }
  }

  weft.exec.kernel @tie_anchor attributes {} {
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    weft.exec.variant @tie_a attributes {
      origin = "tie-a",
      requires = [@generic_base]
    } {
    }
    weft.exec.variant @tie_b attributes {
      origin = "tie-b",
      requires = [@generic_base]
    } {
    }
  }

  weft.exec.kernel @fallback_only_anchor attributes {} {
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
	    weft.exec.variant @only_path attributes {
	      fallback_role = "conservative",
	      origin = "only",
	      requires = [@generic_base]
    } {
    }
  }

  weft.exec.kernel @no_preference_tie_anchor attributes {} {
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    weft.exec.variant @no_pref_a attributes {
      origin = "no-pref-a",
      requires = [@generic_base]
    } {
    }
    weft.exec.variant @no_pref_b attributes {
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
  auto staticPlanOrError = weft::transforms::planKernelVariantSelection(
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
  auto tiePlanOrError = weft::transforms::planKernelVariantSelection(
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
      weft::transforms::planKernelVariantSelection(
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
      weft::transforms::planKernelVariantSelection(
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
  passManager.addPass(weft::transforms::createSelectVariantsPass(registry));
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
      weft::transforms::createSelectVariantsPass(registry));
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
  weft.exec.kernel @fallback_tie_anchor attributes {} {
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    weft.exec.variant @fallback_ir_first attributes {
      fallback_role = "conservative",
      origin = "fallback-tie",
      requires = [@generic_base]
    } {
    }
    weft.exec.variant @non_fallback_ir_second attributes {
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
  auto planOrError = weft::transforms::planKernelVariantSelection(
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
  weft.exec.kernel @runtime_anchor attributes {} {
    weft.exec.capability @generic_probe {
      id = "generic.probe",
      kind = "runtime",
      status = "missing"
    }
    weft.exec.capability @generic_baseline_a {
      id = "generic.baseline.a",
      kind = "toolchain"
    }
    weft.exec.capability @generic_baseline_b {
      id = "generic.baseline.b",
      kind = "toolchain"
    }
	    weft.exec.variant @expensive_ir_first_fallback attributes {
	      fallback_role = "conservative",
	      origin = "expensive-fallback",
	      requires = [@generic_baseline_a]
    } {
    }
    weft.exec.variant @guarded_fast attributes {
      condition = "opaque(condition:fast)",
      guard = "opaque_guard_fast",
      origin = "guarded-fast",
      policy = "opaque_policy_fast",
      requires = [@generic_probe]
    } {
    }
    weft.exec.variant @guarded_tie_a attributes {
      guard = "opaque_guard_tie_a",
      origin = "guarded-tie-a",
      requires = [@generic_probe]
    } {
    }
    weft.exec.variant @guarded_tie_b attributes {
      policy = "opaque_policy_tie_b",
      origin = "guarded-tie-b",
      requires = [@generic_probe]
    } {
    }
	    weft.exec.variant @cheap_ranked_fallback attributes {
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
  auto planOrError = weft::transforms::planKernelVariantSelection(
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

  // Compile-time selection attribution ([D-4] (1)) over the RuntimeDispatch kind:
  // this is the ONLY plan kind that exercises requires_runtime_guard=true and the
  // "unavailable" keys_evaluated verdict, so assert both here (>=2 feasible
  // fallbacks ranked by an explicit owner formula => reason "prior". A neutral
  // original-order tie remains `static_order`; `measured` is still reserved).
  std::string dispatchRecord =
      weft::transforms::buildSelectionAttributionRecord(
          plan, capabilities, /*noTimestamp=*/true);
  if (int result =
          expect(dispatchRecord.find("\"requires_runtime_guard\":true") !=
                     std::string::npos,
                 "attribution record marks the guarded dispatch cases"))
    return result;
  if (int result =
          expect(dispatchRecord.find("\"generic_probe\":\"unavailable\"") !=
                     std::string::npos,
                 "attribution record derives the unavailable keys_evaluated verdict"))
    return result;
  if (int result =
          expect(dispatchRecord.find("\"reason\":\"prior\"") !=
                         std::string::npos &&
                     dispatchRecord.find("\"reason\":\"static_order\"") ==
                         std::string::npos,
                 "runtime dispatch with explicit formula ranking is prior"))
    return result;

  mlir::OpBuilder builder(&context);
  DispatchOp createdDispatch;
  if (int result =
          expectSuccess(weft::transforms::materializeRuntimeDispatchPlan(
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
      weft::transforms::createCheckCapabilityRequiresPass());
  if (int result =
          expect(mlir::succeeded(passManager.run(*module)),
                 "check-capability-requires accepts guarded dispatch and available fallback"))
    return result;

  return 0;
}

int runInjectedRegistrySelectionPassTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @pass_injected_registry attributes {} {
    weft.exec.capability @runtime_probe {
      id = "runtime.probe",
      kind = "runtime-offload",
      status = "missing"
    }
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    weft.exec.variant @guarded_fast attributes {
      condition = "runtime_probe_available",
      origin = "guarded-fast",
      requires = [@runtime_probe]
    } {
    }
	    weft.exec.variant @portable_fallback attributes {
	      fallback_role = "conservative",
	      origin = "portable-fallback",
	      requires = [@generic_base]
    } {
    }
    weft.exec.variant @available_slow attributes {
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
  passManager.addPass(weft::transforms::createSelectVariantsPass(registry));
  passManager.addPass(weft::transforms::createCheckCapabilityRequiresPass());
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
  weft.exec.kernel @conflict_dispatch_anchor attributes {} {
    weft.exec.capability @fast_runtime {
      id = "generic.fast.runtime",
      kind = "runtime",
      relations = #weft.capability_relations<conflicts = ["build.policy.disable_fast_runtime"]>,
      status = "available"
    }
    weft.exec.capability @disable_fast_profile {
      id = "generic.build.profile",
      kind = "build-policy",
      relations = #weft.capability_relations<provides = ["build.policy.disable_fast_runtime"]>,
      status = "available"
    }
    weft.exec.capability @baseline_capability {
      id = "generic.baseline",
      kind = "toolchain",
      status = "available"
    }
    weft.exec.variant @conflicting_fast attributes {
      origin = "conflicting-fast",
      requires = [@fast_runtime]
    } {
    }
    weft.exec.variant @portable_fallback attributes {
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
  auto planOrError = weft::transforms::planKernelVariantSelection(
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

  // Compile-time selection attribution ([D-4] (1)): the RuntimeDispatch kind also
  // exercises the "conflicting" keys_evaluated verdict (available-but-conflicting
  // capability) alongside a guarded case (requires_runtime_guard=true).
  std::string dispatchRecord =
      weft::transforms::buildSelectionAttributionRecord(
          plan, capabilities, /*noTimestamp=*/true);
  if (int result =
          expect(dispatchRecord.find("\"requires_runtime_guard\":true") !=
                     std::string::npos,
                 "attribution record marks the guarded conflicting case"))
    return result;
  if (int result =
          expect(dispatchRecord.find("\"fast_runtime\":\"conflicting\"") !=
                     std::string::npos,
                 "attribution record derives the conflicting keys_evaluated verdict"))
    return result;

  mlir::OpBuilder builder(&context);
  DispatchOp createdDispatch;
  if (int result =
          expectSuccess(weft::transforms::materializeRuntimeDispatchPlan(
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
      weft::transforms::createCheckCapabilityRequiresPass());
  if (int result =
          expect(mlir::succeeded(passManager.run(*module)),
                 "check-capability-requires accepts conflict-aware dispatch"))
    return result;

  return 0;
}

int runBuiltinRVVScalarFallbackSelectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @rvv_plus_scalar attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.dequantize_row_q4_0_problem @canonical_problem {qk = 32 : i64, weight_block_stride = 18 : i64, weight_d_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    weft.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    weft.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @rvv_typed_body attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %runtime_n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = weft_rvv.setvl %runtime_n {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
      } : !weft_rvv.vl
    }
  }

  weft.exec.kernel @scalar_only attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.dequantize_row_q4_0_problem @canonical_problem {qk = 32 : i64, weight_block_stride = 18 : i64, weight_d_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }

  weft.exec.kernel @rvv_only attributes {construction_domain = "riscv-execution"} {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    weft.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    weft.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    weft.exec.variant @rvv_typed_body attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %runtime_n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = weft_rvv.setvl %runtime_n {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
      } : !weft_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse built-in RVV/scalar selection module");

  ExtensionBundleRegistry bundles;
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(
              weft::plugin::registerBuiltinExtensionBundlePlugins(
                  bundles, registry),
              "register built-in extension bundle frontdoor for RVV and "
              "scalar fallback plugins"))
    return result;
  ExtensionPluginRegistry scalarMaterializationRegistry;
  if (int result = expectSuccess(
          weft::plugin::registerScalarExtensionPlugin(
              scalarMaterializationRegistry),
          "register scalar fallback plugin for explicit typed RVV fixture"))
    return result;

  mlir::OpBuilder builder(&context);
  KernelOp rvvScalarKernel = findKernel(*module, "rvv_plus_scalar");
  TargetCapabilitySet rvvScalarCapabilities =
      TargetCapabilitySet::buildFromKernel(rvvScalarKernel);
  llvm::SmallVector<VariantOp, 2> materializedVariants;
  llvm::Expected<mlir::Operation *> rvvScalarProblem =
      weft::plugin::resolveCanonicalProblem(rvvScalarKernel);
  if (!rvvScalarProblem)
    return fail(llvm::toString(rvvScalarProblem.takeError()));
  VariantProposalRequest rvvScalarRequest(
      *rvvScalarProblem, rvvScalarKernel, rvvScalarCapabilities);
  if (int result = expectSuccess(
          weft::transforms::collectAndMaterializeVariantProposals(
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
                        weft::plugin::scalar::
                            getScalarFallbackFirstSliceVariantName());
  if (int result = expect(rvvVariant && scalarVariant,
                          "materialized RVV and scalar variants are present"))
    return result;
  if (int result =
          expect(getStringAttr(rvvVariant.getOperation(), "origin") ==
                         weft::plugin::rvv::getRVVExtensionPluginName() &&
                     getStringAttr(scalarVariant.getOperation(), "origin") ==
                         weft::plugin::scalar::
                             getScalarExtensionPluginName(),
                 "materialized variants preserve plugin origins"))
    return result;
  if (int result =
          expect(getStringAttr(scalarVariant.getOperation(),
                               weft::plugin::kVariantFallbackRoleAttrName) ==
                     weft::plugin::kConservativeFallbackRoleValue,
                 "scalar fallback proposal materializes generic fallback role"))
    return result;
  if (int result = expectSuccess(
          registry.verifyKernelVariantLegality(rvvScalarKernel,
                                               rvvScalarCapabilities),
          "built-in materialized variants pass plugin legality"))
    return result;

  auto rvvScalarPlanOrError =
      weft::transforms::planKernelVariantSelection(
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
          expectSuccess(weft::transforms::materializeSelectedVariantMarker(
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
  llvm::Expected<mlir::Operation *> scalarOnlyProblem =
      weft::plugin::resolveCanonicalProblem(scalarOnlyKernel);
  if (!scalarOnlyProblem)
    return fail(llvm::toString(scalarOnlyProblem.takeError()));
  VariantProposalRequest scalarOnlyRequest(
      *scalarOnlyProblem, scalarOnlyKernel, scalarOnlyCapabilities);
  if (int result = expectSuccess(
          weft::transforms::collectAndMaterializeVariantProposals(
              builder, scalarMaterializationRegistry, scalarOnlyRequest,
              &scalarOnlyVariants),
          "materialize scalar-only fallback proposal"))
    return result;
  if (int result =
          expect(scalarOnlyVariants.size() == 1,
                 "scalar-only capability materializes one fallback variant"))
    return result;
  auto scalarOnlyPlanOrError =
      weft::transforms::planKernelVariantSelection(
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
  auto rvvOnlyPlanOrError = weft::transforms::planKernelVariantSelection(
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
          expectSuccess(weft::transforms::materializeSelectedVariantMarker(
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
  weft.exec.kernel @empty_anchor attributes {} {
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
  }

  weft.exec.kernel @no_fallback_anchor attributes {} {
    weft.exec.capability @generic_probe {
      id = "generic.probe",
      kind = "runtime",
      status = "missing"
    }
    weft.exec.variant @guarded_only attributes {
      condition = "opaque_runtime_probe",
      origin = "guarded-only",
      requires = [@generic_probe]
    } {
    }
  }

  weft.exec.kernel @unguarded_unavailable_anchor attributes {} {
    weft.exec.capability @generic_missing {
      id = "generic.missing",
      kind = "runtime",
      status = "missing"
    }
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
	    weft.exec.variant @unguarded_unavailable attributes {
	      origin = "unguarded-unavailable",
	      requires = [@generic_missing]
	    } {
	    }
	  }

  weft.exec.kernel @conflict_without_fallback_anchor attributes {} {
    weft.exec.capability @fast_runtime {
      id = "generic.fast.runtime",
      kind = "runtime",
      relations = #weft.capability_relations<conflicts = ["build.policy.disable_fast_runtime"]>,
      status = "available"
    }
    weft.exec.capability @disable_fast_profile {
      id = "generic.build.profile",
      kind = "build-policy",
      relations = #weft.capability_relations<provides = ["build.policy.disable_fast_runtime"]>,
      status = "available"
    }
    weft.exec.variant @conflicting_only attributes {
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
  auto emptyPlanOrError = weft::transforms::planKernelVariantSelection(
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
          weft::transforms::planKernelVariantSelection(
              noFallbackKernel,
              TargetCapabilitySet::buildFromKernel(noFallbackKernel), registry),
          {"no plugin-provided conflict-free conservative fallback candidate",
           "no_fallback_anchor"}))
    return result;

  KernelOp unguardedKernel =
      findKernel(*module, "unguarded_unavailable_anchor");
  if (int result = expectPlanErrorContains(
          weft::transforms::planKernelVariantSelection(
              unguardedKernel,
              TargetCapabilitySet::buildFromKernel(unguardedKernel), registry),
          {"no plugin-provided conflict-free conservative fallback candidate",
           "unguarded_unavailable_anchor"}))
    return result;

  KernelOp conflictWithoutFallbackKernel =
      findKernel(*module, "conflict_without_fallback_anchor");
  if (int result = expectPlanErrorContains(
          weft::transforms::planKernelVariantSelection(
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
  weft.exec.kernel @existing_dispatch_anchor attributes {} {
    weft.exec.capability @generic_probe {
      id = "generic.probe",
      kind = "runtime",
      status = "missing"
    }
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    weft.exec.variant @guarded_path attributes {
      policy = "opaque_runtime_policy",
      origin = "guarded",
      requires = [@generic_probe]
    } {
    }
	    weft.exec.variant @fallback_path attributes {
	      fallback_role = "conservative",
	      origin = "fallback",
	      requires = [@generic_base]
	    } {
    }
    weft.exec.dispatch attributes {} {
      weft.exec.case @guarded_path {policy = "preexisting_policy"}
      weft.exec.fallback @fallback_path
    }
  }

  weft.exec.kernel @other_anchor attributes {} {
    weft.exec.capability @generic_other {
      id = "generic.other",
      kind = "toolchain"
    }
    weft.exec.variant @other_path attributes {
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
  auto planOrError = weft::transforms::planKernelVariantSelection(
      kernel, TargetCapabilitySet::buildFromKernel(kernel), registry);
  if (!planOrError)
    return fail("existing dispatch planning failed: " +
                llvm::toString(planOrError.takeError()));

  mlir::OpBuilder builder(&context);
  if (int result = expectErrorContains(
          weft::transforms::materializeRuntimeDispatchPlan(
              builder, *planOrError),
          {"already contains a direct weft.exec.dispatch",
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
      weft::transforms::VariantSelectionCase{
          crossKernelPlan.fallback, otherEstimate, 0, true, true, true, false});
  crossKernelPlan.dispatchCases.push_back(
      weft::transforms::VariantSelectionCase{
          findDirectVariant(kernel, "guarded_path"),
          VariantCostEstimate(), 0, false, false, true, true});

  if (int result = expectErrorContains(
          weft::transforms::materializeRuntimeDispatchPlan(
              builder, crossKernelPlan),
          {"dispatch case variant is not a direct child",
           "other_anchor", "guarded_path"}))
    return result;

  VariantSelectionPlan missingKernelPlan;
  missingKernelPlan.kind = VariantSelectionKind::RuntimeDispatch;
  if (int result = expectErrorContains(
          weft::transforms::materializeRuntimeDispatchPlan(
              builder, missingKernelPlan),
          {"requires a weft.exec.kernel", "<missing>"}))
    return result;

  return 0;
}

int runCostFailurePropagationTests(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @cost_failure_anchor attributes {} {
    weft.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    weft.exec.variant @unknown_origin attributes {
      origin = "missing-plugin",
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
            weft::transforms::planKernelVariantSelection(
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
            weft::transforms::planKernelVariantSelection(
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
            weft::transforms::planKernelVariantSelection(
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
            weft::transforms::planKernelVariantSelection(
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
            weft::transforms::planKernelVariantSelection(
                KernelOp(), registry),
            {"requires a weft.exec.kernel", "<missing>"}))
      return result;
  }

  return 0;
}

// [SEL-2] cross-paradigm silent-misfire FALSIFIER + capability-prior commit-timing lock
// (G4 IME campaign M0, 实验总纲 T4b persistent negative control).
//
// Reproduces — as a regression guard that runs every check-weft — the
// "matrix/vector co-bid silent misfire" the exec-level capability prior closes:
//
//   * BAD config (capability-BLIND constant scores == the pre-SEL-1-T5 IME plugin:
//     matrix score 20.0 > vector base 1.0 under the registry's ASCENDING stable_sort):
//     for a whole-matrix GEMM co-bid the matrix paradigm SILENTLY loses — chosen =
//     vector, NO error. This is the P7 潜伏错选 ([SEL-2]).
//   * GOOD config (capability-DERIVED prior == post SEL-1-T5: matrix score 0.5 < vector
//     base 1.0): the SAME GEMM co-bid → chosen = matrix. The prior makes the misfire
//     vanish.
//   * The two verdicts DIFFER (bad chosen = vector, good chosen = matrix): the assertion
//     that proves the prior layer actually removes the misfire, not merely that a good
//     config happens to work.
//
// The constant scores are INJECTED via a test plugin (the sanctioned constant-score
// path — the real plugin cannot re-emit its old blind 20 for a GEMM shape now that
// SEL-1-T5 landed). The capability-DERIVATION of the 0.5-vs-20 scores from the
// spacemit.ime exact canonical problem geometry is proven separately against the PRODUCTION RVV +
// IME plugins by test/Transforms/VariantSelection/capability-prior-ime-gemm-over-rvv.mlir.
// This falsifier and that lit are the two halves of the same closure: the lit proves the
// score is derived; this test proves the two score regimes flip the winner.
//
// [SEL-2] hard timing obligation (capability prior consulted BEFORE variant commit):
// planKernelVariantSelection consults the prior (via rankKernelVariantsByCost) to build
// the plan; the commit (materializeSelectedVariantMarker) is a strict downstream consumer
// of that already-built plan. We lock this by asserting the COMMITTED marker carries the
// derived prior score (0.5) and targets the derived winner — a commit artifact cannot
// carry the prior's verdict unless the prior ran first.
int runSel2CrossParadigmMisfireFalsifierTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @gemm_cobid attributes {} {
    weft.exec.capability @compute_base {
      id = "compute.base",
      kind = "toolchain"
    }
    weft.exec.variant @vector_paradigm_body attributes {
      origin = "vector-paradigm",
      requires = [@compute_base]
    } {
    }
    weft.exec.variant @matrix_paradigm_body attributes {
      origin = "matrix-paradigm",
      requires = [@compute_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse SEL-2 cross-paradigm falsifier module");

  KernelOp gemmKernel = findKernel(*module, "gemm_cobid");
  VariantOp vectorVariant =
      findDirectVariant(gemmKernel, "vector_paradigm_body");
  VariantOp matrixVariant =
      findDirectVariant(gemmKernel, "matrix_paradigm_body");
  if (int result = expect(gemmKernel && vectorVariant && matrixVariant,
                          "GEMM co-bid kernel exposes a vector and a matrix "
                          "paradigm variant"))
    return result;
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(gemmKernel);

  // BAD config: capability-blind constant scores (pre-SEL-1-T5). Matrix 20 > vector 1.
  SelectionCostPlugin vectorBaseBlind("vector-paradigm", 1.0);
  SelectionCostPlugin matrixBlind("matrix-paradigm", 20.0);
  ExtensionPluginRegistry blindRegistry;
  if (int result = expectSuccess(blindRegistry.registerPlugin(vectorBaseBlind),
                                 "register blind vector base"))
    return result;
  if (int result = expectSuccess(blindRegistry.registerPlugin(matrixBlind),
                                 "register blind matrix constant"))
    return result;

  auto blindPlanOrError = weft::transforms::planKernelVariantSelection(
      gemmKernel, capabilities, blindRegistry);
  if (!blindPlanOrError)
    return fail("blind-config selection failed: " +
                llvm::toString(blindPlanOrError.takeError()));
  VariantSelectionPlan blindPlan = std::move(*blindPlanOrError);
  if (int result =
          expect(blindPlan.kind == VariantSelectionKind::StaticVariant &&
                     blindPlan.selectedVariant == vectorVariant,
                 "BAD config MISFIRES: blind ascending 20>1 silently picks the "
                 "vector paradigm for the GEMM (matrix loses, no error)"))
    return result;

  // GOOD config: capability-derived prior (post SEL-1-T5). Matrix 0.5 < vector 1.
  SelectionCostPlugin vectorBaseDerived("vector-paradigm", 1.0);
  SelectionCostPlugin matrixDerived("matrix-paradigm", 0.5);
  ExtensionPluginRegistry derivedRegistry;
  if (int result =
          expectSuccess(derivedRegistry.registerPlugin(vectorBaseDerived),
                        "register derived vector base"))
    return result;
  if (int result = expectSuccess(derivedRegistry.registerPlugin(matrixDerived),
                                 "register derived matrix prior"))
    return result;

  auto derivedPlanOrError = weft::transforms::planKernelVariantSelection(
      gemmKernel, capabilities, derivedRegistry);
  if (!derivedPlanOrError)
    return fail("derived-config selection failed: " +
                llvm::toString(derivedPlanOrError.takeError()));
  VariantSelectionPlan derivedPlan = std::move(*derivedPlanOrError);
  if (int result =
          expect(derivedPlan.kind == VariantSelectionKind::StaticVariant &&
                     derivedPlan.selectedVariant == matrixVariant,
                 "GOOD config: capability-derived 0.5<1 prior picks the matrix "
                 "paradigm for the GEMM (misfire eliminated)"))
    return result;

  // The verdicts DIFFER: this is the load-bearing falsifier assertion that the prior
  // layer really removes the silent misfire (bad chosen = vector, good chosen = matrix).
  if (int result =
          expect(blindPlan.selectedVariant != derivedPlan.selectedVariant &&
                     blindPlan.selectedVariant == vectorVariant &&
                     derivedPlan.selectedVariant == matrixVariant,
                 "capability prior flips the GEMM winner vector->matrix: the "
                 "blind-config misfire is gone under the derived prior"))
    return result;

  // [SEL-2] timing: the COMMIT is a strict downstream consumer of the prior-built plan.
  // Materialize from the GOOD plan and assert the marker carries the derived prior score
  // and targets the derived winner — proving the prior was consulted before the commit.
  mlir::OpBuilder builder(&context);
  DiagnosticOp marker;
  if (int result = expectSuccess(
          weft::transforms::materializeSelectedVariantMarker(
              builder, derivedPlan, &marker),
          "materialize the capability-prior-committed selected marker"))
    return result;
  std::optional<double> committedScore =
      getFloatAttr(marker.getOperation(), "preference_score");
  if (int result =
          expect(marker && getTarget(marker.getOperation()) ==
                               matrixVariant.getSymName() &&
                     getStringAttr(marker.getOperation(), "selection_kind") ==
                         "static-variant" &&
                     committedScore.has_value() && *committedScore == 0.5,
                 "[SEL-2] commit consumes the prior: committed marker targets "
                 "the matrix variant and carries the derived prior score 0.5"))
    return result;

  return 0;
}

// T4b — FOUR-CONFIGURATION selector ablation matrix (cost-model × scenario).
// (G4 IME campaign, 实验总纲 §4.2 "选择器四配置" + T4b table + [SEL-2] 专项.)
//
// This SYSTEMATIZES the scattered priors — the M0 falsifier's two GEMM cells
// (runSel2CrossParadigmMisfireFalsifierTest, capability-BLIND vs capability-
// DERIVED) plus the T5c M-aware decode boundary — into the complete 2x2 ablation
// the science plan calls for, and adds the two DECODE cells that pin down the
// PRECISION of the fix. The matrix is:
//
//                     | GEMM  (M>=M*, compute-bound prefill) | decode (M<M*, memory-bound)
//   ------------------+--------------------------------------+---------------------------
//   capability-BLIND  | (a) SILENT MISFIRE reproduced        | (d) accidentally CORRECT
//   constant cost     |     20>1 asc -> chosen = VECTOR (X)  |     20>1 -> chosen = VECTOR (ok)
//   (pre SEL-1-T5)    |                                      |
//   ------------------+--------------------------------------+---------------------------
//   capability-       | (b) MISFIRE vanishes                 | (c) roofline PARITY
//   DERIVED cost      |     0.5<1 -> chosen = MATRIX (ok)    |     1.0==1.0 -> chosen = VECTOR (ok)
//   (SEL-1-T5 + T5c)  |                                      |     (matrix NOT preferred)
//
// The ROOFLINE-correct paradigm per scenario: GEMM wants the systolic MATRIX
// array (M reaches the crossover M*, compute-bound); decode wants the RVV VECTOR
// path (M below M*, memory-bound — the compute-isolated micro-advantage does not
// transduce, micro↛e2e). The four cells + four cross-config readings prove the
// capability-derived + M-aware prior PRECISELY removes the GEMM silent misfire
// WITHOUT over-correcting decode.
//
// PROVENANCE / defense-in-depth honesty (照 T5c):
//   * The DERIVED row (b, c) is separately proven end-to-end on the PRODUCTION
//     RVV + IME plugins by the lit tests capability-prior-ime-gemm-over-rvv.mlir
//     (GEMM matM=256) and capability-prior-ime-mstar-writeback.mlir (GEMM matM=64
//     + decode single-fragment). The BLIND row (a, d) is gtest-only by
//     construction: the production plugin CANNOT re-emit its old blind 20 for a
//     GEMM shape now that SEL-1-T5 landed, so the blind constants are INJECTED via
//     the sanctioned test-plugin constant-score path (exactly as the M0 falsifier
//     does).
//   * Cell (c) exercises the T5c M<M* roofline-PARITY cost (1.0). That parity
//     branch is auditable DEFENSE-IN-DEPTH on the cost layer, NOT a production-
//     reachable tiled-emission path (the tiled-shape derivation fail-closes below
//     macM — no remainder path — so a real tiled GEMM reaching cost always has
//     matM >= macM). The REACHABLE production decode path is the single MAC
//     fragment (cost 20), proven by capability-prior-ime-mstar-writeback.mlir and
//     re-observed here as cell (c') — decode is DOUBLY excluded from the matrix
//     paradigm (parity 1.0 AND single-fragment 20 both keep the vector path).
//
// [NG-4]: this is a MECHANISM ablation — no perf numbers, no throughput, no beat
// claim. The `prior` reason enum flip stays a SEPARATE canon-gated burn-down step
// (NOT performed here). Behavior is only OBSERVED, never changed.
int runT4bFourConfigSelectorAblationTest(mlir::MLIRContext &context) {
  // The same paradigm-abstract co-bid kernel as the M0 falsifier: a vector and a
  // matrix paradigm variant co-bidding on ONE contraction. The vector variant is
  // declared FIRST (lower original IR index) — the deterministic tie-break that
  // decides an equal-score PARITY (cell (c)).
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @paradigm_cobid attributes {} {
    weft.exec.capability @compute_base {
      id = "compute.base",
      kind = "toolchain"
    }
    weft.exec.variant @vector_paradigm_body attributes {
      origin = "vector-paradigm",
      requires = [@compute_base]
    } {
    }
    weft.exec.variant @matrix_paradigm_body attributes {
      origin = "matrix-paradigm",
      requires = [@compute_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse T4b four-config ablation module");

  KernelOp kernel = findKernel(*module, "paradigm_cobid");
  VariantOp vectorVariant = findDirectVariant(kernel, "vector_paradigm_body");
  VariantOp matrixVariant = findDirectVariant(kernel, "matrix_paradigm_body");
  if (int result =
          expect(kernel && vectorVariant && matrixVariant,
                 "co-bid kernel exposes a vector and a matrix paradigm variant"))
    return result;
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  // One ablation cell = inject (matrixScore, vectorScore) constant costs via the
  // sanctioned test-plugin path and read back the winning variant. The BLIND
  // scores model the pre-SEL-1-T5 capability-blind constants; the DERIVED scores
  // model the post SEL-1-T5 + T5c capability-derived, M-aware prior. The cell only
  // PLANS (no IR mutation); the committed marker is materialized once, below.
  auto runCell = [&](double matrixScore, double vectorScore, const char *label,
                     VariantOp &winnerOut) -> int {
    SelectionCostPlugin vectorBase("vector-paradigm", vectorScore);
    SelectionCostPlugin matrix("matrix-paradigm", matrixScore);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(vectorBase),
                                   llvm::Twine(label) + ": register vector base"))
      return result;
    if (int result = expectSuccess(registry.registerPlugin(matrix),
                                   llvm::Twine(label) + ": register matrix"))
      return result;
    auto planOrError = weft::transforms::planKernelVariantSelection(
        kernel, capabilities, registry);
    if (!planOrError)
      return fail(llvm::Twine(label) + " selection failed: " +
                  llvm::toString(planOrError.takeError()));
    VariantSelectionPlan plan = std::move(*planOrError);
    if (int result =
            expect(plan.kind == VariantSelectionKind::StaticVariant,
                   llvm::Twine(label) + ": expected a static-variant selection"))
      return result;
    winnerOut = plan.selectedVariant;
    return 0;
  };

  // Config axis (rows): BLIND constant (matrix 20 > vector 1) vs DERIVED prior.
  // Scenario axis (cols): GEMM (roofline-correct = MATRIX) vs decode (= VECTOR).
  constexpr double kBlindMatrixCost = 20.0;      // pre-SEL-1-T5 capability-blind
  constexpr double kVectorBaseCost = 1.0;        // RVV vector base (both regimes)
  constexpr double kDerivedGemmMatrixCost = 0.5; // SEL-1-T5 GEMM preference (M>=M*)
  constexpr double kDerivedDecodeParityCost = 1.0; // T5c M<M* roofline parity (=base)
  constexpr double kDerivedFragmentMacCost = 20.0; // single MAC fragment leaf cost

  VariantOp aWinner, bWinner, cWinner, dWinner, cFragmentWinner;
  // (a) BLIND × GEMM.
  if (int result = runCell(kBlindMatrixCost, kVectorBaseCost, "(a) blind×GEMM",
                           aWinner))
    return result;
  // (b) DERIVED × GEMM.
  if (int result = runCell(kDerivedGemmMatrixCost, kVectorBaseCost,
                           "(b) derived×GEMM", bWinner))
    return result;
  // (c) DERIVED × decode (roofline parity).
  if (int result = runCell(kDerivedDecodeParityCost, kVectorBaseCost,
                           "(c) derived×decode(parity)", cWinner))
    return result;
  // (d) BLIND × decode.
  if (int result = runCell(kBlindMatrixCost, kVectorBaseCost, "(d) blind×decode",
                           dWinner))
    return result;
  // (c') DERIVED × decode via the single MAC fragment leaf cost (double-exclusion).
  if (int result = runCell(kDerivedFragmentMacCost, kVectorBaseCost,
                           "(c') derived×decode(fragment)", cFragmentWinner))
    return result;

  // ---- Per-cell verdicts --------------------------------------------------
  // (a) BLIND × GEMM = SILENT MISFIRE reproduced: chosen = vector (WRONG — the
  //     GEMM roofline wants the matrix array; blind 20>1 ascending silently picks
  //     vector, no error). This is the P7 潜伏错选 ([SEL-2]).
  if (int result =
          expect(aWinner == vectorVariant,
                 "(a) blind×GEMM MISFIRES: blind 20>1 ascending silently selects "
                 "the vector paradigm for a GEMM the roofline wants on the matrix "
                 "array (no error)"))
    return result;
  // (b) DERIVED × GEMM = misfire VANISHES: chosen = matrix (RIGHT).
  if (int result =
          expect(bWinner == matrixVariant,
                 "(b) derived×GEMM: capability-derived 0.5<1 selects the matrix "
                 "paradigm — the silent misfire is eliminated"))
    return result;
  // (c) DERIVED × decode (parity) = matrix NOT preferred: chosen = vector (RIGHT).
  //     The M<M* roofline-parity score (1.0) TIES the vector base; the
  //     deterministic tie-break (equal score -> non-fallback -> original IR order,
  //     vector declared first) keeps the vector path.
  if (int result =
          expect(cWinner == vectorVariant,
                 "(c) derived×decode: M<M* roofline PARITY (1.0==base) does NOT "
                 "prefer the matrix paradigm; the tie-break keeps the vector path"))
    return result;
  // (d) BLIND × decode = ACCIDENTALLY correct: chosen = vector (RIGHT — but only
  //     because the SAME blind 20>1 that MISFIRED for GEMM happens to match the
  //     decode roofline).
  if (int result =
          expect(dWinner == vectorVariant,
                 "(d) blind×decode: the SAME blind 20>1 that misfired for GEMM is "
                 "ACCIDENTALLY correct for decode (the roofline wants vector here)"))
    return result;
  // (c') DOUBLE-EXCLUSION: the derived single MAC fragment leaf cost (20>1) ALSO
  //     keeps the vector path — decode is excluded from the matrix paradigm by
  //     BOTH derived cost mechanisms (parity 1.0 AND single-fragment 20).
  if (int result =
          expect(cFragmentWinner == vectorVariant,
                 "(c') derived×decode fragment leaf cost 20>1 ALSO keeps the "
                 "vector path — decode is doubly excluded from the matrix paradigm"))
    return result;

  // ---- Ablation reading 1: silent misfire REPRODUCE -> VANISH (GEMM column) --
  // (a) chose vector (WRONG); (b) chose matrix (RIGHT); the verdicts DIFFER — the
  // capability-derived prior REMOVES the misfire (not merely a config that works).
  if (int result =
          expect(aWinner == vectorVariant && bWinner == matrixVariant &&
                     aWinner != bWinner,
                 "ABLATION-1 (GEMM column): the capability-derived prior flips the "
                 "GEMM winner vector->matrix — the blind silent misfire is "
                 "reproduced then eliminated"))
    return result;

  // ---- Ablation reading 2: decode NOT over-corrected (decode column) ---------
  // (d) blind×decode and (c) derived×decode BOTH chose vector — the fix leaves the
  // decode verdict UNCHANGED (no over-correction to the matrix paradigm on a
  // compute-micro signal the memory-bound decode roofline never realizes).
  if (int result =
          expect(dWinner == vectorVariant && cWinner == vectorVariant &&
                     dWinner == cWinner,
                 "ABLATION-2 (decode column): blind and derived AGREE on the "
                 "vector path for decode — the fix does not over-correct decode "
                 "to the matrix paradigm"))
    return result;

  // ---- Ablation reading 3: PRECISION (the load-bearing 2x2 diagonal) ---------
  // Going blind -> derived flips ONLY the GEMM verdict (a->b: vector->matrix) and
  // leaves the decode verdict fixed (d->c: vector->vector). The capability-derived
  // + M-aware prior is therefore a PRECISE fix, not a blanket matrix bias.
  if (int result =
          expect(aWinner != bWinner && dWinner == cWinner,
                 "ABLATION-3 (precision): blind->derived flips the GEMM winner but "
                 "NOT the decode winner — the prior precisely removes the misfire"))
    return result;

  // ---- Ablation reading 4: the blind cost is SCENARIO-BLIND ------------------
  // (a) and (d) inject IDENTICAL scores (matrix 20 > vector 1) and reach the
  // IDENTICAL verdict (vector) — yet (a) is WRONG (GEMM) and (d) is RIGHT (decode).
  // A single scenario-blind constant cannot separate a GEMM from a decode. The
  // derived cost IS scenario-aware: it scores the SAME matrix variant 0.5 for the
  // GEMM but at parity (1.0) for decode (0.5 != 1.0).
  if (int result =
          expect(aWinner == dWinner &&
                     kDerivedGemmMatrixCost != kDerivedDecodeParityCost,
                 "ABLATION-4 (scenario-blindness): the blind constant gives GEMM "
                 "and decode the SAME verdict (cannot tell them apart) while the "
                 "derived cost is scenario-aware (GEMM 0.5 != decode parity 1.0)"))
    return result;

  // ---- Full-chain commit lock (decode side) ---------------------------------
  // Complement the M0 falsifier's GEMM-side materialization (matrix winner, score
  // 0.5). Materialize the derived decode-PARITY plan and assert the committed
  // marker targets the VECTOR variant and carries the parity score (1.0): the
  // committed artifact tracks the decode verdict — the matrix paradigm is NOT
  // committed for decode even at cost parity.
  SelectionCostPlugin decodeVectorBase("vector-paradigm", kVectorBaseCost);
  SelectionCostPlugin decodeMatrixParity("matrix-paradigm",
                                          kDerivedDecodeParityCost);
  ExtensionPluginRegistry decodeRegistry;
  if (int result = expectSuccess(decodeRegistry.registerPlugin(decodeVectorBase),
                                 "register decode-parity vector base"))
    return result;
  if (int result =
          expectSuccess(decodeRegistry.registerPlugin(decodeMatrixParity),
                        "register decode-parity matrix"))
    return result;
  auto decodePlanOrError = weft::transforms::planKernelVariantSelection(
      kernel, capabilities, decodeRegistry);
  if (!decodePlanOrError)
    return fail("decode-parity selection failed: " +
                llvm::toString(decodePlanOrError.takeError()));
  VariantSelectionPlan decodePlan = std::move(*decodePlanOrError);
  mlir::OpBuilder builder(&context);
  DiagnosticOp decodeMarker;
  if (int result = expectSuccess(
          weft::transforms::materializeSelectedVariantMarker(
              builder, decodePlan, &decodeMarker),
          "materialize the derived decode-parity committed marker"))
    return result;
  std::optional<double> committedDecodeScore =
      getFloatAttr(decodeMarker.getOperation(), "preference_score");
  if (int result =
          expect(decodeMarker &&
                     getTarget(decodeMarker.getOperation()) ==
                         vectorVariant.getSymName() &&
                     getStringAttr(decodeMarker.getOperation(),
                                   "selection_kind") == "static-variant" &&
                     committedDecodeScore.has_value() &&
                     *committedDecodeScore == 1.0,
                 "decode-parity commit targets the vector variant and carries the "
                 "parity score 1.0 — the matrix paradigm is not committed for "
                 "decode"))
    return result;

  return 0;
}

int runConstructionDomainGateTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @domain_gate attributes {construction_domain = "test-domain"} {
    weft.exec.capability @available {id = "available", kind = "test", status = "available"}
    weft.exec.variant @local attributes {origin = "local-owner", requires = [@available]} {
    }
    weft.exec.variant @foreign attributes {origin = "foreign-owner", requires = [@available]} {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    return fail("failed to parse construction-domain selection test module");

  KernelOp kernel = findKernel(*module, "domain_gate");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  SelectionCostPlugin local("local-owner", 1.0);
  SelectionCostPlugin foreign("foreign-owner", 0.5, true,
                              CostBehavior::Valid, "foreign-domain");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(local),
                                 "register local-domain selection owner"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(foreign),
                                 "register foreign-domain selection owner"))
    return result;

  return expectPlanErrorContains(
      weft::transforms::planKernelVariantSelection(kernel, capabilities,
                                                   registry),
      {"declares construction domain 'test-domain'", "variant @foreign",
       "origin plugin 'foreign-owner'",
       "foreign construction domain 'foreign-domain'"});
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  weft::registerAllDialects(dialectRegistry);
  ExtensionBundleRegistry dialectBundles;
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(
              weft::plugin::registerBuiltinExtensionBundlePlugins(
                  dialectBundles, dialectPlugins),
              "register built-in extension bundle frontdoor for plugin "
              "dialects"))
    return result;
  weft::registerPluginDialects(dialectPlugins, dialectRegistry);

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
  if (int result = runSel2CrossParadigmMisfireFalsifierTest(context))
    return result;
  if (int result = runT4bFourConfigSelectorAblationTest(context))
    return result;
  if (int result = runConstructionDomainGateTest(context))
    return result;

  llvm::outs() << "variant selection planning smoke test passed\n";
  return 0;
}
