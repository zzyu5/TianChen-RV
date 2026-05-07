#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>

using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantCostEstimate;
using tianchenrv::plugin::VariantCostRankingEntry;
using tianchenrv::plugin::VariantCostRequest;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

namespace {

enum class CostBehavior {
  Valid,
  Failure,
  MissingScore,
  NonFiniteScore,
  NegativeScore,
  EmptyExplanation,
  EmptyPolicy,
};

class CostPlugin final : public ExtensionPlugin {
public:
  CostPlugin(llvm::StringRef name, double score, bool enabled = true,
             CostBehavior behavior = CostBehavior::Valid)
      : name(name.str()), availableScore(score), unavailableScore(score),
        enabled(enabled), behavior(behavior) {}

  CostPlugin &setCapabilityScore(llvm::StringRef id, double available,
                                 double unavailable) {
    capabilityID = id.str();
    availableScore = available;
    unavailableScore = unavailable;
    return *this;
  }

  CostPlugin &setExplanation(llvm::StringRef value) {
    explanation = value.str();
    explanationSet = true;
    return *this;
  }

  CostPlugin &setPolicy(llvm::StringRef value) {
    policy = value.str();
    policySet = true;
    return *this;
  }

  CostPlugin &setCallOrder(llvm::SmallVectorImpl<std::string> *order) {
    callOrder = order;
    return *this;
  }

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
    ++costCalls;
    observedVariantSymbols.push_back(request.getVariant().getSymName().str());
    observedKernelSymbols.push_back(request.getKernel().getSymName().str());
    observedCapabilityCounts.push_back(request.getCapabilities().size());
    observedGenericFastAvailable =
        request.getCapabilities().isCapabilityAvailableByID("generic.fast");

    if (const CapabilityDescriptor *capability =
            request.getCapabilities().lookupBySymbolName("generic_fast")) {
      observedGenericFastKind = capability->getKind().str();
    }

    if (callOrder)
      callOrder->push_back(request.getVariant().getSymName().str());

    if (behavior == CostBehavior::Failure)
      return llvm::make_error<llvm::StringError>(
          "plugin-local cost failed", llvm::errc::invalid_argument);

    out = VariantCostEstimate();
    out.setOriginPlugin(name);
    out.setVariantSymbol(request.getVariant().getSymName());
    out.setExplicitPreference(true);

    double score = availableScore;
    if (!capabilityID.empty()) {
      score = request.getCapabilities().isCapabilityAvailableByID(capabilityID)
                  ? availableScore
                  : unavailableScore;
    }

    switch (behavior) {
    case CostBehavior::Valid:
    case CostBehavior::Failure:
      out.setScore(score);
      break;
    case CostBehavior::MissingScore:
      break;
    case CostBehavior::NonFiniteScore:
      out.setScore(std::numeric_limits<double>::infinity());
      break;
    case CostBehavior::NegativeScore:
      out.setScore(-1.0);
      break;
    case CostBehavior::EmptyExplanation:
      out.setScore(score);
      out.setExplanation("  ");
      break;
    case CostBehavior::EmptyPolicy:
      out.setScore(score);
      out.setPolicy("");
      break;
    }

    if (explanationSet)
      out.setExplanation(explanation);
    if (policySet)
      out.setPolicy(policy);

    return llvm::Error::success();
  }

  unsigned getCostCalls() const { return costCalls; }
  llvm::ArrayRef<std::string> getObservedVariantSymbols() const {
    return observedVariantSymbols;
  }
  llvm::ArrayRef<std::string> getObservedKernelSymbols() const {
    return observedKernelSymbols;
  }
  llvm::ArrayRef<std::size_t> getObservedCapabilityCounts() const {
    return observedCapabilityCounts;
  }
  bool sawGenericFastAvailable() const { return observedGenericFastAvailable; }
  llvm::StringRef getObservedGenericFastKind() const {
    return observedGenericFastKind;
  }

private:
  std::string name;
  std::string capabilityID;
  double availableScore;
  double unavailableScore;
  bool enabled;
  CostBehavior behavior;
  bool explanationSet = false;
  std::string explanation;
  bool policySet = false;
  std::string policy;
  llvm::SmallVectorImpl<std::string> *callOrder = nullptr;
  llvm::SmallVector<PluginCapability, 1> capabilities;
  mutable unsigned costCalls = 0;
  mutable llvm::SmallVector<std::string, 4> observedVariantSymbols;
  mutable llvm::SmallVector<std::string, 4> observedKernelSymbols;
  mutable llvm::SmallVector<std::size_t, 4> observedCapabilityCounts;
  mutable bool observedGenericFastAvailable = false;
  mutable std::string observedGenericFastKind;
};

class DefaultCostPlugin final : public ExtensionPlugin {
public:
  explicit DefaultCostPlugin(llvm::StringRef name) : name(name.str()) {}

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
    return fail("expected variant cost error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("variant cost error text missing '") +
                  fragment + "': " + message);
  }
  return 0;
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

std::string printModule(mlir::ModuleOp module) {
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module->print(stream);
  stream.flush();
  return printed;
}

int runSingleVariantCostRoutingTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @cost_anchor attributes {} {
    tcrv.exec.capability @generic_fast {
      id = "generic.fast",
      kind = "generic-feature"
    }
    tcrv.exec.capability @generic_optional {
      id = "generic.optional",
      kind = "policy",
      status = "disabled"
    }
    tcrv.exec.variant @alpha_path attributes {
      origin = "alpha",
      requires = [@generic_fast]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse single variant cost module");

  KernelOp kernel = findKernel(*module, "cost_anchor");
  VariantOp variant = findDirectVariant(kernel, "alpha_path");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  CostPlugin alpha("alpha", 8.0);
  alpha.setCapabilityScore("generic.fast", 2.0, 8.0)
      .setExplanation("generic capability score")
      .setPolicy("prefer lower generic score");
  CostPlugin nonOrigin("non-origin", 1.0);
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(alpha), "register alpha"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(nonOrigin),
                                 "register non-origin"))
    return result;

  VariantCostEstimate estimate;
  VariantCostRequest request(variant, kernel, capabilities);
  if (int result = expectSuccess(registry.estimateVariantCost(request, estimate),
                                 "estimate single variant cost"))
    return result;

  if (int result = expect(alpha.getCostCalls() == 1,
                          "origin plugin cost hook is called exactly once"))
    return result;
  if (int result = expect(nonOrigin.getCostCalls() == 0,
                          "non-origin plugin cost hook is not called"))
    return result;
  if (int result = expect(alpha.getObservedVariantSymbols().size() == 1 &&
                              alpha.getObservedVariantSymbols()[0] ==
                                  "alpha_path",
                          "cost hook receives expected variant symbol"))
    return result;
  if (int result = expect(alpha.getObservedKernelSymbols().size() == 1 &&
                              alpha.getObservedKernelSymbols()[0] ==
                                  "cost_anchor",
                          "cost hook receives expected enclosing kernel"))
    return result;
  if (int result = expect(alpha.getObservedCapabilityCounts().size() == 1 &&
                              alpha.getObservedCapabilityCounts()[0] == 2 &&
                              alpha.sawGenericFastAvailable() &&
                              alpha.getObservedGenericFastKind() ==
                                  "generic-feature",
                          "cost hook receives expected TargetCapabilitySet"))
    return result;
  if (int result = expect(estimate.hasScore() && estimate.getScore() == 2.0,
                          "cost can depend on generic capability availability"))
    return result;
  if (int result = expect(estimate.hasExplicitPreference(),
                          "origin plugin marks explicit selection preference"))
    return result;
  if (int result = expect(estimate.getOriginPlugin() == "alpha" &&
                              estimate.getVariantSymbol() == "alpha_path",
                          "estimate preserves origin and variant symbol"))
    return result;
  if (int result = expect(estimate.hasExplanation() &&
                              estimate.getExplanation() ==
                                  "generic capability score" &&
                              estimate.hasPolicy() &&
                              estimate.getPolicy() ==
                                  "prefer lower generic score",
                          "non-empty explanation and policy are preserved"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "successful cost routing keeps module valid"))
    return result;

  return 0;
}

int runDefaultNeutralEstimateTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @default_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @default_path attributes {
      origin = "default",
      requires = [@generic_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse default cost module");

  KernelOp kernel = findKernel(*module, "default_anchor");
  VariantOp variant = findDirectVariant(kernel, "default_path");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  DefaultCostPlugin plugin("default");
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(plugin), "register default"))
    return result;

  VariantCostEstimate estimate;
  VariantCostRequest request(variant, kernel, capabilities);
  if (int result = expectSuccess(registry.estimateVariantCost(request, estimate),
                                 "estimate default neutral cost"))
    return result;

  if (int result = expect(estimate.hasScore() && estimate.getScore() == 0.0,
                          "default cost hook returns neutral finite score"))
    return result;
  if (int result = expect(!estimate.hasExplicitPreference(),
                          "default hook records no explicit plugin preference"))
    return result;
  if (int result = expect(estimate.getOriginPlugin() == "default" &&
                              estimate.getVariantSymbol() == "default_path",
                          "default cost hook records origin and variant"))
    return result;

  return 0;
}

int runKernelCollectionAndRankingTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @ranking_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @third_path attributes {
      origin = "third",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @tie_a_path attributes {
      origin = "tie-a",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @tie_b_path attributes {
      origin = "tie-b",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @low_path attributes {
      origin = "low",
      requires = [@generic_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse ranking cost module");

  KernelOp kernel = findKernel(*module, "ranking_anchor");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  std::string beforeCollection = printModule(*module);
  unsigned variantCountBefore = countDirectVariants(kernel);

  llvm::SmallVector<std::string, 4> callOrder;
  CostPlugin third("third", 3.0);
  CostPlugin tieA("tie-a", 1.0);
  CostPlugin tieB("tie-b", 1.0);
  CostPlugin low("low", 0.5);
  third.setCallOrder(&callOrder);
  tieA.setCallOrder(&callOrder);
  tieB.setCallOrder(&callOrder);
  low.setCallOrder(&callOrder);

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(tieB), "register tie-b"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(low), "register low"))
    return result;
  if (int result =
          expectSuccess(registry.registerPlugin(third), "register third"))
    return result;
  if (int result =
          expectSuccess(registry.registerPlugin(tieA), "register tie-a"))
    return result;

  llvm::SmallVector<VariantCostRankingEntry, 4> collected;
  if (int result =
          expectSuccess(registry.collectKernelVariantCosts(kernel, capabilities,
                                                           collected),
                        "collect kernel variant costs"))
    return result;

  if (int result = expect(callOrder.size() == 4 &&
                              callOrder[0] == "third_path" &&
                              callOrder[1] == "tie_a_path" &&
                              callOrder[2] == "tie_b_path" &&
                              callOrder[3] == "low_path",
                          "cost collection visits direct variants in IR order"))
    return result;
  if (int result = expect(collected.size() == 4 &&
                              collected[0].estimate.getVariantSymbol() ==
                                  "third_path" &&
                              collected[1].estimate.getVariantSymbol() ==
                                  "tie_a_path" &&
                              collected[2].estimate.getVariantSymbol() ==
                                  "tie_b_path" &&
                              collected[3].estimate.getVariantSymbol() ==
                                  "low_path",
                          "collected costs preserve original IR order"))
    return result;
  if (int result = expect(countDirectVariants(kernel) == variantCountBefore,
                          "cost collection does not erase variants"))
    return result;
  if (int result = expect(printModule(*module) == beforeCollection,
                          "cost collection does not mutate IR"))
    return result;

  llvm::SmallVector<VariantCostRankingEntry, 4> ranked;
  if (int result =
          expectSuccess(registry.rankKernelVariantsByCost(kernel, capabilities,
                                                          ranked),
                        "rank kernel variant costs"))
    return result;

  if (int result = expect(ranked.size() == 4 &&
                              ranked[0].estimate.getVariantSymbol() ==
                                  "low_path" &&
                              ranked[1].estimate.getVariantSymbol() ==
                                  "tie_a_path" &&
                              ranked[2].estimate.getVariantSymbol() ==
                                  "tie_b_path" &&
                              ranked[3].estimate.getVariantSymbol() ==
                                  "third_path",
                          "ranking is deterministic and stable for ties"))
    return result;
  if (int result = expect(ranked[1].originalIndex < ranked[2].originalIndex,
                          "equal score ranking keeps original IR order key"))
    return result;

  return 0;
}

int runExplicitPreferenceAvailabilityRankingTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @preference_availability_anchor attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
    tcrv.exec.variant @default_no_preference attributes {
      origin = "default",
      requires = [@generic_base]
    } {
    }
    tcrv.exec.variant @explicit_preference attributes {
      origin = "explicit",
      requires = [@generic_base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse preference availability ranking module");

  KernelOp kernel = findKernel(*module, "preference_availability_anchor");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  DefaultCostPlugin defaultPlugin("default");
  CostPlugin explicitPlugin("explicit", 10.0);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(defaultPlugin),
                                 "register default no-preference plugin"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(explicitPlugin),
                                 "register explicit preference plugin"))
    return result;

  llvm::SmallVector<VariantCostRankingEntry, 2> ranked;
  if (int result =
          expectSuccess(registry.rankKernelVariantsByCost(kernel, capabilities,
                                                          ranked),
                        "rank explicit preference ahead of no-preference"))
    return result;

  if (int result =
          expect(ranked.size() == 2 &&
                     ranked[0].estimate.getVariantSymbol() ==
                         "explicit_preference" &&
                     ranked[0].estimate.hasExplicitPreference() &&
                     ranked[1].estimate.getVariantSymbol() ==
                         "default_no_preference" &&
                     !ranked[1].estimate.hasExplicitPreference(),
                 "explicit plugin preference availability outranks default no-preference"))
    return result;

  return 0;
}

int runNegativeCostTests(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @negative_anchor attributes {} {
    tcrv.exec.capability @generic_fast {
      id = "generic.fast",
      kind = "generic-feature"
    }
    tcrv.exec.variant @unknown_path attributes {
      origin = "missing-plugin",
      requires = [@generic_fast]
    } {
    }
    tcrv.exec.variant @disabled_path attributes {
      origin = "disabled",
      requires = [@generic_fast]
    } {
    }
    tcrv.exec.variant @failing_path attributes {
      origin = "failing",
      requires = [@generic_fast]
    } {
    }
    tcrv.exec.variant @well_formed_path attributes {
      origin = "well-formed",
      requires = [@generic_fast]
    } {
    }
  }
  tcrv.exec.kernel @other_anchor attributes {} {
    tcrv.exec.capability @generic_other {
      id = "generic.other",
      kind = "toolchain"
    }
    tcrv.exec.variant @other_path attributes {
      origin = "other",
      requires = [@generic_other]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse negative cost module");

  KernelOp kernel = findKernel(*module, "negative_anchor");
  KernelOp otherKernel = findKernel(*module, "other_anchor");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  TargetCapabilitySet emptyCapabilities =
      TargetCapabilitySet::buildFromKernel(KernelOp());

  {
    ExtensionPluginRegistry registry;
    VariantCostEstimate estimate;
    VariantCostRequest request(VariantOp(), kernel, capabilities);
    if (int result =
            expectErrorContains(registry.estimateVariantCost(request, estimate),
                                {"requires a materialized tcrv.exec.variant",
                                 "<missing>", "negative_anchor"}))
      return result;
  }

  {
    ExtensionPluginRegistry registry;
    VariantCostEstimate estimate;
    VariantCostRequest request(findDirectVariant(kernel, "well_formed_path"),
                               KernelOp(), emptyCapabilities);
    if (int result =
            expectErrorContains(registry.estimateVariantCost(request, estimate),
                                {"requires an enclosing tcrv.exec.kernel",
                                 "well_formed_path", "<missing>"}))
      return result;
  }

  {
    ExtensionPluginRegistry registry;
    VariantCostEstimate estimate;
    VariantCostRequest request(findDirectVariant(kernel, "well_formed_path"),
                               otherKernel, capabilities);
    if (int result =
            expectErrorContains(registry.estimateVariantCost(request, estimate),
                                {"variant is not enclosed by the request "
                                 "tcrv.exec.kernel",
                                 "well_formed_path", "other_anchor"}))
      return result;
  }

  {
    VariantOp variant = findDirectVariant(kernel, "well_formed_path");
    mlir::Attribute oldOrigin = variant->getAttr("origin");
    variant->removeAttr("origin");

    ExtensionPluginRegistry registry;
    VariantCostEstimate estimate;
    VariantCostRequest request(variant, kernel, capabilities);
    if (int result =
            expectErrorContains(registry.estimateVariantCost(request, estimate),
                                {"requires non-empty string attribute 'origin'",
                                 "well_formed_path", "negative_anchor"}))
      return result;

    variant->setAttr("origin", oldOrigin);
  }

  {
    VariantOp variant = findDirectVariant(kernel, "well_formed_path");
    mlir::Attribute oldOrigin = variant->getAttr("origin");
    variant->setAttr("origin", mlir::StringAttr::get(&context, ""));

    ExtensionPluginRegistry registry;
    VariantCostEstimate estimate;
    VariantCostRequest request(variant, kernel, capabilities);
    if (int result =
            expectErrorContains(registry.estimateVariantCost(request, estimate),
                                {"requires non-empty string attribute 'origin'",
                                 "well_formed_path", "negative_anchor"}))
      return result;

    variant->setAttr("origin", oldOrigin);
  }

  {
    ExtensionPluginRegistry registry;
    VariantCostEstimate estimate;
    VariantCostRequest request(findDirectVariant(kernel, "unknown_path"),
                               kernel, capabilities);
    if (int result =
            expectErrorContains(registry.estimateVariantCost(request, estimate),
                                {"unknown origin plugin 'missing-plugin'",
                                 "unknown_path", "negative_anchor"}))
      return result;
  }

  {
    CostPlugin disabled("disabled", 1.0, false);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(disabled),
                                   "register disabled cost plugin"))
      return result;

    VariantCostEstimate estimate;
    VariantCostRequest request(findDirectVariant(kernel, "disabled_path"),
                               kernel, capabilities);
    if (int result =
            expectErrorContains(registry.estimateVariantCost(request, estimate),
                                {"origin plugin 'disabled' is disabled",
                                 "disabled_path", "negative_anchor"}))
      return result;
    if (int result = expect(disabled.getCostCalls() == 0,
                            "disabled origin plugin hook is not called"))
      return result;
  }

  {
    CostPlugin failing("failing", 1.0, true, CostBehavior::Failure);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(failing),
                                   "register failing cost plugin"))
      return result;

    VariantCostEstimate estimate;
    VariantCostRequest request(findDirectVariant(kernel, "failing_path"),
                               kernel, capabilities);
    if (int result =
            expectErrorContains(registry.estimateVariantCost(request, estimate),
                                {"origin plugin 'failing' failed cost estimate",
                                 "failing_path", "negative_anchor",
                                 "plugin-local cost failed"}))
      return result;
    if (int result = expect(failing.getCostCalls() == 1,
                            "failing origin plugin hook is called once"))
      return result;
  }

  struct InvalidEstimateCase {
    llvm::StringRef pluginName;
    CostBehavior behavior;
    llvm::StringRef expectedFragment;
  };

  InvalidEstimateCase cases[] = {
      {"missing-score", CostBehavior::MissingScore, "score is missing"},
      {"non-finite", CostBehavior::NonFiniteScore, "score must be finite"},
      {"negative", CostBehavior::NegativeScore, "score must be non-negative"},
      {"empty-explanation", CostBehavior::EmptyExplanation,
       "explanation must be non-empty when present"},
      {"empty-policy", CostBehavior::EmptyPolicy,
       "policy must be non-empty when present"},
  };

  for (const InvalidEstimateCase &testCase : cases) {
    VariantOp variant = findDirectVariant(kernel, "well_formed_path");
    mlir::Attribute oldOrigin = variant->getAttr("origin");
    variant->setAttr("origin",
                     mlir::StringAttr::get(&context, testCase.pluginName));

    CostPlugin plugin(testCase.pluginName, 1.0, true, testCase.behavior);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   llvm::Twine("register ") +
                                       testCase.pluginName))
      return result;

    VariantCostEstimate estimate;
    VariantCostRequest request(variant, kernel, capabilities);
    if (int result =
            expectErrorContains(registry.estimateVariantCost(request, estimate),
                                {"produced invalid cost estimate",
                                 testCase.expectedFragment, "well_formed_path",
                                 "negative_anchor"}))
      return result;

    variant->setAttr("origin", oldOrigin);
  }

  {
    ExtensionPluginRegistry registry;
    llvm::SmallVector<VariantCostRankingEntry, 1> entries;
    if (int result =
            expectErrorContains(registry.collectKernelVariantCosts(KernelOp(),
                                                                   entries),
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

  if (int result = runSingleVariantCostRoutingTest(context))
    return result;
  if (int result = runDefaultNeutralEstimateTest(context))
    return result;
  if (int result = runKernelCollectionAndRankingTest(context))
    return result;
  if (int result = runExplicitPreferenceAvailabilityRankingTest(context))
    return result;
  if (int result = runNegativeCostTests(context))
    return result;

  llvm::outs() << "plugin variant cost smoke test passed\n";
  return 0;
}
