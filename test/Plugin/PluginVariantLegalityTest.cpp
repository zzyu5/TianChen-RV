#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
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
#include <string>

using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantLegalityRequest;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::CapabilityOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

namespace {

class LegalityPlugin final : public ExtensionPlugin {
public:
  LegalityPlugin(llvm::StringRef name, bool enabled = true,
                 llvm::StringRef failureMessage = {},
                 llvm::SmallVectorImpl<std::string> *callOrder = nullptr)
      : name(name.str()), enabled(enabled),
        failureMessage(failureMessage.str()), callOrder(callOrder) {}

  llvm::StringRef getName() const override { return name; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return enabled; }

  llvm::Error
  verifyVariantLegality(const VariantLegalityRequest &request) const override {
    ++legalityCalls;
    observedVariantSymbols.push_back(request.getVariant().getSymName().str());
    observedKernelSymbols.push_back(request.getKernel().getSymName().str());
    observedCapabilityCounts.push_back(request.getCapabilities().size());
    observedGenericAlphaAvailable =
        request.getCapabilities().isCapabilityAvailableByID("generic.alpha");

    if (const CapabilityDescriptor *capability =
            request.getCapabilities().lookupBySymbolName("generic_alpha")) {
      observedGenericAlphaKind = capability->getKind().str();
    }

    if (callOrder)
      callOrder->push_back(request.getVariant().getSymName().str());

    if (!failureMessage.empty())
      return llvm::make_error<llvm::StringError>(
          failureMessage, llvm::errc::invalid_argument);

    return llvm::Error::success();
  }

  unsigned getLegalityCalls() const { return legalityCalls; }
  llvm::ArrayRef<std::string> getObservedVariantSymbols() const {
    return observedVariantSymbols;
  }
  llvm::ArrayRef<std::string> getObservedKernelSymbols() const {
    return observedKernelSymbols;
  }
  llvm::ArrayRef<std::size_t> getObservedCapabilityCounts() const {
    return observedCapabilityCounts;
  }
  bool sawGenericAlphaAvailable() const {
    return observedGenericAlphaAvailable;
  }
  llvm::StringRef getObservedGenericAlphaKind() const {
    return observedGenericAlphaKind;
  }

private:
  std::string name;
  bool enabled;
  std::string failureMessage;
  llvm::SmallVectorImpl<std::string> *callOrder;
  llvm::SmallVector<PluginCapability, 1> capabilities;
  mutable unsigned legalityCalls = 0;
  mutable llvm::SmallVector<std::string, 4> observedVariantSymbols;
  mutable llvm::SmallVector<std::string, 4> observedKernelSymbols;
  mutable llvm::SmallVector<std::size_t, 4> observedCapabilityCounts;
  mutable bool observedGenericAlphaAvailable = false;
  mutable std::string observedGenericAlphaKind;
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
    return fail("expected variant legality error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("variant legality error text missing '") +
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

CapabilityOp findDirectCapability(KernelOp kernel, llvm::StringRef name) {
  if (!kernel || kernel.getBody().empty())
    return CapabilityOp();

  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto capability = llvm::dyn_cast<CapabilityOp>(operation);
    if (capability && capability.getSymName() == name)
      return capability;
  }
  return CapabilityOp();
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

std::string printModule(mlir::ModuleOp module) {
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module->print(stream);
  stream.flush();
  return printed;
}

int runSingleVariantRoutingTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @single_anchor attributes {} {
    tcrv.exec.capability @generic_alpha {
      id = "generic.alpha",
      kind = "toolchain"
    }
    tcrv.exec.capability @generic_beta {
      id = "generic.beta",
      kind = "runtime",
      status = "available"
    }
    tcrv.exec.variant @alpha_path attributes {
      origin = "alpha",
      requires = [@generic_alpha]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse single variant legality module");

  KernelOp kernel = findKernel(*module, "single_anchor");
  VariantOp variant = findDirectVariant(kernel, "alpha_path");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  LegalityPlugin alpha("alpha");
  LegalityPlugin nonOrigin("non-origin");
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(alpha), "register alpha"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(nonOrigin),
                                 "register non-origin"))
    return result;

  VariantLegalityRequest request(variant, kernel, capabilities);
  if (int result = expectSuccess(registry.verifyVariantLegality(request),
                                 "verify single variant legality"))
    return result;

  if (int result = expect(alpha.getLegalityCalls() == 1,
                          "origin plugin hook is called exactly once"))
    return result;
  if (int result = expect(nonOrigin.getLegalityCalls() == 0,
                          "non-origin plugin hook is not called"))
    return result;
  if (int result = expect(alpha.getObservedVariantSymbols().size() == 1 &&
                              alpha.getObservedVariantSymbols()[0] ==
                                  "alpha_path",
                          "hook receives expected variant symbol"))
    return result;
  if (int result = expect(alpha.getObservedKernelSymbols().size() == 1 &&
                              alpha.getObservedKernelSymbols()[0] ==
                                  "single_anchor",
                          "hook receives expected enclosing kernel"))
    return result;
  if (int result = expect(alpha.getObservedCapabilityCounts().size() == 1 &&
                              alpha.getObservedCapabilityCounts()[0] == 2 &&
                              alpha.sawGenericAlphaAvailable() &&
                              alpha.getObservedGenericAlphaKind() ==
                                  "toolchain",
                          "hook receives expected TargetCapabilitySet"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "successful legality keeps module valid"))
    return result;

  return 0;
}

int runKernelOrderAndNoMutationTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @ordered_anchor attributes {} {
    tcrv.exec.capability @generic_alpha {
      id = "generic.alpha",
      kind = "toolchain"
    }
    tcrv.exec.capability @generic_beta {
      id = "generic.beta",
      kind = "runtime"
    }
    tcrv.exec.variant @first_path attributes {
      origin = "first",
      requires = [@generic_alpha]
    } {
    }
    tcrv.exec.variant @second_path attributes {
      origin = "second",
      requires = [@generic_beta]
    } {
    }
    tcrv.exec.dispatch attributes {} {
      tcrv.exec.case @second_path {policy = "preserve_existing_dispatch"}
      tcrv.exec.fallback @first_path
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse ordered legality module");

  KernelOp kernel = findKernel(*module, "ordered_anchor");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  std::string beforeVerification = printModule(*module);
  unsigned dispatchCountBefore = countDirectDispatches(kernel);

  llvm::SmallVector<std::string, 4> callOrder;
  LegalityPlugin first("first", true, {}, &callOrder);
  LegalityPlugin second("second", true, {}, &callOrder);
  LegalityPlugin nonOrigin("non-origin", true, {}, &callOrder);
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(first), "register first"))
    return result;
  if (int result =
          expectSuccess(registry.registerPlugin(nonOrigin),
                        "register non-origin"))
    return result;
  if (int result =
          expectSuccess(registry.registerPlugin(second), "register second"))
    return result;

  if (int result = expectSuccess(
          registry.verifyKernelVariantLegality(kernel, capabilities),
          "verify kernel variant legality"))
    return result;

  if (int result = expect(callOrder.size() == 2 &&
                              callOrder[0] == "first_path" &&
                              callOrder[1] == "second_path",
                          "kernel verification follows direct variant IR order"))
    return result;
  if (int result = expect(first.getLegalityCalls() == 1 &&
                              second.getLegalityCalls() == 1 &&
                              nonOrigin.getLegalityCalls() == 0,
                          "kernel verification calls only origin plugins"))
    return result;
  if (int result = expect(countDirectDispatches(kernel) == dispatchCountBefore,
                          "legality verification preserves dispatch structure"))
    return result;
  if (int result =
          expect(printModule(*module) == beforeVerification,
                 "legality verification does not mutate materialized IR"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "verified kernel module remains valid"))
    return result;

  return 0;
}

int runNegativeLegalityTests(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @negative_anchor attributes {} {
    tcrv.exec.capability @generic_alpha {
      id = "generic.alpha",
      kind = "toolchain"
    }
    tcrv.exec.variant @unknown_path attributes {
      origin = "missing-plugin",
      requires = [@generic_alpha]
    } {
    }
    tcrv.exec.variant @disabled_path attributes {
      origin = "disabled",
      requires = [@generic_alpha]
    } {
    }
    tcrv.exec.variant @failing_path attributes {
      origin = "rejecting",
      requires = [@generic_alpha]
    } {
    }
    tcrv.exec.variant @well_formed_path attributes {
      origin = "well-formed",
      requires = [@generic_alpha]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse negative legality module");

  KernelOp kernel = findKernel(*module, "negative_anchor");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  TargetCapabilitySet emptyCapabilities =
      TargetCapabilitySet::buildFromKernel(KernelOp());

  {
    ExtensionPluginRegistry registry;
    VariantLegalityRequest request(findDirectVariant(kernel, "unknown_path"),
                                   kernel, capabilities);
    if (int result =
            expectErrorContains(registry.verifyVariantLegality(request),
                                {"unknown origin plugin 'missing-plugin'",
                                 "unknown_path", "negative_anchor"}))
      return result;
  }

  {
    LegalityPlugin disabled("disabled", false);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(disabled),
                                   "register disabled legality plugin"))
      return result;
    VariantLegalityRequest request(findDirectVariant(kernel, "disabled_path"),
                                   kernel, capabilities);
    if (int result =
            expectErrorContains(registry.verifyVariantLegality(request),
                                {"origin plugin 'disabled' is disabled",
                                 "disabled_path", "negative_anchor"}))
      return result;
    if (int result = expect(disabled.getLegalityCalls() == 0,
                            "disabled origin plugin hook is not called"))
      return result;
  }

  {
    LegalityPlugin rejecting("rejecting", true,
                             "plugin-local legality failed");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(rejecting),
                                   "register rejecting legality plugin"))
      return result;
    VariantLegalityRequest request(findDirectVariant(kernel, "failing_path"),
                                   kernel, capabilities);
    if (int result =
            expectErrorContains(registry.verifyVariantLegality(request),
                                {"origin plugin 'rejecting' rejected variant",
                                 "failing_path", "negative_anchor",
                                 "plugin-local legality failed"}))
      return result;
    if (int result = expect(rejecting.getLegalityCalls() == 1,
                            "rejecting origin plugin hook is called once"))
      return result;
  }

  {
    ExtensionPluginRegistry registry;
    VariantLegalityRequest request(VariantOp(), kernel, capabilities);
    if (int result =
            expectErrorContains(registry.verifyVariantLegality(request),
                                {"requires a materialized tcrv.exec.variant",
                                 "<missing>", "negative_anchor"}))
      return result;
  }

  {
    ExtensionPluginRegistry registry;
    VariantLegalityRequest request(
        findDirectVariant(kernel, "well_formed_path"), KernelOp(),
        emptyCapabilities);
    if (int result =
            expectErrorContains(registry.verifyVariantLegality(request),
                                {"requires an enclosing tcrv.exec.kernel",
                                 "well_formed_path", "<missing>"}))
      return result;
  }

  {
    ExtensionPluginRegistry registry;
    if (int result =
            expectErrorContains(registry.verifyKernelVariantLegality(KernelOp()),
                                {"requires a tcrv.exec.kernel", "<missing>"}))
      return result;
  }

  return 0;
}

int runDuplicateCapabilitySetStopsLegalityConsumerTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @duplicate_identity_anchor attributes {} {
    tcrv.exec.capability @generic_alpha {
      id = "generic.alpha",
      kind = "toolchain"
    }
    tcrv.exec.capability @generic_beta {
      id = "generic.beta",
      kind = "toolchain"
    }
    tcrv.exec.variant @alpha_path attributes {
      origin = "alpha",
      requires = [@generic_alpha]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse duplicate identity legality module");

  KernelOp kernel = findKernel(*module, "duplicate_identity_anchor");
  CapabilityOp beta = findDirectCapability(kernel, "generic_beta");
  if (int result = expect(static_cast<bool>(beta),
                          "second capability is present before mutation"))
    return result;

  beta->setAttr("id", mlir::StringAttr::get(&context, "generic.alpha"));

  LegalityPlugin alpha("alpha");
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(alpha), "register alpha"))
    return result;

  if (int result = expectErrorContains(
          registry.verifyKernelVariantLegality(kernel),
          {"TargetCapabilitySet", "kernel extraction from "
                                  "@duplicate_identity_anchor",
           "duplicate capability id \"generic.alpha\"", "@generic_beta",
           "existing symbol @generic_alpha"}))
    return result;
  if (int result = expect(
          alpha.getLegalityCalls() == 0,
          "duplicate capability set construction fails before legality hook"))
    return result;

  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runSingleVariantRoutingTest(context))
    return result;
  if (int result = runKernelOrderAndNoMutationTest(context))
    return result;
  if (int result = runNegativeLegalityTests(context))
    return result;
  if (int result = runDuplicateCapabilitySetStopsLegalityConsumerTest(context))
    return result;

  llvm::outs() << "plugin variant legality smoke test passed\n";
  return 0;
}
