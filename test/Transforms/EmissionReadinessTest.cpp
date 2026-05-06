#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/EmissionReadiness.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
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
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmissionStatus;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

namespace {

enum class EmissionBehavior {
  Supported,
  PluginFailure,
  MissingStatus,
  SupportedEmptyPath,
  UnsupportedEmptyReason,
};

class EmissionPlugin final : public ExtensionPlugin {
public:
  EmissionPlugin(llvm::StringRef name, bool enabled = true,
                 EmissionBehavior behavior = EmissionBehavior::Supported)
      : name(name.str()), enabled(enabled), behavior(behavior) {}

  llvm::StringRef getName() const override { return name; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return enabled; }

  llvm::Error checkVariantEmissionReadiness(
      const VariantEmissionRequest &request,
      VariantEmissionStatus &out) const override {
    ++readinessCalls;
    observedKernelSymbols.push_back(request.getKernel().getSymName().str());
    observedVariantSymbols.push_back(request.getVariant().getSymName().str());
    observedRoles.push_back(
        tianchenrv::plugin::stringifyVariantEmissionRole(request.getRole())
            .str());
    observedCapabilityCounts.push_back(request.getCapabilities().size());

    if (behavior == EmissionBehavior::PluginFailure)
      return llvm::make_error<llvm::StringError>(
          "plugin-local emission backend selection failed",
          llvm::errc::invalid_argument);

    if (behavior == EmissionBehavior::MissingStatus) {
      out = VariantEmissionStatus();
      return llvm::Error::success();
    }

    if (behavior == EmissionBehavior::SupportedEmptyPath) {
      out = VariantEmissionStatus::getSupported(
          name, request.getVariant().getSymName(), "");
      return llvm::Error::success();
    }

    if (behavior == EmissionBehavior::UnsupportedEmptyReason) {
      out = VariantEmissionStatus::getUnsupported(
          name, request.getVariant().getSymName(), "");
      return llvm::Error::success();
    }

    std::string path;
    llvm::raw_string_ostream stream(path);
    stream << name << "::emit::"
           << tianchenrv::plugin::stringifyVariantEmissionRole(
                  request.getRole())
           << "::" << request.getVariant().getSymName();
    stream.flush();
    out = VariantEmissionStatus::getSupported(
        name, request.getVariant().getSymName(), path);
    return llvm::Error::success();
  }

  unsigned getReadinessCalls() const { return readinessCalls; }
  llvm::ArrayRef<std::string> getObservedKernelSymbols() const {
    return observedKernelSymbols;
  }
  llvm::ArrayRef<std::string> getObservedVariantSymbols() const {
    return observedVariantSymbols;
  }
  llvm::ArrayRef<std::string> getObservedRoles() const {
    return observedRoles;
  }
  llvm::ArrayRef<unsigned> getObservedCapabilityCounts() const {
    return observedCapabilityCounts;
  }

private:
  std::string name;
  bool enabled;
  EmissionBehavior behavior;
  llvm::SmallVector<PluginCapability, 1> capabilities;
  mutable unsigned readinessCalls = 0;
  mutable llvm::SmallVector<std::string, 4> observedKernelSymbols;
  mutable llvm::SmallVector<std::string, 4> observedVariantSymbols;
  mutable llvm::SmallVector<std::string, 4> observedRoles;
  mutable llvm::SmallVector<unsigned, 4> observedCapabilityCounts;
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
    return fail("expected emission readiness error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("emission readiness error text missing '") +
                  fragment + "': " + message);
  }
  return 0;
}

void registerCoreDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  context.appendDialectRegistry(dialectRegistry);
  context.loadAllAvailableDialects();
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

DispatchOp findDirectDispatch(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return DispatchOp();

  for (mlir::Operation &operation : kernel.getBody().front()) {
    if (auto dispatch = llvm::dyn_cast<DispatchOp>(operation))
      return dispatch;
  }
  return DispatchOp();
}

DispatchCaseOp findFirstDispatchCase(DispatchOp dispatch) {
  if (!dispatch || dispatch.getBody().empty())
    return DispatchCaseOp();

  for (mlir::Operation &operation : dispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(operation))
      return dispatchCase;
  }
  return DispatchCaseOp();
}

FallbackOp findFallback(DispatchOp dispatch) {
  if (!dispatch || dispatch.getBody().empty())
    return FallbackOp();

  for (mlir::Operation &operation : dispatch.getBody().front()) {
    if (auto fallback = llvm::dyn_cast<FallbackOp>(operation))
      return fallback;
  }
  return FallbackOp();
}

mlir::FlatSymbolRefAttr getSymbolRef(mlir::MLIRContext &context,
                                     llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(&context, symbol);
}

const char *getDirectKernelSource(llvm::StringRef kernelName = "direct") {
  (void)kernelName;
  return R"mlir(
module {
  tcrv.exec.kernel @direct {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
  }
}
)mlir";
}

int runRegistrySupportedDirectVariantTest(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, getDirectKernelSource());
  if (!module)
    return fail("failed to parse direct emission readiness module");

  KernelOp kernel = findKernel(*module, "direct");
  VariantOp variant = findDirectVariant(kernel, "fast");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(plugin),
                        "register supported emission readiness plugin"))
    return result;

  VariantEmissionStatus status;
  VariantEmissionRequest request(variant, kernel, capabilities,
                                 VariantEmissionRole::DirectVariant);
  if (int result = expectSuccess(
          registry.checkVariantEmissionReadiness(request, status),
          "registry routes direct variant emission readiness"))
    return result;

  if (int result = expect(status.isSupported(),
                          "supported emission status is preserved"))
    return result;
  if (int result = expect(status.getOriginPlugin() == "mock-emitter",
                          "emission status preserves plugin context"))
    return result;
  if (int result = expect(status.getVariantSymbol() == "fast",
                          "emission status preserves variant context"))
    return result;
  if (int result = expect(status.getEmissionPath().contains(
                              "mock-emitter::emit::direct variant::fast"),
                          "emission status preserves plugin-owned path"))
    return result;
  if (int result = expect(plugin.getObservedKernelSymbols()[0] == "direct",
                          "request preserves kernel context"))
    return result;
  if (int result = expect(plugin.getObservedCapabilityCounts()[0] == 1,
                          "request carries target capability set"))
    return result;

  return 0;
}

int runInjectedDirectPassTest(mlir::MLIRContext &context) {
  const char *source = R"mlir(
module {
  tcrv.exec.kernel @direct_pass {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.variant @fallback attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse injected direct pass module");

  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(plugin),
                                 "register direct pass plugin"))
    return result;

  mlir::PassManager passManager(&context);
  passManager.addPass(
      tianchenrv::transforms::createCheckEmissionPathsPass(registry));
  if (int result =
          expect(mlir::succeeded(passManager.run(*module)),
                 "injected registry direct emission pass succeeds"))
    return result;

  if (int result = expect(plugin.getReadinessCalls() == 2,
                          "direct pass checks every direct variant"))
    return result;
  return expect(plugin.getObservedRoles()[0] == "direct variant",
                "direct pass uses direct variant role");
}

int runInjectedDispatchPassTest(mlir::MLIRContext &context) {
  const char *source = R"mlir(
module {
  tcrv.exec.kernel @dispatch_pass {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base],
      policy = "fast_path"
    } {
    }
    tcrv.exec.variant @medium attributes {
      origin = "mock-emitter",
      requires = [@base],
      policy = "medium_path"
    } {
    }
    tcrv.exec.variant @fallback attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @fast {policy = "fast_path"}
      tcrv.exec.case @medium {policy = "medium_path"}
      tcrv.exec.fallback @fallback
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse injected dispatch pass module");

  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(plugin),
                                 "register dispatch pass plugin"))
    return result;

  mlir::PassManager passManager(&context);
  passManager.addPass(
      tianchenrv::transforms::createCheckEmissionPathsPass(registry));
  if (int result =
          expect(mlir::succeeded(passManager.run(*module)),
                 "injected registry dispatch emission pass succeeds"))
    return result;

  if (int result = expect(plugin.getReadinessCalls() == 3,
                          "dispatch pass checks every case and fallback"))
    return result;
  if (int result = expect(plugin.getObservedRoles()[0] == "dispatch case",
                          "dispatch pass checks case role"))
    return result;
  return expect(plugin.getObservedRoles()[2] == "dispatch fallback",
                "dispatch pass checks fallback role");
}

int runRegistryNegativeTests(mlir::MLIRContext &context) {
  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register missing variant plugin"))
      return result;

    VariantEmissionStatus status;
    if (int result = expectErrorContains(
            registry.checkVariantEmissionReadiness(
                VariantEmissionRequest(VariantOp(), kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                status),
            {"materialized tcrv.exec.variant", "kernel @direct"}))
      return result;
  }

  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register missing kernel plugin"))
      return result;

    VariantEmissionStatus status;
    if (int result = expectErrorContains(
            registry.checkVariantEmissionReadiness(
                VariantEmissionRequest(variant, KernelOp(), capabilities,
                                       VariantEmissionRole::DirectVariant),
                status),
            {"variant @fast", "kernel <missing>",
             "enclosing tcrv.exec.kernel"}))
      return result;
  }

  {
    const char *source = R"mlir(
module {
  tcrv.exec.kernel @left {
    tcrv.exec.capability @base {id = "generic.base", kind = "generic"}
    tcrv.exec.variant @fast attributes {origin = "mock-emitter", requires = [@base]} {
    }
  }
  tcrv.exec.kernel @right {
    tcrv.exec.capability @base {id = "generic.base", kind = "generic"}
  }
}
)mlir";
    mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
    KernelOp left = findKernel(*module, "left");
    KernelOp right = findKernel(*module, "right");
    VariantOp variant = findDirectVariant(left, "fast");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(right);
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register cross-kernel plugin"))
      return result;

    VariantEmissionStatus status;
    if (int result = expectErrorContains(
            registry.checkVariantEmissionReadiness(
                VariantEmissionRequest(variant, right, capabilities,
                                       VariantEmissionRole::DirectVariant),
                status),
            {"variant @fast", "kernel @right", "not directly enclosed"}))
      return result;
  }

  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    variant->removeAttr("origin");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register missing origin plugin"))
      return result;

    VariantEmissionStatus status;
    if (int result = expectErrorContains(
            registry.checkVariantEmissionReadiness(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                status),
            {"variant @fast", "kernel @direct",
             "non-empty string attribute 'origin'"}))
      return result;
  }

  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    variant->setAttr("origin", mlir::StringAttr::get(&context, ""));
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register empty origin plugin"))
      return result;

    VariantEmissionStatus status;
    if (int result = expectErrorContains(
            registry.checkVariantEmissionReadiness(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                status),
            {"variant @fast", "non-empty string attribute 'origin'"}))
      return result;
  }

  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    variant->setAttr("origin",
                     mlir::StringAttr::get(&context, "missing-emitter"));
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

    VariantEmissionStatus status;
    ExtensionPluginRegistry registry;
    if (int result = expectErrorContains(
            registry.checkVariantEmissionReadiness(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                status),
            {"variant @fast", "unknown origin plugin 'missing-emitter'"}))
      return result;
  }

  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    variant->setAttr("origin",
                     mlir::StringAttr::get(&context, "disabled-emitter"));
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("disabled-emitter", false);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register disabled emission plugin"))
      return result;

    VariantEmissionStatus status;
    if (int result = expectErrorContains(
            registry.checkVariantEmissionReadiness(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                status),
            {"variant @fast", "origin plugin 'disabled-emitter' is disabled"}))
      return result;
  }

  for (EmissionBehavior behavior :
       {EmissionBehavior::MissingStatus, EmissionBehavior::SupportedEmptyPath,
        EmissionBehavior::UnsupportedEmptyReason,
        EmissionBehavior::PluginFailure}) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter", true, behavior);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register malformed emission plugin"))
      return result;

    VariantEmissionStatus status;
    llvm::Error error = registry.checkVariantEmissionReadiness(
        VariantEmissionRequest(variant, kernel, capabilities,
                               VariantEmissionRole::DirectVariant),
        status);
    if (behavior == EmissionBehavior::PluginFailure) {
      if (int result = expectErrorContains(
              std::move(error),
              {"failed emission readiness query",
               "plugin-local emission backend selection failed"}))
        return result;
      continue;
    }

    if (int result = expectErrorContains(
            std::move(error),
            {"invalid emission readiness result", "origin plugin"}))
      return result;
  }

  return 0;
}

int expectStructuralErrorHasNoPluginCalls(
    mlir::MLIRContext &context, llvm::StringRef source,
    void (*mutate)(mlir::MLIRContext &, KernelOp),
    std::initializer_list<llvm::StringRef> fragments) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse structural negative module");

  KernelOp kernel = findKernel(*module, "dispatch_negative");
  mutate(context, kernel);

  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(plugin),
                                 "register structural negative plugin"))
    return result;

  if (int result = expectErrorContains(
          tianchenrv::transforms::checkKernelEmissionPaths(kernel, registry),
          fragments))
    return result;

  return expect(plugin.getReadinessCalls() == 0,
                "dispatch structural failures are diagnosed before plugin "
                "routing");
}

const char *getDispatchKernelSource() {
  return R"mlir(
module {
  tcrv.exec.kernel @dispatch_negative {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.variant @fallback attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @fast
      tcrv.exec.fallback @fallback
    }
  }
}
)mlir";
}

int runStructuralDispatchNegativeTests(mlir::MLIRContext &context) {
  if (int result = expectStructuralErrorHasNoPluginCalls(
          context, getDispatchKernelSource(),
          [](mlir::MLIRContext &, KernelOp kernel) {
            DispatchCaseOp dispatchCase =
                findFirstDispatchCase(findDirectDispatch(kernel));
            dispatchCase->removeAttr("target");
          },
          {"missing a variant symbol reference target"}))
    return result;

  if (int result = expectStructuralErrorHasNoPluginCalls(
          context, getDispatchKernelSource(),
          [](mlir::MLIRContext &context, KernelOp kernel) {
            DispatchCaseOp dispatchCase =
                findFirstDispatchCase(findDirectDispatch(kernel));
            dispatchCase->setAttr("target",
                                  getSymbolRef(context, "does_not_exist"));
          },
          {"dispatch target @does_not_exist",
           "does not resolve to a direct sibling tcrv.exec.variant"}))
    return result;

  if (int result = expectStructuralErrorHasNoPluginCalls(
          context, getDispatchKernelSource(),
          [](mlir::MLIRContext &context, KernelOp kernel) {
            DispatchCaseOp dispatchCase =
                findFirstDispatchCase(findDirectDispatch(kernel));
            dispatchCase->setAttr("target", getSymbolRef(context, "base"));
          },
          {"dispatch target @base", "not a tcrv.exec.variant"}))
    return result;

  const char *nestedSource = R"mlir(
module {
  tcrv.exec.kernel @dispatch_negative {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @outer attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
      tcrv.exec.variant @nested attributes {
        origin = "mock-emitter",
        requires = [@base]
      } {
      }
    }
    tcrv.exec.variant @fallback attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @outer
      tcrv.exec.fallback @fallback
    }
  }
}
)mlir";
  if (int result = expectStructuralErrorHasNoPluginCalls(
          context, nestedSource,
          [](mlir::MLIRContext &context, KernelOp kernel) {
            DispatchCaseOp dispatchCase =
                findFirstDispatchCase(findDirectDispatch(kernel));
            dispatchCase->setAttr("target", getSymbolRef(context, "nested"));
          },
          {"dispatch target @nested", "not a direct sibling"}))
    return result;

  if (int result = expectStructuralErrorHasNoPluginCalls(
          context, getDispatchKernelSource(),
          [](mlir::MLIRContext &context, KernelOp kernel) {
            FallbackOp fallback = findFallback(findDirectDispatch(kernel));
            fallback->setAttr("target", getSymbolRef(context, "fast"));
          },
          {"duplicate dispatch emission reference to variant @fast"}))
    return result;

  return 0;
}

int runRVVUnsupportedEmissionTest() {
  mlir::MLIRContext context;
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);

  tianchenrv::plugin::rvv::RVVExtensionPlugin rvvPlugin;
  rvvPlugin.registerDialects(dialectRegistry);
  context.appendDialectRegistry(dialectRegistry);
  context.loadAllAvailableDialects();

  const char *source = R"mlir(
module {
  tcrv.exec.kernel @rvv_emission {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV emission readiness module");

  KernelOp kernel = findKernel(*module, "rvv_emission");
  VariantOp variant = findDirectVariant(kernel, "rvv_first_slice");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  VariantEmissionStatus pluginStatus;
  VariantEmissionRequest request(variant, kernel, capabilities,
                                 VariantEmissionRole::DirectVariant);
  if (int result = expectSuccess(
          rvvPlugin.checkVariantEmissionReadiness(request, pluginStatus),
          "RVV plugin returns explicit unsupported emission status"))
    return result;
  if (int result = expect(pluginStatus.isUnsupported(),
                          "RVV metadata-only first slice is unsupported for "
                          "emission"))
    return result;
  if (int result =
          expect(pluginStatus.getReason().contains("no RVV lowering"),
                 "RVV unsupported reason names missing lowering boundary"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(rvvPlugin),
                        "register RVV plugin for emission readiness"))
    return result;

  VariantEmissionStatus registryStatus;
  if (int result = expectErrorContains(
          registry.checkVariantEmissionReadiness(request, registryStatus),
          {"rvv-plugin", "kernel @rvv_emission", "variant @rvv_first_slice",
           "unsupported emission path", "metadata-only",
           "no RVV lowering, runtime ABI, or executable emission path",
           "not RVV hardware/toolchain/runtime/correctness/performance "
           "evidence"}))
    return result;

  return 0;
}

} // namespace

int main() {
  mlir::MLIRContext context;
  registerCoreDialects(context);

  if (int result = runRegistrySupportedDirectVariantTest(context))
    return result;
  if (int result = runInjectedDirectPassTest(context))
    return result;
  if (int result = runInjectedDispatchPassTest(context))
    return result;
  if (int result = runRegistryNegativeTests(context))
    return result;
  if (int result = runStructuralDispatchNegativeTests(context))
    return result;
  if (int result = runRVVUnsupportedEmissionTest())
    return result;

  llvm::outs() << "emission readiness tests passed\n";
  return 0;
}
