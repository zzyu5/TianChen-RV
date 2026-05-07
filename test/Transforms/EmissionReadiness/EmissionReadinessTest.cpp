#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
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
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmissionStatus;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

namespace {
namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;

enum class EmissionBehavior {
  Supported,
  PluginFailure,
  MissingStatus,
  SupportedEmptyPath,
  UnsupportedEmptyReason,
};

enum class EmissionPlanBehavior {
  Supported,
  Unsupported,
  PluginFailure,
  MissingStatus,
  SupportedEmptyEmissionKind,
  SupportedEmptyLoweringPipeline,
  SupportedEmptyRuntimeABI,
  SupportedEmptyRuntimeABIKind,
  SupportedEmptyRuntimeABIName,
  SupportedEmptyRuntimeGlueRole,
  SupportedEmptyArtifactKind,
  SupportedEmptyExplanation,
  SupportedUnboundedExplanation,
  MissingRequiredCapabilityRefs,
  MismatchedRequiredCapabilityRef,
  UnsupportedEmptyDiagnostic,
  MismatchedVariantSymbol,
  MismatchedRole,
};

class EmissionPlugin final : public ExtensionPlugin {
public:
  EmissionPlugin(llvm::StringRef name, bool enabled = true,
                 EmissionBehavior behavior = EmissionBehavior::Supported,
                 EmissionPlanBehavior planBehavior =
                     EmissionPlanBehavior::Supported)
      : name(name.str()), enabled(enabled), behavior(behavior),
        planBehavior(planBehavior) {}

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

  llvm::Error buildVariantEmissionPlan(
      const VariantEmissionRequest &request,
      VariantEmissionPlan &out) const override {
    ++planCalls;
    observedPlanKernelSymbols.push_back(request.getKernel().getSymName().str());
    observedPlanVariantSymbols.push_back(
        request.getVariant().getSymName().str());
    observedPlanRoles.push_back(
        tianchenrv::plugin::stringifyVariantEmissionRole(request.getRole())
            .str());

    if (planBehavior == EmissionPlanBehavior::PluginFailure)
      return llvm::make_error<llvm::StringError>(
          "plugin-local emission plan construction failed",
          llvm::errc::invalid_argument);

    if (planBehavior == EmissionPlanBehavior::MissingStatus) {
      out = VariantEmissionPlan();
      return llvm::Error::success();
    }

    if (planBehavior == EmissionPlanBehavior::Unsupported ||
        planBehavior == EmissionPlanBehavior::UnsupportedEmptyDiagnostic) {
      out = VariantEmissionPlan::getUnsupported(
          name, request.getKernel().getSymName(),
          request.getVariant().getSymName(), request.getRole(),
          planBehavior == EmissionPlanBehavior::Unsupported
              ? "mock plugin reports unsupported selected emission path"
              : "");
      out.setRuntimeABIKind("mock-unsupported-runtime-abi-kind");
      out.setRuntimeABIName("mock.unsupported.runtime.abi.v1");
      out.setRuntimeGlueRole("mock-unsupported-runtime-glue-role");
      if (planBehavior != EmissionPlanBehavior::MissingRequiredCapabilityRefs)
        if (llvm::Error error =
                out.setRequiredCapabilitySymbolsFromVariant(
                    request.getVariant()))
          return error;
      return llvm::Error::success();
    }

    llvm::StringRef variantSymbol = request.getVariant().getSymName();
    if (planBehavior == EmissionPlanBehavior::MismatchedVariantSymbol)
      variantSymbol = "wrong_variant";

    VariantEmissionRole role = request.getRole();
    if (planBehavior == EmissionPlanBehavior::MismatchedRole)
      role = request.getRole() == VariantEmissionRole::DispatchCase
                 ? VariantEmissionRole::DirectVariant
                 : VariantEmissionRole::DispatchCase;

    std::string explanation =
        planBehavior == EmissionPlanBehavior::SupportedUnboundedExplanation
            ? std::string(600, 'x')
            : "mock plugin-owned lowering/runtime route for selected path";

    out = VariantEmissionPlan::getSupported(
        name, request.getKernel().getSymName(), variantSymbol, role,
        planBehavior == EmissionPlanBehavior::SupportedEmptyEmissionKind
            ? ""
            : "metadata-intent",
        planBehavior == EmissionPlanBehavior::SupportedEmptyLoweringPipeline
            ? ""
            : "mock.lowering.pipeline.v1",
        planBehavior == EmissionPlanBehavior::SupportedEmptyRuntimeABI
            ? ""
            : "mock.runtime.abi.v1",
        planBehavior == EmissionPlanBehavior::SupportedEmptyArtifactKind
            ? ""
            : "compiler-emission-plan",
        planBehavior == EmissionPlanBehavior::SupportedEmptyExplanation
            ? ""
            : explanation);
    out.setRuntimeABIKind(
        planBehavior == EmissionPlanBehavior::SupportedEmptyRuntimeABIKind
            ? ""
            : "mock-runtime-abi-kind");
    out.setRuntimeABIName(
        planBehavior == EmissionPlanBehavior::SupportedEmptyRuntimeABIName
            ? ""
            : "mock.runtime.abi.v1");
    out.setRuntimeGlueRole(
        planBehavior == EmissionPlanBehavior::SupportedEmptyRuntimeGlueRole
            ? ""
            : "mock-runtime-glue-role");
    if (planBehavior != EmissionPlanBehavior::MissingRequiredCapabilityRefs)
      if (llvm::Error error =
              out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
        return error;
    if (planBehavior == EmissionPlanBehavior::MismatchedRequiredCapabilityRef) {
      out.clearRequiredCapabilitySymbols();
      out.addRequiredCapabilitySymbol("other_capability");
    }
    return llvm::Error::success();
  }

  unsigned getReadinessCalls() const { return readinessCalls; }
  unsigned getPlanCalls() const { return planCalls; }
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
  llvm::ArrayRef<std::string> getObservedPlanKernelSymbols() const {
    return observedPlanKernelSymbols;
  }
  llvm::ArrayRef<std::string> getObservedPlanVariantSymbols() const {
    return observedPlanVariantSymbols;
  }
  llvm::ArrayRef<std::string> getObservedPlanRoles() const {
    return observedPlanRoles;
  }

private:
  std::string name;
  bool enabled;
  EmissionBehavior behavior;
  EmissionPlanBehavior planBehavior;
  llvm::SmallVector<PluginCapability, 1> capabilities;
  mutable unsigned readinessCalls = 0;
  mutable unsigned planCalls = 0;
  mutable llvm::SmallVector<std::string, 4> observedKernelSymbols;
  mutable llvm::SmallVector<std::string, 4> observedVariantSymbols;
  mutable llvm::SmallVector<std::string, 4> observedRoles;
  mutable llvm::SmallVector<unsigned, 4> observedCapabilityCounts;
  mutable llvm::SmallVector<std::string, 4> observedPlanKernelSymbols;
  mutable llvm::SmallVector<std::string, 4> observedPlanVariantSymbols;
  mutable llvm::SmallVector<std::string, 4> observedPlanRoles;
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
  if (!error) {
    std::string expected;
    llvm::raw_string_ostream stream(expected);
    for (llvm::StringRef fragment : fragments) {
      if (!expected.empty())
        stream << ", ";
      stream << fragment;
    }
    stream.flush();
    return fail(llvm::Twine("expected emission readiness error containing: ") +
                expected);
  }

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("emission readiness error text missing '") +
                  fragment + "': " + message);
  }
  return 0;
}

int expectSupportedPlan(const VariantEmissionPlan &plan,
                        llvm::StringRef expectedKernel,
                        llvm::StringRef expectedVariant,
                        VariantEmissionRole expectedRole) {
  if (int result = expect(plan.isSupported(), "emission plan is supported"))
    return result;
  if (int result = expect(plan.getOriginPlugin() == "mock-emitter",
                          "emission plan preserves origin plugin"))
    return result;
  if (int result = expect(plan.getKernelSymbol() == expectedKernel,
                          "emission plan preserves kernel symbol"))
    return result;
  if (int result = expect(plan.getVariantSymbol() == expectedVariant,
                          "emission plan preserves variant symbol"))
    return result;
  if (int result = expect(plan.getRole() == expectedRole,
                          "emission plan preserves selected-path role"))
    return result;
  if (int result = expect(plan.getEmissionKind() == "metadata-intent",
                          "emission plan carries generic emission kind"))
    return result;
  if (int result =
          expect(plan.getLoweringPipeline() == "mock.lowering.pipeline.v1",
                 "emission plan carries lowering pipeline id"))
    return result;
  if (int result =
          expect(plan.getRuntimeABI() == "mock.runtime.abi.v1",
                 "emission plan carries runtime ABI id"))
    return result;
  if (int result =
          expect(plan.getRuntimeABIKind() == "mock-runtime-abi-kind",
                 "emission plan carries runtime ABI kind"))
    return result;
  if (int result =
          expect(plan.getRuntimeABIName() == "mock.runtime.abi.v1",
                 "emission plan carries runtime ABI name"))
    return result;
  if (int result =
          expect(plan.getRuntimeGlueRole() == "mock-runtime-glue-role",
                 "emission plan carries runtime glue role"))
    return result;
  if (int result =
          expect(plan.getRequiredCapabilitySymbols().size() == 1 &&
                     plan.getRequiredCapabilitySymbols().front() == "base",
                 "emission plan carries required capability refs"))
    return result;
  if (int result =
          expect(plan.getArtifactKind() == "compiler-emission-plan",
                 "emission plan carries artifact kind"))
    return result;
  return expect(plan.getExplanation().contains("plugin-owned"),
                "emission plan carries explanation");
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

bool isEmissionPlanDiagnostic(DiagnosticOp diagnostic) {
  auto reason = diagnostic->getAttrOfType<mlir::StringAttr>(
      execDiagnostic::kReasonAttrName);
  return reason && reason.getValue() == execDiagnostic::kEmissionPlanReasonValue;
}

llvm::SmallVector<DiagnosticOp, 4>
collectDirectEmissionPlanDiagnostics(KernelOp kernel) {
  llvm::SmallVector<DiagnosticOp, 4> diagnostics;
  if (!kernel || kernel.getBody().empty())
    return diagnostics;

  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto diagnostic = llvm::dyn_cast<DiagnosticOp>(operation);
    if (diagnostic && isEmissionPlanDiagnostic(diagnostic))
      diagnostics.push_back(diagnostic);
  }
  return diagnostics;
}

llvm::StringRef getStringAttr(mlir::Operation *operation,
                              llvm::StringRef name) {
  auto attr = operation->getAttrOfType<mlir::StringAttr>(name);
  return attr ? attr.getValue() : llvm::StringRef();
}

llvm::StringRef getTargetAttr(DiagnosticOp diagnostic) {
  auto attr = diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(
      execDiagnostic::kTargetAttrName);
  return attr ? attr.getValue() : llvm::StringRef();
}

void addMockLoweringBoundary(mlir::MLIRContext &context, KernelOp kernel,
                             llvm::StringRef variantSymbol,
                             llvm::StringRef origin,
                             llvm::StringRef role) {
  VariantOp variant = findDirectVariant(kernel, variantSymbol);
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  mlir::OperationState state(kernel.getLoc(), DiagnosticOp::getOperationName());
  state.addAttribute("reason",
                     mlir::StringAttr::get(&context, "mock-lowering-boundary"));
  state.addAttribute(
      "message",
      mlir::StringAttr::get(
          &context,
          "mock selected lowering boundary used by generic emission tests"));
  state.addAttribute("source_kernel",
                     mlir::StringAttr::get(&context, kernel.getSymName()));
  state.addAttribute("selected_variant",
                     mlir::FlatSymbolRefAttr::get(&context, variantSymbol));
  state.addAttribute("origin", mlir::StringAttr::get(&context, origin));
  state.addAttribute("role", mlir::StringAttr::get(&context, role));
  state.addAttribute("status",
                     mlir::StringAttr::get(&context, "metadata-only"));
  if (variant) {
    if (auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires"))
      state.addAttribute("required_capabilities", requires);
  }
  builder.create(state);
}

void addMockDirectBoundary(mlir::MLIRContext &context, KernelOp kernel,
                           llvm::StringRef variantSymbol,
                           llvm::StringRef origin = "mock-emitter") {
  addMockLoweringBoundary(context, kernel, variantSymbol, origin,
                          "direct variant");
}

int expectDiagnosticStringAttr(DiagnosticOp diagnostic, llvm::StringRef name,
                               llvm::StringRef expected,
                               llvm::Twine context) {
  return expect(getStringAttr(diagnostic.getOperation(), name) == expected,
                context);
}

int expectSupportedEmissionPlanDiagnostic(DiagnosticOp diagnostic,
                                          llvm::StringRef expectedTarget,
                                          llvm::StringRef expectedRole) {
  if (int result = expect(getTargetAttr(diagnostic) == expectedTarget,
                          "diagnostic preserves variant target"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kOriginAttrName, "mock-emitter",
          "diagnostic preserves plugin origin"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kRoleAttrName, expectedRole,
          "diagnostic preserves selected-path role"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kStatusAttrName,
          execDiagnostic::kEmissionPlanSupportedStatusValue,
          "diagnostic marks supported status"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kSeverityAttrName,
          execDiagnostic::kEmissionPlanSupportedSeverityValue,
          "diagnostic marks info severity"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kPlanKindAttrName,
          execDiagnostic::kEmissionPlanPlanKindValue,
          "diagnostic carries generic plan kind"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kEmissionKindAttrName, "metadata-intent",
          "diagnostic carries emission kind"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kLoweringPipelineAttrName,
          "mock.lowering.pipeline.v1",
          "diagnostic carries lowering pipeline"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kRuntimeABIAttrName,
          "mock.runtime.abi.v1", "diagnostic carries runtime ABI"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kRuntimeABIKindAttrName,
          "mock-runtime-abi-kind", "diagnostic carries runtime ABI kind"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kRuntimeABINameAttrName,
          "mock.runtime.abi.v1", "diagnostic carries runtime ABI name"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kRuntimeGlueRoleAttrName,
          "mock-runtime-glue-role", "diagnostic carries runtime glue role"))
    return result;
  auto requiredCapabilities = diagnostic->getAttrOfType<mlir::ArrayAttr>(
      execDiagnostic::kRequiredCapabilitiesAttrName);
  if (int result =
          expect(requiredCapabilities && requiredCapabilities.size() == 1,
                 "diagnostic carries required capability refs"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kArtifactKindAttrName,
          "compiler-emission-plan", "diagnostic carries artifact kind"))
    return result;
  return expect(getStringAttr(diagnostic.getOperation(),
                              execDiagnostic::kMessageAttrName)
                    .contains("plugin-owned"),
                "diagnostic message carries plugin-owned explanation");
}

int expectUnsupportedEmissionPlanDiagnostic(DiagnosticOp diagnostic,
                                            llvm::StringRef expectedTarget,
                                            llvm::StringRef expectedOrigin,
                                            llvm::StringRef expectedFragment) {
  if (int result = expect(getTargetAttr(diagnostic) == expectedTarget,
                          "unsupported diagnostic preserves variant target"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kOriginAttrName, expectedOrigin,
          "unsupported diagnostic preserves plugin origin"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kStatusAttrName,
          execDiagnostic::kEmissionPlanUnsupportedStatusValue,
          "unsupported diagnostic marks unsupported status"))
    return result;
  if (int result = expectDiagnosticStringAttr(
          diagnostic, execDiagnostic::kSeverityAttrName,
          execDiagnostic::kEmissionPlanUnsupportedSeverityValue,
          "unsupported diagnostic marks error severity"))
    return result;
  if (int result =
          expect(!getStringAttr(diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABIKindAttrName)
                      .empty(),
                 "unsupported diagnostic carries runtime ABI kind"))
    return result;
  if (int result =
          expect(!getStringAttr(diagnostic.getOperation(),
                                execDiagnostic::kRuntimeABINameAttrName)
                      .empty(),
                 "unsupported diagnostic carries runtime ABI name"))
    return result;
  if (int result =
          expect(!getStringAttr(diagnostic.getOperation(),
                                execDiagnostic::kRuntimeGlueRoleAttrName)
                      .empty(),
                 "unsupported diagnostic carries runtime glue role"))
    return result;
  if (int result = expect(diagnostic->hasAttr(
                              execDiagnostic::kRequiredCapabilitiesAttrName),
                          "unsupported diagnostic carries capability refs"))
    return result;
  return expect(getStringAttr(diagnostic.getOperation(),
                              execDiagnostic::kMessageAttrName)
                    .contains(expectedFragment),
                "unsupported diagnostic carries plugin diagnostic text");
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

int runRegistrySupportedEmissionPlanTest(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, getDirectKernelSource());
  if (!module)
    return fail("failed to parse direct emission plan module");

  KernelOp kernel = findKernel(*module, "direct");
  VariantOp variant = findDirectVariant(kernel, "fast");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(plugin),
                                 "register supported emission plan plugin"))
    return result;

  VariantEmissionPlan plan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              plan),
          "registry routes direct variant emission plan"))
    return result;

  if (int result = expectSupportedPlan(plan, "direct", "fast",
                                       VariantEmissionRole::DirectVariant))
    return result;
  if (int result = expect(plugin.getPlanCalls() == 1,
                          "emission plan hook is invoked once"))
    return result;
  return expect(plugin.getReadinessCalls() == 0,
                "emission planning does not imply readiness hook reuse");
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

int runSelectedMarkerPassTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @selected_marker {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "fast-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.variant @slow attributes {
      origin = "slow-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.diagnostic {
      message = "fast selected by generic selection marker",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @fast
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse selected marker emission readiness module");

  EmissionPlugin fastPlugin("fast-emitter");
  EmissionPlugin slowPlugin("slow-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(fastPlugin),
                                 "register selected marker fast plugin"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(slowPlugin),
                                 "register selected marker slow plugin"))
    return result;

  KernelOp kernel = findKernel(*module, "selected_marker");
  addMockDirectBoundary(context, kernel, "fast", "fast-emitter");

  mlir::PassManager passManager(&context);
  passManager.addPass(
      tianchenrv::transforms::createCheckEmissionPathsPass(registry));
  if (int result = expect(mlir::succeeded(passManager.run(*module)),
                          "selected marker emission pass succeeds"))
    return result;

  if (int result =
          expect(fastPlugin.getReadinessCalls() == 1,
                 "selected marker routes only selected variant to plugin"))
    return result;
  if (int result =
          expect(slowPlugin.getReadinessCalls() == 0,
                 "selected marker prevents conservative all-variant routing"))
    return result;
  if (int result =
          expect(fastPlugin.getObservedVariantSymbols().size() == 1 &&
                     fastPlugin.getObservedVariantSymbols()[0] == "fast",
                 "selected marker preserves selected target variant"))
    return result;
  return expect(fastPlugin.getObservedRoles().size() == 1 &&
                    fastPlugin.getObservedRoles()[0] == "direct variant",
                "selected marker uses target-neutral direct variant role");
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

  KernelOp kernel = findKernel(*module, "dispatch_pass");
  addMockLoweringBoundary(context, kernel, "fast", "mock-emitter",
                          "dispatch case");
  addMockLoweringBoundary(context, kernel, "medium", "mock-emitter",
                          "dispatch case");
  addMockLoweringBoundary(context, kernel, "fallback", "mock-emitter",
                          "dispatch fallback");

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

int runDispatchEmissionPlanCollectionTest(mlir::MLIRContext &context) {
  const char *source = R"mlir(
module {
  tcrv.exec.kernel @dispatch_plan {
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
    return fail("failed to parse dispatch emission plan module");

  KernelOp kernel = findKernel(*module, "dispatch_plan");
  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(plugin),
                                 "register dispatch emission plan plugin"))
    return result;

  addMockLoweringBoundary(context, kernel, "fast", "mock-emitter",
                          "dispatch case");
  addMockLoweringBoundary(context, kernel, "medium", "mock-emitter",
                          "dispatch case");
  addMockLoweringBoundary(context, kernel, "fallback", "mock-emitter",
                          "dispatch fallback");

  llvm::SmallVector<VariantEmissionPlan, 4> plans;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectKernelEmissionPlans(kernel, plans,
                                                             registry),
          "collect dispatch emission plans"))
    return result;

  if (int result = expect(plans.size() == 3,
                          "dispatch collection returns case and fallback plans"))
    return result;
  if (int result =
          expectSupportedPlan(plans[0], "dispatch_plan", "fast",
                              VariantEmissionRole::DispatchCase))
    return result;
  if (int result =
          expectSupportedPlan(plans[1], "dispatch_plan", "medium",
                              VariantEmissionRole::DispatchCase))
    return result;
  if (int result =
          expectSupportedPlan(plans[2], "dispatch_plan", "fallback",
                              VariantEmissionRole::DispatchFallback))
    return result;
  if (int result = expect(plugin.getObservedPlanVariantSymbols()[0] == "fast" &&
                              plugin.getObservedPlanVariantSymbols()[1] ==
                                  "medium" &&
                              plugin.getObservedPlanVariantSymbols()[2] ==
                                  "fallback",
                          "dispatch emission plans preserve deterministic "
                          "dispatch order"))
    return result;
  return expect(plugin.getReadinessCalls() == 0,
                "emission plan collection does not call readiness hook");
}

int runSelectedMarkerEmissionPlanCollectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @selected_plan {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.variant @slow attributes {
      origin = "slow-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.diagnostic {
      message = "fast selected by generic planner",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @fast
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse selected marker emission plan module");

  KernelOp kernel = findKernel(*module, "selected_plan");
  EmissionPlugin fastPlugin("mock-emitter");
  EmissionPlugin slowPlugin("slow-emitter", true,
                            EmissionBehavior::Supported,
                            EmissionPlanBehavior::PluginFailure);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(fastPlugin),
                                 "register selected plan fast plugin"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(slowPlugin),
                                 "register selected plan slow plugin"))
    return result;

  addMockDirectBoundary(context, kernel, "fast", "mock-emitter");

  llvm::SmallVector<VariantEmissionPlan, 2> plans;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectKernelEmissionPlans(kernel, plans,
                                                             registry),
          "collect selected marker emission plan"))
    return result;

  if (int result = expect(plans.size() == 1,
                          "selected marker returns only selected plan"))
    return result;
  if (int result = expectSupportedPlan(plans[0], "selected_plan", "fast",
                                       VariantEmissionRole::DirectVariant))
    return result;
  if (int result =
          expect(slowPlugin.getPlanCalls() == 0,
                 "unselected unsupported variant does not fail selected plan"))
    return result;
  return expect(fastPlugin.getPlanCalls() == 1,
                "selected marker invokes selected origin plugin once");
}

int runConservativeDirectEmissionPlanCollectionTest(
    mlir::MLIRContext &context) {
  const char *source = R"mlir(
module {
  tcrv.exec.kernel @direct_plan {
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
    return fail("failed to parse conservative direct emission plan module");

  KernelOp kernel = findKernel(*module, "direct_plan");
  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(plugin),
                                 "register conservative emission plan plugin"))
    return result;

  llvm::SmallVector<VariantEmissionPlan, 2> plans;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectKernelEmissionPlans(kernel, plans,
                                                             registry),
          "collect conservative direct emission plans"))
    return result;

  if (int result = expect(plans.size() == 2,
                          "no dispatch or selected marker plans all variants"))
    return result;
  if (int result = expectSupportedPlan(plans[0], "direct_plan", "fast",
                                       VariantEmissionRole::DirectVariant))
    return result;
  return expectSupportedPlan(plans[1], "direct_plan", "fallback",
                             VariantEmissionRole::DirectVariant);
}

int runSelectedEmissionPlanMaterializationPassTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @materialize_selected {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.variant @slow attributes {
      origin = "slow-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.diagnostic {
      message = "fast selected by generic planner",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @fast
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse selected emission-plan materialization module");

  EmissionPlugin fastPlugin("mock-emitter");
  EmissionPlugin slowPlugin("slow-emitter", true, EmissionBehavior::Supported,
                            EmissionPlanBehavior::PluginFailure);
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(fastPlugin),
                                 "register materialization fast plugin"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(slowPlugin),
                                 "register materialization slow plugin"))
    return result;

  KernelOp kernelBeforePass = findKernel(*module, "materialize_selected");
  addMockDirectBoundary(context, kernelBeforePass, "fast", "mock-emitter");

  mlir::PassManager passManager(&context);
  passManager.addPass(
      tianchenrv::transforms::createMaterializeEmissionPlansPass(registry));
  if (int result = expect(mlir::succeeded(passManager.run(*module)),
                          "injected materialize emission plans pass succeeds"))
    return result;

  KernelOp kernel = findKernel(*module, "materialize_selected");
  llvm::SmallVector<DiagnosticOp, 4> diagnostics =
      collectDirectEmissionPlanDiagnostics(kernel);
  if (int result = expect(diagnostics.size() == 1,
                          "selected materialization emits one plan diagnostic"))
    return result;
  if (int result = expectSupportedEmissionPlanDiagnostic(
          diagnostics[0], "fast", "direct variant"))
    return result;
  if (int result =
          expect(slowPlugin.getPlanCalls() == 0,
                 "unselected unsupported variant is not materialized"))
    return result;

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module->print(stream);
  stream.flush();
  return expect(llvm::StringRef(printed).contains("tcrv.exec.diagnostic") &&
                    llvm::StringRef(printed).contains("emission_kind") &&
                    llvm::StringRef(printed).contains("lowering_pipeline") &&
                    llvm::StringRef(printed).contains("runtime_abi") &&
                    llvm::StringRef(printed).contains("artifact_kind"),
                "printed MLIR contains structured emission-plan metadata");
}

int runDispatchEmissionPlanMaterializationOrderTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @materialize_dispatch {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.variant @medium attributes {
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
      tcrv.exec.fallback @fallback
      tcrv.exec.case @fast {policy = "fast_path"}
      tcrv.exec.case @medium {policy = "medium_path"}
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse dispatch materialization module");

  KernelOp kernel = findKernel(*module, "materialize_dispatch");
  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(plugin),
                                 "register dispatch materialization plugin"))
    return result;

  addMockLoweringBoundary(context, kernel, "fast", "mock-emitter",
                          "dispatch case");
  addMockLoweringBoundary(context, kernel, "medium", "mock-emitter",
                          "dispatch case");
  addMockLoweringBoundary(context, kernel, "fallback", "mock-emitter",
                          "dispatch fallback");

  if (int result = expectSuccess(
          tianchenrv::transforms::materializeKernelEmissionPlanDiagnostics(
              kernel, registry),
          "materialize dispatch emission plans"))
    return result;

  llvm::SmallVector<DiagnosticOp, 4> diagnostics =
      collectDirectEmissionPlanDiagnostics(kernel);
  if (int result = expect(diagnostics.size() == 3,
                          "dispatch materialization emits all path diagnostics"))
    return result;
  if (int result = expect(getTargetAttr(diagnostics[0]) == "fast" &&
                              getTargetAttr(diagnostics[1]) == "medium" &&
                              getTargetAttr(diagnostics[2]) == "fallback",
                          "dispatch diagnostics order cases before fallback"))
    return result;
  if (int result = expectSupportedEmissionPlanDiagnostic(
          diagnostics[0], "fast", "dispatch case"))
    return result;
  return expectSupportedEmissionPlanDiagnostic(diagnostics[2], "fallback",
                                               "dispatch fallback");
}

int runConservativeEmissionPlanMaterializationTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @materialize_direct {
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
    return fail("failed to parse conservative materialization module");

  KernelOp kernel = findKernel(*module, "materialize_direct");
  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(plugin),
                                 "register conservative materialization plugin"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::transforms::materializeKernelEmissionPlanDiagnostics(
              kernel, registry),
          "materialize conservative direct emission plans"))
    return result;

  llvm::SmallVector<DiagnosticOp, 4> diagnostics =
      collectDirectEmissionPlanDiagnostics(kernel);
  if (int result = expect(diagnostics.size() == 2,
                          "conservative materialization emits all variants"))
    return result;
  return expect(getTargetAttr(diagnostics[0]) == "fast" &&
                    getTargetAttr(diagnostics[1]) == "fallback",
                "conservative materialization preserves direct variant order");
}

int expectMaterializationErrorLeavesDiagnosticCount(
    mlir::MLIRContext &context, llvm::StringRef source,
    ExtensionPluginRegistry &registry,
    std::initializer_list<llvm::StringRef> fragments,
    unsigned expectedDiagnosticCount) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse materialization negative module");

  KernelOp kernel;
  module->walk([&](KernelOp candidate) {
    if (!kernel)
      kernel = candidate;
  });
  if (!kernel)
    return fail("materialization negative module has no kernel");

  unsigned before = collectDirectEmissionPlanDiagnostics(kernel).size();
  if (int result = expect(before == expectedDiagnosticCount,
                          "negative materialization starts with expected "
                          "diagnostic count"))
    return result;
  if (int result = expectErrorContains(
          tianchenrv::transforms::materializeKernelEmissionPlanDiagnostics(
              kernel, registry),
          fragments))
    return result;
  unsigned after = collectDirectEmissionPlanDiagnostics(kernel).size();
  return expect(after == before,
                "failed materialization leaves diagnostic count unchanged");
}

int runEmissionPlanMaterializationNegativeTests(mlir::MLIRContext &context) {
  {
    ExtensionPluginRegistry registry;
    if (int result = expectMaterializationErrorLeavesDiagnosticCount(
            context, getDirectKernelSource(), registry,
            {"variant emission plan collection failed",
             "unknown origin plugin 'mock-emitter'"},
            0))
      return result;
  }

  {
    EmissionPlugin plugin("mock-emitter", false);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register disabled materialization plugin"))
      return result;
    if (int result = expectMaterializationErrorLeavesDiagnosticCount(
            context, getDirectKernelSource(), registry,
            {"origin plugin 'mock-emitter' is disabled"}, 0))
      return result;
  }

  {
    EmissionPlugin plugin("mock-emitter", true, EmissionBehavior::Supported,
                          EmissionPlanBehavior::SupportedEmptyLoweringPipeline);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register malformed plan plugin"))
      return result;
    if (int result = expectMaterializationErrorLeavesDiagnosticCount(
            context, getDirectKernelSource(), registry,
            {"invalid emission plan",
             "supported or metadata-only plan requires non-empty lowering "
             "pipeline"},
            0))
      return result;
  }

  {
    EmissionPlugin plugin("mock-emitter", true, EmissionBehavior::Supported,
                          EmissionPlanBehavior::MismatchedVariantSymbol);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register mismatched variant plan plugin"))
      return result;
    if (int result = expectMaterializationErrorLeavesDiagnosticCount(
            context, getDirectKernelSource(), registry,
            {"invalid emission plan", "does not match request variant"}, 0))
      return result;
  }

  {
    constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @duplicate_selected_marker {
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
    tcrv.exec.diagnostic {message = "fast", reason = "variant-selected", selection_kind = "static-variant", target = @fast}
    tcrv.exec.diagnostic {message = "fallback", reason = "variant-selected", selection_kind = "fallback-only", target = @fallback}
  }
}
)mlir";
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register duplicate marker plugin"))
      return result;
    if (int result = expectMaterializationErrorLeavesDiagnosticCount(
            context, source, registry,
            {"at most one direct selected-path diagnostic marker"}, 0))
      return result;
    if (int result = expect(plugin.getPlanCalls() == 0,
                            "duplicate markers fail before plugin planning"))
      return result;
  }

  {
    constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @missing_dispatch_target {
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
      tcrv.exec.case @fast {policy = "fast"}
      tcrv.exec.fallback @fallback
    }
  }
}
)mlir";
    mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
    if (!module)
      return fail("failed to parse missing dispatch target module");

    KernelOp kernel = findKernel(*module, "missing_dispatch_target");
    DispatchCaseOp dispatchCase = findFirstDispatchCase(findDirectDispatch(kernel));
    dispatchCase->setAttr("target", getSymbolRef(context, "missing"));

    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register missing dispatch target plugin"))
      return result;
    unsigned before = collectDirectEmissionPlanDiagnostics(kernel).size();
    if (int result = expect(before == 0,
                            "missing dispatch target starts with no "
                            "emission-plan diagnostics"))
      return result;
    if (int result = expectErrorContains(
            tianchenrv::transforms::materializeKernelEmissionPlanDiagnostics(
                kernel, registry),
            {"dispatch target @missing",
             "does not resolve to a direct sibling tcrv.exec.variant"}))
      return result;
    unsigned after = collectDirectEmissionPlanDiagnostics(kernel).size();
    if (int result = expect(after == before,
                            "missing dispatch target leaves diagnostics "
                            "unchanged"))
      return result;
    if (int result = expect(plugin.getPlanCalls() == 0,
                            "missing dispatch target fails before planning"))
      return result;
  }

  {
    constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @stale_boundary_with_existing_plan {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.variant @old_fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.diagnostic {
      message = "fast selected by generic planner",
      reason = "variant-selected",
      selection_kind = "static-variant",
      target = @fast
    }
    tcrv.exec.diagnostic {
      message = "stale mock boundary",
      origin = "mock-emitter",
      reason = "mock-lowering-boundary",
      required_capabilities = [@base],
      role = "direct variant",
      selected_variant = @old_fast,
      source_kernel = "stale_boundary_with_existing_plan",
      status = "metadata-only"
    }
    tcrv.exec.diagnostic {
      message = "existing unsupported plan",
      origin = "mock-emitter",
      reason = "emission_plan",
      required_capabilities = [@base],
      role = "direct variant",
      runtime_abi_kind = "mock-runtime-abi-kind",
      runtime_abi_name = "mock.runtime.abi.v1",
      runtime_glue_role = "mock-runtime-glue-role",
      status = "unsupported",
      target = @fast
    }
  }
}
)mlir";
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register stale boundary plugin"))
      return result;
    if (int result = expectMaterializationErrorLeavesDiagnosticCount(
            context, source, registry,
            {"stale lowering boundary", "selected_variant @old_fast",
             "not selected by the current dispatch"},
            1))
      return result;
    if (int result = expect(plugin.getPlanCalls() == 0,
                            "stale boundary fails before planning"))
      return result;
  }

  {
    constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @preexisting_emission_plan {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
    tcrv.exec.diagnostic {
      message = "existing unsupported plan",
      origin = "mock-emitter",
      reason = "emission_plan",
      required_capabilities = [@base],
      role = "direct variant",
      runtime_abi_kind = "mock-runtime-abi-kind",
      runtime_abi_name = "mock.runtime.abi.v1",
      runtime_glue_role = "mock-runtime-glue-role",
      status = "unsupported",
      target = @fast
    }
  }
}
)mlir";
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register preexisting diagnostic plugin"))
      return result;
    if (int result = expectMaterializationErrorLeavesDiagnosticCount(
            context, source, registry,
            {"requires no pre-existing emission-plan diagnostics",
             "target @fast"},
            1))
      return result;
  }

  return 0;
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
    variant->removeAttr("origin");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register missing-origin plan plugin"))
      return result;

    VariantEmissionPlan plan;
    if (int result = expectErrorContains(
            registry.buildVariantEmissionPlan(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                plan),
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

int runEmissionPlanRegistryTests(mlir::MLIRContext &context) {
  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter");
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register unsupported plan plugin"))
      return result;

    VariantEmissionPlan plan;
    if (int result = expectSuccess(
            registry.buildVariantEmissionPlan(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                plan),
            "build supported plan before unsupported mutation"))
      return result;
  }

  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter", true, EmissionBehavior::Supported,
                          EmissionPlanBehavior::Unsupported);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register unsupported plan plugin"))
      return result;

    VariantEmissionPlan plan;
    if (int result = expectSuccess(
            registry.buildVariantEmissionPlan(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                plan),
            "unsupported emission plan is a structured diagnostic result"))
      return result;
    if (int result = expect(plan.isUnsupported(),
                            "unsupported emission plan keeps unsupported "
                            "status"))
      return result;
    if (int result = expect(plan.getOriginPlugin() == "mock-emitter" &&
                                plan.getKernelSymbol() == "direct" &&
                                plan.getVariantSymbol() == "fast" &&
                                plan.getRole() ==
                                    VariantEmissionRole::DirectVariant,
                            "unsupported emission plan carries generic "
                            "context"))
      return result;
    if (int result =
            expect(plan.getDiagnostic().contains("unsupported selected"),
                   "unsupported emission plan carries diagnostic text"))
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
                                   "register empty-origin plan plugin"))
      return result;

    VariantEmissionPlan plan;
    if (int result = expectErrorContains(
            registry.buildVariantEmissionPlan(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                plan),
            {"variant @fast", "kernel @direct",
             "non-empty string attribute 'origin'"}))
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

    VariantEmissionPlan plan;
    ExtensionPluginRegistry registry;
    if (int result = expectErrorContains(
            registry.buildVariantEmissionPlan(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                plan),
            {"emission plan collection", "variant @fast",
             "unknown origin plugin 'missing-emitter'"}))
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
                                   "register disabled plan plugin"))
      return result;

    VariantEmissionPlan plan;
    if (int result = expectErrorContains(
            registry.buildVariantEmissionPlan(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                plan),
            {"variant @fast", "origin plugin 'disabled-emitter' is disabled"}))
      return result;
  }

  struct PlanNegativeCase {
    EmissionPlanBehavior behavior;
    llvm::StringRef expectedFragment;
  };

  const PlanNegativeCase negativeCases[] = {
      {EmissionPlanBehavior::MissingStatus, "status is missing"},
      {EmissionPlanBehavior::SupportedEmptyEmissionKind,
       "non-empty emission kind"},
      {EmissionPlanBehavior::SupportedEmptyLoweringPipeline,
       "non-empty lowering pipeline"},
      {EmissionPlanBehavior::SupportedEmptyRuntimeABI,
       "non-empty runtime ABI"},
      {EmissionPlanBehavior::SupportedEmptyRuntimeABIKind,
       "non-empty runtime ABI kind"},
      {EmissionPlanBehavior::SupportedEmptyRuntimeABIName,
       "non-empty runtime ABI name"},
      {EmissionPlanBehavior::SupportedEmptyRuntimeGlueRole,
       "non-empty runtime glue role"},
      {EmissionPlanBehavior::SupportedEmptyArtifactKind,
       "non-empty artifact kind"},
      {EmissionPlanBehavior::SupportedEmptyExplanation,
       "non-empty explanation"},
      {EmissionPlanBehavior::SupportedUnboundedExplanation,
       "explanation must be bounded single-line metadata"},
      {EmissionPlanBehavior::MissingRequiredCapabilityRefs,
       "non-empty required capability refs"},
      {EmissionPlanBehavior::MismatchedRequiredCapabilityRef,
       "not a safe subset of selected variant requires metadata"},
      {EmissionPlanBehavior::UnsupportedEmptyDiagnostic,
       "non-empty diagnostic"},
      {EmissionPlanBehavior::MismatchedVariantSymbol,
       "does not match request variant"},
      {EmissionPlanBehavior::MismatchedRole, "does not match request role"},
  };

  for (const PlanNegativeCase &negativeCase : negativeCases) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter", true, EmissionBehavior::Supported,
                          negativeCase.behavior);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register malformed plan plugin"))
      return result;

    VariantEmissionPlan plan;
    if (int result = expectErrorContains(
            registry.buildVariantEmissionPlan(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                plan),
            {"invalid emission plan", negativeCase.expectedFragment}))
      return result;
  }

  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, getDirectKernelSource());
    KernelOp kernel = findKernel(*module, "direct");
    VariantOp variant = findDirectVariant(kernel, "fast");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    EmissionPlugin plugin("mock-emitter", true, EmissionBehavior::Supported,
                          EmissionPlanBehavior::PluginFailure);
    ExtensionPluginRegistry registry;
    if (int result = expectSuccess(registry.registerPlugin(plugin),
                                   "register failing plan plugin"))
      return result;

    VariantEmissionPlan plan;
    if (int result = expectErrorContains(
            registry.buildVariantEmissionPlan(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                plan),
            {"failed emission plan query",
             "plugin-local emission plan construction failed"}))
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

int expectPlanStructuralErrorHasNoPluginCalls(
    mlir::MLIRContext &context, llvm::StringRef source,
    void (*mutate)(mlir::MLIRContext &, KernelOp),
    std::initializer_list<llvm::StringRef> fragments) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse plan structural negative module");

  KernelOp kernel = findKernel(*module, "dispatch_negative");
  mutate(context, kernel);

  EmissionPlugin plugin("mock-emitter");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(plugin),
                                 "register plan structural negative plugin"))
    return result;

  llvm::SmallVector<VariantEmissionPlan, 4> plans;
  if (int result = expectErrorContains(
          tianchenrv::transforms::collectKernelEmissionPlans(kernel, plans,
                                                             registry),
          fragments))
    return result;

  return expect(plugin.getPlanCalls() == 0,
                "plan structural failures are diagnosed before plugin "
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

const char *getSelectedMarkerKernelSource() {
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
    tcrv.exec.diagnostic {
      message = "fast selected by generic planner",
      reason = "variant-selected",
      selection_kind = "static-variant",
      target = @fast
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

  if (int result = expectStructuralErrorHasNoPluginCalls(
          context, getSelectedMarkerKernelSource(),
          [](mlir::MLIRContext &, KernelOp kernel) {
            for (mlir::Operation &operation : kernel.getBody().front()) {
              auto diagnostic = llvm::dyn_cast<DiagnosticOp>(operation);
              if (!diagnostic)
                continue;
              diagnostic->removeAttr("selection_kind");
              break;
            }
          },
          {"selected-path diagnostic marker", "selection_kind"}))
    return result;

  if (int result = expectStructuralErrorHasNoPluginCalls(
          context, getSelectedMarkerKernelSource(),
          [](mlir::MLIRContext &context, KernelOp kernel) {
            for (mlir::Operation &operation : kernel.getBody().front()) {
              auto diagnostic = llvm::dyn_cast<DiagnosticOp>(operation);
              if (!diagnostic)
                continue;
              diagnostic->setAttr("selection_kind",
                                  mlir::StringAttr::get(&context,
                                                        "runtime-dispatch"));
              break;
            }
          },
          {"unsupported selection_kind 'runtime-dispatch'"}))
    return result;

  if (int result = expectStructuralErrorHasNoPluginCalls(
          context, getSelectedMarkerKernelSource(),
          [](mlir::MLIRContext &context, KernelOp kernel) {
            mlir::OpBuilder builder(&context);
            builder.setInsertionPointToEnd(&kernel.getBody().front());
            mlir::OperationState state(kernel.getLoc(),
                                       DiagnosticOp::getOperationName());
            state.addAttribute("reason",
                               mlir::StringAttr::get(&context,
                                                     "variant-selected"));
            state.addAttribute("message",
                               mlir::StringAttr::get(&context,
                                                     "fallback also selected"));
            state.addAttribute("selection_kind",
                               mlir::StringAttr::get(&context,
                                                     "fallback-only"));
            state.addAttribute("target", getSymbolRef(context, "fallback"));
            builder.create(state);
          },
          {"at most one direct selected-path diagnostic marker"}))
    return result;

  return 0;
}

int runEmissionPlanStructuralNegativeTests(mlir::MLIRContext &context) {
  if (int result = expectPlanStructuralErrorHasNoPluginCalls(
          context, getDispatchKernelSource(),
          [](mlir::MLIRContext &, KernelOp kernel) {
            DispatchCaseOp dispatchCase =
                findFirstDispatchCase(findDirectDispatch(kernel));
            dispatchCase->removeAttr("target");
          },
          {"missing a variant symbol reference target"}))
    return result;

  if (int result = expectPlanStructuralErrorHasNoPluginCalls(
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

  if (int result = expectPlanStructuralErrorHasNoPluginCalls(
          context, getSelectedMarkerKernelSource(),
          [](mlir::MLIRContext &context, KernelOp kernel) {
            mlir::OpBuilder builder(&context);
            builder.setInsertionPointToEnd(&kernel.getBody().front());
            mlir::OperationState state(kernel.getLoc(),
                                       DiagnosticOp::getOperationName());
            state.addAttribute("reason",
                               mlir::StringAttr::get(&context,
                                                     "variant-selected"));
            state.addAttribute("message",
                               mlir::StringAttr::get(&context,
                                                     "fallback also selected"));
            state.addAttribute("selection_kind",
                               mlir::StringAttr::get(&context,
                                                     "fallback-only"));
            state.addAttribute("target", getSymbolRef(context, "fallback"));
            builder.create(state);
          },
          {"at most one direct selected-path diagnostic marker"}))
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

  VariantEmissionPlan rvvPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(request, rvvPlan),
          "RVV plugin returns explicit unsupported emission plan"))
    return result;
  if (int result = expect(rvvPlan.isUnsupported(),
                          "RVV metadata-only first slice has unsupported "
                          "emission plan status"))
    return result;
  if (int result = expect(rvvPlan.getOriginPlugin() == "rvv-plugin" &&
                              rvvPlan.getKernelSymbol() == "rvv_emission" &&
                              rvvPlan.getVariantSymbol() == "rvv_first_slice" &&
                              rvvPlan.getRole() ==
                                  VariantEmissionRole::DirectVariant,
                          "RVV unsupported emission plan carries generic "
                          "context"))
    return result;
  if (int result = expect(rvvPlan.getRuntimeABIKind() ==
                              "rvv-plugin-deferred-runtime-abi",
                          "RVV unsupported emission plan carries plugin-owned "
                          "runtime ABI kind"))
    return result;
  if (int result = expect(rvvPlan.getRuntimeABIName() ==
                              "rvv-executable-runtime-abi-deferred",
                          "RVV unsupported emission plan carries runtime ABI "
                          "name"))
    return result;
  if (int result =
          expect(rvvPlan.getRuntimeGlueRole() == "deferred-rvv-runtime-glue",
                 "RVV unsupported emission plan carries runtime glue role"))
    return result;
  if (int result = expect(rvvPlan.getRequiredCapabilitySymbols().size() == 1 &&
                              rvvPlan.getRequiredCapabilitySymbols().front() ==
                                  "rvv",
                          "RVV unsupported emission plan preserves capability "
                          "refs"))
    return result;
  if (int result =
          expect(rvvPlan.getDiagnostic().contains("no RVV lowering pipeline") &&
                     rvvPlan.getDiagnostic().contains("runtime ABI") &&
                     rvvPlan.getDiagnostic().contains("not RVV hardware"),
                 "RVV unsupported emission plan carries structured boundary "
                 "diagnostic"))
    return result;

  llvm::SmallVector<VariantEmissionPlan, 1> plans;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectKernelEmissionPlans(kernel, plans,
                                                             registry),
          "collect RVV unsupported emission plan"))
    return result;
  if (int result = expect(plans.size() == 1 && plans[0].isUnsupported(),
                          "RVV plan collection preserves unsupported plan"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::transforms::materializeKernelEmissionPlanDiagnostics(
              kernel, registry),
          "materialize RVV unsupported emission plan diagnostic"))
    return result;
  llvm::SmallVector<DiagnosticOp, 1> diagnostics =
      collectDirectEmissionPlanDiagnostics(kernel);
  if (int result = expect(diagnostics.size() == 1,
                          "RVV materialization emits one unsupported "
                          "diagnostic"))
    return result;
  if (int result = expectUnsupportedEmissionPlanDiagnostic(
          diagnostics[0], "rvv_first_slice", "rvv-plugin",
          "no RVV lowering pipeline"))
    return result;
  if (int result =
          expect(getStringAttr(diagnostics[0].getOperation(),
                               execDiagnostic::kMessageAttrName)
                     .contains("not RVV hardware/toolchain/runtime/"
                               "correctness/performance evidence"),
                 "RVV unsupported diagnostic avoids runtime/correctness/"
                 "performance claim"))
    return result;

  return 0;
}

} // namespace

int main() {
  mlir::MLIRContext context;
  registerCoreDialects(context);

  if (int result = runRegistrySupportedDirectVariantTest(context))
    return result;
  if (int result = runRegistrySupportedEmissionPlanTest(context))
    return result;
  if (int result = runInjectedDirectPassTest(context))
    return result;
  if (int result = runSelectedMarkerPassTest(context))
    return result;
  if (int result = runInjectedDispatchPassTest(context))
    return result;
  if (int result = runDispatchEmissionPlanCollectionTest(context))
    return result;
  if (int result = runSelectedMarkerEmissionPlanCollectionTest(context))
    return result;
  if (int result = runConservativeDirectEmissionPlanCollectionTest(context))
    return result;
  if (int result = runSelectedEmissionPlanMaterializationPassTest(context))
    return result;
  if (int result = runDispatchEmissionPlanMaterializationOrderTest(context))
    return result;
  if (int result = runConservativeEmissionPlanMaterializationTest(context))
    return result;
  if (int result = runEmissionPlanMaterializationNegativeTests(context))
    return result;
  if (int result = runRegistryNegativeTests(context))
    return result;
  if (int result = runEmissionPlanRegistryTests(context))
    return result;
  if (int result = runStructuralDispatchNegativeTests(context))
    return result;
  if (int result = runEmissionPlanStructuralNegativeTests(context))
    return result;
  if (int result = runRVVUnsupportedEmissionTest())
    return result;

  llvm::outs() << "emission readiness tests passed\n";
  return 0;
}
