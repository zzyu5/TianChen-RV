#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLFunctionalExtras.h"
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
using tianchenrv::plugin::VariantLoweringBoundaryRequest;
using tianchenrv::plugin::VariantLoweringBoundaryResult;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::rvv::LoweringBoundaryOp;

namespace {

enum class BoundaryBehavior {
  Materialized,
  NoBoundary,
  Unsupported,
  PluginFailure,
  MissingStatus,
  MismatchedOrigin,
  MismatchedVariant,
  MismatchedRole,
  NullMaterializedOp,
};

class BoundaryPlugin final : public ExtensionPlugin {
public:
  BoundaryPlugin(llvm::StringRef name, bool enabled = true,
                 BoundaryBehavior behavior = BoundaryBehavior::Materialized)
      : name(name.str()), enabled(enabled), behavior(behavior) {}

  llvm::StringRef getName() const override { return name; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return enabled; }

  llvm::Error materializeSelectedLoweringBoundary(
      const VariantLoweringBoundaryRequest &request,
      VariantLoweringBoundaryResult &out) const override {
    ++calls;
    observedRoles.push_back(
        tianchenrv::plugin::stringifyVariantEmissionRole(request.getRole())
            .str());

    if (behavior == BoundaryBehavior::PluginFailure)
      return llvm::make_error<llvm::StringError>(
          "plugin-local lowering-boundary builder failed",
          llvm::errc::invalid_argument);

    if (behavior == BoundaryBehavior::MissingStatus) {
      out = VariantLoweringBoundaryResult();
      return llvm::Error::success();
    }

    llvm::StringRef kernelSymbol = request.getKernel().getSymName();
    llvm::StringRef variantSymbol = request.getVariant().getSymName();
    VariantEmissionRole role = request.getRole();

    if (behavior == BoundaryBehavior::MismatchedOrigin) {
      out = VariantLoweringBoundaryResult::getNoBoundary(
          "other-plugin", kernelSymbol, variantSymbol, role,
          "mismatched origin test response");
      return llvm::Error::success();
    }

    if (behavior == BoundaryBehavior::MismatchedVariant) {
      out = VariantLoweringBoundaryResult::getNoBoundary(
          name, kernelSymbol, "other_variant", role,
          "mismatched variant test response");
      return llvm::Error::success();
    }

    if (behavior == BoundaryBehavior::MismatchedRole) {
      out = VariantLoweringBoundaryResult::getNoBoundary(
          name, kernelSymbol, variantSymbol, VariantEmissionRole::DispatchCase,
          "mismatched role test response");
      return llvm::Error::success();
    }

    if (behavior == BoundaryBehavior::Unsupported) {
      out = VariantLoweringBoundaryResult::getUnsupported(
          name, kernelSymbol, variantSymbol, role,
          "mock plugin intentionally reports unsupported lowering boundary");
      return llvm::Error::success();
    }

    if (behavior == BoundaryBehavior::NoBoundary) {
      out = VariantLoweringBoundaryResult::getNoBoundary(
          name, kernelSymbol, variantSymbol, role,
          "mock plugin selected path has no lowering-boundary op");
      return llvm::Error::success();
    }

    mlir::Operation *operation = nullptr;
    if (behavior != BoundaryBehavior::NullMaterializedOp) {
      mlir::OperationState state(request.getVariant().getLoc(),
                                 DiagnosticOp::getOperationName());
      state.addAttribute("reason",
                         request.getBuilder().getStringAttr(
                             "mock-lowering-boundary"));
      state.addAttribute("target",
                         mlir::FlatSymbolRefAttr::get(
                             request.getBuilder().getContext(),
                             request.getVariant().getSymName()));
      operation = request.getBuilder().create(state);
    }

    out = VariantLoweringBoundaryResult::getMaterialized(
        name, kernelSymbol, variantSymbol, role, operation);
    return llvm::Error::success();
  }

  mutable unsigned calls = 0;
  mutable llvm::SmallVector<std::string, 4> observedRoles;

private:
  std::string name;
  bool enabled;
  BoundaryBehavior behavior;
  llvm::SmallVector<PluginCapability, 0> capabilities;
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
    return fail("expected RVV lowering-boundary error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("RVV lowering-boundary error text missing '") +
                  fragment + "': " + message);
  }
  return 0;
}

int registerBuiltins(ExtensionPluginRegistry &plugins) {
  return expectSuccess(tianchenrv::plugin::registerBuiltinExtensionPlugins(plugins),
                       "register built-in plugins");
}

void registerDialects(const ExtensionPluginRegistry &plugins,
                      mlir::DialectRegistry &registry) {
  tianchenrv::registerAllDialects(registry);
  tianchenrv::registerPluginDialects(plugins, registry);
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

KernelOp findKernel(mlir::ModuleOp module, llvm::StringRef name) {
  KernelOp kernel;
  module.walk([&](KernelOp candidate) {
    if (candidate.getSymName() == name)
      kernel = candidate;
  });
  return kernel;
}

VariantOp findDirectVariant(KernelOp kernel, llvm::StringRef name) {
  if (!kernel || kernel.getBody().empty())
    return VariantOp();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<VariantOp>(op);
    if (variant && variant.getSymName() == name)
      return variant;
  }
  return VariantOp();
}

LoweringBoundaryOp findDirectBoundary(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return LoweringBoundaryOp();

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op))
      return boundary;
  }
  return LoweringBoundaryOp();
}

unsigned countDirectBoundaries(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return 0;

  unsigned count = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (llvm::isa<LoweringBoundaryOp>(op))
      ++count;
  }
  return count;
}

unsigned countDirectFallbacks(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return 0;

  unsigned count = 0;
  kernel.walk([&](FallbackOp fallback) {
    if (fallback->getParentOfType<KernelOp>() == kernel)
      ++count;
  });
  return count;
}

int expectBoundaryTarget(LoweringBoundaryOp boundary,
                         llvm::StringRef expectedTarget,
                         llvm::StringRef expectedRole) {
  if (int result =
          expect(static_cast<bool>(boundary), "boundary op exists"))
    return result;

  auto target =
      boundary->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
  if (int result = expect(target && target.getValue() == expectedTarget,
                          "boundary selected_variant matches expected RVV"))
    return result;

  auto role = boundary->getAttrOfType<mlir::StringAttr>("role");
  if (int result = expect(role && role.getValue() == expectedRole,
                          "boundary role matches selected path role"))
    return result;

  auto status = boundary->getAttrOfType<mlir::StringAttr>("status");
  if (int result = expect(status && status.getValue() == "unsupported",
                          "boundary remains unsupported"))
    return result;

  auto origin = boundary->getAttrOfType<mlir::StringAttr>("origin");
  if (int result = expect(origin && origin.getValue() == "rvv-plugin",
                          "boundary records RVV origin plugin"))
    return result;

  auto requiredCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>("required_capabilities");
  if (int result =
          expect(requiredCapabilities && !requiredCapabilities.empty(),
                 "boundary records required capability references"))
    return result;

  return 0;
}

int runSelectedRVVWithScalarFallbackBoundaryTest() {
  ExtensionPluginRegistry plugins;
  if (int result = registerBuiltins(plugins))
    return result;

  mlir::DialectRegistry registry;
  registerDialects(plugins, registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_plus_scalar {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_vector_shape = "i32m1",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m1_t",
      tcrv_rvv.selected_vector_suffix = "i32m1",
      tcrv_rvv.selected_setvl_suffix = "e32m1"
    } {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_first_slice {condition = "capability_available"}
      tcrv.exec.fallback @scalar_fallback_first_slice
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV plus scalar fallback IR");

  KernelOp kernel = findKernel(*module, "rvv_plus_scalar");
  if (int result = expect(static_cast<bool>(kernel), "kernel exists"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::plugin::materializeSelectedLoweringBoundaries(kernel,
                                                                    plugins),
          "materialize RVV boundary for selected dispatch case"))
    return result;

  if (int result =
          expect(countDirectBoundaries(kernel) == 1,
                 "exactly one RVV boundary is materialized"))
    return result;
  if (int result =
          expect(countDirectFallbacks(kernel) == 1,
                 "scalar fallback remains in tcrv.exec.dispatch"))
    return result;
  return expectBoundaryTarget(findDirectBoundary(kernel), "rvv_first_slice",
                              "dispatch case");
}

int runScalarOnlyDoesNotMaterializeRVVBoundaryTest() {
  ExtensionPluginRegistry plugins;
  if (int result = registerBuiltins(plugins))
    return result;

  mlir::DialectRegistry registry;
  registerDialects(plugins, registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @scalar_only {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      message = "select scalar fallback envelope",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      severity = "note",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse scalar-only IR");

  KernelOp kernel = findKernel(*module, "scalar_only");
  if (int result = expectSuccess(
          tianchenrv::plugin::materializeSelectedLoweringBoundaries(kernel,
                                                                    plugins),
          "scalar-only selected path is a no-op for RVV boundary"))
    return result;

  return expect(countDirectBoundaries(kernel) == 0,
                "scalar-only selected path creates no RVV boundary");
}

int runRVVOnlySelectedMarkerBoundaryTest() {
  ExtensionPluginRegistry plugins;
  if (int result = registerBuiltins(plugins))
    return result;

  mlir::DialectRegistry registry;
  registerDialects(plugins, registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_only {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_vector_shape = "i32m1",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m1_t",
      tcrv_rvv.selected_vector_suffix = "i32m1",
      tcrv_rvv.selected_setvl_suffix = "e32m1"
    } {
    }
    tcrv.exec.diagnostic {
      message = "select RVV first-slice metadata path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
    tcrv.exec.diagnostic {
      message = "no plugin-provided conservative fallback candidate is available",
      reason = "fallback-coverage-missing",
      selection_kind = "missing-conservative-fallback",
      severity = "warning",
      status = "missing",
      target = @rvv_first_slice
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV-only selected marker IR");

  KernelOp kernel = findKernel(*module, "rvv_only");
  if (int result = expectSuccess(
          tianchenrv::plugin::materializeSelectedLoweringBoundaries(kernel,
                                                                    plugins),
          "materialize RVV direct selected boundary"))
    return result;

  if (int result =
          expect(countDirectBoundaries(kernel) == 1,
                 "RVV-only selected marker creates one boundary"))
    return result;
  return expectBoundaryTarget(findDirectBoundary(kernel), "rvv_first_slice",
                              "direct variant");
}

int runMalformedRVVVariantRejectedBeforeBoundaryTest() {
  ExtensionPluginRegistry plugins;
  if (int result = registerBuiltins(plugins))
    return result;

  mlir::DialectRegistry registry;
  registerDialects(plugins, registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @bad_rvv {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    tcrv.exec.diagnostic {
      message = "select malformed RVV variant",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse malformed RVV variant IR");

  KernelOp kernel = findKernel(*module, "bad_rvv");
  return expectErrorContains(
      tianchenrv::plugin::materializeSelectedLoweringBoundaries(kernel,
                                                               plugins),
      {"failed lowering-boundary materialization", "tcrv_rvv.policy"});
}

int runEmissionPlanStatusesRemainBoundedTest() {
  ExtensionPluginRegistry plugins;
  if (int result = registerBuiltins(plugins))
    return result;

  mlir::DialectRegistry registry;
  registerDialects(plugins, registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @emission_statuses {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_vector_shape = "i32m1",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m1_t",
      tcrv_rvv.selected_vector_suffix = "i32m1",
      tcrv_rvv.selected_setvl_suffix = "e32m1"
    } {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse emission status IR");

  KernelOp kernel = findKernel(*module, "emission_statuses");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  VariantEmissionPlan rvvPlan;
  VariantEmissionRequest rvvRequest(
      findDirectVariant(kernel, "rvv_first_slice"), kernel, capabilities,
      VariantEmissionRole::DispatchCase);
  if (int result = expectSuccess(plugins.buildVariantEmissionPlan(rvvRequest,
                                                                  rvvPlan),
                                 "collect RVV emission plan"))
    return result;
  if (int result = expect(rvvPlan.isUnsupported(),
                          "RVV emission plan remains unsupported"))
    return result;

  VariantEmissionPlan scalarPlan;
  VariantEmissionRequest scalarRequest(
      findDirectVariant(kernel, "scalar_fallback_first_slice"), kernel,
      capabilities, VariantEmissionRole::DispatchFallback);
  if (int result = expectSuccess(
          plugins.buildVariantEmissionPlan(scalarRequest, scalarPlan),
          "collect scalar fallback emission plan"))
    return result;
  return expect(scalarPlan.isUnsupported() &&
                    scalarPlan.getEmissionKind() ==
                        "scalar-fallback-unsupported-emission",
                "scalar fallback emission plan is unsupported fail-closed");
}

int runGenericRegistryRoutingAndDiagnosticTest() {
  mlir::DialectRegistry dialects;
  tianchenrv::registerAllDialects(dialects);
  mlir::MLIRContext context(dialects);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @generic {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic",
      status = "available"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-boundary",
      requires = [@base]
    } {
    }
  }
}
)mlir";

  auto runRegistryCase = [&](BoundaryBehavior behavior,
                             std::initializer_list<llvm::StringRef> fragments) {
    mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
    if (!module)
      return fail("failed to parse generic lowering-boundary registry IR");

    KernelOp kernel = findKernel(*module, "generic");
    VariantOp variant = findDirectVariant(kernel, "fast");
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToEnd(&kernel.getBody().front());

    BoundaryPlugin plugin("mock-boundary", true, behavior);
    ExtensionPluginRegistry registry;
    if (int result =
            expectSuccess(registry.registerPlugin(plugin),
                          "register generic lowering-boundary mock plugin"))
      return result;

    VariantLoweringBoundaryResult boundaryResult;
    llvm::Error error = registry.materializeSelectedLoweringBoundary(
        VariantLoweringBoundaryRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant,
                                       builder),
        boundaryResult);
    if (fragments.size() == 0) {
      if (int result =
              expectSuccess(std::move(error), "route generic boundary request"))
        return result;
      return expect(plugin.calls == 1 && boundaryResult.isMaterialized(),
                    "registry routes materialized boundary result through "
                    "origin plugin");
    }

    return expectErrorContains(std::move(error), fragments);
  };

  if (int result = runRegistryCase(BoundaryBehavior::Materialized, {}))
    return result;
  if (int result =
          runRegistryCase(BoundaryBehavior::Unsupported,
                          {"reported unsupported lowering-boundary "
                           "materialization"}))
    return result;
  if (int result = runRegistryCase(
          BoundaryBehavior::MissingStatus,
          {"invalid lowering-boundary result", "status is missing"}))
    return result;
  if (int result = runRegistryCase(
          BoundaryBehavior::MismatchedOrigin,
          {"invalid lowering-boundary result", "result origin 'other-plugin'",
           "variant origin 'mock-boundary'"}))
    return result;
  if (int result = runRegistryCase(
          BoundaryBehavior::MismatchedVariant,
          {"invalid lowering-boundary result", "result variant @other_variant",
           "request variant @fast"}))
    return result;
  if (int result = runRegistryCase(
          BoundaryBehavior::MismatchedRole,
          {"invalid lowering-boundary result", "result role 'dispatch case'",
           "request role 'direct variant'"}))
    return result;
  if (int result = runRegistryCase(
          BoundaryBehavior::NullMaterializedOp,
          {"invalid lowering-boundary result", "non-null operation"}))
    return result;

  return 0;
}

int runGenericRegistryOriginAndShapeErrorsTest() {
  mlir::DialectRegistry dialects;
  tianchenrv::registerAllDialects(dialects);
  mlir::MLIRContext context(dialects);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @generic {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic",
      status = "available"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-boundary",
      requires = [@base]
    } {
      tcrv.exec.variant @nested attributes {
        origin = "mock-boundary",
        requires = [@base]
      } {
      }
    }
  }
}
)mlir";

  auto runOriginCase = [&](llvm::function_ref<void(VariantOp)> mutate,
                           const BoundaryPlugin &plugin,
                           std::initializer_list<llvm::StringRef> fragments) {
    mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
    if (!module)
      return fail("failed to parse generic origin diagnostic IR");

    KernelOp kernel = findKernel(*module, "generic");
    VariantOp variant = findDirectVariant(kernel, "fast");
    mutate(variant);

    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToEnd(&kernel.getBody().front());

    ExtensionPluginRegistry registry;
    if (!plugin.getName().empty())
      if (int result = expectSuccess(registry.registerPlugin(plugin),
                                     "register origin diagnostic plugin"))
        return result;

    VariantLoweringBoundaryResult boundaryResult;
    return expectErrorContains(
        registry.materializeSelectedLoweringBoundary(
            VariantLoweringBoundaryRequest(variant, kernel, capabilities,
                                           VariantEmissionRole::DirectVariant,
                                           builder),
            boundaryResult),
        fragments);
  };

  BoundaryPlugin enabled("mock-boundary");
  if (int result = runOriginCase(
          [](VariantOp variant) { variant->removeAttr("origin"); }, enabled,
          {"non-empty string attribute 'origin'"}))
    return result;

  if (int result = runOriginCase(
          [&](VariantOp variant) {
            variant->setAttr("origin",
                             mlir::StringAttr::get(&context, "missing-plugin"));
          },
          enabled, {"unknown origin plugin 'missing-plugin'"}))
    return result;

  BoundaryPlugin disabled("disabled-boundary", false);
  if (int result = runOriginCase(
          [&](VariantOp variant) {
            variant->setAttr(
                "origin", mlir::StringAttr::get(&context, "disabled-boundary"));
          },
          disabled, {"origin plugin 'disabled-boundary' is disabled"}))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse non-direct variant diagnostic IR");
  KernelOp kernel = findKernel(*module, "generic");
  VariantOp nested;
  kernel->walk([&](VariantOp variant) {
    if (variant.getSymName() == "nested")
      nested = variant;
  });
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  BoundaryPlugin plugin("mock-boundary");
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(plugin),
                        "register non-direct diagnostic plugin"))
    return result;

  VariantLoweringBoundaryResult boundaryResult;
  return expectErrorContains(
      registry.materializeSelectedLoweringBoundary(
          VariantLoweringBoundaryRequest(nested, kernel, capabilities,
                                         VariantEmissionRole::DirectVariant,
                                         builder),
          boundaryResult),
      {"variant @nested", "not directly enclosed"});
}

int runGenericSelectedPathStructuralErrorsTest() {
  mlir::DialectRegistry dialects;
  tianchenrv::registerAllDialects(dialects);
  mlir::MLIRContext context(dialects);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral duplicateMarkerSource = R"mlir(
module {
  tcrv.exec.kernel @duplicate_marker {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic",
      status = "available"
    }
    tcrv.exec.variant @fast attributes {origin = "mock-boundary", requires = [@base]} {
    }
    tcrv.exec.diagnostic {
      message = "select fast mock path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @fast
    }
    tcrv.exec.diagnostic {
      message = "select fast mock path again",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @fast
    }
  }
}
)mlir";

  {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        parseModule(context, duplicateMarkerSource);
    if (!module)
      return fail("failed to parse duplicate selected marker IR");
    KernelOp kernel = findKernel(*module, "duplicate_marker");
    BoundaryPlugin plugin("mock-boundary");
    ExtensionPluginRegistry registry;
    if (int result =
            expectSuccess(registry.registerPlugin(plugin),
                          "register duplicate marker diagnostic plugin"))
      return result;
    if (int result = expectErrorContains(
            tianchenrv::plugin::materializeSelectedLoweringBoundaries(kernel,
                                                                     registry),
            {"at most one direct selected-path diagnostic marker"}))
      return result;
    if (int result = expect(plugin.calls == 0,
                            "duplicate selected markers fail before plugin"))
      return result;
  }

  constexpr llvm::StringLiteral duplicateDispatchSource = R"mlir(
module {
  tcrv.exec.kernel @duplicate_dispatch_ref {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic",
      status = "available"
    }
    tcrv.exec.variant @fast attributes {origin = "mock-boundary", requires = [@base]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @fast
      tcrv.exec.fallback @fast
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, duplicateDispatchSource);
  if (!module)
    return fail("failed to parse duplicate dispatch reference IR");
  KernelOp kernel = findKernel(*module, "duplicate_dispatch_ref");
  BoundaryPlugin plugin("mock-boundary");
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(plugin),
                        "register duplicate dispatch diagnostic plugin"))
    return result;
  if (int result = expectErrorContains(
          tianchenrv::plugin::materializeSelectedLoweringBoundaries(kernel,
                                                                   registry),
          {"duplicate selected lowering-boundary reference to variant @fast"}))
    return result;
  return expect(plugin.calls == 0,
                "malformed dispatch references fail before plugin");
}

} // namespace

int main() {
  if (int result = runSelectedRVVWithScalarFallbackBoundaryTest())
    return result;
  if (int result = runScalarOnlyDoesNotMaterializeRVVBoundaryTest())
    return result;
  if (int result = runRVVOnlySelectedMarkerBoundaryTest())
    return result;
  if (int result = runMalformedRVVVariantRejectedBeforeBoundaryTest())
    return result;
  if (int result = runEmissionPlanStatusesRemainBoundedTest())
    return result;
  if (int result = runGenericRegistryRoutingAndDiagnosticTest())
    return result;
  if (int result = runGenericRegistryOriginAndShapeErrorsTest())
    return result;
  if (int result = runGenericSelectedPathStructuralErrorsTest())
    return result;

  llvm::outs() << "RVV lowering boundary tests passed\n";
  return 0;
}
