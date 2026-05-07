#include "TianChenRV/Plugin/RVV/RVVLoweringBoundary.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::rvv::LoweringBoundaryOp;

namespace {

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

  auto capabilitySummary =
      boundary->getAttrOfType<mlir::StringAttr>("capability_summary");
  return expect(capabilitySummary &&
                    capabilitySummary.getValue().contains("rvv"),
                "boundary records RVV capability summary");
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
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
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
          tianchenrv::plugin::rvv::materializeRVVLoweringBoundaries(kernel,
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
      message = "select scalar fallback metadata route",
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
          tianchenrv::plugin::rvv::materializeRVVLoweringBoundaries(kernel,
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
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
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
          tianchenrv::plugin::rvv::materializeRVVLoweringBoundaries(kernel,
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
      tianchenrv::plugin::rvv::materializeRVVLoweringBoundaries(kernel,
                                                               plugins),
      {"failed plugin legality", "tcrv_rvv.policy"});
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
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
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
  return expect(scalarPlan.isMetadataOnly(),
                "scalar fallback emission plan remains metadata-only");
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

  llvm::outs() << "RVV lowering boundary tests passed\n";
  return 0;
}
