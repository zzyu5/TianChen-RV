#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"
#include "TianChenRV/Transforms/VariantSelection.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantCostEstimate;
using tianchenrv::plugin::VariantCostRequest;
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmissionStatus;
using tianchenrv::plugin::VariantProposal;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::transforms::VariantSelectionKind;
using tianchenrv::transforms::VariantSelectionPlan;

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
                        std::initializer_list<llvm::StringRef> needles) {
  if (!error)
    return fail("expected error but operation succeeded");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef needle : needles) {
    if (!llvm::StringRef(message).contains(needle))
      return fail("expected error to contain '" + needle +
                  "' but got: " + message);
  }
  return 0;
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

KernelOp findKernel(mlir::ModuleOp module, llvm::StringRef symbolName) {
  KernelOp result;
  module.walk([&](KernelOp kernel) {
    if (kernel.getSymName() == symbolName)
      result = kernel;
  });
  return result;
}

mlir::func::FuncOp findHighLevelPlaceholder(mlir::ModuleOp module) {
  mlir::func::FuncOp result;
  module.walk([&](mlir::func::FuncOp function) {
    if (!result)
      result = function;
  });
  return result;
}

VariantOp findVariant(KernelOp kernel, llvm::StringRef symbolName) {
  VariantOp result;
  kernel->walk([&](VariantOp variant) {
    if (variant.getSymName() == symbolName)
      result = variant;
  });
  return result;
}

int runRegistrationAndCapabilityMetadataTest() {
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar fallback plugin"))
    return result;

  const auto *plugin = registry.lookupPlugin(
      tianchenrv::plugin::scalar::getScalarExtensionPluginName());
  if (int result = expect(plugin, "registered scalar fallback plugin is visible"))
    return result;
  if (int result =
          expect(plugin->getVersion() ==
                     tianchenrv::plugin::scalar::
                         getScalarExtensionPluginVersion(),
                 "scalar fallback plugin version is stable"))
    return result;

  const PluginCapability *capability = registry.lookupCapabilityByID(
      tianchenrv::plugin::scalar::getScalarFallbackCapabilityID());
  if (int result = expect(
          capability &&
              capability->getKind() ==
                  tianchenrv::plugin::scalar::
                      getScalarFallbackCapabilityKind(),
          "scalar fallback capability metadata is registered"))
    return result;

  llvm::SmallVector<const tianchenrv::plugin::ExtensionPlugin *, 2> enabled;
  registry.getEnabledPlugins(enabled);
  if (int result =
          expect(enabled.size() == 1 && enabled.front() == plugin,
                 "scalar fallback plugin is enabled through registry"))
    return result;

  return 0;
}

int runProposalGatingTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @available_scalar attributes {} {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }

  tcrv.exec.kernel @unavailable_scalar attributes {} {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "unavailable"
    }
  }

  tcrv.exec.kernel @missing_scalar attributes {} {
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse scalar proposal gating module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp available = findKernel(*module, "available_scalar");
  KernelOp unavailable = findKernel(*module, "unavailable_scalar");
  KernelOp missing = findKernel(*module, "missing_scalar");
  if (int result =
          expect(highLevelOp && available && unavailable && missing,
                 "proposal gating module contains all anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar fallback plugin for proposal gating"))
    return result;

  TargetCapabilitySet availableCapabilities =
      TargetCapabilitySet::buildFromKernel(available);
  VariantProposalRequest availableRequest(highLevelOp.getOperation(), available,
                                          availableCapabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result = expectSuccess(
          registry.collectVariantProposals(availableRequest, proposals),
          "available scalar fallback capability collects proposal"))
    return result;
  if (int result =
          expect(proposals.size() == 1,
                 "available scalar fallback capability proposes one variant"))
    return result;
  if (int result =
          expect(proposals.front().getVariantName() ==
                         tianchenrv::plugin::scalar::
                             getScalarFallbackFirstSliceVariantName() &&
                     proposals.front().getOriginPlugin() ==
                         tianchenrv::plugin::scalar::
                             getScalarExtensionPluginName() &&
                     proposals.front().getPolicy() ==
                         tianchenrv::plugin::scalar::getScalarFallbackPolicy(),
                 "scalar fallback proposal preserves stable metadata"))
    return result;
  if (int result =
          expect(proposals.front().getRequiredCapabilityIDs().size() == 1 &&
                     proposals.front().getRequiredCapabilityIDs().front() ==
                         tianchenrv::plugin::scalar::
                             getScalarFallbackCapabilityID(),
                 "scalar fallback proposal requires fallback capability id"))
    return result;

  TargetCapabilitySet unavailableCapabilities =
      TargetCapabilitySet::buildFromKernel(unavailable);
  VariantProposalRequest unavailableRequest(highLevelOp.getOperation(),
                                            unavailable,
                                            unavailableCapabilities);
  proposals.clear();
  if (int result = expectSuccess(
          registry.collectVariantProposals(unavailableRequest, proposals),
          "unavailable scalar fallback capability query succeeds"))
    return result;
  if (int result =
          expect(proposals.empty(),
                 "unavailable scalar fallback capability produces no proposal"))
    return result;

  TargetCapabilitySet missingCapabilities =
      TargetCapabilitySet::buildFromKernel(missing);
  VariantProposalRequest missingRequest(highLevelOp.getOperation(), missing,
                                        missingCapabilities);
  proposals.clear();
  if (int result = expectSuccess(
          registry.collectVariantProposals(missingRequest, proposals),
          "missing scalar fallback capability query succeeds"))
    return result;
  if (int result =
          expect(proposals.empty(),
                 "missing scalar fallback capability produces no proposal"))
    return result;

  VariantProposalRequest noHighLevelOpRequest(nullptr, available,
                                              availableCapabilities);
  proposals.clear();
  if (int result = expectSuccess(
          registry.collectVariantProposals(noHighLevelOpRequest, proposals),
          "missing high-level op query succeeds"))
    return result;
  if (int result =
          expect(proposals.empty(),
                 "missing high-level op produces no scalar proposal"))
    return result;

  return 0;
}

int runMaterializationSelectionAndEmissionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @scalar_only attributes {} {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse scalar materialization module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "scalar_only");
  if (int result =
          expect(highLevelOp && kernel, "materialization module has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar fallback plugin for materialization"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize scalar fallback variant proposal"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "one scalar fallback variant is materialized"))
    return result;

  VariantOp variant = materializedVariants.front();
  if (int result =
          expect(variant.getSymName() ==
                     tianchenrv::plugin::scalar::
                         getScalarFallbackFirstSliceVariantName(),
                 "materialized scalar fallback variant has stable symbol"))
    return result;
  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  if (int result =
          expect(originAttr &&
                     originAttr.getValue() ==
                     tianchenrv::plugin::scalar::
                         getScalarExtensionPluginName(),
                 "materialized scalar fallback variant has scalar origin"))
    return result;
  auto policyAttr = variant->getAttrOfType<mlir::StringAttr>("policy");
  if (int result =
          expect(policyAttr &&
                     policyAttr.getValue() ==
                     tianchenrv::plugin::scalar::getScalarFallbackPolicy(),
                 "materialized scalar fallback variant preserves policy"))
    return result;

  auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (int result =
          expect(requiresAttr && requiresAttr.size() == 1,
                 "materialized scalar fallback variant has one requirement"))
    return result;
  auto requiredSymbol =
      llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiresAttr[0]);
  if (int result = expect(
          requiredSymbol &&
              requiredSymbol.getValue() ==
                  tianchenrv::plugin::scalar::
                      getScalarFallbackPreferredCapabilitySymbol(),
          "materialized scalar fallback variant requires @scalar_fallback"))
    return result;

  if (int result =
          expect(mlir::succeeded(mlir::verify(*module)),
                 "materialized scalar fallback module verifies"))
    return result;
  if (int result =
          expectSuccess(registry.verifyKernelVariantLegality(kernel,
                                                             capabilities),
                        "scalar fallback legality accepts materialized variant"))
    return result;

  VariantCostEstimate estimate;
  if (int result = expectSuccess(
          registry.estimateVariantCost(
              VariantCostRequest(variant, kernel, capabilities), estimate),
          "scalar fallback cost estimate routes through plugin"))
    return result;
  if (int result =
          expect(estimate.hasScore() && estimate.getScore() == 1000.0 &&
                     estimate.getOriginPlugin() ==
                         tianchenrv::plugin::scalar::
                             getScalarExtensionPluginName() &&
                     estimate.getVariantSymbol() == variant.getSymName(),
                 "scalar fallback cost metadata is plugin-owned"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("scalar fallback selection planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan selectionPlan = std::move(*planOrError);
  if (int result =
          expect(selectionPlan.kind == VariantSelectionKind::FallbackOnly &&
                     selectionPlan.fallback == variant &&
                     selectionPlan.selectedVariant == variant,
                 "scalar fallback-only variant participates in selection"))
    return result;

  DiagnosticOp marker;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize scalar fallback selected-path marker"))
    return result;
  if (int result = expect(marker, "selected-path marker was created"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectSuccess(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          "scalar fallback emission readiness is plugin-owned"))
    return result;
  if (int result =
          expect(status.isSupported() &&
                     status.getEmissionPath() ==
                         "portable-scalar-fallback-metadata-route",
                 "scalar fallback readiness reports supported metadata route"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "scalar fallback emission plan is plugin-owned"))
    return result;
  if (int result =
          expect(emissionPlan.isSupported() &&
                     emissionPlan.getOriginPlugin() ==
                         tianchenrv::plugin::scalar::
                             getScalarExtensionPluginName() &&
                     emissionPlan.getKernelSymbol() == kernel.getSymName() &&
                     emissionPlan.getVariantSymbol() == variant.getSymName() &&
                     emissionPlan.getEmissionKind() ==
                         "portable-scalar-fallback" &&
                     emissionPlan.getLoweringPipeline() ==
                         "mlir-default-scalar-lowering" &&
                     emissionPlan.getRuntimeABI() == "none-required" &&
                     emissionPlan.getArtifactKind() == "mlir-lowering-plan",
                 "scalar fallback emission plan records stable route metadata"))
    return result;

  return 0;
}

int runLegalityRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @scalar_legality_rejections attributes {} {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @missing_requirement attributes {
      origin = "scalar-plugin",
      requires = []
    } {
    }
  }

  tcrv.exec.kernel @scalar_unavailable_rejection attributes {} {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "unavailable"
    }
    tcrv.exec.variant @requires_unavailable attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse scalar legality rejection module");

  KernelOp missingRequirementKernel =
      findKernel(*module, "scalar_legality_rejections");
  KernelOp unavailableKernel =
      findKernel(*module, "scalar_unavailable_rejection");
  VariantOp missingRequirement =
      findVariant(missingRequirementKernel, "missing_requirement");
  VariantOp requiresUnavailable =
      findVariant(unavailableKernel, "requires_unavailable");
  if (int result =
          expect(missingRequirementKernel && unavailableKernel &&
                     missingRequirement && requiresUnavailable,
                 "legality rejection module has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar fallback plugin for legality negatives"))
    return result;

  TargetCapabilitySet availableCapabilities =
      TargetCapabilitySet::buildFromKernel(missingRequirementKernel);
  if (int result = expectErrorContains(
          registry.verifyVariantLegality(tianchenrv::plugin::
                                             VariantLegalityRequest(
                                                 missingRequirement,
                                                 missingRequirementKernel,
                                                 availableCapabilities)),
          {"scalar fallback", "must require capability id",
           "scalar.fallback"}))
    return result;

  TargetCapabilitySet unavailableCapabilities =
      TargetCapabilitySet::buildFromKernel(unavailableKernel);
  if (int result = expectErrorContains(
          registry.verifyVariantLegality(tianchenrv::plugin::
                                             VariantLegalityRequest(
                                                 requiresUnavailable,
                                                 unavailableKernel,
                                                 unavailableCapabilities)),
          {"scalar fallback", "requires an available capability id",
           "scalar.fallback"}))
    return result;

  return 0;
}

} // namespace

int main() {
  if (int result = runRegistrationAndCapabilityMetadataTest())
    return result;

  mlir::DialectRegistry dialectRegistry;
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            dialectPlugins),
                        "register scalar fallback plugin for dialect context"))
    return result;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(dialectPlugins, dialectRegistry);
  dialectRegistry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runProposalGatingTest(context))
    return result;
  if (int result = runMaterializationSelectionAndEmissionTest(context))
    return result;
  if (int result = runLegalityRejectionTest(context))
    return result;

  llvm::outs() << "scalar fallback extension plugin smoke test passed\n";
  return 0;
}
