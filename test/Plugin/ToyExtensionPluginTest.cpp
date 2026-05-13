#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Toy/IR/ToyDialect.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"
#include "TianChenRV/Transforms/VariantSelection.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
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
using tianchenrv::plugin::VariantProposalDecline;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::toy::ComputeSkeletonOp;
using tianchenrv::tcrv::toy::LoweringBoundaryOp;
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

VariantOp findVariant(KernelOp kernel, llvm::StringRef symbolName) {
  VariantOp result;
  if (!kernel)
    return result;
  kernel->walk([&](VariantOp variant) {
    if (variant.getSymName() == symbolName)
      result = variant;
  });
  return result;
}

LoweringBoundaryOp findToyBoundary(KernelOp kernel,
                                   llvm::StringRef selectedVariantSymbol) {
  LoweringBoundaryOp result;
  if (!kernel || kernel.getBody().empty())
    return result;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    if (selectedVariant && selectedVariant.getValue() == selectedVariantSymbol)
      result = boundary;
  }
  return result;
}

ComputeSkeletonOp findToyComputeRole(KernelOp kernel,
                                     llvm::StringRef selectedVariantSymbol) {
  ComputeSkeletonOp result;
  if (!kernel || kernel.getBody().empty())
    return result;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto compute = llvm::dyn_cast<ComputeSkeletonOp>(op);
    if (!compute)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    if (selectedVariant && selectedVariant.getValue() == selectedVariantSymbol)
      result = compute;
  }
  return result;
}

mlir::Attribute findProposalAttribute(const VariantProposal &proposal,
                                      llvm::StringRef attrName) {
  for (mlir::NamedAttribute attribute : proposal.getPluginAttributes()) {
    if (attribute.getName().getValue() == attrName)
      return attribute.getValue();
  }
  return {};
}

int expectProposalStringAttr(const VariantProposal &proposal,
                             llvm::StringRef attrName,
                             llvm::StringRef expectedValue) {
  auto attr = llvm::dyn_cast_if_present<mlir::StringAttr>(
      findProposalAttribute(proposal, attrName));
  if (int result =
          expect(static_cast<bool>(attr),
                 llvm::Twine("proposal carries string attribute ") + attrName))
    return result;
  return expect(attr.getValue() == expectedValue,
                llvm::Twine("proposal string attribute ") + attrName +
                    " preserves expected value");
}

int runRegistrationAndCapabilityMetadataTest() {
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerToyExtensionPlugin(registry),
                        "register Toy plugin"))
    return result;

  const auto *plugin = registry.lookupPlugin(
      tianchenrv::plugin::toy::getToyExtensionPluginName());
  if (int result = expect(plugin, "registered Toy plugin is visible"))
    return result;
  if (int result =
          expect(plugin->getVersion() ==
                     tianchenrv::plugin::toy::getToyExtensionPluginVersion(),
                 "Toy plugin version is stable"))
    return result;

  const PluginCapability *capability = registry.lookupCapabilityByID(
      tianchenrv::plugin::toy::getToyTemplateCapabilityID());
  if (int result =
          expect(capability &&
                     capability->getKind() ==
                         tianchenrv::plugin::toy::
                             getToyTemplateCapabilityKind(),
                 "Toy template capability metadata is registered"))
    return result;

  const auto &manifest =
      tianchenrv::plugin::toy::getToyConstructionManifest();
  const auto &realization =
      tianchenrv::plugin::toy::getToyTypedRoleGraphRealization();
  if (int result = expectSuccess(
          tianchenrv::plugin::toy::verifyToyConstructionManifest(manifest),
          "Toy construction manifest verifies"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::toy::verifyToyTypedRoleGraphRealization(
              manifest, realization),
          "Toy typed role graph verifies"))
    return result;
  llvm::Expected<tianchenrv::plugin::toy::ToyGeneratedOutputRoute> route =
      tianchenrv::plugin::toy::buildToyGeneratedOutputRoute(manifest,
                                                            realization);
  if (!route)
    return fail("Toy generated output route failed: " +
                llvm::toString(route.takeError()));
  if (int result =
          expect(route->steps.size() == 4 &&
                     route->steps[2].operationName ==
                         "tcrv_toy.compute_skeleton" &&
                     route->steps[2].emitCCall == "__tcrv_toy_compute",
                 "Toy generated route preserves ordered compute role"))
    return result;

  return expectErrorContains(
      tianchenrv::plugin::registerToyExtensionPlugin(registry),
      {"duplicate TianChen-RV extension plugin", "toy-plugin"});
}

int runProposalGatingAndDeclineTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @available_toy attributes {} {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
  }

  tcrv.exec.kernel @missing_toy attributes {} {
  }

  tcrv.exec.kernel @unavailable_toy attributes {} {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "unavailable",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
  }

  tcrv.exec.kernel @malformed_toy attributes {} {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "wrong-template-abi",
      handoff_kind = "toy-lowering-template"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse Toy proposal gating module");

  KernelOp available = findKernel(*module, "available_toy");
  KernelOp missing = findKernel(*module, "missing_toy");
  KernelOp unavailable = findKernel(*module, "unavailable_toy");
  KernelOp malformed = findKernel(*module, "malformed_toy");
  if (int result =
          expect(available && missing && unavailable && malformed,
                 "proposal gating module contains all Toy kernels"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerToyExtensionPlugin(registry),
                        "register Toy plugin for proposal gating"))
    return result;

  TargetCapabilitySet availableCapabilities =
      TargetCapabilitySet::buildFromKernel(available);
  VariantProposalRequest availableRequest(available.getOperation(), available,
                                          availableCapabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  llvm::SmallVector<VariantProposalDecline, 1> declines;
  if (int result = expectSuccess(
          registry.collectVariantProposals(availableRequest, proposals,
                                           &declines),
          "available Toy capability collects proposal"))
    return result;
  if (int result =
          expect(proposals.size() == 1 && declines.empty(),
                 "available Toy capability proposes one variant"))
    return result;

  const VariantProposal &proposal = proposals.front();
  if (int result =
          expect(proposal.getVariantName() ==
                         tianchenrv::plugin::toy::
                             getToyTemplateFirstSliceVariantName() &&
                     proposal.getOriginPlugin() ==
                         tianchenrv::plugin::toy::
                             getToyExtensionPluginName() &&
                     proposal.getPolicy() ==
                         tianchenrv::plugin::toy::getToyTemplatePolicy(),
                 "Toy proposal preserves stable generic metadata"))
    return result;
  if (int result =
          expect(proposal.getRequiredCapabilityIDs().size() == 1 &&
                     proposal.getRequiredCapabilityIDs().front() ==
                         tianchenrv::plugin::toy::getToyTemplateCapabilityID(),
                 "Toy proposal requires toy.template capability id"))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, tianchenrv::plugin::toy::getToyTemplateABIAttrName(),
          tianchenrv::plugin::toy::getToyExpectedTemplateABI()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, tianchenrv::plugin::toy::getToyHandoffKindAttrName(),
          tianchenrv::plugin::toy::getToyExpectedHandoffKind()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_toy.construction_protocol",
          tianchenrv::plugin::toy::getToyConstructionManifest()
              .protocolVersion))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_toy.semantic_role_graph",
          tianchenrv::plugin::toy::getToyConstructionManifest()
              .semanticRoleGraph))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_toy.typed_role_realization",
          tianchenrv::plugin::toy::getToyTypedRoleRealizationSummary()))
    return result;

  auto expectNoProposal = [&](KernelOp kernel, llvm::StringRef context) -> int {
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    VariantProposalRequest request(kernel.getOperation(), kernel, capabilities);
    proposals.clear();
    declines.clear();
    if (int result = expectSuccess(
            registry.collectVariantProposals(request, proposals, &declines),
            llvm::Twine(context) + " query succeeds"))
      return result;
    return expect(proposals.empty() && declines.empty(),
                  llvm::Twine(context) +
                      " produces no proposal without an available Toy "
                      "capability");
  };

  if (int result = expectNoProposal(missing, "missing Toy capability"))
    return result;
  if (int result = expectNoProposal(unavailable, "unavailable Toy capability"))
    return result;

  TargetCapabilitySet malformedCapabilities =
      TargetCapabilitySet::buildFromKernel(malformed);
  VariantProposalRequest malformedRequest(malformed.getOperation(), malformed,
                                          malformedCapabilities);
  proposals.clear();
  declines.clear();
  if (int result = expectSuccess(
          registry.collectVariantProposals(malformedRequest, proposals,
                                           &declines),
          "malformed Toy capability decline is recoverable"))
    return result;
  return expect(proposals.empty() && declines.size() == 1 &&
                    declines.front().getPluginName() ==
                        tianchenrv::plugin::toy::getToyExtensionPluginName() &&
                    declines.front().getReason().contains("template_abi"),
                "malformed Toy capability records plugin-local decline");
}

int runPipelineHookTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @toy_template_kernel attributes {} {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse Toy pipeline hook module");

  KernelOp kernel = findKernel(*module, "toy_template_kernel");
  if (int result = expect(kernel, "Toy pipeline module has kernel anchor"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerToyExtensionPlugin(registry),
                        "register Toy plugin for pipeline hook"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(kernel.getOperation(), kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize Toy proposal"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "Toy capability materializes one variant"))
    return result;

  VariantOp toyVariant = findVariant(
      kernel,
      tianchenrv::plugin::toy::getToyTemplateFirstSliceVariantName());
  if (int result = expect(toyVariant, "Toy variant is materialized"))
    return result;
  if (int result =
          expect(toyVariant->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     tianchenrv::plugin::toy::getToyExtensionPluginName(),
                 "Toy variant has Toy origin"))
    return result;
  if (int result =
          expect(toyVariant
                         ->getAttrOfType<mlir::StringAttr>(
                             tianchenrv::plugin::toy::
                                 getToyTemplateABIAttrName())
                         .getValue() ==
                     tianchenrv::plugin::toy::getToyExpectedTemplateABI(),
                 "Toy variant carries template ABI metadata"))
    return result;

  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "materialized Toy module verifies"))
    return result;
  if (int result =
          expectSuccess(registry.verifyKernelVariantLegality(kernel,
                                                             capabilities),
                        "Toy legality accepts materialized variant"))
    return result;

  VariantCostEstimate estimate;
  if (int result = expectSuccess(
          registry.estimateVariantCost(
              VariantCostRequest(toyVariant, kernel, capabilities), estimate),
          "Toy cost estimate routes through plugin"))
    return result;
  if (int result =
          expect(estimate.hasScore() && estimate.getScore() == 50.0 &&
                     estimate.hasExplicitPreference() &&
                     estimate.getOriginPlugin() ==
                         tianchenrv::plugin::toy::
                             getToyExtensionPluginName() &&
                     estimate.getVariantSymbol() == toyVariant.getSymName(),
                 "Toy cost metadata is plugin-owned"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("Toy selection planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan selectionPlan = std::move(*planOrError);
  if (int result =
          expect(selectionPlan.kind == VariantSelectionKind::StaticVariant &&
                     selectionPlan.selectedVariant == toyVariant &&
                     !selectionPlan.fallback && selectionPlan.dispatchCases.empty(),
                 "Toy selected path uses the generic static variant plan"))
    return result;

  DiagnosticOp marker;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize Toy selected marker"))
    return result;
  if (int result = expect(marker, "Toy selected marker was created"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::plugin::materializeSelectedLoweringBoundaries(
              kernel, capabilities, registry),
          "materialize Toy selected boundary"))
    return result;

  LoweringBoundaryOp boundary =
      findToyBoundary(kernel, toyVariant.getSymName());
  if (int result =
          expect(boundary,
                 "Toy selected boundary is materialized through plugin"))
    return result;
  if (int result =
          expect(boundary->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     tianchenrv::plugin::toy::
                         getToyExtensionPluginName() &&
                     boundary->getAttrOfType<mlir::StringAttr>("role")
                         .getValue() == "direct variant" &&
                     boundary->getAttrOfType<mlir::StringAttr>("status")
                         .getValue() == "metadata-only" &&
                     boundary->getAttrOfType<mlir::StringAttr>("template_abi")
                         .getValue() ==
                         tianchenrv::plugin::toy::
                             getToyExpectedTemplateABI(),
                 "Toy boundary records metadata-only template handoff"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "Toy boundary module verifies"))
    return result;

  ComputeSkeletonOp computeRole =
      findToyComputeRole(kernel, toyVariant.getSymName());
  if (int result =
          expect(computeRole,
                 "Toy selected compute role op is materialized through plugin"))
    return result;
  if (int result =
          expect(computeRole
                         ->getAttrOfType<mlir::StringAttr>("typed_role")
                         .getValue() ==
                         "toy.role.compute.compute_skeleton" &&
                     computeRole
                         ->getAttrOfType<mlir::StringAttr>("source_role")
                         .getValue() == "compute" &&
                     computeRole
                         ->getAttrOfType<mlir::StringAttr>(
                             "role_specific_interface")
                         .getValue() == "TCRVComputeOpInterface" &&
                     computeRole
                         ->getAttrOfType<mlir::StringAttr>("emitc_call")
                         .getValue() == "__tcrv_toy_compute",
                 "Toy compute role op carries typed construction metadata"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::toy::verifyToyComputeRoleOpInterface(
              tianchenrv::plugin::toy::getToyConstructionManifest(),
              tianchenrv::plugin::toy::getToyTypedRoleGraphRealization(),
              computeRole.getOperation()),
          "Toy compute role op validates against construction protocol"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectSuccess(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(toyVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          "Toy emission readiness is plugin-owned"))
    return result;
  if (int result =
          expect(status.isMetadataOnly() &&
                     status.getEmissionPath().contains("toy-template"),
                 "Toy readiness reports metadata-only template route"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(toyVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "Toy emission plan is plugin-owned"))
    return result;
  if (int result =
          expect(emissionPlan.isSupported() &&
                     emissionPlan.getOriginPlugin() ==
                         tianchenrv::plugin::toy::
                             getToyExtensionPluginName() &&
                     emissionPlan.getKernelSymbol() == kernel.getSymName() &&
                     emissionPlan.getVariantSymbol() ==
                         toyVariant.getSymName() &&
                     emissionPlan.getEmissionKind() ==
                         tianchenrv::plugin::toy::
                             getToyMetadataEmissionKind() &&
                     emissionPlan.getLoweringPipeline() ==
                         tianchenrv::plugin::toy::getToyMetadataRouteID() &&
                     emissionPlan.getArtifactKind() ==
                         tianchenrv::plugin::toy::
                             getToyMetadataArtifactKind() &&
                     emissionPlan.getRuntimeABIKind() ==
                         tianchenrv::plugin::toy::
                             getToyMetadataRuntimeABIKind() &&
                     emissionPlan.getRuntimeABIName() ==
                         tianchenrv::plugin::toy::
                             getToyExpectedTemplateABI() &&
                     emissionPlan.getRuntimeGlueRole() ==
                         tianchenrv::plugin::toy::
                             getToyMetadataRuntimeGlueRole() &&
                     emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                     emissionPlan.getRequiredCapabilitySymbols().front() ==
                         tianchenrv::plugin::toy::
                             getToyTemplatePreferredCapabilitySymbol(),
                 "Toy emission plan records stable exportable metadata route"))
    return result;
  if (int result =
          expect(emissionPlan.getSelectedPlanMetadata().size() == 10,
                 "Toy emission plan records construction selected metadata"))
    return result;

  return 0;
}

} // namespace

int main() {
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerToyExtensionPlugin(
                            dialectPlugins),
                        "register Toy plugin for dialect setup"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(dialectPlugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runRegistrationAndCapabilityMetadataTest())
    return result;
  if (int result = runProposalGatingAndDeclineTest(context))
    return result;
  if (int result = runPipelineHookTest(context))
    return result;

  llvm::outs() << "Toy extension plugin template smoke test passed\n";
  return 0;
}
