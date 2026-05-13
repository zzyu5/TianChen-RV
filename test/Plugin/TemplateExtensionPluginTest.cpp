#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Template/IR/TemplateDialect.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
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
using tianchenrv::plugin::VariantSelectedPlanMetadata;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::template_ext::LoweringBoundaryOp;
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

LoweringBoundaryOp findTemplateBoundary(KernelOp kernel,
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

const VariantSelectedPlanMetadata *
findEmissionPlanMetadata(const VariantEmissionPlan &plan,
                         llvm::StringRef name) {
  for (const VariantSelectedPlanMetadata &metadata :
       plan.getSelectedPlanMetadata()) {
    if (metadata.name == name)
      return &metadata;
  }
  return nullptr;
}

int expectEmissionPlanMetadata(const VariantEmissionPlan &plan,
                               llvm::StringRef name,
                               llvm::StringRef expectedValue,
                               llvm::StringRef expectedRole) {
  const VariantSelectedPlanMetadata *metadata =
      findEmissionPlanMetadata(plan, name);
  if (int result =
          expect(metadata, llvm::Twine("emission plan carries metadata ") + name))
    return result;
  return expect(metadata->value == expectedValue && metadata->role == expectedRole &&
                    !metadata->note.empty(),
                llvm::Twine("emission plan metadata ") + name +
                    " preserves expected value and role");
}

int runConstructionManifestTest() {
  const auto &manifest =
      tianchenrv::plugin::template_ext::getTemplateConstructionManifest();
  if (int result = expectSuccess(
          tianchenrv::plugin::template_ext::
              verifyTemplateConstructionManifest(manifest),
          "Template construction manifest verifies"))
    return result;

  if (int result = expect(
          manifest.protocolVersion ==
                  "extension-family-construction-protocol.v1" &&
              manifest.archetype == "custom-riscv-extension-minimal" &&
              manifest.semanticRoleGraph == "configure->load->compute->store",
          "Template manifest exposes construction protocol, archetype, and role graph"))
    return result;
  if (int result = expect(
          manifest.family.pluginName ==
                  tianchenrv::plugin::template_ext::
                      getTemplateExtensionPluginName() &&
              manifest.family.capabilityID ==
                  tianchenrv::plugin::template_ext::
                      getTemplateExtensionCapabilityID() &&
              manifest.family.concreteNamespace == "tcrv_template",
          "Template manifest family declaration agrees with plugin metadata"))
    return result;
  if (int result =
          expect(manifest.semanticRoles.size() == 4 &&
                     manifest.semanticRoles[0].role == "configure" &&
                     manifest.semanticRoles[1].role == "load" &&
                     manifest.semanticRoles[2].role == "compute" &&
                     manifest.semanticRoles[3].role == "store",
                 "Template manifest records ordered semantic role graph"))
    return result;
  for (const auto &role : manifest.semanticRoles) {
    if (int result = expect(
            role.commonInterfaces.contains("TCRVExtensionOpInterface") &&
                role.commonInterfaces.contains("TCRVEmitCLowerableInterface"),
            llvm::Twine("Template role realizes common interfaces: ") +
                role.role))
      return result;
  }
  if (int result = expect(
          tianchenrv::plugin::template_ext::
              getTemplateConstructionInterfaceRealization()
                  .contains("compute=TCRVExtensionOpInterface") &&
              tianchenrv::plugin::template_ext::
                  getTemplateConstructionInterfaceRealization()
                      .contains("TCRVComputeOpInterface"),
          "Template manifest exposes common-interface realization mapping"))
    return result;
  if (int result = expect(
          manifest.emitcRoute.routeID ==
                  tianchenrv::plugin::template_ext::
                      getTemplateMetadataRouteID() &&
              manifest.emitcRoute.requiredHeader ==
                  "template_extension_intrinsics.h" &&
              manifest.emitcRoute.roleToCallMap.contains(
                  "compute=__tcrv_template_compute"),
          "Template manifest exposes plugin-owned EmitC route mapping"))
    return result;
  return expect(
      manifest.evidenceProfile.contains("parse_verify") &&
          manifest.evidenceProfile.contains("interface") &&
          manifest.evidenceProfile.contains("emitc_route_mapping") &&
          manifest.evidenceProfile.contains("generated_output"),
      "Template manifest records focused evidence profile");
}

int runRegistrationAndCapabilityMetadataTest() {
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTemplateExtensionPlugin(registry),
                        "register Template plugin"))
    return result;

  const auto *plugin = registry.lookupPlugin(
      tianchenrv::plugin::template_ext::getTemplateExtensionPluginName());
  if (int result = expect(plugin, "registered Template plugin is visible"))
    return result;
  if (int result =
          expect(plugin->getVersion() ==
                     tianchenrv::plugin::template_ext::getTemplateExtensionPluginVersion(),
                 "Template plugin version is stable"))
    return result;

  const PluginCapability *capability = registry.lookupCapabilityByID(
      tianchenrv::plugin::template_ext::getTemplateExtensionCapabilityID());
  if (int result =
          expect(capability &&
                     capability->getKind() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateExtensionCapabilityKind(),
                 "Template extension capability metadata is registered"))
    return result;

  const auto &manifest =
      tianchenrv::plugin::template_ext::getTemplateConstructionManifest();
  if (int result =
          expect(manifest.family.pluginName == plugin->getName() &&
                     manifest.family.capabilityID == capability->getID() &&
                     manifest.family.capabilityKind == capability->getKind(),
                 "Template construction manifest agrees with registered plugin "
                 "capability"))
    return result;

  return expectErrorContains(
      tianchenrv::plugin::registerTemplateExtensionPlugin(registry),
      {"duplicate TianChen-RV extension plugin", "template-plugin"});
}

int runProposalGatingAndDeclineTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @available_template attributes {} {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }

  tcrv.exec.kernel @missing_template attributes {} {
  }

  tcrv.exec.kernel @unavailable_template attributes {} {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "unavailable",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }

  tcrv.exec.kernel @malformed_template attributes {} {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "wrong-integration-contract",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse Template proposal gating module");

  KernelOp available = findKernel(*module, "available_template");
  KernelOp missing = findKernel(*module, "missing_template");
  KernelOp unavailable = findKernel(*module, "unavailable_template");
  KernelOp malformed = findKernel(*module, "malformed_template");
  if (int result =
          expect(available && missing && unavailable && malformed,
                 "proposal gating module contains all Template kernels"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTemplateExtensionPlugin(registry),
                        "register Template plugin for proposal gating"))
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
          "available Template capability collects proposal"))
    return result;
  if (int result =
          expect(proposals.size() == 1 && declines.empty(),
                 "available Template capability proposes one variant"))
    return result;

  const VariantProposal &proposal = proposals.front();
  if (int result =
          expect(proposal.getVariantName() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateExtensionFirstSliceVariantName() &&
                     proposal.getOriginPlugin() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateExtensionPluginName() &&
                     proposal.getPolicy() ==
                         tianchenrv::plugin::template_ext::getTemplateExtensionPolicy(),
                 "Template proposal preserves stable generic metadata"))
    return result;
  if (int result =
          expect(proposal.getRequiredCapabilityIDs().size() == 1 &&
                     proposal.getRequiredCapabilityIDs().front() ==
                         tianchenrv::plugin::template_ext::getTemplateExtensionCapabilityID(),
                 "Template proposal requires template.extension capability id"))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, tianchenrv::plugin::template_ext::getTemplateIntegrationContractAttrName(),
          tianchenrv::plugin::template_ext::getTemplateExpectedIntegrationContract()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, tianchenrv::plugin::template_ext::getTemplateHandoffKindAttrName(),
          tianchenrv::plugin::template_ext::getTemplateExpectedHandoffKind()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_template.construction_protocol",
          tianchenrv::plugin::template_ext::
              getTemplateConstructionProtocolVersion()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_template.archetype",
          tianchenrv::plugin::template_ext::
              getTemplateConstructionArchetype()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_template.semantic_role_graph",
          tianchenrv::plugin::template_ext::
              getTemplateConstructionSemanticRoleGraph()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_template.common_interface_realization",
          tianchenrv::plugin::template_ext::
              getTemplateConstructionInterfaceRealization()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_template.emitc_route_mapping",
          tianchenrv::plugin::template_ext::getTemplateMetadataRouteID()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_template.evidence_profile",
          tianchenrv::plugin::template_ext::
              getTemplateConstructionEvidenceProfile()))
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
                      " produces no proposal without an available Template "
                      "capability");
  };

  if (int result = expectNoProposal(missing, "missing Template capability"))
    return result;
  if (int result = expectNoProposal(unavailable, "unavailable Template capability"))
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
          "malformed Template capability decline is recoverable"))
    return result;
  return expect(proposals.empty() && declines.size() == 1 &&
                    declines.front().getPluginName() ==
                        tianchenrv::plugin::template_ext::getTemplateExtensionPluginName() &&
                    declines.front().getReason().contains("integration_contract"),
                "malformed Template capability records plugin-local decline");
}

int runPipelineHookTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @template_extension_kernel attributes {} {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse Template pipeline hook module");

  KernelOp kernel = findKernel(*module, "template_extension_kernel");
  if (int result = expect(kernel, "Template pipeline module has kernel anchor"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTemplateExtensionPlugin(registry),
                        "register Template plugin for pipeline hook"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(kernel.getOperation(), kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize Template proposal"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "Template capability materializes one variant"))
    return result;

  VariantOp templateVariant = findVariant(
      kernel,
      tianchenrv::plugin::template_ext::getTemplateExtensionFirstSliceVariantName());
  if (int result = expect(templateVariant, "Template variant is materialized"))
    return result;
  if (int result =
          expect(templateVariant->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     tianchenrv::plugin::template_ext::getTemplateExtensionPluginName(),
                 "Template variant has Template origin"))
    return result;
  if (int result =
          expect(templateVariant
                         ->getAttrOfType<mlir::StringAttr>(
                             tianchenrv::plugin::template_ext::
                                 getTemplateIntegrationContractAttrName())
                         .getValue() ==
                     tianchenrv::plugin::template_ext::getTemplateExpectedIntegrationContract(),
                 "Template variant carries integration contract metadata"))
    return result;
  if (int result =
          expect(templateVariant
                         ->getAttrOfType<mlir::StringAttr>(
                             "tcrv_template.construction_protocol")
                         .getValue() ==
                     tianchenrv::plugin::template_ext::
                         getTemplateConstructionProtocolVersion() &&
                     templateVariant
                             ->getAttrOfType<mlir::StringAttr>(
                                 "tcrv_template.semantic_role_graph")
                             .getValue() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateConstructionSemanticRoleGraph() &&
                     templateVariant
                             ->getAttrOfType<mlir::StringAttr>(
                                 "tcrv_template.common_interface_realization")
                             .getValue()
                             .contains("TCRVComputeOpInterface"),
                 "Template variant carries code-consumed construction manifest "
                 "metadata"))
    return result;

  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "materialized Template module verifies"))
    return result;
  if (int result =
          expectSuccess(registry.verifyKernelVariantLegality(kernel,
                                                             capabilities),
                        "Template legality accepts materialized variant"))
    return result;

  VariantCostEstimate estimate;
  if (int result = expectSuccess(
          registry.estimateVariantCost(
              VariantCostRequest(templateVariant, kernel, capabilities), estimate),
          "Template cost estimate routes through plugin"))
    return result;
  if (int result =
          expect(estimate.hasScore() && estimate.getScore() == 50.0 &&
                     estimate.hasExplicitPreference() &&
                     estimate.getOriginPlugin() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateExtensionPluginName() &&
                     estimate.getVariantSymbol() == templateVariant.getSymName(),
                 "Template cost metadata is plugin-owned"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("Template selection planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan selectionPlan = std::move(*planOrError);
  if (int result =
          expect(selectionPlan.kind == VariantSelectionKind::StaticVariant &&
                     selectionPlan.selectedVariant == templateVariant &&
                     !selectionPlan.fallback && selectionPlan.dispatchCases.empty(),
                 "Template selected path uses the generic static variant plan"))
    return result;

  DiagnosticOp marker;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize Template selected marker"))
    return result;
  if (int result = expect(marker, "Template selected marker was created"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::plugin::materializeSelectedLoweringBoundaries(
              kernel, capabilities, registry),
          "materialize Template selected boundary"))
    return result;

  LoweringBoundaryOp boundary =
      findTemplateBoundary(kernel, templateVariant.getSymName());
  if (int result =
          expect(boundary,
                 "Template selected boundary is materialized through plugin"))
    return result;
  if (int result =
          expect(boundary->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     tianchenrv::plugin::template_ext::
                         getTemplateExtensionPluginName() &&
                     boundary->getAttrOfType<mlir::StringAttr>("role")
                         .getValue() == "direct variant" &&
                     boundary->getAttrOfType<mlir::StringAttr>("status")
                         .getValue() == "metadata-only" &&
                     boundary->getAttrOfType<mlir::StringAttr>("integration_contract")
                         .getValue() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateExpectedIntegrationContract(),
                 "Template boundary records metadata-only template handoff"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "Template boundary module verifies"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectSuccess(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(templateVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          "Template emission readiness is plugin-owned"))
    return result;
  if (int result =
          expect(status.isMetadataOnly() &&
                     status.getEmissionPath().contains("template-extension"),
                 "Template readiness reports metadata-only template route"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(templateVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "Template emission plan is plugin-owned"))
    return result;
  if (int result =
          expect(emissionPlan.isSupported() &&
                     emissionPlan.getOriginPlugin() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateExtensionPluginName() &&
                     emissionPlan.getKernelSymbol() == kernel.getSymName() &&
                     emissionPlan.getVariantSymbol() ==
                         templateVariant.getSymName() &&
                     emissionPlan.getEmissionKind() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateMetadataEmissionKind() &&
                     emissionPlan.getLoweringPipeline() ==
                         tianchenrv::plugin::template_ext::getTemplateMetadataRouteID() &&
                     emissionPlan.getArtifactKind() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateMetadataArtifactKind() &&
                     emissionPlan.getRuntimeABIKind() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateMetadataRuntimeABIKind() &&
                     emissionPlan.getRuntimeABIName() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateExpectedIntegrationContract() &&
                     emissionPlan.getRuntimeGlueRole() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateMetadataRuntimeGlueRole() &&
                     emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                     emissionPlan.getRequiredCapabilitySymbols().front() ==
                         tianchenrv::plugin::template_ext::
                             getTemplateExtensionPreferredCapabilitySymbol(),
                 "Template emission plan records stable exportable metadata route"))
    return result;

  if (int result =
          expect(emissionPlan.getSelectedPlanMetadata().size() == 9,
                 "Template emission plan records full construction selected-plan "
                 "metadata"))
    return result;
  if (int result = expectEmissionPlanMetadata(
          emissionPlan,
          tianchenrv::plugin::template_ext::
              getTemplateConstructionProtocolMetadataName(),
          tianchenrv::plugin::template_ext::
              getTemplateConstructionProtocolVersion(),
          tianchenrv::plugin::template_ext::
              getTemplateConstructionProtocolMetadataRole()))
    return result;
  if (int result = expectEmissionPlanMetadata(
          emissionPlan,
          tianchenrv::plugin::template_ext::
              getTemplateConstructionArchetypeMetadataName(),
          tianchenrv::plugin::template_ext::getTemplateConstructionArchetype(),
          tianchenrv::plugin::template_ext::
              getTemplateConstructionArchetypeMetadataRole()))
    return result;
  if (int result = expectEmissionPlanMetadata(
          emissionPlan,
          tianchenrv::plugin::template_ext::
              getTemplateCommonInterfaceRealizationMetadataName(),
          tianchenrv::plugin::template_ext::
              getTemplateConstructionInterfaceRealization(),
          tianchenrv::plugin::template_ext::
              getTemplateCommonInterfaceRealizationMetadataRole()))
    return result;
  if (int result = expectEmissionPlanMetadata(
          emissionPlan,
          tianchenrv::plugin::template_ext::
              getTemplateEvidenceProfileMetadataName(),
          tianchenrv::plugin::template_ext::
              getTemplateConstructionEvidenceProfile(),
          tianchenrv::plugin::template_ext::
              getTemplateEvidenceProfileMetadataRole()))
    return result;

  return 0;
}

} // namespace

int main() {
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTemplateExtensionPlugin(
                            dialectPlugins),
                        "register Template plugin for dialect setup"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(dialectPlugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runConstructionManifestTest())
    return result;
  if (int result = runRegistrationAndCapabilityMetadataTest())
    return result;
  if (int result = runProposalGatingAndDeclineTest(context))
    return result;
  if (int result = runPipelineHookTest(context))
    return result;

  llvm::outs() << "Template extension plugin template smoke test passed\n";
  return 0;
}
