#include "Weft/InitWeftDialects.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Template/IR/TemplateDialect.h"
#include "Weft/Plugin/Template/TemplateConstructionProtocol.h"
#include "Weft/Plugin/Template/TemplateExtensionPlugin.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Transforms/VariantMaterialization.h"
#include "Weft/Transforms/VariantSelection.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>
#include <utility>

using weft::plugin::ExtensionPluginRegistry;
using weft::plugin::PluginCapability;
using weft::plugin::VariantCostEstimate;
using weft::plugin::VariantCostRequest;
using weft::plugin::VariantEmissionPlan;
using weft::plugin::VariantEmissionRequest;
using weft::plugin::VariantEmissionRole;
using weft::plugin::VariantEmissionStatus;
using weft::plugin::VariantProposal;
using weft::plugin::VariantProposalDecline;
using weft::plugin::VariantProposalRequest;
using weft::conversion::emitc::WEFTEmitCLowerableOpInterface;
using weft::support::TargetCapabilitySet;
using weft::exec::DiagnosticOp;
using weft::exec::KernelOp;
using weft::exec::VariantOp;
using weft::template_ext::ComputeSkeletonOp;
using weft::template_ext::LoweringBoundaryOp;
using weft::transforms::VariantSelectionKind;
using weft::transforms::VariantSelectionPlan;

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

ComputeSkeletonOp findTemplateComputeRoleOp(
    KernelOp kernel, llvm::StringRef selectedVariantSymbol) {
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

int runConstructionManifestTest() {
  const auto &manifest =
      weft::plugin::template_ext::getTemplateConstructionManifest();
  if (int result = expectSuccess(
          weft::plugin::template_ext::
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
                  weft::plugin::template_ext::
                      getTemplateExtensionPluginName() &&
              manifest.family.capabilityID ==
                  weft::plugin::template_ext::
                      getTemplateExtensionCapabilityID() &&
              manifest.family.concreteNamespace == "weft_template",
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
            role.commonInterfaces.contains("WEFTExtensionOpInterface") &&
                role.commonInterfaces.contains("WEFTEmitCLowerableInterface"),
            llvm::Twine("Template role realizes common interfaces: ") +
                role.role))
      return result;
  }
  if (int result = expect(
          weft::plugin::template_ext::
              getTemplateConstructionInterfaceRealization()
                  .contains("compute=WEFTExtensionOpInterface") &&
              weft::plugin::template_ext::
                  getTemplateConstructionInterfaceRealization()
                      .contains("WEFTComputeOpInterface"),
          "Template manifest exposes common-interface realization mapping"))
    return result;
  const auto &realization =
      weft::plugin::template_ext::getTemplateTypedRoleGraphRealization();
  if (int result = expectSuccess(
          weft::plugin::template_ext::
              verifyTemplateTypedRoleGraphRealization(manifest, realization),
          "Template typed role graph realization verifies"))
    return result;
  if (int result =
          expect(realization.roles.size() == 4 &&
                     realization.roles[2].typedRoleID ==
                         "template.role.compute.compute_skeleton" &&
                     realization.roles[2].roleSpecificInterface ==
                         "WEFTComputeOpInterface" &&
                     realization.roles[2].emitCLowerableInterface ==
                         "WEFTEmitCLowerableInterface" &&
                     weft::plugin::template_ext::
                         getTemplateTypedRoleRealizationSummary()
                             .contains("compute:template.role.compute"),
                 "Template typed role realization exposes concrete compute role "
                 "surface"))
    return result;
  if (int result = expect(
          manifest.emitcRoute.routeID ==
                  "template-extension-compute-skeleton-emitc-route" &&
              manifest.emitcRoute.artifactKind ==
                  "riscv-elf-relocatable-object" &&
              manifest.emitcRoute.emissionKind ==
                  "materialized-emitc-cpp-template-compute-skeleton-module",
          "Template manifest exposes materialized EmitC route fields"))
    return result;
  const auto &route =
      weft::plugin::template_ext::getTemplateEmitCConstructionRoute();
  if (int result = expectSuccess(
          weft::plugin::template_ext::
              verifyTemplateEmitCConstructionRouteMapping(
                  route.routeID, route.emissionKind, route.artifactKind,
                  route.loweringBoundaryOpName, route.runtimeABI,
                  route.runtimeABIKind, route.runtimeABIName,
                  route.runtimeGlueRole),
          "Template EmitC construction route mapping verifies"))
    return result;
  if (int result = expectSuccess(
          weft::plugin::template_ext::
              verifyTemplateTargetArtifactBundleMapping(
                  route.headerRouteID, route.headerArtifactKind,
                  route.bundleComponentGroup, route.objectHandoffKind,
                  route.emitCToCppTranslateRouteID),
          "Template object/header bundle mapping verifies"))
    return result;
  return expect(
      manifest.evidenceProfile.contains("parse_verify") &&
          manifest.evidenceProfile.contains("interface") &&
          manifest.evidenceProfile.contains("emitc_route_mapping") &&
          manifest.evidenceProfile.contains("materialized_emitc_module") &&
          manifest.evidenceProfile.contains("generated_cpp_compile"),
      "Template manifest records focused evidence profile");
}

int runTypedRoleGraphValidationTest() {
  namespace template_ext = weft::plugin::template_ext;
  const auto &manifest = template_ext::getTemplateConstructionManifest();
  const auto &realization =
      template_ext::getTemplateTypedRoleGraphRealization();

  if (int result = expectSuccess(
          template_ext::verifyTemplateTypedRoleGraphRealization(manifest,
                                                                realization),
          "Template typed role graph validates without generated output"))
    return result;

  {
    template_ext::TemplateConstructionManifest bad = manifest;
    llvm::SmallVector<template_ext::TemplateConstructionSemanticRole, 4> roles(
        manifest.semanticRoles.begin(), manifest.semanticRoles.end());
    std::swap(roles[1], roles[2]);
    roles[1].order = 1;
    roles[2].order = 2;
    bad.semanticRoles = roles;

    if (int result = expectErrorContains(
            template_ext::verifyTemplateConstructionManifest(bad),
            {"semantic role graph entry", "Template role order"}))
      return result;
  }

  {
    template_ext::TemplateTypedRoleGraphRealization bad = realization;
    llvm::SmallVector<template_ext::TemplateTypedRoleInterfaceRealization, 4>
        roles(realization.roles.begin(), realization.roles.end());
    roles.pop_back();
    bad.roles = roles;

    if (int result = expectErrorContains(
            template_ext::verifyTemplateTypedRoleGraphRealization(manifest,
                                                                  bad),
            {"typed role realization", "exactly one role object"}))
      return result;
  }

  {
    template_ext::TemplateTypedRoleGraphRealization bad = realization;
    llvm::SmallVector<template_ext::TemplateTypedRoleInterfaceRealization, 4>
        roles(realization.roles.begin(), realization.roles.end());
    std::swap(roles[1], roles[2]);
    roles[1].order = 1;
    roles[2].order = 2;
    bad.roles = roles;

    if (int result = expectErrorContains(
            template_ext::verifyTemplateTypedRoleGraphRealization(manifest,
                                                                  bad),
            {"typed role realization entry", "not ordered"}))
      return result;
  }

  {
    template_ext::TemplateTypedRoleGraphRealization bad = realization;
    llvm::SmallVector<template_ext::TemplateTypedRoleInterfaceRealization, 4>
        roles(realization.roles.begin(), realization.roles.end());
    roles[2].operationName = "weft_template.stale_compute_skeleton";
    bad.roles = roles;

    if (int result = expectErrorContains(
            template_ext::verifyTemplateTypedRoleGraphRealization(manifest,
                                                                  bad),
            {"typed role realization entry", "operation",
             "does not match manifest operation"}))
      return result;
  }

  {
    template_ext::TemplateTypedRoleGraphRealization bad = realization;
    llvm::SmallVector<template_ext::TemplateTypedRoleInterfaceRealization, 4>
        roles(realization.roles.begin(), realization.roles.end());
    roles[2].roleSpecificInterface = "WEFTMemoryOpInterface";
    bad.roles = roles;

    if (int result = expectErrorContains(
            template_ext::verifyTemplateTypedRoleGraphRealization(manifest,
                                                                  bad),
            {"typed role realization entry",
             "role-specific common interface", "WEFTComputeOpInterface"}))
      return result;
  }

  {
    template_ext::TemplateConstructionManifest bad = manifest;
    llvm::SmallVector<template_ext::TemplateConstructionSemanticRole, 4> roles(
        manifest.semanticRoles.begin(), manifest.semanticRoles.end());
    roles[2].commonInterfaces =
        "WEFTExtensionOpInterface+WEFTEmitCLowerableInterface";
    bad.semanticRoles = roles;

    if (int result = expectErrorContains(
            template_ext::verifyTemplateConstructionManifest(bad),
            {"semantic role 'compute'", "WEFTComputeOpInterface"}))
      return result;
  }

  return 0;
}

int runTemplateComputeRoleOpInterfaceTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @template_compute_role_interface attributes {} {
    weft.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
    weft.exec.variant @template_zero_core_first_slice attributes {
      origin = "template-plugin",
      requires = [@template_extension]
    } {
    }
    weft_template.compute_skeleton {
      origin = "template-plugin",
      required_capabilities = [@template_extension],
      role = "direct variant",
      role_order = 2 : i64,
      role_specific_interface = "WEFTComputeOpInterface",
      selected_variant = @template_zero_core_first_slice,
      source_kernel = "template_compute_role_interface",
      source_role = "compute",
      status = "role-op-boundary",
      template_reason = "Template ODS role-op boundary only",
      typed_role = "template.role.compute.compute_skeleton"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse Template compute role-op module");
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "Template compute role-op module verifies"))
    return result;

  KernelOp kernel = findKernel(*module, "template_compute_role_interface");
  ComputeSkeletonOp compute = findTemplateComputeRoleOp(
      kernel,
      weft::plugin::template_ext::
          getTemplateExtensionFirstSliceVariantName());
  if (int result =
          expect(compute, "Template ODS compute role op is materialized"))
    return result;

  auto lowerable =
      llvm::dyn_cast<WEFTEmitCLowerableOpInterface>(compute.getOperation());
  if (int result =
          expect(lowerable,
                 "Template compute role op implements "
                 "WEFTEmitCLowerableOpInterface"))
    return result;
  if (int result =
          expect(lowerable.getWEFTEmitCLowerableSourceOpName() ==
                         ComputeSkeletonOp::getOperationName() &&
                     lowerable.getWEFTEmitCLowerableSourceRole() == "compute",
                 "Template compute role op exposes interface source op and "
                 "role"))
    return result;

  const auto &manifest =
      weft::plugin::template_ext::getTemplateConstructionManifest();
  const auto &realization =
      weft::plugin::template_ext::getTemplateTypedRoleGraphRealization();
  if (int result = expectSuccess(
          weft::plugin::template_ext::
              verifyTemplateComputeRoleOpInterface(manifest, realization,
                                                   compute.getOperation()),
          "Template construction validation accepts ODS compute role op"))
    return result;

  compute->setAttr("source_role", mlir::StringAttr::get(&context, "load"));
  if (int result = expectErrorContains(
          weft::plugin::template_ext::
              verifyTemplateComputeRoleOpInterface(manifest, realization,
                                                  compute.getOperation()),
          {"WEFTEmitCLowerableOpInterface source role", "compute"}))
    return result;
  compute->setAttr("source_role", mlir::StringAttr::get(&context, "compute"));

  compute->setAttr(
      "typed_role",
      mlir::StringAttr::get(&context, "template.role.compute.stale"));
  if (int result = expectErrorContains(
          weft::plugin::template_ext::
              verifyTemplateComputeRoleOpInterface(manifest, realization,
                                                   compute.getOperation()),
          {"compute role op typed_role", "typed compute role realization"}))
    return result;
  compute->setAttr(
      "typed_role",
      mlir::StringAttr::get(&context,
                            "template.role.compute.compute_skeleton"));

  if (int result = expectErrorContains(
          weft::plugin::template_ext::
              verifyTemplateComputeRoleOpInterface(
                  manifest, realization, kernel.getOperation()),
          {"must implement WEFTEmitCLowerableOpInterface"}))
    return result;

  return 0;
}

int runRegistrationAndCapabilityMetadataTest() {
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(weft::plugin::registerTemplateExtensionPlugin(registry),
                        "register Template plugin"))
    return result;

  const auto *plugin = registry.lookupPlugin(
      weft::plugin::template_ext::getTemplateExtensionPluginName());
  if (int result = expect(plugin, "registered Template plugin is visible"))
    return result;
  if (int result =
          expect(plugin->getVersion() ==
                     weft::plugin::template_ext::getTemplateExtensionPluginVersion(),
                 "Template plugin version is stable"))
    return result;

  const PluginCapability *capability = registry.lookupCapabilityByID(
      weft::plugin::template_ext::getTemplateExtensionCapabilityID());
  if (int result =
          expect(capability &&
                     capability->getKind() ==
                         weft::plugin::template_ext::
                             getTemplateExtensionCapabilityKind(),
                 "Template extension capability metadata is registered"))
    return result;

  const auto &manifest =
      weft::plugin::template_ext::getTemplateConstructionManifest();
  if (int result =
          expect(manifest.family.pluginName == plugin->getName() &&
                     manifest.family.capabilityID == capability->getID() &&
                     manifest.family.capabilityKind == capability->getKind(),
                 "Template construction manifest agrees with registered plugin "
                 "capability"))
    return result;

  return expectErrorContains(
      weft::plugin::registerTemplateExtensionPlugin(registry),
      {"duplicate Weft-RV extension plugin", "template-plugin"});
}

int runProposalGatingAndDeclineTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @available_template attributes {} {
    weft.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }

  weft.exec.kernel @missing_template attributes {} {
  }

  weft.exec.kernel @unavailable_template attributes {} {
    weft.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "unavailable",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }

  weft.exec.kernel @malformed_template attributes {} {
    weft.exec.capability @template_extension {
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
          expectSuccess(weft::plugin::registerTemplateExtensionPlugin(registry),
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
                         weft::plugin::template_ext::
                             getTemplateExtensionFirstSliceVariantName() &&
                     proposal.getOriginPlugin() ==
                         weft::plugin::template_ext::
                             getTemplateExtensionPluginName() &&
                     proposal.getPolicy() ==
                         weft::plugin::template_ext::getTemplateExtensionPolicy(),
                 "Template proposal preserves stable generic metadata"))
    return result;
  if (int result =
          expect(proposal.getRequiredCapabilityIDs().size() == 1 &&
                     proposal.getRequiredCapabilityIDs().front() ==
                         weft::plugin::template_ext::getTemplateExtensionCapabilityID(),
                 "Template proposal requires template.extension capability id"))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, weft::plugin::template_ext::getTemplateIntegrationContractAttrName(),
          weft::plugin::template_ext::getTemplateExpectedIntegrationContract()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, weft::plugin::template_ext::getTemplateHandoffKindAttrName(),
          weft::plugin::template_ext::getTemplateExpectedHandoffKind()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "weft_template.construction_protocol",
          weft::plugin::template_ext::
              getTemplateConstructionProtocolVersion()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "weft_template.archetype",
          weft::plugin::template_ext::
              getTemplateConstructionArchetype()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "weft_template.semantic_role_graph",
          weft::plugin::template_ext::
              getTemplateConstructionSemanticRoleGraph()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "weft_template.common_interface_realization",
          weft::plugin::template_ext::
              getTemplateConstructionInterfaceRealization()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "weft_template.typed_role_realization",
          weft::plugin::template_ext::
              getTemplateTypedRoleRealizationSummary()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "weft_template.emitc_route_mapping",
          weft::plugin::template_ext::getTemplateConstructionManifest()
              .emitcRoute.routeID))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "weft_template.evidence_profile",
          weft::plugin::template_ext::
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
                        weft::plugin::template_ext::getTemplateExtensionPluginName() &&
                    declines.front().getReason().contains("integration_contract"),
                "malformed Template capability records plugin-local decline");
}

int runPipelineHookTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @template_extension_kernel attributes {} {
    weft.exec.capability @template_extension {
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
          expectSuccess(weft::plugin::registerTemplateExtensionPlugin(registry),
                        "register Template plugin for pipeline hook"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(kernel.getOperation(), kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          weft::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize Template proposal"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "Template capability materializes one variant"))
    return result;

  VariantOp templateVariant = findVariant(
      kernel,
      weft::plugin::template_ext::getTemplateExtensionFirstSliceVariantName());
  if (int result = expect(templateVariant, "Template variant is materialized"))
    return result;
  if (int result =
          expect(templateVariant->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     weft::plugin::template_ext::getTemplateExtensionPluginName(),
                 "Template variant has Template origin"))
    return result;
  if (int result =
          expect(templateVariant
                         ->getAttrOfType<mlir::StringAttr>(
                             weft::plugin::template_ext::
                                 getTemplateIntegrationContractAttrName())
                         .getValue() ==
                     weft::plugin::template_ext::getTemplateExpectedIntegrationContract(),
                 "Template variant carries integration contract metadata"))
    return result;
  if (int result =
          expect(templateVariant
                         ->getAttrOfType<mlir::StringAttr>(
                             "weft_template.construction_protocol")
                         .getValue() ==
                     weft::plugin::template_ext::
                         getTemplateConstructionProtocolVersion() &&
                     templateVariant
                             ->getAttrOfType<mlir::StringAttr>(
                                 "weft_template.semantic_role_graph")
                             .getValue() ==
                         weft::plugin::template_ext::
                             getTemplateConstructionSemanticRoleGraph() &&
                     templateVariant
                             ->getAttrOfType<mlir::StringAttr>(
                                 "weft_template.common_interface_realization")
                             .getValue()
                             .contains("WEFTComputeOpInterface") &&
                     templateVariant
                             ->getAttrOfType<mlir::StringAttr>(
                                 "weft_template.typed_role_realization")
                             .getValue()
                             .contains("compute:template.role.compute"),
                 "Template variant carries code-consumed construction manifest "
                 "and typed role metadata"))
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
                         weft::plugin::template_ext::
                             getTemplateExtensionPluginName() &&
                     estimate.getVariantSymbol() == templateVariant.getSymName(),
                 "Template cost metadata is plugin-owned"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      weft::transforms::planKernelVariantSelection(kernel, capabilities,
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
          weft::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize Template selected marker"))
    return result;
  if (int result = expect(marker, "Template selected marker was created"))
    return result;

  if (int result = expectSuccess(
          weft::plugin::materializeSelectedLoweringBoundaries(
              kernel, capabilities, registry),
          "materialize Template selected boundary"))
    return result;

  LoweringBoundaryOp oldBoundary =
      findTemplateBoundary(kernel, templateVariant.getSymName());
  if (int result =
          expect(!oldBoundary,
                 "Template selected path does not materialize the old "
                 "metadata-only lowering_boundary"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "Template selected role-boundary module verifies"))
    return result;

  ComputeSkeletonOp computeRole =
      findTemplateComputeRoleOp(kernel, templateVariant.getSymName());
  if (int result =
          expect(computeRole,
                 "Template selected path materializes a compute role op"))
    return result;
  if (int result = expectSuccess(
          registry.validateSelectedLoweringBoundary(
              weft::plugin::VariantLoweringBoundaryValidationRequest(
                  templateVariant, kernel, capabilities,
                  VariantEmissionRole::DirectVariant,
                  computeRole.getOperation())),
          "Template selected compute role validates as lowering boundary"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectSuccess(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(templateVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          "Template emission readiness is checked through route builder"))
    return result;
  const auto &constructionRoute =
      weft::plugin::template_ext::getTemplateEmitCConstructionRoute();
  if (int result =
          expect(status.isSupported() &&
                     status.getOriginPlugin() ==
                         weft::plugin::template_ext::
                             getTemplateExtensionPluginName() &&
                     status.getVariantSymbol() == templateVariant.getSymName() &&
                     status.getEmissionPath() == constructionRoute.routeID,
                 "Template emission readiness is supported only after "
                 "selected role route exists"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(templateVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "Template emission plan is plugin-owned"))
    return result;
  bool sawRouteMetadata = false;
  bool sawSourceOpMetadata = false;
  bool sawSourceInterfaceMetadata = false;
  for (const auto &metadata : emissionPlan.getArtifactMetadata()) {
    if (metadata.key ==
            weft::plugin::template_ext::
                getTemplateEmitCRouteMappingMetadataName() &&
        metadata.value == constructionRoute.routeID)
      sawRouteMetadata = true;
    if (metadata.key ==
            weft::plugin::template_ext::
                getTemplateSourceOpMetadataName() &&
        metadata.value == constructionRoute.loweringBoundaryOpName)
      sawSourceOpMetadata = true;
    if (metadata.key ==
            weft::plugin::template_ext::
                getTemplateSourceOpInterfaceMetadataName() &&
        metadata.value == "WEFTEmitCLowerableOpInterface")
      sawSourceInterfaceMetadata = true;
  }
  if (int result =
          expect(emissionPlan.isSupported() &&
                     emissionPlan.getOriginPlugin() ==
                         weft::plugin::template_ext::
                             getTemplateExtensionPluginName() &&
                     emissionPlan.getKernelSymbol() == kernel.getSymName() &&
                     emissionPlan.getVariantSymbol() ==
                         templateVariant.getSymName() &&
                     emissionPlan.getLoweringPipeline() ==
                         constructionRoute.routeID &&
                     emissionPlan.getArtifactKind() ==
                         constructionRoute.artifactKind &&
                     emissionPlan.getRuntimeABI() ==
                         constructionRoute.runtimeABI &&
                     emissionPlan.getRuntimeABIKind() ==
                         constructionRoute.runtimeABIKind &&
                     emissionPlan.getRuntimeABIName() ==
                         constructionRoute.runtimeABIName &&
                     emissionPlan.getLoweringBoundaryOpName() ==
                         constructionRoute.loweringBoundaryOpName &&
                     emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                     emissionPlan.getRequiredCapabilitySymbols().front() ==
                         weft::plugin::template_ext::
                             getTemplateExtensionPreferredCapabilitySymbol() &&
                     sawRouteMetadata && sawSourceOpMetadata &&
                     sawSourceInterfaceMetadata,
                 "Template emission plan carries materialized EmitC route "
                 "metadata"))
    return result;

  return 0;
}

} // namespace

int main() {
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(weft::plugin::registerTemplateExtensionPlugin(
                            dialectPlugins),
                        "register Template plugin for dialect setup"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  weft::registerAllDialects(dialectRegistry);
  weft::registerPluginDialects(dialectPlugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runConstructionManifestTest())
    return result;
  if (int result = runTypedRoleGraphValidationTest())
    return result;
  if (int result = runTemplateComputeRoleOpInterfaceTest(context))
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
