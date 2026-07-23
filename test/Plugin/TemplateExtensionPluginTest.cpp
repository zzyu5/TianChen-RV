#include "Weft/InitWeftDialects.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Template/IR/TemplateDialect.h"
#include "Weft/Plugin/Template/TemplateFamilyContract.h"
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

mlir::Operation *resolveExactProblem(KernelOp kernel) {
  llvm::Expected<mlir::Operation *> problem =
      weft::plugin::resolveCanonicalProblem(kernel);
  if (problem)
    return *problem;
  llvm::errs() << "FAIL: cannot resolve exact canonical problem for kernel @"
               << (kernel ? kernel.getSymName() : "<missing>") << ": "
               << llvm::toString(problem.takeError()) << "\n";
  return nullptr;
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

  const auto &route =
      weft::plugin::template_ext::getTemplateArtifactRoute();
  if (int result = expect(
          route.artifactKind == "riscv-elf-relocatable-object" &&
              route.loweringBoundaryOpName ==
                  "weft_template.compute_skeleton",
          "Template artifact projection names the typed compute body"))
    return result;

  return expectErrorContains(
      weft::plugin::registerTemplateExtensionPlugin(registry),
      {"duplicate Weft-RV extension plugin", "template-plugin"});
}

int runProposalGatingAndDeclineTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @available_template attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }

  weft.exec.kernel @missing_template attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }

  weft.exec.kernel @unavailable_template attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "unavailable",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }

  weft.exec.kernel @malformed_template attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
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
  mlir::Operation *availableProblem = resolveExactProblem(available);
  if (int result = expect(availableProblem,
                          "available Template kernel binds exact problem"))
    return result;
  VariantProposalRequest availableRequest(availableProblem, available,
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
  for (llvm::StringRef legacyConstructionAttr : {
           "weft_template.construction_protocol", "weft_template.archetype",
           "weft_template.semantic_role_graph",
           "weft_template.common_interface_realization",
           "weft_template.typed_role_realization",
           "weft_template.emitc_route_mapping",
           "weft_template.evidence_profile"}) {
    if (int result = expect(
            !findProposalAttribute(proposal, legacyConstructionAttr),
            llvm::Twine(
                "Template proposal omits legacy construction metadata '") +
                legacyConstructionAttr + "'"))
      return result;
  }

  auto expectNoProposal = [&](KernelOp kernel, llvm::StringRef context) -> int {
    TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
    mlir::Operation *problem = resolveExactProblem(kernel);
    if (!problem)
      return 1;
    VariantProposalRequest request(problem, kernel, capabilities);
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
  mlir::Operation *malformedProblem = resolveExactProblem(malformed);
  if (int result = expect(malformedProblem,
                          "malformed Template kernel binds exact problem"))
    return result;
  VariantProposalRequest malformedRequest(malformedProblem, malformed,
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
  weft.exec.kernel @template_extension_kernel attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
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
  mlir::Operation *problem = resolveExactProblem(kernel);
  if (int result =
          expect(problem, "Template pipeline kernel binds exact problem"))
    return result;
  VariantProposalRequest request(problem, kernel, capabilities);
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
  for (llvm::StringRef legacyConstructionAttr : {
           "weft_template.construction_protocol", "weft_template.archetype",
           "weft_template.semantic_role_graph",
           "weft_template.common_interface_realization",
           "weft_template.typed_role_realization",
           "weft_template.emitc_route_mapping",
           "weft_template.evidence_profile"}) {
    if (int result = expect(
            !templateVariant->hasAttr(legacyConstructionAttr),
            llvm::Twine(
                "materialized Template variant omits legacy construction metadata '") +
                legacyConstructionAttr + "'"))
      return result;
  }

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
              VariantCostRequest(templateVariant, kernel, problem,
                                 capabilities),
              estimate),
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
                                     VariantEmissionRole::DirectVariant,
                                     computeRole.getOperation()),
              status),
          "Template emission readiness consumes exact construction result"))
    return result;
  const auto &constructionRoute =
      weft::plugin::template_ext::getTemplateArtifactRoute();
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
                                     VariantEmissionRole::DirectVariant,
                                     computeRole.getOperation()),
              emissionPlan),
          "Template emission plan is plugin-owned"))
    return result;
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
                     emissionPlan.getArtifactMetadata().empty(),
                 "Template emission plan derives artifact identity directly "
                 "from the exact typed body and route contract"))
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

  if (int result = runRegistrationAndCapabilityMetadataTest())
    return result;
  if (int result = runProposalGatingAndDeclineTest(context))
    return result;
  if (int result = runPipelineHookTest(context))
    return result;

  llvm::outs() << "Template extension plugin template smoke test passed\n";
  return 0;
}
