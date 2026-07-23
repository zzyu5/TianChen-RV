#include "Weft/InitWeftDialects.h"
#include "Weft/Dialect/Toy/IR/ToyDialect.h"
#include "Weft/Plugin/BuiltinExtensionPlugins.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "Weft/Plugin/Toy/ToyFamilyContract.h"
#include "Weft/Plugin/Toy/ToyExtensionPlugin.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Transforms/VariantMaterialization.h"
#include "Weft/Transforms/VariantSelection.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>

using weft::plugin::ExtensionPluginRegistry;
using weft::plugin::ExtensionBundleRegistry;
using weft::plugin::PluginCapability;
using weft::plugin::SourceFrontDoorPassRegistration;
using weft::plugin::VariantCostEstimate;
using weft::plugin::VariantCostRequest;
using weft::plugin::VariantEmissionPlan;
using weft::plugin::VariantEmissionRequest;
using weft::plugin::VariantEmissionRole;
using weft::plugin::VariantEmissionStatus;
using weft::plugin::VariantProposal;
using weft::plugin::VariantProposalDecline;
using weft::plugin::VariantProposalRequest;
using weft::support::TargetCapabilitySet;
using weft::exec::DiagnosticOp;
using weft::exec::KernelOp;
using weft::exec::VariantOp;
using weft::toy::ComputeSkeletonOp;
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
          expectSuccess(weft::plugin::registerToyExtensionPlugin(registry),
                        "register Toy plugin"))
    return result;

  const auto *plugin = registry.lookupPlugin(
      weft::plugin::toy::getToyExtensionPluginName());
  if (int result = expect(plugin, "registered Toy plugin is visible"))
    return result;
  if (int result =
          expect(plugin->getVersion() ==
                     weft::plugin::toy::getToyExtensionPluginVersion(),
                 "Toy plugin version is stable"))
    return result;

  const PluginCapability *capability = registry.lookupCapabilityByID(
      weft::plugin::toy::getToyTemplateCapabilityID());
  if (int result =
          expect(capability &&
                     capability->getKind() ==
                         weft::plugin::toy::
                             getToyTemplateCapabilityKind(),
                 "Toy template capability metadata is registered"))
    return result;

  llvm::SmallVector<SourceFrontDoorPassRegistration, 2> sourceFrontDoorPasses;
  if (int result = expectSuccess(registry.collectSourceFrontDoorPasses(
                                     sourceFrontDoorPasses),
                                 "Toy source front-door pass collection "
                                 "succeeds"))
    return result;
  if (int result =
          expect(sourceFrontDoorPasses.size() == 1,
                 "Toy plugin contributes one source front-door pass"))
    return result;
  if (int result =
          expect(sourceFrontDoorPasses.front().getOwnerPlugin() ==
                     weft::plugin::toy::getToyExtensionPluginName(),
                 "Toy source front-door pass is owned by Toy plugin"))
    return result;
  if (int result =
          expect(sourceFrontDoorPasses.front().getArgument() ==
                     "weft-toy-materialize-template-source-front-door",
                 "Toy source front-door pass keeps the public pass argument"))
    return result;
  if (int result = expect(static_cast<bool>(
                              sourceFrontDoorPasses.front().getFactory()),
                          "Toy source front-door pass factory is present"))
    return result;

  // Historical manifest/role-graph replay tests were retired with the second
  // construction authority.  Artifact constants are checked directly below.

  const auto &route = weft::plugin::toy::getToyArtifactRoute();
  if (int result = expect(
          route.artifactKind == "riscv-elf-relocatable-object" &&
              route.loweringBoundaryOpName == "weft_toy.compute_skeleton" &&
              !weft::plugin::toy::getToyRuntimeABIParameters().empty(),
          "Toy artifact projection is bound to its typed compute body"))
    return result;
  return expectErrorContains(
      weft::plugin::registerToyExtensionPlugin(registry),
      {"duplicate Weft-RV extension plugin", "toy-plugin"});
}

int runBuiltinSourceFrontDoorCollectionTest() {
  ExtensionBundleRegistry bundles;
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          weft::plugin::registerBuiltinExtensionBundlePlugins(
              bundles, registry),
          "register built-in extension bundle frontdoor for source front-door "
          "collection"))
    return result;

  llvm::SmallVector<SourceFrontDoorPassRegistration, 8> sourceFrontDoorPasses;
  if (int result = expectSuccess(
          registry.collectSourceFrontDoorPasses(sourceFrontDoorPasses),
          "collect built-in source front-door pass registrations"))
    return result;

  int rvvIndex = -1;
  int toyIndex = -1;
  int tensorextLiteIndex = -1;
  for (auto [index, pass] : llvm::enumerate(sourceFrontDoorPasses)) {
    if (pass.getArgument().contains("source-seed"))
      return fail("built-in source front-door collection resurrected "
                  "source-seed public API");
    if (pass.getOwnerPlugin() == "rvv-plugin" &&
        pass.getArgument() ==
            "weft-rvv-materialize-vector-binary-source-front-door")
      rvvIndex = static_cast<int>(index);
    if (pass.getOwnerPlugin() ==
            weft::plugin::toy::getToyExtensionPluginName() &&
        pass.getArgument() ==
            "weft-toy-materialize-template-source-front-door")
      toyIndex = static_cast<int>(index);
    if (pass.getOwnerPlugin() ==
            weft::plugin::tensorext_lite::
                getTensorExtLiteExtensionPluginName() &&
        pass.getArgument() ==
            "weft-tensorext-lite-materialize-fragment-mma-source-front-door")
      tensorextLiteIndex = static_cast<int>(index);
  }

  if (int result =
          expect(rvvIndex >= 0, "built-in registry exposes RVV source front "
                                "door through the common interface"))
    return result;
  if (int result =
          expect(toyIndex >= 0, "built-in registry exposes Toy source front "
                                "door through the common interface"))
    return result;
  if (int result = expect(tensorextLiteIndex >= 0,
                          "built-in registry exposes TensorExtLite source "
                          "front door through the common interface"))
    return result;
  return expect(rvvIndex < toyIndex && toyIndex < tensorextLiteIndex,
                "built-in source front-door order follows registry order");
}

int runProposalGatingAndDeclineTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @available_toy attributes {construction_domain = "riscv-execution"} {
    weft.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
  }

  weft.exec.kernel @missing_toy attributes {construction_domain = "riscv-execution"} {
  }

  weft.exec.kernel @unavailable_toy attributes {construction_domain = "riscv-execution"} {
    weft.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "unavailable",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
  }

  weft.exec.kernel @malformed_toy attributes {construction_domain = "riscv-execution"} {
    weft.exec.capability @toy_template {
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
          expectSuccess(weft::plugin::registerToyExtensionPlugin(registry),
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
                         weft::plugin::toy::
                             getToyTemplateFirstSliceVariantName() &&
                     proposal.getOriginPlugin() ==
                         weft::plugin::toy::
                             getToyExtensionPluginName() &&
                     proposal.getPolicy() ==
                         weft::plugin::toy::getToyTemplatePolicy(),
                 "Toy proposal preserves stable generic metadata"))
    return result;
  if (int result =
          expect(proposal.getRequiredCapabilityIDs().size() == 1 &&
                     proposal.getRequiredCapabilityIDs().front() ==
                         weft::plugin::toy::getToyTemplateCapabilityID(),
                 "Toy proposal requires toy.template capability id"))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, weft::plugin::toy::getToyTemplateABIAttrName(),
          weft::plugin::toy::getToyExpectedTemplateABI()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, weft::plugin::toy::getToyHandoffKindAttrName(),
          weft::plugin::toy::getToyExpectedHandoffKind()))
    return result;
  for (llvm::StringRef legacyConstructionAttr : {
           "weft_toy.construction_protocol", "weft_toy.archetype",
           "weft_toy.semantic_role_graph",
           "weft_toy.common_interface_realization",
           "weft_toy.typed_role_realization",
           "weft_toy.emitc_route_mapping", "weft_toy.evidence_profile"}) {
    if (int result = expect(
            !findProposalAttribute(proposal, legacyConstructionAttr),
            llvm::Twine("Toy proposal omits legacy construction metadata '") +
                legacyConstructionAttr + "'"))
      return result;
  }

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
                        weft::plugin::toy::getToyExtensionPluginName() &&
                    declines.front().getReason().contains("template_abi"),
                "malformed Toy capability records plugin-local decline");
}

int runPipelineHookTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @toy_template_kernel attributes {construction_domain = "riscv-execution"} {
    weft.exec.capability @toy_template {
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
          expectSuccess(weft::plugin::registerToyExtensionPlugin(registry),
                        "register Toy plugin for pipeline hook"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(kernel.getOperation(), kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          weft::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize Toy proposal"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "Toy capability materializes one variant"))
    return result;

  VariantOp toyVariant = findVariant(
      kernel,
      weft::plugin::toy::getToyTemplateFirstSliceVariantName());
  if (int result = expect(toyVariant, "Toy variant is materialized"))
    return result;
  if (int result =
          expect(toyVariant->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     weft::plugin::toy::getToyExtensionPluginName(),
                 "Toy variant has Toy origin"))
    return result;
  if (int result =
          expect(toyVariant
                         ->getAttrOfType<mlir::StringAttr>(
                             weft::plugin::toy::
                                 getToyTemplateABIAttrName())
                         .getValue() ==
                     weft::plugin::toy::getToyExpectedTemplateABI(),
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
                         weft::plugin::toy::
                             getToyExtensionPluginName() &&
                     estimate.getVariantSymbol() == toyVariant.getSymName(),
                 "Toy cost metadata is plugin-owned"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      weft::transforms::planKernelVariantSelection(kernel, capabilities,
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
          weft::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize Toy selected marker"))
    return result;
  if (int result = expect(marker, "Toy selected marker was created"))
    return result;

  if (int result = expectSuccess(
          weft::plugin::materializeSelectedLoweringBoundaries(
              kernel, capabilities, registry),
          "materialize Toy selected boundary"))
    return result;

  ComputeSkeletonOp computeRole =
      findToyComputeRole(kernel, toyVariant.getSymName());
  if (int result =
          expect(computeRole,
                 "Toy selected path materializes a compute role boundary"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "Toy boundary module verifies"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectSuccess(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(toyVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant,
                                     computeRole.getOperation()),
              status),
          "Toy emission readiness consumes exact construction result"))
    return result;
  const auto &routeSpec =
      weft::plugin::toy::getToyArtifactRoute();
  if (int result =
          expect(status.isSupported() &&
                     status.getEmissionPath() == routeSpec.routeID,
                 "Toy emission readiness reports supported EmitC route"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(toyVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant,
                                     computeRole.getOperation()),
              emissionPlan),
          "Toy emission plan is plugin-owned"))
    return result;
  if (int result =
          expect(emissionPlan.isSupported() &&
                     emissionPlan.getOriginPlugin() ==
                         weft::plugin::toy::
                             getToyExtensionPluginName() &&
                     emissionPlan.getKernelSymbol() == kernel.getSymName() &&
                     emissionPlan.getVariantSymbol() ==
                         toyVariant.getSymName() &&
                     emissionPlan.getLoweringPipeline() ==
                         routeSpec.routeID &&
                     emissionPlan.getEmissionKind() ==
                         routeSpec.emissionKind &&
                     emissionPlan.getArtifactKind() ==
                         routeSpec.artifactKind &&
                     emissionPlan.getRuntimeABI() ==
                         routeSpec.runtimeABI &&
                     emissionPlan.getRuntimeABIKind() ==
                         routeSpec.runtimeABIKind &&
                     emissionPlan.getRuntimeABIName() ==
                         routeSpec.runtimeABIName &&
                     emissionPlan.getRuntimeGlueRole() ==
                         routeSpec.runtimeGlueRole &&
                     emissionPlan.getLoweringBoundaryOpName() ==
                         routeSpec.loweringBoundaryOpName &&
                     weft::support::runtimeABIParametersEqual(
                         emissionPlan.getRuntimeABIParameters(),
                         weft::plugin::toy::
                             getToyRuntimeABIParameters()) &&
                     emissionPlan.getArtifactMetadata().empty() &&
                     emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                     emissionPlan.getRequiredCapabilitySymbols().front() ==
                         weft::plugin::toy::
                             getToyTemplatePreferredCapabilitySymbol(),
                 "Toy emission plan derives artifact identity directly from "
                 "the exact typed body and route contract"))
    return result;

  return 0;
}

} // namespace

int main() {
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(weft::plugin::registerToyExtensionPlugin(
                            dialectPlugins),
                        "register Toy plugin for dialect setup"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  weft::registerAllDialects(dialectRegistry);
  weft::registerPluginDialects(dialectPlugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runRegistrationAndCapabilityMetadataTest())
    return result;
  if (int result = runBuiltinSourceFrontDoorCollectionTest())
    return result;
  if (int result = runProposalGatingAndDeclineTest(context))
    return result;
  if (int result = runPipelineHookTest(context))
    return result;

  llvm::outs() << "Toy extension plugin template smoke test passed\n";
  return 0;
}
