#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"
#include "TianChenRV/Transforms/VariantSelection.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::SourceFrontDoorPassRegistration;
using tianchenrv::plugin::VariantCostEstimate;
using tianchenrv::plugin::VariantCostRequest;
using tianchenrv::plugin::VariantEmitCLowerableRequest;
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
using tianchenrv::tcrv::tensorext_lite::ConfigSkeletonOp;
using tianchenrv::tcrv::tensorext_lite::LoadFragSkeletonOp;
using tianchenrv::tcrv::tensorext_lite::TileMmaSkeletonOp;
using tianchenrv::tcrv::tensorext_lite::LoweringBoundaryOp;
using tianchenrv::tcrv::tensorext_lite::StoreFragSkeletonOp;
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

LoweringBoundaryOp findTensorExtLiteBoundary(KernelOp kernel,
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

TileMmaSkeletonOp findTensorExtLiteNestedComputeRole(
    VariantOp variant, llvm::StringRef selectedVariantSymbol) {
  TileMmaSkeletonOp result;
  if (!variant || variant.getBody().empty())
    return result;

  for (mlir::Operation &op : variant.getBody().front()) {
    auto compute = llvm::dyn_cast<TileMmaSkeletonOp>(op);
    if (!compute)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    if (selectedVariant && selectedVariant.getValue() == selectedVariantSymbol)
      result = compute;
  }
  return result;
}

void materializeTensorExtLiteRoleSequence(mlir::OpBuilder &builder,
                                          KernelOp kernel,
                                          VariantOp variant,
                                          bool reorderLoadAndTile = false,
                                          unsigned roleCount = 4) {
  mlir::Block &body = variant.getBody().front();
  builder.setInsertionPointToEnd(&body);
  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  llvm::ArrayRef<tianchenrv::plugin::tensorext_lite::
                     TensorExtLiteFragmentMmaRoleStep>
      roleSteps = tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentMmaRoleSteps();
  const unsigned orderedIndices[] = {0, 1, 2, 3};
  const unsigned reorderedIndices[] = {0, 2, 1, 3};
  llvm::ArrayRef<unsigned> indices =
      reorderLoadAndTile ? llvm::ArrayRef<unsigned>(reorderedIndices)
                         : llvm::ArrayRef<unsigned>(orderedIndices);
  for (unsigned specIndex : indices.take_front(roleCount)) {
    const auto &step = roleSteps[specIndex];
    mlir::OperationState state(variant.getLoc(), step.operationName);
    state.addAttribute("source_kernel",
                       builder.getStringAttr(kernel.getSymName()));
    state.addAttribute("selected_variant",
                       mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                    variant.getSymName()));
    state.addAttribute("origin",
                       builder.getStringAttr(
                           tianchenrv::plugin::tensorext_lite::
                               getTensorExtLiteExtensionPluginName()));
    state.addAttribute(
        "role",
        builder.getStringAttr(tianchenrv::plugin::stringifyVariantEmissionRole(
            VariantEmissionRole::DirectVariant)));
    state.addAttribute("status", builder.getStringAttr("role-op-boundary"));
    state.addAttribute("required_capabilities", variantRequires);
    state.addAttribute("typed_role", builder.getStringAttr(step.typedRoleID));
    state.addAttribute("role_order",
                       builder.getI64IntegerAttr(step.order));
    state.addAttribute("source_role", builder.getStringAttr(step.sourceRole));
    state.addAttribute("role_specific_interface",
                       builder.getStringAttr(step.roleSpecificInterface));
    builder.create(state);
  }
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
          expectSuccess(tianchenrv::plugin::registerTensorExtLiteExtensionPlugin(registry),
                        "register TensorExtLite plugin"))
    return result;

  const auto *plugin = registry.lookupPlugin(
      tianchenrv::plugin::tensorext_lite::getTensorExtLiteExtensionPluginName());
  if (int result = expect(plugin, "registered TensorExtLite plugin is visible"))
    return result;
  if (int result =
          expect(plugin->getVersion() ==
                     tianchenrv::plugin::tensorext_lite::getTensorExtLiteExtensionPluginVersion(),
                 "TensorExtLite plugin version is stable"))
    return result;

  const PluginCapability *capability = registry.lookupCapabilityByID(
      tianchenrv::plugin::tensorext_lite::getTensorExtLiteFragmentCapabilityID());
  if (int result =
          expect(capability &&
                     capability->getKind() ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteFragmentCapabilityKind(),
                 "TensorExtLite fragment capability metadata is registered"))
    return result;

  llvm::SmallVector<SourceFrontDoorPassRegistration, 2> sourceFrontDoorPasses;
  if (int result = expectSuccess(registry.collectSourceFrontDoorPasses(
                                     sourceFrontDoorPasses),
                                 "TensorExtLite source front-door pass "
                                 "collection succeeds"))
    return result;
  if (int result =
          expect(sourceFrontDoorPasses.size() == 1,
                 "TensorExtLite plugin contributes one source front-door pass"))
    return result;
  if (int result =
          expect(sourceFrontDoorPasses.front().getOwnerPlugin() ==
                     tianchenrv::plugin::tensorext_lite::
                         getTensorExtLiteExtensionPluginName(),
                 "TensorExtLite source front-door pass is owned by "
                 "TensorExtLite plugin"))
    return result;
  if (int result =
          expect(sourceFrontDoorPasses.front().getArgument() ==
                     "tcrv-tensorext-lite-materialize-fragment-mma-source-front-door",
                 "TensorExtLite source front-door pass keeps the public pass "
                 "argument"))
    return result;
  if (int result = expect(static_cast<bool>(
                              sourceFrontDoorPasses.front().getFactory()),
                          "TensorExtLite source front-door pass factory is "
                          "present"))
    return result;

  tianchenrv::plugin::ExtensionBundle bundle(
      "tensorext-lite-extension-bundle",
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteExtensionPluginName(),
      tianchenrv::plugin::registerTensorExtLiteExtensionPlugin);
  if (int result = expectSuccess(
          plugin->configureTargetSupportExtensionBundle(bundle),
          "TensorExtLite target-support extension bundle configures"))
    return result;
  const auto &route = tianchenrv::plugin::tensorext_lite::
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
  if (int result =
          expect(bundle.getRequiredDialectNames().size() == 1 &&
                     bundle.getRequiredDialectNames().front() ==
                         "tcrv_tensorext_lite" &&
                     bundle.getLoweringBoundaryOps().size() == 1 &&
                     bundle.getLoweringBoundaryOps().front() ==
                         route.loweringBoundaryOpName &&
                     static_cast<bool>(
                         bundle
                             .getTargetArtifactExporterBundleRegistrationFn()),
                 "TensorExtLite extension bundle publishes required dialect, "
                 "selected lowering-boundary op, and target artifact "
                 "exporter registration"))
    return result;

  const auto &manifest =
      tianchenrv::plugin::tensorext_lite::getTensorExtLiteConstructionManifest();
  const auto &realization =
      tianchenrv::plugin::tensorext_lite::getTensorExtLiteTypedRoleGraphRealization();
  if (int result = expectSuccess(
          tianchenrv::plugin::tensorext_lite::verifyTensorExtLiteConstructionManifest(manifest),
          "TensorExtLite construction manifest verifies"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::tensorext_lite::verifyTensorExtLiteTypedRoleGraphRealization(
              manifest, realization),
          "TensorExtLite typed role graph verifies"))
    return result;
  if (int result =
          expect(realization.roles.size() == 4 &&
                     realization.roles[0].operationName ==
                         "tcrv_tensorext_lite.config_skeleton" &&
                     realization.roles[1].operationName ==
                         "tcrv_tensorext_lite.load_frag_skeleton" &&
                     realization.roles[2].role == "tile_mma" &&
                     realization.roles[2].operationName ==
                         "tcrv_tensorext_lite.tile_mma_skeleton" &&
                     realization.roles[2].roleSpecificInterface ==
                         "TCRVComputeOpInterface" &&
                     realization.roles[2].emitCLowerableInterface ==
                         "TCRVEmitCLowerableInterface" &&
                     realization.roles[3].operationName ==
                         "tcrv_tensorext_lite.store_frag_skeleton" &&
                     realization.roles[3].emitCLowerableInterface ==
                         "TCRVEmitCLowerableInterface",
                 "TensorExtLite typed role graph preserves ordered role sequence"))
    return result;

  llvm::ArrayRef<tianchenrv::plugin::tensorext_lite::
                     TensorExtLiteFragmentMmaRoleStep>
      roleSteps = tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentMmaRoleSteps();
  if (int result =
          expect(roleSteps.size() == realization.roles.size() &&
                     roleSteps[0].operationName ==
                         realization.roles[0].operationName &&
                     roleSteps[1].sourceRole == "load_frag" &&
                     roleSteps[2].typedRoleID ==
                         realization.roles[2].typedRoleID &&
                     roleSteps[3].callee ==
                         route.storeFragCallee &&
                     tianchenrv::plugin::tensorext_lite::
                         getTensorExtLiteFragmentMmaSourceOps() ==
                         "tcrv_tensorext_lite.config_skeleton->"
                         "tcrv_tensorext_lite.load_frag_skeleton->"
                         "tcrv_tensorext_lite.tile_mma_skeleton->"
                         "tcrv_tensorext_lite.store_frag_skeleton",
                 "TensorExtLite construction protocol exposes ordered "
                 "role-step and route-callee data"))
    return result;
  if (int result =
          expect(manifest.emitcRoute.routeID == route.routeID &&
                     manifest.emitcRoute.emissionKind == route.emissionKind &&
                     manifest.emitcRoute.artifactKind == route.artifactKind &&
                     route.artifactKind == "riscv-elf-relocatable-object" &&
                     manifest.evidenceProfile.contains(
                         "materialized_emitc_module"),
                 "TensorExtLite construction manifest records the EmitC route "
                 "with bounded object artifact authority"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::tensorext_lite::
              verifyTensorExtLiteConstructionProtocolReady(),
          "TensorExtLite construction protocol ready check validates active route"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::tensorext_lite::
              verifyTensorExtLiteFragmentMmaEmitCConstructionRouteMapping(
                  route.routeID, route.emissionKind, route.artifactKind,
                  route.runtimeABI, route.runtimeABIKind, route.runtimeABIName,
                  route.runtimeGlueRole),
          "TensorExtLite active EmitC route mapping validates"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::tensorext_lite::
              verifyTensorExtLiteFragmentMmaTargetArtifactBundleMapping(
                  route.headerRouteID, route.headerArtifactKind,
                  route.bundleComponentGroup, route.objectHandoffKind,
                  route.emitCToCppTranslateRouteID),
          "TensorExtLite target artifact bundle mapping validates"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::tensorext_lite::
              verifyTensorExtLiteFragmentMmaArtifactMetadata(
                  tianchenrv::plugin::tensorext_lite::
                      getTensorExtLiteFragmentMmaArtifactMetadata(),
                  "TensorExtLite construction test"),
          "TensorExtLite artifact metadata validates from protocol"))
    return result;
  llvm::SmallVector<tianchenrv::support::ArtifactMetadataEntry, 12>
      staleProtocolMetadata(
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteFragmentMmaArtifactMetadata()
                  .begin(),
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteFragmentMmaArtifactMetadata()
                  .end());
  staleProtocolMetadata[2].value = "tcrv_tensorext_lite.stale_skeleton";
  if (int result = expectErrorContains(
          tianchenrv::plugin::tensorext_lite::
              verifyTensorExtLiteFragmentMmaArtifactMetadata(
                  staleProtocolMetadata,
                  "TensorExtLite construction test stale metadata"),
          {tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteSourceOpsMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteFragmentMmaSourceOps()}))
    return result;
  if (int result = expectErrorContains(
          tianchenrv::plugin::tensorext_lite::
              verifyTensorExtLiteFragmentMmaEmitCConstructionRouteMapping(
                  "tensorext-lite-fragment-mma-no-active-emitc-route",
                  route.emissionKind, route.artifactKind, route.runtimeABI,
                  route.runtimeABIKind, route.runtimeABIName,
                  route.runtimeGlueRole),
          {"TensorExtLite EmitC route id", route.routeID}))
    return result;

  return expectErrorContains(
      tianchenrv::plugin::registerTensorExtLiteExtensionPlugin(registry),
      {"duplicate TianChen-RV extension plugin", "tensorext-lite-plugin"});
}

int runProposalGatingAndDeclineTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @available_tensorext_lite attributes {} {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
  }

  tcrv.exec.kernel @missing_tensorext_lite attributes {} {
  }

  tcrv.exec.kernel @unavailable_tensorext_lite attributes {} {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "unavailable",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
  }

  tcrv.exec.kernel @malformed_tensorext_lite attributes {} {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "wrong-fragment-abi",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse TensorExtLite proposal gating module");

  KernelOp available = findKernel(*module, "available_tensorext_lite");
  KernelOp missing = findKernel(*module, "missing_tensorext_lite");
  KernelOp unavailable = findKernel(*module, "unavailable_tensorext_lite");
  KernelOp malformed = findKernel(*module, "malformed_tensorext_lite");
  if (int result =
          expect(available && missing && unavailable && malformed,
                 "proposal gating module contains all TensorExtLite kernels"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTensorExtLiteExtensionPlugin(registry),
                        "register TensorExtLite plugin for proposal gating"))
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
          "available TensorExtLite capability collects proposal"))
    return result;
  if (int result =
          expect(proposals.size() == 1 && declines.empty(),
                 "available TensorExtLite capability proposes one variant"))
    return result;

  const VariantProposal &proposal = proposals.front();
  if (int result =
          expect(proposal.getVariantName() ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteFragmentFirstSliceVariantName() &&
                     proposal.getOriginPlugin() ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteExtensionPluginName() &&
                     proposal.getPolicy() ==
                         tianchenrv::plugin::tensorext_lite::getTensorExtLiteFragmentPolicy(),
                 "TensorExtLite proposal preserves stable generic metadata"))
    return result;
  if (int result =
          expect(proposal.getRequiredCapabilityIDs().size() == 1 &&
                     proposal.getRequiredCapabilityIDs().front() ==
                         tianchenrv::plugin::tensorext_lite::getTensorExtLiteFragmentCapabilityID(),
                 "TensorExtLite proposal requires tensorext_lite.tile_mma capability id"))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, tianchenrv::plugin::tensorext_lite::getTensorExtLiteFragmentABIAttrName(),
          tianchenrv::plugin::tensorext_lite::getTensorExtLiteExpectedFragmentABI()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, tianchenrv::plugin::tensorext_lite::getTensorExtLiteHandoffKindAttrName(),
          tianchenrv::plugin::tensorext_lite::getTensorExtLiteExpectedHandoffKind()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_tensorext_lite.construction_protocol",
          tianchenrv::plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
              .protocolVersion))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_tensorext_lite.semantic_role_graph",
          tianchenrv::plugin::tensorext_lite::getTensorExtLiteConstructionManifest()
              .semanticRoleGraph))
    return result;
  if (int result = expectProposalStringAttr(
          proposal, "tcrv_tensorext_lite.typed_role_realization",
          tianchenrv::plugin::tensorext_lite::getTensorExtLiteTypedRoleRealizationSummary()))
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
                      " produces no proposal without an available TensorExtLite "
                      "capability");
  };

  if (int result = expectNoProposal(missing, "missing TensorExtLite capability"))
    return result;
  if (int result = expectNoProposal(unavailable, "unavailable TensorExtLite capability"))
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
          "malformed TensorExtLite capability decline is recoverable"))
    return result;
  return expect(proposals.empty() && declines.size() == 1 &&
                    declines.front().getPluginName() ==
                        tianchenrv::plugin::tensorext_lite::getTensorExtLiteExtensionPluginName() &&
                    declines.front().getReason().contains("fragment_abi"),
                "malformed TensorExtLite capability records plugin-local decline");
}

int runPipelineHookTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @tensorext_lite_tile_mma_kernel attributes {} {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse TensorExtLite pipeline hook module");

  KernelOp kernel = findKernel(*module, "tensorext_lite_tile_mma_kernel");
  if (int result = expect(kernel, "TensorExtLite pipeline module has kernel anchor"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTensorExtLiteExtensionPlugin(registry),
                        "register TensorExtLite plugin for pipeline hook"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(kernel.getOperation(), kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize TensorExtLite proposal"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "TensorExtLite capability materializes one variant"))
    return result;

  VariantOp tensorext_liteVariant = findVariant(
      kernel,
      tianchenrv::plugin::tensorext_lite::getTensorExtLiteFragmentFirstSliceVariantName());
  if (int result = expect(tensorext_liteVariant, "TensorExtLite variant is materialized"))
    return result;
  if (int result =
          expect(tensorext_liteVariant->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     tianchenrv::plugin::tensorext_lite::getTensorExtLiteExtensionPluginName(),
                 "TensorExtLite variant has TensorExtLite origin"))
    return result;
  if (int result =
          expect(tensorext_liteVariant
                         ->getAttrOfType<mlir::StringAttr>(
                             tianchenrv::plugin::tensorext_lite::
                                 getTensorExtLiteFragmentABIAttrName())
                         .getValue() ==
                     tianchenrv::plugin::tensorext_lite::getTensorExtLiteExpectedFragmentABI(),
                 "TensorExtLite variant carries fragment ABI metadata"))
    return result;

  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "materialized TensorExtLite module verifies"))
    return result;
  if (int result =
          expectSuccess(registry.verifyKernelVariantLegality(kernel,
                                                             capabilities),
                        "TensorExtLite legality accepts materialized variant"))
    return result;

  VariantCostEstimate estimate;
  if (int result = expectSuccess(
          registry.estimateVariantCost(
              VariantCostRequest(tensorext_liteVariant, kernel, capabilities), estimate),
          "TensorExtLite cost estimate routes through plugin"))
    return result;
  if (int result =
          expect(estimate.hasScore() && estimate.getScore() == 50.0 &&
                     estimate.hasExplicitPreference() &&
                     estimate.getOriginPlugin() ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteExtensionPluginName() &&
                     estimate.getVariantSymbol() == tensorext_liteVariant.getSymName(),
                 "TensorExtLite cost metadata is plugin-owned"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("TensorExtLite selection planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan selectionPlan = std::move(*planOrError);
  if (int result =
          expect(selectionPlan.kind == VariantSelectionKind::StaticVariant &&
                     selectionPlan.selectedVariant == tensorext_liteVariant &&
                     !selectionPlan.fallback && selectionPlan.dispatchCases.empty(),
                 "TensorExtLite selected path uses the generic static variant plan"))
    return result;

  DiagnosticOp marker;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize TensorExtLite selected marker"))
    return result;
  if (int result = expect(marker, "TensorExtLite selected marker was created"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::plugin::materializeSelectedLoweringBoundaries(
              kernel, capabilities, registry),
          "materialize TensorExtLite selected boundary"))
    return result;

  LoweringBoundaryOp boundary =
      findTensorExtLiteBoundary(kernel, tensorext_liteVariant.getSymName());
  if (int result =
          expect(boundary,
                 "TensorExtLite selected path materializes a route boundary"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "TensorExtLite boundary and role-body module verifies"))
    return result;

  TileMmaSkeletonOp nestedComputeRole = findTensorExtLiteNestedComputeRole(
      tensorext_liteVariant, tensorext_liteVariant.getSymName());
  if (int result =
          expect(nestedComputeRole,
                 "TensorExtLite selected path materializes tile_mma role op"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::plugin::tensorext_lite::
              verifyTensorExtLiteRoleOpInterface(
                  tianchenrv::plugin::tensorext_lite::
                      getTensorExtLiteConstructionManifest(),
                  tianchenrv::plugin::tensorext_lite::
                      getTensorExtLiteTypedRoleGraphRealization(),
                  nestedComputeRole.getOperation(), "tile_mma"),
          "TensorExtLite tile_mma role op validates through construction interface"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectSuccess(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(tensorext_liteVariant, kernel,
                                     capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          "TensorExtLite emission readiness sees active explicit role route"))
    return result;
  const auto &routeMetadata = tianchenrv::plugin::tensorext_lite::
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
  const auto &manifest = tianchenrv::plugin::tensorext_lite::
      getTensorExtLiteConstructionManifest();
  llvm::ArrayRef<tianchenrv::plugin::tensorext_lite::
                     TensorExtLiteFragmentMmaRoleStep>
      roleSteps = tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentMmaRoleSteps();
  if (int result =
          expect(status.isSupported() &&
                     status.getEmissionPath() == routeMetadata.routeID,
                 "TensorExtLite readiness reports supported active route id"))
    return result;

  VariantEmissionPlan emissionPlan;
  emissionPlan = VariantEmissionPlan();
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(tensorext_liteVariant, kernel,
                                     capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "TensorExtLite emission plan is plugin-owned"))
    return result;
  if (int result =
          expect(emissionPlan.isSupported() &&
                     emissionPlan.getOriginPlugin() ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteExtensionPluginName() &&
                     emissionPlan.getKernelSymbol() == kernel.getSymName() &&
                     emissionPlan.getVariantSymbol() ==
                         tensorext_liteVariant.getSymName() &&
                     emissionPlan.getLoweringPipeline() ==
                         routeMetadata.routeID &&
                     emissionPlan.getEmissionKind() ==
                         routeMetadata.emissionKind &&
                     emissionPlan.getArtifactKind() ==
                         "riscv-elf-relocatable-object" &&
                     emissionPlan.getRuntimeABI() ==
                         routeMetadata.runtimeABI &&
                     emissionPlan.getRuntimeABIKind() ==
                         routeMetadata.runtimeABIKind &&
                     emissionPlan.getRuntimeABIName() ==
                         routeMetadata.runtimeABIName &&
                     emissionPlan.getRuntimeGlueRole() ==
                         routeMetadata.runtimeGlueRole &&
                     emissionPlan.getLoweringBoundaryOpName() ==
                         routeMetadata.loweringBoundaryOpName &&
                     tianchenrv::support::runtimeABIParametersEqual(
                         emissionPlan.getRuntimeABIParameters(),
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteFragmentMmaRuntimeABIParameters()) &&
                     emissionPlan.getExplanation().contains(
                         "relocatable object artifact") &&
                     emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                     emissionPlan.getRequiredCapabilitySymbols().front() ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteFragmentPreferredCapabilitySymbol() &&
                     emissionPlan.getArtifactMetadata().size() == 12 &&
                     emissionPlan.getArtifactMetadata()[0].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteEmitCLowerableRouteMetadataName() &&
                     emissionPlan.getArtifactMetadata()[0].value ==
                         routeMetadata.routeID &&
                     emissionPlan.getArtifactMetadata()[1].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteRoleSequenceMetadataName() &&
                     emissionPlan.getArtifactMetadata()[1].value ==
                         manifest.semanticRoleGraph &&
                     emissionPlan.getArtifactMetadata()[2].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteSourceOpsMetadataName() &&
                     emissionPlan.getArtifactMetadata()[2].value ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteFragmentMmaSourceOps() &&
                     emissionPlan.getArtifactMetadata()[3].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteSourceRolesMetadataName() &&
                     emissionPlan.getArtifactMetadata()[3].value ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteFragmentMmaSourceRoles() &&
                     emissionPlan.getArtifactMetadata()[4].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteSourceOpInterfaceMetadataName() &&
                     emissionPlan.getArtifactMetadata()[4].value ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteEmitCLowerableOpInterfaceName() &&
                     emissionPlan.getArtifactMetadata()[5].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteConstructionProtocolMetadataName() &&
                     emissionPlan.getArtifactMetadata()[5].value ==
                         manifest.protocolVersion &&
                     emissionPlan.getArtifactMetadata()[6].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteConstructionArchetypeMetadataName() &&
                     emissionPlan.getArtifactMetadata()[6].value ==
                         manifest.archetype &&
                     emissionPlan.getArtifactMetadata()[7].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteSemanticRoleGraphMetadataName() &&
                     emissionPlan.getArtifactMetadata()[7].value ==
                         manifest.semanticRoleGraph &&
                     emissionPlan.getArtifactMetadata()[8].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteCommonInterfaceRealizationMetadataName() &&
                     emissionPlan.getArtifactMetadata()[8].value ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteConstructionInterfaceRealization() &&
                     emissionPlan.getArtifactMetadata()[9].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteTypedRoleRealizationMetadataName() &&
                     emissionPlan.getArtifactMetadata()[9].value ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteTypedRoleRealizationSummary() &&
                     emissionPlan.getArtifactMetadata()[10].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteEmitCRouteMappingMetadataName() &&
                     emissionPlan.getArtifactMetadata()[10].value ==
                         manifest.emitcRoute.routeID &&
                     emissionPlan.getArtifactMetadata()[11].key ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteEvidenceProfileMetadataName() &&
                     emissionPlan.getArtifactMetadata()[11].value ==
                         manifest.evidenceProfile,
                 "TensorExtLite emission plan is a supported object artifact "
                 "candidate backed by EmitC route provenance"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::tensorext_lite::
              verifyTensorExtLiteFragmentMmaArtifactMetadata(
                  emissionPlan.getArtifactMetadata(),
                  "TensorExtLite emission plan metadata"),
          "TensorExtLite emission-plan artifact metadata matches protocol"))
    return result;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute emitcRoute;
  if (int result = expectSuccess(
          registry.buildVariantEmitCLowerableRoute(
              VariantEmitCLowerableRequest(tensorext_liteVariant, kernel,
                                           capabilities,
                                           VariantEmissionRole::DirectVariant),
              emitcRoute),
          "TensorExtLite route builds through registry"))
    return result;
  if (int result =
          expect(emitcRoute.getRouteID() == routeMetadata.routeID &&
                     emitcRoute.getFunctionDeclarations().size() == 4 &&
                     emitcRoute.getFunctionDeclarations()[0].name ==
                         roleSteps[0].callee &&
                     emitcRoute.getSourceOpProvenance().size() == 4 &&
                     emitcRoute.getCallOpaqueSteps().size() == 4 &&
                     emitcRoute.getSourceOpProvenance()[0].role ==
                         roleSteps[0].sourceRole &&
                     emitcRoute.getSourceOpProvenance()[1].role ==
                         roleSteps[1].sourceRole &&
                     emitcRoute.getSourceOpProvenance()[2].role ==
                         roleSteps[2].sourceRole &&
                     emitcRoute.getSourceOpProvenance()[3].role ==
                         roleSteps[3].sourceRole &&
                     emitcRoute.getCallOpaqueSteps()[2].callee ==
                         roleSteps[2].callee,
                 "TensorExtLite route preserves role sequence provenance and "
                 "call-opaque mapping"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::conversion::emitc::
              verifyTCRVEmitCLowerableRouteMaterializesToEmitC(
                  emitcRoute, "tcrv_tensorext_lite_test", {}),
          "TensorExtLite common EmitC materializer accepts the route"))
    return result;

  return 0;
}

int runRoleSequenceOrderingNegativeTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @reordered_tensorext_lite_tile_mma_kernel attributes {} {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse reordered TensorExtLite role-sequence module");

  KernelOp kernel =
      findKernel(*module, "reordered_tensorext_lite_tile_mma_kernel");
  if (int result =
          expect(kernel, "reordered TensorExtLite module has kernel anchor"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTensorExtLiteExtensionPlugin(
                            registry),
                        "register TensorExtLite plugin for reordered role "
                        "negative"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(kernel.getOperation(), kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize TensorExtLite proposal for reordered role negative"))
    return result;

  VariantOp tensorext_liteVariant = findVariant(
      kernel,
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentFirstSliceVariantName());
  if (int result =
          expect(tensorext_liteVariant,
                 "TensorExtLite reordered negative materializes variant"))
    return result;

  materializeTensorExtLiteRoleSequence(builder, kernel, tensorext_liteVariant,
                                       /*reorderLoadAndTile=*/true);
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "reordered TensorExtLite role module verifies"))
    return result;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute emitcRoute;
  return expectErrorContains(
      registry.buildVariantEmitCLowerableRoute(VariantEmitCLowerableRequest(
          tensorext_liteVariant, kernel, capabilities,
          VariantEmissionRole::DirectVariant),
                                               emitcRoute),
      {"selected TensorExtLite role ops must appear in",
       tianchenrv::plugin::tensorext_lite::
           getTensorExtLiteConstructionManifest()
               .semanticRoleGraph,
       "order"});
}

int runPartialRoleSequenceMaterializationNegativeTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @partial_tensorext_lite_tile_mma_kernel attributes {} {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse partial TensorExtLite role-sequence module");

  KernelOp kernel =
      findKernel(*module, "partial_tensorext_lite_tile_mma_kernel");
  if (int result =
          expect(kernel, "partial TensorExtLite module has kernel anchor"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTensorExtLiteExtensionPlugin(
                            registry),
                        "register TensorExtLite plugin for partial role "
                        "negative"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(kernel.getOperation(), kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize TensorExtLite proposal for partial role negative"))
    return result;

  VariantOp tensorext_liteVariant = findVariant(
      kernel,
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentFirstSliceVariantName());
  if (int result =
          expect(tensorext_liteVariant,
                 "TensorExtLite partial negative materializes variant"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("TensorExtLite partial role selection planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan selectionPlan = std::move(*planOrError);
  DiagnosticOp marker;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize TensorExtLite selected marker for partial role negative"))
    return result;

  materializeTensorExtLiteRoleSequence(builder, kernel, tensorext_liteVariant,
                                       /*reorderLoadAndTile=*/false,
                                       /*roleCount=*/1);
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "partial TensorExtLite role module verifies before "
                          "construction-template fail-closed check"))
    return result;

  return expectErrorContains(
      tianchenrv::plugin::materializeSelectedLoweringBoundaries(
          kernel, capabilities, registry),
      {"partial materialized role sequence",
       tianchenrv::plugin::tensorext_lite::
           getTensorExtLiteConstructionManifest()
               .semanticRoleGraph});
}

int runDuplicateRoleSequenceMaterializationNegativeTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @duplicate_tensorext_lite_tile_mma_kernel attributes {} {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse duplicate TensorExtLite role-sequence module");

  KernelOp kernel =
      findKernel(*module, "duplicate_tensorext_lite_tile_mma_kernel");
  if (int result =
          expect(kernel, "duplicate TensorExtLite module has kernel anchor"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTensorExtLiteExtensionPlugin(
                            registry),
                        "register TensorExtLite plugin for duplicate role "
                        "negative"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(kernel.getOperation(), kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize TensorExtLite proposal for duplicate role negative"))
    return result;

  VariantOp tensorext_liteVariant = findVariant(
      kernel,
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentFirstSliceVariantName());
  if (int result =
          expect(tensorext_liteVariant,
                 "TensorExtLite duplicate negative materializes variant"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("TensorExtLite duplicate role selection planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan selectionPlan = std::move(*planOrError);
  DiagnosticOp marker;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize TensorExtLite selected marker for duplicate role negative"))
    return result;

  materializeTensorExtLiteRoleSequence(builder, kernel, tensorext_liteVariant);
  materializeTensorExtLiteRoleSequence(builder, kernel, tensorext_liteVariant,
                                       /*reorderLoadAndTile=*/false,
                                       /*roleCount=*/1);
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "duplicate TensorExtLite role module verifies before "
                          "construction-template fail-closed check"))
    return result;

  return expectErrorContains(
      tianchenrv::plugin::materializeSelectedLoweringBoundaries(
          kernel, capabilities, registry),
      {"duplicate materialized role op",
       "tcrv_tensorext_lite.config_skeleton"});
}

} // namespace

int main() {
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerTensorExtLiteExtensionPlugin(
                            dialectPlugins),
                        "register TensorExtLite plugin for dialect setup"))
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
  if (int result = runRoleSequenceOrderingNegativeTest(context))
    return result;
  if (int result = runPartialRoleSequenceMaterializationNegativeTest(context))
    return result;
  if (int result = runDuplicateRoleSequenceMaterializationNegativeTest(context))
    return result;

  llvm::outs() << "TensorExtLite extension plugin fragment smoke test passed\n";
  return 0;
}
