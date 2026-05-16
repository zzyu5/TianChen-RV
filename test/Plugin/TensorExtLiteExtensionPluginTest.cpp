#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
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

TileMmaSkeletonOp findTensorExtLiteComputeRole(KernelOp kernel,
                                     llvm::StringRef selectedVariantSymbol) {
  TileMmaSkeletonOp result;
  if (!kernel || kernel.getBody().empty())
    return result;

  for (mlir::Operation &op : kernel.getBody().front()) {
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

struct TensorExtLiteTestRoleSpec {
  llvm::StringRef operationName;
  llvm::StringRef typedRole;
  llvm::StringRef sourceRole;
  llvm::StringRef roleSpecificInterface;
  int64_t roleOrder = 0;
};

void materializeTensorExtLiteRoleSequence(mlir::OpBuilder &builder,
                                          KernelOp kernel,
                                          VariantOp variant) {
  static constexpr TensorExtLiteTestRoleSpec kRoleSpecs[] = {
      {"tcrv_tensorext_lite.config_skeleton", "tel.role.config",
       "configure", "TCRVConfigOpInterface", 0},
      {"tcrv_tensorext_lite.load_frag_skeleton", "tel.role.load_frag",
       "load_frag", "TCRVMemoryOpInterface", 1},
      {"tcrv_tensorext_lite.tile_mma_skeleton", "tel.role.tile_mma",
       "tile_mma", "TCRVComputeOpInterface", 2},
      {"tcrv_tensorext_lite.store_frag_skeleton", "tel.role.store_frag",
       "store_frag", "TCRVMemoryOpInterface", 3},
  };

  mlir::Block &body = variant.getBody().front();
  builder.setInsertionPointToEnd(&body);
  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  for (const TensorExtLiteTestRoleSpec &spec : kRoleSpecs) {
    mlir::OperationState state(variant.getLoc(), spec.operationName);
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
    state.addAttribute("typed_role", builder.getStringAttr(spec.typedRole));
    state.addAttribute("role_order",
                       builder.getI64IntegerAttr(spec.roleOrder));
    state.addAttribute("source_role", builder.getStringAttr(spec.sourceRole));
    state.addAttribute("role_specific_interface",
                       builder.getStringAttr(spec.roleSpecificInterface));
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

  const auto &route = tianchenrv::plugin::tensorext_lite::
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
  if (int result =
          expect(manifest.emitcRoute.routeID == route.routeID &&
                     manifest.emitcRoute.emissionKind == route.emissionKind &&
                     manifest.emitcRoute.artifactKind == route.artifactKind &&
                     route.artifactKind == "unsupported-emission-diagnostic" &&
                     manifest.evidenceProfile.contains(
                         "materialized_emitc_module"),
                 "TensorExtLite construction manifest records the EmitC route "
                 "without target artifact authority"))
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
          expect(!boundary,
                 "TensorExtLite selected path does not materialize a route boundary"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "TensorExtLite no-boundary module verifies"))
    return result;

  TileMmaSkeletonOp computeRole =
      findTensorExtLiteComputeRole(kernel, tensorext_liteVariant.getSymName());
  if (int result =
          expect(!computeRole,
                 "TensorExtLite selected path does not materialize a tile_mma role op"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectErrorContains(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(tensorext_liteVariant, kernel,
                                     capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          {"reported unsupported emission path",
           "requires one materialized tcrv_tensorext_lite.config_skeleton"}))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectErrorContains(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(tensorext_liteVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          {"requires one materialized tcrv_tensorext_lite.config_skeleton"}))
    return result;

  materializeTensorExtLiteRoleSequence(builder, kernel, tensorext_liteVariant);
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "TensorExtLite explicit role-body module verifies"))
    return result;

  TileMmaSkeletonOp nestedComputeRole = findTensorExtLiteNestedComputeRole(
      tensorext_liteVariant, tensorext_liteVariant.getSymName());
  if (int result =
          expect(nestedComputeRole,
                 "TensorExtLite explicit body materializes tile_mma role op"))
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

  status = VariantEmissionStatus();
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
  if (int result =
          expect(status.isSupported() &&
                     status.getEmissionPath() == routeMetadata.routeID,
                 "TensorExtLite readiness reports supported active route id"))
    return result;

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
          expect(emissionPlan.isUnsupported() &&
                     emissionPlan.getOriginPlugin() ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteExtensionPluginName() &&
                     emissionPlan.getKernelSymbol() == kernel.getSymName() &&
                     emissionPlan.getVariantSymbol() ==
                         tensorext_liteVariant.getSymName() &&
                     emissionPlan.getLoweringPipeline().empty() &&
                     emissionPlan.getEmissionKind().empty() &&
                     emissionPlan.getArtifactKind().empty() &&
                     emissionPlan.getLoweringBoundaryOpName().empty() &&
                     emissionPlan.getDiagnostic().contains(
                         "target artifact export remains unsupported") &&
                     emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                     emissionPlan.getRequiredCapabilitySymbols().front() ==
                         tianchenrv::plugin::tensorext_lite::
                             getTensorExtLiteFragmentPreferredCapabilitySymbol(),
                 "TensorExtLite emission plan is an unsupported diagnostic "
                 "while EmitC route materialization remains separate"))
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
                     emitcRoute.getSourceOpProvenance().size() == 4 &&
                     emitcRoute.getCallOpaqueSteps().size() == 4 &&
                     emitcRoute.getSourceOpProvenance()[0].role ==
                         "configure" &&
                     emitcRoute.getSourceOpProvenance()[1].role ==
                         "load_frag" &&
                     emitcRoute.getSourceOpProvenance()[2].role ==
                         "tile_mma" &&
                     emitcRoute.getSourceOpProvenance()[3].role ==
                         "store_frag" &&
                     emitcRoute.getCallOpaqueSteps()[2].callee ==
                         routeMetadata.tileMmaCallee,
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

  llvm::outs() << "TensorExtLite extension plugin fragment smoke test passed\n";
  return 0;
}
