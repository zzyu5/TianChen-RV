#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/RVVScalarBinaryFamily.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"
#include "TianChenRV/Transforms/VariantSelection.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
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
using tianchenrv::plugin::VariantLoweringBoundaryRequest;
using tianchenrv::plugin::VariantLoweringBoundaryResult;
using tianchenrv::plugin::VariantProposal;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::conversion::emitc::TCRVEmitCLowerableOpInterface;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::target::rvv_scalar::RVVScalarBinaryFamilyRecord;
using tianchenrv::tcrv::scalar::LoweringBoundaryOp;
using tianchenrv::tcrv::scalar::I32VAddMicrokernelOp;
using tianchenrv::tcrv::scalar::I32VMulMicrokernelOp;
using tianchenrv::tcrv::scalar::I32VSubMicrokernelOp;
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

int expectDeletedScalarDirectCEmissionPlan(
    const VariantEmissionPlan &emissionPlan, llvm::Twine context) {
  return expect(emissionPlan.isUnsupported() &&
                    emissionPlan.getOriginPlugin() ==
                        tianchenrv::plugin::scalar::
                            getScalarExtensionPluginName() &&
                    emissionPlan.getRuntimeABIKind() ==
                        "unsupported-plugin-runtime-abi" &&
                    emissionPlan.getRuntimeABIName() ==
                        "unsupported-emission-runtime-abi" &&
                    emissionPlan.getRuntimeGlueRole() ==
                        "no-runtime-glue-unsupported" &&
                    emissionPlan.getEmissionKind().empty() &&
                    emissionPlan.getLoweringPipeline().empty() &&
                    emissionPlan.getRuntimeABI().empty() &&
                    emissionPlan.getArtifactKind().empty() &&
                    emissionPlan.getRequiredCapabilitySymbols().empty() &&
                    emissionPlan.getRuntimeABIParameters().empty() &&
                    emissionPlan.getSelectedPlanMetadata().empty() &&
                    emissionPlan.getDiagnostic().contains(
                        "scalar direct C source exporter was deleted"),
                context);
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

LoweringBoundaryOp findScalarBoundary(KernelOp kernel,
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

I32VAddMicrokernelOp findScalarMicrokernel(
    KernelOp kernel, llvm::StringRef selectedVariantSymbol) {
  I32VAddMicrokernelOp result;
  if (!kernel || kernel.getBody().empty())
    return result;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto microkernel = llvm::dyn_cast<I32VAddMicrokernelOp>(op);
    if (!microkernel)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    if (selectedVariant && selectedVariant.getValue() == selectedVariantSymbol)
      result = microkernel;
  }
  return result;
}

I32VSubMicrokernelOp findScalarSubMicrokernel(
    KernelOp kernel, llvm::StringRef selectedVariantSymbol) {
  I32VSubMicrokernelOp result;
  if (!kernel || kernel.getBody().empty())
    return result;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto microkernel = llvm::dyn_cast<I32VSubMicrokernelOp>(op);
    if (!microkernel)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    if (selectedVariant && selectedVariant.getValue() == selectedVariantSymbol)
      result = microkernel;
  }
  return result;
}

I32VMulMicrokernelOp findScalarMulMicrokernel(
    KernelOp kernel, llvm::StringRef selectedVariantSymbol) {
  I32VMulMicrokernelOp result;
  if (!kernel || kernel.getBody().empty())
    return result;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto microkernel = llvm::dyn_cast<I32VMulMicrokernelOp>(op);
    if (!microkernel)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    if (selectedVariant && selectedVariant.getValue() == selectedVariantSymbol)
      result = microkernel;
  }
  return result;
}

mlir::Operation *findScalarMicrokernelOperationByName(
    KernelOp kernel, llvm::StringRef selectedVariantSymbol,
    llvm::StringRef operationName) {
  if (!kernel || kernel.getBody().empty())
    return nullptr;

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != operationName)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    if (selectedVariant && selectedVariant.getValue() == selectedVariantSymbol)
      return &op;
  }
  return nullptr;
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

  tcrv.exec.kernel @available_scalar_vadd attributes {
    tcrv_frontend_lowering = "i32-vadd"
  } {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }

  tcrv.exec.kernel @available_scalar_vsub attributes {
    tcrv_frontend_lowering = "i32-vsub"
  } {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }

  tcrv.exec.kernel @available_scalar_vmul attributes {
    tcrv_frontend_lowering = "i32-vmul"
  } {
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
  KernelOp availableVAdd = findKernel(*module, "available_scalar_vadd");
  KernelOp availableVSub = findKernel(*module, "available_scalar_vsub");
  KernelOp availableVMul = findKernel(*module, "available_scalar_vmul");
  KernelOp unavailable = findKernel(*module, "unavailable_scalar");
  KernelOp missing = findKernel(*module, "missing_scalar");
  if (int result =
          expect(highLevelOp && available && availableVAdd && availableVSub &&
                     availableVMul && unavailable && missing,
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
                         tianchenrv::plugin::scalar::getScalarFallbackPolicy() &&
                     proposals.front().getFallbackRole() ==
                         tianchenrv::plugin::VariantFallbackRole::
                             ConservativeFallback,
                 "scalar fallback proposal preserves stable metadata"))
    return result;
  if (int result =
          expect(proposals.front().getRequiredCapabilityIDs().size() == 1 &&
                     proposals.front().getRequiredCapabilityIDs().front() ==
                         tianchenrv::plugin::scalar::
                             getScalarFallbackCapabilityID(),
                 "scalar fallback proposal requires fallback capability id"))
    return result;
  bool hasElementCount = false;
  for (mlir::NamedAttribute attr : proposals.front().getPluginAttributes()) {
    llvm::StringRef name = attr.getName().getValue();
    if (name == "tcrv_scalar.element_count") {
      hasElementCount = true;
    }
  }
  if (int result =
          expect(!hasElementCount,
                 "default scalar i32-vadd proposal carries no element-count "
                 "route metadata"))
    return result;

  auto expectDescriptorlessScalarProposalForKernel =
      [&](KernelOp kernel, llvm::StringRef context) -> int {
    TargetCapabilitySet kernelCapabilities =
        TargetCapabilitySet::buildFromKernel(kernel);
    VariantProposalRequest kernelRequest(highLevelOp.getOperation(), kernel,
                                         kernelCapabilities);
    proposals.clear();
    if (int result = expectSuccess(
            registry.collectVariantProposals(kernelRequest, proposals),
            llvm::Twine("collect scalar proposal for ") + context))
      return result;
    if (int result = expect(proposals.size() == 1,
                            llvm::Twine("one scalar proposal for ") + context))
      return result;

    for (mlir::NamedAttribute attr : proposals.front().getPluginAttributes()) {
      llvm::StringRef name = attr.getName().getValue();
      if (name == "tcrv_scalar.element_count")
        return fail(llvm::Twine("default scalar proposal for ") + context +
                    " must not carry element-count route metadata");
    }
    return 0;
  };

  if (int result = expectDescriptorlessScalarProposalForKernel(
          availableVAdd, "frontend-lowered i32-vadd"))
    return result;
  if (int result = expectDescriptorlessScalarProposalForKernel(
          availableVSub, "frontend-lowered i32-vsub"))
    return result;
  if (int result = expectDescriptorlessScalarProposalForKernel(
          availableVMul, "frontend-lowered i32-vmul"))
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

  auto fallbackRoleAttr = variant->getAttrOfType<mlir::StringAttr>(
      tianchenrv::plugin::kVariantFallbackRoleAttrName);
  if (int result =
          expect(fallbackRoleAttr &&
                     fallbackRoleAttr.getValue() ==
                         tianchenrv::plugin::kConservativeFallbackRoleValue,
                 "materialized scalar fallback variant preserves generic fallback role"))
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
                     estimate.hasExplicitPreference() &&
                     estimate.getOriginPlugin() ==
                         tianchenrv::plugin::scalar::
                             getScalarExtensionPluginName() &&
                     estimate.getVariantSymbol() == variant.getSymName() &&
                     estimate.getFallbackRole() ==
                         tianchenrv::plugin::VariantFallbackRole::
                             ConservativeFallback,
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

  VariantLoweringBoundaryResult boundaryResult;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&kernel.getBody().front());
    if (int result = expectSuccess(
            registry.materializeSelectedLoweringBoundary(
                VariantLoweringBoundaryRequest(
                    variant, kernel, capabilities,
                    VariantEmissionRole::DirectVariant, builder),
                boundaryResult),
            "scalar fallback selected boundary materializes through plugin"))
      return result;
  }
  if (int result =
          expect(boundaryResult.isMaterialized(),
                 "scalar fallback boundary result is materialized"))
    return result;
  auto scalarBoundary =
      llvm::dyn_cast_or_null<LoweringBoundaryOp>(
          boundaryResult.getMaterializedOperation());
  if (int result = expect(scalarBoundary,
                          "scalar fallback boundary is a tcrv_scalar op"))
    return result;
  if (int result =
          expect(scalarBoundary->getParentOp() == kernel.getOperation(),
                 "scalar fallback boundary is a direct kernel child"))
    return result;
  if (int result =
          expect(scalarBoundary->getAttrOfType<mlir::StringAttr>(
                     "source_kernel")
                         .getValue() == kernel.getSymName(),
                 "scalar boundary preserves source kernel metadata"))
    return result;
  if (int result =
          expect(scalarBoundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
                     "selected_variant")
                         .getValue() == variant.getSymName(),
                 "scalar boundary preserves selected variant reference"))
    return result;
  if (int result =
          expect(scalarBoundary->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     tianchenrv::plugin::scalar::
                         getScalarExtensionPluginName(),
                 "scalar boundary preserves origin plugin metadata"))
    return result;
  if (int result =
          expect(scalarBoundary->getAttrOfType<mlir::StringAttr>("role")
                         .getValue() == "direct variant" &&
                     scalarBoundary->getAttrOfType<mlir::StringAttr>("status")
                         .getValue() == "metadata-only",
                 "scalar boundary records direct metadata-only selected path"))
    return result;
  if (int result =
          expect(scalarBoundary->getAttrOfType<mlir::StringAttr>(
                     "fallback_reason")
                         .getValue()
                         .contains("plugin-owned metadata only"),
                 "scalar boundary carries non-executable fallback reason"))
    return result;
  auto boundaryRequires =
      scalarBoundary->getAttrOfType<mlir::ArrayAttr>("required_capabilities");
  if (int result = expect(boundaryRequires && boundaryRequires == requiresAttr,
                          "scalar boundary preserves capability references"))
    return result;
  I32VAddMicrokernelOp scalarMicrokernel =
      findScalarMicrokernel(kernel, variant.getSymName());
  if (int result =
          expect(scalarMicrokernel,
                 "descriptorless scalar fallback materializes a typed "
                 "microkernel"))
    return result;
  if (int result =
          expect(scalarMicrokernel->getParentOp() == kernel.getOperation(),
                 "scalar fallback microkernel is a direct kernel child"))
    return result;
  if (int result =
          expect(scalarMicrokernel->getAttrOfType<mlir::StringAttr>(
                     "source_kernel")
                         .getValue() == kernel.getSymName(),
                 "scalar microkernel preserves source kernel metadata"))
    return result;
  if (int result =
          expect(scalarMicrokernel->getAttrOfType<mlir::FlatSymbolRefAttr>(
                     "selected_variant")
                         .getValue() == variant.getSymName(),
                 "scalar microkernel preserves selected variant reference"))
    return result;
  if (int result =
          expect(scalarMicrokernel->getAttrOfType<mlir::StringAttr>("role")
                         .getValue() == "direct variant" &&
                     scalarMicrokernel
                             ->getAttrOfType<mlir::IntegerAttr>(
                                 "element_count")
                             .getInt() == 16,
                 "scalar microkernel records direct role and bounded element count"))
    return result;
  auto microkernelRequires =
      scalarMicrokernel->getAttrOfType<mlir::ArrayAttr>(
          "required_capabilities");
  if (int result = expect(microkernelRequires &&
                              microkernelRequires == requiresAttr,
                          "scalar microkernel preserves capability references"))
    return result;
  if (int result =
          expect(mlir::succeeded(mlir::verify(*module)),
                 "scalar boundary and microkernel module verifies after materialization"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectErrorContains(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          {"reported unsupported emission path",
           "scalar direct C source exporter was deleted"}))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "scalar fallback emission plan is plugin-owned"))
    return result;
  if (int result = expect(
          emissionPlan.getKernelSymbol() == kernel.getSymName() &&
              emissionPlan.getVariantSymbol() == variant.getSymName(),
          "scalar fallback emission plan records kernel and variant"))
    return result;
  if (int result = expectDeletedScalarDirectCEmissionPlan(
          emissionPlan,
          "scalar fallback emission plan records deleted source route"))
    return result;

  return 0;
}

int runDescriptorlessScalarSubMaterializationTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @scalar_vsub attributes {
    tcrv_frontend_lowering = "i32-vsub"
  } {
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
    return fail("failed to parse scalar descriptorless vsub module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "scalar_vsub");
  if (int result =
          expect(highLevelOp && kernel,
                 "scalar descriptorless vsub test has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar plugin for descriptorless vsub"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result =
          expectSuccess(registry.collectVariantProposals(request, proposals),
                        "collect scalar vsub proposal"))
    return result;
  if (int result = expect(proposals.size() == 1,
                          "scalar vsub collects one proposal"))
    return result;
  bool hasElementCount = false;
  for (mlir::NamedAttribute attr : proposals.front().getPluginAttributes()) {
    if (attr.getName().getValue() == "tcrv_scalar.element_count")
      hasElementCount = true;
  }
  if (int result =
          expect(!hasElementCount,
                 "scalar vsub proposal is descriptorless typed i32"))
    return result;

  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize scalar vsub proposal"))
    return result;
  if (int result = expect(materializedVariants.size() == 1,
                          "one scalar vsub variant is materialized"))
    return result;
  VariantOp variant = materializedVariants.front();

  VariantLoweringBoundaryResult boundaryResult;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&kernel.getBody().front());
    if (int result = expectSuccess(
            registry.materializeSelectedLoweringBoundary(
                VariantLoweringBoundaryRequest(
                    variant, kernel, capabilities,
                    VariantEmissionRole::DirectVariant, builder),
                boundaryResult),
            "materialize descriptorless scalar vsub boundary"))
      return result;
  }
  if (int result =
          expect(boundaryResult.isMaterialized(),
                 "descriptorless scalar vsub boundary materializes"))
    return result;
  if (int result =
          expect(findScalarSubMicrokernel(kernel, variant.getSymName()),
                 "typed-source scalar vsub path materializes vsub op"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "build scalar vsub emission plan"))
    return result;
  if (int result = expectDeletedScalarDirectCEmissionPlan(
          emissionPlan,
          "scalar vsub emission plan records deleted source route"))
    return result;

  return 0;
}

int runDescriptorlessScalarMulMaterializationTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @scalar_vmul attributes {
    tcrv_frontend_lowering = "i32-vmul"
  } {
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
    return fail("failed to parse scalar descriptorless vmul module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "scalar_vmul");
  if (int result =
          expect(highLevelOp && kernel,
                 "scalar descriptorless vmul test has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar plugin for descriptorless vmul"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result =
          expectSuccess(registry.collectVariantProposals(request, proposals),
                        "collect scalar vmul proposal"))
    return result;
  if (int result = expect(proposals.size() == 1,
                          "scalar vmul collects one proposal"))
    return result;
  bool hasElementCount = false;
  for (mlir::NamedAttribute attr : proposals.front().getPluginAttributes()) {
    if (attr.getName().getValue() == "tcrv_scalar.element_count")
      hasElementCount = true;
  }
  if (int result =
          expect(!hasElementCount,
                 "scalar vmul proposal is descriptorless typed i32"))
    return result;

  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize scalar vmul proposal"))
    return result;
  if (int result = expect(materializedVariants.size() == 1,
                          "one scalar vmul variant is materialized"))
    return result;
  VariantOp variant = materializedVariants.front();

  VariantLoweringBoundaryResult boundaryResult;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&kernel.getBody().front());
    if (int result = expectSuccess(
            registry.materializeSelectedLoweringBoundary(
                VariantLoweringBoundaryRequest(
                    variant, kernel, capabilities,
                    VariantEmissionRole::DirectVariant, builder),
                boundaryResult),
            "materialize descriptorless scalar vmul boundary"))
      return result;
  }
  if (int result =
          expect(boundaryResult.isMaterialized(),
                 "descriptorless scalar vmul boundary materializes"))
    return result;
  if (int result =
          expect(findScalarMulMicrokernel(kernel, variant.getSymName()),
                 "typed-source scalar vmul path materializes vmul op"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "build scalar vmul emission plan"))
    return result;
  if (int result = expectDeletedScalarDirectCEmissionPlan(
          emissionPlan,
          "scalar vmul emission plan records deleted source route"))
    return result;

  return 0;
}

int runDescriptorlessScalarI64MaterializationCase(
    mlir::MLIRContext &context,
    const RVVScalarBinaryFamilyRecord &family,
    llvm::StringRef diagnosticLabel) {
  std::string kernelName =
      (llvm::Twine("scalar_") + family.scalar.functionStem).str();
  std::string source =
      (llvm::Twine(R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @)mlir") +
       kernelName + R"mlir( attributes {
    tcrv_frontend_lowering = ")mlir" +
       family.frontendLowering + R"mlir("
  } {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}
)mlir")
          .str();

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail(llvm::Twine("failed to parse scalar descriptorless ") +
                diagnosticLabel + " module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, kernelName);
  if (int result =
          expect(highLevelOp && kernel,
                 llvm::Twine("scalar descriptorless ") + diagnosticLabel +
                     " test has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        llvm::Twine("register scalar plugin for "
                                    "descriptorless ") +
                            diagnosticLabel))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result =
          expectSuccess(registry.collectVariantProposals(request, proposals),
                        llvm::Twine("collect scalar ") + diagnosticLabel +
                            " proposal"))
    return result;
  if (int result = expect(proposals.size() == 1,
                          llvm::Twine("scalar ") + diagnosticLabel +
                              " collects one proposal"))
    return result;

  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          llvm::Twine("materialize scalar ") + diagnosticLabel + " proposal"))
    return result;
  if (int result = expect(materializedVariants.size() == 1,
                          llvm::Twine("one scalar ") + diagnosticLabel +
                              " variant is materialized"))
    return result;
  VariantOp variant = materializedVariants.front();

  VariantLoweringBoundaryResult boundaryResult;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&kernel.getBody().front());
    if (int result = expectSuccess(
            registry.materializeSelectedLoweringBoundary(
                VariantLoweringBoundaryRequest(
                    variant, kernel, capabilities,
                    VariantEmissionRole::DirectVariant, builder),
                boundaryResult),
            llvm::Twine("materialize descriptorless scalar ") +
                diagnosticLabel + " boundary"))
      return result;
  }
  if (int result =
          expect(boundaryResult.isMaterialized(),
                 llvm::Twine("descriptorless scalar ") + diagnosticLabel +
                     " boundary materializes"))
    return result;
  mlir::Operation *microkernel = findScalarMicrokernelOperationByName(
      kernel, variant.getSymName(), family.scalar.microkernelOpName);
  if (int result =
          expect(microkernel,
                 llvm::Twine("typed scalar ") + diagnosticLabel +
                     " path materializes i64 op"))
    return result;
  if (int result =
          expect(static_cast<bool>(
                     llvm::isa<TCRVEmitCLowerableOpInterface>(
                         microkernel)),
                 llvm::Twine("scalar ") + diagnosticLabel +
                     " microkernel implements generated EmitC lowerable op "
                     "interface"))
    return result;
  if (int result = expect(
          microkernel->getAttrOfType<mlir::StringAttr>("source_kernel")
                  .getValue() == kernel.getSymName() &&
              microkernel->getAttrOfType<mlir::FlatSymbolRefAttr>(
                  "selected_variant")
                      .getValue() == variant.getSymName(),
          llvm::Twine("scalar ") + diagnosticLabel +
              " microkernel preserves source and selected variant"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          llvm::Twine("build scalar ") + diagnosticLabel + " emission plan"))
    return result;
  if (int result = expectDeletedScalarDirectCEmissionPlan(
          emissionPlan,
          llvm::Twine("scalar ") + diagnosticLabel +
              " emission plan records deleted source route"))
    return result;

  return 0;
}

int runDescriptorlessScalarI64VAddMaterializationTest(
    mlir::MLIRContext &context) {
  return runDescriptorlessScalarI64MaterializationCase(
      context, tianchenrv::target::rvv_scalar::getI64VAddFamilyRegistrationRecord(),
      "i64-vadd");
}

int runDescriptorlessScalarI64VSubMaterializationTest(
    mlir::MLIRContext &context) {
  return runDescriptorlessScalarI64MaterializationCase(
      context, tianchenrv::target::rvv_scalar::getI64VSubFamilyRegistrationRecord(),
      "i64-vsub");
}

int runDescriptorlessScalarI64VMulMaterializationTest(
    mlir::MLIRContext &context) {
  return runDescriptorlessScalarI64MaterializationCase(
      context, tianchenrv::target::rvv_scalar::getI64VMulFamilyRegistrationRecord(),
      "i64-vmul");
}

int runBoundaryMaterializationRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @scalar_boundary_rejections attributes {} {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.capability @portable {
      id = "portable",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @malformed_scalar_selected attributes {
      origin = "scalar-plugin",
      requires = [@portable]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse scalar boundary rejection module");

  KernelOp kernel = findKernel(*module, "scalar_boundary_rejections");
  VariantOp malformed = findVariant(kernel, "malformed_scalar_selected");
  if (int result = expect(kernel && malformed,
                          "boundary rejection module has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar fallback plugin for boundary rejection"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&kernel.getBody().front());
  VariantLoweringBoundaryResult boundaryResult;
  return expectErrorContains(
      registry.materializeSelectedLoweringBoundary(
          VariantLoweringBoundaryRequest(malformed, kernel, capabilities,
                                         VariantEmissionRole::DirectVariant,
                                         builder),
          boundaryResult),
      {"selected scalar fallback variant @malformed_scalar_selected",
       "failed plugin legality before boundary materialization",
       "must require capability id", "scalar.fallback"});
}

int runRVVDeclineStillMaterializesScalarBoundaryTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @rvv_decline_scalar_boundary attributes {} {
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
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV-decline scalar boundary module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "rvv_decline_scalar_boundary");
  if (int result =
          expect(highLevelOp && kernel,
                 "RVV-decline scalar boundary module has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerBuiltinExtensionPlugins(
                            registry),
                        "register built-in plugins for RVV decline coverage"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 2> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize scalar proposal after recoverable RVV decline"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "RVV decline leaves one scalar fallback proposal"))
    return result;

  VariantOp scalarVariant =
      findVariant(kernel, tianchenrv::plugin::scalar::
                              getScalarFallbackFirstSliceVariantName());
  if (int result = expect(scalarVariant,
                          "scalar fallback variant exists after RVV decline"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("RVV-decline scalar selection planning failed: " +
                llvm::toString(planOrError.takeError()));

  DiagnosticOp marker;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeSelectedVariantMarker(
              builder, *planOrError, &marker),
          "materialize selected marker after RVV decline"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::plugin::materializeSelectedLoweringBoundaries(
              kernel, capabilities, registry),
          "materialize scalar boundary after RVV decline"))
    return result;

  if (int result =
          expect(findScalarBoundary(kernel, scalarVariant.getSymName()),
                 "RVV decline does not block scalar boundary materialization"))
    return result;
  if (int result =
          expect(findScalarMicrokernel(kernel, scalarVariant.getSymName()),
                 "RVV decline scalar fallback materializes microkernel"))
    return result;
  if (int result =
          expect(mlir::succeeded(mlir::verify(*module)),
                 "RVV-decline scalar boundary module verifies"))
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
  if (int result = runDescriptorlessScalarSubMaterializationTest(context))
    return result;
  if (int result = runDescriptorlessScalarMulMaterializationTest(context))
    return result;
  if (int result = runDescriptorlessScalarI64VAddMaterializationTest(context))
    return result;
  if (int result = runDescriptorlessScalarI64VSubMaterializationTest(context))
    return result;
  if (int result = runDescriptorlessScalarI64VMulMaterializationTest(context))
    return result;
  if (int result = runBoundaryMaterializationRejectionTest(context))
    return result;
  if (int result = runRVVDeclineStillMaterializesScalarBoundaryTest(context))
    return result;
  if (int result = runLegalityRejectionTest(context))
    return result;

  llvm::outs() << "scalar fallback extension plugin smoke test passed\n";
  return 0;
}
