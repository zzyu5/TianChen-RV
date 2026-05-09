#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
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
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::target::i32_binary::I32BinaryFamilyDescriptor;
using tianchenrv::target::i32_binary::I32BinaryFamilyKind;
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
  bool hasDescriptor = false;
  bool hasElementCount = false;
  for (mlir::NamedAttribute attr : proposals.front().getPluginAttributes()) {
    llvm::StringRef name = attr.getName().getValue();
    if (name == "tcrv_scalar.lowering_descriptor") {
      auto value = llvm::dyn_cast<mlir::StringAttr>(attr.getValue());
      hasDescriptor = value &&
                      value.getValue() ==
                          tianchenrv::target::i32_binary::
                              getI32VAddFamilyDescriptor()
                                  .loweringDescriptor;
    }
    if (name == "tcrv_scalar.element_count") {
      auto value = llvm::dyn_cast<mlir::IntegerAttr>(attr.getValue());
      hasElementCount = value && value.getInt() == 16;
    }
  }
  if (int result =
          expect(hasDescriptor && hasElementCount,
                 "scalar fallback proposal carries finite i32-vadd descriptor"))
    return result;

  auto expectScalarDescriptorForKernel =
      [&](KernelOp kernel, llvm::StringRef expectedDescriptor,
          llvm::StringRef context) -> int {
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

    bool descriptorMatches = false;
    bool elementCountMatches = false;
    for (mlir::NamedAttribute attr : proposals.front().getPluginAttributes()) {
      llvm::StringRef name = attr.getName().getValue();
      if (name == "tcrv_scalar.lowering_descriptor") {
        auto value = llvm::dyn_cast<mlir::StringAttr>(attr.getValue());
        descriptorMatches = value && value.getValue() == expectedDescriptor;
      }
      if (name == "tcrv_scalar.element_count") {
        auto value = llvm::dyn_cast<mlir::IntegerAttr>(attr.getValue());
        elementCountMatches = value && value.getInt() == 16;
      }
    }
    return expect(descriptorMatches && elementCountMatches,
                  llvm::Twine("scalar proposal descriptor matches ") +
                      context);
  };

  if (int result = expectScalarDescriptorForKernel(
          availableVAdd,
          tianchenrv::target::i32_binary::getI32VAddFamilyDescriptor()
              .loweringDescriptor,
          "frontend-lowered i32-vadd"))
    return result;
  if (int result = expectScalarDescriptorForKernel(
          availableVSub,
          tianchenrv::target::i32_binary::getI32VSubFamilyDescriptor()
              .loweringDescriptor,
          "frontend-lowered i32-vsub"))
    return result;
  if (int result = expectScalarDescriptorForKernel(
          availableVMul,
          tianchenrv::target::i32_binary::getI32VMulFamilyDescriptor()
              .loweringDescriptor,
          "frontend-lowered i32-vmul"))
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
                 "scalar fallback descriptor materializes a microkernel"))
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
                         "scalar-explicit-i32-vadd-microkernel-c-source-export",
                 "scalar fallback readiness reports supported source route"))
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
                         "scalar-explicit-i32-vadd-microkernel-c-source" &&
                     emissionPlan.getLoweringPipeline() ==
                         "tcrv-export-scalar-microkernel-c" &&
                     emissionPlan.getRuntimeABI() ==
                         "scalar-i32-vadd-runtime-callable-c-abi.v1" &&
                     emissionPlan.getRuntimeABIKind() ==
                         "scalar-runtime-callable-c-abi" &&
                     emissionPlan.getRuntimeABIName() ==
                         "scalar-i32-vadd-runtime-callable-c-function.v1" &&
                     emissionPlan.getRuntimeGlueRole() ==
                         "runtime-callable-i32-vadd-fallback-function" &&
                     emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                     emissionPlan.getRequiredCapabilitySymbols().front() ==
                         "scalar_fallback" &&
                     emissionPlan.getRuntimeABIParameters().size() == 4 &&
                     emissionPlan.getRuntimeABIParameters()[0].role ==
                         tianchenrv::support::RuntimeABIParameterRole::
                             LHSInputBuffer &&
                     emissionPlan.getRuntimeABIParameters()[3].role ==
                         tianchenrv::support::RuntimeABIParameterRole::
                             RuntimeElementCount &&
                     emissionPlan.getRuntimeABIParameters()[3].ownership ==
                         tianchenrv::support::RuntimeABIParameterOwnership::
                             TargetExportABIOwned &&
                     emissionPlan.getArtifactKind() ==
                         "runtime-callable-c-source",
                 "scalar fallback emission plan records stable supported source route"))
    return result;

  return 0;
}

int runDescriptorBackedScalarSubMaterializationTest(
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

  const I32BinaryFamilyDescriptor &family =
      tianchenrv::target::i32_binary::getI32VSubFamilyDescriptor();

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse scalar descriptor-backed vsub module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "scalar_vsub");
  if (int result =
          expect(highLevelOp && kernel,
                 "scalar descriptor-backed vsub test has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar plugin for descriptor-backed vsub"))
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
  bool hasDescriptor = false;
  for (mlir::NamedAttribute attr : proposals.front().getPluginAttributes()) {
    if (attr.getName().getValue() == "tcrv_scalar.lowering_descriptor") {
      auto value = llvm::dyn_cast<mlir::StringAttr>(attr.getValue());
      hasDescriptor = value && value.getValue() == family.loweringDescriptor;
    }
  }
  if (int result =
          expect(hasDescriptor,
                 "scalar vsub proposal consumes registry lowering descriptor"))
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
            "materialize descriptor-backed scalar vsub boundary"))
      return result;
  }
  if (int result =
          expect(boundaryResult.isMaterialized(),
                 "descriptor-backed scalar vsub boundary materializes"))
    return result;
  if (int result =
          expect(findScalarSubMicrokernel(kernel, variant.getSymName()),
                 "registry-backed scalar vsub descriptor materializes vsub op"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "build scalar vsub emission plan"))
    return result;
  if (int result =
          expect(emissionPlan.isSupported() &&
                     emissionPlan.getEmissionKind() ==
                         family.scalar.emissionKind &&
                     emissionPlan.getLoweringPipeline() ==
                         family.scalar.routeID &&
                     emissionPlan.getRuntimeABI() ==
                         family.scalar.runtimeABI &&
                     emissionPlan.getRuntimeABIKind() ==
                         family.scalar.runtimeABIKind &&
                     emissionPlan.getRuntimeABIName() ==
                         family.scalar.runtimeABIName &&
                     emissionPlan.getRuntimeGlueRole() ==
                         family.scalar.runtimeGlueRole,
                 "scalar vsub emission plan consumes registry facts"))
    return result;

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&kernel.getBody().front());
    mlir::OperationState staleState(
        variant.getLoc(), I32VAddMicrokernelOp::getOperationName());
    staleState.addAttribute(
        "selected_variant",
        mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                     variant.getSymName()));
    staleState.addAttribute("role", builder.getStringAttr("direct variant"));
    builder.create(staleState);
  }

  VariantEmissionStatus staleStatus;
  return expectErrorContains(
      registry.checkVariantEmissionReadiness(
          VariantEmissionRequest(variant, kernel, capabilities,
                                 VariantEmissionRole::DirectVariant),
          staleStatus),
      {"descriptor requires", family.scalar.microkernelOpName});
}

int runDescriptorBackedScalarMulMaterializationTest(
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

  const I32BinaryFamilyDescriptor &family =
      tianchenrv::target::i32_binary::getI32VMulFamilyDescriptor();

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse scalar descriptor-backed vmul module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "scalar_vmul");
  if (int result =
          expect(highLevelOp && kernel,
                 "scalar descriptor-backed vmul test has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar plugin for descriptor-backed vmul"))
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
  bool hasDescriptor = false;
  for (mlir::NamedAttribute attr : proposals.front().getPluginAttributes()) {
    if (attr.getName().getValue() == "tcrv_scalar.lowering_descriptor") {
      auto value = llvm::dyn_cast<mlir::StringAttr>(attr.getValue());
      hasDescriptor = value && value.getValue() == family.loweringDescriptor;
    }
  }
  if (int result =
          expect(hasDescriptor,
                 "scalar vmul proposal consumes registry lowering descriptor"))
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
            "materialize descriptor-backed scalar vmul boundary"))
      return result;
  }
  if (int result =
          expect(boundaryResult.isMaterialized(),
                 "descriptor-backed scalar vmul boundary materializes"))
    return result;
  if (int result =
          expect(findScalarMulMicrokernel(kernel, variant.getSymName()),
                 "registry-backed scalar vmul descriptor materializes vmul op"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "build scalar vmul emission plan"))
    return result;
  if (int result =
          expect(emissionPlan.isSupported() &&
                     emissionPlan.getEmissionKind() ==
                         family.scalar.emissionKind &&
                     emissionPlan.getLoweringPipeline() ==
                         family.scalar.routeID &&
                     emissionPlan.getRuntimeABI() ==
                         family.scalar.runtimeABI &&
                     emissionPlan.getRuntimeABIKind() ==
                         family.scalar.runtimeABIKind &&
                     emissionPlan.getRuntimeABIName() ==
                         family.scalar.runtimeABIName &&
                     emissionPlan.getRuntimeGlueRole() ==
                         family.scalar.runtimeGlueRole,
                 "scalar vmul emission plan consumes registry facts"))
    return result;

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&kernel.getBody().front());
    mlir::OperationState staleState(
        variant.getLoc(), I32VAddMicrokernelOp::getOperationName());
    staleState.addAttribute(
        "selected_variant",
        mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                     variant.getSymName()));
    staleState.addAttribute("role", builder.getStringAttr("direct variant"));
    builder.create(staleState);
  }

  VariantEmissionStatus staleStatus;
  return expectErrorContains(
      registry.checkVariantEmissionReadiness(
          VariantEmissionRequest(variant, kernel, capabilities,
                                 VariantEmissionRole::DirectVariant),
          staleStatus),
      {"descriptor requires", family.scalar.microkernelOpName});
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
  if (int result = runDescriptorBackedScalarSubMaterializationTest(context))
    return result;
  if (int result = runDescriptorBackedScalarMulMaterializationTest(context))
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
