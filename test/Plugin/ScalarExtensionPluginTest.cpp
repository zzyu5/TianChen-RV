#include "Weft/InitWeftDialects.h"
#include "Weft/Dialect/Scalar/IR/ScalarDialect.h"
#include "Weft/Plugin/BuiltinExtensionPlugins.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Transforms/VariantMaterialization.h"
#include "Weft/Transforms/VariantSelection.h"

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

using weft::plugin::ExtensionPluginRegistry;
using weft::plugin::ExtensionBundleRegistry;
using weft::plugin::FamilyConstructionResult;
using weft::plugin::PluginCapability;
using weft::plugin::VariantCostEstimate;
using weft::plugin::VariantCostRequest;
using weft::plugin::VariantEmissionPlan;
using weft::plugin::VariantEmissionRequest;
using weft::plugin::VariantEmissionRole;
using weft::plugin::VariantEmissionStatus;
using weft::plugin::VariantLoweringBoundaryRequest;
using weft::plugin::VariantLoweringBoundaryResult;
using weft::plugin::VariantProposal;
using weft::plugin::VariantProposalRequest;
using weft::support::CapabilityAvailability;
using weft::support::CapabilityDescriptor;
using weft::support::TargetCapabilitySet;
using weft::exec::CapabilityRelationsAttr;
using weft::exec::DiagnosticOp;
using weft::exec::KernelOp;
using weft::exec::VariantOp;
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

int expectScalarUnsupportedEmissionPlan(
    const VariantEmissionPlan &emissionPlan, llvm::Twine context) {
  return expect(emissionPlan.isUnsupported() &&
                    emissionPlan.getOriginPlugin() ==
                        weft::plugin::scalar::
                            getScalarExtensionPluginName() &&
                    emissionPlan.getEmissionKind() ==
                        "scalar-fallback-unsupported-emission" &&
                    emissionPlan.getLoweringPipeline() ==
                        "scalar-no-constructed-body-route" &&
                    emissionPlan.getRuntimeABI() ==
                        "scalar-no-constructed-body-abi" &&
                    emissionPlan.getRuntimeABIKind() ==
                        "unsupported-plugin-runtime-abi" &&
                    emissionPlan.getRuntimeABIName() ==
                        "unsupported-emission-runtime-abi" &&
                    emissionPlan.getRuntimeGlueRole() ==
                        "no-runtime-glue-unsupported" &&
                    emissionPlan.getArtifactKind() ==
                        "unsupported-emission-diagnostic" &&
                    emissionPlan.getDiagnostic().contains(
                        "exact constructed final typed body") &&
                    emissionPlan.getRuntimeABIParameters().empty() &&
                    emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                    emissionPlan.getRequiredCapabilitySymbols().front() ==
                        "scalar_fallback" &&
                    emissionPlan.getExplanation().empty(),
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

bool hasScalarLoweringBoundary(KernelOp kernel) {
  if (!kernel || kernel.getBody().empty())
    return false;

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() == "weft_scalar.lowering_boundary")
      return true;
  }
  return false;
}

int runRegistrationAndCapabilityMetadataTest() {
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(weft::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar fallback plugin"))
    return result;

  const auto *plugin = registry.lookupPlugin(
      weft::plugin::scalar::getScalarExtensionPluginName());
  if (int result = expect(plugin, "registered scalar fallback plugin is visible"))
    return result;
  if (int result =
          expect(plugin->getVersion() ==
                     weft::plugin::scalar::
                         getScalarExtensionPluginVersion(),
                 "scalar fallback plugin version is stable"))
    return result;

  const PluginCapability *capability = registry.lookupCapabilityByID(
      weft::plugin::scalar::getScalarFallbackCapabilityID());
  if (int result = expect(
          capability &&
              capability->getKind() ==
                  weft::plugin::scalar::
                      getScalarFallbackCapabilityKind(),
          "scalar fallback capability metadata is registered"))
    return result;

  llvm::SmallVector<const weft::plugin::ExtensionPlugin *, 2> enabled;
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

  weft.exec.kernel @available_scalar attributes {} {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }

  weft.exec.kernel @unavailable_scalar attributes {} {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "unavailable"
    }
  }

  weft.exec.kernel @missing_scalar attributes {} {
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
          expectSuccess(weft::plugin::registerScalarExtensionPlugin(
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
                         weft::plugin::scalar::
                             getScalarFallbackFirstSliceVariantName() &&
                     proposals.front().getOriginPlugin() ==
                         weft::plugin::scalar::
                             getScalarExtensionPluginName() &&
                     proposals.front().getPolicy() ==
                         weft::plugin::scalar::getScalarFallbackPolicy() &&
                     proposals.front().getFallbackRole() ==
                         weft::plugin::VariantFallbackRole::
                             ConservativeFallback,
                 "scalar fallback proposal preserves stable metadata"))
    return result;
  if (int result =
          expect(proposals.front().getRequiredCapabilityIDs().size() == 1 &&
                     proposals.front().getRequiredCapabilityIDs().front() ==
                         weft::plugin::scalar::
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

  weft.exec.kernel @scalar_only attributes {} {
    weft.exec.capability @scalar_fallback {
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
          expectSuccess(weft::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar fallback plugin for materialization"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          weft::transforms::collectAndMaterializeVariantProposals(
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
                     weft::plugin::scalar::
                         getScalarFallbackFirstSliceVariantName(),
                 "materialized scalar fallback variant has stable symbol"))
    return result;
  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  if (int result =
          expect(originAttr &&
                     originAttr.getValue() ==
                     weft::plugin::scalar::
                         getScalarExtensionPluginName(),
                 "materialized scalar fallback variant has scalar origin"))
    return result;
  auto policyAttr = variant->getAttrOfType<mlir::StringAttr>("policy");
  if (int result =
          expect(policyAttr &&
                     policyAttr.getValue() ==
                     weft::plugin::scalar::getScalarFallbackPolicy(),
                 "materialized scalar fallback variant preserves policy"))
    return result;

  auto fallbackRoleAttr = variant->getAttrOfType<mlir::StringAttr>(
      weft::plugin::kVariantFallbackRoleAttrName);
  if (int result =
          expect(fallbackRoleAttr &&
                     fallbackRoleAttr.getValue() ==
                         weft::plugin::kConservativeFallbackRoleValue,
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
                  weft::plugin::scalar::
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
                         weft::plugin::scalar::
                             getScalarExtensionPluginName() &&
                     estimate.getVariantSymbol() == variant.getSymName() &&
                     estimate.getFallbackRole() ==
                         weft::plugin::VariantFallbackRole::
                             ConservativeFallback,
                 "scalar fallback cost metadata is plugin-owned"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      weft::transforms::planKernelVariantSelection(kernel, capabilities,
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
          weft::transforms::materializeSelectedVariantMarker(
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
            "Scalar construction requires no separate lowering boundary"))
      return result;
  }
  if (int result =
          expect(boundaryResult.isNoBoundary() &&
                     !hasScalarLoweringBoundary(kernel),
                 "scalar fallback reports no metadata lowering boundary"))
    return result;
  if (int result =
          expect(mlir::succeeded(mlir::verify(*module)),
                 "scalar fallback module verifies without metadata boundary"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectErrorContains(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          {"scalar-plugin", "reported unsupported emission path",
           "exact constructed final typed body", "not emittable"}))
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
  if (int result = expectScalarUnsupportedEmissionPlan(
          emissionPlan,
          "descriptorless no-body scalar fallback is unsupported fail-closed"))
    return result;

  return 0;
}

int runTypedFinalBodyConstructionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @scalar_immediate {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @scalar_immediate_variant attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft_scalar.compute_skeleton {source_kernel = "scalar_immediate", selected_variant = @scalar_immediate_variant, scalar_immediate = 7 : i64}
  }
  weft.exec.kernel @scalar_tq2 {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @scalar_tq2_variant attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft_scalar.tq2_0_q8_k_vec_dot {source_kernel = "scalar_tq2", selected_variant = @scalar_tq2_variant, qk = 256 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 64 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64}
  }
  weft.exec.kernel @scalar_q4 {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @scalar_q4_variant attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft_scalar.dequantize_row_q4_0 {source_kernel = "scalar_q4", selected_variant = @scalar_q4_variant, qk = 32 : i64, weight_block_stride = 18 : i64, weight_d_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse Scalar typed-body construction module");

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          weft::plugin::registerScalarExtensionPlugin(registry),
          "register Scalar plugin for typed-body construction"))
    return result;

  auto construct = [&](llvm::StringRef kernelName,
                       llvm::StringRef variantName,
                       FamilyConstructionResult &result) -> int {
    KernelOp kernel = findKernel(*module, kernelName);
    VariantOp variant = findVariant(kernel, variantName);
    if (int status = expect(kernel && variant,
                            "typed-body construction anchors exist"))
      return status;
    return expectSuccess(registry.constructFormulaPlansForVariant(
                             *module, variant, result),
                         "construct exact Scalar final typed body");
  };

  FamilyConstructionResult immediateResult;
  if (int result = construct("scalar_immediate", "scalar_immediate_variant",
                             immediateResult))
    return result;
  auto immediate = llvm::dyn_cast_if_present<weft::scalar::ImmediateCallBodyOp>(
      immediateResult.getOperation());
  if (int result = expect(
          immediateResult.hasFinalBody() && immediate &&
              immediate.getScalarImmediate() == 7,
          "compute_skeleton is consumed into an exact immediate_call_body"))
    return result;

  FamilyConstructionResult tq2Result;
  if (int result =
          construct("scalar_tq2", "scalar_tq2_variant", tq2Result))
    return result;
  auto tq2 = llvm::dyn_cast_if_present<weft::scalar::PackedTernaryDotBodyOp>(
      tq2Result.getOperation());
  weft::scalar::TernaryBlockLoopOp tq2Block;
  weft::scalar::TernaryPlaneGroupLoopOp tq2Group;
  weft::scalar::TernaryPlaneLoopOp tq2Plane;
  weft::scalar::TernaryLaneLoopOp tq2Lane;
  weft::scalar::TernaryDecodeMacOp tq2Decode;
  weft::scalar::TernaryScaleFoldOp tq2Fold;
  bool tq2HasStore = false;
  if (tq2 && !tq2.getBody().empty()) {
    mlir::Block &root = tq2.getBody().front();
    if (root.getOperations().size() == 2) {
      tq2Block = llvm::dyn_cast<weft::scalar::TernaryBlockLoopOp>(
          &root.front());
      tq2HasStore = llvm::isa<weft::scalar::TernaryStoreOp>(root.back());
    }
  }
  if (tq2Block && !tq2Block.getBody().empty()) {
    mlir::Block &block = tq2Block.getBody().front();
    if (block.getOperations().size() == 2) {
      tq2Group = llvm::dyn_cast<weft::scalar::TernaryPlaneGroupLoopOp>(
          &block.front());
      tq2Fold = llvm::dyn_cast<weft::scalar::TernaryScaleFoldOp>(
          &block.back());
    }
  }
  if (tq2Group && !tq2Group.getBody().empty() &&
      tq2Group.getBody().front().getOperations().size() == 1)
    tq2Plane = llvm::dyn_cast<weft::scalar::TernaryPlaneLoopOp>(
        &tq2Group.getBody().front().front());
  if (tq2Plane && !tq2Plane.getBody().empty() &&
      tq2Plane.getBody().front().getOperations().size() == 1)
    tq2Lane = llvm::dyn_cast<weft::scalar::TernaryLaneLoopOp>(
        &tq2Plane.getBody().front().front());
  if (tq2Lane && !tq2Lane.getBody().empty() &&
      tq2Lane.getBody().front().getOperations().size() == 1)
    tq2Decode = llvm::dyn_cast<weft::scalar::TernaryDecodeMacOp>(
        &tq2Lane.getBody().front().front());
  if (int result = expect(
          tq2Result.hasFinalBody() && tq2 && tq2Block && tq2Group &&
              tq2Plane && tq2Lane && tq2Decode && tq2Fold && tq2HasStore &&
              tq2Block.getQk() == 256 &&
              tq2Block.getWeightBlockStride() == 66 &&
              tq2Group.getUpperBound() == 64 && tq2Group.getStep() == 32 &&
              tq2Plane.getUpperBound() == 4 &&
              tq2Plane.getFieldBits() == 2 &&
              tq2Lane.getUpperBound() == 32 &&
              tq2Decode.getFieldMask() == 3 &&
              tq2Decode.getDecodeZeroPoint() == 1 &&
              tq2Fold.getWeightDByteOffset() == 64,
          "tq2 source is consumed into the complete typed loop/decode/fold "
          "plan tree"))
    return result;

  FamilyConstructionResult q4Result;
  if (int result = construct("scalar_q4", "scalar_q4_variant", q4Result))
    return result;
  auto q4 = llvm::dyn_cast_if_present<weft::scalar::PackedAffineDequantBodyOp>(
      q4Result.getOperation());
  weft::scalar::AffineBlockLoopOp q4Block;
  weft::scalar::AffineBlockScaleOp q4Scale;
  weft::scalar::AffineQuantBaseOp q4QuantBase;
  weft::scalar::AffinePackedByteLoopOp q4Packed;
  weft::scalar::AffineDecodeScaleScatterOp q4Decode;
  if (q4 && !q4.getBody().empty() &&
      q4.getBody().front().getOperations().size() == 1)
    q4Block = llvm::dyn_cast<weft::scalar::AffineBlockLoopOp>(
        &q4.getBody().front().front());
  if (q4Block && !q4Block.getBody().empty()) {
    mlir::Block &block = q4Block.getBody().front();
    if (block.getOperations().size() == 3) {
      auto iterator = block.begin();
      q4Scale = llvm::dyn_cast<weft::scalar::AffineBlockScaleOp>(
          &*iterator++);
      q4QuantBase = llvm::dyn_cast<weft::scalar::AffineQuantBaseOp>(
          &*iterator++);
      q4Packed = llvm::dyn_cast<weft::scalar::AffinePackedByteLoopOp>(
          &*iterator);
    }
  }
  if (q4Packed && !q4Packed.getBody().empty() &&
      q4Packed.getBody().front().getOperations().size() == 1)
    q4Decode = llvm::dyn_cast<weft::scalar::AffineDecodeScaleScatterOp>(
        &q4Packed.getBody().front().front());
  if (int result = expect(
          q4Result.hasFinalBody() && q4 && q4Block && q4Scale && q4QuantBase &&
              q4Packed && q4Decode && q4Block.getQk() == 32 &&
              q4Block.getWeightBlockStride() == 18 &&
              q4Scale.getWeightDByteOffset() == 0 &&
              q4QuantBase.getWeightQuantByteOffset() == 2 &&
              q4Packed.getUpperBound() == 16 &&
              q4Decode.getLowFieldShift() == 0 &&
              q4Decode.getHighFieldShift() == 4 &&
              q4Decode.getFieldMask() == 15 &&
              q4Decode.getDecodeZeroPoint() == 8 &&
              q4Decode.getHighOutputDelta() == 16,
          "q4 source is consumed into the complete typed "
          "block/decode/scale/scatter plan tree"))
    return result;

  unsigned sourceCount = 0;
  module->walk([&](mlir::Operation *op) {
    if (llvm::isa<weft::scalar::ComputeSkeletonOp,
                  weft::scalar::TernaryQ2Q8BlockDotOp,
                  weft::scalar::DequantizeRowQ4Op>(op))
      ++sourceCount;
  });
  if (int result = expect(sourceCount == 0,
                          "Scalar source problems do not survive construction"))
    return result;

  KernelOp tq2Kernel = findKernel(*module, "scalar_tq2");
  VariantOp tq2Variant = findVariant(tq2Kernel, "scalar_tq2_variant");
  TargetCapabilitySet tq2Capabilities =
      TargetCapabilitySet::buildFromKernel(tq2Kernel);
  VariantEmissionRequest emissionRequest(
      tq2Variant, tq2Kernel, tq2Capabilities,
      VariantEmissionRole::DirectVariant, tq2.getOperation());
  VariantEmissionStatus status;
  if (int result = expectSuccess(
          registry.checkVariantEmissionReadiness(emissionRequest, status),
          "exact Scalar typed body is emission-ready"))
    return result;
  VariantEmissionPlan plan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(emissionRequest, plan),
          "build exact Scalar typed-body artifact plan"))
    return result;
  if (int result = expect(
          status.isSupported() && plan.isSupported() &&
              plan.getRuntimeABI() == "scalar-tq2-q8-block-dot-c-abi.v1" &&
              plan.getRuntimeABIParameters().size() == 4 &&
              plan.getLoweringBoundaryOpName() ==
                  weft::scalar::PackedTernaryDotBodyOp::getOperationName(),
          "artifact planning consumes the exact body and reports its real ABI"))
    return result;

  FamilyConstructionResult repeated;
  if (int result = construct("scalar_tq2", "scalar_tq2_variant", repeated))
    return result;
  if (int result = expect(
          repeated.getOperation() == tq2.getOperation(),
          "bound Scalar construction is idempotent on its exact final body"))
    return result;

  return expect(mlir::succeeded(mlir::verify(*module)),
                "Scalar typed-body construction module verifies");
}

int runInvalidTernaryGeometryConstructionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @scalar_tq2_invalid {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @scalar_tq2_invalid_variant attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft_scalar.tq2_0_q8_k_vec_dot {source_kernel = "scalar_tq2_invalid", selected_variant = @scalar_tq2_invalid_variant, qk = 255 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 64 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64}
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse invalid Scalar ternary geometry module");

  KernelOp kernel = findKernel(*module, "scalar_tq2_invalid");
  VariantOp variant = findVariant(kernel, "scalar_tq2_invalid_variant");
  if (int result = expect(kernel && variant,
                          "invalid ternary construction anchors exist"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          weft::plugin::registerScalarExtensionPlugin(registry),
          "register Scalar plugin for invalid ternary construction"))
    return result;

  FamilyConstructionResult construction;
  if (int result = expectErrorContains(
          registry.constructFormulaPlansForVariant(*module, variant,
                                                   construction),
          {"canonical qk/stride/offset geometry"}))
    return result;

  unsigned sourceCount = 0;
  unsigned finalBodyCount = 0;
  kernel.walk([&](mlir::Operation *op) {
    if (llvm::isa<weft::scalar::TernaryQ2Q8BlockDotOp>(op))
      ++sourceCount;
    if (llvm::isa<weft::scalar::PackedTernaryDotBodyOp>(op))
      ++finalBodyCount;
  });
  if (int result = expect(
          sourceCount == 1 && finalBodyCount == 0 &&
              !construction.getOperation(),
          "invalid ternary geometry preserves the source problem and creates "
          "no final body"))
    return result;

  return expect(mlir::succeeded(mlir::verify(*module)),
                "invalid ternary source remains structurally valid after "
                "formula rejection");
}

int runBoundaryMaterializationRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @scalar_boundary_rejections attributes {} {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.capability @portable {
      id = "portable",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @malformed_scalar_selected attributes {
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
          expectSuccess(weft::plugin::registerScalarExtensionPlugin(
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

int runRVVDeclineKeepsScalarFallbackEnvelopeBoundarylessTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  weft.exec.kernel @rvv_decline_scalar_envelope attributes {} {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV-decline scalar envelope module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "rvv_decline_scalar_envelope");
  if (int result =
          expect(highLevelOp && kernel,
                 "RVV-decline scalar envelope module has anchors"))
    return result;

  ExtensionBundleRegistry bundles;
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(
              weft::plugin::registerBuiltinExtensionBundlePlugins(
                  bundles, registry),
              "register built-in extension bundle frontdoor for RVV decline "
              "coverage"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 2> materializedVariants;
  if (int result = expectSuccess(
          weft::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize scalar proposal after recoverable RVV decline"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "RVV decline leaves one scalar fallback proposal"))
    return result;

  VariantOp scalarVariant =
      findVariant(kernel, weft::plugin::scalar::
                              getScalarFallbackFirstSliceVariantName());
  if (int result = expect(scalarVariant,
                          "scalar fallback variant exists after RVV decline"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      weft::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("RVV-decline scalar selection planning failed: " +
                llvm::toString(planOrError.takeError()));

  DiagnosticOp marker;
  if (int result = expectSuccess(
          weft::transforms::materializeSelectedVariantMarker(
              builder, *planOrError, &marker),
          "materialize selected marker after RVV decline"))
    return result;

  if (int result = expectSuccess(
          weft::plugin::materializeSelectedLoweringBoundaries(
              kernel, capabilities, registry),
          "fallback-only scalar envelope does not require boundary materialization"))
    return result;

  if (int result =
          expect(!hasScalarLoweringBoundary(kernel),
                 "RVV decline does not materialize a scalar lowering boundary"))
    return result;
  if (int result =
          expect(mlir::succeeded(mlir::verify(*module)),
                 "RVV-decline scalar envelope module verifies"))
    return result;

  return 0;
}

int runLegalityRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @scalar_legality_rejections attributes {} {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @missing_requirement attributes {
      origin = "scalar-plugin",
      requires = []
    } {
    }
  }

  weft.exec.kernel @scalar_unavailable_rejection attributes {} {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "unavailable"
    }
    weft.exec.variant @requires_unavailable attributes {
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
          expectSuccess(weft::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar fallback plugin for legality negatives"))
    return result;

  TargetCapabilitySet availableCapabilities =
      TargetCapabilitySet::buildFromKernel(missingRequirementKernel);
  if (int result = expectErrorContains(
          registry.verifyVariantLegality(weft::plugin::
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
          registry.verifyVariantLegality(weft::plugin::
                                             VariantLegalityRequest(
                                                 requiresUnavailable,
                                                 unavailableKernel,
                                                 unavailableCapabilities)),
          {"scalar fallback", "requires an available capability id",
           "scalar.fallback"}))
    return result;

  return 0;
}

// [F-6] independent-family acceptance ([core-invariants] [F-6], [L-2]
// independent-attached). Two machine-checked conjuncts:
//   (1) the scalar family's capability-predicate transitive `implies` closure
//       ∩ {rvv, rvv.*} = ∅ (built on the new
//       TargetCapabilitySet::impliedClosureAvoidsRVVNamespace helper), with a
//       non-vacuous negative control proving the helper fires when the closure
//       DOES reach the rvv.* namespace directly or transitively; and
//   (2) a vector-absent instance (only scalar.fallback, no rvv) whose scalar
//       variant is only_feasible AND is actually selected -- FallbackOnly with
//       the scalar variant as the chosen selectedVariant, not a dead metadata
//       shell that never wins selection.
//
// Reconciliation with the plugin's 3 fail-closed emission segments and the
// out-of-scope emission unit test (runMaterializationSelectionAndEmissionTest):
// those lock Unsupported for the descriptorless *metadata* variant, which is
// correct -- the conservative fallback envelope has no body to emit. The
// selected variant is proved non-dead-shell by the sibling emittable machine
// check test/Transforms/VariantSelection/f6-independent-scalar-family-emittable.mlir,
// where the SAME vector-absent family carries a typed body and the
// --weft-scalar-emitc-to-cpp route lowers it to real pure-scalar C.
int runFamilyIndependenceAcceptanceTest(mlir::MLIRContext &context) {
  auto impliesOnly = [&](llvm::StringRef impliedID) {
    return CapabilityRelationsAttr::get(
        &context, /*provides=*/{},
        /*implies=*/{mlir::StringAttr::get(&context, impliedID)},
        /*conflicts=*/{});
  };

  // --- conjunct (1): closure ∩ rvv.* = ∅ for the scalar fallback family. ---
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  weft.exec.kernel @only_feasible_scalar attributes {} {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse F-6 vector-absent scalar module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "only_feasible_scalar");
  if (int result = expect(highLevelOp && kernel,
                          "F-6 vector-absent scalar module has anchors"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  const CapabilityDescriptor *scalarFallback = capabilities.lookupByID(
      weft::plugin::scalar::getScalarFallbackCapabilityID());
  if (int result = expect(scalarFallback,
                          "F-6 kernel exposes the scalar.fallback capability"))
    return result;
  if (int result =
          expect(capabilities.impliedClosureAvoidsRVVNamespace(*scalarFallback),
                 "F-6 conjunct 1: scalar.fallback implies-closure ∩ rvv.* = ∅"))
    return result;

  // Non-vacuous negative controls: the helper MUST report non-independence when
  // the transitive implies closure reaches the rvv.* namespace, directly or
  // through an intermediate hop, and MUST NOT trip on a bare-prefix lookalike.
  TargetCapabilitySet directRVV;
  if (int result = expectSuccess(
          directRVV.tryAddCapability(CapabilityDescriptor(
              "fake_scalar_direct", "scalar.fallback.direct", "fallback",
              "available", CapabilityAvailability::Available, /*properties=*/{},
              impliesOnly("rvv"))),
          "add direct-rvv negative-control capability"))
    return result;
  const CapabilityDescriptor *directSeed =
      directRVV.lookupByID("scalar.fallback.direct");
  if (int result = expect(
          directSeed &&
              !directRVV.impliedClosureAvoidsRVVNamespace(*directSeed),
          "F-6 negative control: implies rvv (exact) is NOT independent"))
    return result;

  TargetCapabilitySet dottedRVV;
  if (int result = expectSuccess(
          dottedRVV.tryAddCapability(CapabilityDescriptor(
              "fake_scalar_dotted", "scalar.fallback.dotted", "fallback",
              "available", CapabilityAvailability::Available, /*properties=*/{},
              impliesOnly("rvv.zvfh"))),
          "add dotted-rvv negative-control capability"))
    return result;
  const CapabilityDescriptor *dottedSeed =
      dottedRVV.lookupByID("scalar.fallback.dotted");
  if (int result = expect(
          dottedSeed &&
              !dottedRVV.impliedClosureAvoidsRVVNamespace(*dottedSeed),
          "F-6 negative control: implies rvv.zvfh is NOT independent"))
    return result;

  // Transitive reach: seed implies mid, mid implies rvv.zve32f (a closure leaf).
  TargetCapabilitySet transitiveRVV;
  if (int result = expectSuccess(
          transitiveRVV.tryAddCapability(CapabilityDescriptor(
              "fake_scalar_transitive", "scalar.fallback.transitive", "fallback",
              "available", CapabilityAvailability::Available, /*properties=*/{},
              impliesOnly("bridge.mid"))),
          "add transitive-rvv seed capability"))
    return result;
  if (int result = expectSuccess(
          transitiveRVV.tryAddCapability(CapabilityDescriptor(
              "bridge_mid", "bridge.mid", "profile", "available",
              CapabilityAvailability::Available, /*properties=*/{},
              impliesOnly("rvv.zve32f"))),
          "add transitive-rvv bridge capability"))
    return result;
  const CapabilityDescriptor *transitiveSeed =
      transitiveRVV.lookupByID("scalar.fallback.transitive");
  if (int result =
          expect(transitiveSeed &&
                     !transitiveRVV.impliedClosureAvoidsRVVNamespace(
                         *transitiveSeed),
                 "F-6 negative control: transitive reach to rvv.zve32f is NOT "
                 "independent"))
    return result;

  // Bare-prefix guard: "rvvish" is not in the rvv.* namespace; must stay
  // independent so the intersection test is a true namespace test, not substring.
  TargetCapabilitySet lookalikeRVV;
  if (int result = expectSuccess(
          lookalikeRVV.tryAddCapability(CapabilityDescriptor(
              "fake_scalar_lookalike", "scalar.fallback.lookalike", "fallback",
              "available", CapabilityAvailability::Available, /*properties=*/{},
              impliesOnly("rvvish"))),
          "add bare-prefix lookalike capability"))
    return result;
  const CapabilityDescriptor *lookalikeSeed =
      lookalikeRVV.lookupByID("scalar.fallback.lookalike");
  if (int result = expect(
          lookalikeSeed &&
              lookalikeRVV.impliedClosureAvoidsRVVNamespace(*lookalikeSeed),
          "F-6 guard: 'rvvish' is outside the rvv.* namespace (independent)"))
    return result;

  // --- conjunct (2): the scalar variant is only_feasible AND truly selected. ---
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(weft::plugin::registerScalarExtensionPlugin(
                            registry),
                        "register scalar fallback plugin for F-6 selection"))
    return result;

  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          weft::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize scalar fallback variant for F-6"))
    return result;
  if (int result = expect(materializedVariants.size() == 1,
                          "F-6 vector-absent instance materializes exactly one "
                          "scalar variant"))
    return result;
  VariantOp scalarVariant = materializedVariants.front();

  llvm::Expected<VariantSelectionPlan> planOrError =
      weft::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("F-6 scalar selection planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan selectionPlan = std::move(*planOrError);
  if (int result = expect(
          selectionPlan.kind == VariantSelectionKind::FallbackOnly &&
              selectionPlan.selectedVariant == scalarVariant &&
              selectionPlan.fallback == scalarVariant,
          "F-6 conjunct 2: the only-feasible scalar variant is actually "
          "selected (not a dead shell)"))
    return result;
  if (int result =
          expect(scalarVariant.getSymName() ==
                     weft::plugin::scalar::
                         getScalarFallbackFirstSliceVariantName(),
                 "F-6 selected variant is the canonical scalar fallback slice"))
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
          expectSuccess(weft::plugin::registerScalarExtensionPlugin(
                            dialectPlugins),
                        "register scalar fallback plugin for dialect context"))
    return result;
  weft::registerAllDialects(dialectRegistry);
  weft::registerPluginDialects(dialectPlugins, dialectRegistry);
  dialectRegistry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runProposalGatingTest(context))
    return result;
  if (int result = runMaterializationSelectionAndEmissionTest(context))
    return result;
  if (int result = runTypedFinalBodyConstructionTest(context))
    return result;
  if (int result = runInvalidTernaryGeometryConstructionTest(context))
    return result;
  if (int result = runBoundaryMaterializationRejectionTest(context))
    return result;
  if (int result =
          runRVVDeclineKeepsScalarFallbackEnvelopeBoundarylessTest(context))
    return result;
  if (int result = runLegalityRejectionTest(context))
    return result;
  if (int result = runFamilyIndependenceAcceptanceTest(context))
    return result;

  llvm::outs() << "scalar fallback extension plugin smoke test passed\n";
  return 0;
}
