#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Offload/IR/OffloadDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Transforms/EmissionReadiness.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"
#include "TianChenRV/Transforms/VariantSelection.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
using tianchenrv::plugin::VariantLoweringBoundaryResult;
using tianchenrv::plugin::VariantProposal;
using tianchenrv::plugin::VariantProposalDecline;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::offload::LoweringBoundaryOp;
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
  return module.lookupSymbol<mlir::func::FuncOp>("high_level_placeholder");
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

LoweringBoundaryOp findOffloadBoundary(KernelOp kernel,
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

int runRegistrationAndCapabilityMetadataTest() {
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerOffloadExtensionPlugin(
                            registry),
                        "register offload plugin"))
    return result;

  const auto *plugin = registry.lookupPlugin(
      tianchenrv::plugin::offload::getOffloadExtensionPluginName());
  if (int result = expect(plugin, "registered offload plugin is visible"))
    return result;
  if (int result =
          expect(plugin->getVersion() ==
                     tianchenrv::plugin::offload::
                         getOffloadExtensionPluginVersion(),
                 "offload plugin version is stable"))
    return result;

  const PluginCapability *capability = registry.lookupCapabilityByID(
      tianchenrv::plugin::offload::getOffloadRuntimeCapabilityID());
  if (int result =
          expect(capability &&
                     capability->getKind() ==
                         tianchenrv::plugin::offload::
                             getOffloadRuntimeCapabilityKind(),
                 "offload runtime capability metadata is registered"))
    return result;

  return expectErrorContains(
      tianchenrv::plugin::registerOffloadExtensionPlugin(registry),
      {"duplicate TianChen-RV extension plugin", "offload-plugin"});
}

int runProposalGatingAndDeclineTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @available_offload attributes {} {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
  }

  tcrv.exec.kernel @missing_offload attributes {} {
  }

  tcrv.exec.kernel @malformed_offload attributes {} {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "sophgo-vendor-runtime",
      handoff_kind = "runtime-offload"
    }
  }

  tcrv.exec.kernel @misclassified_custom_isa_offload attributes {} {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "custom-isa",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
  }

  tcrv.exec.kernel @vendor_string_only attributes {vendor_hint = "sophgo"} {
    tcrv.exec.capability @vendor_runtime {
      id = "sophgo.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse offload proposal gating module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp available = findKernel(*module, "available_offload");
  KernelOp missing = findKernel(*module, "missing_offload");
  KernelOp malformed = findKernel(*module, "malformed_offload");
  KernelOp misclassified =
      findKernel(*module, "misclassified_custom_isa_offload");
  KernelOp vendorOnly = findKernel(*module, "vendor_string_only");
  if (int result = expect(highLevelOp && available && missing && malformed &&
                              misclassified && vendorOnly,
                          "proposal gating module contains all anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerOffloadExtensionPlugin(
                            registry),
                        "register offload plugin for proposal gating"))
    return result;

  TargetCapabilitySet availableCapabilities =
      TargetCapabilitySet::buildFromKernel(available);
  VariantProposalRequest availableRequest(highLevelOp.getOperation(), available,
                                          availableCapabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  llvm::SmallVector<VariantProposalDecline, 1> declines;
  if (int result = expectSuccess(
          registry.collectVariantProposals(availableRequest, proposals,
                                           &declines),
          "available offload runtime capability collects proposal"))
    return result;
  if (int result =
          expect(proposals.size() == 1 && declines.empty(),
                 "available offload runtime capability proposes one variant"))
    return result;
  const VariantProposal &proposal = proposals.front();
  if (int result =
          expect(proposal.getVariantName() ==
                         tianchenrv::plugin::offload::
                             getOffloadRuntimeFirstSliceVariantName() &&
                     proposal.getOriginPlugin() ==
                         tianchenrv::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     proposal.getPolicy() ==
                         tianchenrv::plugin::offload::
                             getOffloadFirstSlicePolicy(),
                 "offload proposal preserves stable generic metadata"))
    return result;
  if (int result =
          expect(proposal.getRequiredCapabilityIDs().size() == 1 &&
                     proposal.getRequiredCapabilityIDs().front() ==
                         tianchenrv::plugin::offload::
                             getOffloadRuntimeCapabilityID(),
                 "offload proposal requires offload.runtime capability id"))
    return result;
  if (int result = expectProposalStringAttr(
          proposal,
          tianchenrv::plugin::offload::getOffloadRuntimeABIAttrName(),
          tianchenrv::plugin::offload::getOffloadExpectedRuntimeABI()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal,
          tianchenrv::plugin::offload::getOffloadHandoffKindAttrName(),
          tianchenrv::plugin::offload::getOffloadExpectedHandoffKind()))
    return result;

  TargetCapabilitySet missingCapabilities =
      TargetCapabilitySet::buildFromKernel(missing);
  VariantProposalRequest missingRequest(highLevelOp.getOperation(), missing,
                                        missingCapabilities);
  proposals.clear();
  declines.clear();
  if (int result = expectSuccess(
          registry.collectVariantProposals(missingRequest, proposals,
                                           &declines),
          "missing offload capability query succeeds"))
    return result;
  if (int result =
          expect(proposals.empty() && declines.empty(),
                 "missing offload capability produces no proposal or decline"))
    return result;

  TargetCapabilitySet malformedCapabilities =
      TargetCapabilitySet::buildFromKernel(malformed);
  VariantProposalRequest malformedRequest(highLevelOp.getOperation(), malformed,
                                          malformedCapabilities);
  proposals.clear();
  declines.clear();
  if (int result = expectSuccess(
          registry.collectVariantProposals(malformedRequest, proposals,
                                           &declines),
          "malformed offload capability decline is recoverable"))
    return result;
  if (int result =
          expect(proposals.empty() && declines.size() == 1 &&
                     declines.front().getPluginName() ==
                         tianchenrv::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     declines.front().getReason().contains("runtime_abi"),
                 "malformed offload capability records plugin-local decline"))
    return result;

  TargetCapabilitySet misclassifiedCapabilities =
      TargetCapabilitySet::buildFromKernel(misclassified);
  VariantProposalRequest misclassifiedRequest(
      highLevelOp.getOperation(), misclassified, misclassifiedCapabilities);
  proposals.clear();
  declines.clear();
  if (int result = expectSuccess(
          registry.collectVariantProposals(misclassifiedRequest, proposals,
                                           &declines),
          "custom-ISA offload capability misclassification decline is "
          "recoverable"))
    return result;
  if (int result =
          expect(proposals.empty() && declines.size() == 1 &&
                     declines.front().getPluginName() ==
                         tianchenrv::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     declines.front().getReason().contains(
                         "kind must be 'runtime-offload'"),
                 "offload.runtime modeled as custom ISA records "
                 "plugin-local decline"))
    return result;

  TargetCapabilitySet vendorOnlyCapabilities =
      TargetCapabilitySet::buildFromKernel(vendorOnly);
  VariantProposalRequest vendorOnlyRequest(highLevelOp.getOperation(),
                                           vendorOnly, vendorOnlyCapabilities);
  proposals.clear();
  declines.clear();
  if (int result = expectSuccess(
          registry.collectVariantProposals(vendorOnlyRequest, proposals,
                                           &declines),
          "vendor string only query succeeds"))
    return result;
  return expect(proposals.empty() && declines.empty(),
                "vendor/Sophgo strings do not trigger offload support without "
                "explicit offload.runtime capability");
}

int runMaterializationSelectionAndEmissionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @offload_plus_scalar attributes {} {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "n",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse offload materialization module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "offload_plus_scalar");
  if (int result =
          expect(highLevelOp && kernel, "materialization module has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerBuiltinExtensionPlugins(
                            registry),
                        "register built-in plugins for offload materialization"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 2> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize offload and scalar proposals"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 2,
                 "offload plus scalar materializes two variants"))
    return result;

  VariantOp offloadVariant = findVariant(
      kernel, tianchenrv::plugin::offload::
                  getOffloadRuntimeFirstSliceVariantName());
  VariantOp scalarVariant = findVariant(
      kernel, tianchenrv::plugin::scalar::
                  getScalarFallbackFirstSliceVariantName());
  if (int result = expect(offloadVariant && scalarVariant,
                          "offload and scalar variants are materialized"))
    return result;

  if (int result =
          expect(offloadVariant->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     tianchenrv::plugin::offload::
                         getOffloadExtensionPluginName(),
                 "offload variant has offload origin"))
    return result;
  if (int result =
          expect(offloadVariant
                         ->getAttrOfType<mlir::StringAttr>(
                             tianchenrv::plugin::offload::
                                 getOffloadRuntimeABIAttrName())
                         .getValue() ==
                     tianchenrv::plugin::offload::
                         getOffloadExpectedRuntimeABI(),
                 "offload variant carries runtime ABI metadata"))
    return result;

  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "materialized offload module verifies"))
    return result;
  if (int result =
          expectSuccess(registry.verifyKernelVariantLegality(kernel,
                                                             capabilities),
                        "offload legality accepts materialized variant"))
    return result;

  VariantCostEstimate estimate;
  if (int result = expectSuccess(
          registry.estimateVariantCost(
              VariantCostRequest(offloadVariant, kernel, capabilities),
              estimate),
          "offload cost estimate routes through plugin"))
    return result;
  if (int result =
          expect(estimate.hasScore() && estimate.getScore() == 10.0 &&
                     estimate.hasExplicitPreference() &&
                     estimate.getOriginPlugin() ==
                         tianchenrv::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     estimate.getVariantSymbol() == offloadVariant.getSymName(),
                 "offload cost metadata is plugin-owned"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("offload selection planning failed: " +
                llvm::toString(planOrError.takeError()));
  VariantSelectionPlan selectionPlan = std::move(*planOrError);
  if (int result =
          expect(selectionPlan.kind == VariantSelectionKind::StaticVariant &&
                     selectionPlan.selectedVariant == offloadVariant &&
                     selectionPlan.fallback == scalarVariant &&
                     selectionPlan.dispatchCases.empty(),
                 "offload selected path keeps printable metadata static while "
                 "recording scalar fallback coverage"))
    return result;

  DiagnosticOp marker;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize offload selected marker"))
    return result;
  if (int result = expect(marker, "selected marker was created"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::plugin::materializeSelectedLoweringBoundaries(
              kernel, capabilities, registry),
          "materialize offload and scalar selected boundaries"))
    return result;

  LoweringBoundaryOp offloadBoundary =
      findOffloadBoundary(kernel, offloadVariant.getSymName());
  if (int result =
          expect(offloadBoundary,
                 "offload selected boundary is materialized through plugin"))
    return result;
  if (int result =
          expect(offloadBoundary->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     tianchenrv::plugin::offload::
                         getOffloadExtensionPluginName() &&
                     offloadBoundary->getAttrOfType<mlir::StringAttr>("role")
                         .getValue() == "direct variant" &&
                     offloadBoundary->getAttrOfType<mlir::StringAttr>("status")
                         .getValue() == "metadata-only" &&
                     offloadBoundary
                             ->getAttrOfType<mlir::StringAttr>("runtime_abi")
                             .getValue() ==
                         tianchenrv::plugin::offload::
                             getOffloadExpectedRuntimeABI(),
                 "offload boundary records metadata-only runtime handoff"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "offload boundary module verifies"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectSuccess(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(offloadVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          "offload emission readiness is plugin-owned"))
    return result;
  if (int result =
          expect(status.isMetadataOnly() &&
                     status.getEmissionPath().contains("offload-handoff"),
                 "offload readiness reports metadata-only handoff route"))
    return result;

  VariantEmissionPlan emissionPlan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(offloadVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              emissionPlan),
          "offload emission plan is plugin-owned"))
    return result;
  if (int result =
          expect(emissionPlan.isUnsupported() &&
                     emissionPlan.getOriginPlugin() ==
                         tianchenrv::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     emissionPlan.getKernelSymbol() == kernel.getSymName() &&
                     emissionPlan.getVariantSymbol() ==
                         offloadVariant.getSymName() &&
                     emissionPlan.getLoweringPipeline().empty() &&
                     emissionPlan.getArtifactKind().empty() &&
                     emissionPlan.getRuntimeABIParameters().empty() &&
                     emissionPlan.getSelectedPlanMetadata().empty() &&
                     emissionPlan.getDiagnostic().contains(
                         "descriptor artifact export has been deleted") &&
                     emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                     emissionPlan.getRequiredCapabilitySymbols().front() ==
                         tianchenrv::plugin::offload::
                             getOffloadRuntimePreferredCapabilitySymbol(),
                 "offload emission plan fails closed after descriptor route "
                 "deletion"))
    return result;

  if (int result = expectSuccess(
          tianchenrv::transforms::materializeKernelEmissionPlanDiagnostics(
              kernel, capabilities, registry),
          "materialize offload emission-plan diagnostics"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "offload emission-plan module verifies"))
    return result;

  return 0;
}

int runLegalityRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @offload_custom_isa_misclassification_rejected {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "custom-isa",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
  }

  tcrv.exec.kernel @offload_legality_rejections attributes {} {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @missing_offload_requirement attributes {
      origin = "offload-plugin",
      requires = [@scalar_fallback],
      tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
    tcrv.exec.variant @missing_runtime_abi_metadata attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse offload legality rejection module");

  KernelOp kernel = findKernel(*module, "offload_legality_rejections");
  KernelOp customISA =
      findKernel(*module, "offload_custom_isa_misclassification_rejected");
  VariantOp missingRequirement =
      findVariant(kernel, "missing_offload_requirement");
  VariantOp missingABI =
      findVariant(kernel, "missing_runtime_abi_metadata");
  VariantOp customISAVariant =
      findVariant(customISA, "offload_runtime_first_slice");
  if (int result = expect(kernel && customISA && missingRequirement &&
                              missingABI && customISAVariant,
                          "legality rejection module has anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerOffloadExtensionPlugin(
                            registry),
                        "register offload plugin for legality negatives"))
    return result;

  TargetCapabilitySet customISACapabilities =
      TargetCapabilitySet::buildFromKernel(customISA);
  if (int result = expectErrorContains(
          registry.verifyVariantLegality(
              tianchenrv::plugin::VariantLegalityRequest(
                  customISAVariant, customISA, customISACapabilities)),
          {"runtime-offload", "kind must be 'runtime-offload'"}))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  if (int result = expectErrorContains(
          registry.verifyVariantLegality(
              tianchenrv::plugin::VariantLegalityRequest(
                  missingRequirement, kernel, capabilities)),
          {"runtime-offload", "must require capability id",
           "offload.runtime"}))
    return result;

  return expectErrorContains(
      registry.verifyVariantLegality(
          tianchenrv::plugin::VariantLegalityRequest(missingABI, kernel,
                                                     capabilities)),
      {"runtime-offload", "tcrv_offload.runtime_abi"});
}

} // namespace

int main() {
  if (int result = runRegistrationAndCapabilityMetadataTest())
    return result;

  mlir::DialectRegistry dialectRegistry;
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerBuiltinExtensionPlugins(
                            dialectPlugins),
                        "register built-in plugins for dialect context"))
    return result;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(dialectPlugins, dialectRegistry);
  dialectRegistry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runProposalGatingAndDeclineTest(context))
    return result;
  if (int result = runMaterializationSelectionAndEmissionTest(context))
    return result;
  if (int result = runLegalityRejectionTest(context))
    return result;

  llvm::outs() << "runtime-offload extension plugin smoke test passed\n";
  return 0;
}
