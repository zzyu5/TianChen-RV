#include "Weft/InitWeftDialects.h"
#include "Weft/Dialect/Offload/IR/OffloadDialect.h"
#include "Weft/Plugin/BuiltinExtensionPlugins.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/Offload/OffloadExtensionPlugin.h"
#include "Weft/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Support/RuntimeABI.h"
#include "Weft/Transforms/EmissionReadiness.h"
#include "Weft/Transforms/VariantMaterialization.h"
#include "Weft/Transforms/VariantSelection.h"

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

using weft::plugin::ExtensionPluginRegistry;
using weft::plugin::ExtensionBundleRegistry;
using weft::plugin::PluginCapability;
using weft::plugin::VariantCostEstimate;
using weft::plugin::VariantCostRequest;
using weft::plugin::VariantEmissionPlan;
using weft::plugin::VariantEmissionRequest;
using weft::plugin::VariantEmissionRole;
using weft::plugin::VariantEmissionStatus;
using weft::plugin::VariantLoweringBoundaryResult;
using weft::plugin::VariantProposal;
using weft::plugin::VariantProposalDecline;
using weft::plugin::VariantProposalRequest;
using weft::support::TargetCapabilitySet;
using weft::exec::DiagnosticOp;
using weft::exec::KernelOp;
using weft::exec::VariantOp;
using weft::offload::LoweringBoundaryOp;
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
          expectSuccess(weft::plugin::registerOffloadExtensionPlugin(
                            registry),
                        "register offload plugin"))
    return result;

  const auto *plugin = registry.lookupPlugin(
      weft::plugin::offload::getOffloadExtensionPluginName());
  if (int result = expect(plugin, "registered offload plugin is visible"))
    return result;
  if (int result =
          expect(plugin->getVersion() ==
                     weft::plugin::offload::
                         getOffloadExtensionPluginVersion(),
                 "offload plugin version is stable"))
    return result;

  const PluginCapability *capability = registry.lookupCapabilityByID(
      weft::plugin::offload::getOffloadRuntimeCapabilityID());
  if (int result =
          expect(capability &&
                     capability->getKind() ==
                         weft::plugin::offload::
                             getOffloadRuntimeCapabilityKind(),
                 "offload runtime capability metadata is registered"))
    return result;

  return expectErrorContains(
      weft::plugin::registerOffloadExtensionPlugin(registry),
      {"duplicate Weft-RV extension plugin", "offload-plugin"});
}

int runProposalGatingAndDeclineTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @available_offload attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
  }

  weft.exec.kernel @missing_offload attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }

  weft.exec.kernel @malformed_offload attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "sophgo-vendor-runtime",
      handoff_kind = "runtime-offload"
    }
  }

  weft.exec.kernel @misclassified_custom_isa_offload attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "custom-isa",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
  }

  weft.exec.kernel @vendor_string_only attributes {construction_domain = "riscv-execution", problem = @canonical_problem, vendor_hint = "sophgo"} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @vendor_runtime {
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

  KernelOp available = findKernel(*module, "available_offload");
  KernelOp missing = findKernel(*module, "missing_offload");
  KernelOp malformed = findKernel(*module, "malformed_offload");
  KernelOp misclassified =
      findKernel(*module, "misclassified_custom_isa_offload");
  KernelOp vendorOnly = findKernel(*module, "vendor_string_only");
  if (int result = expect(available && missing && malformed && misclassified &&
                              vendorOnly,
                          "proposal gating module contains all anchors"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(weft::plugin::registerOffloadExtensionPlugin(
                            registry),
                        "register offload plugin for proposal gating"))
    return result;

  TargetCapabilitySet availableCapabilities =
      TargetCapabilitySet::buildFromKernel(available);
  mlir::Operation *availableProblem = resolveExactProblem(available);
  if (int result =
          expect(availableProblem, "available Offload kernel binds exact problem"))
    return result;
  VariantProposalRequest availableRequest(availableProblem, available,
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
                         weft::plugin::offload::
                             getOffloadRuntimeFirstSliceVariantName() &&
                     proposal.getOriginPlugin() ==
                         weft::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     proposal.getPolicy() ==
                         weft::plugin::offload::
                             getOffloadFirstSlicePolicy(),
                 "offload proposal preserves stable generic metadata"))
    return result;
  if (int result =
          expect(proposal.getRequiredCapabilityIDs().size() == 1 &&
                     proposal.getRequiredCapabilityIDs().front() ==
                         weft::plugin::offload::
                             getOffloadRuntimeCapabilityID(),
                 "offload proposal requires offload.runtime capability id"))
    return result;
  if (int result = expectProposalStringAttr(
          proposal,
          weft::plugin::offload::getOffloadRuntimeABIAttrName(),
          weft::plugin::offload::getOffloadExpectedRuntimeABI()))
    return result;
  if (int result = expectProposalStringAttr(
          proposal,
          weft::plugin::offload::getOffloadHandoffKindAttrName(),
          weft::plugin::offload::getOffloadExpectedHandoffKind()))
    return result;

  TargetCapabilitySet missingCapabilities =
      TargetCapabilitySet::buildFromKernel(missing);
  mlir::Operation *missingProblem = resolveExactProblem(missing);
  if (int result =
          expect(missingProblem, "missing Offload kernel binds exact problem"))
    return result;
  VariantProposalRequest missingRequest(missingProblem, missing,
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
  mlir::Operation *malformedProblem = resolveExactProblem(malformed);
  if (int result = expect(
          malformedProblem, "malformed Offload kernel binds exact problem"))
    return result;
  VariantProposalRequest malformedRequest(malformedProblem, malformed,
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
                         weft::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     declines.front().getReason().contains("runtime_abi"),
                 "malformed offload capability records plugin-local decline"))
    return result;

  TargetCapabilitySet misclassifiedCapabilities =
      TargetCapabilitySet::buildFromKernel(misclassified);
  mlir::Operation *misclassifiedProblem = resolveExactProblem(misclassified);
  if (int result = expect(
          misclassifiedProblem,
          "misclassified Offload kernel binds exact problem"))
    return result;
  VariantProposalRequest misclassifiedRequest(
      misclassifiedProblem, misclassified, misclassifiedCapabilities);
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
                         weft::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     declines.front().getReason().contains(
                         "kind must be 'runtime-offload'"),
                 "offload.runtime modeled as custom ISA records "
                 "plugin-local decline"))
    return result;

  TargetCapabilitySet vendorOnlyCapabilities =
      TargetCapabilitySet::buildFromKernel(vendorOnly);
  mlir::Operation *vendorOnlyProblem = resolveExactProblem(vendorOnly);
  if (int result = expect(
          vendorOnlyProblem, "vendor-only Offload kernel binds exact problem"))
    return result;
  VariantProposalRequest vendorOnlyRequest(vendorOnlyProblem, vendorOnly,
                                           vendorOnlyCapabilities);
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
  weft.exec.kernel @offload_plus_scalar attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.dequantize_row_q4_0_problem @canonical_problem {qk = 32 : i64, weight_block_stride = 18 : i64, weight_d_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    weft.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    weft.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    weft.exec.runtime_param @abi_runtime_element_count {
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

  KernelOp kernel = findKernel(*module, "offload_plus_scalar");
  if (int result = expect(kernel, "materialization module has kernel anchor"))
    return result;

  ExtensionBundleRegistry bundles;
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(
              weft::plugin::registerBuiltinExtensionBundlePlugins(
                  bundles, registry),
              "register built-in extension bundle frontdoor for offload "
              "materialization"))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  mlir::Operation *problem = resolveExactProblem(kernel);
  if (int result =
          expect(problem, "Offload materialization kernel binds exact problem"))
    return result;
  VariantProposalRequest request(problem, kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 2> materializedVariants;
  if (int result = expectSuccess(
          weft::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize offload and scalar proposals"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 2,
                 "offload plus scalar materializes two variants"))
    return result;

  VariantOp offloadVariant = findVariant(
      kernel, weft::plugin::offload::
                  getOffloadRuntimeFirstSliceVariantName());
  VariantOp scalarVariant = findVariant(
      kernel, weft::plugin::scalar::
                  getScalarFallbackFirstSliceVariantName());
  if (int result = expect(offloadVariant && scalarVariant,
                          "offload and scalar variants are materialized"))
    return result;

  if (int result =
          expect(offloadVariant->getAttrOfType<mlir::StringAttr>("origin")
                         .getValue() ==
                     weft::plugin::offload::
                         getOffloadExtensionPluginName(),
                 "offload variant has offload origin"))
    return result;
  if (int result =
          expect(offloadVariant
                         ->getAttrOfType<mlir::StringAttr>(
                             weft::plugin::offload::
                                 getOffloadRuntimeABIAttrName())
                         .getValue() ==
                     weft::plugin::offload::
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
              VariantCostRequest(offloadVariant, kernel, problem,
                                 capabilities),
              estimate),
          "offload cost estimate routes through plugin"))
    return result;
  if (int result =
          expect(estimate.hasScore() && estimate.getScore() == 10.0 &&
                     estimate.hasExplicitPreference() &&
                     estimate.getOriginPlugin() ==
                         weft::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     estimate.getVariantSymbol() == offloadVariant.getSymName(),
                 "offload cost metadata is plugin-owned"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      weft::transforms::planKernelVariantSelection(kernel, capabilities,
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
          weft::transforms::materializeSelectedVariantMarker(
              builder, selectionPlan, &marker),
          "materialize offload selected marker"))
    return result;
  if (int result = expect(marker, "selected marker was created"))
    return result;

  if (int result = expectErrorContains(
          weft::plugin::materializeSelectedLoweringBoundaries(
              kernel, capabilities, registry),
          {"selected owner did not construct an executable final body before "
           "boundary exposure",
           "offload delegation plan has no executable implementation"}))
    return result;

  LoweringBoundaryOp offloadBoundary =
      findOffloadBoundary(kernel, offloadVariant.getSymName());
  if (int result = expect(
          offloadBoundary &&
              offloadBoundary.getSourceKernel() == kernel.getSymName() &&
              offloadBoundary.getOrigin() ==
                  weft::plugin::offload::getOffloadExtensionPluginName() &&
              offloadBoundary.getRole() == "direct variant" &&
              offloadBoundary.getStatus() == "no-active-route" &&
              offloadBoundary.getRuntimeAbi() ==
                  weft::plugin::offload::getOffloadExpectedRuntimeABI() &&
              offloadBoundary.getHandoffKind() == "runtime-offload" &&
              offloadBoundary.getHandoffReason().has_value() &&
              offloadBoundary.getHandoffReason()->contains(
                  "no executable external implementation"),
          "unsupported Offload construction preserves its typed diagnostic "
          "carrier while boundary exposure fails closed"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "fail-closed Offload boundary module verifies"))
    return result;

  VariantEmissionStatus status;
  if (int result = expectErrorContains(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(offloadVariant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              status),
          {"reported unsupported emission path",
           "no active materialized EmitC"}))
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
                         weft::plugin::offload::
                             getOffloadExtensionPluginName() &&
                     emissionPlan.getKernelSymbol() == kernel.getSymName() &&
                     emissionPlan.getVariantSymbol() ==
                         offloadVariant.getSymName() &&
                     emissionPlan.getLoweringPipeline().empty() &&
                     emissionPlan.getArtifactKind().empty() &&
                     emissionPlan.getRuntimeABIParameters().empty() &&
                     emissionPlan.getDiagnostic().contains(
                         "no active executable lowering") &&
                     emissionPlan.getRequiredCapabilitySymbols().size() == 1 &&
                     emissionPlan.getRequiredCapabilitySymbols().front() ==
                         weft::plugin::offload::
                             getOffloadRuntimePreferredCapabilitySymbol(),
                 "offload emission plan fails closed without executable target "
                 "route"))
    return result;

  if (int result = expectSuccess(
          weft::transforms::materializeKernelEmissionPlanDiagnostics(
              kernel, capabilities, registry),
          "materialize fail-closed offload emission diagnostic"))
    return result;
  bool foundUnsupportedOffloadPlan = false;
  kernel.walk([&](DiagnosticOp diagnostic) {
    auto reason =
        diagnostic->getAttrOfType<mlir::StringAttr>("reason");
    auto status =
        diagnostic->getAttrOfType<mlir::StringAttr>("status");
    auto target = diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>("target");
    if (reason && reason.getValue() == "emission_plan" && status &&
        status.getValue() == "unsupported" && target &&
        target.getValue() == offloadVariant.getSymName())
      foundUnsupportedOffloadPlan = true;
  });
  if (int result = expect(
          foundUnsupportedOffloadPlan,
          "offload delegation plan routes to an explicit unsupported "
          "emission diagnostic"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "offload emission-plan module verifies"))
    return result;

  return 0;
}

int runLegalityRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @offload_custom_isa_misclassification_rejected attributes {problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "custom-isa",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    weft.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      weft_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      weft_offload.handoff_kind = "runtime-offload"
    } {
    }
  }

  weft.exec.kernel @offload_legality_rejections attributes {problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @missing_offload_requirement attributes {
      origin = "offload-plugin",
      requires = [@scalar_fallback],
      weft_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      weft_offload.handoff_kind = "runtime-offload"
    } {
    }
    weft.exec.variant @missing_runtime_abi_metadata attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      weft_offload.handoff_kind = "runtime-offload"
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
          expectSuccess(weft::plugin::registerOffloadExtensionPlugin(
                            registry),
                        "register offload plugin for legality negatives"))
    return result;

  TargetCapabilitySet customISACapabilities =
      TargetCapabilitySet::buildFromKernel(customISA);
  mlir::Operation *customISAProblem = resolveExactProblem(customISA);
  if (int result =
          expect(customISAProblem, "custom-ISA kernel binds exact problem"))
    return result;
  if (int result = expectErrorContains(
          registry.verifyVariantLegality(
              weft::plugin::VariantLegalityRequest(
                  customISAVariant, customISA, customISAProblem,
                  customISACapabilities)),
          {"runtime-offload", "kind must be 'runtime-offload'"}))
    return result;

  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  mlir::Operation *problem = resolveExactProblem(kernel);
  if (int result =
          expect(problem, "Offload legality kernel binds exact problem"))
    return result;
  if (int result = expectErrorContains(
          registry.verifyVariantLegality(
              weft::plugin::VariantLegalityRequest(
                  missingRequirement, kernel, problem, capabilities)),
          {"runtime-offload", "must require capability id",
           "offload.runtime"}))
    return result;

  return expectErrorContains(
      registry.verifyVariantLegality(
          weft::plugin::VariantLegalityRequest(missingABI, kernel, problem,
                                               capabilities)),
      {"runtime-offload", "weft_offload.runtime_abi"});
}

} // namespace

int main() {
  if (int result = runRegistrationAndCapabilityMetadataTest())
    return result;

  mlir::DialectRegistry dialectRegistry;
  ExtensionBundleRegistry dialectBundles;
  ExtensionPluginRegistry dialectPlugins;
  if (int result =
          expectSuccess(
              weft::plugin::registerBuiltinExtensionBundlePlugins(
                  dialectBundles, dialectPlugins),
              "register built-in extension bundle frontdoor for dialect "
              "context"))
    return result;
  weft::registerAllDialects(dialectRegistry);
  weft::registerPluginDialects(dialectPlugins, dialectRegistry);

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
