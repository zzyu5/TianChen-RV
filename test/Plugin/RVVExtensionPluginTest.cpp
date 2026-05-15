#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <initializer_list>
#include <string>

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmissionStatus;
using tianchenrv::plugin::VariantLegalityRequest;
using tianchenrv::plugin::VariantLoweringBoundaryRequest;
using tianchenrv::plugin::VariantLoweringBoundaryResult;
using tianchenrv::plugin::VariantProposal;
using tianchenrv::plugin::VariantProposalDecline;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::plugin::rvv::RVVProbeCapabilityFacts;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

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
                        std::initializer_list<llvm::StringRef> fragments) {
  if (!error)
    return fail("expected error");
  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("error text missing '") + fragment +
                  "': " + message);
  }
  return 0;
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

KernelOp findKernel(mlir::ModuleOp module, llvm::StringRef name) {
  KernelOp kernel;
  module->walk([&](KernelOp candidate) {
    if (candidate.getSymName() == name)
      kernel = candidate;
  });
  return kernel;
}

VariantOp findVariant(KernelOp kernel, llvm::StringRef name) {
  VariantOp variant;
  if (!kernel || kernel.getBody().empty())
    return variant;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto candidate = llvm::dyn_cast<VariantOp>(op);
    if (candidate && candidate.getSymName() == name)
      variant = candidate;
  }
  return variant;
}

mlir::func::FuncOp findHighLevelPlaceholder(mlir::ModuleOp module) {
  return module.lookupSymbol<mlir::func::FuncOp>("high_level_placeholder");
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
  if (int result = expect(static_cast<bool>(attr),
                          llvm::Twine("proposal carries ") + attrName))
    return result;
  return expect(attr.getValue() == expectedValue,
                llvm::Twine("proposal preserves ") + attrName);
}

int expectProposalIntegerAttr(const VariantProposal &proposal,
                              llvm::StringRef attrName,
                              std::int64_t expectedValue) {
  auto attr = llvm::dyn_cast_if_present<mlir::IntegerAttr>(
      findProposalAttribute(proposal, attrName));
  if (int result = expect(static_cast<bool>(attr),
                          llvm::Twine("proposal carries ") + attrName))
    return result;
  return expect(attr.getInt() == expectedValue,
                llvm::Twine("proposal preserves ") + attrName);
}

bool proposalHasSelectedBinaryMetadata(const VariantProposal &proposal) {
  for (mlir::NamedAttribute attribute : proposal.getPluginAttributes()) {
    llvm::StringRef name = attribute.getName().getValue();
    if (name.contains("selected") && name.contains("_binary"))
      return true;
  }
  return false;
}

RVVProbeCapabilityFacts makeSuccessfulProbeFacts() {
  RVVProbeCapabilityFacts facts;
  facts.architecture = "riscv64";
  facts.hartCount = 64;
  facts.vlenbBytes = 16;
  facts.i32M1LaneCount = 4;
  facts.isaVectorHints = "rv64gcv_zvl128b";
  facts.clangAvailable = true;
  facts.clangVersion = "clang version 18.1.3";
  facts.cmakeAvailable = true;
  facts.cmakeVersion = "cmake version 3.28.3";
  facts.minimalRVVCompileRunSucceeded = true;
  facts.selectedMarch = "rv64gcv";
  facts.selectedMABI = "lp64d";
  return facts;
}

int runRegistrationAndCapabilityMetadataTest() {
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin"))
    return result;
  if (int result = expect(registry.size() == 1,
                          "RVV plugin registration adds one plugin"))
    return result;
  if (int result = expect(registry.lookupPlugin(
                              tianchenrv::plugin::rvv::
                                  getRVVExtensionPluginName()) != nullptr,
                          "registered RVV plugin is lookup-visible"))
    return result;

  llvm::SmallVector<PluginCapability, 16> capabilities;
  registry.collectCapabilities(capabilities);
  if (int result = expect(capabilities.size() == 14,
                          "RVV plugin exposes RVV/vector-shape capabilities"))
    return result;
  if (int result = expect(registry.lookupCapabilityByID("rvv") != nullptr,
                          "RVV capability id lookup succeeds"))
    return result;
  return expect(registry.lookupCapabilityByID(
                    tianchenrv::target::rvv::
                        getRVVI32BinarySelectedVectorShapeCapabilityID()) !=
                    nullptr,
                "RVV selected vector-shape selector capability lookup succeeds");
}

int runCapabilityProfileTest() {
  tianchenrv::plugin::rvv::RVVExtensionPlugin plugin;
  llvm::Expected<TargetCapabilitySet> capabilitiesOrError =
      plugin.buildTargetCapabilitiesFromProbeFacts(makeSuccessfulProbeFacts());
  if (!capabilitiesOrError)
    return fail("valid RVV probe facts were rejected: " +
                llvm::toString(capabilitiesOrError.takeError()));

  TargetCapabilitySet capabilities = std::move(*capabilitiesOrError);
  if (int result = expect(capabilities.isCapabilityAvailableByID("rvv"),
                          "RVV profile exposes available rvv capability"))
    return result;
  if (int result = expect(capabilities.isCapabilityAvailableByID(
                              tianchenrv::plugin::rvv::
                                  getRVVI32M1SEW32CapabilityID()),
                          "RVV profile exposes i32m1 SEW capability"))
    return result;
  return expect(capabilities.isCapabilityAvailableByID(
                    tianchenrv::plugin::rvv::
                        getRVVProbeCompileRunCapabilityID()),
                "RVV profile exposes compile/run capability");
}

int runMissingAndDeclineProposalTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @missing_rvv {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
  }

  tcrv.exec.kernel @missing_hart_count {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV decline module");
  mlir::func::FuncOp highLevel = findHighLevelPlaceholder(*module);

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for decline test"))
    return result;

  KernelOp missingRVV = findKernel(*module, "missing_rvv");
  TargetCapabilitySet missingCapabilities =
      TargetCapabilitySet::buildFromKernel(missingRVV);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result = expectSuccess(
          registry.collectVariantProposals(
              VariantProposalRequest(highLevel.getOperation(), missingRVV,
                                     missingCapabilities),
              proposals),
          "collect missing RVV proposals"))
    return result;
  if (int result = expect(proposals.empty(),
                          "RVV plugin proposes nothing without rvv"))
    return result;

  KernelOp missingHart = findKernel(*module, "missing_hart_count");
  TargetCapabilitySet hartCapabilities =
      TargetCapabilitySet::buildFromKernel(missingHart);
  llvm::SmallVector<VariantProposalDecline, 1> declines;
  if (int result = expectSuccess(
          registry.collectVariantProposals(
              VariantProposalRequest(highLevel.getOperation(), missingHart,
                                     hartCapabilities),
              proposals, &declines),
          "collect missing hart-count RVV proposal decline"))
    return result;
  if (int result = expect(proposals.empty() && !declines.empty(),
                          "missing hart count is a recoverable RVV decline"))
    return result;
  return expect(llvm::StringRef(declines.front().getReason())
                    .contains("rvv.hart_count"),
                "decline reason names the missing bounded capability");
}

int runProposalMaterializationAndBoundaryTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @rvv_metadata_only {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_vlenb {
      id = "rvv.vlenb_bytes",
      kind = "uarch",
      bytes = 16 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lanes {
      id = "rvv.i32_m1_lane_count",
      kind = "uarch",
      lanes = 4 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      value = "rv64gcv",
      status = "available"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV proposal module");
  KernelOp kernel = findKernel(*module, "rvv_metadata_only");
  mlir::func::FuncOp highLevel = findHighLevelPlaceholder(*module);
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for proposal test"))
    return result;

  VariantProposalRequest request(highLevel.getOperation(), kernel, capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result = expectSuccess(registry.collectVariantProposals(
                                     request, proposals),
                                 "collect RVV proposal"))
    return result;
  if (int result = expect(proposals.size() == 1,
                          "RVV capability facts produce one proposal"))
    return result;
  if (int result = expect(!proposalHasSelectedBinaryMetadata(proposals[0]),
                          "RVV proposal carries no selected-binary metadata"))
    return result;
  if (int result =
          expect(proposals[0].getRequiredCapabilityIDs().size() == 5,
                 "RVV proposal requires rvv plus selected vector-shape ids"))
    return result;
  if (int result =
          expectProposalStringAttr(proposals[0], "tcrv_rvv.required_march",
                                   "rv64gcv"))
    return result;
  if (int result = expectProposalStringAttr(
          proposals[0], "tcrv_rvv.selected_vector_shape", "i32m1"))
    return result;
  if (int result = expectProposalIntegerAttr(
          proposals[0], "tcrv_rvv.selected_vector_sew", 32))
    return result;
  if (int result = expectProposalIntegerAttr(
          proposals[0], "tcrv_rvv.vlenb_bytes", 16))
    return result;

  mlir::OpBuilder builder(module->getContext());
  llvm::SmallVector<VariantOp, 1> materialized;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeVariantProposals(
              builder, request, proposals, &materialized),
          "materialize RVV proposal"))
    return result;
  if (int result = expect(materialized.size() == 1,
                          "one RVV variant was materialized"))
    return result;
  VariantOp variant = materialized.front();

  if (int result =
          expectSuccess(registry.verifyVariantLegality(
                            VariantLegalityRequest(variant, kernel,
                                                   capabilities)),
                        "verify RVV metadata-only variant legality"))
    return result;

  VariantEmissionStatus readiness;
  if (int result = expectErrorContains(
          registry.checkVariantEmissionReadiness(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              readiness),
          {"unsupported emission path", "no materialized EmitC lowering"}))
    return result;

  VariantEmissionPlan plan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              plan),
          "build RVV emission plan"))
    return result;
  if (int result = expect(plan.isUnsupported(),
                          "RVV emission plan is unsupported"))
    return result;
  if (int result = expect(plan.getRuntimeABIKind() ==
                              "unsupported-plugin-runtime-abi",
                          "RVV emission plan has unsupported ABI kind"))
    return result;
  for (const auto &metadata : plan.getSelectedPlanMetadata()) {
    llvm::StringRef name(metadata.name);
    if (name.contains("selected") && name.contains("_binary"))
      return fail("RVV emission plan retained selected-binary metadata");
  }

  VariantLoweringBoundaryResult boundaryResult;
  builder.setInsertionPointToEnd(&kernel.getBody().front());
  if (int result = expectSuccess(
          registry.materializeSelectedLoweringBoundary(
              VariantLoweringBoundaryRequest(
                  variant, kernel, capabilities,
                  VariantEmissionRole::DirectVariant, builder),
              boundaryResult),
          "materialize RVV unsupported lowering boundary"))
    return result;
  if (int result = expect(boundaryResult.isMaterialized(),
                          "RVV lowering boundary materializes metadata op"))
    return result;
  auto boundary = llvm::dyn_cast<tianchenrv::tcrv::rvv::LoweringBoundaryOp>(
      boundaryResult.getMaterializedOperation());
  if (int result = expect(static_cast<bool>(boundary),
                          "RVV boundary is tcrv_rvv.lowering_boundary"))
    return result;
  if (int result =
          expectSuccess(registry.validateSelectedLoweringBoundary(
                            tianchenrv::plugin::
                                VariantLoweringBoundaryValidationRequest(
                                    variant, kernel, capabilities,
                                    VariantEmissionRole::DirectVariant,
                                    boundary.getOperation())),
                        "validate RVV lowering boundary"))
    return result;

  return 0;
}

int runPolicyLegalityRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @bad_policy {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = "not_a_typed_policy",
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_vector_shape = "i32m1",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m1_t",
      tcrv_rvv.selected_vector_suffix = "i32m1",
      tcrv_rvv.selected_setvl_suffix = "e32m1"
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV policy rejection module");
  KernelOp kernel = findKernel(*module, "bad_policy");
  VariantOp variant = findVariant(kernel, "rvv_first_slice");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  tianchenrv::plugin::rvv::RVVExtensionPlugin plugin;
  return expectErrorContains(
      plugin.verifyVariantLegality(
          VariantLegalityRequest(variant, kernel, capabilities)),
      {"tcrv_rvv.policy", "requires typed"});
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runRegistrationAndCapabilityMetadataTest())
    return result;
  if (int result = runCapabilityProfileTest())
    return result;
  if (int result = runMissingAndDeclineProposalTest(context))
    return result;
  if (int result = runProposalMaterializationAndBoundaryTest(context))
    return result;
  if (int result = runPolicyLegalityRejectionTest(context))
    return result;

  llvm::outs() << "RVV extension plugin smoke test passed\n";
  return 0;
}
