#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
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
using tianchenrv::plugin::VariantLegalityRequest;
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
  if (int result = expect(capabilities.size() == 1,
                          "RVV plugin exposes only the base RVV capability"))
    return result;
  return expect(registry.lookupCapabilityByID("rvv") != nullptr,
                "RVV capability id lookup succeeds");
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
  llvm::SmallVector<const tianchenrv::support::CapabilityDescriptor *, 1>
      configCapabilities;
  capabilities.collectByKind("isa-vector-config", configCapabilities);
  if (int result = expect(configCapabilities.empty(),
                          "RVV probe profile does not manufacture vector "
                          "config capabilities"))
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

  KernelOp availableRVVNoBody = findKernel(*module, "missing_hart_count");
  TargetCapabilitySet availableRVVCapabilities =
      TargetCapabilitySet::buildFromKernel(availableRVVNoBody);
  llvm::SmallVector<VariantProposalDecline, 1> declines;
  if (int result = expectSuccess(
          registry.collectVariantProposals(
              VariantProposalRequest(highLevel.getOperation(),
                                     availableRVVNoBody,
                                     availableRVVCapabilities),
              proposals, &declines),
          "collect RVV no-body proposal decline"))
    return result;
  if (int result = expect(proposals.empty() && !declines.empty(),
                          "available RVV no-body input is a recoverable RVV "
                          "decline"))
    return result;
  return expect(llvm::StringRef(declines.front().getReason())
                    .contains("explicit typed tcrv_rvv extension-family IR"),
                "decline reason explains explicit typed RVV IR requirement");
}

int runCapabilityOnlyProposalAndTypedBodyRequirementTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @rvv_no_body_capability_only {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
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

  tcrv.exec.kernel @rvv_missing_typed_body_variant {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_missing_typed_body attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV capability-only module");
  KernelOp kernel = findKernel(*module, "rvv_no_body_capability_only");
  mlir::func::FuncOp highLevel = findHighLevelPlaceholder(*module);
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for typed-body requirement test"))
    return result;

  VariantProposalRequest request(highLevel.getOperation(), kernel, capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  llvm::SmallVector<VariantProposalDecline, 1> declines;
  if (int result = expectSuccess(
          registry.collectVariantProposals(request, proposals, &declines),
          "collect RVV capability-only proposal surface"))
    return result;
  if (int result =
          expect(proposals.empty(), "RVV no-body capability produces no proposal"))
    return result;
  if (int result =
          expect(!declines.empty() &&
                     llvm::StringRef(declines.front().getReason())
                         .contains("explicit typed tcrv_rvv "
                                   "extension-family IR"),
                 "RVV decline explains explicit typed IR requirement"))
    return result;

  mlir::OpBuilder builder(module->getContext());
  llvm::SmallVector<VariantOp, 1> materialized;
  if (int result =
          expectErrorContains(
              tianchenrv::transforms::collectAndMaterializeVariantProposals(
                  builder, registry, request, &materialized),
              {"no viable plugin proposals",
               "explicit typed tcrv_rvv extension-family IR"}))
    return result;

  KernelOp staleKernel = findKernel(*module, "rvv_missing_typed_body_variant");
  TargetCapabilitySet staleCapabilities =
      TargetCapabilitySet::buildFromKernel(staleKernel);
  VariantOp staleVariant = findVariant(staleKernel, "rvv_missing_typed_body");
  if (int result =
          expectErrorContains(
              registry.verifyVariantLegality(VariantLegalityRequest(
                  staleVariant, staleKernel, staleCapabilities)),
              {"explicit typed RVV extension-family body"}))
    return result;
  VariantEmissionPlan stalePlan;
  if (int result = expectErrorContains(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(staleVariant, staleKernel,
                                     staleCapabilities,
                                     VariantEmissionRole::DirectVariant),
              stalePlan),
          {"explicit typed RVV extension-family body"}))
    return result;

  return 0;
}

int runMetadataOnlyVariantLegalityRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @metadata_without_typed_body_variant {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.variant @rvv_metadata_without_typed_body attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = "not_a_typed_policy"
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV metadata rejection module");
  KernelOp kernel = findKernel(*module, "metadata_without_typed_body_variant");
  VariantOp variant = findVariant(kernel, "rvv_metadata_without_typed_body");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);
  tianchenrv::plugin::rvv::RVVExtensionPlugin plugin;
  return expectErrorContains(
      plugin.verifyVariantLegality(
          VariantLegalityRequest(variant, kernel, capabilities)),
      {"explicit typed RVV extension-family body"});
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
  if (int result =
          runCapabilityOnlyProposalAndTypedBodyRequirementTest(context))
    return result;
  if (int result = runMetadataOnlyVariantLegalityRejectionTest(context))
    return result;

  llvm::outs() << "RVV extension plugin smoke test passed\n";
  return 0;
}
