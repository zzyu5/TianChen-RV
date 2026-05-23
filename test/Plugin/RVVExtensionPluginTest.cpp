#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
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

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::SourceFrontDoorPassRegistration;
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmissionStatus;
using tianchenrv::plugin::VariantEmitCLowerableRequest;
using tianchenrv::plugin::VariantLegalityRequest;
using tianchenrv::plugin::VariantLoweringBoundaryRequest;
using tianchenrv::plugin::VariantLoweringBoundaryResult;
using tianchenrv::plugin::VariantLoweringBoundaryValidationRequest;
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
  if (!error) {
    if (!fragments.size())
      return fail("expected error");
    return fail(llvm::Twine("expected error containing '") +
                *fragments.begin() + "'");
  }
  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("error text missing '") + fragment +
                  "': " + message);
  }
  return 0;
}

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context,
                                              llvm::StringRef source) {
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

mlir::Operation *findFirstNestedOp(VariantOp variant, llvm::StringRef name) {
  mlir::Operation *found = nullptr;
  if (!variant)
    return found;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found || op == variant.getOperation())
      return;
    if (op->getName().getStringRef() == name)
      found = op;
  });
  return found;
}

mlir::func::FuncOp findHighLevelPlaceholder(mlir::ModuleOp module) {
  return module.lookupSymbol<mlir::func::FuncOp>("high_level_placeholder");
}

RVVProbeCapabilityFacts makeSuccessfulProbeFacts() {
  RVVProbeCapabilityFacts facts;
  facts.architecture = "riscv64";
  facts.hartCount = 64;
  facts.vlenbBytes = 16;
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
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin"))
    return result;
  if (int result = expect(registry.size() == 1,
                          "RVV plugin registration adds one plugin"))
    return result;
  if (int result = expect(
          registry.lookupPlugin(
              tianchenrv::plugin::rvv::getRVVExtensionPluginName()) != nullptr,
          "registered RVV plugin is lookup-visible"))
    return result;

  llvm::SmallVector<PluginCapability, 16> capabilities;
  registry.collectCapabilities(capabilities);
  if (int result = expect(capabilities.size() == 1,
                          "RVV plugin exposes only the base RVV capability"))
    return result;
  if (int result = expect(registry.lookupCapabilityByID("rvv") != nullptr,
                          "RVV capability id lookup succeeds"))
    return result;

  llvm::SmallVector<SourceFrontDoorPassRegistration, 2> sourceFrontDoorPasses;
  if (int result = expectSuccess(
          registry.collectSourceFrontDoorPasses(sourceFrontDoorPasses),
          "collect RVV source front-door pass registrations"))
    return result;
  if (int result =
          expect(sourceFrontDoorPasses.size() == 1,
                 "RVV plugin exposes one source front-door pass registration"))
    return result;
  if (int result =
          expect(sourceFrontDoorPasses.front().getOwnerPlugin() ==
                     tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
                 "RVV source front-door pass is owned by RVV plugin"))
    return result;
  if (int result =
          expect(sourceFrontDoorPasses.front().getArgument() ==
                     "tcrv-rvv-fail-closed-legacy-vector-source-front-door",
                 "RVV source front-door pass exposes the fail-closed legacy argument"))
    return result;
  if (int result = expect(
          !sourceFrontDoorPasses.front().isDefaultArtifactFrontDoorEligible(),
          "RVV source front-door pass is explicit-only and not default "
          "artifact-front-door eligible"))
    return result;
  return expect(static_cast<bool>(sourceFrontDoorPasses.front().getFactory()),
                "RVV source front-door pass factory is present");
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
  return expect(
      capabilities.isCapabilityAvailableByID(
          tianchenrv::plugin::rvv::getRVVProbeCompileRunCapabilityID()),
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
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
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
  if (int result =
          expect(proposals.empty(), "RVV plugin proposes nothing without rvv"))
    return result;

  KernelOp availableRVVNoBody = findKernel(*module, "missing_hart_count");
  TargetCapabilitySet availableRVVCapabilities =
      TargetCapabilitySet::buildFromKernel(availableRVVNoBody);
  llvm::SmallVector<VariantProposalDecline, 1> declines;
  if (int result =
          expectSuccess(registry.collectVariantProposals(
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
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin for typed-body requirement test"))
    return result;

  VariantProposalRequest request(highLevel.getOperation(), kernel,
                                 capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  llvm::SmallVector<VariantProposalDecline, 1> declines;
  if (int result = expectSuccess(
          registry.collectVariantProposals(request, proposals, &declines),
          "collect RVV capability-only proposal surface"))
    return result;
  if (int result = expect(proposals.empty(),
                          "RVV no-body capability produces no proposal"))
    return result;
  if (int result = expect(!declines.empty() &&
                              llvm::StringRef(declines.front().getReason())
                                  .contains("explicit typed tcrv_rvv "
                                            "extension-family IR"),
                          "RVV decline explains explicit typed IR requirement"))
    return result;

  mlir::OpBuilder builder(module->getContext());
  llvm::SmallVector<VariantOp, 1> materialized;
  if (int result = expectErrorContains(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materialized),
          {"no viable plugin proposals",
           "explicit typed tcrv_rvv extension-family IR"}))
    return result;

  KernelOp staleKernel = findKernel(*module, "rvv_missing_typed_body_variant");
  TargetCapabilitySet staleCapabilities =
      TargetCapabilitySet::buildFromKernel(staleKernel);
  VariantOp staleVariant = findVariant(staleKernel, "rvv_missing_typed_body");
  if (int result = expectErrorContains(
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
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  tianchenrv::plugin::rvv::RVVExtensionPlugin plugin;
  return expectErrorContains(
      plugin.verifyVariantLegality(
          VariantLegalityRequest(variant, kernel, capabilities)),
      {"explicit typed RVV extension-family body"});
}

int runWithVLSelectedLoweringBoundaryTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_i32m1_boundary_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}

    tcrv.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_add, sew = 32 : i64, source_kernel = "rvv_i32m1_boundary_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }

    tcrv.exec.variant @rvv_i32_sub attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_sub, sew = 32 : i64, source_kernel = "rvv_i32m1_boundary_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %diff = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "sub"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %diff, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }

    tcrv.exec.variant @rvv_i32_mul attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_mul, sew = 32 : i64, source_kernel = "rvv_i32m1_boundary_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %product = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "mul"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %product, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV with_vl boundary module");
  KernelOp kernel = findKernel(*module, "rvv_i32m1_boundary_kernel");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin for with_vl boundary test"))
    return result;

  mlir::OpBuilder builder(module->getContext());
  for (llvm::StringRef variantName :
       {"rvv_i32_add", "rvv_i32_sub", "rvv_i32_mul"}) {
    VariantOp variant = findVariant(kernel, variantName);
    mlir::Operation *withVL = findFirstNestedOp(variant, "tcrv_rvv.with_vl");
    if (int result = expect(withVL != nullptr,
                            llvm::Twine("found with_vl for @") + variantName))
      return result;

    VariantLoweringBoundaryResult boundaryResult;
    if (int result = expectSuccess(
            registry.materializeSelectedLoweringBoundary(
                VariantLoweringBoundaryRequest(
                    variant, kernel, capabilities,
                    VariantEmissionRole::DirectVariant, builder),
                boundaryResult),
            llvm::Twine("materialize with_vl selected boundary for @") +
                variantName))
      return result;
    if (int result =
            expect(boundaryResult.isMaterialized(),
                   llvm::Twine("RVV selected boundary is materialized for @") +
                       variantName))
      return result;
    if (int result = expect(
            boundaryResult.getMaterializedOperation() == withVL,
            llvm::Twine("materialized boundary is existing with_vl for @") +
                variantName))
      return result;
    if (int result = expectSuccess(
            registry.validateSelectedLoweringBoundary(
                VariantLoweringBoundaryValidationRequest(
                    variant, kernel, capabilities,
                    VariantEmissionRole::DirectVariant, withVL)),
            llvm::Twine("validate with_vl selected boundary for @") +
                variantName))
      return result;

    VariantEmissionPlan plan;
    if (int result = expectSuccess(
            registry.buildVariantEmissionPlan(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                plan),
            llvm::Twine("build construction-checked emission plan for @") +
                variantName))
      return result;
    llvm::Expected<
        tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription>
        routeDescription =
            tianchenrv::plugin::rvv::describeRVVSelectedBodyEmitCRoute(
                VariantEmitCLowerableRequest(
                    variant, kernel, capabilities,
                    VariantEmissionRole::DirectVariant));
    if (!routeDescription)
      return fail(llvm::Twine("describe selected-body route for @") +
                  variantName + ": " +
                  llvm::toString(routeDescription.takeError()));
    if (int result = expect(
            plan.isSupported(),
            llvm::Twine("RVV emission plan is supported for @") + variantName))
      return result;
    if (int result = expect(
            plan.getLoweringPipeline() ==
                routeDescription->targetArtifactRouteID,
            llvm::Twine("RVV emission plan uses the materialized EmitC target "
                        "artifact route from selected-body description for @") +
                variantName))
      return result;
    if (int result = expect(
            plan.getRuntimeABIName() == routeDescription->runtimeABIName,
            llvm::Twine("RVV emission plan uses selected-body runtime ABI "
                        "description for @") +
                variantName))
      return result;
    if (int result = expect(
            plan.getArtifactKind() == "riscv-elf-relocatable-object",
            llvm::Twine("RVV emission plan advertises object artifact for @") +
                variantName))
      return result;
    if (int result = expect(
            plan.getRuntimeABIParameters().size() == 4,
            llvm::Twine("RVV emission plan carries callable ABI parameters "
                        "for @") +
                variantName))
      return result;
    if (int result = expect(
            routeDescription->sew == 32 && routeDescription->lmul == "m1" &&
                routeDescription->tailPolicy == "agnostic" &&
                routeDescription->maskPolicy == "agnostic" &&
                routeDescription->boundaryOpName == "tcrv_rvv.with_vl" &&
                routeDescription->vectorTypeName ==
                    "!tcrv_rvv.vector<i32, \"m1\">" &&
                routeDescription->vectorCType == "vint32m1_t" &&
                routeDescription->setVLIntrinsic == "__riscv_vsetvl_e32m1" &&
                routeDescription->vectorLoadIntrinsic ==
                    "__riscv_vle32_v_i32m1" &&
                routeDescription->storeIntrinsic == "__riscv_vse32_v_i32m1" &&
                routeDescription->rhsBroadcastIntrinsic.empty(),
            llvm::Twine("RVV selected-body route profile carries "
                        "typed config and intrinsic mapping for @") +
                variantName))
      return result;
    if (int result = expectSuccess(
            tianchenrv::plugin::rvv::
                verifyRVVSelectedBodyEmitCRouteDescription(
                    *routeDescription,
                    "RVV selected-body provider identity test"),
            llvm::Twine("verify provider-derived selected-body identity for @") +
                variantName))
      return result;
    if (routeDescription->operation ==
        tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Add) {
      tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription
          staleIntrinsic = *routeDescription;
      staleIntrinsic.intrinsic = "__riscv_vsub_vv_i32m1";
      if (int result = expectErrorContains(
              tianchenrv::plugin::rvv::
                  verifyRVVSelectedBodyEmitCRouteDescription(
                      staleIntrinsic,
                      "RVV selected-body stale intrinsic test"),
              {"compute intrinsic", "__riscv_vadd_vv_i32m1",
               "__riscv_vsub_vv_i32m1"}))
        return result;

      tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription
          staleArtifactRoute = *routeDescription;
      staleArtifactRoute.targetArtifactRouteID =
          "rvv-stale-target-artifact-route";
      if (int result = expectErrorContains(
              tianchenrv::plugin::rvv::
                  verifyRVVSelectedBodyEmitCRouteDescription(
                      staleArtifactRoute,
                      "RVV selected-body stale target artifact route test"),
              {"target artifact route id",
               tianchenrv::plugin::rvv::
                   getRVVSelectedBodyTargetArtifactRouteID(),
               "rvv-stale-target-artifact-route"}))
        return result;

      tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription
          staleEmitCRoute = *routeDescription;
      staleEmitCRoute.emitCRouteID = "rvv-stale-emitc-route";
      if (int result = expectErrorContains(
              tianchenrv::plugin::rvv::
                  verifyRVVSelectedBodyEmitCRouteDescription(
                      staleEmitCRoute,
                      "RVV selected-body stale EmitC route id test"),
              {"EmitC route id", routeDescription->emitCRouteID,
               "rvv-stale-emitc-route"}))
        return result;

      tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription
          staleRuntimeABI = *routeDescription;
      staleRuntimeABI.runtimeABIName = "rvv-stale-callable-c-abi.v1";
      if (int result = expectErrorContains(
              tianchenrv::plugin::rvv::
                  verifyRVVSelectedBodyEmitCRouteDescription(
                      staleRuntimeABI,
                      "RVV selected-body stale runtime ABI label test"),
              {"runtime ABI name", routeDescription->runtimeABIName,
               "rvv-stale-callable-c-abi.v1"}))
        return result;

      tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription
          staleConfigContract = *routeDescription;
      staleConfigContract.configContractID =
          "rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1";
      if (int result = expectErrorContains(
              tianchenrv::plugin::rvv::
                  verifyRVVSelectedBodyEmitCRouteDescription(
                      staleConfigContract,
                      "RVV selected-body stale config contract test"),
              {"config contract", routeDescription->configContractID,
               staleConfigContract.configContractID}))
        return result;
    }

    tianchenrv::plugin::rvv::RVVSelectedBodyConstructionMetadataFacts facts =
        tianchenrv::plugin::rvv::
            getRVVSelectedBodyConstructionMetadataFacts(*routeDescription);
    llvm::Expected<llvm::SmallVector<
        tianchenrv::support::ArtifactMetadataEntry, 16>>
        expectedConstructionMetadata =
            tianchenrv::plugin::rvv::
                getRVVSelectedBodyConstructionArtifactMetadata(facts);
    if (!expectedConstructionMetadata)
      return fail(llvm::Twine("build provider-derived construction metadata "
                              "for @") +
                  variantName + ": " +
                  llvm::toString(expectedConstructionMetadata.takeError()));
    if (plan.getArtifactMetadata().size() <
        expectedConstructionMetadata->size())
      return fail(llvm::Twine("RVV emission plan missing construction "
                              "metadata for @") +
                  variantName);
    for (std::size_t index = 0; index < expectedConstructionMetadata->size();
         ++index) {
      const tianchenrv::support::ArtifactMetadataEntry &got =
          plan.getArtifactMetadata()[index];
      const tianchenrv::support::ArtifactMetadataEntry &want =
          (*expectedConstructionMetadata)[index];
      if (got.key != want.key || got.value != want.value)
        return fail(llvm::Twine("RVV emission plan construction metadata[") +
                    llvm::Twine(index) + "] for @" + variantName +
                    " must mirror provider-derived route facts");
    }

    VariantEmissionStatus readiness;
    if (int result = expectSuccess(
            registry.checkVariantEmissionReadiness(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                readiness),
            llvm::Twine("RVV emission readiness succeeds for @") + variantName))
      return result;
    if (int result =
            expect(readiness.isSupported() &&
                       readiness.getEmissionPath() ==
                           routeDescription->targetArtifactRouteID,
                   llvm::Twine("RVV emission readiness names the selected-body "
                               "description target artifact route for @") +
                       variantName))
      return result;
  }

  VariantOp addVariant = findVariant(kernel, "rvv_i32_add");
  mlir::Operation *setvl = findFirstNestedOp(addVariant, "tcrv_rvv.setvl");
  return expectErrorContains(
      registry.validateSelectedLoweringBoundary(
          VariantLoweringBoundaryValidationRequest(
              addVariant, kernel, capabilities,
              VariantEmissionRole::DirectVariant, setvl)),
      {"selected RVV typed lowering boundary must be the existing "
       "tcrv_rvv.with_vl operation"});
}

int runWithVLSelectedLoweringBoundaryDuplicateRejectionTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_i32m1_duplicate_boundary_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add_duplicate_with_vl attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_add_duplicate_with_vl, sew = 32 : i64, source_kernel = "rvv_i32m1_duplicate_boundary_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_add_duplicate_with_vl, sew = 32 : i64, source_kernel = "rvv_i32m1_duplicate_boundary_kernel", status = "selected-lowering-boundary"} {
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV duplicate with_vl boundary module");
  KernelOp kernel = findKernel(*module, "rvv_i32m1_duplicate_boundary_kernel");
  VariantOp variant = findVariant(kernel, "rvv_i32_add_duplicate_with_vl");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin for duplicate boundary test"))
    return result;

  mlir::OpBuilder builder(module->getContext());
  VariantLoweringBoundaryResult boundaryResult;
  return expectErrorContains(
      registry.materializeSelectedLoweringBoundary(
          VariantLoweringBoundaryRequest(variant, kernel, capabilities,
                                         VariantEmissionRole::DirectVariant,
                                         builder),
          boundaryResult),
      {"selected RVV typed lowering boundary requires exactly one "
       "tcrv_rvv.with_vl op"});
}

int runStaleWithVLRouteMetadataDoesNotAuthorizeEmissionTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_stale_route_metadata_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_stale_route_metadata attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-binary-add-emitc-route", selected_path_role = "direct variant", selected_variant = @rvv_stale_route_metadata, sew = 32 : i64, source_kernel = "rvv_stale_route_metadata_kernel", status = "selected-lowering-boundary"} {
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV stale route metadata module");
  KernelOp kernel = findKernel(*module, "rvv_stale_route_metadata_kernel");
  VariantOp variant = findVariant(kernel, "rvv_stale_route_metadata");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin for stale route metadata test"))
    return result;

  VariantEmissionPlan plan;
  return expectErrorContains(
      registry.buildVariantEmissionPlan(
          VariantEmissionRequest(variant, kernel, capabilities,
                                 VariantEmissionRole::DirectVariant),
          plan),
      {"bounded generic RVV vector-load route requires exactly two "
       "tcrv_rvv.load ops"});
}

int runLMULM2SelectedBodyRouteTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_unsupported_lmul_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_m2_add attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_m2_add, sew = 32 : i64, source_kernel = "rvv_unsupported_lmul_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
        %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV unsupported LMUL selected-body module");
  KernelOp kernel = findKernel(*module, "rvv_unsupported_lmul_kernel");
  VariantOp variant = findVariant(kernel, "rvv_i32_m2_add");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin for LMUL m2 route test"))
    return result;

  VariantEmissionPlan plan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              plan),
          "build RVV LMUL m2 selected-body emission plan"))
    return result;
  if (int result = expect(
          plan.isSupported() &&
              plan.getRuntimeABI() ==
                  tianchenrv::plugin::rvv::getRVVSelectedBodyRuntimeABIName(
                      tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::
                          Add),
          "RVV LMUL m2 body keeps the stable callable ABI while typed config "
          "owns LMUL semantics"))
    return result;

  bool sawLMULM2Metadata = false;
  for (const tianchenrv::support::ArtifactMetadataEntry &entry :
       plan.getArtifactMetadata()) {
    if (entry.key == "tcrv_rvv.lmul" && entry.value == "m2")
      sawLMULM2Metadata = true;
  }
  if (int result = expect(
          sawLMULM2Metadata,
          "RVV LMUL m2 emission plan mirrors typed selected-body LMUL m2"))
    return result;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute route;
  if (int result = expectSuccess(
          registry.buildVariantEmitCLowerableRoute(
              VariantEmitCLowerableRequest(variant, kernel, capabilities,
                                           VariantEmissionRole::DirectVariant),
              route),
          "build RVV LMUL m2 selected-body EmitC route"))
    return result;
  if (int result =
          expect(route.getForLoops().size() == 1 &&
                     route.getForLoops().front().bodySteps.size() == 5 &&
                     route.getForLoops().front().bodySteps[0].callee ==
                         "__riscv_vsetvl_e32m2" &&
                     route.getForLoops().front().bodySteps[1].callee ==
                         "__riscv_vle32_v_i32m2" &&
                     route.getForLoops().front().bodySteps[2].callee ==
                         "__riscv_vle32_v_i32m2" &&
                     route.getForLoops().front().bodySteps[3].callee ==
                         "__riscv_vadd_vv_i32m2" &&
                     route.getForLoops().front().bodySteps[4].callee ==
                         "__riscv_vse32_v_i32m2",
                 "RVV LMUL m2 route derives m2 setvl/load/add/store "
                 "intrinsics from typed config"))
    return result;
  if (int result = expectSuccess(
      tianchenrv::conversion::emitc::
          verifyTCRVEmitCLowerableRouteMaterializesToEmitC(
              route, "tcrv_rvv_lmul_m2_selected_body", {}),
      "RVV LMUL m2 EmitC lowerable route materializes to EmitC"))
    return result;

  mlir::Builder builder(&context);
  auto setvl = llvm::dyn_cast_or_null<tianchenrv::tcrv::rvv::SetVLOp>(
      findFirstNestedOp(variant, "tcrv_rvv.setvl"));
  auto withVL = llvm::dyn_cast_or_null<tianchenrv::tcrv::rvv::WithVLOp>(
      findFirstNestedOp(variant, "tcrv_rvv.with_vl"));
  if (int result =
          expect(setvl && withVL, "LMUL m2 route test locates setvl/with_vl"))
    return result;
  setvl->setAttr("lmul", builder.getStringAttr("m1"));
  withVL->setAttr("lmul", builder.getStringAttr("m1"));

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute staleConfigRoute;
  return expectErrorContains(
      registry.buildVariantEmitCLowerableRoute(
          VariantEmitCLowerableRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
          staleConfigRoute),
      {"selected RVV typed config resolver requires lhs vector LMUL 'm2' to "
       "match selected config LMUL 'm1'"});
}

int runBroadcastSelectedBodyRouteTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_i32m1_broadcast_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add_broadcast attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_add_broadcast, sew = 32 : i64, source_kernel = "rvv_i32m1_broadcast_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.broadcast_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV broadcast selected-body module");
  KernelOp kernel = findKernel(*module, "rvv_i32m1_broadcast_kernel");
  VariantOp variant = findVariant(kernel, "rvv_i32_add_broadcast");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin for broadcast route test"))
    return result;

  VariantEmissionPlan plan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              plan),
          "build RVV generic broadcast selected-body emission plan"))
    return result;

  llvm::Expected<tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription>
      routeDescription =
          tianchenrv::plugin::rvv::describeRVVSelectedBodyEmitCRoute(
              VariantEmitCLowerableRequest(
                  variant, kernel, capabilities,
                  VariantEmissionRole::DirectVariant));
  if (!routeDescription)
    return fail("describe RVV generic broadcast selected-body route: " +
                llvm::toString(routeDescription.takeError()));
  if (int result =
          expect(plan.isSupported() &&
                     routeDescription->operation ==
                         tianchenrv::plugin::rvv::
                             RVVSelectedBodyOperationKind::Add &&
                     routeDescription->memoryForm ==
                         tianchenrv::plugin::rvv::
                             RVVSelectedBodyMemoryForm::RHSBroadcastLoad &&
                     routeDescription->typedComputeOpName ==
                         "tcrv_rvv.binary" &&
                     routeDescription->rhsBroadcastIntrinsic ==
                         "__riscv_vmv_v_x_i32m1",
                 "RVV generic broadcast route is provider-derived from typed "
                 "body structure"))
    return result;
  return expectSuccess(
      tianchenrv::plugin::rvv::verifyRVVSelectedBodyEmitCRouteDescription(
          *routeDescription,
          "RVV generic broadcast selected-body route description"),
      "verify RVV generic broadcast selected-body route description");
}

int runScalarBroadcastElementwisePlanValidationTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_scalar_broadcast_elementwise_plan_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_scalar_broadcast_add attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_scalar_broadcast_add, sew = 32 : i64, source_kernel = "rvv_scalar_broadcast_elementwise_plan_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV scalar-broadcast elementwise module");
  KernelOp kernel =
      findKernel(*module, "rvv_scalar_broadcast_elementwise_plan_kernel");
  VariantOp variant = findVariant(kernel, "rvv_scalar_broadcast_add");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  llvm::Expected<tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription>
      routeDescription =
          tianchenrv::plugin::rvv::describeRVVSelectedBodyEmitCRoute(
              VariantEmitCLowerableRequest(
                  variant, kernel, capabilities,
                  VariantEmissionRole::DirectVariant));
  if (!routeDescription)
    return fail("describe RVV scalar-broadcast elementwise route: " +
                llvm::toString(routeDescription.takeError()));
  if (int result =
          expect(routeDescription->operation ==
                         tianchenrv::plugin::rvv::
                             RVVSelectedBodyOperationKind::ScalarBroadcastAdd &&
                     routeDescription->memoryForm ==
                         tianchenrv::plugin::rvv::
                             RVVSelectedBodyMemoryForm::RHSScalarBroadcast &&
                     routeDescription
                             ->scalarBroadcastElementwiseRouteFamilyPlanID ==
                         "rvv-scalar-broadcast-elementwise-route-family-plan.v1" &&
                     routeDescription->targetLeafProfile ==
                         "rvv-v1-e32m1-scalar-broadcast-elementwise-leaf-profile.v1" &&
                     routeDescription->providerSupportedMirror ==
                         "provider_supported_mirror:rvv-scalar-broadcast-elementwise-plan-validated" &&
                     routeDescription->requiredHeaderDeclarations ==
                         "stddef.h,stdint.h,riscv_vector.h" &&
                     routeDescription->cTypeMappingSummary ==
                         "vl:size_t,lhs:signed-e32m1,rhs_scalar:i32,result:signed-e32m1" &&
                     routeDescription->runtimeABIOrder ==
                         "lhs,rhs_scalar,out,n" &&
                     routeDescription->routeOperandBindingSummary ==
                         "rvv-route-operand-binding:scalar_broadcast_add.v1;"
                         "lhs=lhs-input-buffer:lhs:runtime-abi-mirror|"
                         "materialized-load-base|scalar-broadcast-lhs-call|"
                         "header-mirror;"
                         "rhs_scalar=rhs-scalar-value:rhs_scalar:"
                         "runtime-abi-mirror|scalar-broadcast-rhs-call|"
                         "header-mirror;"
                         "out=output-buffer:out:runtime-abi-mirror|"
                         "materialized-store-base|header-mirror;"
                         "n=runtime-element-count:n:runtime-abi-mirror|"
                         "setvl-avl|loop-control|header-mirror" &&
                     routeDescription->rhsBroadcastIntrinsic ==
                         "__riscv_vmv_v_x_i32m1" &&
                     routeDescription->intrinsic ==
                         "__riscv_vadd_vv_i32m1" &&
                     routeDescription->storeIntrinsic ==
                         "__riscv_vse32_v_i32m1",
                 "RVV scalar-broadcast elementwise route records validated "
                 "family plan facts"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVSelectedBodyEmitCRouteDescription(
              *routeDescription,
              "RVV scalar-broadcast elementwise route description"),
          "verify RVV scalar-broadcast elementwise route description"))
    return result;

  auto expectStaleFieldRejected =
      [&](tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription stale,
          std::initializer_list<llvm::StringRef> expected) -> int {
    return expectErrorContains(
        tianchenrv::plugin::rvv::verifyRVVSelectedBodyEmitCRouteDescription(
            stale, "RVV scalar-broadcast stale plan field test"),
        expected);
  };

  tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription stale =
      *routeDescription;
  stale.providerSupportedMirror = "supported";
  if (int result = expectStaleFieldRejected(
          stale, {"provider_supported_mirror",
                  "provider_supported_mirror:rvv-scalar-broadcast-elementwise-plan-validated",
                  "supported"}))
    return result;

  stale = *routeDescription;
  stale.scalarBroadcastElementwiseRouteFamilyPlanID =
      "rvv-stale-helper-selected-plan.v1";
  if (int result = expectStaleFieldRejected(
          stale, {"scalar-broadcast elementwise route family plan",
                  "rvv-scalar-broadcast-elementwise-route-family-plan.v1",
                  "rvv-stale-helper-selected-plan.v1"}))
    return result;

  stale = *routeDescription;
  stale.targetLeafProfile = "route-id-selected-profile";
  if (int result = expectStaleFieldRejected(
          stale, {"target leaf profile",
                  "rvv-v1-e32m1-scalar-broadcast-elementwise-leaf-profile.v1",
                  "route-id-selected-profile"}))
    return result;

  stale = *routeDescription;
  stale.rhsBroadcastIntrinsic = "__riscv_vle32_v_i32m1";
  if (int result = expectStaleFieldRejected(
          stale, {"RHS broadcast intrinsic", "__riscv_vmv_v_x_i32m1",
                  "__riscv_vle32_v_i32m1"}))
    return result;

  stale = *routeDescription;
  stale.runtimeABIOrder = "lhs,rhs,out,n";
  return expectStaleFieldRejected(
      stale, {"runtime ABI order", "lhs,rhs_scalar,out,n",
              "lhs,rhs,out,n"});
}

int runScalarBroadcastAndSplatRouteFamilyProviderPlanTest() {
  using tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  using tianchenrv::plugin::rvv::RVVSelectedBodyRouteAnalysis;
  using tianchenrv::plugin::rvv::
      isRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyConsumer;
  using tianchenrv::plugin::rvv::
      isRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyConsumer;
  using tianchenrv::plugin::rvv::
      verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyProviderPlans;
  using tianchenrv::plugin::rvv::
      verifyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyProviderPlans;

  for (RVVSelectedBodyOperationKind op :
       {RVVSelectedBodyOperationKind::ScalarBroadcastAdd,
        RVVSelectedBodyOperationKind::ScalarBroadcastSub,
        RVVSelectedBodyOperationKind::ScalarBroadcastMul}) {
    if (int result = expect(
            isRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyConsumer(op),
            "scalar-broadcast add/sub/mul must be scalar-broadcast "
            "elementwise family consumers"))
      return result;
  }
  if (int result = expect(
          !isRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::RuntimeI32SplatStore),
          "runtime_i32_splat_store must stay outside scalar-broadcast "
          "elementwise family"))
    return result;
  if (int result = expect(
          isRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::RuntimeI32SplatStore),
          "runtime_i32_splat_store must be a runtime scalar splat-store "
          "family consumer"))
    return result;
  if (int result = expect(
          !isRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::ScalarBroadcastAdd),
          "scalar_broadcast_add must stay outside runtime scalar splat-store "
          "family"))
    return result;

  RVVSelectedBodyRouteAnalysis missingScalarBroadcastPlan;
  missingScalarBroadcastPlan.description.operation =
      RVVSelectedBodyOperationKind::ScalarBroadcastAdd;
  if (int result = expectErrorContains(
          verifyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyProviderPlans(
              missingScalarBroadcastPlan,
              "scalar-broadcast provider unit test"),
          {"requires the scalar-broadcast elementwise route-family plan",
           "scalar_broadcast_add"}))
    return result;

  RVVSelectedBodyRouteAnalysis staleScalarBroadcastNonConsumer;
  staleScalarBroadcastNonConsumer.description.operation =
      RVVSelectedBodyOperationKind::Add;
  staleScalarBroadcastNonConsumer.scalarBroadcastElementwiseRouteFamilyPlan
      .emplace();
  staleScalarBroadcastNonConsumer.scalarBroadcastElementwiseRouteFamilyPlan
      ->operation = RVVSelectedBodyOperationKind::ScalarBroadcastAdd;
  if (int result = expectErrorContains(
          verifyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyProviderPlans(
              staleScalarBroadcastNonConsumer,
              "scalar-broadcast provider unit test"),
          {"must not carry a scalar-broadcast elementwise route-family plan",
           "add"}))
    return result;

  RVVSelectedBodyRouteAnalysis missingRuntimeSplatPlan;
  missingRuntimeSplatPlan.description.operation =
      RVVSelectedBodyOperationKind::RuntimeI32SplatStore;
  if (int result = expectErrorContains(
          verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyProviderPlans(
              missingRuntimeSplatPlan,
              "runtime scalar splat-store provider unit test"),
          {"requires the runtime scalar splat-store route-family plan",
           "runtime_i32_splat_store"}))
    return result;

  RVVSelectedBodyRouteAnalysis staleRuntimeSplatNonConsumer;
  staleRuntimeSplatNonConsumer.description.operation =
      RVVSelectedBodyOperationKind::ScalarBroadcastAdd;
  staleRuntimeSplatNonConsumer.runtimeScalarSplatStoreRouteFamilyPlan.emplace();
  staleRuntimeSplatNonConsumer.runtimeScalarSplatStoreRouteFamilyPlan
      ->operation = RVVSelectedBodyOperationKind::RuntimeI32SplatStore;
  return expectErrorContains(
      verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyProviderPlans(
          staleRuntimeSplatNonConsumer,
          "runtime scalar splat-store provider unit test"),
      {"must not carry a runtime scalar splat-store route-family plan",
       "scalar_broadcast_add"});
}

int runWideningConversionRouteFamilyProviderPlanTest() {
  using tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  using tianchenrv::plugin::rvv::RVVSelectedBodyRouteAnalysis;
  using tianchenrv::plugin::rvv::
      isRVVSelectedBodyWideningConversionRouteFamilyConsumer;
  using tianchenrv::plugin::rvv::
      verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans;

  if (int result = expect(
          isRVVSelectedBodyWideningConversionRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::WidenI32ToI64),
          "widen_i32_to_i64 must be a widening conversion family consumer"))
    return result;
  if (int result = expect(
          isRVVSelectedBodyWideningConversionRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::WidenI16ToI32),
          "widen_i16_to_i32 must be a widening conversion family consumer"))
    return result;
  if (int result = expect(
          !isRVVSelectedBodyWideningConversionRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::WideningMAccAdd),
          "widening_macc_add must stay outside widening conversion family"))
    return result;
  if (int result = expect(
          !isRVVSelectedBodyWideningConversionRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::Add),
          "plain add must stay outside widening conversion family"))
    return result;

  RVVSelectedBodyRouteAnalysis missingI32ToI64Plan;
  missingI32ToI64Plan.description.operation =
      RVVSelectedBodyOperationKind::WidenI32ToI64;
  if (int result = expectErrorContains(
          verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans(
              missingI32ToI64Plan,
              "widening conversion provider unit test"),
          {"requires the widening conversion route-family plan",
           "widen_i32_to_i64"}))
    return result;

  RVVSelectedBodyRouteAnalysis missingI16ToI32Plan;
  missingI16ToI32Plan.description.operation =
      RVVSelectedBodyOperationKind::WidenI16ToI32;
  if (int result = expectErrorContains(
          verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans(
              missingI16ToI32Plan,
              "widening conversion provider unit test"),
          {"requires the widening conversion route-family plan",
           "widen_i16_to_i32"}))
    return result;

  RVVSelectedBodyRouteAnalysis staleNonConsumer;
  staleNonConsumer.description.operation = RVVSelectedBodyOperationKind::Add;
  staleNonConsumer.wideningConversionRouteFamilyPlan.emplace();
  staleNonConsumer.wideningConversionRouteFamilyPlan->operation =
      RVVSelectedBodyOperationKind::WidenI32ToI64;
  return expectErrorContains(
      verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans(
          staleNonConsumer, "widening conversion provider unit test"),
      {"must not carry a widening conversion route-family plan", "add"});
}

int runMaskedAddSelectedBodyPolicyRouteTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_i32m1_masked_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_masked_add attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_masked_add, sew = 32 : i64, source_kernel = "rvv_i32m1_masked_add_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask = tcrv_rvv.compare %lhs, %rhs, %vl {kind = "eq"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %sum = tcrv_rvv.masked_binary %mask, %lhs, %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV masked add selected-body module");
  KernelOp kernel = findKernel(*module, "rvv_i32m1_masked_add_kernel");
  VariantOp variant = findVariant(kernel, "rvv_i32_masked_add");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin for masked add policy route test"))
    return result;

  VariantEmissionPlan plan;
  if (int result = expectSuccess(
          registry.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              plan),
          "build RVV masked add selected-body emission plan"))
    return result;

  llvm::Expected<tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription>
      routeDescription =
          tianchenrv::plugin::rvv::describeRVVSelectedBodyEmitCRoute(
              VariantEmitCLowerableRequest(
                  variant, kernel, capabilities,
                  VariantEmissionRole::DirectVariant));
  if (!routeDescription)
    return fail("describe RVV generic masked add selected-body route: " +
                llvm::toString(routeDescription.takeError()));
  if (int result =
          expect(plan.isSupported() &&
                     routeDescription->operation ==
                         tianchenrv::plugin::rvv::
                             RVVSelectedBodyOperationKind::MaskedAdd &&
                     routeDescription->typedComputeOpName ==
                         "tcrv_rvv.masked_binary" &&
                     routeDescription->maskRole ==
                         "predicate-mask-produced-by-compare" &&
                     routeDescription->maskSource ==
                         "compare-produced-mask-same-vl-scope" &&
                     routeDescription->inactiveLaneContract ==
                         "masked-off-lanes-preserve-passthrough-vector" &&
                     routeDescription->maskedPassthroughLayout ==
                         "passthrough-vector-preserves-inactive-lanes" &&
                     routeDescription->compareIntrinsic ==
                         "__riscv_vmseq_vv_i32m1_b32" &&
                     routeDescription->intrinsic == "__riscv_vadd_vv_i32m1" &&
                     routeDescription->maskedMergeIntrinsic ==
                         "__riscv_vmerge_vvm_i32m1",
                 "RVV masked add route derives mask role, inactive-lane "
                 "contract, and policy leaves from typed body facts"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::
              verifyRVVSelectedBodyEmitCRouteDescription(
                  *routeDescription,
                  "RVV generic masked add selected-body route description"),
          "verify RVV generic masked add selected-body route description"))
    return result;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute route;
  if (int result = expectSuccess(
          registry.buildVariantEmitCLowerableRoute(
              VariantEmitCLowerableRequest(variant, kernel, capabilities,
                                           VariantEmissionRole::DirectVariant),
              route),
          "build RVV masked add selected-body EmitC route"))
    return result;
  if (int result =
          expect(route.getForLoops().size() == 1 &&
                     route.getForLoops().front().bodySteps.size() == 7 &&
                     route.getForLoops().front().bodySteps[3].callee ==
                         "__riscv_vmseq_vv_i32m1_b32" &&
                     route.getForLoops().front().bodySteps[4].callee ==
                         "__riscv_vadd_vv_i32m1" &&
                     route.getForLoops().front().bodySteps[5].callee ==
                         "__riscv_vmerge_vvm_i32m1",
                 "RVV masked add EmitC route carries explicit compare, "
                 "active add, and passthrough-preserving merge steps"))
    return result;

  tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription staleMaskRole =
      *routeDescription;
  staleMaskRole.maskRole = "";
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::
              verifyRVVSelectedBodyEmitCRouteDescription(
                  staleMaskRole,
                  "RVV masked add missing mask role route description"),
          {"mask role", "predicate-mask-produced-by-compare"}))
    return result;

  tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription
      staleInactiveLaneContract = *routeDescription;
  staleInactiveLaneContract.inactiveLaneContract = "";
  return expectErrorContains(
      tianchenrv::plugin::rvv::verifyRVVSelectedBodyEmitCRouteDescription(
          staleInactiveLaneContract,
          "RVV masked add missing inactive-lane route description"),
      {"inactive lane contract",
       "masked-off-lanes-preserve-passthrough-vector"});
}

int runContractionTargetLeafProfileValidationTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_contraction_leaf_profile_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_computed_mask_strided_dot attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_computed_mask_strided_dot, sew = 32 : i64, source_kernel = "rvv_contraction_leaf_profile_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %cmp_rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %lhs_vec = tcrv_rvv.strided_load %lhs, %lhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %rhs_vec = tcrv_rvv.strided_load %rhs, %rhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %mask = tcrv_rvv.compare %cmp_lhs_vec, %cmp_rhs_vec, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %sum = tcrv_rvv.masked_widening_dot_reduce %mask, %lhs_vec, %rhs_vec, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", kind = "signed_masked_widening_dot_reduce_add", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-dot-reduction-lane0-to-output-scalar"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV contraction leaf-profile module");
  KernelOp kernel = findKernel(*module, "rvv_contraction_leaf_profile_kernel");
  VariantOp variant = findVariant(kernel, "rvv_computed_mask_strided_dot");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  llvm::Expected<tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription>
      routeDescription =
          tianchenrv::plugin::rvv::describeRVVSelectedBodyEmitCRoute(
              VariantEmitCLowerableRequest(
                  variant, kernel, capabilities,
                  VariantEmissionRole::DirectVariant));
  if (!routeDescription)
    return fail("describe RVV contraction leaf-profile route: " +
                llvm::toString(routeDescription.takeError()));
  if (int result = expect(
          routeDescription->targetLeafProfile ==
                  "rvv-v1-i16mf2-i32m1-contraction-leaf-profile.v1" &&
              routeDescription->providerSupportedMirror ==
                  "provider_supported_mirror:rvv-contraction-family-plan-validated" &&
              routeDescription->requiredHeaderDeclarations ==
                  "stddef.h,stdint.h,riscv_vector.h" &&
              routeDescription->cTypeMappingSummary ==
                  "vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32" &&
              routeDescription->sourceVectorLoadIntrinsic ==
                  "__riscv_vle16_v_i16mf2" &&
              routeDescription->stridedLoadIntrinsic ==
                  "__riscv_vlse16_v_i16mf2" &&
              routeDescription->wideningProductIntrinsic ==
                  "__riscv_vwmul_vv_i32m1" &&
              routeDescription->maskedWideningProductIntrinsic ==
                  "__riscv_vwmul_vv_i32m1_m" &&
              routeDescription->intrinsic ==
                  "__riscv_vredsum_vs_i32m1_i32m1" &&
              routeDescription->storeIntrinsic == "__riscv_vse32_v_i32m1" &&
              routeDescription->inactiveLaneZeroingRequirement ==
                  "masked-widening-products-zero-inactive-lanes-before-reduction",
          "RVV contraction route records validated target profile, leaves, "
          "headers, type mapping, support mirror, and zeroing requirement"))
    return result;

  auto expectStaleFieldRejected =
      [&](tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription stale,
          std::initializer_list<llvm::StringRef> expected) -> int {
    return expectErrorContains(
        tianchenrv::plugin::rvv::verifyRVVSelectedBodyEmitCRouteDescription(
            stale, "RVV contraction stale target-leaf/profile test"),
        expected);
  };

  tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription stale =
      *routeDescription;
  stale.targetLeafProfile = "metadata-selected-profile";
  if (int result = expectStaleFieldRejected(
          stale, {"target leaf profile",
                  "rvv-v1-i16mf2-i32m1-contraction-leaf-profile.v1",
                  "metadata-selected-profile"}))
    return result;

  stale = *routeDescription;
  stale.providerSupportedMirror = "supported";
  if (int result = expectStaleFieldRejected(
          stale, {"provider_supported_mirror",
                  "provider_supported_mirror:rvv-contraction-family-plan-validated",
                  "supported"}))
    return result;

  stale = *routeDescription;
  stale.sourceVectorLoadIntrinsic = "__riscv_vle32_v_i32m1";
  if (int result = expectStaleFieldRejected(
          stale, {"source vector-load intrinsic",
                  "__riscv_vle16_v_i16mf2",
                  "__riscv_vle32_v_i32m1"}))
    return result;

  stale = *routeDescription;
  stale.stridedLoadIntrinsic = "__riscv_vlse32_v_i32m1";
  if (int result = expectStaleFieldRejected(
          stale, {"strided-load intrinsic", "__riscv_vlse16_v_i16mf2",
                  "__riscv_vlse32_v_i32m1"}))
    return result;

  stale = *routeDescription;
  stale.wideningProductIntrinsic = "__riscv_vmul_vv_i32m1";
  if (int result = expectStaleFieldRejected(
          stale, {"widening product intrinsic", "__riscv_vwmul_vv_i32m1",
                  "__riscv_vmul_vv_i32m1"}))
    return result;

  stale = *routeDescription;
  stale.maskedWideningProductIntrinsic = "__riscv_vwmul_vv_i32m1";
  if (int result = expectStaleFieldRejected(
          stale, {"masked widening product intrinsic",
                  "__riscv_vwmul_vv_i32m1_m",
                  "__riscv_vwmul_vv_i32m1"}))
    return result;

  stale = *routeDescription;
  stale.intrinsic = "__riscv_vadd_vv_i32m1";
  if (int result = expectStaleFieldRejected(
          stale, {"compute intrinsic", "__riscv_vredsum_vs_i32m1_i32m1",
                  "__riscv_vadd_vv_i32m1"}))
    return result;

  stale = *routeDescription;
  stale.storeIntrinsic = "__riscv_vsse32_v_i32m1";
  if (int result = expectStaleFieldRejected(
          stale, {"store intrinsic", "__riscv_vse32_v_i32m1",
                  "__riscv_vsse32_v_i32m1"}))
    return result;

  stale = *routeDescription;
  stale.inactiveLaneZeroingRequirement = "mask-policy-metadata";
  if (int result = expectStaleFieldRejected(
          stale, {"inactive-lane zeroing requirement",
                  "masked-widening-products-zero-inactive-lanes-before-reduction",
                  "mask-policy-metadata"}))
    return result;

  stale = *routeDescription;
  stale.emitCRouteID = "rvv-i32m1-stale-route";
  return expectStaleFieldRejected(
      stale, {"EmitC route id", routeDescription->emitCRouteID,
              "rvv-i32m1-stale-route"});
}

int runComputedMaskSelectRouteFamilyProviderPlanTest() {
  using tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  using tianchenrv::plugin::rvv::RVVSelectedBodyRouteAnalysis;
  using tianchenrv::plugin::rvv::
      isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer;
  using tianchenrv::plugin::rvv::
      verifyRVVSelectedBodyComputedMaskSelectRouteFamilyProviderPlans;

  if (int result = expect(
          isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::ComputedMaskSelect),
          "computed_mask_select must be a computed-mask select family "
          "consumer"))
    return result;
  if (int result = expect(
          isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect),
          "runtime_scalar_cmp_select must be a computed-mask select family "
          "consumer"))
    return result;
  if (int result = expect(
          isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::
                  RuntimeScalarDualCompareMaskAndSelect),
          "runtime_scalar_dual_cmp_mask_and_select must be a computed-mask "
          "select family consumer"))
    return result;
  if (int result = expect(
          !isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer(
              RVVSelectedBodyOperationKind::CmpSelect),
          "plain cmp_select must remain outside computed-mask select family"))
    return result;

  RVVSelectedBodyRouteAnalysis missingPlan;
  missingPlan.description.operation =
      RVVSelectedBodyOperationKind::ComputedMaskSelect;
  if (int result = expectErrorContains(
          verifyRVVSelectedBodyComputedMaskSelectRouteFamilyProviderPlans(
              missingPlan, "computed-mask select provider unit test"),
          {"requires the computed-mask select route-family plan",
           "computed_mask_select"}))
    return result;

  RVVSelectedBodyRouteAnalysis staleNonConsumer;
  staleNonConsumer.description.operation =
      RVVSelectedBodyOperationKind::CmpSelect;
  staleNonConsumer.computedMaskSelectRouteFamilyPlan.emplace();
  staleNonConsumer.computedMaskSelectRouteFamilyPlan->operation =
      RVVSelectedBodyOperationKind::ComputedMaskSelect;
  if (int result = expectErrorContains(
          verifyRVVSelectedBodyComputedMaskSelectRouteFamilyProviderPlans(
              staleNonConsumer, "computed-mask select provider unit test"),
          {"must not carry a computed-mask select route-family plan",
           "cmp_select"}))
    return result;

  return 0;
}

int runCompareSelectSelectedBodyRouteTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_i32m1_compare_select_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_cmp_select attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_cmp_select, sew = 32 : i64, source_kernel = "rvv_i32m1_compare_select_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %mask = tcrv_rvv.i32_cmp_eq %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1_mask
        %selected = tcrv_rvv.i32_select %mask, %lhs, %rhs, %vl : !tcrv_rvv.i32m1_mask, !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %selected, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV compare/select selected-body module");
  KernelOp kernel = findKernel(*module, "rvv_i32m1_compare_select_kernel");
  VariantOp variant = findVariant(kernel, "rvv_i32_cmp_select");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin for compare/select route test"))
    return result;

  VariantEmissionPlan plan;
  return expectErrorContains(
      registry.buildVariantEmissionPlan(
          VariantEmissionRequest(variant, kernel, capabilities,
                                 VariantEmissionRole::DirectVariant),
          plan),
      {"legacy selected-body op 'tcrv_rvv.i32_load' is fail-closed during "
       "RVV Stage1",
       "Stage2 routes must use generic tcrv_rvv.load, "
       "tcrv_rvv.broadcast_load, tcrv_rvv.splat, tcrv_rvv.strided_load, "
       "tcrv_rvv.binary, tcrv_rvv.index_load, tcrv_rvv.indexed_load, "
       "tcrv_rvv.segment2_load, tcrv_rvv.segment2_store, "
       "tcrv_rvv.indexed_store, "
       "tcrv_rvv.mask_load, tcrv_rvv.compare, tcrv_rvv.masked_binary, "
       "tcrv_rvv.select, tcrv_rvv.reduce, tcrv_rvv.standalone_reduce, "
       "tcrv_rvv.masked_standalone_reduce, "
       "tcrv_rvv.macc, tcrv_rvv.masked_macc, "
       "tcrv_rvv.widening_convert, tcrv_rvv.move, "
       "tcrv_rvv.widening_dot_reduce, tcrv_rvv.masked_widening_dot_reduce, "
       "tcrv_rvv.masked_move, tcrv_rvv.masked_load, "
       "tcrv_rvv.masked_strided_load, tcrv_rvv.masked_indexed_load, "
       "tcrv_rvv.masked_indexed_store, tcrv_rvv.masked_segment2_load, "
       "tcrv_rvv.masked_store, "
       "tcrv_rvv.masked_strided_store, tcrv_rvv.store, and "
       "tcrv_rvv.strided_store"});
}

int runOutOfOrderSelectedRoleSequenceRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_i32m1_out_of_order_role_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add_out_of_order_roles attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_add_out_of_order_roles, sew = 32 : i64, source_kernel = "rvv_i32m1_out_of_order_role_kernel", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV out-of-order role module");
  KernelOp kernel = findKernel(*module, "rvv_i32m1_out_of_order_role_kernel");
  VariantOp variant = findVariant(kernel, "rvv_i32_add_out_of_order_roles");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV plugin for out-of-order role test"))
    return result;

  VariantEmissionPlan plan;
  return expectErrorContains(
      registry.buildVariantEmissionPlan(
          VariantEmissionRequest(variant, kernel, capabilities,
                                 VariantEmissionRole::DirectVariant),
          plan),
      {"selected RVV EmitC route", "construction order 3", "expected 0"});
}

int runRouteOperandBindingPlanValidationTest() {
  using tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription;
  using tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  using tianchenrv::plugin::rvv::RVVRouteOperandBinding;
  using tianchenrv::plugin::rvv::RVVRouteOperandBindingPlan;
  using tianchenrv::support::RuntimeABIParameterRole;
  using tianchenrv::support::makeTargetExportABIParameter;

  auto addBinding =
      [](RVVRouteOperandBindingPlan &plan, llvm::StringRef logicalOperand,
         tianchenrv::support::RuntimeABIParameter parameter,
         std::initializer_list<llvm::StringRef> uses) {
        RVVRouteOperandBinding binding;
        binding.logicalOperand = logicalOperand.str();
        binding.parameter = std::move(parameter);
        for (llvm::StringRef use : uses)
          binding.materializedUses.push_back(use.str());
        plan.bindings.push_back(std::move(binding));
      };

  auto makeClosureDescription =
      [](RVVSelectedBodyOperationKind operation, llvm::StringRef abiOrder,
         const RVVRouteOperandBindingPlan &plan) {
        RVVSelectedBodyEmitCRouteDescription description;
        description.operation = operation;
        description.runtimeABIOrder = abiOrder;
        description.routeOperandBindingPlanID = plan.planID;
        description.routeOperandBindingSummary =
            tianchenrv::plugin::rvv::stringifyRVVRouteOperandBindingPlan(plan);
        for (const RVVRouteOperandBinding &binding : plan.bindings)
          description.runtimeABIParameters.push_back(binding.parameter);
        return description;
      };

  RVVRouteOperandBindingPlan plan;
  plan.planID = "rvv-route-operand-binding:macc_add.v1";
  addBinding(plan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"runtime-abi-mirror", "materialized-load-base"});
  addBinding(plan, "rhs",
             makeTargetExportABIParameter(
                 "rhs", "const int32_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"runtime-abi-mirror", "materialized-load-base"});
  addBinding(plan, "acc",
             makeTargetExportABIParameter(
                 "acc", "const int32_t *",
                 RuntimeABIParameterRole::AccumulatorInputBuffer),
             {"runtime-abi-mirror", "materialized-accumulator-load-base"});
  addBinding(plan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"runtime-abi-mirror", "materialized-store-base"});
  addBinding(plan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control"});

  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              plan, "rvv-route-operand-binding:macc_add.v1",
              "lhs,rhs,acc,out,n", "route operand binding unit test"),
          "valid route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan badOrder = plan;
  badOrder.bindings[2].parameter.cName = "out";
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              badOrder, "rvv-route-operand-binding:macc_add.v1",
              "lhs,rhs,acc,out,n", "route operand binding unit test"),
          {"runtime ABI mirror order"}))
    return result;

  RVVRouteOperandBindingPlan swappedNRole = plan;
  swappedNRole.bindings[4].parameter.role =
      RuntimeABIParameterRole::SourceByteStride;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedNRole, "rvv-route-operand-binding:macc_add.v1",
              "lhs,rhs,acc,out,n", "route operand binding unit test"),
          {"logical operand 'n'", "runtime-element-count",
           "source-byte-stride"}))
    return result;

  RVVRouteOperandBindingPlan duplicate = plan;
  duplicate.bindings.push_back(plan.bindings.front());
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              duplicate, "rvv-route-operand-binding:macc_add.v1",
              "lhs,rhs,acc,out,n", "route operand binding unit test"),
          {"duplicate logical operand"}))
    return result;

  RVVRouteOperandBindingPlan duplicateRole = plan;
  duplicateRole.bindings[1].parameter.role =
      RuntimeABIParameterRole::LHSInputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              duplicateRole, "rvv-route-operand-binding:macc_add.v1",
              "lhs,rhs,acc,out,n", "route operand binding unit test"),
          {"logical operand 'rhs'", "rhs-input-buffer", "lhs-input-buffer"}))
    return result;

  RVVRouteOperandBindingPlan wideningMAccPlan;
  wideningMAccPlan.planID = "rvv-route-operand-binding:widening_macc_add.v1";
  addBinding(wideningMAccPlan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int16_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "src-load", "wmacc-lhs", "src-i16mf2"});
  addBinding(wideningMAccPlan, "rhs",
             makeTargetExportABIParameter(
                 "rhs", "const int16_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi", "src-load", "wmacc-rhs", "src-i16mf2"});
  addBinding(wideningMAccPlan, "acc",
             makeTargetExportABIParameter(
                 "acc", "const int32_t *",
                 RuntimeABIParameterRole::AccumulatorInputBuffer),
             {"abi", "acc-load", "wmacc-acc", "acc-i32m1"});
  addBinding(wideningMAccPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "res-store", "res-i32m1"});
  addBinding(wideningMAccPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              wideningMAccPlan,
              "rvv-route-operand-binding:widening_macc_add.v1",
              "lhs,rhs,acc,out,n", "valid widening macc binding plan"),
          "valid widening macc route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedWideningAccumulator =
      wideningMAccPlan;
  swappedWideningAccumulator.bindings[2].parameter.role =
      RuntimeABIParameterRole::OutputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedWideningAccumulator,
              "rvv-route-operand-binding:widening_macc_add.v1",
              "lhs,rhs,acc,out,n", "route operand binding unit test"),
          {"logical operand 'acc'", "accumulator-input-buffer",
           "output-buffer"}))
    return result;

  llvm::Expected<const tianchenrv::support::RuntimeABIParameter *>
      wrongWideningWidthUse = tianchenrv::plugin::rvv::
          getRVVRouteOperandBindingParameter(
              wideningMAccPlan, "lhs", "res-i32m1",
              "route operand binding unit test");
  if (int result = expectErrorContains(wrongWideningWidthUse.takeError(),
                                       {"materialized use",
                                        "res-i32m1"}))
    return result;

  RVVRouteOperandBindingPlan widenI32ToI64Plan;
  widenI32ToI64Plan.planID =
      "rvv-route-operand-binding:widen_i32_to_i64.v1";
  addBinding(widenI32ToI64Plan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "src-load", "convert-src", "src-i32m1",
              "relation-signed-i32m1-to-i64m2", "hdr"});
  addBinding(widenI32ToI64Plan, "out",
             makeTargetExportABIParameter("out", "int64_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "res-store", "convert-result", "res-i64m2",
              "relation-signed-i32m1-to-i64m2", "hdr"});
  addBinding(widenI32ToI64Plan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop", "hdr"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              widenI32ToI64Plan,
              "rvv-route-operand-binding:widen_i32_to_i64.v1",
              "lhs,out,n", "valid widen i32 to i64 binding plan"),
          "valid widen_i32_to_i64 route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedWidenConversionOut = widenI32ToI64Plan;
  swappedWidenConversionOut.bindings[1].parameter.role =
      RuntimeABIParameterRole::LHSInputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedWidenConversionOut,
              "rvv-route-operand-binding:widen_i32_to_i64.v1",
              "lhs,out,n", "route operand binding unit test"),
          {"logical operand 'out'", "output-buffer", "lhs-input-buffer"}))
    return result;

  llvm::Expected<const tianchenrv::support::RuntimeABIParameter *>
      wrongWidenConversionDirection = tianchenrv::plugin::rvv::
          getRVVRouteOperandBindingParameter(
              widenI32ToI64Plan, "lhs",
              "relation-signed-i16mf2-to-i32m1",
              "route operand binding unit test");
  if (int result = expectErrorContains(
          wrongWidenConversionDirection.takeError(),
          {"materialized use", "relation-signed-i16mf2-to-i32m1"}))
    return result;

  RVVRouteOperandBindingPlan widenI16ToI32Plan;
  widenI16ToI32Plan.planID =
      "rvv-route-operand-binding:widen_i16_to_i32.v1";
  addBinding(widenI16ToI32Plan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int16_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "src-load", "convert-src", "src-i16mf2",
              "relation-signed-i16mf2-to-i32m1", "hdr"});
  addBinding(widenI16ToI32Plan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "res-store", "convert-result", "res-i32m1",
              "relation-signed-i16mf2-to-i32m1", "hdr"});
  addBinding(widenI16ToI32Plan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop", "hdr"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              widenI16ToI32Plan,
              "rvv-route-operand-binding:widen_i16_to_i32.v1",
              "lhs,out,n", "valid widen i16 to i32 binding plan"),
          "valid widen_i16_to_i32 route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan wideningDotPlan;
  wideningDotPlan.planID =
      "rvv-route-operand-binding:widening_dot_reduce.v1";
  addBinding(wideningDotPlan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int16_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "ld", "dot-lhs", "i16"});
  addBinding(wideningDotPlan, "rhs",
             makeTargetExportABIParameter(
                 "rhs", "const int16_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi", "ld", "dot-rhs", "i16"});
  addBinding(wideningDotPlan, "acc",
             makeTargetExportABIParameter(
                 "acc", "const int32_t *",
                 RuntimeABIParameterRole::AccumulatorInputBuffer),
             {"abi", "seed", "red", "i32"});
  addBinding(wideningDotPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "store", "i32"});
  addBinding(wideningDotPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              wideningDotPlan,
              "rvv-route-operand-binding:widening_dot_reduce.v1",
              "lhs,rhs,acc,out,n", "valid widening dot binding plan"),
          "valid widening dot route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedDotAccumulator = wideningDotPlan;
  swappedDotAccumulator.bindings[2].parameter.role =
      RuntimeABIParameterRole::OutputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedDotAccumulator,
              "rvv-route-operand-binding:widening_dot_reduce.v1",
              "lhs,rhs,acc,out,n", "route operand binding unit test"),
          {"logical operand 'acc'", "accumulator-input-buffer",
           "output-buffer"}))
    return result;

  llvm::Expected<const tianchenrv::support::RuntimeABIParameter *>
      wrongDotWidthUse = tianchenrv::plugin::rvv::
          getRVVRouteOperandBindingParameter(
              wideningDotPlan, "lhs", "i32",
              "route operand binding unit test");
  if (int result =
          expectErrorContains(wrongDotWidthUse.takeError(),
                              {"materialized use", "i32"}))
    return result;

  RVVRouteOperandBindingPlan stridedWideningDotPlan;
  stridedWideningDotPlan.planID =
      "rvv-route-operand-binding:strided_widening_dot_reduce.v1";
  addBinding(stridedWideningDotPlan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int16_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "sld", "dot-lhs", "i16"});
  addBinding(stridedWideningDotPlan, "rhs",
             makeTargetExportABIParameter(
                 "rhs", "const int16_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi", "sld", "dot-rhs", "i16"});
  addBinding(stridedWideningDotPlan, "acc",
             makeTargetExportABIParameter(
                 "acc", "const int32_t *",
                 RuntimeABIParameterRole::AccumulatorInputBuffer),
             {"abi", "seed", "red", "i32"});
  addBinding(stridedWideningDotPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "store", "i32"});
  addBinding(stridedWideningDotPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop"});
  addBinding(stridedWideningDotPlan, "lhs_stride",
             makeTargetExportABIParameter(
                 "lhs_stride", "size_t",
                 RuntimeABIParameterRole::LHSInputStride),
             {"abi", "str", "addr"});
  addBinding(stridedWideningDotPlan, "rhs_stride",
             makeTargetExportABIParameter(
                 "rhs_stride", "size_t",
                 RuntimeABIParameterRole::RHSInputStride),
             {"abi", "str", "addr"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              stridedWideningDotPlan,
              "rvv-route-operand-binding:strided_widening_dot_reduce.v1",
              "lhs,rhs,acc,out,n,lhs_stride,rhs_stride",
              "valid strided widening dot binding plan"),
          "valid strided widening dot route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedDotStride = stridedWideningDotPlan;
  swappedDotStride.bindings[6].parameter.role =
      RuntimeABIParameterRole::LHSInputStride;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedDotStride,
              "rvv-route-operand-binding:strided_widening_dot_reduce.v1",
              "lhs,rhs,acc,out,n,lhs_stride,rhs_stride",
              "route operand binding unit test"),
          {"logical operand 'rhs_stride'", "rhs-input-stride",
           "lhs-input-stride"}))
    return result;

  RVVRouteOperandBindingPlan maskedWideningDotPlan;
  maskedWideningDotPlan.planID =
      "rvv-route-operand-binding:masked_widening_dot_reduce.v1";
  addBinding(maskedWideningDotPlan, "cmp_lhs",
             makeTargetExportABIParameter(
                 "cmp_lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "cmp", "mask"});
  addBinding(maskedWideningDotPlan, "cmp_rhs",
             makeTargetExportABIParameter(
                 "cmp_rhs", "const int32_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi", "cmp", "mask"});
  addBinding(maskedWideningDotPlan, "dot_lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int16_t *",
                 RuntimeABIParameterRole::DotLHSInputBuffer),
             {"abi", "ld", "mlhs", "i16"});
  addBinding(maskedWideningDotPlan, "dot_rhs",
             makeTargetExportABIParameter(
                 "rhs", "const int16_t *",
                 RuntimeABIParameterRole::DotRHSInputBuffer),
             {"abi", "ld", "mrhs", "i16"});
  addBinding(maskedWideningDotPlan, "acc",
             makeTargetExportABIParameter(
                 "acc", "const int32_t *",
                 RuntimeABIParameterRole::AccumulatorInputBuffer),
             {"abi", "seed", "red", "i32"});
  addBinding(maskedWideningDotPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "store", "i32"});
  addBinding(maskedWideningDotPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              maskedWideningDotPlan,
              "rvv-route-operand-binding:masked_widening_dot_reduce.v1",
              "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n",
              "valid computed-mask widening dot binding plan"),
          "valid computed-mask widening dot route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedMaskedDotLHS = maskedWideningDotPlan;
  swappedMaskedDotLHS.bindings[2].parameter.role =
      RuntimeABIParameterRole::LHSInputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedMaskedDotLHS,
              "rvv-route-operand-binding:masked_widening_dot_reduce.v1",
              "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n",
              "route operand binding unit test"),
          {"logical operand 'dot_lhs'", "dot-lhs-input-buffer",
           "lhs-input-buffer"}))
    return result;

  llvm::Expected<const tianchenrv::support::RuntimeABIParameter *>
      maskedMirrorMismatch = tianchenrv::plugin::rvv::
          getRVVRouteOperandBindingParameter(
              maskedWideningDotPlan, "cmp_lhs", "mlhs",
              "route operand binding unit test");
  if (int result =
          expectErrorContains(maskedMirrorMismatch.takeError(),
                              {"materialized use", "mlhs"}))
    return result;

  RVVRouteOperandBindingPlan maskedStridedWideningDotPlan;
  maskedStridedWideningDotPlan.planID =
      "rvv-route-operand-binding:masked_strided_wdot.v1";
  addBinding(maskedStridedWideningDotPlan, "cmp_lhs",
             makeTargetExportABIParameter(
                 "cmp_lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "cmp", "mask"});
  addBinding(maskedStridedWideningDotPlan, "cmp_rhs",
             makeTargetExportABIParameter(
                 "cmp_rhs", "const int32_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi", "cmp", "mask"});
  addBinding(maskedStridedWideningDotPlan, "dot_lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int16_t *",
                 RuntimeABIParameterRole::DotLHSInputBuffer),
             {"abi", "sld", "mlhs", "i16"});
  addBinding(maskedStridedWideningDotPlan, "dot_rhs",
             makeTargetExportABIParameter(
                 "rhs", "const int16_t *",
                 RuntimeABIParameterRole::DotRHSInputBuffer),
             {"abi", "sld", "mrhs", "i16"});
  addBinding(maskedStridedWideningDotPlan, "acc",
             makeTargetExportABIParameter(
                 "acc", "const int32_t *",
                 RuntimeABIParameterRole::AccumulatorInputBuffer),
             {"abi", "seed", "red", "i32"});
  addBinding(maskedStridedWideningDotPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "store", "i32"});
  addBinding(maskedStridedWideningDotPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop"});
  addBinding(maskedStridedWideningDotPlan, "lhs_stride",
             makeTargetExportABIParameter(
                 "lhs_stride", "size_t",
                 RuntimeABIParameterRole::LHSInputStride),
             {"abi", "str", "addr"});
  addBinding(maskedStridedWideningDotPlan, "rhs_stride",
             makeTargetExportABIParameter(
                 "rhs_stride", "size_t",
                 RuntimeABIParameterRole::RHSInputStride),
             {"abi", "str", "addr"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              maskedStridedWideningDotPlan,
              "rvv-route-operand-binding:"
              "masked_strided_wdot.v1",
              "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride",
              "valid computed-mask strided widening dot binding plan"),
          "valid computed-mask strided widening dot route operand binding "
          "plan"))
    return result;

  RVVRouteOperandBindingPlan swappedMaskedStridedDotStride =
      maskedStridedWideningDotPlan;
  swappedMaskedStridedDotStride.bindings[8].parameter.role =
      RuntimeABIParameterRole::LHSInputStride;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedMaskedStridedDotStride,
              "rvv-route-operand-binding:"
              "masked_strided_wdot.v1",
              "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride",
              "route operand binding unit test"),
          {"logical operand 'rhs_stride'", "rhs-input-stride",
           "lhs-input-stride"}))
    return result;

  llvm::Expected<const tianchenrv::support::RuntimeABIParameter *> missingUse =
      tianchenrv::plugin::rvv::getRVVRouteOperandBindingParameter(
          plan, "acc", "materialized-store-base",
          "route operand binding unit test");
  if (int result =
          expectErrorContains(missingUse.takeError(), {"materialized use"}))
    return result;

  llvm::Expected<const tianchenrv::support::RuntimeABIParameter *>
      missingOperand = tianchenrv::plugin::rvv::
          getRVVRouteOperandBindingParameter(
              plan, "stride_bytes", "materialized-strided-load-stride",
              "route operand binding unit test");
  if (int result =
          expectErrorContains(missingOperand.takeError(),
                              {"exactly one logical operand", "stride_bytes"}))
    return result;

  RVVRouteOperandBindingPlan stridedLoadPlan;
  stridedLoadPlan.planID =
      "rvv-route-operand-binding:strided_load_unit_store.v1";
  addBinding(stridedLoadPlan, "src",
             makeTargetExportABIParameter(
                 "src", "const int32_t *",
                 RuntimeABIParameterRole::SourceInputBuffer),
             {"runtime-abi-mirror", "materialized-strided-load-base"});
  addBinding(stridedLoadPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"runtime-abi-mirror", "materialized-store-base"});
  addBinding(stridedLoadPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control"});
  addBinding(stridedLoadPlan, "stride_bytes",
             makeTargetExportABIParameter(
                 "stride_bytes", "size_t",
                 RuntimeABIParameterRole::SourceByteStride),
             {"runtime-abi-mirror", "materialized-strided-load-stride",
              "materialized-byte-address"});

  RVVRouteOperandBindingPlan swappedStrideAndN = stridedLoadPlan;
  swappedStrideAndN.bindings[2].parameter.role =
      RuntimeABIParameterRole::SourceByteStride;
  swappedStrideAndN.bindings[3].parameter.role =
      RuntimeABIParameterRole::RuntimeElementCount;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedStrideAndN,
              "rvv-route-operand-binding:strided_load_unit_store.v1",
              "src,out,n,stride_bytes", "route operand binding unit test"),
          {"logical operand 'n'", "runtime-element-count",
           "source-byte-stride"}))
    return result;

  RVVRouteOperandBindingPlan indexedGatherPlan;
  indexedGatherPlan.planID =
      "rvv-route-operand-binding:indexed_gather_unit_store.v1";
  addBinding(indexedGatherPlan, "data",
             makeTargetExportABIParameter(
                 "data", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"runtime-abi-mirror", "materialized-indexed-data-base",
              "indexed-load-base"});
  addBinding(indexedGatherPlan, "index",
             makeTargetExportABIParameter(
                 "index", "const uint32_t *",
                 RuntimeABIParameterRole::IndexInputBuffer),
             {"runtime-abi-mirror", "materialized-index-load-base",
              "index-offset-scale"});
  addBinding(indexedGatherPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"runtime-abi-mirror", "materialized-store-base"});
  addBinding(indexedGatherPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              indexedGatherPlan,
              "rvv-route-operand-binding:indexed_gather_unit_store.v1",
              "data,index,out,n", "valid indexed gather binding plan"),
          "valid indexed gather route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedGatherIndexAndData = indexedGatherPlan;
  swappedGatherIndexAndData.bindings[1].parameter.role =
      RuntimeABIParameterRole::LHSInputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedGatherIndexAndData,
              "rvv-route-operand-binding:indexed_gather_unit_store.v1",
              "data,index,out,n", "route operand binding unit test"),
          {"logical operand 'index'", "index-input-buffer",
           "lhs-input-buffer"}))
    return result;

  RVVRouteOperandBindingPlan indexedScatterPlan;
  indexedScatterPlan.planID =
      "rvv-route-operand-binding:indexed_scatter_unit_load.v1";
  addBinding(indexedScatterPlan, "src",
             makeTargetExportABIParameter(
                 "src", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"runtime-abi-mirror", "materialized-load-base", "move-source"});
  addBinding(indexedScatterPlan, "index",
             makeTargetExportABIParameter(
                 "index", "const uint32_t *",
                 RuntimeABIParameterRole::IndexInputBuffer),
             {"runtime-abi-mirror", "materialized-index-load-base",
              "index-offset-scale"});
  addBinding(indexedScatterPlan, "dst",
             makeTargetExportABIParameter("dst", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"runtime-abi-mirror", "materialized-indexed-store-base"});
  addBinding(indexedScatterPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              indexedScatterPlan,
              "rvv-route-operand-binding:indexed_scatter_unit_load.v1",
              "src,index,dst,n", "valid indexed scatter binding plan"),
          "valid indexed scatter route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedScatterDestination = indexedScatterPlan;
  swappedScatterDestination.bindings[2].parameter.role =
      RuntimeABIParameterRole::LHSInputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedScatterDestination,
              "rvv-route-operand-binding:indexed_scatter_unit_load.v1",
              "src,index,dst,n", "route operand binding unit test"),
          {"logical operand 'dst'", "output-buffer", "lhs-input-buffer"}))
    return result;

  RVVRouteOperandBindingPlan segment2DeinterleavePlan;
  segment2DeinterleavePlan.planID =
      "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1";
  addBinding(segment2DeinterleavePlan, "src",
             makeTargetExportABIParameter(
                 "src", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"runtime-abi-mirror", "seg-load-base", "src-mem"});
  addBinding(segment2DeinterleavePlan, "out0",
             makeTargetExportABIParameter(
                 "out0", "int32_t *",
                 RuntimeABIParameterRole::SegmentField0OutputBuffer),
             {"runtime-abi-mirror", "field0-store-base", "field0-role",
              "dst-mem"});
  addBinding(segment2DeinterleavePlan, "out1",
             makeTargetExportABIParameter(
                 "out1", "int32_t *",
                 RuntimeABIParameterRole::SegmentField1OutputBuffer),
             {"runtime-abi-mirror", "field1-store-base", "field1-role",
              "dst-mem"});
  addBinding(segment2DeinterleavePlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              segment2DeinterleavePlan,
              "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1",
              "src,out0,out1,n",
              "valid segment2 deinterleave binding plan"),
          "valid segment2 deinterleave route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedDeinterleaveField = segment2DeinterleavePlan;
  swappedDeinterleaveField.bindings[1].parameter.role =
      RuntimeABIParameterRole::SegmentField1OutputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedDeinterleaveField,
              "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1",
              "src,out0,out1,n", "route operand binding unit test"),
          {"logical operand 'out0'", "segment-field0-output-buffer",
           "segment-field1-output-buffer"}))
    return result;

  RVVRouteOperandBindingPlan segment2InterleavePlan;
  segment2InterleavePlan.planID =
      "rvv-route-operand-binding:segment2_interleave_unit_load.v1";
  addBinding(segment2InterleavePlan, "src0",
             makeTargetExportABIParameter(
                 "src0", "const int32_t *",
                 RuntimeABIParameterRole::SegmentField0InputBuffer),
             {"runtime-abi-mirror", "field0-load-base", "field0-role",
              "src0-mem", "tuple-field0"});
  addBinding(segment2InterleavePlan, "src1",
             makeTargetExportABIParameter(
                 "src1", "const int32_t *",
                 RuntimeABIParameterRole::SegmentField1InputBuffer),
             {"runtime-abi-mirror", "field1-load-base", "field1-role",
              "src1-mem", "tuple-field1"});
  addBinding(segment2InterleavePlan, "dst",
             makeTargetExportABIParameter(
                 "dst", "int32_t *",
                 RuntimeABIParameterRole::SegmentInterleavedOutputBuffer),
             {"runtime-abi-mirror", "seg-store-base", "dst-mem"});
  addBinding(segment2InterleavePlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              segment2InterleavePlan,
              "rvv-route-operand-binding:segment2_interleave_unit_load.v1",
              "src0,src1,dst,n", "valid segment2 interleave binding plan"),
          "valid segment2 interleave route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedInterleaveField = segment2InterleavePlan;
  swappedInterleaveField.bindings[0].parameter.role =
      RuntimeABIParameterRole::SegmentField1InputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedInterleaveField,
              "rvv-route-operand-binding:segment2_interleave_unit_load.v1",
              "src0,src1,dst,n", "route operand binding unit test"),
          {"logical operand 'src0'", "segment-field0-input-buffer",
           "segment-field1-input-buffer"}))
    return result;

  RVVRouteOperandBindingPlan scalarBroadcastPlan;
  scalarBroadcastPlan.planID =
      "rvv-route-operand-binding:scalar_broadcast_add.v1";
  addBinding(scalarBroadcastPlan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"runtime-abi-mirror", "materialized-load-base",
              "scalar-broadcast-lhs-call", "header-mirror"});
  addBinding(scalarBroadcastPlan, "rhs_scalar",
             makeTargetExportABIParameter("rhs_scalar", "int32_t",
                                          RuntimeABIParameterRole::RHSScalarValue),
             {"runtime-abi-mirror", "scalar-broadcast-rhs-call",
              "header-mirror"});
  addBinding(scalarBroadcastPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"runtime-abi-mirror", "materialized-store-base",
              "header-mirror"});
  addBinding(scalarBroadcastPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control",
              "header-mirror"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              scalarBroadcastPlan,
              "rvv-route-operand-binding:scalar_broadcast_add.v1",
              "lhs,rhs_scalar,out,n", "route operand binding unit test"),
          "valid scalar broadcast route operand binding plan"))
    return result;
  auto scalarBroadcastRHSHeader =
      tianchenrv::plugin::rvv::getRVVRouteOperandBindingParameter(
          scalarBroadcastPlan, "rhs_scalar", "header-mirror",
          "scalar broadcast RHS scalar header mirror lookup");
  if (!scalarBroadcastRHSHeader)
    return fail(llvm::Twine("valid scalar broadcast RHS scalar header mirror "
                            "lookup: ") +
                llvm::toString(scalarBroadcastRHSHeader.takeError()));

  RVVRouteOperandBindingPlan missingScalarHeader = scalarBroadcastPlan;
  missingScalarHeader.bindings[1].materializedUses.pop_back();
  auto missingScalarHeaderLookup =
      tianchenrv::plugin::rvv::getRVVRouteOperandBindingParameter(
          missingScalarHeader, "rhs_scalar", "header-mirror",
          "scalar broadcast RHS scalar header mirror lookup");
  if (missingScalarHeaderLookup)
    return fail("expected missing scalar broadcast RHS scalar header mirror "
                "lookup to fail");
  if (int result = expectErrorContains(
          missingScalarHeaderLookup.takeError(),
          {"requires logical operand 'rhs_scalar'",
           "materialized use 'header-mirror'"}))
    return result;

  RVVRouteOperandBindingPlan badScalarRole = scalarBroadcastPlan;
  badScalarRole.bindings[1].parameter.role =
      RuntimeABIParameterRole::RHSInputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              badScalarRole,
              "rvv-route-operand-binding:scalar_broadcast_add.v1",
              "lhs,rhs_scalar,out,n", "route operand binding unit test"),
          {"logical operand 'rhs_scalar'", "rhs-scalar-value",
           "rhs-input-buffer"}))
    return result;

  RVVRouteOperandBindingPlan standaloneReductionPlan;
  standaloneReductionPlan.planID =
      "rvv-route-operand-binding:standalone_reduce_add.v1";
  addBinding(standaloneReductionPlan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"runtime-abi-mirror", "materialized-load-base"});
  addBinding(standaloneReductionPlan, "acc",
             makeTargetExportABIParameter(
                 "acc", "const int32_t *",
                 RuntimeABIParameterRole::AccumulatorInputBuffer),
             {"runtime-abi-mirror", "standalone-initial-accumulator-call"});
  addBinding(standaloneReductionPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"runtime-abi-mirror", "standalone-accumulator-state-load",
              "materialized-store-base"});
  addBinding(standaloneReductionPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              standaloneReductionPlan,
              "rvv-route-operand-binding:standalone_reduce_add.v1",
              "lhs,acc,out,n", "route operand binding unit test"),
          "valid standalone reduction route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan maskedLoadStorePlan;
  maskedLoadStorePlan.planID =
      "rvv-route-operand-binding:masked_unit_load_store.v1";
  addBinding(maskedLoadStorePlan, "src",
             makeTargetExportABIParameter(
                 "src", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"runtime-abi-mirror", "materialized-masked-load-base",
              "masked-load-source-call", "header-mirror"});
  addBinding(maskedLoadStorePlan, "mask",
             makeTargetExportABIParameter(
                 "mask", "const int32_t *",
                 RuntimeABIParameterRole::MaskInputBuffer),
             {"runtime-abi-mirror", "materialized-mask-load-base",
              "masked-load-mask-call"});
  addBinding(maskedLoadStorePlan, "dst",
             makeTargetExportABIParameter(
                 "dst", "int32_t *",
                 RuntimeABIParameterRole::OutputBuffer),
             {"runtime-abi-mirror", "materialized-old-destination-load-base",
              "masked-load-passthrough-call", "materialized-store-base",
              "header-mirror"});
  addBinding(maskedLoadStorePlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control",
              "header-mirror"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              maskedLoadStorePlan,
              "rvv-route-operand-binding:masked_unit_load_store.v1",
              "src,mask,dst,n", "route operand binding unit test"),
          "valid masked load-store route operand binding plan"))
    return result;

  RVVSelectedBodyEmitCRouteDescription maskedLoadStoreDescription =
      makeClosureDescription(RVVSelectedBodyOperationKind::MaskedUnitLoadStore,
                             "src,mask,dst,n", maskedLoadStorePlan);
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingClosure(
              maskedLoadStorePlan, maskedLoadStoreDescription,
              "masked load-store closure unit test"),
          "valid masked load-store route operand binding closure"))
    return result;

  RVVSelectedBodyEmitCRouteDescription staleMaskedLoadStoreDescription =
      maskedLoadStoreDescription;
  staleMaskedLoadStoreDescription.routeOperandBindingSummary = "stale";
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingClosure(
              maskedLoadStorePlan, staleMaskedLoadStoreDescription,
              "masked load-store closure unit test"),
          {"mirror summary", "stale"}))
    return result;

  RVVRouteOperandBindingPlan missingMaskedLoadStorePlan =
      maskedLoadStorePlan;
  missingMaskedLoadStorePlan.planID.clear();
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingClosure(
              missingMaskedLoadStorePlan, maskedLoadStoreDescription,
              "masked load-store closure unit test"),
          {"requires plan id",
           "rvv-route-operand-binding:masked_unit_load_store.v1"}))
    return result;

  RVVSelectedBodyEmitCRouteDescription wrongMaskedLoadStoreMirror =
      maskedLoadStoreDescription;
  wrongMaskedLoadStoreMirror.routeOperandBindingPlanID =
      "rvv-route-operand-binding:masked_unit_store.v1";
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingClosure(
              maskedLoadStorePlan, wrongMaskedLoadStoreMirror,
              "masked load-store closure unit test"),
          {"mirror plan id",
           "rvv-route-operand-binding:masked_unit_load_store.v1",
           "rvv-route-operand-binding:masked_unit_store.v1"}))
    return result;

  RVVSelectedBodyEmitCRouteDescription swappedRuntimeMirror =
      maskedLoadStoreDescription;
  std::swap(swappedRuntimeMirror.runtimeABIParameters[0],
            swappedRuntimeMirror.runtimeABIParameters[1]);
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingClosure(
              maskedLoadStorePlan, swappedRuntimeMirror,
              "masked load-store closure unit test"),
          {"runtime ABI parameter mirrors", "binding plan"}))
    return result;

  RVVRouteOperandBindingPlan swappedMaskedLoadMask = maskedLoadStorePlan;
  swappedMaskedLoadMask.bindings[1].parameter.role =
      RuntimeABIParameterRole::RHSInputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedMaskedLoadMask,
              "rvv-route-operand-binding:masked_unit_load_store.v1",
              "src,mask,dst,n", "route operand binding unit test"),
          {"logical operand 'mask'", "mask-input-buffer",
           "rhs-input-buffer"}))
    return result;

  RVVRouteOperandBindingPlan maskedStorePlan;
  maskedStorePlan.planID = "rvv-route-operand-binding:masked_unit_store.v1";
  addBinding(maskedStorePlan, "src",
             makeTargetExportABIParameter(
                 "src", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"runtime-abi-mirror", "materialized-load-base"});
  addBinding(maskedStorePlan, "mask",
             makeTargetExportABIParameter(
                 "mask", "const int32_t *",
                 RuntimeABIParameterRole::MaskInputBuffer),
             {"runtime-abi-mirror", "materialized-mask-load-base"});
  addBinding(maskedStorePlan, "dst",
             makeTargetExportABIParameter("dst", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"runtime-abi-mirror", "materialized-masked-store-base"});
  addBinding(maskedStorePlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"runtime-abi-mirror", "setvl-avl", "loop-control"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              maskedStorePlan,
              "rvv-route-operand-binding:masked_unit_store.v1",
              "src,mask,dst,n", "route operand binding unit test"),
          "valid masked store route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan computedMaskUnitLoadStorePlan;
  computedMaskUnitLoadStorePlan.planID =
      "rvv-route-operand-binding:computed_masked_unit_load_store.v1";
  addBinding(computedMaskUnitLoadStorePlan, "cmp_lhs",
             makeTargetExportABIParameter(
                 "cmp_lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi-mirror", "cmp-lhs-load", "compare-lhs-call"});
  addBinding(computedMaskUnitLoadStorePlan, "cmp_rhs",
             makeTargetExportABIParameter(
                 "cmp_rhs", "const int32_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi-mirror", "cmp-rhs-load", "compare-rhs-call"});
  addBinding(computedMaskUnitLoadStorePlan, "src",
             makeTargetExportABIParameter(
                 "src", "const int32_t *",
                 RuntimeABIParameterRole::SourceInputBuffer),
             {"abi-mirror", "materialized-masked-load-base",
              "masked-load-source-call"});
  addBinding(computedMaskUnitLoadStorePlan, "dst",
             makeTargetExportABIParameter(
                 "dst", "int32_t *",
                 RuntimeABIParameterRole::OutputBuffer),
             {"abi-mirror", "old-dst-load", "masked-load-passthrough-call",
              "materialized-store-base", "header-mirror"});
  addBinding(computedMaskUnitLoadStorePlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi-mirror", "setvl-avl", "loop-control", "header-mirror"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              computedMaskUnitLoadStorePlan,
              "rvv-route-operand-binding:computed_masked_unit_load_store.v1",
              "cmp_lhs,cmp_rhs,src,dst,n",
              "route operand binding unit test"),
          "valid computed-mask unit load-store route operand binding plan"))
    return result;

  RVVSelectedBodyEmitCRouteDescription computedMaskUnitLoadStoreDescription =
      makeClosureDescription(
          RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore,
          "cmp_lhs,cmp_rhs,src,dst,n", computedMaskUnitLoadStorePlan);
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingClosure(
              computedMaskUnitLoadStorePlan,
              computedMaskUnitLoadStoreDescription,
              "computed-mask unit load-store closure unit test"),
          "valid computed-mask unit load-store route operand binding closure"))
    return result;

  RVVSelectedBodyEmitCRouteDescription wrongComputedMaskOperation =
      makeClosureDescription(
          RVVSelectedBodyOperationKind::ComputedMaskStridedStore,
          "cmp_lhs,cmp_rhs,src,dst,n", computedMaskUnitLoadStorePlan);
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingClosure(
              computedMaskUnitLoadStorePlan, wrongComputedMaskOperation,
              "computed-mask unit load-store closure unit test"),
          {"requires plan id",
           "rvv-route-operand-binding:computed_masked_strided_store.v1"}))
    return result;

  llvm::Expected<const tianchenrv::support::RuntimeABIParameter *>
      missingComputedMaskStoreUse = tianchenrv::plugin::rvv::
          getRVVRouteOperandBindingParameter(
              computedMaskUnitLoadStorePlan, "dst", "strided-store",
              "route operand binding unit test");
  if (int result =
          expectErrorContains(missingComputedMaskStoreUse.takeError(),
                              {"materialized use", "strided-store"}))
    return result;

  RVVRouteOperandBindingPlan computedMaskStridedStorePlan;
  computedMaskStridedStorePlan.planID =
      "rvv-route-operand-binding:computed_masked_strided_store.v1";
  addBinding(computedMaskStridedStorePlan, "cmp_lhs",
             makeTargetExportABIParameter(
                 "cmp_lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi-mirror", "cmp-lhs-load", "cmp-lhs-call"});
  addBinding(computedMaskStridedStorePlan, "cmp_rhs",
             makeTargetExportABIParameter(
                 "cmp_rhs", "const int32_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi-mirror", "cmp-rhs-load", "cmp-rhs-call"});
  addBinding(computedMaskStridedStorePlan, "src",
             makeTargetExportABIParameter(
                 "src", "const int32_t *",
                 RuntimeABIParameterRole::SourceInputBuffer),
             {"abi-mirror", "src-load", "mstr-store-src-call"});
  addBinding(computedMaskStridedStorePlan, "dst",
             makeTargetExportABIParameter("dst", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi-mirror", "mstr-store-base", "header-mirror"});
  addBinding(computedMaskStridedStorePlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi-mirror", "setvl-avl", "loop-control"});
  addBinding(computedMaskStridedStorePlan, "dst_stride_bytes",
             makeTargetExportABIParameter(
                 "dst_stride_bytes", "size_t",
                 RuntimeABIParameterRole::DestinationByteStride),
             {"abi-mirror", "mstr-store-stride", "byte", "header-mirror"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              computedMaskStridedStorePlan,
              "rvv-route-operand-binding:computed_masked_strided_store.v1",
              "cmp_lhs,cmp_rhs,src,dst,n,dst_stride_bytes",
              "route operand binding unit test"),
          "valid computed-mask strided store route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedComputedStrideAndN =
      computedMaskStridedStorePlan;
  swappedComputedStrideAndN.bindings[4].parameter.role =
      RuntimeABIParameterRole::DestinationByteStride;
  swappedComputedStrideAndN.bindings[5].parameter.role =
      RuntimeABIParameterRole::RuntimeElementCount;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedComputedStrideAndN,
              "rvv-route-operand-binding:computed_masked_strided_store.v1",
              "cmp_lhs,cmp_rhs,src,dst,n,dst_stride_bytes",
              "route operand binding unit test"),
          {"logical operand 'n'", "runtime-element-count",
           "destination-byte-stride"}))
    return result;

  RVVRouteOperandBindingPlan addPlan;
  addPlan.planID = "rvv-route-operand-binding:add.v1";
  addBinding(addPlan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "load-base", "binary-lhs-call"});
  addBinding(addPlan, "rhs",
             makeTargetExportABIParameter(
                 "rhs", "const int32_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi", "load-base", "binary-rhs-call"});
  addBinding(addPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "store-base", "header"});
  addBinding(addPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop-control", "header"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              addPlan, "rvv-route-operand-binding:add.v1", "lhs,rhs,out,n",
              "valid add route operand binding plan"),
          "valid add route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedAddLHSRHS = addPlan;
  swappedAddLHSRHS.bindings[0].parameter.role =
      RuntimeABIParameterRole::RHSInputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedAddLHSRHS, "rvv-route-operand-binding:add.v1",
              "lhs,rhs,out,n", "route operand binding unit test"),
          {"logical operand 'lhs'", "lhs-input-buffer", "rhs-input-buffer"}))
    return result;

  RVVRouteOperandBindingPlan computedMaskSelectPlan;
  computedMaskSelectPlan.planID =
      "rvv-route-operand-binding:computed_mask_select.v1";
  addBinding(computedMaskSelectPlan, "cmp_lhs",
             makeTargetExportABIParameter(
                 "cmp_lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "cmp-lhs-load", "compare-lhs-call"});
  addBinding(computedMaskSelectPlan, "cmp_rhs",
             makeTargetExportABIParameter(
                 "cmp_rhs", "const int32_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi", "cmp-rhs-load", "compare-rhs-call"});
  addBinding(computedMaskSelectPlan, "true_value",
             makeTargetExportABIParameter(
                 "true_value", "const int32_t *",
                 RuntimeABIParameterRole::TrueValueInputBuffer),
             {"abi", "true-load", "select-true-call"});
  addBinding(computedMaskSelectPlan, "false_value",
             makeTargetExportABIParameter(
                 "false_value", "const int32_t *",
                 RuntimeABIParameterRole::FalseValueInputBuffer),
             {"abi", "false-load", "select-false-call"});
  addBinding(computedMaskSelectPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "store-base", "header"});
  addBinding(computedMaskSelectPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop-control", "header"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              computedMaskSelectPlan,
              "rvv-route-operand-binding:computed_mask_select.v1",
              "cmp_lhs,cmp_rhs,true_value,false_value,out,n",
              "valid computed-mask select route operand binding plan"),
          "valid computed-mask select route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedTrueFalse = computedMaskSelectPlan;
  swappedTrueFalse.bindings[2].parameter.role =
      RuntimeABIParameterRole::FalseValueInputBuffer;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedTrueFalse,
              "rvv-route-operand-binding:computed_mask_select.v1",
              "cmp_lhs,cmp_rhs,true_value,false_value,out,n",
              "route operand binding unit test"),
          {"logical operand 'true_value'", "true-value-input-buffer",
           "false-value-input-buffer"}))
    return result;

  RVVRouteOperandBindingPlan stridedAddPlan;
  stridedAddPlan.planID = "rvv-route-operand-binding:strided_add.v1";
  addBinding(stridedAddPlan, "lhs",
             makeTargetExportABIParameter(
                 "lhs", "const int32_t *",
                 RuntimeABIParameterRole::LHSInputBuffer),
             {"abi", "lhs-load-base", "binary-lhs-call"});
  addBinding(stridedAddPlan, "rhs",
             makeTargetExportABIParameter(
                 "rhs", "const int32_t *",
                 RuntimeABIParameterRole::RHSInputBuffer),
             {"abi", "rhs-load-base", "binary-rhs-call"});
  addBinding(stridedAddPlan, "out",
             makeTargetExportABIParameter("out", "int32_t *",
                                          RuntimeABIParameterRole::OutputBuffer),
             {"abi", "store-base", "header"});
  addBinding(stridedAddPlan, "n",
             makeTargetExportABIParameter(
                 "n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount),
             {"abi", "setvl-avl", "loop-control", "header"});
  addBinding(stridedAddPlan, "lhs_stride",
             makeTargetExportABIParameter(
                 "lhs_stride", "size_t",
                 RuntimeABIParameterRole::LHSInputStride),
             {"abi", "lhs-load-stride", "lhs-byte-addr", "header"});
  addBinding(stridedAddPlan, "rhs_stride",
             makeTargetExportABIParameter(
                 "rhs_stride", "size_t",
                 RuntimeABIParameterRole::RHSInputStride),
             {"abi", "rhs-load-stride", "rhs-byte-addr", "header"});
  addBinding(stridedAddPlan, "out_stride",
             makeTargetExportABIParameter(
                 "out_stride", "size_t",
                 RuntimeABIParameterRole::OutputStride),
             {"abi", "store-stride", "out-byte-addr", "header"});
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              stridedAddPlan, "rvv-route-operand-binding:strided_add.v1",
              "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride",
              "valid strided add route operand binding plan"),
          "valid strided add route operand binding plan"))
    return result;

  RVVRouteOperandBindingPlan swappedStridedRHSAndOutStride = stridedAddPlan;
  swappedStridedRHSAndOutStride.bindings[5].parameter.role =
      RuntimeABIParameterRole::OutputStride;
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::verifyRVVRouteOperandBindingPlan(
              swappedStridedRHSAndOutStride,
              "rvv-route-operand-binding:strided_add.v1",
              "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride",
              "route operand binding unit test"),
          {"logical operand 'rhs_stride'", "rhs-input-stride",
           "output-stride"}))
    return result;

  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();
  context.getOrLoadDialect<tianchenrv::tcrv::rvv::TCRVRVVDialect>();

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
  if (int result = runWithVLSelectedLoweringBoundaryTest(context))
    return result;
  if (int result =
          runWithVLSelectedLoweringBoundaryDuplicateRejectionTest(context))
    return result;
  if (int result =
          runStaleWithVLRouteMetadataDoesNotAuthorizeEmissionTest(context))
    return result;
  if (int result = runLMULM2SelectedBodyRouteTest(context))
    return result;
  if (int result = runBroadcastSelectedBodyRouteTest(context))
    return result;
  if (int result = runScalarBroadcastElementwisePlanValidationTest(context))
    return result;
  if (int result = runScalarBroadcastAndSplatRouteFamilyProviderPlanTest())
    return result;
  if (int result = runWideningConversionRouteFamilyProviderPlanTest())
    return result;
  if (int result = runMaskedAddSelectedBodyPolicyRouteTest(context))
    return result;
  if (int result = runContractionTargetLeafProfileValidationTest(context))
    return result;
  if (int result = runComputedMaskSelectRouteFamilyProviderPlanTest())
    return result;
  if (int result = runCompareSelectSelectedBodyRouteTest(context))
    return result;
  if (int result = runOutOfOrderSelectedRoleSequenceRejectionTest(context))
    return result;
  if (int result = runRouteOperandBindingPlanValidationTest())
    return result;

  llvm::outs() << "RVV extension plugin smoke test passed\n";
  return 0;
}
