#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
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
       "tcrv_rvv.indexed_store, tcrv_rvv.compare, "
       "tcrv_rvv.masked_binary, tcrv_rvv.select, tcrv_rvv.reduce, "
       "tcrv_rvv.macc, tcrv_rvv.widening_convert, tcrv_rvv.move, "
       "tcrv_rvv.store, and tcrv_rvv.strided_store"});
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
  if (int result = runMaskedAddSelectedBodyPolicyRouteTest(context))
    return result;
  if (int result = runCompareSelectSelectedBodyRouteTest(context))
    return result;
  if (int result = runOutOfOrderSelectedRoleSequenceRejectionTest(context))
    return result;

  llvm::outs() << "RVV extension plugin smoke test passed\n";
  return 0;
}
