#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
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
using tianchenrv::plugin::SourceSeedPassRegistration;
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmissionStatus;
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
  if (int result = expect(registry.lookupCapabilityByID("rvv") != nullptr,
                          "RVV capability id lookup succeeds"))
    return result;

  llvm::SmallVector<SourceSeedPassRegistration, 2> sourceSeedPasses;
  if (int result = expectSuccess(
          registry.collectSourceSeedPasses(sourceSeedPasses),
          "collect RVV source-seed pass registrations"))
    return result;
  if (int result =
          expect(sourceSeedPasses.size() == 1,
                 "RVV plugin exposes one source-seed pass registration"))
    return result;
  if (int result = expect(sourceSeedPasses.front().getOwnerPlugin() ==
                              tianchenrv::plugin::rvv::
                                  getRVVExtensionPluginName(),
                          "RVV source-seed pass is owned by RVV plugin"))
    return result;
  if (int result =
          expect(sourceSeedPasses.front().getArgument() ==
                     "tcrv-rvv-materialize-i32m1-selected-boundary-seed",
                 "RVV source-seed pass keeps the public pass argument"))
    return result;
  return expect(static_cast<bool>(sourceSeedPasses.front().getFactory()),
                "RVV source-seed pass factory is present");
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
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }

    tcrv.exec.variant @rvv_i32_sub attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %diff = tcrv_rvv.i32_sub %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %diff, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }

    tcrv.exec.variant @rvv_i32_mul attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %product = tcrv_rvv.i32_mul %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %product, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV with_vl boundary module");
  KernelOp kernel = findKernel(*module, "rvv_i32m1_boundary_kernel");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for with_vl boundary test"))
    return result;

  mlir::OpBuilder builder(module->getContext());
  for (llvm::StringRef variantName : {"rvv_i32_add", "rvv_i32_sub",
                                      "rvv_i32_mul"}) {
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
    if (int result =
            expect(boundaryResult.getMaterializedOperation() == withVL,
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
    if (int result =
            expect(plan.isUnsupported(),
                   llvm::Twine("RVV emission plan is unsupported for @") +
                       variantName))
      return result;
    if (int result = expect(
            llvm::StringRef(plan.getDiagnostic())
                .contains("target artifact route authority has been deleted"),
            llvm::Twine("RVV emission plan reports deleted target artifact "
                        "route for @") +
                variantName))
      return result;

    VariantEmissionStatus readiness;
    if (int result = expectErrorContains(
            registry.checkVariantEmissionReadiness(
                VariantEmissionRequest(variant, kernel, capabilities,
                                       VariantEmissionRole::DirectVariant),
                readiness),
            {"reported unsupported emission path",
             "target artifact route authority has been deleted"}))
      return result;
  }

  VariantOp addVariant = findVariant(kernel, "rvv_i32_add");
  mlir::Operation *setvl = findFirstNestedOp(addVariant, "tcrv_rvv.setvl");
  return expectErrorContains(
      registry.validateSelectedLoweringBoundary(
          VariantLoweringBoundaryValidationRequest(
              addVariant, kernel, capabilities,
              VariantEmissionRole::DirectVariant, setvl)),
      {"selected RVV i32m1 lowering boundary must be the existing "
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
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
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
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for duplicate boundary test"))
    return result;

  mlir::OpBuilder builder(module->getContext());
  VariantLoweringBoundaryResult boundaryResult;
  return expectErrorContains(
      registry.materializeSelectedLoweringBoundary(
          VariantLoweringBoundaryRequest(
              variant, kernel, capabilities,
              VariantEmissionRole::DirectVariant, builder),
          boundaryResult),
      {"selected RVV i32m1 lowering boundary requires exactly one "
       "tcrv_rvv.with_vl op"});
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

  llvm::outs() << "RVV extension plugin smoke test passed\n";
  return 0;
}
