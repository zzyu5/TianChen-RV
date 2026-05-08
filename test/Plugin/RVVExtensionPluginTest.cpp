#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/Passes.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"
#include "TianChenRV/Transforms/VariantSelection.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>
#include <utility>

using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmissionStatus;
using tianchenrv::plugin::VariantProposal;
using tianchenrv::plugin::VariantProposalDecline;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::plugin::rvv::RVVProbeCapabilityFacts;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::rvv::MaskPolicy;
using tianchenrv::tcrv::rvv::PolicyAttr;
using tianchenrv::tcrv::rvv::TailPolicy;
using tianchenrv::transforms::VariantSelectionKind;
using tianchenrv::transforms::VariantSelectionPlan;

namespace {

class MalformedRVVLikePlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return tianchenrv::plugin::rvv::getRVVExtensionPluginName();
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool supportsOperation(const VariantProposalRequest &request) const override {
    return request.getHighLevelOp() && request.getKernel() &&
           request.getCapabilities().isCapabilityAvailableByID(
               tianchenrv::plugin::rvv::getRVVCapabilityID());
  }

  llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const override {
    (void)request;
    VariantProposal proposal("rvv_malformed_first_slice", getName());
    proposal.addRequiredCapabilityID("rvv.missing");
    out.push_back(proposal);
    return llvm::Error::success();
  }

private:
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

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

int expectExpectedErrorContains(llvm::Expected<TargetCapabilitySet> value,
                                std::initializer_list<llvm::StringRef>
                                    fragments) {
  if (value)
    return fail("expected TargetCapabilitySet construction error");
  return expectErrorContains(value.takeError(), fragments);
}

int expectRVVPolicyAttr(mlir::Attribute attribute, TailPolicy expectedTail,
                        MaskPolicy expectedMask) {
  auto policy = llvm::dyn_cast_if_present<PolicyAttr>(attribute);
  if (int result =
          expect(static_cast<bool>(policy),
                 "RVV policy metadata is a typed #tcrv_rvv.policy attribute"))
    return result;
  if (int result = expect(policy.getTail() == expectedTail,
                          "RVV policy tail metadata matches expectation"))
    return result;
  return expect(policy.getMask() == expectedMask,
                "RVV policy mask metadata matches expectation");
}

int expectStringAttr(mlir::Operation *operation, llvm::StringRef attrName,
                     llvm::StringRef expectedValue) {
  auto attr = operation->getAttrOfType<mlir::StringAttr>(attrName);
  if (int result = expect(static_cast<bool>(attr),
                          llvm::Twine("expected string attribute ") + attrName))
    return result;
  return expect(attr.getValue() == expectedValue,
                llvm::Twine("string attribute ") + attrName +
                    " preserves expected value");
}

int expectIntegerAttr(mlir::Operation *operation, llvm::StringRef attrName,
                      int64_t expectedValue) {
  auto attr = operation->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (int result =
          expect(static_cast<bool>(attr),
                 llvm::Twine("expected integer attribute ") + attrName))
    return result;
  return expect(attr.getInt() == expectedValue,
                llvm::Twine("integer attribute ") + attrName +
                    " preserves expected value");
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

int expectProposalIntegerAttr(const VariantProposal &proposal,
                              llvm::StringRef attrName, int64_t expectedValue) {
  auto attr = llvm::dyn_cast_if_present<mlir::IntegerAttr>(
      findProposalAttribute(proposal, attrName));
  if (int result =
          expect(static_cast<bool>(attr),
                 llvm::Twine("proposal carries integer attribute ") + attrName))
    return result;
  return expect(attr.getInt() == expectedValue,
                llvm::Twine("proposal integer attribute ") + attrName +
                    " preserves expected value");
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

mlir::func::FuncOp findHighLevelPlaceholder(mlir::ModuleOp module) {
  return module.lookupSymbol<mlir::func::FuncOp>("high_level_placeholder");
}

VariantProposalRequest makeRequest(mlir::Operation *highLevelOp,
                                   KernelOp kernel,
                                   TargetCapabilitySet &capabilities) {
  return VariantProposalRequest(highLevelOp, kernel, capabilities);
}

RVVProbeCapabilityFacts makeSuccessfulProbeFacts() {
  RVVProbeCapabilityFacts facts;
  facts.architecture = "riscv64";
  facts.hartCount = 64;
  facts.vlenbBytes = 16;
  facts.i32M1LaneCount = 4;
  facts.isaVectorHints =
      "isa: rv64imafdcv_zicsr_zifencei_zihpm_zve32f_zve64d_zvfh_zvl128b";
  facts.clangAvailable = true;
  facts.clangVersion = "clang version 18.1.3";
  facts.cmakeAvailable = true;
  facts.cmakeVersion = "cmake version 3.28.3";
  facts.minimalRVVCompileRunSucceeded = true;
  facts.selectedMarch = "rv64gcv";
  facts.selectedMABI = "lp64d";
  facts.sourceSHA256 =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  facts.binarySHA256 =
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
  return facts;
}

bool containsForbiddenIdentityText(llvm::StringRef text) {
  return text.contains("PASSWORD") || text.contains("hunter2") ||
         text.contains("PRIVATE KEY") || text.contains("ssh rvv") ||
         text.contains("provider") || text.contains("benchmark") ||
         text.contains("latency") || text.contains("throughput") ||
         text.contains("rvv_probe_ok");
}

int expectCapabilityIdentityIsSanitized(
    const CapabilityDescriptor &capability) {
  std::string identity;
  llvm::raw_string_ostream stream(identity);
  stream << capability.getSymbolName() << "|" << capability.getID() << "|"
         << capability.getKind() << "|" << capability.getStatus();
  stream.flush();
  return expect(!containsForbiddenIdentityText(identity),
                "capability identity excludes raw logs, secrets, providers, "
                "and performance measurements");
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
  if (int result = expectErrorContains(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          {"duplicate TianChen-RV extension plugin", "rvv-plugin"}))
    return result;

  llvm::SmallVector<PluginCapability, 4> capabilities;
  registry.collectCapabilities(capabilities);
  if (int result = expect(capabilities.size() == 1,
                          "RVV plugin exposes one first-slice capability"))
    return result;
  if (int result =
          expect(capabilities[0].getID() ==
                         tianchenrv::plugin::rvv::getRVVCapabilityID() &&
                     capabilities[0].getKind() ==
                         tianchenrv::plugin::rvv::getRVVCapabilityKind() &&
                     !capabilities[0].getDescription().empty(),
                 "RVV capability metadata is explicit and documented"))
    return result;
  if (int result = expect(registry.lookupCapabilityByID("rvv") != nullptr,
                          "RVV capability id lookup succeeds"))
    return result;

  ExtensionPluginRegistry builtinRegistry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerBuiltinExtensionPlugins(builtinRegistry),
          "register built-in extension plugins"))
    return result;
  if (int result =
          expect(builtinRegistry.lookupPlugin("rvv-plugin") != nullptr &&
                     builtinRegistry.lookupPlugin("offload-plugin") !=
                         nullptr &&
                     builtinRegistry.lookupPlugin("scalar-plugin") != nullptr &&
                     builtinRegistry.size() == 3,
                 "built-in registration owns safe RVV, offload, and scalar "
                 "plugin lifetimes"))
    return result;
  if (int result = expectErrorContains(
          tianchenrv::plugin::registerBuiltinExtensionPlugins(builtinRegistry),
          {"duplicate TianChen-RV extension plugin", "rvv-plugin"}))
    return result;

  return 0;
}

int runMissingAndUnavailableProposalTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @missing_rvv attributes {} {
    tcrv.exec.capability @generic_base {
      id = "generic.base",
      kind = "toolchain"
    }
  }

  tcrv.exec.kernel @unavailable_rvv attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "disabled"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV missing/unavailable module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  if (int result =
          expect(static_cast<bool>(highLevelOp),
                 "high-level placeholder is present for RVV proposals"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for missing/unavailable test"))
    return result;

  for (llvm::StringRef kernelName : {"missing_rvv", "unavailable_rvv"}) {
    KernelOp kernel = findKernel(*module, kernelName);
    if (int result =
            expect(static_cast<bool>(kernel),
                   llvm::Twine("kernel is present: ") + kernelName))
      return result;

    TargetCapabilitySet capabilities =
        TargetCapabilitySet::buildFromKernel(kernel);
    VariantProposalRequest request =
        makeRequest(highLevelOp.getOperation(), kernel, capabilities);
    llvm::SmallVector<VariantProposal, 1> proposals;
    if (int result =
            expectSuccess(registry.collectVariantProposals(request, proposals),
                          llvm::Twine("collect no RVV proposals for ") +
                              kernelName))
      return result;
    if (int result =
            expect(proposals.empty(),
                   llvm::Twine("RVV plugin proposes nothing without available "
                               "RVV capability for ") +
                       kernelName))
      return result;
  }

  return 0;
}

int runRVVCapabilityProfileTest(mlir::MLIRContext &context) {
  tianchenrv::plugin::rvv::RVVExtensionPlugin plugin;
  llvm::Expected<TargetCapabilitySet> capabilitiesOrError =
      plugin.buildTargetCapabilitiesFromProbeFacts(makeSuccessfulProbeFacts());
  if (!capabilitiesOrError)
    return fail("valid RVV probe facts were rejected: " +
                llvm::toString(capabilitiesOrError.takeError()));

  TargetCapabilitySet capabilities = std::move(*capabilitiesOrError);
  if (int result = expect(capabilities.size() == 9,
                          "RVV probe facts produce deterministic capabilities"))
    return result;
  if (int result = expect(capabilities.isCapabilityAvailableByID("rvv"),
                          "RVV profile exposes available rvv capability"))
    return result;
  if (int result = expect(capabilities.isCapabilityAvailableBySymbolName(
                              tianchenrv::plugin::rvv::
                                  getRVVPreferredCapabilitySymbol()),
                          "RVV profile exposes documented @rvv symbol"))
    return result;

  const CapabilityDescriptor *hartCount = capabilities.lookupByID(
      tianchenrv::plugin::rvv::getRVVHartCountCapabilityID());
  if (int result = expect(hartCount && hartCount->getKind() == "uarch" &&
                              hartCount->getProperty("count") == "64",
                          "RVV profile preserves hart count as uarch fact"))
    return result;
  if (int result = expect(hartCount && hartCount->providesID(
                                           tianchenrv::support::
                                               getTargetHartCountCapabilityID()),
                          "RVV hart count provides generic target hart-count "
                          "capability relation"))
    return result;
  if (int result = expect(capabilities.lookupProviderByID(
                              tianchenrv::support::
                                  getTargetHartCountCapabilityID()) ==
                              hartCount,
                          "generic target hart-count capability resolves to "
                          "the RVV hart-count provider"))
    return result;

  const CapabilityDescriptor *vlenb = capabilities.lookupByID(
      tianchenrv::plugin::rvv::getRVVVLenBBytesCapabilityID());
  if (int result = expect(vlenb && vlenb->getKind() == "uarch" &&
                              vlenb->getProperty("bytes") == "16",
                          "RVV profile preserves vlenb bytes as uarch fact"))
    return result;

  const CapabilityDescriptor *i32Lanes = capabilities.lookupByID(
      tianchenrv::plugin::rvv::getRVVI32M1LaneCountCapabilityID());
  if (int result =
          expect(i32Lanes && i32Lanes->getKind() == "uarch" &&
                     i32Lanes->getProperty("lanes") == "4",
                 "RVV profile preserves i32 m1 lane count as uarch fact"))
    return result;

  const CapabilityDescriptor *clang = capabilities.lookupByID(
      tianchenrv::plugin::rvv::getRVVClangToolchainCapabilityID());
  if (int result = expect(clang && clang->getKind() == "toolchain" &&
                              clang->getProperty("version") ==
                                  "clang version 18.1.3",
                          "RVV profile preserves clang toolchain version"))
    return result;

  const CapabilityDescriptor *cmake = capabilities.lookupByID(
      tianchenrv::plugin::rvv::getRVVCMakeToolchainCapabilityID());
  if (int result = expect(cmake && cmake->getKind() == "toolchain" &&
                              cmake->getProperty("version") ==
                                  "cmake version 3.28.3",
                          "RVV profile preserves cmake toolchain version"))
    return result;

  const CapabilityDescriptor *compileRun = capabilities.lookupByID(
      tianchenrv::plugin::rvv::getRVVProbeCompileRunCapabilityID());
  if (int result = expect(compileRun &&
                              compileRun->getProperty("selected_march") ==
                                  "rv64gcv" &&
                              compileRun->getProperty("selected_mabi") ==
                                  "lp64d",
                          "RVV profile preserves compile/run march and mabi"))
    return result;

  llvm::ArrayRef<CapabilityDescriptor> orderedCapabilities =
      capabilities.getCapabilities();
  if (int result =
          expect(orderedCapabilities[0].getSymbolName() == "rvv" &&
                     orderedCapabilities[1].getSymbolName() ==
                         tianchenrv::plugin::rvv::
                             getRVVHartCountCapabilitySymbol() &&
                     orderedCapabilities[2].getSymbolName() ==
                         tianchenrv::plugin::rvv::
                             getRVVVLenBBytesCapabilitySymbol() &&
                     orderedCapabilities[3].getSymbolName() ==
                         tianchenrv::plugin::rvv::
                             getRVVI32M1LaneCountCapabilitySymbol() &&
                     orderedCapabilities[4].getSymbolName() ==
                         tianchenrv::plugin::rvv::
                             getRVVClangToolchainCapabilitySymbol() &&
                     orderedCapabilities[5].getSymbolName() ==
                         tianchenrv::plugin::rvv::
                             getRVVCMakeToolchainCapabilitySymbol(),
                 "RVV profile capability ordering is deterministic"))
    return result;

  for (const CapabilityDescriptor &capability : orderedCapabilities) {
    if (int result = expectCapabilityIdentityIsSanitized(capability))
      return result;
  }

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @profile_rvv attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV capability profile module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "profile_rvv");
  if (int result = expect(highLevelOp && kernel,
                          "RVV capability profile test has op and kernel"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for capability profile test"))
    return result;

  VariantProposalRequest request =
      makeRequest(highLevelOp.getOperation(), kernel, capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result =
          expectSuccess(registry.collectVariantProposals(request, proposals),
                        "profile-derived capabilities feed RVV proposals"))
    return result;
  if (int result = expect(proposals.size() == 1,
                          "profile-derived RVV capabilities propose variant"))
    return result;
  if (int result = expectProposalStringAttr(
          proposals[0], "tcrv_rvv.required_march", "rv64gcv"))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.vlenb_bytes", 16))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.i32_m1_lanes", 4))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.element_count", 16))
    return result;

  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize RVV proposal from profile capabilities"))
    return result;
  VariantOp variant = materializedVariants.front();

  VariantEmissionStatus readiness;
  if (int result = expectSuccess(
          plugin.checkVariantEmissionReadiness(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              readiness),
          "RVV profile capabilities still route emission readiness"))
    return result;
  if (int result = expect(readiness.isUnsupported() &&
                              readiness.getReason().contains(
                                  "no RVV lowering"),
                          "RVV profile does not enable emission readiness"))
    return result;

  VariantEmissionPlan plan;
  if (int result = expectSuccess(
          plugin.buildVariantEmissionPlan(
              VariantEmissionRequest(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant),
              plan),
          "RVV profile capabilities still route emission plan"))
    return result;
  if (int result = expect(plan.isUnsupported() &&
                              plan.getDiagnostic().contains(
                                  "no RVV lowering pipeline"),
                          "RVV profile does not enable emission plan support"))
    return result;
  if (int result = expect(plan.getRuntimeABIKind() ==
                              "rvv-plugin-deferred-runtime-abi",
                          "RVV unsupported plan records plugin-owned runtime "
                          "ABI kind"))
    return result;
  if (int result = expect(plan.getRuntimeABIName() ==
                              "rvv-executable-runtime-abi-deferred",
                          "RVV unsupported plan records deferred runtime ABI "
                          "name"))
    return result;
  if (int result =
          expect(plan.getRuntimeGlueRole() == "deferred-rvv-runtime-glue",
                 "RVV unsupported plan records runtime glue role"))
    return result;
  if (int result = expect(plan.getRequiredCapabilitySymbols().size() == 1 &&
                              plan.getRequiredCapabilitySymbols().front() ==
                                  "rvv",
                          "RVV unsupported plan preserves required capability "
                          "refs"))
    return result;

  tianchenrv::plugin::VariantCostEstimate estimate;
  if (int result = expectSuccess(
          plugin.estimateVariantCost(
              tianchenrv::plugin::VariantCostRequest(variant, kernel,
                                                     capabilities),
              estimate),
          "RVV profile capacity feeds cost metadata"))
    return result;
  if (int result =
          expect(estimate.getScore() == 0.25 &&
                     estimate.getExplanation().contains("i32_m1_lanes=4") &&
                     estimate.getExplanation().contains(
                         "not a runtime performance claim"),
                 "RVV capacity is exposed only as plugin-local heuristic "
                 "metadata"))
    return result;

  RVVProbeCapabilityFacts wideFacts = makeSuccessfulProbeFacts();
  wideFacts.vlenbBytes = 128;
  wideFacts.i32M1LaneCount = 32;
  llvm::Expected<TargetCapabilitySet> wideCapabilitiesOrError =
      plugin.buildTargetCapabilitiesFromProbeFacts(wideFacts);
  if (!wideCapabilitiesOrError) {
    std::string message = llvm::toString(wideCapabilitiesOrError.takeError());
    return fail(llvm::Twine("wide RVV probe facts build target capabilities: ") +
                message);
  }

  TargetCapabilitySet wideCapabilities = std::move(*wideCapabilitiesOrError);
  VariantProposalRequest wideRequest =
      makeRequest(highLevelOp.getOperation(), kernel, wideCapabilities);
  llvm::SmallVector<VariantProposal, 1> wideProposals;
  if (int result =
          expectSuccess(registry.collectVariantProposals(wideRequest,
                                                        wideProposals),
                        "wide RVV capacity facts feed RVV proposals"))
    return result;
  if (int result =
          expect(wideProposals.size() == 1,
                 "wide RVV capability facts still propose one variant"))
    return result;
  if (int result = expectProposalIntegerAttr(
          wideProposals[0], "tcrv_rvv.i32_m1_lanes", 32))
    return result;
  if (int result = expectProposalIntegerAttr(
          wideProposals[0], "tcrv_rvv.element_count", 64))
    return result;

  return 0;
}

int runRVVCapabilityProfileRejectionTest() {
  tianchenrv::plugin::rvv::RVVExtensionPlugin plugin;

  RVVProbeCapabilityFacts facts = makeSuccessfulProbeFacts();
  facts.isaVectorHints = "";
  if (int result = expectExpectedErrorContains(
          plugin.buildTargetCapabilitiesFromProbeFacts(facts),
          {"ISA/vector hint", "required"}))
    return result;

  facts = makeSuccessfulProbeFacts();
  facts.architecture = "x86_64";
  if (int result = expectExpectedErrorContains(
          plugin.buildTargetCapabilitiesFromProbeFacts(facts),
          {"architecture", "riscv64"}))
    return result;

  facts = makeSuccessfulProbeFacts();
  facts.minimalRVVCompileRunSucceeded = false;
  if (int result = expectExpectedErrorContains(
          plugin.buildTargetCapabilitiesFromProbeFacts(facts),
          {"compile/run", "required"}))
    return result;

  facts = makeSuccessfulProbeFacts();
  facts.clangAvailable = false;
  if (int result = expectExpectedErrorContains(
          plugin.buildTargetCapabilitiesFromProbeFacts(facts),
          {"clang availability", "required"}))
    return result;

  facts = makeSuccessfulProbeFacts();
  facts.cmakeAvailable = false;
  if (int result = expectExpectedErrorContains(
          plugin.buildTargetCapabilitiesFromProbeFacts(facts),
          {"cmake availability", "required"}))
    return result;

  facts = makeSuccessfulProbeFacts();
  facts.vlenbBytes = 18;
  facts.i32M1LaneCount = 4;
  if (int result = expectExpectedErrorContains(
          plugin.buildTargetCapabilitiesFromProbeFacts(facts),
          {"vlenb bytes", "i32 lane multiple"}))
    return result;

  facts = makeSuccessfulProbeFacts();
  facts.i32M1LaneCount = 8;
  if (int result = expectExpectedErrorContains(
          plugin.buildTargetCapabilitiesFromProbeFacts(facts),
          {"i32 m1 lane count", "divided by four"}))
    return result;

  facts = makeSuccessfulProbeFacts();
  facts.clangVersion = "PASSWORD=hunter2";
  if (int result = expectExpectedErrorContains(
          plugin.buildTargetCapabilitiesFromProbeFacts(facts),
          {"clang version", "secret-like"}))
    return result;

  return 0;
}

int runRVVCapabilityPropertyDecisionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral positiveSource = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @mlir_property_rvv attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl256b",
      status = "available",
      vendor_note = "kept-generic"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_vlenb_bytes {
      id = "rvv.vlenb_bytes",
      kind = "uarch",
      bytes = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lanes {
      id = "rvv.i32_m1_lane_count",
      kind = "uarch",
      lanes = 8 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @sophgo_runtime {
      id = "sophgo.runtime",
      kind = "runtime-offload",
      mode = "pcie",
      runtime = "sophgo",
      status = "available"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, positiveSource);
  if (!module)
    return fail("failed to parse RVV property decision positive module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "mlir_property_rvv");
  if (int result = expect(highLevelOp && kernel,
                          "RVV property decision test has op and kernel"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  const CapabilityDescriptor *rvvCapability = capabilities.lookupByID("rvv");
  if (int result =
          expect(rvvCapability &&
                     rvvCapability->getProperty("isa_vector_hints") ==
                         "rv64gcv_zvl256b" &&
                     rvvCapability->getProperty("vendor_note") ==
                         "kept-generic",
                 "RVV plugin sees preserved non-core MLIR properties"))
    return result;
  const CapabilityDescriptor *sophgo =
      capabilities.lookupByID("sophgo.runtime");
  if (int result = expect(sophgo && sophgo->getKind() == "runtime-offload" &&
                              sophgo->getProperty("runtime") == "sophgo" &&
                              sophgo->getProperty("mode") == "pcie",
                          "Sophgo/offload remains generic descriptor data"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for property decision test"))
    return result;

  VariantProposalRequest request =
      makeRequest(highLevelOp.getOperation(), kernel, capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result =
          expectSuccess(registry.collectVariantProposals(request, proposals),
                        "collect RVV proposal from preserved MLIR properties"))
    return result;
  if (int result =
          expect(proposals.size() == 1 &&
                     proposals[0].getCondition() ==
                         "rvv_capability_properties_available" &&
                     proposals[0].getGuard() ==
                         "plugin_local_rvv_property_evidence" &&
                     proposals[0].getRequiredCapabilityIDs().size() == 1 &&
                     proposals[0].getRequiredCapabilityIDs()[0] == "rvv",
                 "RVV property evidence enables proposal metadata"))
    return result;
  if (int result = expectProposalStringAttr(
          proposals[0], "tcrv_rvv.required_march", "rv64gcv"))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.vlenb_bytes", 32))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.i32_m1_lanes", 8))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.element_count", 32))
    return result;

  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize RVV property-enabled proposal"))
    return result;
  if (int result = expect(materializedVariants.size() == 1,
                          "materialized one property-enabled RVV variant"))
    return result;

  VariantOp variant = materializedVariants.front();
  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (int result =
          expect(originAttr && originAttr.getValue() == "rvv-plugin" &&
                     requiresAttr && requiresAttr.size() == 1,
                 "materialized RVV variant preserves origin and requires"))
    return result;
  auto requiredSymbol =
      llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiresAttr[0]);
  if (int result =
          expect(requiredSymbol && requiredSymbol.getValue() == "rvv",
                 "materialized RVV variant keeps generic @rvv requirement"))
    return result;
  if (int result =
          expectSuccess(registry.verifyVariantLegality(
                            tianchenrv::plugin::VariantLegalityRequest(
                                variant, kernel, capabilities)),
                        "RVV legality accepts property-enabled variant"))
    return result;

  auto variantVLenB =
      variant->getAttrOfType<mlir::IntegerAttr>("tcrv_rvv.vlenb_bytes");
  auto variantI32Lanes =
      variant->getAttrOfType<mlir::IntegerAttr>("tcrv_rvv.i32_m1_lanes");
  if (int result =
          expect(variantVLenB && variantVLenB.getInt() == 32 &&
                     variantI32Lanes && variantI32Lanes.getInt() == 8,
                 "materialized RVV variant preserves capability-derived "
                 "capacity metadata"))
    return result;
  if (int result = expectIntegerAttr(variant.getOperation(),
                                     "tcrv_rvv.element_count", 32))
    return result;

  auto expectProposalDecline =
      [&](llvm::StringRef kernelName, llvm::StringRef source,
          std::initializer_list<llvm::StringRef> fragments) -> int {
    mlir::OwningOpRef<mlir::ModuleOp> negativeModule =
        parseModule(context, source);
    if (!negativeModule)
      return fail(llvm::Twine("failed to parse negative RVV property module ") +
                  kernelName);

    mlir::func::FuncOp negativeHighLevelOp =
        findHighLevelPlaceholder(*negativeModule);
    KernelOp negativeKernel = findKernel(*negativeModule, kernelName);
    TargetCapabilitySet negativeCapabilities =
        TargetCapabilitySet::buildFromKernel(negativeKernel);
    VariantProposalRequest negativeRequest =
        makeRequest(negativeHighLevelOp.getOperation(), negativeKernel,
                    negativeCapabilities);
    llvm::SmallVector<VariantProposal, 1> negativeProposals;
    llvm::SmallVector<VariantProposalDecline, 1> negativeDeclines;
    if (int result = expectSuccess(
            registry.collectVariantProposals(negativeRequest, negativeProposals,
                                             &negativeDeclines),
            llvm::Twine("collect recoverable RVV proposal decline for ") +
                kernelName))
      return result;
    if (int result =
            expect(negativeProposals.empty(),
                   llvm::Twine("RVV recoverable decline proposes nothing for ") +
                       kernelName))
      return result;
    if (int result =
            expect(negativeDeclines.size() == 1 &&
                       negativeDeclines[0].getPluginName() ==
                           tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
                   llvm::Twine("one RVV decline diagnostic is collected for ") +
                       kernelName))
      return result;

    llvm::StringRef message = negativeDeclines[0].getReason();
    for (llvm::StringRef fragment : fragments) {
      if (!message.contains(fragment))
        return fail(llvm::Twine("decline text missing '") + fragment +
                    "': " + message);
    }
    return 0;
  };

  if (int result = expectProposalDecline(
          "missing_hint",
          R"mlir(
module {
  func.func @high_level_placeholder() { return }
  tcrv.exec.kernel @missing_hint attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
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
)mlir",
          {"rvv", "isa_vector_hints", "requires preserved property"}))
    return result;

  if (int result = expectProposalDecline(
          "malformed_harts",
          R"mlir(
module {
  func.func @high_level_placeholder() { return }
  tcrv.exec.kernel @malformed_harts attributes {} {
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
      count = "sixty-four",
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
)mlir",
          {"rvv.hart_count", "count", "positive integer"}))
    return result;

  if (int result = expectProposalDecline(
          "secret_hint",
          R"mlir(
module {
  func.func @high_level_placeholder() { return }
  tcrv.exec.kernel @secret_hint attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "TOKEN=rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
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
)mlir",
          {"isa_vector_hints", "secret-like"}))
    return result;

  if (int result = expectProposalDecline(
          "conflicting_march",
          R"mlir(
module {
  func.func @high_level_placeholder() { return }
  tcrv.exec.kernel @conflicting_march attributes {} {
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
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gc"
    }
  }
}
)mlir",
          {"conflicting RVV property values", "rvv.toolchain.march",
           "selected_march"}))
    return result;

  constexpr llvm::StringLiteral mismatchedCapacitySource = R"mlir(
module {
  tcrv.exec.kernel @mismatched_capacity attributes {} {
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
    tcrv.exec.capability @rvv_vlenb_bytes {
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
    tcrv.exec.variant @wrong_lanes attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.vlenb_bytes = 16 : i64,
      tcrv_rvv.i32_m1_lanes = 8 : i64
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> mismatchedCapacityModule =
      parseModule(context, mismatchedCapacitySource);
  if (!mismatchedCapacityModule)
    return fail("failed to parse mismatched RVV capacity module");
  KernelOp mismatchedCapacityKernel =
      findKernel(*mismatchedCapacityModule, "mismatched_capacity");
  TargetCapabilitySet mismatchedCapacityCapabilities =
      TargetCapabilitySet::buildFromKernel(mismatchedCapacityKernel);
  VariantOp wrongLanes;
  mismatchedCapacityKernel->walk([&](VariantOp candidate) {
    if (candidate.getSymName() == "wrong_lanes")
      wrongLanes = candidate;
  });

  if (int result = expectErrorContains(
          registry.verifyVariantLegality(
              tianchenrv::plugin::VariantLegalityRequest(
                  wrongLanes, mismatchedCapacityKernel,
                  mismatchedCapacityCapabilities)),
          {"capacity metadata", "i32 lanes", "vlenb bytes"}))
    return result;

  constexpr llvm::StringLiteral unsupportedMetadataSource = R"mlir(
module {
  tcrv.exec.kernel @unsupported_metadata attributes {} {
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
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.variant @wrong_march attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gc"
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> unsupportedModule =
      parseModule(context, unsupportedMetadataSource);
  if (!unsupportedModule)
    return fail("failed to parse unsupported RVV metadata module");
  KernelOp unsupportedKernel =
      findKernel(*unsupportedModule, "unsupported_metadata");
  TargetCapabilitySet unsupportedCapabilities =
      TargetCapabilitySet::buildFromKernel(unsupportedKernel);
  VariantOp wrongMarch;
  unsupportedKernel->walk([&](VariantOp candidate) {
    if (candidate.getSymName() == "wrong_march")
      wrongMarch = candidate;
  });

  if (int result = expectErrorContains(
          registry.verifyVariantLegality(
              tianchenrv::plugin::VariantLegalityRequest(
                  wrongMarch, unsupportedKernel, unsupportedCapabilities)),
          {"tcrv_rvv.required_march", "selected_march"}))
    return result;

  return 0;
}

int runRVVModuleTargetProfileCapacityDecisionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.target @module_rvv_capacity_profile {
    id = "rvv.profile.capacity",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.probe.compile_run"],
    architecture = "riscv64",
    isa_vector_hints = "rv64gcv_zvl256b",
    count = 64 : i64,
    bytes = 32 : i64,
    lanes = 8 : i64,
    selected_march = "rv64gcv",
    status = "available"
  }

  tcrv.exec.kernel @profile_capacity_rvv attributes {target = @module_rvv_capacity_profile} {
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse module target-profile RVV capacity module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "profile_capacity_rvv");
  if (int result = expect(highLevelOp && kernel,
                          "module target-profile capacity test has op and "
                          "kernel"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  const CapabilityDescriptor *profile =
      capabilities.lookupBySymbolName("module_rvv_capacity_profile");
  if (int result =
          expect(profile && profile->satisfiesID("rvv") &&
                     profile->satisfiesID("rvv.vlenb_bytes") &&
                     profile->satisfiesID("rvv.i32_m1_lane_count") &&
                     profile->getProperty("bytes") == "32" &&
                     profile->getProperty("lanes") == "8",
                 "module target profile carries relation-provided RVV "
                 "capacity facts"))
    return result;
  const CapabilityDescriptor *vlenbProvider =
      capabilities.lookupProviderByID("rvv.vlenb_bytes");
  const CapabilityDescriptor *i32LaneProvider =
      capabilities.lookupProviderByID("rvv.i32_m1_lane_count");
  if (int result = expect(vlenbProvider == profile &&
                              i32LaneProvider == profile,
                          "RVV capacity provider lookup resolves the module "
                          "target profile relation"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for module target-profile "
                        "capacity test"))
    return result;

  VariantProposalRequest request =
      makeRequest(highLevelOp.getOperation(), kernel, capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result = expectSuccess(
          registry.collectVariantProposals(request, proposals),
          "module target-profile capacity facts feed RVV proposal"))
    return result;
  if (int result =
          expect(proposals.size() == 1,
                 "module target-profile capacity facts propose one RVV "
                 "variant"))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.vlenb_bytes", 32))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.i32_m1_lanes", 8))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.element_count", 32))
    return result;

  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize RVV proposal from module target-profile capacity"))
    return result;
  if (int result =
          expect(materializedVariants.size() == 1,
                 "materialized one RVV variant from module target profile"))
    return result;

  VariantOp variant = materializedVariants.front();
  auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  auto requiredSymbol =
      requiresAttr && !requiresAttr.empty()
          ? llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiresAttr[0])
          : mlir::FlatSymbolRefAttr();
  if (int result =
          expect(requiredSymbol &&
                     requiredSymbol.getValue() ==
                         "module_rvv_capacity_profile",
                 "materialized relation-provider RVV variant requires the "
                 "module target profile symbol"))
    return result;
  if (int result =
          expectSuccess(registry.verifyVariantLegality(
                            tianchenrv::plugin::VariantLegalityRequest(
                                variant, kernel, capabilities)),
                        "RVV legality accepts module profile capacity facts"))
    return result;

  tianchenrv::plugin::VariantCostEstimate estimate;
  if (int result = expectSuccess(
          registry.estimateVariantCost(
              tianchenrv::plugin::VariantCostRequest(variant, kernel,
                                                     capabilities),
              estimate),
          "RVV cost estimate consumes module target-profile capacity facts"))
    return result;
  if (int result =
          expect(estimate.getScore() == 0.125 &&
                     estimate.getExplanation().contains("i32_m1_lanes=8") &&
                     estimate.getExplanation().contains(
                         "not a runtime performance claim"),
                 "module target-profile RVV capacity is only plugin-local "
                 "selection heuristic metadata"))
    return result;

  return 0;
}

int runAvailableRVVEndToEndTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @available_rvv attributes {} {
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
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV available module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "available_rvv");
  if (int result = expect(highLevelOp && kernel,
                          "available RVV test has high-level op and kernel"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  if (int result = expect(capabilities.isCapabilityAvailableByID("rvv"),
                          "RVV capability is available by id"))
    return result;
  if (int result =
          expect(capabilities.isCapabilityAvailableBySymbolName(
                     tianchenrv::plugin::rvv::
                         getRVVPreferredCapabilitySymbol()),
                 "RVV capability is available by documented symbol"))
    return result;
  const CapabilityDescriptor *rvvCapability = capabilities.lookupByID("rvv");
  if (int result =
          expect(rvvCapability &&
                     rvvCapability->getProperty("isa_vector_hints") ==
                         "rv64gcv_zvl128b",
                 "RVV plugin test sees preserved MLIR RVV property"))
    return result;

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for available test"))
    return result;

  VariantProposalRequest request =
      makeRequest(highLevelOp.getOperation(), kernel, capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result =
          expectSuccess(registry.collectVariantProposals(request, proposals),
                        "collect available RVV proposal"))
    return result;
  if (int result = expect(proposals.size() == 1,
                          "available RVV capability produces one proposal"))
    return result;
  if (int result =
          expect(proposals[0].getVariantName() ==
                         tianchenrv::plugin::rvv::
                             getRVVFirstSliceVariantName() &&
                     proposals[0].getOriginPlugin() ==
                         tianchenrv::plugin::rvv::
                             getRVVExtensionPluginName() &&
                     proposals[0].getRequiredCapabilityIDs().size() == 1 &&
                     proposals[0].getRequiredCapabilityIDs()[0] == "rvv" &&
                     !proposals[0].getGuard().empty() &&
                     !proposals[0].getPolicy().empty(),
                 "RVV proposal has deterministic plugin-owned metadata"))
    return result;
  llvm::ArrayRef<mlir::NamedAttribute> proposalAttributes =
      proposals[0].getPluginAttributes();
  if (int result =
          expect(proposalAttributes.size() == 4,
                 "RVV proposal carries typed policy and property evidence "
                 "attributes"))
    return result;
  mlir::Attribute rvvPolicy =
      findProposalAttribute(proposals[0],
                            tianchenrv::plugin::rvv::getRVVPolicyAttrName());
  if (int result = expect(static_cast<bool>(rvvPolicy),
                          "RVV proposal carries tcrv_rvv.policy"))
    return result;
  if (int result = expectRVVPolicyAttr(rvvPolicy, TailPolicy::Agnostic,
                                       MaskPolicy::Agnostic))
    return result;
  if (int result = expectProposalStringAttr(
          proposals[0], "tcrv_rvv.required_march", "rv64gcv"))
    return result;
  if (int result =
          expectProposalStringAttr(proposals[0],
                                   "tcrv_rvv.lowering_descriptor",
                                   "i32-vadd-microkernel.v1"))
    return result;
  if (int result =
          expectProposalIntegerAttr(proposals[0], "tcrv_rvv.element_count", 16))
    return result;

  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "materialize RVV proposal through generic helper"))
    return result;
  if (int result = expect(materializedVariants.size() == 1,
                          "one RVV variant is materialized"))
    return result;

  VariantOp variant = materializedVariants[0];
  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  auto rvvPolicyAttr =
      variant->getAttr(tianchenrv::plugin::rvv::getRVVPolicyAttrName());
  auto requiredMarchAttr =
      variant->getAttrOfType<mlir::StringAttr>("tcrv_rvv.required_march");
  if (int result =
          expect(originAttr && originAttr.getValue() == "rvv-plugin" &&
                     requiresAttr && requiresAttr.size() == 1 &&
                     static_cast<bool>(rvvPolicyAttr) && requiredMarchAttr &&
                     requiredMarchAttr.getValue() == "rv64gcv",
                 "materialized RVV variant has typed origin, requires, and "
                 "plugin-owned property metadata"))
    return result;
  if (int result =
          expectStringAttr(variant.getOperation(),
                           "tcrv_rvv.lowering_descriptor",
                           "i32-vadd-microkernel.v1"))
    return result;
  if (int result = expectIntegerAttr(variant.getOperation(),
                                     "tcrv_rvv.element_count", 16))
    return result;
  if (int result = expectRVVPolicyAttr(rvvPolicyAttr, TailPolicy::Agnostic,
                                       MaskPolicy::Agnostic))
    return result;
  auto requiredSymbol =
      llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiresAttr[0]);
  if (int result =
          expect(requiredSymbol && requiredSymbol.getValue() == "rvv",
                 "materialized RVV variant requires @rvv symbol"))
    return result;
  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "materialized RVV module verifies structurally"))
    return result;

  if (int result =
          expectSuccess(registry.verifyKernelVariantLegality(kernel,
                                                             capabilities),
                        "RVV plugin legality accepts materialized variant"))
    return result;
  if (int result =
          expectStringAttr(variant.getOperation(), "policy",
                           "metadata_only_first_slice"))
    return result;

  mlir::PassManager passManager(&context);
  passManager.addPass(
      tianchenrv::transforms::createCheckCapabilityRequiresPass());
  if (int result = expect(mlir::succeeded(passManager.run(*module)),
                          "generic capability pass accepts RVV variant"))
    return result;

  llvm::Expected<VariantSelectionPlan> planOrError =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         registry);
  if (!planOrError)
    return fail("RVV selection planning failed: " +
                llvm::toString(planOrError.takeError()));

  VariantSelectionPlan plan = std::move(*planOrError);
  if (int result = expect(plan.kind == VariantSelectionKind::StaticVariant &&
                              plan.selectedVariant == variant &&
                              !plan.fallback &&
                              plan.missingFallbackCoverage &&
                              plan.rankedVariants.size() == 1 &&
                              plan.rankedVariants[0].variant == variant,
                          "selection consumes RVV-origin variant without "
                          "inventing a fallback through ExtensionPluginRegistry"))
    return result;

  ExtensionPluginRegistry emptyRegistry;
  llvm::Expected<VariantSelectionPlan> emptyPlan =
      tianchenrv::transforms::planKernelVariantSelection(kernel, capabilities,
                                                         emptyRegistry);
  if (emptyPlan)
    return fail("empty registry unexpectedly planned an RVV-origin variant");
  if (int result = expectErrorContains(
          emptyPlan.takeError(), {"unknown origin plugin", "rvv-plugin"}))
    return result;

  return 0;
}

int runRVVPolicyLegalityRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_policy_legality attributes {} {
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
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.variant @missing_policy attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.variant @wrong_type_policy attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.policy = "not_a_typed_policy"
    } {
    }
    tcrv.exec.variant @inconsistent_policy attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = undisturbed, mask = agnostic>
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV policy legality rejection module");

  KernelOp kernel = findKernel(*module, "rvv_policy_legality");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(registry),
                        "register RVV plugin for policy legality negatives"))
    return result;

  VariantOp missingPolicy;
  VariantOp wrongTypePolicy;
  VariantOp inconsistentPolicy;
  kernel->walk([&](VariantOp variant) {
    if (variant.getSymName() == "missing_policy")
      missingPolicy = variant;
    if (variant.getSymName() == "wrong_type_policy")
      wrongTypePolicy = variant;
    if (variant.getSymName() == "inconsistent_policy")
      inconsistentPolicy = variant;
  });

  if (int result = expectErrorContains(
          registry.verifyVariantLegality(
              tianchenrv::plugin::VariantLegalityRequest(missingPolicy, kernel,
                                                         capabilities)),
          {"tcrv_rvv.policy", "requires typed"}))
    return result;
  if (int result = expectErrorContains(
          registry.verifyVariantLegality(
              tianchenrv::plugin::VariantLegalityRequest(wrongTypePolicy,
                                                         kernel, capabilities)),
          {"tcrv_rvv.policy", "#tcrv_rvv.policy"}))
    return result;
  if (int result = expectErrorContains(
          registry.verifyVariantLegality(
              tianchenrv::plugin::VariantLegalityRequest(
                  inconsistentPolicy, kernel, capabilities)),
          {"tcrv_rvv.policy", "agnostic tail/mask"}))
    return result;

  return 0;
}

int runMalformedProposalRejectionTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @malformed_rvv attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse malformed RVV test module");

  mlir::func::FuncOp highLevelOp = findHighLevelPlaceholder(*module);
  KernelOp kernel = findKernel(*module, "malformed_rvv");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  MalformedRVVLikePlugin malformed;
  ExtensionPluginRegistry registry;
  if (int result =
          expectSuccess(registry.registerPlugin(malformed),
                        "register malformed RVV-like plugin"))
    return result;

  VariantProposalRequest request =
      makeRequest(highLevelOp.getOperation(), kernel, capabilities);
  llvm::SmallVector<VariantProposal, 1> proposals;
  if (int result = expectErrorContains(
          registry.collectVariantProposals(request, proposals),
          {"rvv-plugin", "rvv_malformed_first_slice",
           "unknown capability id", "rvv.missing"}))
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
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(
                            dialectPlugins),
                        "register RVV plugin for dialect context"))
    return result;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(dialectPlugins, dialectRegistry);
  dialectRegistry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runMissingAndUnavailableProposalTest(context))
    return result;
  if (int result = runRVVCapabilityProfileTest(context))
    return result;
  if (int result = runRVVCapabilityProfileRejectionTest())
    return result;
  if (int result = runRVVCapabilityPropertyDecisionTest(context))
    return result;
  if (int result = runRVVModuleTargetProfileCapacityDecisionTest(context))
    return result;
  if (int result = runAvailableRVVEndToEndTest(context))
    return result;
  if (int result = runRVVPolicyLegalityRejectionTest(context))
    return result;
  if (int result = runMalformedProposalRejectionTest(context))
    return result;

  llvm::outs() << "RVV extension plugin smoke test passed\n";
  return 0;
}
