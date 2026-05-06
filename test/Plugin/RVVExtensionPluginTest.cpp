#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
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

using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantProposal;
using tianchenrv::plugin::VariantProposalRequest;
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
                     builtinRegistry.size() == 1,
                 "built-in registration owns a safe RVV plugin lifetime"))
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
      status = "available"
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
          expect(proposalAttributes.size() == 1,
                 "RVV proposal carries one plugin-owned typed attribute"))
    return result;
  if (int result = expect(proposalAttributes[0].getName().getValue() ==
                              tianchenrv::plugin::rvv::getRVVPolicyAttrName(),
                          "RVV proposal attribute name is tcrv_rvv.policy"))
    return result;
  if (int result = expectRVVPolicyAttr(proposalAttributes[0].getValue(),
                                       TailPolicy::Agnostic,
                                       MaskPolicy::Agnostic))
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
  if (int result =
          expect(originAttr && originAttr.getValue() == "rvv-plugin" &&
                     requiresAttr && requiresAttr.size() == 1 &&
                     static_cast<bool>(rvvPolicyAttr),
                 "materialized RVV variant has typed origin, requires, and "
                 "plugin-owned policy"))
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
  if (int result = expect(plan.kind == VariantSelectionKind::FallbackOnly &&
                              plan.fallback == variant &&
                              plan.rankedVariants.size() == 1 &&
                              plan.rankedVariants[0].variant == variant,
                          "selection consumes RVV-origin variant through "
                          "ExtensionPluginRegistry"))
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
      status = "available"
    }
    tcrv.exec.variant @missing_policy attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice"
    } {
    }
    tcrv.exec.variant @wrong_type_policy attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = "not_a_typed_policy"
    } {
    }
    tcrv.exec.variant @inconsistent_policy attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
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
  if (int result = runAvailableRVVEndToEndTest(context))
    return result;
  if (int result = runRVVPolicyLegalityRejectionTest(context))
    return result;
  if (int result = runMalformedProposalRejectionTest(context))
    return result;

  llvm::outs() << "RVV extension plugin smoke test passed\n";
  return 0;
}
