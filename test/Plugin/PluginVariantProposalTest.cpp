#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
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
using tianchenrv::plugin::VariantProposalCollectionResult;
using tianchenrv::plugin::VariantProposalDecline;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

namespace {

class ProposalPlugin final : public ExtensionPlugin {
public:
  ProposalPlugin(llvm::StringRef name, llvm::StringRef supportCapabilityID,
                 llvm::StringRef proposalName, llvm::StringRef originPlugin,
                 llvm::StringRef requiredCapabilityID,
                 llvm::StringRef requiredCapabilitySymbol, bool enabled = true,
                 bool includeRequiredCapabilitySymbol = true)
      : name(name.str()), supportCapabilityID(supportCapabilityID.str()),
        proposalName(proposalName.str()), originPlugin(originPlugin.str()),
        requiredCapabilityID(requiredCapabilityID.str()),
        requiredCapabilitySymbol(requiredCapabilitySymbol.str()),
        enabled(enabled),
        includeRequiredCapabilitySymbol(includeRequiredCapabilitySymbol) {}

  llvm::StringRef getName() const override { return name; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return enabled; }

  bool supportsOperation(const VariantProposalRequest &request) const override {
    ++supportCalls;

    if (request.getHighLevelOp())
      observedHighLevelOpName =
          request.getHighLevelOp()->getName().getStringRef().str();
    if (request.getKernel())
      observedKernelName = request.getKernel().getSymName().str();
    observedCapabilityCount = request.getCapabilities().size();

    return request.getHighLevelOp() && request.getKernel() &&
           request.getCapabilities().isCapabilityAvailableByID(
               supportCapabilityID);
  }

  llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const override {
    (void)request;
    ++proposalCalls;

    VariantProposal proposal(proposalName, originPlugin);
    proposal.addRequiredCapabilityID(requiredCapabilityID);
    if (includeRequiredCapabilitySymbol)
      proposal.addRequiredCapabilitySymbol(requiredCapabilitySymbol);
    proposal.setGuard("generic_capability_available");
    out.push_back(proposal);
    return llvm::Error::success();
  }

  unsigned getSupportCalls() const { return supportCalls; }
  unsigned getProposalCalls() const { return proposalCalls; }
  llvm::StringRef getObservedHighLevelOpName() const {
    return observedHighLevelOpName;
  }
  llvm::StringRef getObservedKernelName() const { return observedKernelName; }
  std::size_t getObservedCapabilityCount() const {
    return observedCapabilityCount;
  }

private:
  std::string name;
  std::string supportCapabilityID;
  std::string proposalName;
  std::string originPlugin;
  std::string requiredCapabilityID;
  std::string requiredCapabilitySymbol;
  bool enabled;
  bool includeRequiredCapabilitySymbol;
  llvm::SmallVector<PluginCapability, 1> capabilities;
  mutable unsigned supportCalls = 0;
  mutable unsigned proposalCalls = 0;
  mutable std::string observedHighLevelOpName;
  mutable std::string observedKernelName;
  mutable std::size_t observedCapabilityCount = 0;
};

class DecliningPlugin final : public ExtensionPlugin {
public:
  DecliningPlugin(llvm::StringRef name, llvm::StringRef supportCapabilityID,
                  llvm::StringRef reason)
      : name(name.str()), supportCapabilityID(supportCapabilityID.str()),
        reason(reason.str()) {}

  llvm::StringRef getName() const override { return name; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool supportsOperation(const VariantProposalRequest &request) const override {
    return request.getHighLevelOp() && request.getKernel() &&
           request.getCapabilities().isCapabilityAvailableByID(
               supportCapabilityID);
  }

  llvm::Error
  collectVariantProposals(const VariantProposalRequest &request,
                          VariantProposalCollectionResult &out) const override {
    (void)request;
    out.addRecoverableDecline(name, reason);
    return llvm::Error::success();
  }

private:
  std::string name;
  std::string supportCapabilityID;
  std::string reason;
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
    return fail("expected proposal collection error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("proposal error text missing '") + fragment +
                  "': " + message);
  }
  return 0;
}

KernelOp findKernel(mlir::ModuleOp module) {
  KernelOp kernel;
  module->walk([&](KernelOp candidate) { kernel = candidate; });
  return kernel;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  dialectRegistry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @proposal_source attributes {} {
    tcrv.exec.capability @generic_vector {
      id = "generic.vector",
      kind = "generic-execution"
    }
    tcrv.exec.capability @generic_toolchain {
      id = "generic.toolchain",
      kind = "toolchain"
    }
    tcrv.exec.capability @generic_disabled {
      id = "generic.disabled",
      kind = "runtime",
      status = "disabled"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    return fail("failed to parse plugin proposal test module");

  mlir::func::FuncOp highLevelOp =
      module->lookupSymbol<mlir::func::FuncOp>("high_level_placeholder");
  if (int result =
          expect(static_cast<bool>(highLevelOp),
                 "high-level placeholder operation is present"))
    return result;

  KernelOp kernel = findKernel(*module);
  if (int result = expect(static_cast<bool>(kernel), "kernel is present"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  if (int result =
          expect(capabilities.size() == 3,
                 "capabilities are collected from parsed tcrv.exec.kernel"))
    return result;
  if (int result = expect(capabilities.isCapabilityAvailableByID(
                              "generic.vector"),
                          "generic vector capability is available by id"))
    return result;
  if (int result = expect(!capabilities.isCapabilityAvailableByID(
                              "generic.disabled"),
                          "generic disabled capability is unavailable by id"))
    return result;

  ProposalPlugin first("first", "generic.vector", "first_path", "first",
                       "generic.vector", "generic_vector");
  ProposalPlugin disabled("disabled", "generic.vector", "disabled_path",
                          "disabled", "generic.vector", "generic_vector", false);
  ProposalPlugin unsupported("unsupported", "generic.missing",
                             "unsupported_path", "unsupported",
                             "generic.missing", "generic_missing");
  ProposalPlugin second("second", "generic.toolchain", "second_path", "second",
                        "generic.toolchain", "generic_toolchain");

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(first),
                                 "register first"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(disabled),
                                 "register disabled"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(unsupported),
                                 "register unsupported"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(second),
                                 "register second"))
    return result;

  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  llvm::SmallVector<VariantProposal, 4> proposals;
  if (int result =
          expectSuccess(registry.collectVariantProposals(request, proposals),
                        "collect variant proposals"))
    return result;

  if (int result = expect(proposals.size() == 2,
                          "only enabled supported plugins propose variants"))
    return result;
  if (int result = expect(proposals[0].getVariantName() == "first_path" &&
                              proposals[1].getVariantName() == "second_path",
                          "proposal order follows registration order"))
    return result;
  if (int result = expect(proposals[0].getOriginPlugin() == "first" &&
                              proposals[1].getOriginPlugin() == "second",
                          "proposal origins are preserved"))
    return result;
  if (int result = expect(proposals[0].getRequiredCapabilityIDs().size() == 1 &&
                              proposals[0].getRequiredCapabilityIDs()[0] ==
                                  "generic.vector" &&
                              proposals[0].getRequiredCapabilitySymbols()
                                      .size() == 1 &&
                              proposals[0].getRequiredCapabilitySymbols()[0] ==
                                  "generic_vector" &&
                              proposals[0].getGuard() ==
                                  "generic_capability_available",
                          "proposal metadata is compiler-visible and generic"))
    return result;

  if (int result = expect(first.getSupportCalls() == 1 &&
                              first.getProposalCalls() == 1 &&
                              second.getSupportCalls() == 1 &&
                              second.getProposalCalls() == 1,
                          "supported enabled plugins are queried and invoked"))
    return result;
  if (int result = expect(disabled.getSupportCalls() == 0 &&
                              disabled.getProposalCalls() == 0,
                          "disabled plugin is skipped before support query"))
    return result;
  if (int result = expect(unsupported.getSupportCalls() == 1 &&
                              unsupported.getProposalCalls() == 0,
                          "unsupported plugin proposal hook is not called"))
    return result;
  if (int result = expect(first.getObservedHighLevelOpName() == "func.func" &&
                              first.getObservedKernelName() ==
                                  "proposal_source" &&
                              first.getObservedCapabilityCount() == 3,
                          "plugin observes MLIR op, kernel, and capability set"))
    return result;

  DecliningPlugin recoverableDecline("recoverable-decline", "generic.vector",
                                     "missing plugin-local evidence");
  ProposalPlugin validAfterDecline(
      "valid-after-decline", "generic.vector", "valid_after_decline_path",
      "valid-after-decline", "generic.vector", "generic_vector");
  ExtensionPluginRegistry fallbackPreservingRegistry;
  if (int result =
          expectSuccess(fallbackPreservingRegistry.registerPlugin(
                            recoverableDecline),
                        "register recoverable decline plugin"))
    return result;
  if (int result =
          expectSuccess(fallbackPreservingRegistry.registerPlugin(
                            validAfterDecline),
                        "register valid-after-decline plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> fallbackPreservedProposals;
  llvm::SmallVector<VariantProposalDecline, 1> fallbackDeclines;
  if (int result = expectSuccess(
          fallbackPreservingRegistry.collectVariantProposals(
              request, fallbackPreservedProposals, &fallbackDeclines),
          "collect valid proposal after recoverable decline"))
    return result;
  if (int result =
          expect(fallbackPreservedProposals.size() == 1 &&
                     fallbackPreservedProposals[0].getVariantName() ==
                         "valid_after_decline_path",
                 "recoverable decline preserves later valid proposal"))
    return result;
  if (int result =
          expect(fallbackDeclines.size() == 1 &&
                     fallbackDeclines[0].getPluginName() ==
                         "recoverable-decline" &&
                     fallbackDeclines[0].getReason() ==
                         "missing plugin-local evidence",
                 "recoverable decline diagnostic is preserved"))
    return result;
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> materializedFallbackVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::materializeVariantProposals(
              builder, request, fallbackPreservedProposals,
              &materializedFallbackVariants),
          "materialize valid proposal after recoverable decline"))
    return result;
  if (int result =
          expect(materializedFallbackVariants.size() == 1 &&
                     materializedFallbackVariants[0].getSymName() ==
                         "valid_after_decline_path",
                 "valid proposal after decline remains materializable"))
    return result;

  ProposalPlugin invalidBeforeValid(
      "invalid-before-valid", "generic.vector", "", "invalid-before-valid",
      "generic.vector", "generic_vector");
  ProposalPlugin validAfterInvalid(
      "valid-after-invalid", "generic.vector", "valid_after_invalid_path",
      "valid-after-invalid", "generic.vector", "generic_vector");
  ExtensionPluginRegistry invalidBeforeValidRegistry;
  if (int result = expectSuccess(
          invalidBeforeValidRegistry.registerPlugin(invalidBeforeValid),
          "register invalid-before-valid plugin"))
    return result;
  if (int result = expectSuccess(
          invalidBeforeValidRegistry.registerPlugin(validAfterInvalid),
          "register valid-after-invalid plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> invalidBeforeValidProposals;
  if (int result = expectErrorContains(
          invalidBeforeValidRegistry.collectVariantProposals(
              request, invalidBeforeValidProposals),
          {"invalid-before-valid", "variant name must be non-empty"}))
    return result;

  DecliningPlugin declineFirst("decline-first", "generic.vector",
                               "first bounded decline");
  DecliningPlugin declineSecond("decline-second", "generic.vector",
                                "second bounded decline");
  ExtensionPluginRegistry noViableRegistry;
  if (int result =
          expectSuccess(noViableRegistry.registerPlugin(declineFirst),
                        "register first no-viable decline plugin"))
    return result;
  if (int result =
          expectSuccess(noViableRegistry.registerPlugin(declineSecond),
                        "register second no-viable decline plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> noViableProposals;
  llvm::SmallVector<VariantProposalDecline, 2> noViableDeclines;
  if (int result = expectSuccess(
          noViableRegistry.collectVariantProposals(request, noViableProposals,
                                                   &noViableDeclines),
          "collect deterministic recoverable declines"))
    return result;
  if (int result =
          expect(noViableProposals.empty() && noViableDeclines.size() == 2,
                 "all-decline collection has no valid proposals and two "
                 "diagnostics"))
    return result;
  if (int result =
          expect(noViableDeclines[0].getPluginName() == "decline-first" &&
                     noViableDeclines[0].getReason() ==
                         "first bounded decline" &&
                     noViableDeclines[1].getPluginName() == "decline-second" &&
                     noViableDeclines[1].getReason() ==
                         "second bounded decline",
                 "decline diagnostics preserve registration order"))
    return result;
  if (int result = expectErrorContains(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, noViableRegistry, request),
          {"no viable plugin proposals",
           "decline-first: first bounded decline",
           "decline-second: second bounded decline"}))
    return result;

  ProposalPlugin emptyName("empty-name", "generic.vector", "", "empty-name",
                           "generic.vector", "generic_vector");
  ExtensionPluginRegistry emptyNameRegistry;
  if (int result = expectSuccess(emptyNameRegistry.registerPlugin(emptyName),
                                 "register empty-name plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> invalidNameProposals;
  if (int result = expectErrorContains(
          emptyNameRegistry.collectVariantProposals(request,
                                                    invalidNameProposals),
          {"empty-name", "variant name must be non-empty"}))
    return result;

  ProposalPlugin emptyOrigin("empty-origin", "generic.vector",
                             "originless_path", "", "generic.vector",
                             "generic_vector");
  ExtensionPluginRegistry emptyOriginRegistry;
  if (int result = expectSuccess(
          emptyOriginRegistry.registerPlugin(emptyOrigin),
          "register empty-origin plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> invalidOriginProposals;
  if (int result = expectErrorContains(
          emptyOriginRegistry.collectVariantProposals(request,
                                                      invalidOriginProposals),
          {"empty-origin", "origin plugin must be non-empty"}))
    return result;

  ProposalPlugin emptyRequiredID("empty-required-id", "generic.vector",
                                 "empty_required_id_path", "empty-required-id",
                                 "", "generic_vector");
  ExtensionPluginRegistry emptyRequiredIDRegistry;
  if (int result = expectSuccess(
          emptyRequiredIDRegistry.registerPlugin(emptyRequiredID),
          "register empty-required-id plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> emptyRequiredIDProposals;
  if (int result = expectErrorContains(
          emptyRequiredIDRegistry.collectVariantProposals(
              request, emptyRequiredIDProposals),
          {"empty-required-id", "empty_required_id_path",
           "required capability id must be non-empty"}))
    return result;

  ProposalPlugin unknownRequiredID(
      "unknown-required-id", "generic.vector", "unknown_required_id_path",
      "unknown-required-id", "generic.missing", "generic_vector");
  ExtensionPluginRegistry unknownRequiredIDRegistry;
  if (int result = expectSuccess(
          unknownRequiredIDRegistry.registerPlugin(unknownRequiredID),
          "register unknown-required-id plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> unknownRequiredIDProposals;
  if (int result = expectErrorContains(
          unknownRequiredIDRegistry.collectVariantProposals(
              request, unknownRequiredIDProposals),
          {"unknown-required-id", "unknown_required_id_path",
           "unknown capability id", "generic.missing"}))
    return result;

  ProposalPlugin unavailableRequiredID(
      "unavailable-required-id", "generic.vector",
      "unavailable_required_id_path", "unavailable-required-id",
      "generic.disabled", "generic_vector");
  ExtensionPluginRegistry unavailableRequiredIDRegistry;
  if (int result = expectSuccess(
          unavailableRequiredIDRegistry.registerPlugin(unavailableRequiredID),
          "register unavailable-required-id plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> unavailableRequiredIDProposals;
  if (int result = expectErrorContains(
          unavailableRequiredIDRegistry.collectVariantProposals(
              request, unavailableRequiredIDProposals),
          {"unavailable-required-id", "unavailable_required_id_path",
           "unavailable capability id", "generic.disabled", "generic_disabled",
           "status = \"disabled\""}))
    return result;

  ProposalPlugin emptyRequiredSymbol(
      "empty-required-symbol", "generic.vector", "empty_required_symbol_path",
      "empty-required-symbol", "generic.vector", "");
  ExtensionPluginRegistry emptyRequiredSymbolRegistry;
  if (int result = expectSuccess(
          emptyRequiredSymbolRegistry.registerPlugin(emptyRequiredSymbol),
          "register empty-required-symbol plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> emptyRequiredSymbolProposals;
  if (int result = expectErrorContains(
          emptyRequiredSymbolRegistry.collectVariantProposals(
              request, emptyRequiredSymbolProposals),
          {"empty-required-symbol", "empty_required_symbol_path",
           "required capability symbol reference must be non-empty"}))
    return result;

  ProposalPlugin unknownRequiredSymbol(
      "unknown-required-symbol", "generic.vector",
      "unknown_required_symbol_path", "unknown-required-symbol",
      "generic.vector", "generic_missing");
  ExtensionPluginRegistry unknownRequiredSymbolRegistry;
  if (int result = expectSuccess(
          unknownRequiredSymbolRegistry.registerPlugin(unknownRequiredSymbol),
          "register unknown-required-symbol plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> unknownRequiredSymbolProposals;
  if (int result = expectErrorContains(
          unknownRequiredSymbolRegistry.collectVariantProposals(
              request, unknownRequiredSymbolProposals),
          {"unknown-required-symbol", "unknown_required_symbol_path",
           "unknown capability symbol", "generic_missing"}))
    return result;

  ProposalPlugin unavailableRequiredSymbol(
      "unavailable-required-symbol", "generic.vector",
      "unavailable_required_symbol_path", "unavailable-required-symbol",
      "generic.vector", "generic_disabled");
  ExtensionPluginRegistry unavailableRequiredSymbolRegistry;
  if (int result = expectSuccess(
          unavailableRequiredSymbolRegistry.registerPlugin(
              unavailableRequiredSymbol),
          "register unavailable-required-symbol plugin"))
    return result;
  llvm::SmallVector<VariantProposal, 1> unavailableRequiredSymbolProposals;
  if (int result = expectErrorContains(
          unavailableRequiredSymbolRegistry.collectVariantProposals(
              request, unavailableRequiredSymbolProposals),
          {"unavailable-required-symbol", "unavailable_required_symbol_path",
           "unavailable capability symbol", "generic_disabled",
           "generic.disabled", "status = \"disabled\""}))
    return result;

  llvm::outs() << "plugin variant proposal smoke test passed\n";
  return 0;
}
