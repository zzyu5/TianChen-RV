#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;

namespace {

class ProposalPlugin final : public ExtensionPlugin {
public:
  ProposalPlugin(llvm::StringRef name, llvm::StringRef supportCapabilityID,
                 llvm::StringRef proposalName, llvm::StringRef originPlugin,
                 llvm::StringRef requiredCapabilitySymbol,
                 bool enabled = true)
      : name(name.str()), supportCapabilityID(supportCapabilityID.str()),
        proposalName(proposalName.str()), originPlugin(originPlugin.str()),
        requiredCapabilitySymbol(requiredCapabilitySymbol.str()),
        enabled(enabled) {}

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
    proposal.addRequiredCapabilityID(supportCapabilityID);
    if (!requiredCapabilitySymbol.empty())
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
  std::string requiredCapabilitySymbol;
  bool enabled;
  llvm::SmallVector<PluginCapability, 1> capabilities;
  mutable unsigned supportCalls = 0;
  mutable unsigned proposalCalls = 0;
  mutable std::string observedHighLevelOpName;
  mutable std::string observedKernelName;
  mutable std::size_t observedCapabilityCount = 0;
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
                       "generic_vector");
  ProposalPlugin disabled("disabled", "generic.vector", "disabled_path",
                          "disabled", "generic_vector", false);
  ProposalPlugin unsupported("unsupported", "generic.missing",
                             "unsupported_path", "unsupported",
                             "generic_missing");
  ProposalPlugin second("second", "generic.toolchain", "second_path", "second",
                        "generic_toolchain");

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

  ProposalPlugin emptyName("empty-name", "generic.vector", "", "empty-name",
                           "generic_vector");
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
                             "originless_path", "", "generic_vector");
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

  llvm::outs() << "plugin variant proposal smoke test passed\n";
  return 0;
}
