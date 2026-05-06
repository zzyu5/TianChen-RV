#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/Passes.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
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
using tianchenrv::plugin::VariantProposal;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

namespace {

class MaterializationPlugin final : public ExtensionPlugin {
public:
  MaterializationPlugin(llvm::StringRef name, llvm::StringRef supportID,
                        VariantProposal proposal)
      : name(name.str()), supportID(supportID.str()),
        proposal(std::move(proposal)) {}

  llvm::StringRef getName() const override { return name; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool supportsOperation(const VariantProposalRequest &request) const override {
    return request.getHighLevelOp() && request.getKernel() &&
           request.getCapabilities().isCapabilityAvailableByID(supportID);
  }

  llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const override {
    (void)request;
    out.push_back(proposal);
    return llvm::Error::success();
  }

private:
  std::string name;
  std::string supportID;
  VariantProposal proposal;
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
    return fail("expected variant materialization error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("materialization error text missing '") +
                  fragment + "': " + message);
  }
  return 0;
}

mlir::OwningOpRef<mlir::ModuleOp> parseTestModule(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @materialization_anchor attributes {} {
    tcrv.exec.capability @generic_alpha {
      id = "generic.alpha",
      kind = "generic-execution"
    }
    tcrv.exec.capability @generic_beta {
      id = "generic.beta",
      kind = "toolchain",
      status = "available"
    }
  }

  tcrv.exec.kernel @duplicate_anchor attributes {} {
    tcrv.exec.capability @generic_alpha {
      id = "generic.alpha",
      kind = "generic-execution"
    }
    tcrv.exec.variant @existing_path attributes {
      origin = "existing-plugin",
      requires = [@generic_alpha]
    } {
    }
  }
}
)mlir";

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

llvm::SmallVector<VariantOp, 4> collectDirectVariants(KernelOp kernel) {
  llvm::SmallVector<VariantOp, 4> variants;
  if (!kernel || kernel.getBody().empty())
    return variants;

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (auto variant = llvm::dyn_cast<VariantOp>(op))
      variants.push_back(variant);
  }
  return variants;
}

int expectRequires(VariantOp variant,
                   std::initializer_list<llvm::StringRef> symbols) {
  mlir::ArrayAttr requires = variant.getRequiresAttr();
  if (int result = expect(static_cast<bool>(requires),
                          "materialized variant has requires array"))
    return result;
  if (int result = expect(requires.size() == symbols.size(),
                          "materialized requires count matches expectation"))
    return result;

  unsigned index = 0;
  for (llvm::StringRef symbol : symbols) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requires[index++]);
    if (int result =
            expect(static_cast<bool>(symbolRef),
                   "requires entry is a FlatSymbolRefAttr"))
      return result;
    if (int result = expect(symbolRef.getValue() == symbol,
                            llvm::Twine("requires entry maps to @") + symbol))
      return result;
  }

  return 0;
}

int expectStringAttr(VariantOp variant, llvm::StringRef attrName,
                     llvm::StringRef expectedValue) {
  auto attr = variant->getAttrOfType<mlir::StringAttr>(attrName);
  if (int result = expect(static_cast<bool>(attr),
                          llvm::Twine("materialized variant has ") +
                              attrName + " metadata"))
    return result;

  return expect(attr.getValue() == expectedValue,
                llvm::Twine("materialized variant preserves ") + attrName +
                    " metadata");
}

VariantProposal makeProposal(llvm::StringRef name, llvm::StringRef origin) {
  return VariantProposal(name, origin);
}

int expectDirectMaterializationError(
    mlir::MLIRContext &context, VariantProposal proposal,
    std::initializer_list<llvm::StringRef> fragments) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseTestModule(context);
  if (!module)
    return fail("failed to parse test module for negative materialization");

  KernelOp kernel = findKernel(*module, "materialization_anchor");
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(nullptr, kernel, capabilities);
  mlir::OpBuilder builder(&context);

  unsigned beforeCount = collectDirectVariants(kernel).size();
  if (int result = expectErrorContains(
          tianchenrv::transforms::materializeVariantProposals(
              builder, request, llvm::ArrayRef<VariantProposal>(proposal)),
          fragments))
    return result;

  return expect(collectDirectVariants(kernel).size() == beforeCount,
                "failed materialization does not mutate kernel");
}

int runPositiveMaterializationTest(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseTestModule(context);
  if (!module)
    return fail("failed to parse variant materialization test module");

  mlir::func::FuncOp highLevelOp =
      module->lookupSymbol<mlir::func::FuncOp>("high_level_placeholder");
  if (int result =
          expect(static_cast<bool>(highLevelOp),
                 "high-level placeholder operation is present"))
    return result;

  KernelOp kernel = findKernel(*module, "materialization_anchor");
  if (int result = expect(static_cast<bool>(kernel), "kernel is present"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  if (int result =
          expect(capabilities.isCapabilityAvailableByID("generic.alpha"),
                 "alpha capability is available by id"))
    return result;
  if (int result =
          expect(capabilities.isCapabilityAvailableBySymbolName("generic_beta"),
                 "beta capability is available by symbol"))
    return result;

  VariantProposal first = makeProposal("first_path", "first-plugin");
  first.addRequiredCapabilityID("generic.alpha");
  first.setCondition("generic_condition_for_future_dispatch");
  first.setGuard("generic_guard_for_future_dispatch");

  VariantProposal second = makeProposal("second_path", "second-plugin");
  second.addRequiredCapabilitySymbol("generic_beta");
  second.setPolicy("generic_policy_for_future_dispatch");

  MaterializationPlugin firstPlugin("first-plugin", "generic.alpha", first);
  MaterializationPlugin secondPlugin("second-plugin", "generic.beta", second);

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(firstPlugin),
                                 "register first materialization plugin"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(secondPlugin),
                                 "register second materialization plugin"))
    return result;

  VariantProposalRequest request(highLevelOp.getOperation(), kernel,
                                 capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 4> materializedVariants;
  if (int result = expectSuccess(
          tianchenrv::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &materializedVariants),
          "collect and materialize variant proposals"))
    return result;

  if (int result = expect(materializedVariants.size() == 2,
                          "two proposals materialize"))
    return result;
  if (int result = expect(materializedVariants[0].getSymName() ==
                                  "first_path" &&
                              materializedVariants[1].getSymName() ==
                                  "second_path",
                          "materialized order follows proposal order"))
    return result;
  if (int result = expect(materializedVariants[0].getOrigin().value_or("") ==
                                  "first-plugin" &&
                              materializedVariants[1].getOrigin().value_or("") ==
                                  "second-plugin",
                          "origin metadata is preserved"))
    return result;
  if (int result = expectRequires(materializedVariants[0], {"generic_alpha"}))
    return result;
  if (int result = expectRequires(materializedVariants[1], {"generic_beta"}))
    return result;
  if (int result = expectStringAttr(
          materializedVariants[0], "condition",
          "generic_condition_for_future_dispatch"))
    return result;
  if (int result = expectStringAttr(
          materializedVariants[0], "guard",
          "generic_guard_for_future_dispatch"))
    return result;
  if (int result = expectStringAttr(
          materializedVariants[1], "policy",
          "generic_policy_for_future_dispatch"))
    return result;
  if (int result = expect(!materializedVariants[0].getBody().empty() &&
                              !materializedVariants[1].getBody().empty(),
                          "materialized variants have typed MLIR body blocks"))
    return result;

  llvm::SmallVector<VariantOp, 4> directVariants =
      collectDirectVariants(kernel);
  if (int result = expect(directVariants.size() == 2,
                          "variants are nested directly in the kernel"))
    return result;
  if (int result = expect(directVariants[0] == materializedVariants[0] &&
                              directVariants[1] == materializedVariants[1],
                          "kernel child order is deterministic"))
    return result;

  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "materialized module verifies"))
    return result;

  std::string printedModule;
  llvm::raw_string_ostream stream(printedModule);
  module->print(stream);
  stream.flush();
  mlir::OwningOpRef<mlir::ModuleOp> reparsedModule =
      mlir::parseSourceString<mlir::ModuleOp>(printedModule, &context);
  if (int result = expect(static_cast<bool>(reparsedModule),
                          "printed materialized IR parses"))
    return result;
  KernelOp reparsedKernel =
      findKernel(*reparsedModule, "materialization_anchor");
  if (int result = expect(collectDirectVariants(reparsedKernel).size() == 2,
                          "reparsed IR walks as typed variant ops"))
    return result;

  mlir::PassManager passManager(&context);
  passManager.addPass(
      tianchenrv::transforms::createCheckCapabilityRequiresPass());
  if (int result =
          expect(mlir::succeeded(passManager.run(*module)),
                 "check-capability-requires accepts materialized requirements"))
    return result;

  return 0;
}

int runNegativeMaterializationTests(mlir::MLIRContext &context) {
  {
    TargetCapabilitySet emptyCapabilities =
        TargetCapabilitySet::buildFromKernel(KernelOp());
    KernelOp missingKernel;
    VariantProposalRequest request(nullptr, missingKernel, emptyCapabilities);
    mlir::OpBuilder builder(&context);
    VariantProposal proposal = makeProposal("missing_kernel_path", "origin");
    if (int result = expectErrorContains(
            tianchenrv::transforms::materializeVariantProposals(
                builder, request, llvm::ArrayRef<VariantProposal>(proposal)),
            {"kernel anchor"}))
      return result;
  }

  {
    mlir::OwningOpRef<mlir::ModuleOp> module = parseTestModule(context);
    if (!module)
      return fail("failed to parse duplicate test module");
    KernelOp kernel = findKernel(*module, "duplicate_anchor");
    TargetCapabilitySet capabilities =
        TargetCapabilitySet::buildFromKernel(kernel);
    VariantProposalRequest request(nullptr, kernel, capabilities);
    mlir::OpBuilder builder(&context);
    VariantProposal proposal = makeProposal("existing_path", "duplicate");
    proposal.addRequiredCapabilityID("generic.alpha");
    unsigned beforeCount = collectDirectVariants(kernel).size();
    if (int result = expectErrorContains(
            tianchenrv::transforms::materializeVariantProposals(
                builder, request, llvm::ArrayRef<VariantProposal>(proposal)),
            {"duplicate", "existing_path", "duplicate_anchor"}))
      return result;
    if (int result = expect(collectDirectVariants(kernel).size() == beforeCount,
                            "duplicate rejection does not mutate kernel"))
      return result;
  }

  {
    VariantProposal proposal = makeProposal("missing_id_path", "id-origin");
    proposal.addRequiredCapabilityID("generic.missing");
    if (int result = expectDirectMaterializationError(
            context, proposal,
            {"id-origin", "missing_id_path", "required capability id",
             "generic.missing"}))
      return result;
  }

  {
    VariantProposal proposal =
        makeProposal("missing_symbol_path", "symbol-origin");
    proposal.addRequiredCapabilitySymbol("generic_missing");
    if (int result = expectDirectMaterializationError(
            context, proposal,
            {"symbol-origin", "missing_symbol_path",
             "required capability symbol", "generic_missing"}))
      return result;
  }

  {
    VariantProposal proposal = makeProposal("", "empty-name-origin");
    proposal.addRequiredCapabilityID("generic.alpha");
    if (int result = expectDirectMaterializationError(
            context, proposal, {"variant name must be non-empty"}))
      return result;
  }

  {
    VariantProposal proposal = makeProposal("bad name", "bad-name-origin");
    proposal.addRequiredCapabilityID("generic.alpha");
    if (int result = expectDirectMaterializationError(
            context, proposal, {"bad-name-origin", "bad name",
                                "cannot be represented"}))
      return result;
  }

  {
    VariantProposal proposal = makeProposal("empty_origin_path", "");
    proposal.addRequiredCapabilityID("generic.alpha");
    if (int result = expectDirectMaterializationError(
            context, proposal, {"empty_origin_path",
                                "origin plugin must be non-empty"}))
      return result;
  }

  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  dialectRegistry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runPositiveMaterializationTest(context))
    return result;
  if (int result = runNegativeMaterializationTests(context))
    return result;

  llvm::outs() << "variant materialization smoke test passed\n";
  return 0;
}
