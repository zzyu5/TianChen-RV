#include "Weft/InitWeftDialects.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Transforms/VariantMaterialization.h"
#include "Weft/Transforms/VariantSelection.h"

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
#include <utility>

using weft::plugin::ExtensionPlugin;
using weft::plugin::ExtensionPluginRegistry;
using weft::plugin::FamilyConstructionRequest;
using weft::plugin::FamilyConstructionResult;
using weft::plugin::PluginCapability;
using weft::plugin::VariantCostEstimate;
using weft::plugin::VariantCostRequest;
using weft::plugin::VariantLegalityRequest;
using weft::plugin::VariantProposal;
using weft::plugin::VariantProposalCollectionResult;
using weft::plugin::VariantProposalDecline;
using weft::plugin::VariantProposalRequest;
using weft::support::TargetCapabilitySet;
using weft::exec::KernelOp;
using weft::exec::VariantOp;

namespace {

class ProposalPlugin final : public ExtensionPlugin {
public:
  ProposalPlugin(llvm::StringRef name, llvm::StringRef supportCapabilityID,
                 llvm::StringRef proposalName, llvm::StringRef originPlugin,
                 llvm::StringRef requiredCapabilityID,
                 llvm::StringRef requiredCapabilitySymbol, bool enabled = true,
                 bool includeRequiredCapabilitySymbol = true,
                 llvm::StringRef constructionDomain = "test-domain")
      : name(name.str()), supportCapabilityID(supportCapabilityID.str()),
        proposalName(proposalName.str()), originPlugin(originPlugin.str()),
        requiredCapabilityID(requiredCapabilityID.str()),
        requiredCapabilitySymbol(requiredCapabilitySymbol.str()),
        enabled(enabled),
        includeRequiredCapabilitySymbol(includeRequiredCapabilitySymbol),
        constructionDomain(constructionDomain.str()) {}

  llvm::StringRef getName() const override { return name; }
  llvm::StringRef getConstructionDomain() const override {
    return constructionDomain;
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return enabled; }

  void collectFormulaDescriptors(
      llvm::SmallVectorImpl<weft::plugin::FormulaDescriptor> &out)
      const override {
    weft::plugin::FormulaDescriptor descriptor(
        getFormulaID(), getName(), "test/mock-proposal",
        weft::plugin::FormulaResultKind::CandidateSet,
        weft::plugin::FormulaConstructionStrength::ConstructedWeak);
    descriptor.getGeometryAxis().set(
        weft::plugin::FormulaAxisUse::Decisive, "MockProposalGeometry");
    descriptor.getGeometryAxis().addConsumedField("canonical-problem");
    descriptor.getCapabilityAxis().set(
        weft::plugin::FormulaAxisUse::Decisive, "TargetCapabilitySet");
    descriptor.getCapabilityAxis().addConsumedField(supportCapabilityID);
    descriptor.getStaticContextAxis().set(
        weft::plugin::FormulaAxisUse::HonestNull,
        "MockProposalNoStaticContext");
    descriptor.addSemanticCase("supported");
    descriptor.addSemanticCase("unsupported");
    descriptor.addProductionEntry("plugin:variant-proposal");
    out.push_back(std::move(descriptor));
  }

  bool supportsOperation(const VariantProposalRequest &request) const override {
    ++supportCalls;

    if (request.getProblem())
      observedHighLevelOpName =
          request.getProblem()->getName().getStringRef().str();
    if (request.getKernel())
      observedKernelName = request.getKernel().getSymName().str();
    observedCapabilityCount = request.getCapabilities().size();

    return request.getProblem() && request.getKernel() &&
           request.getCapabilities().isCapabilityAvailableByID(
               supportCapabilityID);
  }

  llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const override {
    (void)request;
    ++proposalCalls;

    VariantProposal proposal(proposalName, originPlugin);
    proposal.setFormulaID(getFormulaID());
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
  std::string getFormulaID() const { return name + ".proposal.construct"; }
  std::string name;
  std::string supportCapabilityID;
  std::string proposalName;
  std::string originPlugin;
  std::string requiredCapabilityID;
  std::string requiredCapabilitySymbol;
  bool enabled;
  bool includeRequiredCapabilitySymbol;
  std::string constructionDomain;
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
  llvm::StringRef getConstructionDomain() const override {
    return "test-domain";
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool supportsOperation(const VariantProposalRequest &request) const override {
    return request.getProblem() && request.getKernel() &&
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

/// Test-only typed construction owner used to prove that target binding, not a
/// family/origin-name branch in common code, selects an owner domain.  The
/// constructed func.func is only an artifact-neutral typed carrier for this
/// lifecycle test; it deliberately registers no backend or artifact route.
class DomainConstructionOwner final : public ExtensionPlugin {
public:
  DomainConstructionOwner(llvm::StringRef name, llvm::StringRef domain,
                          llvm::StringRef capabilityID,
                          llvm::StringRef capabilitySymbol)
      : name(name.str()), domain(domain.str()),
        capabilityID(capabilityID.str()),
        capabilitySymbol(capabilitySymbol.str()),
        variantName((name + "_candidate").str()) {}

  llvm::StringRef getName() const override { return name; }
  llvm::StringRef getConstructionDomain() const override { return domain; }
  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }
  void registerDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void collectFormulaDescriptors(
      llvm::SmallVectorImpl<weft::plugin::FormulaDescriptor> &out)
      const override {
    weft::plugin::FormulaDescriptor descriptor(
        getFormulaID(), getName(), "test/domain-bound-construction",
        weft::plugin::FormulaResultKind::CandidateSet,
        weft::plugin::FormulaConstructionStrength::ConstructedWeak);
    descriptor.getGeometryAxis().set(
        weft::plugin::FormulaAxisUse::Decisive, "ExactInt8MACProblem");
    descriptor.getGeometryAxis().addConsumedField("physical-problem");
    descriptor.getCapabilityAxis().set(
        weft::plugin::FormulaAxisUse::Decisive, "BoundTargetCapability");
    descriptor.getCapabilityAxis().addConsumedField(capabilityID);
    descriptor.getStaticContextAxis().set(
        weft::plugin::FormulaAxisUse::HonestNull,
        "DomainWitnessNoStaticContext");
    descriptor.addSemanticCase("domain-qualified-candidate");
    descriptor.addProductionEntry("plugin:variant-proposal");
    descriptor.addProductionEntry("plugin:analytic-cost");
    out.push_back(std::move(descriptor));
  }

  bool supportsOperation(const VariantProposalRequest &request) const override {
    ++supportCalls;
    supportProblem = request.getProblem();
    return llvm::isa_and_nonnull<weft::exec::Int8MACProblemOp>(
               request.getProblem()) &&
           request.getCapabilities().isCapabilityAvailableByID(capabilityID);
  }

  llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const override {
    ++proposalCalls;
    proposalProblem = request.getProblem();
    VariantProposal proposal(variantName, name);
    proposal.setFormulaID(getFormulaID());
    proposal.addRequiredCapabilityID(capabilityID);
    proposal.addRequiredCapabilitySymbol(capabilitySymbol);
    out.push_back(std::move(proposal));
    return llvm::Error::success();
  }

  llvm::Error
  verifyVariantLegality(const VariantLegalityRequest &request) const override {
    ++legalityCalls;
    legalityProblem = request.getProblem();
    if (!llvm::isa_and_nonnull<weft::exec::Int8MACProblemOp>(
            request.getProblem()) ||
        !request.getCapabilities().isCapabilityAvailableByID(capabilityID))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "domain witness lost exact P or C_d");
    return llvm::Error::success();
  }

  llvm::Error estimateVariantCost(const VariantCostRequest &request,
                                  VariantCostEstimate &out) const override {
    out = VariantCostEstimate();
    out.setScore(1.0);
    out.setExplicitPreference(true);
    out.setOriginPlugin(name);
    out.setFormulaID(getFormulaID());
    out.setVariantSymbol(request.getVariant().getSymName());
    out.setExplanation("single target-domain-qualified test owner");
    out.setPolicy("analytic test prior inside the legal candidate set");
    return llvm::Error::success();
  }

  llvm::Error constructFormulaPlans(
      const FamilyConstructionRequest &request,
      FamilyConstructionResult &out) const override {
    ++constructionCalls;
    constructionProblem = request.getProblem();
    if (!llvm::isa_and_nonnull<weft::exec::Int8MACProblemOp>(
            request.getProblem()) ||
        !request.getCapabilities().isCapabilityAvailableByID(capabilityID))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "domain construction lost exact P or C_d");

    mlir::OpBuilder builder(request.getModule().getContext());
    builder.setInsertionPointToEnd(&request.getVariant().getBody().front());
    auto function = builder.create<mlir::func::FuncOp>(
        request.getVariant().getLoc(), name + "_typed_body",
        builder.getFunctionType({}, {}));
    mlir::Block *entry = function.addEntryBlock();
    mlir::OpBuilder::atBlockEnd(entry).create<mlir::func::ReturnOp>(
        function.getLoc());
    out = FamilyConstructionResult::getFinalBody(function.getOperation());
    return llvm::Error::success();
  }

  llvm::StringRef getVariantName() const { return variantName; }
  unsigned getSupportCalls() const { return supportCalls; }
  unsigned getProposalCalls() const { return proposalCalls; }
  unsigned getLegalityCalls() const { return legalityCalls; }
  unsigned getConstructionCalls() const { return constructionCalls; }
  bool observedSameProblem(mlir::Operation *problem) const {
    return supportProblem == problem && proposalProblem == problem &&
           legalityProblem == problem && constructionProblem == problem;
  }

private:
  std::string getFormulaID() const { return name + ".domain.construct"; }
  std::string name;
  std::string domain;
  std::string capabilityID;
  std::string capabilitySymbol;
  std::string variantName;
  llvm::SmallVector<PluginCapability, 1> capabilities;
  mutable unsigned supportCalls = 0;
  mutable unsigned proposalCalls = 0;
  mutable unsigned legalityCalls = 0;
  mutable unsigned constructionCalls = 0;
  mutable mlir::Operation *supportProblem = nullptr;
  mutable mlir::Operation *proposalProblem = nullptr;
  mutable mlir::Operation *legalityProblem = nullptr;
  mutable mlir::Operation *constructionProblem = nullptr;
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

int runSecondTargetDomainOwnerWitness(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.capability @domain_a_capability {id = "domain.a.capability", kind = "test-owner"}
  weft.exec.capability @domain_b_capability {id = "domain.b.capability", kind = "test-owner"}
  weft.exec.target @domain_a_profile {id = "domain.a.profile", target_kind = "profile", construction_domain = "domain-a", capability_providers = [@domain_a_capability]}
  weft.exec.target @domain_b_profile {id = "domain.b.profile", target_kind = "profile", construction_domain = "domain-b", capability_providers = [@domain_b_capability]}
  weft.exec.kernel @domain_b_source attributes {target = @domain_b_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    return fail("failed to parse second target-domain owner witness");
  KernelOp kernel = findKernel(*module);
  llvm::Expected<mlir::Operation *> problem =
      weft::plugin::resolveCanonicalProblem(kernel);
  if (!problem)
    return fail(llvm::toString(problem.takeError()));

  DomainConstructionOwner ownerA("domain-a-owner", "domain-a",
                                 "domain.a.capability",
                                 "domain_a_capability");
  DomainConstructionOwner ownerB("domain-b-owner", "domain-b",
                                 "domain.b.capability",
                                 "domain_b_capability");
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(ownerA),
                                 "register domain A owner"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(ownerB),
                                 "register domain B owner"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  VariantProposalRequest request(*problem, kernel, capabilities);
  mlir::OpBuilder builder(&context);
  llvm::SmallVector<VariantOp, 1> variants;
  if (int result = expectSuccess(
          weft::transforms::collectAndMaterializeVariantProposals(
              builder, registry, request, &variants),
          "materialize domain B proposal"))
    return result;
  if (int result = expect(
          variants.size() == 1 &&
              variants.front().getSymName() == ownerB.getVariantName(),
          "target domain B admits only its matching owner candidate"))
    return result;
  if (int result = expect(ownerA.getSupportCalls() == 0 &&
                              ownerA.getProposalCalls() == 0,
                          "foreign domain A exits before owner hooks"))
    return result;

  if (int result = expectSuccess(registry.verifyKernelVariantLegality(kernel),
                                 "verify domain B candidate legality"))
    return result;
  llvm::Expected<weft::transforms::VariantSelectionPlan> selection =
      weft::transforms::planKernelVariantSelection(kernel, registry);
  if (!selection)
    return fail("select domain B candidate: " +
                llvm::toString(selection.takeError()));
  if (int result = expect(
          selection->selectedVariant == variants.front(),
          "thin selector preserves the sole legal domain B candidate"))
    return result;

  FamilyConstructionResult construction;
  if (int result = expectSuccess(
          registry.constructFormulaPlansForVariant(
              *module, selection->selectedVariant, construction),
          "construct domain B artifact-neutral typed body"))
    return result;
  mlir::Operation *root = construction.getOperation();
  if (int result = expect(
          construction.hasFinalBody() && root &&
              llvm::isa<mlir::func::FuncOp>(root) &&
              root->getParentOfType<VariantOp>() == variants.front(),
          "construction returns the exact typed root owned by the selected "
          "variant"))
    return result;
  if (int result = expect(
          ownerB.getSupportCalls() == 1 && ownerB.getProposalCalls() == 1 &&
              ownerB.getLegalityCalls() >= 1 &&
              ownerB.getConstructionCalls() == 1 &&
              ownerB.observedSameProblem(*problem),
          "proposal, legality, and construction consume one physical P under "
          "the target-bound C_d"))
    return result;
  if (int result = expect(ownerA.getLegalityCalls() == 0 &&
                              ownerA.getConstructionCalls() == 0,
                          "foreign domain A never enters legality or "
                          "construction"))
    return result;
  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  weft::registerAllDialects(dialectRegistry);
  dialectRegistry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.capability @generic_vector {
    id = "generic.vector",
    kind = "generic-execution"
  }
  weft.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  weft.exec.capability @generic_disabled {
    id = "generic.disabled",
    kind = "runtime",
    status = "disabled"
  }
  weft.exec.target @proposal_profile {
    id = "proposal.profile",
    target_kind = "profile",
    construction_domain = "test-domain",
    capability_providers = [@generic_vector, @generic_toolchain, @generic_disabled]
  }
  weft.exec.kernel @proposal_source attributes {target = @proposal_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    return fail("failed to parse plugin proposal test module");

  KernelOp kernel = findKernel(*module);
  if (int result = expect(static_cast<bool>(kernel), "kernel is present"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  if (int result = expect(
          capabilities.size() == 4,
          "target profile and its composed capabilities are collected"))
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
  ProposalPlugin foreign("foreign", "generic.vector", "foreign_path",
                         "foreign", "generic.vector", "generic_vector", true,
                         true, "foreign-domain");

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
  if (int result = expectSuccess(registry.registerPlugin(foreign),
                                 "register foreign-domain plugin"))
    return result;

  llvm::Expected<mlir::Operation *> problem =
      weft::plugin::resolveCanonicalProblem(kernel);
  if (!problem)
    return fail(llvm::toString(problem.takeError()));
  VariantProposalRequest request(*problem, kernel, capabilities);
  {
    VariantProposalRequest forged(module->getOperation(), kernel,
                                  capabilities);
    llvm::SmallVector<VariantProposal, 1> forgedProposals;
    if (int result = expectErrorContains(
            registry.collectVariantProposals(forged, forgedProposals),
            {"must carry the exact canonical problem bound by its kernel"}))
      return result;
  }
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
  if (int result = expect(foreign.getSupportCalls() == 0 &&
                              foreign.getProposalCalls() == 0,
                          "foreign-domain plugin is skipped before support "
                          "or proposal hooks"))
    return result;
  if (int result = expect(first.getObservedHighLevelOpName() ==
                                  "weft.exec.int8_mac_problem" &&
                              first.getObservedKernelName() ==
                                  "proposal_source" &&
                              first.getObservedCapabilityCount() == 4,
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
          weft::transforms::materializeVariantProposals(
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
          weft::transforms::collectAndMaterializeVariantProposals(
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

  if (int result = runSecondTargetDomainOwnerWitness(context))
    return result;

  llvm::outs() << "plugin variant proposal smoke test passed\n";
  return 0;
}
