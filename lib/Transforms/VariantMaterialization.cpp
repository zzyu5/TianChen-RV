#include "TianChenRV/Transforms/VariantMaterialization.h"

#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <string>
#include <utility>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZEPLUGINVARIANTS
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::VariantProposal;
using tianchenrv::plugin::VariantProposalDecline;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::CapabilityOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

struct PlannedVariant {
  std::string symbolName;
  std::string originPlugin;
  mlir::ArrayAttr requires;
  std::string condition;
  std::string guard;
  std::string policy;
  std::string fallbackRole;
  llvm::SmallVector<mlir::NamedAttribute, 4> pluginAttributes;
};

llvm::Error makeMaterializationError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      message, llvm::errc::invalid_argument);
}

llvm::Error makeProposalError(const VariantProposal &proposal,
                              llvm::Twine message) {
  return makeMaterializationError(
      llvm::Twine("TianChen-RV variant materialization failed for proposal '") +
      proposal.getVariantName() + "' from origin plugin '" +
      proposal.getOriginPlugin() + "': " + message);
}

bool isValidBareSymbolName(llvm::StringRef value) {
  if (value.empty() || value.trim() != value)
    return false;

  auto isFirst = [](char character) {
    unsigned char value = static_cast<unsigned char>(character);
    return std::isalpha(value) || character == '_' || character == '$';
  };
  auto isRest = [](char character) {
    unsigned char value = static_cast<unsigned char>(character);
    return std::isalnum(value) || character == '_' || character == '$' ||
           character == '.' || character == '-';
  };

  if (!isFirst(value.front()))
    return false;

  return llvm::all_of(value.drop_front(), isRest);
}

bool isCoreVariantAttributeName(llvm::StringRef name) {
  return name == "sym_name" || name == "origin" || name == "requires" ||
         name == "condition" || name == "guard" || name == "policy" ||
         name == plugin::kVariantFallbackRoleAttrName;
}

bool isValidPluginAttributeNameSegment(llvm::StringRef segment) {
  if (segment.empty())
    return false;

  auto isFirst = [](char character) {
    unsigned char value = static_cast<unsigned char>(character);
    return std::isalpha(value) || character == '_';
  };
  auto isRest = [](char character) {
    unsigned char value = static_cast<unsigned char>(character);
    return std::isalnum(value) || character == '_' || character == '$';
  };

  if (!isFirst(segment.front()))
    return false;

  for (char character : segment.drop_front()) {
    if (!isRest(character))
      return false;
  }
  return true;
}

bool isValidPluginAttributeName(llvm::StringRef name) {
  if (name.empty() || name.trim() != name || !name.contains('.'))
    return false;

  llvm::SmallVector<llvm::StringRef, 4> segments;
  name.split(segments, '.');
  if (segments.size() < 2)
    return false;

  return llvm::all_of(segments, isValidPluginAttributeNameSegment);
}

bool kernelHasBody(KernelOp kernel) {
  return kernel && !kernel.getBody().empty();
}

bool kernelHasDirectCapability(KernelOp kernel) {
  if (!kernelHasBody(kernel))
    return false;

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (llvm::isa<CapabilityOp>(op))
      return true;
  }
  return false;
}

std::string getNonEmptyDecisionMetadata(llvm::StringRef value) {
  if (value.trim().empty())
    return {};
  return value.str();
}

std::string formatRecoverableDeclines(
    llvm::ArrayRef<VariantProposalDecline> recoverableDeclines) {
  if (recoverableDeclines.empty())
    return "no enabled extension plugin produced a viable proposal";

  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "recoverable plugin declines in registration order: ";
  llvm::interleave(
      recoverableDeclines,
      [&](const VariantProposalDecline &decline) {
        stream << decline.getPluginName() << ": " << decline.getReason();
      },
      [&] { stream << "; "; });
  return stream.str();
}

llvm::Error makeNoViableProposalError(
    KernelOp kernel, llvm::ArrayRef<VariantProposalDecline> recoverableDeclines) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  stream << "TianChen-RV plugin variant materialization";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  stream << " collected no viable plugin proposals; "
         << formatRecoverableDeclines(recoverableDeclines);
  return makeMaterializationError(stream.str());
}

CapabilityOp lookupCapabilityInKernel(KernelOp kernel,
                                      llvm::StringRef symbolName) {
  return llvm::dyn_cast_or_null<CapabilityOp>(
      mlir::SymbolTable::lookupSymbolIn(
          kernel.getOperation(),
          mlir::StringAttr::get(kernel.getOperation()->getContext(),
                                symbolName)));
}

llvm::Error appendRequiredSymbol(mlir::OpBuilder &builder, KernelOp kernel,
                                 const VariantProposal &proposal,
                                 llvm::StringRef requiredSymbol,
                                 llvm::StringSet<> &seenRequiredSymbols,
                                 llvm::SmallVectorImpl<mlir::Attribute> &out) {
  if (requiredSymbol.trim().empty())
    return makeProposalError(
        proposal, "required capability symbol reference must be non-empty");

  if (!isValidBareSymbolName(requiredSymbol))
    return makeProposalError(
        proposal,
        llvm::Twine("required capability symbol reference @") +
            requiredSymbol +
            " cannot be represented by the current MLIR symbol subset");

  CapabilityOp capabilityOp = lookupCapabilityInKernel(kernel, requiredSymbol);
  if (!capabilityOp)
    return makeProposalError(
        proposal, llvm::Twine("unresolved required capability symbol @") +
                      requiredSymbol + " in kernel @" + kernel.getSymName());

  if (seenRequiredSymbols.insert(requiredSymbol).second)
    out.push_back(mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                               requiredSymbol));

  return llvm::Error::success();
}

llvm::Error appendRequiredID(mlir::OpBuilder &builder, KernelOp kernel,
                             const VariantProposalRequest &request,
                             const VariantProposal &proposal,
                             llvm::StringRef requiredID,
                             llvm::StringSet<> &seenRequiredSymbols,
                             llvm::SmallVectorImpl<mlir::Attribute> &out) {
  if (requiredID.trim().empty())
    return makeProposalError(proposal,
                             "required capability id must be non-empty");

  const CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(requiredID);
  if (!capability)
    return makeProposalError(
        proposal, llvm::Twine("unresolved required capability id \"") +
                      requiredID + "\" in kernel @" + kernel.getSymName());
  if (!capability->isAvailable())
    return makeProposalError(
        proposal, llvm::Twine("required capability id \"") + requiredID +
                      "\" resolves to unavailable capability symbol @" +
                      capability->getSymbolName() + " in kernel @" +
                      kernel.getSymName());

  llvm::StringRef requiredSymbol = capability->getSymbolName();
  if (requiredSymbol.trim().empty())
    return makeProposalError(
        proposal, llvm::Twine("required capability id \"") + requiredID +
                      "\" resolved to an empty capability symbol");

  CapabilityOp capabilityOp = lookupCapabilityInKernel(kernel, requiredSymbol);
  if (!capabilityOp)
    return makeProposalError(
        proposal, llvm::Twine("required capability id \"") + requiredID +
                      "\" resolved to missing capability symbol @" +
                      requiredSymbol + " in kernel @" + kernel.getSymName());

  if (seenRequiredSymbols.insert(requiredSymbol).second)
    out.push_back(mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                               requiredSymbol));

  return llvm::Error::success();
}

llvm::Error
validateAndCopyPluginAttributes(const VariantProposal &proposal,
                                llvm::SmallVectorImpl<mlir::NamedAttribute>
                                    &pluginAttributes) {
  llvm::StringSet<> seenAttributeNames;
  for (mlir::NamedAttribute attribute : proposal.getPluginAttributes()) {
    if (!attribute.getName())
      return makeProposalError(
          proposal, "plugin-owned attribute name must be non-empty");

    llvm::StringRef name = attribute.getName().getValue();
    if (name.empty())
      return makeProposalError(
          proposal, "plugin-owned attribute name must be non-empty");

    if (!attribute.getValue())
      return makeProposalError(
          proposal, llvm::Twine("plugin-owned attribute '") + name +
                        "' must have a non-null MLIR attribute value");

    if (isCoreVariantAttributeName(name))
      return makeProposalError(
          proposal, llvm::Twine("plugin-owned attribute '") + name +
                        "' collides with required tcrv.exec.variant "
                        "attribute");

    if (!isValidPluginAttributeName(name))
      return makeProposalError(
          proposal, llvm::Twine("plugin-owned attribute '") + name +
                        "' must be a dialect-qualified discardable "
                        "attribute name");

    if (!seenAttributeNames.insert(name).second)
      return makeProposalError(
          proposal,
          llvm::Twine("duplicate plugin-owned attribute '") + name + "'");

    pluginAttributes.push_back(attribute);
  }

  return llvm::Error::success();
}

llvm::Error validateAndPlanMaterialization(
    mlir::OpBuilder &builder, const VariantProposalRequest &request,
    llvm::ArrayRef<VariantProposal> proposals,
    llvm::SmallVectorImpl<PlannedVariant> &plannedVariants) {
  KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeMaterializationError(
        "TianChen-RV variant materialization requires a tcrv.exec.kernel "
        "anchor");

  if (!kernelHasBody(kernel))
    return makeMaterializationError(
        llvm::Twine("TianChen-RV variant materialization requires kernel @") +
        kernel.getSymName() + " to have a materializable body block");

  llvm::StringSet<> seenVariantSymbols;
  for (const VariantProposal &proposal : proposals) {
    llvm::StringRef variantName = proposal.getVariantName();
    if (variantName.trim().empty())
      return makeProposalError(proposal, "variant name must be non-empty");

    if (!isValidBareSymbolName(variantName))
      return makeProposalError(
          proposal, llvm::Twine("variant name \"") + variantName +
                        "\" cannot be represented by the current MLIR symbol "
                        "subset");

    if (proposal.getOriginPlugin().trim().empty())
      return makeProposalError(proposal, "origin plugin must be non-empty");

    if (!seenVariantSymbols.insert(variantName).second)
      return makeProposalError(
          proposal, llvm::Twine("duplicate variant symbol @") + variantName +
                        " in materialization input for kernel @" +
                        kernel.getSymName());

    if (mlir::Operation *existingSymbol =
            mlir::SymbolTable::lookupSymbolIn(kernel.getOperation(),
                                              variantName)) {
      return makeProposalError(
          proposal, llvm::Twine("duplicate variant symbol @") + variantName +
                        " in kernel @" + kernel.getSymName() +
                        "; existing symbol op is " +
                        existingSymbol->getName().getStringRef());
    }

    llvm::StringSet<> seenRequiredSymbols;
    llvm::SmallVector<mlir::Attribute, 4> requiredCapabilityRefs;
    for (const std::string &requiredIDStorage :
         proposal.getRequiredCapabilityIDs()) {
      if (llvm::Error error = appendRequiredID(
              builder, kernel, request, proposal, requiredIDStorage,
              seenRequiredSymbols, requiredCapabilityRefs))
        return error;
    }

    for (const std::string &requiredSymbolStorage :
         proposal.getRequiredCapabilitySymbols()) {
      llvm::StringRef requiredSymbol(requiredSymbolStorage);
      const CapabilityDescriptor *capability =
          request.getCapabilities().lookupBySymbolName(requiredSymbol);
      if (!capability) {
        if (requiredSymbol.trim().empty())
          return makeProposalError(
              proposal,
              "required capability symbol reference must be non-empty");
        return makeProposalError(
            proposal, llvm::Twine("unresolved required capability symbol @") +
                          requiredSymbol + " in kernel @" +
                          kernel.getSymName());
      }

      if (llvm::Error error =
              appendRequiredSymbol(builder, kernel, proposal, requiredSymbol,
                                   seenRequiredSymbols,
                                   requiredCapabilityRefs))
        return error;
    }

    llvm::SmallVector<mlir::NamedAttribute, 4> pluginAttributes;
    if (llvm::Error error =
            validateAndCopyPluginAttributes(proposal, pluginAttributes))
      return error;

    plannedVariants.push_back(PlannedVariant{
        variantName.str(), proposal.getOriginPlugin().str(),
        builder.getArrayAttr(requiredCapabilityRefs),
        getNonEmptyDecisionMetadata(proposal.getCondition()),
        getNonEmptyDecisionMetadata(proposal.getGuard()),
        getNonEmptyDecisionMetadata(proposal.getPolicy()),
        proposal.hasFallbackRole()
            ? plugin::stringifyVariantFallbackRole(proposal.getFallbackRole())
                  .str()
            : std::string(),
        std::move(pluginAttributes)});
  }

  return llvm::Error::success();
}

llvm::Error buildExpectedRequiredRefs(
    mlir::OpBuilder &builder, const VariantProposalRequest &request,
    const VariantProposal &proposal,
    llvm::SmallVectorImpl<mlir::Attribute> &requiredCapabilityRefs) {
  KernelOp kernel = request.getKernel();
  llvm::StringSet<> seenRequiredSymbols;

  for (const std::string &requiredIDStorage :
       proposal.getRequiredCapabilityIDs()) {
    if (llvm::Error error = appendRequiredID(
            builder, kernel, request, proposal, requiredIDStorage,
            seenRequiredSymbols, requiredCapabilityRefs))
      return error;
  }

  for (const std::string &requiredSymbolStorage :
       proposal.getRequiredCapabilitySymbols()) {
    llvm::StringRef requiredSymbol(requiredSymbolStorage);
    const CapabilityDescriptor *capability =
        request.getCapabilities().lookupBySymbolName(requiredSymbol);
    if (!capability) {
      if (requiredSymbol.trim().empty())
        return makeProposalError(
            proposal, "required capability symbol reference must be non-empty");
      return makeProposalError(
          proposal, llvm::Twine("unresolved required capability symbol @") +
                        requiredSymbol + " in kernel @" + kernel.getSymName());
    }

    if (llvm::Error error =
            appendRequiredSymbol(builder, kernel, proposal, requiredSymbol,
                                 seenRequiredSymbols, requiredCapabilityRefs))
      return error;
  }

  return llvm::Error::success();
}

bool arrayAttrsEqual(mlir::ArrayAttr lhs, mlir::ArrayAttr rhs) {
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return false;

  for (auto [lhsAttr, rhsAttr] : llvm::zip(lhs, rhs)) {
    if (lhsAttr != rhsAttr)
      return false;
  }
  return true;
}

llvm::Error explainExistingVariantMismatch(VariantOp existing,
                                           const VariantProposal &proposal,
                                           llvm::Twine reason) {
  return makeProposalError(
      proposal, llvm::Twine("existing direct variant @") +
                    existing.getSymName() +
                    " does not exactly match the current plugin proposal: " +
                    reason);
}

llvm::Error existingVariantMatchesProposal(
    mlir::OpBuilder &builder, const VariantProposalRequest &request,
    const VariantProposal &proposal, VariantOp existing) {
  if (!existing)
    return makeProposalError(
        proposal, "existing direct symbol for proposed variant is not a "
                  "tcrv.exec.variant");

  if (existing.getBody().empty())
    return explainExistingVariantMismatch(
        existing, proposal, "variant region is missing its materialized block");

  if (!existing.getBody().front().empty())
    return explainExistingVariantMismatch(
        existing, proposal,
        "variant body is not the empty body produced by proposal "
        "materialization");

  auto originAttr =
      existing->getAttrOfType<mlir::StringAttr>("origin");
  if (!originAttr || originAttr.getValue() != proposal.getOriginPlugin())
    return explainExistingVariantMismatch(existing, proposal,
                                          "origin attribute differs");

  llvm::SmallVector<mlir::Attribute, 4> requiredCapabilityRefs;
  if (llvm::Error error = buildExpectedRequiredRefs(
          builder, request, proposal, requiredCapabilityRefs))
    return error;

  mlir::ArrayAttr expectedRequires = builder.getArrayAttr(requiredCapabilityRefs);
  mlir::ArrayAttr actualRequires =
      existing->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!arrayAttrsEqual(actualRequires, expectedRequires))
    return explainExistingVariantMismatch(existing, proposal,
                                          "requires attribute differs");

  auto requireStringMetadata = [&](llvm::StringRef attrName,
                                   llvm::StringRef expected) -> llvm::Error {
    auto attr = existing->getAttrOfType<mlir::StringAttr>(attrName);
    if (expected.empty()) {
      if (attr)
        return explainExistingVariantMismatch(
            existing, proposal,
            llvm::Twine(attrName) + " attribute is present but proposal omits it");
      return llvm::Error::success();
    }

    if (!attr || attr.getValue() != expected)
      return explainExistingVariantMismatch(
          existing, proposal, llvm::Twine(attrName) + " attribute differs");
    return llvm::Error::success();
  };

  if (llvm::Error error =
          requireStringMetadata("condition",
                                getNonEmptyDecisionMetadata(
                                    proposal.getCondition())))
    return error;
  if (llvm::Error error =
          requireStringMetadata("guard",
                                getNonEmptyDecisionMetadata(proposal.getGuard())))
    return error;
  if (llvm::Error error =
          requireStringMetadata("policy",
                                getNonEmptyDecisionMetadata(
                                    proposal.getPolicy())))
    return error;

  std::string expectedFallbackRole =
      proposal.hasFallbackRole()
          ? plugin::stringifyVariantFallbackRole(proposal.getFallbackRole()).str()
          : std::string();
  if (llvm::Error error = requireStringMetadata(
          plugin::kVariantFallbackRoleAttrName, expectedFallbackRole))
    return error;

  llvm::SmallVector<mlir::NamedAttribute, 4> expectedPluginAttributes;
  if (llvm::Error error =
          validateAndCopyPluginAttributes(proposal, expectedPluginAttributes))
    return error;

  llvm::StringSet<> expectedPluginAttributeNames;
  for (mlir::NamedAttribute attribute : expectedPluginAttributes)
    expectedPluginAttributeNames.insert(attribute.getName().getValue());

  for (mlir::NamedAttribute expectedAttribute : expectedPluginAttributes) {
    mlir::Attribute actual =
        existing->getAttr(expectedAttribute.getName().getValue());
    if (actual != expectedAttribute.getValue())
      return explainExistingVariantMismatch(
          existing, proposal,
          llvm::Twine("plugin-owned attribute '") +
              expectedAttribute.getName().getValue() + "' differs");
  }

  for (mlir::NamedAttribute actualAttribute : existing->getAttrs()) {
    llvm::StringRef name = actualAttribute.getName().getValue();
    if (isCoreVariantAttributeName(name) ||
        expectedPluginAttributeNames.contains(name))
      continue;
    return explainExistingVariantMismatch(
        existing, proposal,
        llvm::Twine("unexpected extra attribute '") + name + "'");
  }

  return llvm::Error::success();
}

llvm::Error filterAlreadyMaterializedProposals(
    mlir::OpBuilder &builder, const VariantProposalRequest &request,
    llvm::ArrayRef<VariantProposal> proposals,
    llvm::SmallVectorImpl<VariantProposal> &proposalsToMaterialize) {
  KernelOp kernel = request.getKernel();
  for (const VariantProposal &proposal : proposals) {
    mlir::Operation *existingSymbol = mlir::SymbolTable::lookupSymbolIn(
        kernel.getOperation(), proposal.getVariantName());
    if (!existingSymbol) {
      proposalsToMaterialize.push_back(proposal);
      continue;
    }

    auto existingVariant = llvm::dyn_cast<VariantOp>(existingSymbol);
    if (!existingVariant)
      return makeProposalError(
          proposal, llvm::Twine("duplicate proposed variant symbol @") +
                        proposal.getVariantName() +
                        " resolves to non-variant op " +
                        existingSymbol->getName().getStringRef() +
                        " in kernel @" + kernel.getSymName());

    if (llvm::Error error = existingVariantMatchesProposal(
            builder, request, proposal, existingVariant))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error materializeKernelPluginVariants(
    mlir::OpBuilder &builder, KernelOp kernel,
    const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<VariantOp> *materializedVariants = nullptr) {
  if (!kernel)
    return makeMaterializationError(
        "TianChen-RV plugin variant materialization requires a "
        "tcrv.exec.kernel");

  if (!kernelHasBody(kernel))
    return makeMaterializationError(
        llvm::Twine("TianChen-RV plugin variant materialization requires "
                    "kernel @") +
        kernel.getSymName() + " to have a materialized body block");

  if (registry.empty())
    return makeMaterializationError(
        llvm::Twine("TianChen-RV plugin variant materialization for kernel @") +
        kernel.getSymName() +
        " requires at least one enabled extension plugin in the registry");

  if (!kernelHasDirectCapability(kernel))
    return makeMaterializationError(
        llvm::Twine("TianChen-RV plugin variant materialization for kernel @") +
        kernel.getSymName() +
        " requires existing direct tcrv.exec.capability anchors");

  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();
  VariantProposalRequest request(kernel.getOperation(), kernel, *capabilities);

  llvm::SmallVector<VariantProposal, 4> proposals;
  llvm::SmallVector<VariantProposalDecline, 2> recoverableDeclines;
  if (llvm::Error error =
          registry.collectVariantProposals(request, proposals,
                                           &recoverableDeclines))
    return error;

  if (proposals.empty())
    return makeNoViableProposalError(kernel, recoverableDeclines);

  llvm::SmallVector<VariantProposal, 4> proposalsToMaterialize;
  if (llvm::Error error = filterAlreadyMaterializedProposals(
          builder, request, proposals, proposalsToMaterialize))
    return error;

  return materializeVariantProposals(builder, request, proposalsToMaterialize,
                                     materializedVariants);
}

class MaterializePluginVariantsPass final
    : public impl::MaterializePluginVariantsBase<
          MaterializePluginVariantsPass> {
public:
  MaterializePluginVariantsPass() : registry(&ownedRegistry) {}

  explicit MaterializePluginVariantsPass(
      const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  MaterializePluginVariantsPass(const MaterializePluginVariantsPass &other)
      : impl::MaterializePluginVariantsBase<MaterializePluginVariantsPass>(
            other),
        registry(other.registry == &other.ownedRegistry ? &ownedRegistry
                                                        : other.registry) {}

  void runOnOperation() override {
    loadPluginDialects();

    mlir::OpBuilder builder(&getContext());
    llvm::SmallVector<KernelOp, 4> kernels;
    getOperation()->walk([&](KernelOp kernel) { kernels.push_back(kernel); });

    for (KernelOp kernel : kernels) {
      if (mlir::failed(runMaterialization(builder, kernel))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  void loadPluginDialects() {
    if (registry->empty())
      return;

    mlir::DialectRegistry dialects;
    registry->registerDialectsForEnabledPlugins(dialects);
    getContext().appendDialectRegistry(dialects);
    getContext().loadAllAvailableDialects();
  }

  mlir::LogicalResult runMaterialization(mlir::OpBuilder &builder,
                                         KernelOp kernel) {
    if (llvm::Error error =
            materializeKernelPluginVariants(builder, kernel, *registry)) {
      std::string message = llvm::toString(std::move(error));
      if (kernel)
        kernel.emitError() << message;
      else
        getOperation()->emitError() << message;
      return mlir::failure();
    }
    return mlir::success();
  }

  ExtensionPluginRegistry ownedRegistry;
  const ExtensionPluginRegistry *registry = nullptr;
};

} // namespace

llvm::Error materializeVariantProposals(
    mlir::OpBuilder &builder, const VariantProposalRequest &request,
    llvm::ArrayRef<plugin::VariantProposal> proposals,
    llvm::SmallVectorImpl<VariantOp> *materializedVariants) {
  llvm::SmallVector<PlannedVariant, 4> plannedVariants;
  if (llvm::Error error = validateAndPlanMaterialization(
          builder, request, proposals, plannedVariants))
    return error;

  KernelOp kernel = request.getKernel();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  for (const PlannedVariant &plannedVariant : plannedVariants) {
    mlir::StringAttr conditionAttr;
    mlir::StringAttr guardAttr;
    mlir::StringAttr policyAttr;
    if (!plannedVariant.condition.empty())
      conditionAttr = builder.getStringAttr(plannedVariant.condition);
    if (!plannedVariant.guard.empty())
      guardAttr = builder.getStringAttr(plannedVariant.guard);
    if (!plannedVariant.policy.empty())
      policyAttr = builder.getStringAttr(plannedVariant.policy);

    VariantOp variant = builder.create<VariantOp>(
        kernel.getLoc(), plannedVariant.symbolName,
        builder.getStringAttr(plannedVariant.originPlugin),
        plannedVariant.requires, conditionAttr, guardAttr, policyAttr);
    for (mlir::NamedAttribute attribute : plannedVariant.pluginAttributes)
      variant->setAttr(attribute.getName(), attribute.getValue());
    if (!plannedVariant.fallbackRole.empty())
      variant->setAttr(plugin::kVariantFallbackRoleAttrName,
                       builder.getStringAttr(plannedVariant.fallbackRole));
    if (variant.getBody().empty())
      variant.getBody().emplaceBlock();
    if (materializedVariants)
      materializedVariants->push_back(variant);
  }

  return llvm::Error::success();
}

llvm::Error collectAndMaterializeVariantProposals(
    mlir::OpBuilder &builder, const plugin::ExtensionPluginRegistry &registry,
    const plugin::VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantOp> *materializedVariants) {
  llvm::SmallVector<plugin::VariantProposal, 4> proposals;
  llvm::SmallVector<plugin::VariantProposalDecline, 2> recoverableDeclines;
  if (llvm::Error error =
          registry.collectVariantProposals(request, proposals,
                                           &recoverableDeclines))
    return error;

  if (proposals.empty())
    return makeNoViableProposalError(request.getKernel(), recoverableDeclines);

  return materializeVariantProposals(builder, request, proposals,
                                     materializedVariants);
}

std::unique_ptr<::mlir::Pass> createMaterializePluginVariantsPass() {
  return std::make_unique<MaterializePluginVariantsPass>();
}

std::unique_ptr<::mlir::Pass> createMaterializePluginVariantsPass(
    const plugin::ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializePluginVariantsPass>(registry);
}

} // namespace tianchenrv::transforms
