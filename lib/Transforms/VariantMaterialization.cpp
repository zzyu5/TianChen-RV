#include "TianChenRV/Transforms/VariantMaterialization.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
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
namespace {

using tianchenrv::plugin::VariantProposal;
using tianchenrv::plugin::VariantProposalRequest;
using tianchenrv::support::CapabilityDescriptor;
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
         name == "condition" || name == "guard" || name == "policy";
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

std::string getNonEmptyDecisionMetadata(llvm::StringRef value) {
  if (value.trim().empty())
    return {};
  return value.str();
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
      request.getCapabilities().lookupByID(requiredID);
  if (!capability)
    return makeProposalError(
        proposal, llvm::Twine("unresolved required capability id \"") +
                      requiredID + "\" in kernel @" + kernel.getSymName());

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
        std::move(pluginAttributes)});
  }

  return llvm::Error::success();
}

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
  if (llvm::Error error = registry.collectVariantProposals(request, proposals))
    return error;

  return materializeVariantProposals(builder, request, proposals,
                                     materializedVariants);
}

} // namespace tianchenrv::transforms
