//===- RVVSelectedTargetCapability.h - Unique selected RVV capability ------===//
//
// The single plugin-local projection from a selected variant's `requires`
// symbols and a canonical TargetCapabilitySet to owned RVV target facts.  It is
// shared by route planning and mechanism-local formula decisions so neither an
// emitter nor a selector can introduce first-provider-wins parsing.
//
//===----------------------------------------------------------------------===//

#ifndef WEFT_PLUGIN_RVV_RVVSELECTEDTARGETCAPABILITY_H
#define WEFT_PLUGIN_RVV_RVVSELECTEDTARGETCAPABILITY_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace weft::plugin::rvv {

struct RVVSelectedTargetCapabilityFacts {
  std::string selectedProviderSymbol;
  std::string selectedProviderID;
  std::string selectedProviderKind;
  std::string rvvSatisfactionKind;
  std::string supportedSEW;
  std::string supportedLMUL;
  std::string rvvVersion;
  std::optional<std::int64_t> minimumVLEN;
  std::optional<std::int64_t> vectorRegisterCount;
  std::string requiredTailPolicy;
  std::string requiredMaskPolicy;
  std::string providerMirror;
  std::string legalityMirror;

  bool hasFacts() const { return !selectedProviderSymbol.empty(); }
};

inline llvm::Error makeRVVSelectedTargetCapabilityError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV RVV selected target-capability gate failed: ") +
          message,
      llvm::errc::invalid_argument);
}

/// The sole token-list predicate for RVV capability allow-list properties.
inline bool rvvCapabilityPropertyListContains(llvm::StringRef value,
                                              llvm::StringRef expectedToken) {
  llvm::SmallVector<llvm::StringRef, 8> tokens;
  llvm::SplitString(value, tokens, ",;| \t\r\n");
  for (llvm::StringRef token : tokens)
    if (token.trim() == expectedToken)
      return true;
  return false;
}

namespace selected_target_capability_detail {

inline llvm::Expected<std::string> readTypedStringProperty(
    const support::CapabilityDescriptor &capability,
    llvm::StringRef propertyName, llvm::StringRef context) {
  mlir::Attribute raw = capability.getPropertyAttribute(propertyName);
  if (!raw) {
    // A descriptor projected from IR must not carry a stringified property
    // without its typed source.  This catches synthetic/lossy seams instead of
    // silently treating an untyped spelling as canonical c.
    if (capability.getProperties().count(propertyName.str()) != 0)
      return makeRVVSelectedTargetCapabilityError(
          llvm::Twine(context) + " selected RVV capability provider @" +
          capability.getSymbolName() + " property '" + propertyName +
          "' has no typed source attribute");
    return std::string();
  }
  auto value = llvm::dyn_cast<mlir::StringAttr>(raw);
  if (!value)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " property '" + propertyName +
        "' must be a typed string attribute");
  llvm::StringRef trimmed = value.getValue().trim();
  if (trimmed.empty())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " property '" + propertyName +
        "' must be absent rather than an explicitly empty allow-list/fact");
  return trimmed.str();
}

inline llvm::Error validateTokenList(
    const support::CapabilityDescriptor &capability, llvm::StringRef value,
    llvm::StringRef propertyName, llvm::ArrayRef<llvm::StringRef> allowed,
    llvm::StringRef context) {
  if (value.empty())
    return llvm::Error::success();
  llvm::SmallVector<llvm::StringRef, 8> tokens;
  llvm::SplitString(value, tokens, ",;| \t\r\n");
  if (tokens.empty())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " property '" + propertyName +
        "' has no capability tokens");
  for (llvm::StringRef token : tokens) {
    bool recognized = false;
    for (llvm::StringRef candidate : allowed)
      recognized |= token.trim() == candidate;
    if (!recognized)
      return makeRVVSelectedTargetCapabilityError(
          llvm::Twine(context) + " selected RVV capability provider @" +
          capability.getSymbolName() + " property '" + propertyName +
          "' contains unknown token '" + token.trim() + "'");
  }
  return llvm::Error::success();
}

inline llvm::Error validatePolicyProperty(
    const support::CapabilityDescriptor &capability, llvm::StringRef value,
    llvm::StringRef propertyName, llvm::StringRef context) {
  if (value.empty())
    return llvm::Error::success();
  if (value == "agnostic" || value == "undisturbed")
    return llvm::Error::success();
  return makeRVVSelectedTargetCapabilityError(
      llvm::Twine(context) + " selected RVV capability provider @" +
      capability.getSymbolName() + " property '" + propertyName +
      "' has unknown policy token '" + value +
      "'; expected 'agnostic', 'undisturbed', or an absent fact");
}

inline std::string joinProviderSymbols(
    llvm::ArrayRef<const support::CapabilityDescriptor *> providers) {
  std::string symbols;
  llvm::raw_string_ostream stream(symbols);
  for (std::size_t index = 0, count = providers.size(); index < count;
       ++index) {
    if (index != 0)
      stream << ", ";
    stream << "@" << providers[index]->getSymbolName();
  }
  return symbols;
}

inline llvm::StringRef
classifyRVVSatisfaction(const support::CapabilityDescriptor &capability) {
  if (capability.getID() == "rvv")
    return "exact";
  if (capability.providesID("rvv"))
    return "provides";
  if (capability.impliesID("rvv"))
    return "implies";
  return "transitive";
}

inline std::string formatProviderMirror(
    const support::CapabilityDescriptor &capability) {
  std::string mirror;
  llvm::raw_string_ostream stream(mirror);
  stream << "selected_capability_provider_mirror:@"
         << capability.getSymbolName() << ";id=" << capability.getID()
         << ";kind=" << capability.getKind()
         << ";rvv=" << classifyRVVSatisfaction(capability);
  return mirror;
}

inline llvm::Error verifyProfileProperties(
    const support::CapabilityDescriptor &capability, llvm::StringRef context) {
  llvm::StringRef architecture =
      capability.getProperty("architecture").trim();
  if (!architecture.empty() && architecture != "riscv64")
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " architecture fact '" + architecture +
        "' is incompatible with RVV selected routes; expected 'riscv64'");

  llvm::StringRef isaVectorHints =
      capability.getProperty("isa_vector_hints").trim();
  if (!isaVectorHints.empty() && !hasRVVVectorHint(isaVectorHints))
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " isa_vector_hints fact '" +
        isaVectorHints + "' does not contain RVV vector ISA evidence");
  return llvm::Error::success();
}

} // namespace selected_target_capability_detail

/// Collect one unique, available, non-conflicting RVV provider selected by the
/// variant.  Duplicate capability ids/symbols are rejected by the caller's
/// buildFromKernelChecked; ambiguity inside `requires` is rejected here.  The
/// minimum-VLEN property is optional for generic routes, but when present it
/// must parse as a positive typed integer.  Mechanisms that require it reject a
/// missing value after this shared projection.
inline llvm::Expected<RVVSelectedTargetCapabilityFacts>
collectRVVSelectedTargetCapabilityFacts(
    weft::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    llvm::StringRef context) {
  if (!variant)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " requires a materialized weft.exec.variant");

  auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requiresAttr || requiresAttr.empty())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) +
        " requires non-empty selected variant requires metadata carrying "
        "exactly one RVV capability provider");

  llvm::SmallVector<const support::CapabilityDescriptor *, 2>
      selectedRVVProviders;
  for (mlir::Attribute entry : requiresAttr) {
    auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(entry);
    if (!symbolRef)
      return makeRVVSelectedTargetCapabilityError(
          llvm::Twine(context) +
          " selected variant requires entry must be a symbol reference");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      return makeRVVSelectedTargetCapabilityError(
          llvm::Twine(context) + " selected variant requires entry @" +
          symbolRef.getValue() +
          " does not resolve in the weft.exec target capability set");

    if (capabilities.satisfiesIDTransitively(*capability, "rvv"))
      selectedRVVProviders.push_back(capability);
  }

  if (selectedRVVProviders.empty())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) +
        " requires exactly one selected RVV capability provider satisfying "
        "id 'rvv'; no selected requires entry satisfied RVV capability");
  if (selectedRVVProviders.size() > 1)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) +
        " requires exactly one selected RVV capability provider satisfying "
        "id 'rvv'; ambiguous selected providers were " +
        selected_target_capability_detail::joinProviderSymbols(
            selectedRVVProviders));

  const support::CapabilityDescriptor &capability =
      *selectedRVVProviders.front();
  if (!capability.isAvailable())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " satisfying id 'rvv' is unavailable "
        "(status = '" +
        capability.getStatus() + "')");

  if (capability.getID() == "rvv" && capability.getKind() != "isa-vector")
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " declares exact id 'rvv' but kind '" +
        capability.getKind() + "'; expected kind 'isa-vector'");
  if (capability.getID() != "rvv" && capability.getKind() != "isa-vector" &&
      capability.getKind() != "profile")
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() +
        " satisfying id 'rvv' must be kind 'isa-vector' or 'profile', got '" +
        capability.getKind() + "'");

  llvm::SmallVector<support::CapabilityConflict, 2> conflicts;
  capabilities.collectAvailableConflictsForCapability(capability, conflicts);
  if (!conflicts.empty())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " conflicts with available provider @" +
        conflicts.front().conflictingCapability->getSymbolName());

  if (llvm::Error error =
          selected_target_capability_detail::verifyProfileProperties(
              capability, context))
    return std::move(error);

  RVVSelectedTargetCapabilityFacts facts;
  facts.selectedProviderSymbol = capability.getSymbolName().str();
  facts.selectedProviderID = capability.getID().str();
  facts.selectedProviderKind = capability.getKind().str();
  facts.rvvSatisfactionKind =
      selected_target_capability_detail::classifyRVVSatisfaction(capability)
          .str();
  llvm::Expected<std::string> supportedSEW =
      selected_target_capability_detail::readTypedStringProperty(
          capability, "supported_sew", context);
  if (!supportedSEW)
    return supportedSEW.takeError();
  llvm::Expected<std::string> supportedLMUL =
      selected_target_capability_detail::readTypedStringProperty(
          capability, "supported_lmul", context);
  if (!supportedLMUL)
    return supportedLMUL.takeError();
  llvm::Expected<std::string> rvvVersion =
      selected_target_capability_detail::readTypedStringProperty(
          capability, "rvv_version", context);
  if (!rvvVersion)
    return rvvVersion.takeError();
  llvm::Expected<std::string> requiredTailPolicy =
      selected_target_capability_detail::readTypedStringProperty(
          capability, "required_tail_policy", context);
  if (!requiredTailPolicy)
    return requiredTailPolicy.takeError();
  llvm::Expected<std::string> requiredMaskPolicy =
      selected_target_capability_detail::readTypedStringProperty(
          capability, "required_mask_policy", context);
  if (!requiredMaskPolicy)
    return requiredMaskPolicy.takeError();

  static const llvm::StringRef kAllowedSEW[] = {"8", "16", "32", "64"};
  static const llvm::StringRef kAllowedLMUL[] = {
      "mf8", "mf4", "mf2", "m1", "m2", "m4", "m8"};
  if (llvm::Error error =
          selected_target_capability_detail::validateTokenList(
              capability, *supportedSEW, "supported_sew", kAllowedSEW,
              context))
    return std::move(error);
  if (llvm::Error error =
          selected_target_capability_detail::validateTokenList(
              capability, *supportedLMUL, "supported_lmul", kAllowedLMUL,
              context))
    return std::move(error);
  if (!rvvVersion->empty() && *rvvVersion != "1.0" && *rvvVersion != "0.7")
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " has unknown rvv_version '" +
        *rvvVersion + "'; expected '1.0', '0.7', or an absent fact");
  if (*rvvVersion == "0.7" &&
      (rvvCapabilityPropertyListContains(*supportedLMUL, "mf8") ||
       rvvCapabilityPropertyListContains(*supportedLMUL, "mf4") ||
       rvvCapabilityPropertyListContains(*supportedLMUL, "mf2")))
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() +
        " conflicts internally: rvv_version=0.7 but supported_lmul contains "
        "a fractional LMUL token");
  if (llvm::Error error =
          selected_target_capability_detail::validatePolicyProperty(
              capability, *requiredTailPolicy, "required_tail_policy",
              context))
    return std::move(error);
  if (llvm::Error error =
          selected_target_capability_detail::validatePolicyProperty(
              capability, *requiredMaskPolicy, "required_mask_policy",
              context))
    return std::move(error);

  facts.supportedSEW = std::move(*supportedSEW);
  facts.supportedLMUL = std::move(*supportedLMUL);
  facts.rvvVersion = std::move(*rvvVersion);
  facts.requiredTailPolicy = std::move(*requiredTailPolicy);
  facts.requiredMaskPolicy = std::move(*requiredMaskPolicy);

  mlir::Attribute rawMinimumVLEN =
      capability.getPropertyAttribute("minimum_vlen");
  if (rawMinimumVLEN) {
    auto minimumVLEN = llvm::dyn_cast<mlir::IntegerAttr>(rawMinimumVLEN);
    if (!minimumVLEN || !minimumVLEN.getType().isSignlessInteger(64))
      return makeRVVSelectedTargetCapabilityError(
          llvm::Twine(context) + " selected RVV capability provider @" +
          capability.getSymbolName() +
          " minimum_vlen must be a typed signless i64 attribute");
    if (minimumVLEN.getInt() <= 0)
      return makeRVVSelectedTargetCapabilityError(
          llvm::Twine(context) + " selected RVV capability provider @" +
          capability.getSymbolName() + " has non-positive minimum_vlen " +
          llvm::Twine(minimumVLEN.getInt()));
    facts.minimumVLEN = minimumVLEN.getInt();
  } else if (capability.getProperties().count("minimum_vlen") != 0) {
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() +
        " minimum_vlen has no typed source attribute");
  }

  mlir::Attribute rawVectorRegisterCount =
      capability.getPropertyAttribute("vreg_count");
  if (rawVectorRegisterCount) {
    auto vectorRegisterCount =
        llvm::dyn_cast<mlir::IntegerAttr>(rawVectorRegisterCount);
    if (!vectorRegisterCount ||
        !vectorRegisterCount.getType().isSignlessInteger(64))
      return makeRVVSelectedTargetCapabilityError(
          llvm::Twine(context) + " selected RVV capability provider @" +
          capability.getSymbolName() +
          " vreg_count must be a typed signless i64 attribute");
    if (vectorRegisterCount.getInt() <= 0)
      return makeRVVSelectedTargetCapabilityError(
          llvm::Twine(context) + " selected RVV capability provider @" +
          capability.getSymbolName() + " has non-positive vreg_count " +
          llvm::Twine(vectorRegisterCount.getInt()));
    facts.vectorRegisterCount = vectorRegisterCount.getInt();
  } else if (capability.getProperties().count("vreg_count") != 0) {
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() +
        " vreg_count has no typed source attribute");
  }
  facts.providerMirror =
      selected_target_capability_detail::formatProviderMirror(capability);
  return facts;
}

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVSELECTEDTARGETCAPABILITY_H
